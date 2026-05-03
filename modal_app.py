"""Modal cloud deployment — DICOM Extractor v5 on auto-scaled CPU containers.

Runs the COMPLETE pipeline on Modal with:
  - Auto-scaled CPU containers (4 series per container)
  - Modal Volume for persistent input + output storage
  - Distributed extraction via .map() across CPU containers
  - Per-series fan-out via .starmap() across CPU containers
  - HIPAA compliance scanning + 97-rule redaction pipeline
  - HTML dashboard, montages, histograms, quality grades, HF dataset upload
  - Job tracking with stage-level progress and cancellation

Default input: /Volumes/T7 Shield/redacted (auto-detected when plugged in)

CLI usage:
  modal run modal_app.py::run --do-redact --upload-hf             # full pipeline
  modal run modal_app.py::upload                                   # upload from T7 Shield
  modal run modal_app.py::extract --study redacted                 # extract only
  modal run modal_app.py::redact --study redacted                  # redact only
  modal run modal_app.py::download --study redacted                # download results
  modal run modal_app.py::list_studies                             # list all studies
  modal run modal_app.py::cleanup                                  # delete stale data
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path, PurePosixPath

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import modal

# ── Modal infrastructure ─────────────────────────────────────────────────────

VOLUME_NAME = "micom-data"
_T7_SHIELD = "/Volumes/T7 Shield/redacted"
_MCAP_LOCAL = "mcap-files"
DEFAULT_LOCAL_INPUT = _T7_SHIELD if Path(_T7_SHIELD).exists() else _MCAP_LOCAL
UPLOAD_WORKERS = 4
UPLOAD_CHUNK_SIZE = 20000
DOWNLOAD_WORKERS = 16
PARALLEL_EXTRACTION_THRESHOLD = 300
DICOM_EXTENSIONS = {".dcm", ".dicom", ".ima", ".img"}

# Shared pip dependencies — single source of truth
_PIP_DEPS = [
    "pydicom>=3.0",
    "SimpleITK>=2.4",
    "nibabel>=5.3",
    "polars>=1.17",
    "numpy>=2.1",
    "rich>=13.9",
    "matplotlib>=3.10",
    "scipy>=1.14",
    "pillow>=11.0",
    "python-dotenv>=1.0",
    "huggingface_hub>=0.25",
    "openai>=1.50",
]

# uv_pip_install resolves and installs in parallel (vs serial pip), which
# typically rebuilds the dep layer ~3-5x faster on cache misses. Layer order
# is stable→volatile so apt_install stays cached when deps shift, and the
# local source directory is injected at startup (default copy=False) so
# editing src/ never invalidates the image.
_base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("fonts-dejavu-core")
    .uv_pip_install(*_PIP_DEPS)
)

cpu_image = _base_image.add_local_dir("src", remote_path="/root/src")

app = modal.App("micom", image=cpu_image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Volume is mounted at /vol. Internal paths are /vol/studies/... and /vol/output/...
# Volume API (put_file, listdir, read_file) uses paths relative to volume root.
# Filesystem paths on the container use the mount point prefix.
MOUNT_POINT = PurePosixPath("/vol")
STUDIES_DIR = MOUNT_POINT / "studies"
OUTPUT_DIR = MOUNT_POINT / "output"

# Volume-relative paths (for volume.batch_upload, volume.listdir, volume.read_file)
_VOL_STUDIES = PurePosixPath("studies")
_VOL_OUTPUT = PurePosixPath("output")

# ── Job tracking ────────────────────────────────────────────────────────────
# Modal Dict stores active job metadata so we can list, check progress, and cancel.
# Keys: call_id → {"study", "pipeline", "status", "started_at", "stage", "detail"}

job_tracker = modal.Dict.from_name("micom-jobs", create_if_missing=True)

PIPELINE_STAGES = {
    "extraction": [
        "extract",
        "conformance",
        "series_gpu",
        "reports",
        "grade",
        "hipaa",
        "hf_upload",
    ],
    "redaction": ["pre_scan", "redact", "post_scan", "register"],
}


def _update_job(call_id: str, *, stage: str = "", detail: str = "", status: str = "running"):
    """Update job progress in the tracker. Safe to call from any container."""
    try:
        existing = job_tracker[call_id]
    except KeyError:
        existing = {}
    existing.update(
        {
            "stage": stage,
            "detail": detail,
            "status": status,
            "updated_at": time.time(),
        }
    )
    job_tracker[call_id] = existing


def _register_job(call_id: str, study: str, pipeline: str):
    """Register a new job in the tracker."""
    job_tracker[call_id] = {
        "study": study,
        "pipeline": pipeline,
        "status": "running",
        "started_at": time.time(),
        "updated_at": time.time(),
        "stage": "queued",
        "detail": "",
    }


def _finish_job(call_id: str, status: str = "completed", detail: str = ""):
    """Mark a job as completed or failed."""
    _update_job(call_id, stage="done", detail=detail, status=status)


def _is_cancelled(call_id: str) -> bool:
    """Check if a job has been cancelled."""
    try:
        return job_tracker[call_id].get("status") == "cancelled"
    except KeyError:
        return False


def _get_hf_token() -> str:
    """Read HF token from local cache (set by `hf auth login`)."""
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token() or ""
    except Exception:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            return token_path.read_text().strip()
        return ""


# ── Upload helpers (tar-based) ──────────────────────────────────────────────
#
# Instead of uploading 355K individual files (blows 500K inode limit),
# pack them into tar archives locally, upload the tars (= ~20 inodes),
# and extract inside the container at runtime.

TAR_CHUNK_FILES = 20000  # files per tar archive


def _discover_dicom_files(local: Path) -> list[Path]:
    """Find all DICOM files recursively. Handles .dcm, .dicom, .ima, extensionless."""
    files = []
    for f in sorted(local.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() in DICOM_EXTENSIONS:
            files.append(f)
        elif f.suffix == "" and f.stat().st_size > 128:
            try:
                with open(f, "rb") as fh:
                    fh.seek(128)
                    if fh.read(4) == b"DICM":
                        files.append(f)
            except Exception:
                pass
    return files


def _volume_has_tars(study_name: str) -> int:
    """Check how many tar archives exist for this study on the volume."""
    remote = str(_VOL_STUDIES / study_name)
    try:
        entries = [e for e in volume.listdir(remote) if e.path.endswith(".tar")]
        return len(entries)
    except Exception:
        return 0


def _ensure_uploaded(local_dir: str, study_name: str = "") -> tuple[str, dict]:
    """Pack DICOM files into tar archives and upload to Modal volume.

    355K files → ~18 tar archives → 18 inodes (vs 355K).
    Each tar is ~3-4GB, uploaded sequentially.
    """
    import tarfile
    import tempfile

    local = Path(local_dir)
    if not local.exists():
        return study_name or "", {"uploaded": 0, "skipped": 0, "failed": 0}

    if not study_name:
        study_name = local.name

    dcm_files = _discover_dicom_files(local)
    if not dcm_files:
        return study_name, {"uploaded": 0, "skipped": 0, "failed": 0, "error": "no DICOM files"}

    total_bytes = sum(f.stat().st_size for f in dcm_files)
    n_tars = (len(dcm_files) + TAR_CHUNK_FILES - 1) // TAR_CHUNK_FILES

    # Check if tars already uploaded
    existing_tars = _volume_has_tars(study_name)
    if existing_tars >= n_tars:
        print(
            f"Study '{study_name}' already on volume ({existing_tars} tar archives) — skipping",
            flush=True,
        )
        return study_name, {
            "uploaded": 0,
            "skipped": len(dcm_files),
            "failed": 0,
            "total_bytes": total_bytes,
            "elapsed": 0,
        }

    print(
        f"Packing {len(dcm_files)} files ({total_bytes / 1024**2:.1f} MB) into {n_tars} tar archives",
        flush=True,
    )
    t0 = time.time()
    uploaded = 0

    for i in range(0, len(dcm_files), TAR_CHUNK_FILES):
        chunk = dcm_files[i : i + TAR_CHUNK_FILES]
        chunk_idx = i // TAR_CHUNK_FILES
        tar_name = f"chunk_{chunk_idx:04d}.tar"
        remote_path = str(_VOL_STUDIES / study_name / tar_name)

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=True) as tmp:
            print(f"  Packing {tar_name} ({len(chunk)} files)...", end="", flush=True)
            with tarfile.open(tmp.name, "w") as tf:
                for f in chunk:
                    tf.add(str(f), arcname=str(f.relative_to(local)))

            tar_size = Path(tmp.name).stat().st_size
            print(f" {tar_size / 1024**2:.0f} MB, uploading...", end="", flush=True)

            with volume.batch_upload(force=True) as batch:
                batch.put_file(tmp.name, remote_path)

            uploaded += len(chunk)
            elapsed = time.time() - t0
            tar_size / max(elapsed - (t0 - t0), 0.01) / 1024**2
            print(f" done [{uploaded}/{len(dcm_files)}]", flush=True)

    elapsed = time.time() - t0
    print(
        f"Upload: {n_tars} tar archives ({total_bytes / 1024**2:.0f} MB) in {elapsed:.1f}s",
        flush=True,
    )
    return study_name, {
        "uploaded": uploaded,
        "skipped": 0,
        "failed": 0,
        "total_bytes": total_bytes,
        "elapsed": round(elapsed, 1),
        "tar_count": n_tars,
    }


# ── Download helpers ─────────────────────────────────────────────────────────


def _download_one_file(vol_path: str, local_path: Path) -> int:
    """Download a single file from volume, streaming to disk."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        n = 0
        with open(local_path, "wb") as f:
            for chunk in volume.read_file(vol_path):
                f.write(chunk)
                n += len(chunk)
        return n
    except Exception:
        return 0


def _parallel_download(
    vol_dir: str,
    local: Path,
    max_workers: int = DOWNLOAD_WORKERS,
) -> tuple[int, int]:
    """Download all files in parallel. vol_dir is volume-relative (no /data prefix)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        entries = [
            e
            for e in volume.listdir(vol_dir, recursive=True)
            if not e.path.endswith("/") and getattr(e, "type", None) != "directory"
        ]
    except Exception:
        return 0, 0

    if not entries:
        return 0, 0

    n = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for entry in entries:
            rel = PurePosixPath(entry.path).relative_to(vol_dir)
            out = local / str(rel)
            futures[pool.submit(_download_one_file, entry.path, out)] = entry.path

        for fut in as_completed(futures):
            total_bytes += fut.result()
            n += 1
            if n % 50 == 0:
                print(f"  [{n}/{len(entries)}] downloaded...")

    return n, total_bytes


# ── Modal functions ──────────────────────────────────────────────────────────


@app.function(
    image=cpu_image,
    volumes={str(MOUNT_POINT): volume},
    timeout=120,
    memory=512,
    cpu=0.5,
    region="us-east",
    scaledown_window=300,
)
@modal.concurrent(max_inputs=64, target_inputs=48)
def extract_file_cloud(fpath: str) -> dict:
    """Extract metadata from a single DICOM file. Distributed via .map().

    region="us-east" colocates with the volume's home region to cut cross-region
    egress on per-file reads. max_inputs=64 lets one container service a full
    .map() chunk in flight (extraction is I/O-bound: pydicom read of metadata
    only, skip_pixels=True). target_inputs=48 keeps headroom so the autoscaler
    spins up a new container before any single one saturates.
    """
    import sys

    sys.path.insert(0, "/root")
    from src.extraction import extract_single_file

    try:
        return extract_single_file(fpath, skip_pixels=True)
    except Exception as e:
        return {"_filepath": fpath, "_filename": fpath.rsplit("/", 1)[-1], "_error": str(e)}


# ── Per-series processing — CPU containers, fanned out ──────────────────────
#
# Series fan out across multiple containers via .starmap(). Each container
# processes 4 series concurrently (CPU-bound: numpy stats, PIL montages).
# Modal auto-scales containers based on demand.


@app.function(
    image=cpu_image,
    volumes={str(MOUNT_POINT): volume},
    timeout=600,
    memory=8192,
    cpu=4.0,
    region="us-east",
    scaledown_window=300,
    retries=modal.Retries(
        max_retries=2,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
)
@modal.concurrent(max_inputs=4)
def process_one_series_gpu(
    uid: str,
    files: list[str],
    meta: dict,
    out_dir: str,
    export_nii: bool,
    idx: int,
    subdir: str,
    records: list[dict],
    conformance: list[dict],
) -> dict:
    """Process a single series on a CPU container.

    All work is CPU-bound (numpy stats, PIL montages, SimpleITK reads).
    max_inputs=4 means 4 series run concurrently per container.
    Modal auto-scales the number of containers based on demand.
    """
    import sys

    sys.path.insert(0, "/root")
    import traceback

    from src.series import process_one_series, reset_sitk_probe

    reset_sitk_probe()

    desc = meta.get("series_description", uid[:12])
    try:
        r = process_one_series(
            uid, files, meta, out_dir, export_nii, idx, subdir, records, conformance
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  [ERROR] Series {idx} '{desc}' failed: {e}\n{tb}")
        return {
            "uid": uid,
            "label": desc,
            "info": meta,
            "vstats": None,
            "montage_path": None,
            "histogram_path": None,
            "enhanced_path": None,
            "series_folder": None,
            "error": str(e),
            "traceback": tb,
        }

    vs = r.vstats
    if vs:
        print(
            f"  s{r.info.get('series_number', '?')} {r.info.get('series_description', '')}: "
            f"{vs.get('volume_shape', [])} SNR={vs.get('volume_snr_estimate', 0):.2f} "
            f"Grade={vs.get('quality_grade', '?')}"
        )

    return {
        "uid": r.uid,
        "label": r.label,
        "info": r.info,
        "vstats": r.vstats,
        "montage_path": r.montage_path,
        "histogram_path": r.histogram_path,
        "enhanced_path": getattr(r, "enhanced_path", None),
        "series_folder": getattr(r, "series_folder", None),
        "error": None,
    }


@app.function(
    image=cpu_image,
    volumes={str(MOUNT_POINT): volume},
    timeout=1800,
    memory=4096,
    cpu=2.0,
    region="us-east",
    scaledown_window=300,
)
def extract_tar_cloud(tar_path: str, extract_dir: str) -> dict:
    """Extract one tar archive into the volume. Distributed via .starmap().

    Each tar is ~3-4 GB and contains up to 20k DICOMs. Running one tar per
    container in parallel scales linearly with tar count; the orchestrator
    used to loop sequentially, which dominated startup for studies with
    18+ tars. The volume's background commit picks up the writes; the
    orchestrator calls volume.reload() before listing files.
    """
    import tarfile as _tf
    from pathlib import Path as _P

    tar_p = _P(tar_path)
    out = _P(extract_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_files = 0
    with _tf.open(str(tar_p), "r") as tf:
        for member in tf:
            tf.extract(member, path=str(out), filter="data")
            if member.isfile():
                n_files += 1

    return {
        "tar": tar_p.name,
        "files": n_files,
        "elapsed_s": round(time.time() - t0, 2),
    }


@app.function(
    image=cpu_image,
    volumes={str(MOUNT_POINT): volume},
    timeout=86400,
    memory=16384,
    cpu=8.0,
    region="us-east",
    secrets=[
        modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "") or _get_hf_token()})
    ],
)
def run_extraction_pipeline(
    study_name: str,
    analyze: bool = False,
    export_nii: bool = False,
    compress: bool = False,
    upload_hf: bool = False,
    hf_repo: str = "",
    hf_private: bool = False,
    annotate: bool = False,
    _call_id: str = "",
) -> dict:
    """Run the FULL extraction pipeline — orchestrator on CPU, series on CPU.

    Architecture:
      - This function runs on a CPU container (cheap, fast cold start)
      - Extraction: distributed via extract_file_cloud.map() across CPU containers
      - Series processing: fanned out via process_one_series_gpu.map() across
        up to 10 CPU GPU containers — each series gets its own 80GB CPU
      - Reports: generated on this CPU container after series results return

    For a study with 15 series, all 15 dispatch to GPU containers simultaneously.
    Modal scales up to 10 containers in parallel = 10 CPUs at once.
    """
    import sys

    sys.path.insert(0, "/root")

    # Reload volume to see files uploaded from the local machine
    volume.reload()

    study_dir = Path(str(STUDIES_DIR)) / study_name
    out_dir = Path(str(OUTPUT_DIR)) / study_name

    if not study_dir.exists():
        return {"error": f"Study not found: {study_dir}"}

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract tar archives if present (parallel via .starmap) ─────────
    # One container per tar, all running concurrently. Replaces the
    # serial-on-orchestrator extract loop that used to dominate startup
    # for studies with 18+ tars.
    tar_files = sorted(study_dir.glob("*.tar"))
    if tar_files:
        extract_dir = Path(str(STUDIES_DIR)) / f"{study_name}_extracted"
        if extract_dir.exists() and any(extract_dir.rglob("*.dcm")):
            print(f"Using previously extracted files in {extract_dir}", flush=True)
        else:
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Extracting {len(tar_files)} tar archives in parallel → {extract_dir}...",
                flush=True,
            )
            t_untar = time.time()
            tar_args = [(str(tf), str(extract_dir)) for tf in tar_files]
            results = list(extract_tar_cloud.starmap(tar_args))
            n_extracted = sum(r.get("files", 0) for r in results)
            volume.reload()
            longest = max((r.get("elapsed_s", 0) for r in results), default=0)
            print(
                f"  Extracted {n_extracted} files from {len(tar_files)} tars "
                f"in {time.time() - t_untar:.1f}s "
                f"(parallel; longest single-tar {longest:.1f}s)",
                flush=True,
            )
        study_dir = extract_dir  # switch to extracted files

    dcm_files = sorted(study_dir.rglob("*.dcm"))
    if not dcm_files:
        for f in sorted(study_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".dcm", ".dicom", ".ima", ""):
                dcm_files.append(f)
        if not dcm_files:
            return {"error": f"No DICOM files in {study_dir}"}

    n_files = len(dcm_files)
    print(f"Found {n_files} DICOM files in {study_dir}", flush=True)
    t0 = time.time()

    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.constants import NON_IMAGE_SOP, SOP_CLASS_NAMES
    from src.exports import export_cross_series_comparison
    from src.extraction import check_conformance, extract_single_file
    from src.quality import grade_study
    from src.report import generate_html_report

    n_workers = 8
    file_paths = [str(f) for f in dcm_files]

    # Helper: check cancellation between stages
    def _check_cancel():
        if _call_id and _is_cancelled(_call_id):
            print("Pipeline cancelled by user")
            _finish_job(_call_id, status="cancelled", detail="Cancelled by user")
            return True
        return False

    def _progress(stage: str, detail: str = ""):
        if _call_id:
            _update_job(_call_id, stage=stage, detail=detail)

    _progress("extract", f"Starting extraction of {n_files} files")

    # ── Stage 1: Extract (cached on volume) ────────────────────────────
    # Cache extraction results as JSON on volume — skip the 34-min extraction
    # on re-runs. Invalidate if file count changes.
    cache_path = out_dir / "extraction_cache.json"
    files_are_local = any(fp.startswith("/tmp/") for fp in file_paths)

    t_ext = time.time()
    MAP_CHUNK = 20000

    all_records = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("_n_files") == n_files:
                all_records = cached["records"]
                print(f"Using cached extraction ({len(all_records)} records)", flush=True)
                _progress("extract", f"cached {len(all_records)} records")
        except Exception:
            pass

    if all_records is None:
        if not files_are_local and n_files >= PARALLEL_EXTRACTION_THRESHOLD:
            print(
                f"Distributed extraction (.map) for {n_files} files in chunks of {MAP_CHUNK}...",
                flush=True,
            )
            all_records = []
            for chunk_start in range(0, n_files, MAP_CHUNK):
                chunk = file_paths[chunk_start : chunk_start + MAP_CHUNK]
                chunk_results = list(extract_file_cloud.map(chunk))
                all_records.extend(chunk_results)
                elapsed = time.time() - t_ext
                rate = len(all_records) / max(elapsed, 0.01)
                eta = (n_files - len(all_records)) / max(rate, 0.01)
                print(
                    f"  [{len(all_records)}/{n_files}] {rate:.0f} files/s, ETA {eta:.0f}s",
                    flush=True,
                )
                _progress("extract", f"{len(all_records)}/{n_files} files ({rate:.0f}/s)")
        else:
            if files_are_local:
                print(f"Local extraction (threaded) for {n_files} files...", flush=True)
            all_records = []
            _last_log = [0]

            def _safe_extract(fp):
                try:
                    return extract_single_file(fp, True)
                except Exception as e:
                    return {"_filepath": fp, "_filename": Path(fp).name, "_error": str(e)}

            with ThreadPoolExecutor(max_workers=n_workers * 4) as pool:
                futs = {pool.submit(_safe_extract, fp): fp for fp in file_paths}
                for fut in as_completed(futs):
                    all_records.append(fut.result())
                done = len(all_records)
                if done - _last_log[0] >= 10000 or done == n_files:
                    elapsed = time.time() - t_ext
                    rate = done / max(elapsed, 0.01)
                    eta = (n_files - done) / max(rate, 0.01)
                    print(f"  [{done}/{n_files}] {rate:.0f} files/s, ETA {eta:.0f}s", flush=True)
                    _last_log[0] = done
                    _progress("extract", f"{done}/{n_files} files ({rate:.0f}/s)")

    # Filter out failed extractions
    errored = [r for r in all_records if r.get("_error")]
    all_records = [r for r in all_records if not r.get("_error")]
    all_records.sort(key=lambda r: r.get("_filename", ""))
    t_extract = time.time() - t_ext
    print(
        f"Extracted {len(all_records)} files in {t_extract:.1f}s"
        + (f" ({len(errored)} failed)" if errored else ""),
        flush=True,
    )

    # Cache extraction results on volume for next run
    if not cache_path.exists() and all_records:
        try:
            cache_path.write_text(
                json.dumps({"_n_files": n_files, "records": all_records}, default=str)
            )
            volume.commit()
            print("  Cached extraction results for next run", flush=True)
        except Exception as e:
            print(f"  Cache write failed (non-fatal): {e}", flush=True)

    if _check_cancel():
        return {"error": "cancelled", "stage": "extract"}

    _progress("conformance", f"Extracted {len(all_records)} files, running conformance checks")

    # Patient info
    r0 = all_records[0]
    patient_info = {
        k: r0.get(f"_{k}", "")
        for k in (
            "patient_id",
            "patient_name",
            "patient_sex",
            "patient_birth_date",
            "patient_weight",
            "study_date",
            "study_description",
            "institution",
            "manufacturer",
            "model",
            "field_strength",
            "software_versions",
            "station_name",
        )
    }

    # Group by series
    groups: dict[str, list[str]] = defaultdict(list)
    series_meta: dict[str, dict] = {}
    for r in all_records:
        uid = r.get("_series_uid", "unknown")
        groups[uid].append(r.get("_filepath", ""))
        if uid not in series_meta:
            series_meta[uid] = {
                "series_number": r.get("_series_number", ""),
                "series_description": r.get("_series_description", ""),
                "modality": r.get("_modality", ""),
                "sop_class_uid": r.get("_sop_class_uid", ""),
            }

    def _sort_key(uid):
        try:
            return (0, int(series_meta.get(uid, {}).get("series_number", 0)))
        except (ValueError, TypeError):
            return (1, str(series_meta.get(uid, {}).get("series_number", "")))

    sorted_uids = sorted(groups.keys(), key=_sort_key)
    image_records = [r for r in all_records if r.get("_sop_class_uid", "") not in NON_IMAGE_SOP]

    # ── Stage 2: Conformance + save (concurrent) ─────────────────────────
    with ThreadPoolExecutor(max_workers=3) as pool:
        conf_fut = pool.submit(check_conformance, image_records)
        json_fut = pool.submit(
            lambda: (out_dir / "dicom_full_dump.json").write_text(
                json.dumps(all_records, indent=2, default=str)
            )
        )

        def _save_csv():
            try:
                import polars as pl

                prio = [
                    "_filename",
                    "_series_number",
                    "_series_description",
                    "_modality",
                    "PatientID",
                    "StudyDate",
                    "Modality",
                    "SeriesDescription",
                    "MagneticFieldStrength",
                    "Manufacturer",
                ]
                all_keys = {k for r in all_records for k in r}
                cols = [c for c in prio if c in all_keys] + sorted(all_keys - set(prio))
                rows = [{c: str(r.get(c, "")) for c in cols} for r in all_records]
                (out_dir / "dicom_metadata.csv").write_csv_via_polars(pl.DataFrame(rows))
            except Exception:
                pass

        csv_fut = pool.submit(_save_csv)
        conformance_issues = conf_fut.result()
        json_fut.result()
        csv_fut.result()

    print(f"Conformance: {len(conformance_issues)} issues")

    if _check_cancel():
        return {"error": "cancelled", "stage": "conformance"}

    # ── Stage 3: Process series ──────────────────────────────────────────
    t_series = time.time()
    image_uids = [
        u
        for u in sorted_uids
        if series_meta.get(u, {}).get("sop_class_uid", "") not in NON_IMAGE_SOP
    ]
    ps_uids = [u for u in sorted_uids if u not in set(image_uids)]

    filepath_to_uid: dict[str, str] = {}
    filename_to_uids: dict[str, list[str]] = defaultdict(list)
    series_subdirs: dict[str, str] = {}
    series_records: dict[str, list[dict]] = defaultdict(list)
    series_conformance: dict[str, list[dict]] = defaultdict(list)

    for uid in image_uids:
        first = Path(groups[uid][0])
        try:
            rel = first.parent.relative_to(study_dir)
            series_subdirs[uid] = str(rel) if str(rel) != "." else ""
        except ValueError:
            series_subdirs[uid] = ""
        for fp in groups[uid]:
            filepath_to_uid[fp] = uid
            filename_to_uids[Path(fp).name].append(uid)

    for r in all_records:
        uid = filepath_to_uid.get(r.get("_filepath", ""))
        if uid:
            series_records[uid].append(r)

    for c in conformance_issues:
        for uid in filename_to_uids.get(c.get("filename", ""), ()):
            series_conformance[uid].append(c)

    # Fan out series processing. If files are local (/tmp from tar extraction),
    # process on this container. Otherwise distribute via .starmap().
    print(f"Processing {len(image_uids)} series...", flush=True)
    _progress("series_gpu", f"Processing {len(image_uids)} series")

    series_args = [
        (
            uid,
            groups[uid],
            series_meta.get(uid, {}),
            str(out_dir),
            export_nii,
            idx + 1,
            series_subdirs.get(uid, ""),
            series_records.get(uid, []),
            series_conformance.get(uid, []),
        )
        for idx, uid in enumerate(image_uids)
    ]

    if files_are_local:
        # Process locally — files are in /tmp, not visible to workers
        import sys

        sys.path.insert(0, "/root")
        from src.series import process_one_series, reset_sitk_probe

        reset_sitk_probe()

        gpu_results = []
        for args in series_args:
            uid, files, meta, od, nii, idx, subdir, recs, conf = args
            desc = meta.get("series_description", uid[:12])
            try:
                r = process_one_series(uid, files, meta, od, nii, idx, subdir, recs, conf)
                gpu_results.append(
                    {
                        "uid": r.uid,
                        "label": r.label,
                        "info": r.info,
                        "vstats": r.vstats,
                        "montage_path": r.montage_path,
                        "histogram_path": r.histogram_path,
                        "enhanced_path": getattr(r, "enhanced_path", None),
                        "series_folder": getattr(r, "series_folder", None),
                        "error": None,
                    }
                )
                vs = r.vstats
                if vs:
                    print(
                        f"  s{r.info.get('series_number', '?')} {r.info.get('series_description', '')}: "
                        f"{vs.get('volume_shape', [])} SNR={vs.get('volume_snr_estimate', 0):.2f} "
                        f"Grade={vs.get('quality_grade', '?')}",
                        flush=True,
                    )
            except Exception as e:
                print(f"  [ERROR] Series {idx} '{desc}': {e}", flush=True)
                gpu_results.append(
                    {
                        "uid": uid,
                        "label": desc,
                        "info": meta,
                        "vstats": None,
                        "montage_path": None,
                        "histogram_path": None,
                        "enhanced_path": None,
                        "series_folder": None,
                        "error": str(e),
                    }
                )
    else:
        gpu_results = list(process_one_series_gpu.starmap(series_args))

    t_proc = time.time() - t_series

    failed_series = [r for r in gpu_results if r.get("error")]
    ok_series = [r for r in gpu_results if not r.get("error")]
    if failed_series:
        print(f"  WARNING: {len(failed_series)}/{len(gpu_results)} series failed:")
        for r in failed_series:
            print(f"    - {r.get('label', '?')}: {r.get('error', '?')}")
    print(f"Processed {len(ok_series)}/{len(gpu_results)} series in {t_proc:.1f}s", flush=True)

    if _check_cancel():
        return {"error": "cancelled", "stage": "series_gpu"}

    # Build info dicts from GPU results (dicts, not SeriesResult objects)
    ps_info = [
        {
            "series_uid": uid,
            "series_number": series_meta[uid].get("series_number", ""),
            "series_description": series_meta[uid].get("series_description", ""),
            "modality": series_meta[uid].get("modality", ""),
            "sop_class": SOP_CLASS_NAMES.get(series_meta[uid].get("sop_class_uid", ""), ""),
            "file_count": len(groups[uid]),
            "has_pixels": False,
            "note": "Presentation state",
        }
        for uid in ps_uids
    ]

    series_info = [r["info"] for r in gpu_results] + ps_info
    comparison_data = {
        r["uid"]: {"label": r["label"], "vstats": r["vstats"]}
        for r in gpu_results
        if r.get("vstats")
    }
    image_paths = {
        f"{r['info'].get('series_number', '?')}_{r['info'].get('series_description', '')}": {
            "montage": r["montage_path"],
            "histogram": r["histogram_path"],
        }
        for r in gpu_results
        if r.get("montage_path")
    }

    # ── Stage 4: Reports ─────────────────────────────────────────────────
    _progress("reports", f"Generating reports for {len(ok_series)} series")
    t_report = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        cross_fut = pool.submit(export_cross_series_comparison, comparison_data, str(out_dir))
        stats = {
            "patient": patient_info,
            "series": series_info,
            "conformance_issues": conformance_issues,
        }
        stats_fut = pool.submit(
            (out_dir / "series_stats.json").write_text, json.dumps(stats, indent=2, default=str)
        )
        cross_path = cross_fut.result()
        generate_html_report(
            patient_info, series_info, conformance_issues, image_paths, cross_path, out_dir, pool
        )
        stats_fut.result()
    print(f"Reports in {time.time() - t_report:.1f}s")

    if _check_cancel():
        return {"error": "cancelled", "stage": "reports"}

    # ── Stage 5: Grade + AI ──────────────────────────────────────────────
    _progress("grade", "Grading study quality")
    all_grades = [
        s.get("quality_analysis", {}).get("quality_grade", {})
        for s in series_info
        if s.get("has_pixels") and s.get("quality_analysis")
    ]
    study_grade = grade_study(all_grades)

    if analyze or annotate:
        if os.environ.get("OPENROUTER_API_KEY"):
            _progress("annotate", "Multi-model OpenRouter annotation + Gemma 4 / Claude synthesis")
            try:
                from src.annotation.cloud import annotate_study_multi

                ann_report = annotate_study_multi(
                    gpu_results,
                    series_info,
                    out_dir,
                    synthesize=True,
                    patient_info=patient_info,
                )
                n_ann = ann_report.get("series_annotated", 0)
                summary = ann_report.get("summary", {})
                print(
                    f"  Annotated {n_ann} series, "
                    f"{summary.get('pathology_detected', 0)} with pathology, "
                    f"{summary.get('disagreements', 0)} disagreements"
                )
                if ann_report.get("narrative_path"):
                    print(f"  Final report → {ann_report['narrative_path']}")
            except Exception as e:
                print(f"  Cloud annotation failed: {e}")
        else:
            print("OPENROUTER_API_KEY not set — skipping cloud annotation")

    if _check_cancel():
        return {"error": "cancelled", "stage": "grade"}

    # ── Stage 6: HIPAA scan ──────────────────────────────────────────────
    _progress("hipaa", f"Scanning {n_files} files for HIPAA compliance")
    from src.hipaa import compliance_report_to_dict, run_hipaa_scan

    # If processing redacted data, tell scanner which tags are de-identified
    de_id_tags = None
    if "redacted" in study_name.lower():
        from src.redaction import PHI_TAGS_BLANK, PHI_TAGS_DATE, PHI_TAGS_HASH, PHI_TAGS_REMOVE

        de_id_tags = PHI_TAGS_HASH | PHI_TAGS_DATE | PHI_TAGS_BLANK | PHI_TAGS_REMOVE

    hipaa_report = run_hipaa_scan(
        file_paths,
        study_name=study_name,
        n_workers=n_workers,
        de_identified_tags=de_id_tags,
    )
    (out_dir / "hipaa_compliance.json").write_text(
        json.dumps(compliance_report_to_dict(hipaa_report), indent=2)
    )
    print(
        f"HIPAA: score {hipaa_report.compliance_score:.0f}/100, {hipaa_report.total_phi_findings} PHI, "
        f"{hipaa_report.risk_summary.get('high', 0)} high-risk"
    )

    volume.commit()

    if _check_cancel():
        return {"error": "cancelled", "stage": "hipaa"}

    # ── Stage 7: HF upload (optional — server-side, no download needed) ──
    hf_url = None
    if upload_hf:
        _progress("hf_upload", "Uploading to Hugging Face")
        print("\n=== Uploading to Hugging Face (server-side) ===")
        try:
            from src.hf_upload import upload_to_huggingface

            hipaa_dict = compliance_report_to_dict(hipaa_report)
            hf_url = upload_to_huggingface(
                out_dir,
                study_name=study_name,
                repo_id=hf_repo,
                patient_info=patient_info,
                series_info=series_info,
                hipaa_report=hipaa_dict,
                study_grade=study_grade,
                private=hf_private,
                token=os.environ.get("HF_TOKEN", ""),
                source_dir=study_dir,
            )
            print(f"  Dataset: {hf_url}")
        except Exception as e:
            print(f"  HF upload failed: {e}")

    total = time.time() - t0
    n_with_vol = sum(1 for r in gpu_results if r.get("vstats"))
    grade_dist = study_grade.get("grade_distribution", {})

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total:.1f}s")
    print("  GPU: 2x CPU per container (auto-scaled via .starmap)")
    print(f"  Files: {n_files} | Series: {len(image_uids)} image + {len(ps_uids)} PS")
    if failed_series:
        print(f"  Failed: {len(failed_series)} series")
    print(f"  Grade: {study_grade.get('grade', '?')} ({study_grade.get('score', 0):.0f}/100)")
    print(f"  HIPAA: {hipaa_report.compliance_score:.0f}/100")
    if hf_url:
        print(f"  HF: {hf_url}")
    print(
        f"  Timing: extract {t_extract:.1f}s + process {t_proc:.1f}s + reports {time.time() - t_report:.1f}s"
    )
    print(f"{'=' * 60}")

    result = {
        "study": study_name,
        "files": n_files,
        "image_series": len(image_uids),
        "with_volumes": n_with_vol,
        "series_succeeded": len(ok_series),
        "series_failed": len(failed_series),
        "failed_series_details": [
            {"label": r.get("label"), "error": r.get("error")} for r in failed_series
        ],
        "ps_series": len(ps_uids),
        "conformance_issues": len(conformance_issues),
        "study_grade": study_grade.get("grade", "?"),
        "study_score": study_grade.get("score", 0),
        "grade_distribution": grade_dist,
        "time_s": round(total, 1),
        "extract_time_s": round(t_extract, 1),
        "process_time_s": round(t_proc, 1),
        "gpu": "CPU (auto-scaled)",
        "storage": "modal_volume",
        "hipaa_score": hipaa_report.compliance_score,
        "hipaa_phi_findings": hipaa_report.total_phi_findings,
        "hipaa_high_risk_files": hipaa_report.risk_summary.get("high", 0),
        "hf_url": hf_url,
        "output": str(out_dir),
    }

    if _call_id:
        _finish_job(
            _call_id, detail=f"Grade {study_grade.get('grade', '?')}, {len(ok_series)} series"
        )

    return result


# ── Redaction ────────────────────────────────────────────────────────────────


@app.function(
    image=cpu_image,
    volumes={str(MOUNT_POINT): volume},
    timeout=120,
    memory=512,
    region="us-east",
    scaledown_window=300,
)
@modal.concurrent(max_inputs=32, target_inputs=24)
def redact_single_file_cloud(fpath: str, out_dir: str, salt: str, date_shift_days: int) -> dict:
    """Redact PHI from a single DICOM file. 66 rules, single-pass verify."""
    import sys

    sys.path.insert(0, "/root")
    from src.redaction import redact_single_file

    r = redact_single_file(fpath, out_dir, salt, date_shift_days, verify=True)
    return {
        "filepath": r.filepath,
        "output": r.output_path,
        "error": r.error,
        "removed": r.tags_removed,
        "blanked": r.tags_blanked,
        "hashed": r.tags_hashed,
        "shifted": r.tags_date_shifted,
        "verified_clean": r.verified_clean,
    }


@app.function(
    image=cpu_image, volumes={str(MOUNT_POINT): volume}, timeout=86400, memory=32768, cpu=8.0
)
def run_redaction_pipeline(
    study_name: str, salt: str = "", date_shift: int = -90, _call_id: str = ""
) -> dict:
    """Full HIPAA pipeline: scan → redact → verify → scan → copy redacted to studies.

    After redaction, copies the clean files into STUDIES_DIR/{study}_redacted
    so run_extraction_pipeline can process them as a normal study. This is
    the key integration point: redacted files become a first-class study that
    the extraction pipeline can run on.

    Flow:
      1. Pre-scan original files (baseline PHI inventory)
      2. Redact all files via .map() (66 rules, single-pass verify)
      3. Post-scan redacted files (confirm PHI removal)
      4. Copy redacted .dcm files → volume:/data/studies/{study}_redacted/
         so `extract --study {study}_redacted` works
    """
    import sys

    sys.path.insert(0, "/root")
    import hashlib
    import os

    from src.hipaa import compliance_report_to_dict, run_hipaa_scan

    # Reload volume to see files uploaded from the local machine
    volume.reload()

    study_dir = Path(str(STUDIES_DIR)) / study_name
    redact_out = Path(str(OUTPUT_DIR)) / f"{study_name}_redacted"
    redacted_study_dir = Path(str(STUDIES_DIR)) / f"{study_name}_redacted"

    if not study_dir.exists():
        return {"error": f"Study not found: {study_dir}"}

    # Extract tar archives if present
    import tarfile as _tarfile

    tar_files = sorted(study_dir.glob("*.tar"))
    if tar_files:
        extract_dir = Path("/tmp/dicom_study") / study_name
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {len(tar_files)} tar archives → {extract_dir}...", flush=True)
        for tf_path in tar_files:
            with _tarfile.open(str(tf_path), "r") as tf:
                tf.extractall(path=str(extract_dir), filter="data")
        study_dir = extract_dir

    redact_out.mkdir(parents=True, exist_ok=True)
    dcm_files = sorted(study_dir.rglob("*.dcm"))
    if not dcm_files:
        # Also try extensionless
        for f in sorted(study_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".dcm", ".dicom", ".ima", ""):
                dcm_files.append(f)
        if not dcm_files:
            return {"error": "No .dcm files found"}

    if not salt:
        salt = hashlib.sha256(os.urandom(32)).hexdigest()[:16]

    file_paths = [str(f) for f in dcm_files]

    def _rprogress(stage: str, detail: str = ""):
        if _call_id:
            _update_job(_call_id, stage=stage, detail=detail)

    def _rcheck_cancel():
        if _call_id and _is_cancelled(_call_id):
            print("Redaction cancelled by user")
            _finish_job(_call_id, status="cancelled", detail="Cancelled by user")
            return True
        return False

    # Step 1: Redact (skip pre-scan — it's slow and informational only)
    _rprogress("redact", f"Redacting {len(dcm_files)} files")
    print(f"Step 1: Redacting ({len(dcm_files)} files, shift={date_shift}d)...", flush=True)
    t0 = time.time()

    files_are_local = any(str(f).startswith("/tmp/") for f in dcm_files)
    n_threads = 16  # 8 CPUs, 16 threads (I/O-bound, GIL-releasing)

    # Progress callback — prints every 5000 files
    _last_progress = [0]

    def _on_progress(done, total):
        if done - _last_progress[0] >= 5000 or done == total:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 0.01)
            eta = (total - done) / max(rate, 0.01)
            print(f"  [{done}/{total}] {rate:.0f} files/s, ETA {eta:.0f}s", flush=True)
            _last_progress[0] = done
            _rprogress("redact", f"{done}/{total} files ({rate:.0f}/s)")

    if files_are_local:
        from src.redaction import redact_files

        redact_out_local = Path("/tmp/redacted_out") / study_name
        redact_out_local.mkdir(parents=True, exist_ok=True)
        summary = redact_files(
            file_paths,
            str(redact_out_local),
            n_workers=n_threads,
            salt=salt,
            date_shift_days=date_shift,
            verify=True,
            on_progress=_on_progress,
        )
        results = []
        for r in summary.results:
            results.append(
                {
                    "filepath": r.filepath,
                    "output": r.output_path,
                    "error": r.error,
                    "removed": r.tags_removed,
                    "blanked": r.tags_blanked,
                    "hashed": r.tags_hashed,
                    "shifted": r.tags_date_shifted,
                    "verified_clean": r.verified_clean,
                }
            )
    else:
        results = list(
            redact_single_file_cloud.starmap(
                [(str(f), str(redact_out), salt, date_shift) for f in dcm_files]
            )
        )

    processed = sum(1 for r in results if not r.get("error"))
    failed = sum(1 for r in results if r.get("error"))
    verified = sum(1 for r in results if r.get("verified_clean"))
    removed = sum(r.get("removed", 0) for r in results)
    blanked = sum(r.get("blanked", 0) for r in results)
    hashed = sum(r.get("hashed", 0) for r in results)
    shifted = sum(r.get("shifted", 0) for r in results)
    t_redact = time.time() - t0
    print(
        f"  Done: {processed} files in {t_redact:.1f}s ({removed} removed, {blanked} blanked, "
        f"{hashed} hashed, {shifted} shifted, {verified}/{processed} verified)",
        flush=True,
    )

    if _rcheck_cancel():
        return {"error": "cancelled", "stage": "redact"}

    # Step 2: Post-scan (sample 1000 files, not all 355K — much faster)
    _rprogress("post_scan", "Verifying redacted files")
    redacted_paths = [r["output"] for r in results if r.get("output") and r["output"]]

    import random as _rnd

    sample_size = min(1000, len(redacted_paths))
    sample_paths = (
        _rnd.sample(redacted_paths, sample_size)
        if len(redacted_paths) > sample_size
        else redacted_paths
    )
    print(
        f"Step 2: HIPAA post-scan (sampling {sample_size}/{len(redacted_paths)} files)...",
        flush=True,
    )

    from src.redaction import PHI_TAGS_BLANK, PHI_TAGS_DATE, PHI_TAGS_HASH, PHI_TAGS_REMOVE

    de_identified = PHI_TAGS_HASH | PHI_TAGS_DATE | PHI_TAGS_BLANK | PHI_TAGS_REMOVE

    post = run_hipaa_scan(
        sample_paths,
        study_name=f"{study_name}_post",
        n_workers=n_threads,
        de_identified_tags=de_identified,
    )
    print(
        f"  Post: score {post.compliance_score:.0f}/100, {post.total_phi_findings} PHI (from {sample_size} sample)",
        flush=True,
    )

    if _rcheck_cancel():
        return {"error": "cancelled", "stage": "post_scan"}

    # Step 4: Pack redacted files into tar archives and register as a study
    _rprogress("register", f"Registering {study_name}_redacted")
    print(f"Step 4: Packing redacted files as '{study_name}_redacted'...", flush=True)

    volume.reload()

    # Collect redacted file paths
    redacted_files = []
    for r in results:
        if r.get("output") and not r.get("error"):
            src = Path(r["output"])
            if src.exists():
                redacted_files.append(src)

    # Pack into tar archives on the volume (keeps inode count low)
    redacted_study_dir.mkdir(parents=True, exist_ok=True)
    n_packed = 0
    for i in range(0, len(redacted_files), TAR_CHUNK_FILES):
        chunk = redacted_files[i : i + TAR_CHUNK_FILES]
        chunk_idx = i // TAR_CHUNK_FILES
        tar_path = redacted_study_dir / f"chunk_{chunk_idx:04d}.tar"
        with _tarfile.open(str(tar_path), "w") as tf:
            for f in chunk:
                tf.add(str(f), arcname=f.name)
        n_packed += len(chunk)

    print(
        f"  Packed {n_packed} redacted files into {(len(redacted_files) + TAR_CHUNK_FILES - 1) // TAR_CHUNK_FILES} tar archives",
        flush=True,
    )

    combined = {
        "post_redaction": compliance_report_to_dict(post),
        "redacted_study": f"{study_name}_redacted",
        "improvement": {
            "files_redacted": processed,
            "files_failed": failed,
            "tags_removed": removed,
            "tags_blanked": blanked,
            "tags_hashed": hashed,
            "tags_date_shifted": shifted,
            "verified_clean": verified,
            "redaction_time_s": round(t_redact, 1),
            "post_score": post.compliance_score,
            "post_phi": post.total_phi_findings,
        },
    }
    (redact_out / "hipaa_compliance.json").write_text(json.dumps(combined, indent=2))
    volume.commit()

    if _call_id:
        _finish_job(
            _call_id,
            detail=f"{processed} files redacted, post-score {post.compliance_score:.0f}/100",
        )

    return combined


# ── CLI entrypoints ──────────────────────────────────────────────────────────


@app.local_entrypoint()
def upload(local_dir: str = DEFAULT_LOCAL_INPUT, study_name: str = ""):
    """Upload DICOM files to Modal Volume (skip existing, parallel, progress)."""
    _ensure_uploaded(local_dir, study_name)


@app.local_entrypoint()
def extract(
    study: str = "",
    local_dir: str = "",
    analyze: bool = False,
    annotate: bool = False,
    do_redact: bool = False,
    upload_hf: bool = False,
    hf_repo: str = "",
    export_nii: bool = False,
    compress: bool = False,
):
    """Extract on cloud. Auto-uploads, optional redact + annotation + HF upload.

    --annotate: multi-model annotation (Gemini + Qwen + GPT-4 + Claude)
    --do-redact: redacts first, then extracts redacted files (PHI-free output)
    --upload-hf: uploads results to Hugging Face directly from cloud (fast)
    """
    if not study and not local_dir:
        t7 = Path(DEFAULT_LOCAL_INPUT)
        if t7.exists():
            local_dir = str(t7)
            study = t7.name
            print(f"Auto-detected: {study}")
        else:
            print("Usage: modal run modal_app.py extract --study <name> [--local-dir ./path]")
            return

    if local_dir:
        study, upload_stats = _ensure_uploaded(local_dir, study)
        if upload_stats.get("error"):
            print(f"Upload error: {upload_stats['error']}")
            return

    if not study:
        print("No study name — pass --study or --local-dir")
        return

    extract_study = study

    if do_redact:
        print(f"\n=== Redacting {study} → {study}_redacted ===")
        redact_result = run_redaction_pipeline.remote(study)
        print(json.dumps(redact_result, indent=2, default=str))
        if not redact_result.get("error"):
            extract_study = redact_result.get("redacted_study", f"{study}_redacted")
            print(f"Extracting redacted study: {extract_study}")

    result = run_extraction_pipeline.remote(
        extract_study,
        analyze,
        export_nii,
        compress,
        upload_hf=upload_hf,
        hf_repo=hf_repo,
        annotate=annotate,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def redact(study: str = "", salt: str = "", date_shift: int = -90):
    """Run full HIPAA redaction: scan → redact → verify → scan again."""
    if not study:
        print("Usage: modal run modal_app.py redact --study <name>")
        return
    result = run_redaction_pipeline.remote(study, salt, date_shift)
    print(json.dumps(result, indent=2, default=str))


@app.local_entrypoint()
def download(study: str = "", local_dir: str = "./output"):
    """Download results from Modal Volume (parallel, 16 threads)."""
    if not study:
        print("Usage: modal run modal_app.py download --study <name>")
        return
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    remote_dir = str(_VOL_OUTPUT / study)
    print(f"Downloading volume:{remote_dir} → {local}")
    t0 = time.time()
    n, total_bytes = _parallel_download(remote_dir, local)
    elapsed = time.time() - t0
    rate = total_bytes / max(elapsed, 0.01) / 1024 / 1024
    print(
        f"Downloaded {n} files ({total_bytes / 1024**2:.1f} MB) in {elapsed:.1f}s ({rate:.1f} MB/s)"
    )


@app.local_entrypoint()
def run(
    local_dir: str = DEFAULT_LOCAL_INPUT,
    out_dir: str = "./output",
    analyze: bool = False,
    annotate: bool = False,
    do_redact: bool = False,
    upload_hf: bool = False,
    hf_repo: str = "",
    hf_private: bool = False,
    export_nii: bool = False,
    compress: bool = False,
):
    """Full pipeline: upload → [redact →] extract → [annotate] → download [→ HF upload].

    --do-redact:  redact first → extract REDACTED files → all output is PHI-free
    --upload-hf:  upload results to Hugging Face as a structured dataset
    Series processing fans out across up to 10 CPU GPU containers in parallel.
    """
    local = Path(local_dir)
    if not local.exists():
        print(f"Error: {local} not found")
        if local_dir == DEFAULT_LOCAL_INPUT:
            print("Default input not found — pass --local-dir")
        return

    t_start = time.time()

    # Step 1: Upload
    print(f"{'=' * 60}")
    print("=== Step 1: Upload ===")
    print(f"{'=' * 60}")
    study_name, upload_stats = _ensure_uploaded(local_dir)

    # Step 2: Redact (optional — runs BEFORE extraction so output is clean)
    redact_result = None
    extract_study = study_name  # which study to extract — original or redacted

    if do_redact:
        print(f"\n{'=' * 60}")
        print("=== Step 2: Redact (HIPAA scan → redact → verify → scan) ===")
        print(f"{'=' * 60}")
        redact_result = run_redaction_pipeline.remote(study_name)
        print(json.dumps(redact_result, indent=2, default=str))

        if not redact_result.get("error"):
            # Extract the REDACTED study, not the original
            extract_study = redact_result.get("redacted_study", f"{study_name}_redacted")
            print(f"\nWill extract redacted study: {extract_study}")
        else:
            print(f"Redaction failed, extracting original: {study_name}")

    # Step 3: Extract (on redacted files if --do-redact, otherwise original)
    print(f"\n{'=' * 60}")
    step_n = 3 if do_redact else 2
    print(f"=== Step {step_n}: Extract '{extract_study}' (2x CPU per container, auto-scaled) ===")
    print(f"{'=' * 60}")
    # HF upload happens server-side inside the pipeline — no separate step needed
    result = run_extraction_pipeline.remote(
        extract_study,
        analyze,
        export_nii,
        compress,
        upload_hf=upload_hf,
        hf_repo=hf_repo,
        hf_private=hf_private,
        annotate=annotate,
    )
    print(json.dumps(result, indent=2))

    if result.get("error"):
        print(f"Extraction failed: {result['error']}")
        return

    # Download
    print(f"\n{'=' * 60}")
    step_n += 1
    print(f"=== Step {step_n}: Download ===")
    print(f"{'=' * 60}")
    dl_local = Path(out_dir)
    dl_local.mkdir(parents=True, exist_ok=True)

    t_dl = time.time()
    n, total_dl = _parallel_download(str(_VOL_OUTPUT / extract_study), dl_local)
    print(f"Downloaded {n} files ({total_dl / 1024**2:.1f} MB) in {time.time() - t_dl:.1f}s")

    if do_redact and redact_result and not redact_result.get("error"):
        redact_remote = str(_VOL_OUTPUT / f"{study_name}_redacted")
        hipaa_dl = dl_local / "redaction"
        hipaa_dl.mkdir(parents=True, exist_ok=True)
        n2, _b2 = _parallel_download(redact_remote, hipaa_dl)
        if n2:
            print(f"Downloaded {n2} redaction files → {hipaa_dl}")

    # Summary
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done in {total_time:.1f}s")
    print(f"  Study:     {study_name}" + (f" → {extract_study} (redacted)" if do_redact else ""))
    print(f"  GPU:       {result.get('gpu', '?')}")
    print(f"  Grade:     {result.get('study_grade', '?')} ({result.get('study_score', 0):.0f}/100)")
    print(
        f"  HIPAA:     {result.get('hipaa_score', '?')}/100 ({result.get('hipaa_phi_findings', '?')} PHI)"
    )
    print(f"  Upload:    {upload_stats.get('uploaded', 0)} new")
    if do_redact and redact_result and not redact_result.get("error"):
        imp = redact_result.get("improvement", {})
        print(
            f"  Redaction: {imp.get('files_redacted', '?')} files, "
            f"post-score {imp.get('post_score', '?')}/100"
        )
    if result.get("hf_url"):
        print(f"  HF:        {result['hf_url']}")
    print(f"  Output:    {out_dir}")
    print(f"{'=' * 60}")


@app.local_entrypoint()
def list_studies():
    """List all studies and results on the volume."""
    for label, remote in [("Studies", str(_VOL_STUDIES)), ("Results", str(_VOL_OUTPUT))]:
        print(f"{label} on volume:{remote}")
        print("-" * 40)
        try:
            for entry in volume.listdir(remote):
                name = PurePosixPath(entry.path).name
                try:
                    files = [
                        f
                        for f in volume.listdir(entry.path, recursive=True)
                        if not f.path.endswith("/")
                    ]
                    print(f"  {name:30s}  ({len(files)} files)")
                except Exception:
                    print(f"  {name}")
        except Exception as e:
            print(f"  (empty: {e})")
        print()


@app.local_entrypoint()
def delete_study(study: str = "", include_output: bool = True):
    """Delete a study (and optionally its output) from the volume."""
    if not study:
        print("Usage: modal run modal_app.py delete-study --study <name>")
        return

    deleted = 0
    paths = [str(_VOL_STUDIES / study)]
    if include_output:
        paths.append(str(_VOL_OUTPUT / study))

    for path in paths:
        try:
            for entry in volume.listdir(path, recursive=True):
                if not entry.path.endswith("/"):
                    volume.remove_file(entry.path)
                    deleted += 1
            print(f"Deleted from {path}")
        except Exception:
            pass

    print(f"Total: {deleted} files deleted")


@app.local_entrypoint()
def cleanup(nuke_all: bool = False):
    """Delete stale data from old uploads and optionally ALL volume data.

    Without --nuke-all: removes data/ prefix (old wrong-path uploads).
    With --nuke-all: removes EVERYTHING (studies/, output/, data/) to free inodes.

    Your volume is at 142% inode usage — run with --nuke-all to clear it.
    """
    prefixes = ["data/studies", "data/output", "data"]
    if nuke_all:
        prefixes = ["studies", "output", "data/studies", "data/output", "data"]

    deleted = 0
    for prefix in prefixes:
        prefix_deleted = 0
        try:
            entries = list(volume.listdir(prefix, recursive=True))
            for entry in entries:
                if not entry.path.endswith("/"):
                    try:
                        volume.remove_file(entry.path)
                        prefix_deleted += 1
                    except Exception:
                        pass
            if prefix_deleted:
                print(f"  Deleted {prefix_deleted} files from {prefix}/")
                deleted += prefix_deleted
        except Exception:
            pass

    print(f"Cleanup done: {deleted} files removed")

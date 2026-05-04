"""Commercial DICOM analysis pipeline — no redaction, Gemma 4 only, ~200GB output.

Reads 355K DICOM files from Modal v2 volume (micom-v2), processes through:

  Stage 1: EXTRACTION
    Montages (3-plane, 6 slices each), histograms (linear + log),
    enhanced views (MIP/MinIP/tissue), quality grades (A-F),
    series stats JSON, DICOM metadata CSV, cross-series comparison

  Stage 2: ADVANCED QUALITY + PER-SLICE EXPORT
    CNR, noise floor, bias field, edge sharpness, histogram separation,
    inter-slice consistency, ML training score (0-100), commercial tier
    Per-slice PNGs: every axial slice + brain window = ~185GB

  Stage 3: GEMMA 4 ANNOTATION (OpenRouter)
    Pass 1: Structured annotation (pathology, anatomy, ML labels, quality)
    Pass 2: Deep tissue analysis (gray/white matter, CSF, vascular, age, disease)
    Pass 3: Synthesis report (§1-§11 radiology dictation + health recommendations)

  Stage 4: DATASET ANALYTICS
    Sequence distribution, protocol completeness, quality breakdown

  Stage 5: HF UPLOAD
    Everything except .html/.tar to HuggingFace (upload_large_folder for 200GB+)

No redaction. No data modification. Raw DICOM data → maximum clinical value.
Uses ONLY Gemma 4 27B via OpenRouter — no cheap models.
Modal Volume v2 — unlimited files, no inode limit.

Usage:
  modal run --detach batch_pipeline.py --skip-upload --repo shubhxho/akai-mri
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path, PurePosixPath

import modal

VOLUME_NAME = "micom-v2"
MOUNT_POINT = PurePosixPath("/vol")
_VOL_STUDIES = PurePosixPath("studies")
_VOL_OUTPUT = PurePosixPath("output")

BATCH_SIZE = 50  # studies per batch (for processing, not upload)
UPLOAD_CHUNK_SIZE = 10000  # files per batch_upload call
UPLOAD_WORKERS = 4

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

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("fonts-dejavu-core")
    .pip_install(*_PIP_DEPS)
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("micom-batch", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


# ── Parallel worker functions ─────────────────────────────────────────────────
#
# Each stage fans out across multiple containers for massive parallelism:
#   - process_study: 1 container per study (20 concurrent) — extraction + quality + slices
#   - annotate_series_worker: 1 call per series (10 concurrent) — Gemma 4 via OpenRouter
#   - process_batch: orchestrator that spawns workers and collects results


@app.function(
    image=image,
    volumes={str(MOUNT_POINT): volume},
    timeout=3600,
    memory=8192,
    cpu=4.0,
)
@modal.concurrent(max_inputs=4)
def process_study(study_dir: str, output_base: str) -> dict:
    """Process ONE study: extract + quality + per-slice PNGs. Runs on its own container."""
    import numpy as np

    study_path = Path(study_dir)
    study_name = study_path.name
    out_dir = Path(output_base) / study_name

    # v2 volume: data should be visible immediately, but retry with reload if not
    if not study_path.exists():
        with contextlib.suppress(Exception):
            volume.reload()
    if not study_path.exists():
        return {"study": study_name, "error": f"not found: {study_dir}"}

    dcm_files = sorted(study_path.rglob("*.dcm"))
    if not dcm_files:
        return {"study": study_name, "error": "no DCM files"}

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Processing {study_name}: {len(dcm_files)} files → {out_dir}")

    # Run extraction pipeline
    try:
        from src.pipeline.run import run_pipeline

        run_pipeline(
            study_path,
            analyze=False,
            export_nii=False,
            out_dir=out_dir,
            workers=4,
            compress=False,
            recursive=True,
        )
    except Exception as e:
        return {
            "study": study_name,
            "error": f"extraction: {e}",
            "files": len(dcm_files),
            "path": str(study_path),
        }

    # Advanced quality + per-slice PNGs
    slice_count = 0
    slice_bytes = 0
    try:
        import json as _json

        import SimpleITK as sitk

        from src.advanced_quality import full_quality_assessment
        from src.export.slice_export import export_all_slices
        from src.helpers import safe_squeeze

        for sd in out_dir.iterdir():
            if not sd.is_dir():
                continue
            detail_files = list(sd.glob("*_detail.json"))
            if not detail_files:
                continue
            try:
                detail = _json.loads(detail_files[0].read_text())
                series_files = detail.get("file_paths", [])
                if not series_files or len(series_files) < 3:
                    continue

                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(series_files[:200])
                reader.SetGlobalWarningDisplay(False)
                vol = sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)
                vol = safe_squeeze(vol)

                if vol.ndim >= 3 and vol.shape[0] >= 3:
                    # Quality
                    aq = full_quality_assessment(vol, detail.get("series_description", ""))
                    detail["advanced_quality"] = aq
                    detail["ml_training_score"] = aq.get("ml_training_score", {})
                    detail_files[0].write_text(_json.dumps(detail, indent=2, default=str))

                    # Per-slice PNGs
                    sr = export_all_slices(
                        vol, sd.name, str(out_dir), windows=["brain"], max_size=512, n_workers=2
                    )
                    slice_count += sr["total_slices"]
                    slice_bytes += sr["total_bytes"]
            except Exception:
                pass
    except Exception:
        pass

    volume.commit()

    return {
        "study": study_name,
        "files": len(dcm_files),
        "slice_pngs": slice_count,
        "slice_gb": round(slice_bytes / 1024**3, 2),
    }


@app.function(
    image=image,
    volumes={str(MOUNT_POINT): volume},
    secrets=[modal.Secret.from_name("openrouter")],
    timeout=300,
    memory=2048,
    cpu=1.0,
)
@modal.concurrent(max_inputs=10)
def annotate_series_worker(
    montage_path: str, series_label: str, quality_ctx: str, ann_dir: str
) -> dict:
    """Annotate ONE series via Gemma 4 on OpenRouter. 10 concurrent calls."""
    import re

    # No volume.reload() — v2 volumes auto-mount latest data
    from src.annotation.cloud import (
        _client,
        _detect_provider,
        annotate_series_multi,
        tissue_analysis_with_model,
    )

    # Pass 1: structured annotation
    result = annotate_series_multi(montage_path, series_label, quality_ctx, models=["gemma4"])

    # Pass 2: tissue analysis
    if result.get("consensus", {}).get("sequence_type"):
        try:
            import json as _json

            provider = _detect_provider()
            client = _client()
            prior = _json.dumps(
                {
                    "sequence": result["consensus"].get("sequence_type", "?"),
                    "pathology_found": result["consensus"].get("pathology", {}).get("found", False),
                },
                default=str,
            )
            tissue = tissue_analysis_with_model(
                client,
                montage_path,
                series_label,
                prior_annotation=prior,
                quality_ctx=quality_ctx,
                model_key="gemma4",
                provider=provider,
            )
            if tissue.get("tissue_analysis"):
                result["tissue_analysis"] = tissue["tissue_analysis"]
        except Exception:
            pass

    # Save
    import json as _json

    safe_name = re.sub(r"[^\w\-]", "_", series_label)
    Path(ann_dir).mkdir(parents=True, exist_ok=True)
    (Path(ann_dir) / f"{safe_name}.json").write_text(_json.dumps(result, indent=2, default=str))
    volume.commit()

    return {
        "series": series_label,
        "sequence": result.get("consensus", {}).get("sequence_type", "?"),
        "pathology": result.get("consensus", {}).get("pathology", {}).get("found", False),
        "has_tissue": bool(result.get("tissue_analysis")),
    }


# ── Orchestrator ─────────────────────────────────────────────────────────────


@app.function(
    volumes={str(MOUNT_POINT): volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("openrouter")],
    timeout=86400,
    memory=16384,
    cpu=8.0,
)
def process_batch(
    batch_name: str, upload_hf: bool = False, hf_repo: str = "", hf_token: str = ""
) -> dict:
    """Parallel pipeline: fan out across 20+ containers for ~2-3 hour completion.

    Architecture:
      Stage 1+2: process_study.map() — 20 containers process studies in parallel
                 Each does: extraction + quality + per-slice PNGs
      Stage 3:   annotate_series_worker.starmap() — 10 concurrent OpenRouter calls
                 Each does: Gemma 4 annotation + tissue analysis
      Stage 4:   Analytics (single container)
      Stage 5:   HF upload

    No redaction. Raw data. Gemma 4 only via OpenRouter.
    """
    import shutil

    volume.reload()

    studies_dir = Path(str(MOUNT_POINT / "studies")) / batch_name
    output_dir = Path(str(MOUNT_POINT / "output")) / batch_name

    if not studies_dir.exists():
        return {"error": f"Batch not found: {studies_dir}"}

    # Find all study subdirectories
    study_subdirs = sorted(
        [d for d in studies_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    # If no subdirs, treat the whole dir as one study
    if not study_subdirs:
        study_subdirs = [studies_dir]

    total_dcm = sum(1 for _ in studies_dir.rglob("*.dcm"))
    print(f"Found {total_dcm} DICOM files across {len(study_subdirs)} studies")

    t_total = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1+2: PARALLEL EXTRACTION + QUALITY + SLICES
    # Fan out: 1 container per study, up to 20 concurrent
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(
        f"STAGE 1+2: Parallel extraction ({len(study_subdirs)} studies, 20 concurrent containers)"
    )
    print(f"{'=' * 60}")

    study_args = [(str(sd), str(output_dir)) for sd in study_subdirs]
    study_results = list(process_study.starmap(study_args))

    total_files = sum(r.get("files", 0) for r in study_results)
    total_slices = sum(r.get("slice_pngs", 0) for r in study_results)
    total_slice_gb = sum(r.get("slice_gb", 0) for r in study_results)
    errors = [r for r in study_results if r.get("error")]

    print(f"\n  Processed: {total_files} files across {len(study_results)} studies")
    print(f"  Slice PNGs: {total_slices} ({total_slice_gb:.1f} GB)")
    if errors:
        print(f"  Errors: {len(errors)} studies failed")
        for e in errors[:5]:
            print(f"    {e.get('study', '?')}: {e.get('error', '?')} (path: {e.get('path', '')})")

    volume.reload()  # See all the outputs from parallel workers

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: PARALLEL GEMMA 4 ANNOTATION (10 concurrent OpenRouter calls)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("STAGE 3: Gemma 4 annotation via OpenRouter (10 concurrent)")
    print(f"{'=' * 60}")

    ann_results = []
    try:
        if os.environ.get("OPENROUTER_API_KEY"):
            from src.annotation.local import _build_quality_context, _is_derivative

            ann_dir = str(output_dir / "annotations")
            Path(ann_dir).mkdir(parents=True, exist_ok=True)

            # Collect all series montages across all study outputs
            ann_args = []
            for sd in output_dir.iterdir():
                if not sd.is_dir():
                    continue
                montages = list(sd.glob("*_multiplane.png"))
                if not montages:
                    continue

                info_parts = sd.name.split("_", 1)
                snum = info_parts[0].replace("s", "") if info_parts else "?"
                sdesc = info_parts[1] if len(info_parts) > 1 else ""
                label = f"Series {snum} — {sdesc}"

                if _is_derivative(label):
                    continue

                detail_files = list(sd.glob("*_detail.json"))
                qa = None
                if detail_files:
                    try:
                        d = json.loads(detail_files[0].read_text())
                        qa = d.get("quality_analysis")
                    except Exception:
                        pass

                quality_ctx = _build_quality_context(qa) if qa else ""
                ann_args.append((str(montages[0]), label, quality_ctx, ann_dir))

            if ann_args:
                print(f"  Annotating {len(ann_args)} series (10 concurrent)...")
                ann_results = list(annotate_series_worker.starmap(ann_args))

                pathology_count = sum(1 for r in ann_results if r.get("pathology"))
                tissue_count = sum(1 for r in ann_results if r.get("has_tissue"))
                print(
                    f"  Done: {len(ann_results)} annotated, {pathology_count} with pathology, {tissue_count} with tissue analysis"
                )

                # Save summary
                (Path(ann_dir) / "study_annotations.json").write_text(
                    json.dumps(
                        {"series_annotated": len(ann_results), "results": ann_results},
                        indent=2,
                        default=str,
                    )
                )
            else:
                print("  No primary series found")
        else:
            print("  OPENROUTER_API_KEY not set — skipping")
    except Exception as e:
        print(f"  Annotation failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4: ANALYTICS
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== Stage 4: Analytics ===")
    try:
        analytics_dir = output_dir / "analytics"
        analytics_dir.mkdir(parents=True, exist_ok=True)
        analytics = {
            "study_name": batch_name,
            "total_files": total_dcm,
            "total_studies": len(study_subdirs),
            "total_slice_pngs": total_slices,
            "total_slice_gb": total_slice_gb,
            "series_annotated": len(ann_results),
            "pathology_detected": sum(1 for r in ann_results if r.get("pathology")),
        }
        (analytics_dir / "dataset_analytics.json").write_text(json.dumps(analytics, indent=2))
        print("  Analytics saved")
    except Exception as e:
        print(f"  Analytics failed: {e}")

    volume.commit()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5: HF UPLOAD
    # ══════════════════════════════════════════════════════════════════════
    hf_url = None
    if upload_hf and hf_repo:
        print(f"\n{'=' * 60}")
        print("STAGE 5: Upload to HF")
        print(f"{'=' * 60}")
        try:
            from huggingface_hub import HfApi, create_repo

            token = hf_token or os.environ.get("HF_TOKEN", "")
            if token:
                api = HfApi(token=token)
                create_repo(hf_repo, repo_type="dataset", private=False, exist_ok=True, token=token)

                staging = Path(tempfile.mkdtemp())
                SKIP_EXT = {".html", ".htm", ".tar"}
                n_staged = 0
                for f in output_dir.rglob("*"):
                    if not f.is_file() or f.name.startswith("."):
                        continue
                    if f.suffix.lower() in SKIP_EXT:
                        continue
                    if f.stat().st_size > 500 * 1024 * 1024:
                        continue
                    if "_internal" in str(f):
                        continue
                    rel = f.relative_to(output_dir)
                    dest = staging / batch_name / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
                    n_staged += 1

                if n_staged:
                    os.environ["HF_HOME"] = str(staging / ".hf_cache")
                    staged_gb = (
                        sum(f.stat().st_size for f in Path(staging).rglob("*") if f.is_file())
                        / 1024**3
                    )
                    print(f"  {n_staged} files ({staged_gb:.1f} GB)")

                    if n_staged > 1000 or staged_gb > 5:
                        api.upload_large_folder(
                            folder_path=str(staging), repo_id=hf_repo, repo_type="dataset"
                        )
                    else:
                        api.upload_folder(
                            folder_path=str(staging),
                            repo_id=hf_repo,
                            repo_type="dataset",
                            token=token,
                            commit_message=f"Add {batch_name}: {n_staged} files",
                        )
                    hf_url = f"https://huggingface.co/datasets/{hf_repo}"
                    print(f"  → {hf_url}")
                shutil.rmtree(staging, ignore_errors=True)
        except Exception as e:
            print(f"  HF upload failed: {e}")

    volume.commit()

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed / 60:.0f} min")
    print(
        f"  Files: {total_files} | Studies: {len(study_results)} | Slices: {total_slices} ({total_slice_gb:.1f}GB)"
    )
    print(
        f"  Annotated: {len(ann_results)} series | Pathology: {sum(1 for r in ann_results if r.get('pathology'))}"
    )
    if hf_url:
        print(f"  HF: {hf_url}")
    print(f"{'=' * 60}")

    return {
        "batch": batch_name,
        "files": total_files,
        "studies": len(study_results),
        "slices": total_slices,
        "slice_gb": total_slice_gb,
        "annotated": len(ann_results),
        "time_min": round(elapsed / 60, 1),
        "hf_url": hf_url,
    }


@app.function(
    volumes={str(MOUNT_POINT): volume},
    timeout=300,
    memory=2048,
)
def cleanup_batch(batch_name: str) -> dict:
    """Delete a batch from the volume to free inodes for the next batch."""
    volume.reload()
    deleted = 0

    for prefix in [
        str(_VOL_STUDIES / batch_name),
        str(_VOL_STUDIES / f"{batch_name}_extracted"),
        str(_VOL_OUTPUT / batch_name),
        str(_VOL_OUTPUT / f"{batch_name}_processed"),
    ]:
        try:
            for entry in volume.listdir(prefix, recursive=True):
                if not entry.path.endswith("/"):
                    volume.remove_file(entry.path)
                    deleted += 1
        except Exception:
            pass

    volume.commit()
    print(f"Cleaned batch {batch_name}: {deleted} files removed")
    return {"batch": batch_name, "deleted": deleted}


# ── Local orchestrator ───────────────────────────────────────────────────────


def _find_study_dirs(root: Path) -> list[Path]:
    """Find all directories that contain DICOM files."""
    studies = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("."):
            continue
        if d.suffix == ".zip":
            continue
        if any(d.rglob("*.dcm")):
            studies.append(d)
    return studies


def _upload_files_direct(study_dirs: list[Path], study_name: str, root: Path):
    """Upload DICOM files directly to v2 volume (no tars needed — unlimited inodes)."""

    all_files = []
    for sd in study_dirs:
        for f in sorted(sd.rglob("*.dcm")):
            rel = str(f.relative_to(root))
            all_files.append((f, rel))

    if not all_files:
        print("  No DICOM files found")
        return

    total_bytes = sum(f.stat().st_size for f, _ in all_files)
    chunks = [
        all_files[i : i + UPLOAD_CHUNK_SIZE] for i in range(0, len(all_files), UPLOAD_CHUNK_SIZE)
    ]
    print(f"  {len(all_files)} files ({total_bytes / 1024**3:.1f}GB), {len(chunks)} chunks")

    t0 = time.time()
    uploaded = 0
    failed = 0

    for chunk_idx, chunk in enumerate(chunks):
        try:
            with volume.batch_upload(force=True) as batch:
                for local_path, rel in chunk:
                    remote = str(_VOL_STUDIES / study_name / rel)
                    batch.put_file(local_path, remote)
            uploaded += len(chunk)
        except Exception as e:
            print(f"  [ERROR] chunk {chunk_idx}: {e}")
            failed += len(chunk)

        elapsed = time.time() - t0
        rate = total_bytes * (uploaded / max(len(all_files), 1)) / max(elapsed, 0.01) / 1024**2
        print(f"  [{uploaded}/{len(all_files)}] {rate:.1f} MB/s ({failed} failed)")

    print(f"  Upload done: {uploaded} files in {time.time() - t0:.0f}s")


def _get_hf_token() -> str:
    """Read HF token."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token() or ""
        except Exception:
            pass
    if not token:
        p = Path.home() / ".cache" / "huggingface" / "token"
        if p.exists():
            token = p.read_text().strip()
    return token


@app.local_entrypoint()
def main(
    input: str = "/Volumes/T7 Shield/Akai - MRI",
    repo: str = "shubhxho/akai-mri",
    batch_size: int = BATCH_SIZE,
    skip_upload: bool = False,
):
    """Process entire Akai MRI folder in streaming batches.

    Upload → extract → annotate (Gemma 4) → quality → upload to HF.
    No redaction. Raw data preserved as-is.
    """
    root = Path(input)
    if not root.exists():
        print(f"Error: {root} not found")
        return

    study_dirs = _find_study_dirs(root)
    print(f"Found {len(study_dirs)} study directories in {root.name}")
    print(f"Processing batch size: {batch_size} studies")
    print(f"HF repo: {repo}")
    print(f"Volume: {VOLUME_NAME} (v2 — unlimited files)")
    print()

    hf_token = _get_hf_token()
    t_start = time.time()

    # ── Step 1: Upload ALL files at once (v2 volume has no inode limit) ──
    if skip_upload:
        print("Skipping upload — files already on volume")
    else:
        print(f"{'=' * 60}")
        print(f"=== Step 1: Upload all {len(study_dirs)} studies to v2 volume ===")
        print(f"{'=' * 60}")
        _upload_files_direct(study_dirs, "akai_mri", root)

    # ── Step 2: Process (extract + annotate + quality + HF upload) ──────
    total_batches = (len(study_dirs) + batch_size - 1) // batch_size
    total_processed = 0
    total_failed = 0

    for batch_idx in range(0, len(study_dirs), batch_size):
        study_dirs[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        batch_name = "akai_mri"  # all files are under one study name

        print(f"\n{'=' * 60}")
        print(f"=== Step 2: Process batch {batch_num}/{total_batches} ===")
        print(f"{'=' * 60}")

        t_batch = time.time()
        try:
            result = process_batch.remote(
                batch_name,
                upload_hf=True,
                hf_repo=repo,
                hf_token=hf_token,
            )
            print(f"  Result: {json.dumps(result, indent=2)}")
            total_processed += result.get("files_input", 0)
        except Exception as e:
            print(f"  ERROR: {e}")
            total_failed += 1

        elapsed_batch = time.time() - t_batch
        elapsed_total = time.time() - t_start
        remaining = total_batches - batch_num
        eta = (elapsed_total / max(batch_num, 1)) * remaining

        print(f"  Batch {batch_num} done in {elapsed_batch:.0f}s, ETA {eta / 60:.0f}min")

        # Only process the first batch since all files are under one study
        # The process_batch function handles all files in the study
        break

    # Final summary
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"ALL DONE in {elapsed / 60:.0f} minutes")
    print(f"  Studies: {len(study_dirs)}")
    print(f"  Files processed: {total_processed}")
    print(f"  Failed: {total_failed}")
    print(f"  Volume: {VOLUME_NAME} (v2)")
    print(f"  HF dataset: https://huggingface.co/datasets/{repo}")
    print(f"{'=' * 60}")

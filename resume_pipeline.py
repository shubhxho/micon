"""Resume pipeline -- runs only the stages that haven't already finished.

Reads detail.json + montages already on the `micom-v2` volume from the
extraction pass and brings the dataset to a shippable state in four stages:

  Stage 2  advanced_quality + per-slice PNGs   (per-series, parallel)
  Stage 3  Gemma 4 annotation via OpenRouter   (per-series, parallel)
  Stage 4  pack each study's slice PNGs into one tar shard  (per-study)
  Stage 5  upload to Hugging Face              (one orchestrator pass)

Each stage is idempotent. Skip flags let you re-run only the tail:
  --skip-quality --skip-annotation         resume from packing onwards
  --skip-quality --skip-annotation --skip-pack   upload only

New: --force bypasses sentinel-based short-circuit (default: skip stages
already marked ok by a prior run).

Usage:
  modal run --detach resume_pipeline.py
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import modal

# Single source of truth for sanitising series labels into filenames.
# Used by both annotate_one (writer) and the orchestrator (existing-set check)
# so they can never drift.
_SAFE_NAME_RE = re.compile(r"[^\w\-]")


def _safe_name(label: str) -> str:
    return _SAFE_NAME_RE.sub("_", label)


VOLUME_NAME = "micom-v2"
MOUNT_POINT = PurePosixPath("/vol")

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

app = modal.App("micom-resume", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


# ── Worker: advanced quality + per-slice PNGs for ONE series ────────────────
#
# Fans out across many containers via .map(). Each worker reads one detail.json,
# loads the SimpleITK volume, runs quality assessment, exports per-slice PNGs,
# and writes detail.json back. No volume.commit() per worker — the orchestrator
# commits once after .map() returns. Failed series log path + exception type.


@app.function(
    image=image,
    volumes={str(MOUNT_POINT): volume},
    timeout=600,
    memory=4096,
    cpu=2.0,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
)
@modal.concurrent(max_inputs=2)
def quality_one_series(detail_path_str: str, output_dir_str: str) -> dict:
    """Run advanced quality + slice export for a single series.

    Returns {ok, skipped, slices, bytes, path, error}. Skipped means the series
    already had advanced_quality (idempotent re-runs are cheap). `output_dir_str`
    is the dataset root so we can compute a deterministic study_id and rewrite
    file_paths to study-relative form (no /vol/... leaks in published JSON).
    """
    import numpy as np
    import SimpleITK as sitk

    detail_path = Path(detail_path_str)
    output_dir = Path(output_dir_str)
    out = {
        "ok": False,
        "skipped": False,
        "slices": 0,
        "bytes": 0,
        "path": detail_path_str,
        "error": None,
    }

    try:
        from src.advanced_quality import full_quality_assessment
        from src.export.slice_export import export_all_slices
        from src.helpers import safe_squeeze

        detail = json.loads(detail_path.read_text())

        if detail.get("advanced_quality"):
            out["skipped"] = True
            return out

        series_files = detail.get("file_paths", [])
        if not series_files or len(series_files) < 3:
            # Localizers, color MIPs, projections, processed maps all routinely
            # have 1-2 files. Not an error -- mark as skipped so the failure
            # breakdown reflects only real problems.
            out["skipped"] = True
            out["skip_reason"] = f"too_few_files ({len(series_files)})"
            return out

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_files[:200])
        reader.SetGlobalWarningDisplay(False)
        vol = sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)
        vol = safe_squeeze(vol)

        if vol.ndim < 3 or vol.shape[0] < 3:
            out["error"] = f"bad shape {tuple(vol.shape)}"
            return out

        aq = full_quality_assessment(vol, detail.get("series_description", ""))
        detail["advanced_quality"] = aq
        detail["ml_training_score"] = aq.get("ml_training_score", {})

        # study_id: deterministic from path relative to output_dir
        try:
            study_id = detail_path.relative_to(output_dir).parts[0]
        except (ValueError, IndexError):
            study_id = detail_path.parent.parent.name
        detail["study_id"] = study_id

        # pipeline_version: timestamp + best-effort git sha from env
        detail["pipeline_version"] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_sha": os.environ.get("GIT_SHA", "unknown"),
        }

        detail_path.write_text(json.dumps(detail, indent=2, default=str))

        series_dir = detail_path.parent
        sr = export_all_slices(
            vol,
            series_dir.name,
            str(series_dir.parent),
            windows=["brain"],
            max_size=512,
            n_workers=2,
        )
        out["slices"] = sr["total_slices"]
        out["bytes"] = sr["total_bytes"]
        out["ok"] = True
        return out

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        print(f"  [WARN] {detail_path.name}: {out['error']}", flush=True)
        return out


# ── Worker: pack one study's slice PNGs into a single tar shard ─────────────
#
# HF cannot ingest 1M+ tiny PNGs in a single repo at any reasonable speed --
# the per-file recovery / hash pass takes 80+ hours at ~5 files/sec because
# overhead dominates. Packing each study's slices into one tar drops the file
# count from 1M to ~1,053 (one per study), which HF handles cleanly. Tarball
# is uncompressed (PNG is already compressed) so the wall time per worker is
# effectively just disk I/O.
#
# Naming convention: {study_id}.slices.tar at the root of the study directory.
# Buyers can stream the tarballs with webdataset / tarfile without unpacking.


@app.function(
    image=image,
    volumes={str(MOUNT_POINT): volume},
    timeout=1800,
    memory=2048,
    cpu=2.0,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
)
@modal.concurrent(max_inputs=4)
def pack_one_study_slices(study_dir_str: str) -> dict:
    """Walk one study directory, tar every slice PNG into a single shard.

    Returns {ok, study, tar_path, n_slices, bytes, skipped, error}. Skipped =
    tarball already exists and is non-empty (idempotent).
    """
    import tarfile

    study_dir = Path(study_dir_str)
    out = {
        "ok": False,
        "skipped": False,
        "study": study_dir.name,
        "tar_path": None,
        "n_slices": 0,
        "bytes": 0,
        "error": None,
    }

    tar_path = study_dir / f"{study_dir.name}.slices.tar"

    # Layout from src/export/slice_export.py:
    #   <study>/slices/<series>/axial_NNNN.png
    #   <study>/slices/<series>/coronal_NNNN.png
    #   <study>/slices/<series>/sagittal_NNNN.png
    #   <study>/slices/<series>/<window>/axial_NNNN.png  (e.g. brain/, bone/)
    # rglob("slices/**/*.png") catches all of them in one sweep.
    slices_root = study_dir / "slices"
    if not slices_root.exists():
        out["skipped"] = True
        out["error"] = "no_slices_dir"
        return out

    slice_pngs = list(slices_root.rglob("*.png"))
    if not slice_pngs:
        out["skipped"] = True
        out["error"] = "empty_slices_dir"
        return out

    # Idempotent: skip iff shard exists, opens cleanly, and entry count
    # matches the current source directory exactly. Strict `==` (not `>=`)
    # so a later slice_export pass that adds windows (e.g. bone) forces a
    # re-pack instead of leaving buyers with a stale tar.
    if tar_path.exists() and tar_path.stat().st_size > 0:
        try:
            with tarfile.open(tar_path, "r") as tf:
                n = sum(1 for _ in tf)
            if n == len(slice_pngs):
                out["skipped"] = True
                out["n_slices"] = n
                out["bytes"] = tar_path.stat().st_size
                out["tar_path"] = str(tar_path)
                return out
            # Mismatch: stale or partial -- re-pack
            tar_path.unlink(missing_ok=True)
        except Exception:
            tar_path.unlink(missing_ok=True)

    try:
        # Stream into tar -- uncompressed (PNG is already compressed).
        with tarfile.open(tar_path, "w") as tf:
            for png in slice_pngs:
                # arcname is study-relative: slices/<series>/.../*.png
                arcname = str(png.relative_to(study_dir))
                tf.add(str(png), arcname=arcname, recursive=False)

        out["ok"] = True
        out["tar_path"] = str(tar_path)
        out["n_slices"] = len(slice_pngs)
        out["bytes"] = tar_path.stat().st_size
        return out

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        if tar_path.exists():
            tar_path.unlink(missing_ok=True)
        return out


# ── Worker: annotate one series via OpenRouter ───────────────────────────────


@app.cls(
    image=image,
    volumes={str(MOUNT_POINT): volume},
    secrets=[modal.Secret.from_name("openrouter")],
    # 900s ceiling -- the slowest OpenRouter response we've seen is ~150s,
    # but bursts during peak hours stack. Workers should never hit this.
    timeout=900,
    memory=1024,
    cpu=0.5,
    retries=modal.Retries(max_retries=2, initial_delay=4.0, backoff_coefficient=2.0),
)
# Higher fan-out -- each worker is mostly waiting on HTTP, so 32 in-flight
# requests per container is comfortable. The class form lets us reuse the
# OpenAI client + module imports across all 32 concurrent calls in a warm
# container instead of paying the import cost once per call.
@modal.concurrent(max_inputs=32)
class Annotator:
    """Cost-aware annotation worker.

    @modal.enter() runs once per container boot; the OpenAI/OpenRouter client
    + provider detection persist across every annotate() call, so a warm
    container amortizes setup over thousands of montages.
    """

    @modal.enter()
    def setup(self):
        from src.annotation.cloud import _client, _detect_provider

        self.provider = _detect_provider()
        self.client = _client()

    @modal.method()
    def annotate(
        self, montage_path: str, series_label: str, quality_ctx: str, ann_dir: str
    ) -> dict:
        """Annotate ONE series. Annotation + tissue prompts race in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        from src.annotation.cloud import (
            _default_lineup,
            annotate_series_multi,
            tissue_analysis_with_model,
        )

        # Cheap-tier lineup unless MICOM_PREMIUM=1 is set on the secret.
        lineup = _default_lineup()
        primary = lineup[0] if lineup else "kimi"

        def _do_annotation():
            return annotate_series_multi(montage_path, series_label, quality_ctx, models=lineup)

        def _do_tissue():
            try:
                prior = json.dumps({"sequence_hint": "see montage"})
                tissue = tissue_analysis_with_model(
                    self.client,
                    montage_path,
                    series_label,
                    prior_annotation=prior,
                    quality_ctx=quality_ctx,
                    model_key=primary,
                    provider=self.provider,
                )
                return tissue.get("tissue_analysis") if tissue else None
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=2) as inner:
            f_annot = inner.submit(_do_annotation)
            f_tissue = inner.submit(_do_tissue)
            result = f_annot.result()
            tissue = f_tissue.result()

        if tissue:
            result["tissue_analysis"] = tissue

        # Orchestrator commits the volume once per stage -- per-worker commits
        # would saturate the metadata service with no benefit at max_inputs=32.
        Path(ann_dir).mkdir(parents=True, exist_ok=True)
        (Path(ann_dir) / f"{_safe_name(series_label)}.json").write_text(
            json.dumps(result, indent=2, default=str)
        )

        consensus = result.get("consensus", {})
        return {
            "series": series_label,
            "sequence": consensus.get("sequence_type", "?"),
            "pathology": consensus.get("pathology", {}).get("found", False),
            "has_tissue": bool(result.get("tissue_analysis")),
            "models_used": result.get("models_used", []),
        }


# ── Main resume function ─────────────────────────────────────────────────────


@app.function(
    image=image,
    volumes={str(MOUNT_POINT): volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("openrouter")],
    timeout=86400,
    memory=16384,
    cpu=8.0,
)
def resume(
    hf_repo: str = "",
    skip_quality: bool = False,
    skip_annotation: bool = False,
    skip_pack: bool = False,
    force: bool = False,
) -> dict:
    """Resume pipeline -- quality + annotation + slice-pack + HF upload."""
    from src.pipeline.log import log, log_error
    from src.pipeline.sentinels import is_done, mark_finished, mark_started, summarize

    volume.reload()
    t0 = time.time()

    output_dir = Path(MOUNT_POINT / "output" / "akai_mri")
    if not output_dir.exists():
        return {"error": f"Output not found: {output_dir}"}

    # ── Find ALL montages recursively ────────────────────────────────────
    print("Scanning for existing montages...")
    all_montages = list(output_dir.rglob("*_multiplane.png"))
    print(f"Found {len(all_montages)} montages across all studies")
    log("startup", "found_montages", n=len(all_montages))

    # Find all detail.json files
    all_details = list(output_dir.rglob("*_detail.json"))
    print(f"Found {len(all_details)} detail.json files")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Advanced quality + per-slice PNGs — FANNED OUT across containers
    # ══════════════════════════════════════════════════════════════════════
    # Each detail.json represents one series. Workers run independently:
    # read detail.json → load volume → quality → slice export → write back.
    # Modal autoscales container count based on demand. ~50-100x faster than
    # the previous sequential single-container loop.
    if not skip_quality:
        if not force and is_done(output_dir, "stage2_quality"):
            print("  Stage 2 already done (sentinel ok). Use --force to re-run.")
        else:
            print(f"\n{'=' * 60}")
            print("STAGE 2: Advanced quality + per-slice PNGs (parallel .map)")
            print(f"{'=' * 60}")

            # Pre-filter: skip details that already have advanced_quality.
            # Avoids paying for a container that immediately returns "skipped".
            already_done = 0
            pending: list[str] = []
            for dp in all_details:
                try:
                    if json.loads(dp.read_text()).get("advanced_quality"):
                        already_done += 1
                        continue
                except Exception:
                    pass
                pending.append(str(dp))

            print(f"  {already_done} series already have advanced_quality (skipping)")
            print(f"  {len(pending)} series to process — fanning out via .map()")

            mark_started(output_dir, "stage2_quality", metadata={"pending": len(pending)})
            log("stage2_quality", "stage_started", pending=len(pending))

            try:
                if pending:
                    t_q = time.time()
                    # starmap so each worker also receives output_dir; needed to
                    # compute study_id deterministically and rewrite file_paths
                    # to study-relative form.
                    args = [(p, str(output_dir)) for p in pending]
                    results = list(quality_one_series.starmap(args, return_exceptions=True))
                    # Coerce any worker exceptions into error dicts so the rest of
                    # the summary code (which calls .get() on each result) works.
                    results = [
                        r
                        if not isinstance(r, BaseException)
                        else {
                            "ok": False,
                            "skipped": False,
                            "error": f"{type(r).__name__}: {r}",
                            "slices": 0,
                            "bytes": 0,
                            "path": "?",
                        }
                        for r in results
                    ]

                    ok = sum(1 for r in results if r.get("ok"))
                    skipped = sum(1 for r in results if r.get("skipped"))
                    errored = [r for r in results if r.get("error")]
                    total_slices = sum(r.get("slices", 0) for r in results)
                    total_slice_bytes = sum(r.get("bytes", 0) for r in results)
                    elapsed = time.time() - t_q

                    print(
                        f"  Quality done in {elapsed / 60:.1f} min: "
                        f"{ok} processed, {skipped} skipped, {len(errored)} failed"
                    )
                    print(f"  Slices: {total_slices} PNGs ({total_slice_bytes / 1024**3:.1f} GB)")
                    log(
                        "stage2_quality",
                        "stage_complete",
                        elapsed_s=elapsed,
                        ok=ok,
                        skipped=skipped,
                        failed=len(errored),
                    )

                    if errored:
                        # Log a sample of errors for diagnosis (don't spam)
                        from collections import Counter

                        err_types = Counter(
                            r["error"].split(":")[0] for r in errored if r.get("error")
                        )
                        print(f"  Failure breakdown: {dict(err_types.most_common(10))}")
                        for r in errored:
                            err_str = r.get("error") or ""
                            err_class, _, err_msg = err_str.partition(":")
                            log(
                                "stage2_quality",
                                "worker_failed",
                                error_class=err_class.strip(),
                                error_msg=err_msg.strip(),
                                path=Path(r.get("path", "?")).name,
                            )

                    mark_finished(
                        output_dir,
                        "stage2_quality",
                        ok=True,
                        errors=len(errored),
                        inputs_processed=ok,
                        metadata={"processed": ok, "skipped": skipped},
                    )
                    # Single commit after the whole map completes -- workers don't commit.
                    # Reload so the orchestrator's mount sees the worker writes
                    # before the next stage scans the directory.
                    volume.commit()
                    volume.reload()
                else:
                    print("  Nothing to do -- all series already processed")
                    mark_finished(
                        output_dir,
                        "stage2_quality",
                        ok=True,
                        errors=0,
                        inputs_processed=0,
                        metadata={"processed": 0, "skipped": already_done},
                    )

            except Exception as e:
                log_error("stage2_quality", "stage_failed", e)
                mark_finished(output_dir, "stage2_quality", ok=False, errors=1)
                volume.commit()
                raise

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: Gemma 4 annotation (only series without annotations)
    # ══════════════════════════════════════════════════════════════════════
    ann_results = []
    if not skip_annotation:
        if not force and is_done(output_dir, "stage3_annotation"):
            print("  Stage 3 already done (sentinel ok). Use --force to re-run.")
        else:
            print(f"\n{'=' * 60}")
            print("STAGE 3: Gemma 4 annotation via OpenRouter")
            print(f"{'=' * 60}")

            mark_started(output_dir, "stage3_annotation", metadata={})
            log("stage3_annotation", "stage_started")

            try:
                if os.environ.get("OPENROUTER_API_KEY"):
                    from src.annotation.local import _build_quality_context, _is_derivative

                    ann_dir = str(output_dir / "annotations")
                    Path(ann_dir).mkdir(parents=True, exist_ok=True)

                    # Check which series already have annotations
                    existing_anns = set()
                    for f in Path(ann_dir).glob("*.json"):
                        if f.name != "study_annotations.json":
                            existing_anns.add(f.stem)

                    # Build annotation args from ALL montages (recursive).
                    ann_args = []
                    derivative_skips = 0
                    for montage in all_montages:
                        series_dir = montage.parent
                        series_name = series_dir.name

                        # Extract series number and description
                        parts = series_name.split("_", 1)
                        snum = parts[0].replace("s", "") if parts else "?"
                        sdesc = parts[1] if len(parts) > 1 else ""
                        # Em-dash, NOT "--", because the safe-name -> filename
                        # mapping is part of the idempotency check against the
                        # ~3,400 annotation files already on disk. Existing files
                        # are `Series_NNNN___<desc>.json` (em-dash + spaces ->
                        # three underscores). Changing the dash here would orphan
                        # them and trigger a full re-annotation pass.
                        label = f"Series {snum} \u2014 {sdesc}"

                        if _is_derivative(label):
                            derivative_skips += 1
                            continue

                        if _safe_name(label) in existing_anns:
                            continue

                        # Get quality context
                        detail_files = list(series_dir.glob("*_detail.json"))
                        qa = None
                        if detail_files:
                            try:
                                d = json.loads(detail_files[0].read_text())
                                qa = d.get("quality_analysis")
                            except Exception:
                                pass

                        quality_ctx = _build_quality_context(qa) if qa else ""
                        ann_args.append((str(montage), label, quality_ctx, ann_dir))

                    print(
                        f"  {len(ann_args)} series to annotate "
                        f"({len(existing_anns)} already done, "
                        f"{derivative_skips} derivatives skipped)"
                    )

                    if ann_args:
                        t_a = time.time()
                        print("  Running 32 concurrent OpenRouter calls per worker...")
                        # Iterate the map -- per-input failures don't kill the run.
                        # return_exceptions=True surfaces the failed items as
                        # exception objects we count instead of crashing the whole
                        # stage on the first hung worker.
                        ann_results = []
                        failed = 0
                        annotator = Annotator()  # pyright: ignore[reportCallIssue]  # Modal PartialFunction
                        for r in annotator.annotate.starmap(ann_args, return_exceptions=True):
                            if isinstance(r, BaseException):
                                failed += 1
                                log(
                                    "stage3_annotation",
                                    "worker_failed",
                                    error_class=type(r).__name__,
                                    error_msg=str(r),
                                )
                            else:
                                ann_results.append(r)

                        pathology_count = sum(1 for r in ann_results if r.get("pathology"))
                        tissue_count = sum(1 for r in ann_results if r.get("has_tissue"))
                        elapsed = (time.time() - t_a) / 60
                        print(
                            f"  Annotated: {len(ann_results)} in {elapsed:.1f} min "
                            f"(failed {failed}, pathology {pathology_count}, tissue {tissue_count})"
                        )
                        log(
                            "stage3_annotation",
                            "stage_complete",
                            elapsed_s=elapsed * 60,
                            ok=len(ann_results),
                            skipped=len(existing_anns),
                            failed=failed,
                        )

                        (Path(ann_dir) / "study_annotations.json").write_text(
                            json.dumps(
                                {
                                    "series_annotated": len(ann_results) + len(existing_anns),
                                    "annotation_failures": failed,
                                    "results": ann_results,
                                },
                                indent=2,
                                default=str,
                            )
                        )

                        mark_finished(
                            output_dir,
                            "stage3_annotation",
                            ok=True,
                            errors=failed,
                            inputs_processed=len(ann_results),
                            metadata={"processed": len(ann_results), "skipped": len(existing_anns)},
                        )
                        volume.commit()
                        volume.reload()

                    else:
                        mark_finished(
                            output_dir,
                            "stage3_annotation",
                            ok=True,
                            errors=0,
                            inputs_processed=0,
                            metadata={"processed": 0, "skipped": len(existing_anns)},
                        )

                else:
                    print("  OPENROUTER_API_KEY not set")
                    mark_finished(
                        output_dir,
                        "stage3_annotation",
                        ok=True,
                        errors=0,
                        metadata={"skipped_reason": "no_api_key"},
                    )

            except Exception as e:
                log_error("stage3_annotation", "stage_failed", e)
                mark_finished(output_dir, "stage3_annotation", ok=False, errors=1)
                volume.commit()
                raise

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4: Pack per-study slice PNGs into tar shards
    # ══════════════════════════════════════════════════════════════════════
    # Run via .map() across all studies. Each worker writes one
    # `<study_id>.slices.tar` at the study root containing every slice PNG
    # in that study. Idempotent: workers skip studies that already have a
    # valid tar. Drops upload file count from ~1M to ~1,053.
    pack_summary = {}
    if not skip_pack:
        if not force and is_done(output_dir, "stage4_pack"):
            print("  Stage 4 already done (sentinel ok). Use --force to re-run.")
        else:
            print(f"\n{'=' * 60}")
            print("STAGE 4: Pack per-study slices into tar shards")
            print(f"{'=' * 60}")

            # Studies are direct children of output_dir.
            study_dirs = [
                str(p)
                for p in output_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".") and p.name != "_internal"
            ]
            print(f"  {len(study_dirs)} studies to pack")

            mark_started(output_dir, "stage4_pack", metadata={"studies": len(study_dirs)})
            log("stage4_pack", "stage_started", studies=len(study_dirs))

            try:
                if study_dirs:
                    t_p = time.time()
                    packed, skipped, failed = 0, 0, 0
                    total_slices, total_bytes = 0, 0
                    errors = []
                    for r in pack_one_study_slices.map(study_dirs, return_exceptions=True):
                        if isinstance(r, BaseException):
                            failed += 1
                            log(
                                "stage4_pack",
                                "worker_failed",
                                error_class=type(r).__name__,
                                error_msg=str(r),
                            )
                            continue
                        if r.get("ok"):
                            packed += 1
                        elif r.get("skipped"):
                            skipped += 1
                        else:
                            failed += 1
                            if r.get("error"):
                                errors.append(r["error"])
                                err_class, _, err_msg = r["error"].partition(":")
                                log(
                                    "stage4_pack",
                                    "worker_failed",
                                    error_class=err_class.strip(),
                                    error_msg=err_msg.strip(),
                                    study=r.get("study", "?"),
                                )
                        total_slices += r.get("n_slices", 0)
                        total_bytes += r.get("bytes", 0)

                    elapsed = (time.time() - t_p) / 60
                    print(
                        f"  Packed in {elapsed:.1f} min: {packed} new, {skipped} already done, {failed} failed"
                    )
                    print(
                        f"  Total: {total_slices:,} slices in {total_bytes / 1024**3:.1f} GB across study tars"
                    )
                    log(
                        "stage4_pack",
                        "stage_complete",
                        elapsed_s=elapsed * 60,
                        ok=packed,
                        skipped=skipped,
                        failed=failed,
                    )

                    if errors:
                        from collections import Counter

                        err_types = Counter(e.split(":")[0] for e in errors)
                        print(f"  Failure breakdown: {dict(err_types.most_common(10))}")

                    pack_summary = {
                        "studies_packed": packed,
                        "studies_skipped": skipped,
                        "studies_failed": failed,
                        "total_slices": total_slices,
                        "total_bytes": total_bytes,
                    }
                    mark_finished(
                        output_dir,
                        "stage4_pack",
                        ok=True,
                        errors=failed,
                        inputs_processed=packed,
                        metadata={"packed": packed, "skipped": skipped},
                    )
                    volume.commit()
                    volume.reload()

                else:
                    mark_finished(
                        output_dir,
                        "stage4_pack",
                        ok=True,
                        errors=0,
                        metadata={"packed": 0, "skipped": 0},
                    )

            except Exception as e:
                log_error("stage4_pack", "stage_failed", e)
                mark_finished(output_dir, "stage4_pack", ok=False, errors=1)
                volume.commit()
                raise

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5: Upload to HF -- direct from the volume mount
    # ══════════════════════════════════════════════════════════════════════
    # Tar shards from Stage 4 ARE shipped (the *.slices.tar files at the
    # study root). The raw `slices/` directories are NOT -- they're now
    # redundant with the tars and would re-introduce the 1M-file problem.
    hf_url = None
    if hf_repo:
        if not force and is_done(output_dir, "stage5_upload"):
            print("  Stage 5 already done (sentinel ok). Use --force to re-run.")
        else:
            print(f"\n{'=' * 60}")
            print("STAGE 5: Upload to HF (direct from volume)")
            print(f"{'=' * 60}")

            mark_started(output_dir, "stage5_upload", metadata={"hf_repo": hf_repo})
            log("stage5_upload", "stage_started", hf_repo=hf_repo)

            try:
                from huggingface_hub import HfApi, create_repo

                token = os.environ.get("HF_TOKEN", "")
                if not token:
                    print("  HF_TOKEN not set -- skipping upload")
                    mark_finished(
                        output_dir,
                        "stage5_upload",
                        ok=True,
                        errors=0,
                        metadata={"skipped_reason": "no_hf_token"},
                    )
                else:
                    api = HfApi(token=token)
                    create_repo(
                        hf_repo, repo_type="dataset", private=False, exist_ok=True, token=token
                    )

                    # Patterns we never want to ship.
                    #
                    # We skip raw `slices/` directories (those 1M PNGs are now
                    # packed into per-study `*.slices.tar` shards by Stage 4).
                    # We DO NOT add a generic "*.tar" exclusion -- the slice tar
                    # shards are exactly what we want to ship.
                    ignore = [
                        "*.dcm",
                        "*.dicom",
                        "*.html",
                        "*.htm",
                        "*.tar.gz",
                        "*.zip",
                        ".*",
                        ".*/**",
                        "_internal/**",
                        "**/_internal/**",
                        ".hf_cache/**",
                        "**/.hf_cache/**",
                        # Raw per-slice PNGs -- replaced by *.slices.tar shards
                        "slices/**",
                        "**/slices/**",
                        "*_slices/**",
                        "**/*_slices/**",
                    ]

                    t_u = time.time()
                    top_level = sorted(p.name for p in output_dir.iterdir())
                    print(f"  Source: {output_dir}")
                    print(f"  Top-level entries: {len(top_level)}")
                    print(f"  Ignore patterns: {len(ignore)}")
                    print("  Starting upload_large_folder...", flush=True)

                    api.upload_large_folder(
                        folder_path=str(output_dir),
                        repo_id=hf_repo,
                        repo_type="dataset",
                        ignore_patterns=ignore,
                    )

                    hf_url = f"https://huggingface.co/datasets/{hf_repo}"
                    elapsed_u = time.time() - t_u
                    print(f"  Upload finished in {elapsed_u / 60:.1f} min")
                    print(f"  -> {hf_url}")
                    log(
                        "stage5_upload",
                        "stage_complete",
                        elapsed_s=elapsed_u,
                        ok=1,
                        skipped=0,
                        failed=0,
                    )

                    mark_finished(
                        output_dir,
                        "stage5_upload",
                        ok=True,
                        errors=0,
                        inputs_processed=1,
                        metadata={"hf_url": hf_url},
                    )
                    volume.commit()

            except Exception as e:
                log_error("stage5_upload", "stage_failed", e)
                mark_finished(output_dir, "stage5_upload", ok=False, errors=1)
                volume.commit()
                raise

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"RESUME DONE in {elapsed / 60:.0f} min")
    print(f"  Montages found: {len(all_montages)}")
    print(f"  Annotated: {len(ann_results)} series")
    if hf_url:
        print(f"  HF: {hf_url}")
    print(f"{'=' * 60}")

    print(json.dumps(summarize(output_dir), indent=2, default=str))

    return {
        "montages": len(all_montages),
        "annotated": len(ann_results),
        "pack": pack_summary,
        "time_min": round(elapsed / 60, 1),
        "hf_url": hf_url,
    }


@app.local_entrypoint()
def main(
    repo: str = "shubhxho/speall-mri",
    skip_quality: bool = False,
    skip_annotation: bool = False,
    skip_pack: bool = False,
    force: bool = False,
):
    """Resume from where batch_pipeline failed -- quality + annotation + pack + HF upload."""
    call = resume.spawn(
        hf_repo=repo,
        skip_quality=skip_quality,
        skip_annotation=skip_annotation,
        skip_pack=skip_pack,
        force=force,
    )
    print("Spawned -- runs on Modal even if you disconnect.")
    print("Check: https://modal.com/apps/shubhxho/main")
    try:
        result = call.get(timeout=86400)
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("Disconnected -- resume continues on Modal.")

"""Resume pipeline — runs ONLY the stages that failed/were skipped.

Extraction already completed for all 1,050 studies. This script:
  1. Finds all existing montages (recursive search)
  2. Runs advanced quality + per-slice PNGs on each
  3. Runs Gemma 4 annotation via OpenRouter on each
  4. Generates analytics
  5. Uploads to HF

Skips anything already done. Reads from micom-v2 volume.

Usage:
  modal run --detach resume_pipeline.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path, PurePosixPath

import modal

VOLUME_NAME = "micom-v2"
MOUNT_POINT = PurePosixPath("/vol")

_PIP_DEPS = [
    "pydicom>=3.0", "SimpleITK>=2.4", "nibabel>=5.3",
    "polars>=1.17", "numpy>=2.1", "rich>=13.9",
    "matplotlib>=3.10", "scipy>=1.14", "pillow>=11.0",
    "python-dotenv>=1.0", "huggingface_hub>=0.25", "openai>=1.50",
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
    timeout=600, memory=4096, cpu=2.0,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
)
@modal.concurrent(max_inputs=2)
def quality_one_series(detail_path_str: str) -> dict:
    """Run advanced quality + slice export for a single series.

    Returns {ok, skipped, slices, bytes, path, error}. Skipped means the series
    already had advanced_quality (idempotent re-runs are cheap).
    """
    import sys; sys.path.insert(0, "/root")
    import numpy as np
    import SimpleITK as sitk

    detail_path = Path(detail_path_str)
    out = {"ok": False, "skipped": False, "slices": 0, "bytes": 0,
           "path": detail_path_str, "error": None}

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
            out["error"] = f"too few files ({len(series_files)})"
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
        detail_path.write_text(json.dumps(detail, indent=2, default=str))

        series_dir = detail_path.parent
        sr = export_all_slices(
            vol, series_dir.name, str(series_dir.parent),
            windows=["brain"], max_size=512, n_workers=2,
        )
        out["slices"] = sr["total_slices"]
        out["bytes"] = sr["total_bytes"]
        out["ok"] = True
        return out

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        print(f"  [WARN] {detail_path.name}: {out['error']}", flush=True)
        return out


# ── Worker: annotate one series via OpenRouter ───────────────────────────────

@app.function(
    image=image,
    volumes={str(MOUNT_POINT): volume},
    secrets=[modal.Secret.from_name("openrouter")],
    timeout=300, memory=2048, cpu=1.0,
)
@modal.concurrent(max_inputs=10)
def annotate_one(montage_path: str, series_label: str, quality_ctx: str, ann_dir: str) -> dict:
    """Annotate ONE series via Gemma 4 on OpenRouter."""
    import sys; sys.path.insert(0, "/root")
    import re

    from src.cloud_analysis import annotate_series_multi, tissue_analysis_with_model, _client, _detect_provider

    result = annotate_series_multi(montage_path, series_label, quality_ctx, models=["gemma4"])

    # Tissue analysis pass
    if result.get("consensus", {}).get("sequence_type"):
        try:
            provider = _detect_provider()
            client = _client()
            prior = json.dumps({
                "sequence": result["consensus"].get("sequence_type", "?"),
                "pathology_found": result["consensus"].get("pathology", {}).get("found", False),
            }, default=str)
            tissue = tissue_analysis_with_model(
                client, montage_path, series_label,
                prior_annotation=prior, quality_ctx=quality_ctx,
                model_key="gemma4", provider=provider,
            )
            if tissue.get("tissue_analysis"):
                result["tissue_analysis"] = tissue["tissue_analysis"]
        except Exception:
            pass

    # Save
    safe_name = re.sub(r'[^\w\-]', '_', series_label)
    Path(ann_dir).mkdir(parents=True, exist_ok=True)
    (Path(ann_dir) / f"{safe_name}.json").write_text(json.dumps(result, indent=2, default=str))
    volume.commit()

    return {
        "series": series_label,
        "sequence": result.get("consensus", {}).get("sequence_type", "?"),
        "pathology": result.get("consensus", {}).get("pathology", {}).get("found", False),
        "has_tissue": bool(result.get("tissue_analysis")),
    }


# ── Main resume function ─────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={str(MOUNT_POINT): volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("openrouter")],
    timeout=86400, memory=16384, cpu=8.0,
)
def resume(hf_repo: str = "", skip_quality: bool = False, skip_annotation: bool = False) -> dict:
    """Resume pipeline from where it failed — skip extraction, do quality + annotation + upload."""
    import sys; sys.path.insert(0, "/root")
    import numpy as np
    import shutil
    import tempfile

    volume.reload()
    t0 = time.time()

    output_dir = Path(str(MOUNT_POINT / "output" / "akai_mri"))
    if not output_dir.exists():
        return {"error": f"Output not found: {output_dir}"}

    # ── Find ALL montages recursively ────────────────────────────────────
    print("Scanning for existing montages...")
    all_montages = list(output_dir.rglob("*_multiplane.png"))
    print(f"Found {len(all_montages)} montages across all studies")

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
        print(f"\n{'='*60}")
        print(f"STAGE 2: Advanced quality + per-slice PNGs (parallel .map)")
        print(f"{'='*60}")

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

        if pending:
            t_q = time.time()
            results = list(quality_one_series.map(pending))

            ok = sum(1 for r in results if r.get("ok"))
            skipped = sum(1 for r in results if r.get("skipped"))
            errored = [r for r in results if r.get("error")]
            total_slices = sum(r.get("slices", 0) for r in results)
            total_slice_bytes = sum(r.get("bytes", 0) for r in results)
            elapsed = time.time() - t_q

            print(f"  Quality done in {elapsed/60:.1f} min: "
                  f"{ok} processed, {skipped} skipped, {len(errored)} failed")
            print(f"  Slices: {total_slices} PNGs ({total_slice_bytes/1024**3:.1f} GB)")
            if errored:
                # Log a sample of errors for diagnosis (don't spam)
                from collections import Counter
                err_types = Counter(r["error"].split(":")[0] for r in errored if r.get("error"))
                print(f"  Failure breakdown: {dict(err_types.most_common(10))}")

            # Single commit after the whole map completes — workers don't commit.
            volume.commit()
        else:
            print(f"  Nothing to do — all series already processed")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: Gemma 4 annotation (only series without annotations)
    # ══════════════════════════════════════════════════════════════════════
    ann_results = []
    if not skip_annotation:
        print(f"\n{'='*60}")
        print(f"STAGE 3: Gemma 4 annotation via OpenRouter")
        print(f"{'='*60}")

        try:
            if os.environ.get("OPENROUTER_API_KEY"):
                from src.ai_analysis import _build_quality_context, _is_derivative

                ann_dir = str(output_dir / "annotations")
                Path(ann_dir).mkdir(parents=True, exist_ok=True)

                # Check which series already have annotations
                existing_anns = set()
                for f in Path(ann_dir).glob("*.json"):
                    if f.name != "study_annotations.json":
                        existing_anns.add(f.stem)

                # Build annotation args from ALL montages (recursive)
                ann_args = []
                for montage in all_montages:
                    series_dir = montage.parent
                    series_name = series_dir.name

                    # Extract series number and description
                    parts = series_name.split("_", 1)
                    snum = parts[0].replace("s", "") if parts else "?"
                    sdesc = parts[1] if len(parts) > 1 else ""
                    label = f"Series {snum} — {sdesc}"

                    if _is_derivative(label):
                        continue

                    # Skip if already annotated
                    import re
                    safe_name = re.sub(r'[^\w\-]', '_', label)
                    if safe_name in existing_anns:
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

                print(f"  {len(ann_args)} series to annotate ({len(existing_anns)} already done)")

                if ann_args:
                    print(f"  Running 10 concurrent OpenRouter calls...")
                    ann_results = list(annotate_one.starmap(ann_args))

                    pathology_count = sum(1 for r in ann_results if r.get("pathology"))
                    tissue_count = sum(1 for r in ann_results if r.get("has_tissue"))
                    print(f"  Annotated: {len(ann_results)}, pathology: {pathology_count}, tissue: {tissue_count}")

                    (Path(ann_dir) / "study_annotations.json").write_text(
                        json.dumps({"series_annotated": len(ann_results) + len(existing_anns), "results": ann_results}, indent=2, default=str)
                    )
                    volume.commit()

            else:
                print(f"  OPENROUTER_API_KEY not set")
        except Exception as e:
            print(f"  Annotation failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5: Upload to HF
    # ══════════════════════════════════════════════════════════════════════
    hf_url = None
    if hf_repo:
        print(f"\n{'='*60}")
        print(f"STAGE 5: Upload to HF")
        print(f"{'='*60}")
        try:
            from huggingface_hub import HfApi, create_repo
            token = os.environ.get("HF_TOKEN", "")
            if token:
                api = HfApi(token=token)
                create_repo(hf_repo, repo_type="dataset", private=False, exist_ok=True, token=token)

                staging = Path(tempfile.mkdtemp())
                SKIP_EXT = {".html", ".htm", ".tar", ".dcm"}
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
                    dest = staging / "akai_mri" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
                    n_staged += 1

                if n_staged:
                    staged_gb = sum(f.stat().st_size for f in Path(staging).rglob("*") if f.is_file()) / 1024**3
                    print(f"  {n_staged} files ({staged_gb:.1f} GB)")
                    os.environ["HF_HOME"] = str(staging / ".hf_cache")

                    if n_staged > 1000 or staged_gb > 5:
                        api.upload_large_folder(folder_path=str(staging), repo_id=hf_repo, repo_type="dataset")
                    else:
                        api.upload_folder(folder_path=str(staging), repo_id=hf_repo, repo_type="dataset", token=token,
                                          commit_message=f"Resume: {n_staged} files")
                    hf_url = f"https://huggingface.co/datasets/{hf_repo}"
                    print(f"  → {hf_url}")
                shutil.rmtree(staging, ignore_errors=True)
        except Exception as e:
            print(f"  HF upload failed: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"RESUME DONE in {elapsed / 60:.0f} min")
    print(f"  Montages found: {len(all_montages)}")
    print(f"  Annotated: {len(ann_results)} series")
    if hf_url:
        print(f"  HF: {hf_url}")
    print(f"{'='*60}")

    return {
        "montages": len(all_montages),
        "annotated": len(ann_results),
        "time_min": round(elapsed / 60, 1),
        "hf_url": hf_url,
    }


@app.local_entrypoint()
def main(
    repo: str = "shubhxho/akai-mri",
    skip_quality: bool = False,
    skip_annotation: bool = False,
):
    """Resume from where batch_pipeline failed — quality + annotation + HF upload."""
    call = resume.spawn(hf_repo=repo, skip_quality=skip_quality, skip_annotation=skip_annotation)
    print(f"Spawned — runs on Modal even if you disconnect.")
    print(f"Check: https://modal.com/apps/shubhxho/main")
    try:
        result = call.get(timeout=86400)
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("Disconnected — resume continues on Modal.")

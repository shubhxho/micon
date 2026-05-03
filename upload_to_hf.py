"""Standalone HF upload script -- generate manifest + push to Hugging Face.

Does ONLY the things needed to refresh the public dataset:

  1. Optionally generate fresh manifest.parquet + study_manifest.parquet
     from current volume contents (default: yes)
  2. Optionally squash HF commit history before upload (useful after a
     streak of partial upload runs that accreted hundreds of commits)
  3. Upload directly from the volume mount via upload_large_folder with
     the project-standard ignore patterns

This avoids spinning up the full resume_pipeline.py with three skip flags
every time you just want to push the latest state to HF.

Usage:
  modal run --detach upload_to_hf.py::upload --hf-repo shubhxho/speall-mri
  modal run upload_to_hf.py                       # generate manifest + upload
  modal run upload_to_hf.py --skip-manifest        # upload without manifest regen
  modal run upload_to_hf.py --squash               # squash history + upload
"""

from __future__ import annotations

import json
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

app = modal.App("speall-upload", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


# Patterns we never want to ship.
#
# We skip raw `slices/` directories (those 1M PNGs are packed into per-study
# `*.slices.tar` shards by Stage 4 of resume_pipeline.py).
# We do NOT add a generic "*.tar" exclusion -- the slice tar shards are
# exactly what we want to ship.
_IGNORE_PATTERNS = [
    "*.dcm", "*.dicom",
    "*.html", "*.htm",
    "*.tar.gz", "*.zip",
    ".*", ".*/**",
    "_internal/**", "**/_internal/**",
    ".hf_cache/**", "**/.hf_cache/**",
    # Raw per-slice PNGs -- replaced by *.slices.tar shards
    "slices/**", "**/slices/**",
    "*_slices/**", "**/*_slices/**",
]


@app.function(
    image=image,
    volumes={str(MOUNT_POINT): volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=86400, memory=16384, cpu=8.0,
)
def upload(
    hf_repo: str = "shubhxho/speall-mri",
    generate_manifest: bool = True,
    super_squash: bool = False,
) -> dict:
    """Generate manifest + optionally squash history + upload to HF.

    Args:
        hf_repo: Hugging Face dataset repo ID (e.g. "shubhxho/speall-mri").
        generate_manifest: If True, regenerate manifest.parquet and
            study_manifest.parquet from the current volume contents before
            uploading.
        super_squash: If True, collapse prior HF commit history into a single
            commit before the upload. Useful after many incremental upload runs
            that accreted hundreds of commits.

    Returns:
        Dict with keys: hf_url, manifest (counts or None), time_min.
    """
    import sys
    sys.path.insert(0, "/root")

    import os

    from huggingface_hub import HfApi, create_repo

    volume.reload()
    t0 = time.time()

    output_dir = Path(MOUNT_POINT) / "output" / "akai_mri"
    if not output_dir.exists():
        return {"error": f"Output dir not found: {output_dir}"}

    # ── Step 1: Generate fresh manifests ─────────────────────────────────────
    manifest_counts: dict | None = None
    if generate_manifest:
        print(f"{'='*60}")
        print("Generating manifests from volume contents...")
        print(f"{'='*60}")
        try:
            from src.build_manifest import write_manifests
            manifest_counts = write_manifests(output_dir, output_dir)
            print(f"  manifest.parquet       -> {manifest_counts['series_rows']} series rows")
            print(f"  study_manifest.parquet -> {manifest_counts['study_rows']} study rows")
            # Commit so parquets are durable on the volume before upload
            volume.commit()
            volume.reload()
        except Exception as exc:
            print(f"  Manifest generation failed: {exc}")
            manifest_counts = {"error": str(exc)}

    # ── Step 2: Create repo + optionally squash history ───────────────────────
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        return {"error": "HF_TOKEN secret not set"}

    api = HfApi(token=token)
    create_repo(hf_repo, repo_type="dataset", private=False, exist_ok=True, token=token)

    if super_squash:
        print(f"\n{'='*60}")
        print("Squashing HF commit history...")
        print(f"{'='*60}")
        try:
            api.super_squash_history(repo_id=hf_repo, repo_type="dataset")
            print("  History squashed.")
        except Exception as exc:
            print(f"  Squash failed (continuing): {exc}")

    # ── Step 3: Upload from volume mount ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Starting upload_large_folder...")
    print(f"{'='*60}")

    top_level = sorted(p.name for p in output_dir.iterdir())
    print(f"  Source path:       {output_dir}")
    print(f"  Top-level entries: {len(top_level)}")
    print(f"  Ignore patterns:   {len(_IGNORE_PATTERNS)}")
    print(f"  Patterns: {_IGNORE_PATTERNS}", flush=True)

    t_u = time.time()
    api.upload_large_folder(
        folder_path=str(output_dir),
        repo_id=hf_repo,
        repo_type="dataset",
        ignore_patterns=_IGNORE_PATTERNS,
    )
    upload_elapsed = time.time() - t_u
    hf_url = f"https://huggingface.co/datasets/{hf_repo}"

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"UPLOAD DONE in {upload_elapsed / 60:.1f} min (total wall time {total_elapsed / 60:.1f} min)")
    print(f"  -> {hf_url}")
    print(f"{'='*60}")

    return {
        "hf_url": hf_url,
        "manifest": manifest_counts,
        "upload_time_min": round(upload_elapsed / 60, 1),
        "time_min": round(total_elapsed / 60, 1),
    }


@app.local_entrypoint()
def main(
    repo: str = "shubhxho/speall-mri",
    skip_manifest: bool = False,
    squash: bool = False,
) -> None:
    """Upload current volume state to Hugging Face.

    Examples:
      modal run upload_to_hf.py
      modal run upload_to_hf.py --skip-manifest
      modal run upload_to_hf.py --squash
      modal run --detach upload_to_hf.py::upload --hf-repo shubhxho/speall-mri
    """
    call = upload.spawn(
        hf_repo=repo,
        generate_manifest=not skip_manifest,
        super_squash=squash,
    )
    print("Spawned -- runs on Modal even if you disconnect.")
    print("Check: https://modal.com/apps/shubhxho/main")
    try:
        result = call.get(timeout=86400)
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("Disconnected -- upload continues on Modal.")

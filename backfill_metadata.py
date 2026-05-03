"""Backfill study_id and pipeline_version into existing detail.json files.

One-shot Modal job that populates provenance metadata on detail.json files
written before the metadata logic landed in resume_pipeline.py.

Usage:
  modal run backfill_metadata.py
  modal run backfill_metadata.py --repo-dir akai_mri
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import modal

VOLUME_NAME = "micom-v2"
MOUNT_POINT = PurePosixPath("/vol")

image = modal.Image.debian_slim(python_version="3.12")

app = modal.App("micom-backfill", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


@app.function(
    image=image,
    volumes={str(MOUNT_POINT): volume},
    timeout=120,
    memory=512,
    cpu=1.0,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
)
def backfill_one(detail_path_str: str, output_dir_str: str) -> dict:
    """Backfill study_id + pipeline_version into one detail.json.

    Returns {ok, skipped, path, error}.
    - skipped=True means study_id was already present (idempotent).
    - ok=True means the file was updated successfully.
    """
    detail_path = Path(detail_path_str)
    output_dir = Path(output_dir_str)
    out = {"ok": False, "skipped": False, "path": detail_path_str, "error": None}

    try:
        detail = json.loads(detail_path.read_text())

        if detail.get("study_id"):
            out["skipped"] = True
            return out

        # Derive study_id the same way quality_one_series does.
        try:
            study_id = detail_path.relative_to(output_dir).parts[0]
        except (ValueError, IndexError):
            study_id = detail_path.parent.parent.name

        detail["study_id"] = study_id
        detail["pipeline_version"] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_sha": os.environ.get("GIT_SHA", "unknown"),
        }

        detail_path.write_text(json.dumps(detail, indent=2, default=str))
        out["ok"] = True
        return out

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


@app.function(
    image=image,
    volumes={str(MOUNT_POINT): volume},
    timeout=3600,
    memory=2048,
    cpu=2.0,
)
def run_backfill(repo_dir: str = "akai_mri") -> dict:
    """Scan volume for detail.json files and fan out backfill_one via starmap."""
    volume.reload()

    output_dir = Path(MOUNT_POINT / "output" / repo_dir)
    if not output_dir.exists():
        return {"error": f"Output not found: {output_dir}"}

    all_details = list(output_dir.rglob("*_detail.json"))
    print(f"Found {len(all_details)} detail.json files under {output_dir}")

    if not all_details:
        return {"ok": 0, "skipped": 0, "failed": 0, "total": 0}

    args = [(str(p), str(output_dir)) for p in all_details]
    results = list(backfill_one.starmap(args, return_exceptions=True))

    ok = 0
    skipped = 0
    failed = 0
    for r in results:
        if isinstance(r, BaseException):
            failed += 1
        elif r.get("ok"):
            ok += 1
        elif r.get("skipped"):
            skipped += 1
        else:
            failed += 1

    volume.commit()
    print(f"Backfill complete: {ok} updated / {skipped} already had study_id / {failed} failed")
    return {"ok": ok, "skipped": skipped, "failed": failed, "total": len(all_details)}


@app.local_entrypoint()
def main(repo_dir: str = "akai_mri"):
    """Scan /vol/output/<repo_dir>/**/*_detail.json and backfill metadata."""
    call = run_backfill.spawn(repo_dir=repo_dir)
    print("Spawned backfill job -- runs on Modal even if you disconnect.")
    print("Check: https://modal.com/apps/shubhxho/main")
    try:
        result = call.get(timeout=3600)
        print(json.dumps(result, indent=2))
        print(
            f"\nSummary: {result.get('ok', 0)} ok / "
            f"{result.get('skipped', 0)} skipped / "
            f"{result.get('failed', 0)} failed"
        )
    except KeyboardInterrupt:
        print("Disconnected -- backfill continues on Modal.")

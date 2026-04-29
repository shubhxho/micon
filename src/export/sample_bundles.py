"""Sample bundle generator — reproducible study slices for buyer evaluation.

Creates:
  - 5-study sample bundle (quick evaluation)
  - 50-study sample bundle (thorough evaluation)
  - Full dataset bundle (all studies)

Each bundle is a self-contained directory with:
  - Clean DICOM hierarchy
  - Per-study manifests
  - Aggregate manifest
  - Validation reports

Reproducible via seed — same seed always produces same sample.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path


def create_sample_bundle(
    source_dir: Path | str,
    output_dir: Path | str,
    n_studies: int = 5,
    seed: int = 42,
) -> dict:
    """Create a reproducible sample bundle of N studies.

    Args:
        source_dir: directory containing study subdirectories
        output_dir: where to write the bundle
        n_studies: number of studies to include
        seed: random seed for reproducibility

    Returns: {"studies_included": N, "total_files": N, "bundle_path": str}
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all study directories (contain .dcm files)
    study_dirs = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and any(d.rglob("*.dcm"))
    ])

    if not study_dirs:
        # Single study — treat source_dir as the study
        study_dirs = [source_dir]

    # Reproducible sample
    rng = random.Random(seed)
    sample = rng.sample(study_dirs, min(n_studies, len(study_dirs)))

    total_files = 0
    studies = []

    for study_dir in sample:
        study_name = study_dir.name
        dest = output_dir / study_name
        dest.mkdir(parents=True, exist_ok=True)

        dcm_files = sorted(study_dir.rglob("*.dcm"))
        for f in dcm_files:
            rel = f.relative_to(study_dir)
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            total_files += 1

        # Copy manifest if exists
        manifest = study_dir / "manifest.json"
        if manifest.exists():
            shutil.copy2(manifest, dest / "manifest.json")

        # Copy validation report if exists
        validation = study_dir / "validation_report.json"
        if validation.exists():
            shutil.copy2(validation, dest / "validation_report.json")

        studies.append(study_name)

    # Write bundle manifest
    bundle_manifest = {
        "bundle_type": f"{n_studies}-study sample",
        "seed": seed,
        "studies_included": len(studies),
        "total_files": total_files,
        "studies": studies,
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(bundle_manifest, indent=2))

    return {
        "studies_included": len(studies),
        "total_files": total_files,
        "bundle_path": str(output_dir),
    }

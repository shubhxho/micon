"""Manifest augmentation for the Speall segmentation pipeline.

Adds segmentation-availability boolean columns and per-study volumetric
statistics to an existing ``manifest.parquet`` (or ``study_manifest.parquet``).

New columns added to the *study-level* manifest
------------------------------------------------
has_brain_mask         bool  -- at least one synthstrip mask exists
has_brain_parcellation bool  -- at least one synthseg mask exists
has_lesion_mask        bool  -- at least one monai_brain_lesion mask exists
brain_volume_cm3       f64   -- total brain volume (labels > 0 in brainmask)
ventricle_volume_cm3   f64   -- ventricle label volume (label 4 in synthseg)
lesion_load_cm3        f64   -- total lesion volume (labels > 0 in lesion mask)

Usage::

    from src.segmentation.extras_for_buyers import augment_manifest
    augment_manifest(
        manifest_path=Path("/data/bids/manifest.parquet"),
        bids_root=Path("/data/bids"),
        out_path=Path("/data/bids/manifest_with_seg.parquet"),
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src._logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------

try:
    import nibabel as nib  # type: ignore[import-untyped]

    _NIBABEL = True
except ImportError:
    nib = None  # type: ignore[assignment]
    _NIBABEL = False

try:
    import numpy as np  # type: ignore[import-untyped]

    _NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY = False

# Label index used in SynthSeg parcellation for lateral ventricles
# (FreeSurfer LUT: Left-Lateral-Ventricle=4, Right-Lateral-Ventricle=43)
_VENTRICLE_LABELS = frozenset({4, 43})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _voxel_volume_mm3(affine: Any) -> float:
    """Return voxel volume in mm^3 from a NIfTI affine."""
    if not _NUMPY:
        return 1.0
    import numpy as _np

    return float(abs(_np.linalg.det(affine[:3, :3])))


def _volume_cm3(mask: Any, voxel_vol_mm3: float, labels: frozenset[int] | None = None) -> float:
    """Sum voxels matching ``labels`` (or all non-zero), convert to cm^3."""
    if not _NUMPY:
        return 0.0

    if labels is None:
        count = int((mask > 0).sum())
    else:
        count = sum(int((mask == lbl).sum()) for lbl in labels)
    return round(count * voxel_vol_mm3 / 1000.0, 4)


def _read_nifti_mask(path: Path) -> tuple[Any, Any] | None:
    """Load NIfTI mask; returns (data, affine) or None on failure."""
    if not _NIBABEL:
        logger.warning("nibabel not available; cannot read NIfTI masks")
        return None
    try:
        img = nib.load(str(path))  # pyright: ignore[reportOptionalMemberAccess]
        return img.get_fdata().astype("uint8"), img.affine  # pyright: ignore[reportAttributeAccessIssue]
    except Exception as exc:
        logger.warning("Could not load mask {}: {}", path, exc)
        return None


def _find_derivative_mask(
    bids_root: Path,
    derivative_name: str,
    subject: str,
    session: str,
    desc_label: str,
) -> Path | None:
    """Return the BIDS derivatives mask path if it exists."""
    from src.segmentation.pipeline import _derivatives_filename, _sanitize

    out_dir = (
        bids_root
        / "derivatives"
        / derivative_name
        / f"sub-{_sanitize(subject)}"
        / f"ses-{_sanitize(session)}"
        / "anat"
    )
    fname = _derivatives_filename(subject, session, desc_label)
    candidate = out_dir / fname
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Per-study stats collection
# ---------------------------------------------------------------------------


def _study_seg_stats(
    bids_root: Path,
    subject: str,
    session: str,
) -> dict[str, Any]:
    """Compute segmentation stats for one subject/session."""
    # brain mask
    brain_mask_path = _find_derivative_mask(
        bids_root, "speall-synthstrip", subject, session, "brainmask"
    )
    has_brain_mask = brain_mask_path is not None
    brain_volume = 0.0
    if has_brain_mask:
        loaded = _read_nifti_mask(brain_mask_path)  # type: ignore[arg-type]
        if loaded is not None:
            brain_volume = _volume_cm3(loaded[0], _voxel_volume_mm3(loaded[1]))

    # parcellation
    parcel_path = _find_derivative_mask(bids_root, "speall-synthseg", subject, session, "parcel")
    has_parcellation = parcel_path is not None
    ventricle_volume = 0.0
    if has_parcellation:
        loaded = _read_nifti_mask(parcel_path)  # type: ignore[arg-type]
        if loaded is not None:
            ventricle_volume = _volume_cm3(
                loaded[0], _voxel_volume_mm3(loaded[1]), _VENTRICLE_LABELS
            )

    # lesion mask
    lesion_path = _find_derivative_mask(bids_root, "speall-lesion", subject, session, "lesion")
    has_lesion_mask = lesion_path is not None
    lesion_load = 0.0
    if has_lesion_mask:
        loaded = _read_nifti_mask(lesion_path)  # type: ignore[arg-type]
        if loaded is not None:
            lesion_load = _volume_cm3(loaded[0], _voxel_volume_mm3(loaded[1]))

    return {
        "has_brain_mask": has_brain_mask,
        "has_brain_parcellation": has_parcellation,
        "has_lesion_mask": has_lesion_mask,
        "brain_volume_cm3": brain_volume,
        "ventricle_volume_cm3": ventricle_volume,
        "lesion_load_cm3": lesion_load,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def augment_manifest(
    manifest_path: Path,
    bids_root: Path,
    out_path: Path | None = None,
) -> pl.DataFrame:
    """Augment a manifest parquet with segmentation availability columns.

    Reads ``manifest_path`` (must have ``study_id`` column), crawls the BIDS
    derivatives layout for segmentation masks, computes volumetric stats,
    and writes an augmented parquet to ``out_path``.

    Args:
        manifest_path: Path to an existing ``manifest.parquet`` or
            ``study_manifest.parquet``.
        bids_root:     Root of the BIDS dataset (contains ``derivatives/``).
        out_path:      Where to write the augmented parquet.  Defaults to
            ``manifest_path.parent / manifest_path.stem + '_with_seg.parquet'``.

    Returns:
        The augmented :class:`polars.DataFrame`.
    """
    if out_path is None:
        stem = manifest_path.stem
        out_path = manifest_path.parent / f"{stem}_with_seg.parquet"

    df = pl.read_parquet(manifest_path)

    if "study_id" not in df.columns:
        raise ValueError("manifest must contain a 'study_id' column")

    # Derive subject/session from study_id (study_id = sub-XXX_ses-YY or just XXX)
    study_ids = df["study_id"].unique().to_list()
    seg_rows: list[dict[str, Any]] = []

    for study_id in study_ids:
        subject, session = _parse_study_id(study_id)
        stats = _study_seg_stats(bids_root, subject, session)
        stats["study_id"] = study_id
        seg_rows.append(stats)

    seg_df = pl.DataFrame(
        seg_rows,
        schema={
            "study_id": pl.Utf8,
            "has_brain_mask": pl.Boolean,
            "has_brain_parcellation": pl.Boolean,
            "has_lesion_mask": pl.Boolean,
            "brain_volume_cm3": pl.Float64,
            "ventricle_volume_cm3": pl.Float64,
            "lesion_load_cm3": pl.Float64,
        },
    )

    result = df.join(seg_df, on="study_id", how="left")
    result.write_parquet(str(out_path))
    logger.info("Wrote augmented manifest -> {} ({} rows)", out_path, len(result))
    return result


def _parse_study_id(study_id: str) -> tuple[str, str]:
    """Extract (subject, session) from a study_id string.

    Handles formats:
    - ``sub-001_ses-01`` -> (``001``, ``01``)
    - ``001``            -> (``001``, ``01``)
    - ``sub-001``        -> (``001``, ``01``)
    """
    import re

    sub_match = re.search(r"sub-([A-Za-z0-9]+)", study_id)
    ses_match = re.search(r"ses-([A-Za-z0-9]+)", study_id)
    subject = sub_match.group(1) if sub_match else re.sub(r"[^A-Za-z0-9]", "", study_id)
    session = ses_match.group(1) if ses_match else "01"
    return subject, session


def compute_cohort_summary(augmented_df: pl.DataFrame) -> dict[str, Any]:
    """Return corpus-level segmentation summary statistics.

    Args:
        augmented_df: DataFrame returned by :func:`augment_manifest`.

    Returns:
        Dict with ``n_studies``, ``n_with_brain_mask``, ``n_with_parcellation``,
        ``n_with_lesion_mask``, ``mean_brain_volume_cm3``,
        ``mean_lesion_load_cm3``.
    """
    n = len(augmented_df)
    return {
        "n_studies": n,
        "n_with_brain_mask": int(augmented_df["has_brain_mask"].sum())
        if "has_brain_mask" in augmented_df.columns
        else 0,
        "n_with_parcellation": int(augmented_df["has_brain_parcellation"].sum())
        if "has_brain_parcellation" in augmented_df.columns
        else 0,
        "n_with_lesion_mask": int(augmented_df["has_lesion_mask"].sum())
        if "has_lesion_mask" in augmented_df.columns
        else 0,
        "mean_brain_volume_cm3": float(augmented_df["brain_volume_cm3"].mean() or 0.0)  # type: ignore[arg-type]
        if "brain_volume_cm3" in augmented_df.columns
        else 0.0,
        "mean_lesion_load_cm3": float(augmented_df["lesion_load_cm3"].mean() or 0.0)  # type: ignore[arg-type]
        if "lesion_load_cm3" in augmented_df.columns
        else 0.0,
    }

"""MONAI integration helpers for the Speall MRI Brain Dataset.

Lazy import: this module imports cleanly without monai or torch installed.
Install extras: pip install "micom[monai]"

Usage
-----
    from pathlib import Path
    from src.loaders.monai import to_monai_dict_dataset, recommend_transforms
    from monai.data import Dataset
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord

    data_dicts = to_monai_dict_dataset(
        manifest_path=Path("manifest.parquet"),
        root=Path("/data/speall"),
        split="train",
        sequence_type="DWI",
    )

    transforms_names = recommend_transforms("DWI")
    print(transforms_names)

    # Build a MONAI Dataset
    transforms = Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]), ToTensord(keys=["image"])])
    dataset = Dataset(data=data_dicts, transform=transforms)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lazy heavy-framework imports
# ---------------------------------------------------------------------------

try:
    import monai  # noqa: F401

    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Recommended transforms per sequence type
# ---------------------------------------------------------------------------

_TRANSFORM_MAP: dict[str, list[str]] = {
    "DWI": [
        "LoadImaged(keys=['image'])",
        "EnsureChannelFirstd(keys=['image'])",
        "Spacingd(keys=['image'], pixdim=(2.0, 2.0, 2.0), mode='bilinear')",
        "ScaleIntensityRanged(keys=['image'], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True)",
        "RandGaussianNoised(keys=['image'], prob=0.2, std=0.05)",
        "RandRotated(keys=['image'], range_x=0.2, range_y=0.2, range_z=0.2, prob=0.3)",
        "ToTensord(keys=['image', 'label'])",
    ],
    "FLAIR": [
        "LoadImaged(keys=['image'])",
        "EnsureChannelFirstd(keys=['image'])",
        "Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear')",
        "NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True)",
        "RandFlipd(keys=['image'], spatial_axis=0, prob=0.5)",
        "RandRotated(keys=['image'], range_x=0.15, range_y=0.15, range_z=0.15, prob=0.3)",
        "ToTensord(keys=['image', 'label'])",
    ],
    "T1": [
        "LoadImaged(keys=['image'])",
        "EnsureChannelFirstd(keys=['image'])",
        "Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear')",
        "ScaleIntensityRanged(keys=['image'], a_min=0, a_max=2000, b_min=0.0, b_max=1.0, clip=True)",
        "RandRotated(keys=['image'], range_x=0.15, range_y=0.15, range_z=0.15, prob=0.3)",
        "RandZoomd(keys=['image'], min_zoom=0.9, max_zoom=1.1, prob=0.2)",
        "ToTensord(keys=['image', 'label'])",
    ],
    "T2": [
        "LoadImaged(keys=['image'])",
        "EnsureChannelFirstd(keys=['image'])",
        "Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear')",
        "NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True)",
        "RandFlipd(keys=['image'], spatial_axis=[0, 1], prob=0.5)",
        "RandRotated(keys=['image'], range_x=0.15, range_y=0.15, range_z=0.15, prob=0.3)",
        "ToTensord(keys=['image', 'label'])",
    ],
    "SWAN": [
        "LoadImaged(keys=['image'])",
        "EnsureChannelFirstd(keys=['image'])",
        "Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear')",
        "ScaleIntensityRanged(keys=['image'], a_min=-100, a_max=500, b_min=0.0, b_max=1.0, clip=True)",
        "RandGaussianSmoothd(keys=['image'], sigma_x=(0.5, 1.5), prob=0.2)",
        "ToTensord(keys=['image', 'label'])",
    ],
    "TOF": [
        "LoadImaged(keys=['image'])",
        "EnsureChannelFirstd(keys=['image'])",
        "Spacingd(keys=['image'], pixdim=(0.5, 0.5, 0.5), mode='bilinear')",
        "ScaleIntensityRanged(keys=['image'], a_min=0, a_max=1500, b_min=0.0, b_max=1.0, clip=True)",
        "RandRotated(keys=['image'], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.2)",
        "ToTensord(keys=['image', 'label'])",
    ],
}

_DEFAULT_TRANSFORMS: list[str] = [
    "LoadImaged(keys=['image'])",
    "EnsureChannelFirstd(keys=['image'])",
    "Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear')",
    "NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True)",
    "ToTensord(keys=['image', 'label'])",
]

# Label encoding for quality grade (A=4, B=3, C=2, D=1, F=0)
_GRADE_TO_LABEL: dict[str, int] = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}


# ---------------------------------------------------------------------------
# Internal helpers (shared with pytorch loader)
# ---------------------------------------------------------------------------


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read parquet file into list of row dicts."""
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(str(path))
        cols = table.to_pydict()
        n = len(table)
        return [{col: cols[col][i] for col in cols} for i in range(n)]
    except ImportError:
        pass

    try:
        import pandas as pd

        df = pd.read_parquet(str(path))
        return df.to_dict(orient="records")
    except ImportError:
        pass

    raise ImportError("Either pyarrow or pandas is required. pip install pyarrow")


def _assign_split(series_uid: str, split: str) -> bool:
    """Deterministic split assignment when splits.parquet is absent."""
    bucket = abs(hash(series_uid)) % 10
    if split == "train":
        return bucket >= 2
    if split == "test":
        return bucket == 0
    if split == "validation":
        return bucket == 1
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_monai_dict_dataset(
    manifest_path: Path,
    root: Path,
    split: str = "train",
    sequence_type: str | None = None,
    quality_grades: list[str] | None = None,
    splits_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert the Speall MRI manifest into MONAI's expected list-of-dicts format.

    Each dict contains:
        ``image``    -- str path to the DICOM series directory (or .nii.gz)
        ``label``    -- int (quality grade encoded: A=4, B=3, C=2, D=1, F=0)
        ``metadata`` -- dict with all manifest scalar fields

    Compatible with ``monai.data.Dataset`` and ``monai.transforms.Compose``.

    Parameters
    ----------
    manifest_path:
        Path to ``manifest.parquet``.
    root:
        Dataset root directory.
    split:
        ``"train"`` | ``"test"`` | ``"validation"`` | ``"all"``.
    sequence_type:
        Filter to one sequence type, e.g. ``"DWI"``.  Case-insensitive.
    quality_grades:
        Optional list of accepted grades, e.g. ``["A", "B"]``.
    splits_path:
        Path to ``splits.parquet``.  If absent, deterministic hash is used.

    Returns
    -------
    list[dict]
        List of MONAI-compatible dicts with ``image``, ``label``, ``metadata``.
    """
    root = Path(root)
    all_rows = _read_parquet_rows(Path(manifest_path))

    # Split membership
    split_uids: set[str] | None = None
    if splits_path and Path(splits_path).exists():
        split_rows = _read_parquet_rows(Path(splits_path))
        split_uids = {r["series_uid"] for r in split_rows if r.get("split") == split}

    result: list[dict[str, Any]] = []
    for row in all_rows:
        uid: str = row.get("series_uid") or ""
        study_id: str = row.get("study_id") or ""

        if split_uids is not None:
            if uid not in split_uids:
                continue
        elif split != "all" and not _assign_split(uid, split):
            continue

        if sequence_type:
            row_seq = (row.get("sequence_type") or "").upper()
            if row_seq != sequence_type.upper():
                continue

        if quality_grades:
            row_grade = row.get("quality_grade") or ""
            if row_grade.upper() not in [g.upper() for g in quality_grades]:
                continue

        # Resolve image path: prefer NIfTI, fall back to DICOM dir
        detail_path_rel = row.get("detail_path") or ""
        series_dir = (
            root / study_id / Path(detail_path_rel).parent if detail_path_rel else root / study_id
        )

        image_path = _resolve_image_path(series_dir)

        grade_str = row.get("quality_grade") or "F"
        label = _GRADE_TO_LABEL.get(grade_str.upper(), 0)

        metadata: dict[str, Any] = {
            "study_id": study_id,
            "series_uid": uid,
            "series_description": row.get("series_description"),
            "sequence_type": row.get("sequence_type"),
            "sequence_confidence": row.get("sequence_confidence"),
            "modality": row.get("modality"),
            "file_count": row.get("file_count"),
            "tr_ms": row.get("tr_ms"),
            "te_ms": row.get("te_ms"),
            "b_value": row.get("b_value"),
            "field_strength_T": row.get("field_strength_T"),
            "plane": row.get("plane"),
            "volume_snr": row.get("volume_snr"),
            "volume_cnr": row.get("volume_cnr"),
            "quality_grade": grade_str,
            "quality_score": row.get("quality_score"),
            "ml_score": row.get("ml_score"),
            "ml_grade": row.get("ml_grade"),
            "commercial_tier": row.get("commercial_tier"),
            "series_dir": str(series_dir),
        }

        result.append(
            {
                "image": str(image_path),
                "label": label,
                "metadata": metadata,
            }
        )

    return result


def recommend_transforms(sequence_type: str) -> list[str]:
    """Return a list of recommended MONAI transform constructor strings.

    Parameters
    ----------
    sequence_type:
        One of ``"DWI"``, ``"FLAIR"``, ``"T1"``, ``"T2"``, ``"SWAN"``, ``"TOF"``.
        Case-insensitive.

    Returns
    -------
    list[str]
        Ordered list of MONAI transform constructor strings ready to eval
        or display to the buyer.

    Examples
    --------
    >>> transforms = recommend_transforms("DWI")
    >>> print("\\n".join(transforms))
    """
    return _TRANSFORM_MAP.get(sequence_type.upper(), _DEFAULT_TRANSFORMS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_image_path(series_dir: Path) -> Path:
    """Resolve best image path for MONAI LoadImaged.

    Priority:
    1. First .nii.gz file
    2. First .nii file
    3. Series DICOM directory itself (MONAI can load DICOM dirs)
    4. series_dir as fallback
    """
    if series_dir.exists():
        for nii in sorted(series_dir.glob("*.nii.gz")):
            return nii
        for nii in sorted(series_dir.glob("*.nii")):
            return nii
    return series_dir

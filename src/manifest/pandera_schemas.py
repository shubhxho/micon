"""
Pandera DataFrame schemas for the Speall MRI manifest parquet files.

Uses the ``pandera.polars`` backend (pandera >= 0.20).

Schemas defined
---------------
- ``MANIFEST_SCHEMA``       -- manifest.parquet (one row per series)
- ``STUDY_MANIFEST_SCHEMA`` -- study_manifest.parquet (one row per study)
- ``SPLITS_SCHEMA``         -- splits.parquet (study_id, split)

Public API
----------
    validate_manifest(df, schema_name) -> tuple[bool, list[str]]

CLI
---
    python -m src.manifest.pandera_schemas --manifest manifests/manifest.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import polars as pl
import pandera.polars as pa
from pandera.errors import SchemaErrors

# ---------------------------------------------------------------------------
# manifest.parquet schema  (mirrors _SERIES_SCHEMA in builder.py)
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA = pa.DataFrameSchema(
    columns={
        "study_id": pa.Column(pl.Utf8, nullable=True),
        "series_uid": pa.Column(pl.Utf8, nullable=True),
        "series_number": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "series_description": pa.Column(pl.Utf8, nullable=True),
        "sequence_type": pa.Column(pl.Utf8, nullable=True),
        "sequence_confidence": pa.Column(
            pl.Utf8,
            pa.Check.isin(["high", "medium", "low", "unknown"]),
            nullable=True,
        ),
        "modality": pa.Column(pl.Utf8, nullable=True),
        "file_count": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "tr_ms": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "te_ms": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "ti_ms": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "fa_deg": pa.Column(pl.Float64, pa.Check.between(0.0, 360.0), nullable=True),
        "b_value": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "field_strength_T": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "plane": pa.Column(
            pl.Utf8,
            pa.Check.isin(["axial", "sagittal", "coronal"]),
            nullable=True,
        ),
        "volume_shape": pa.Column(pl.List(pl.Int64), nullable=True),
        "spacing_mm": pa.Column(pl.List(pl.Float64), nullable=True),
        "fov_mm": pa.Column(pl.List(pl.Float64), nullable=True),
        "volume_snr": pa.Column(pl.Float64, nullable=True),
        "volume_cnr": pa.Column(pl.Float64, nullable=True),
        "volume_entropy": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "quality_grade": pa.Column(
            pl.Utf8,
            pa.Check.isin(["A", "B", "C", "D", "F"]),
            nullable=True,
        ),
        "quality_score": pa.Column(
            pl.Float64,
            pa.Check.between(0.0, 100.0),
            nullable=True,
        ),
        "ml_score": pa.Column(
            pl.Float64,
            pa.Check.between(0.0, 100.0),
            nullable=True,
        ),
        "ml_grade": pa.Column(pl.Utf8, nullable=True),
        "commercial_tier": pa.Column(pl.Utf8, nullable=True),
        "detail_path": pa.Column(pl.Utf8, nullable=True),
        "montage_path": pa.Column(pl.Utf8, nullable=True),
        "has_tar_shard": pa.Column(pl.Boolean, nullable=True),
    },
    strict=False,  # Allow extra columns from future schema evolution
    coerce=False,
)

# ---------------------------------------------------------------------------
# study_manifest.parquet schema  (mirrors _STUDY_SCHEMA in builder.py)
# ---------------------------------------------------------------------------

STUDY_MANIFEST_SCHEMA = pa.DataFrameSchema(
    columns={
        "study_id": pa.Column(pl.Utf8, nullable=True),
        "n_series": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "sequences_present": pa.Column(pl.List(pl.Utf8), nullable=True),
        "total_files": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "dominant_grade": pa.Column(
            pl.Utf8,
            pa.Check.isin(["A", "B", "C", "D", "F"]),
            nullable=True,
        ),
        "mean_ml_score": pa.Column(
            pl.Float64,
            pa.Check.between(0.0, 100.0),
            nullable=True,
        ),
        "has_dwi": pa.Column(pl.Boolean, nullable=True),
        "has_flair": pa.Column(pl.Boolean, nullable=True),
        "has_swan": pa.Column(pl.Boolean, nullable=True),
        "has_tof": pa.Column(pl.Boolean, nullable=True),
        "has_t1": pa.Column(pl.Boolean, nullable=True),
        "has_t2": pa.Column(pl.Boolean, nullable=True),
        "total_size_mb": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "tar_shard_path": pa.Column(pl.Utf8, nullable=True),
    },
    strict=False,
    coerce=False,
)

# ---------------------------------------------------------------------------
# splits.parquet schema  (study_id, split -- written by builder.write_manifests)
# ---------------------------------------------------------------------------

SPLITS_SCHEMA = pa.DataFrameSchema(
    columns={
        "study_id": pa.Column(pl.Utf8, nullable=False),
        "split": pa.Column(
            pl.Utf8,
            pa.Check.isin(["train", "val", "test"]),
            nullable=False,
        ),
    },
    strict=False,
    coerce=False,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "manifest": MANIFEST_SCHEMA,
    "study_manifest": STUDY_MANIFEST_SCHEMA,
    "splits": SPLITS_SCHEMA,
}

_SCHEMA_ALIASES: dict[str, str] = {
    "manifest.parquet": "manifest",
    "study_manifest.parquet": "study_manifest",
    "splits.parquet": "splits",
}


def validate_manifest(
    df: pl.DataFrame,
    schema_name: str,
) -> tuple[bool, list[str]]:
    """Run pandera validation and return ``(valid, errors)``.

    Parameters
    ----------
    df:
        Polars DataFrame to validate.
    schema_name:
        One of ``"manifest"``, ``"study_manifest"``, ``"splits"`` or the
        corresponding filename (``"manifest.parquet"`` etc.).

    Returns
    -------
    tuple[bool, list[str]]
        ``(True, [])`` on success; ``(False, [error_msg, ...])`` on failure.
    """
    resolved = _SCHEMA_ALIASES.get(schema_name, schema_name)
    schema = _SCHEMAS.get(resolved)
    if schema is None:
        raise ValueError(
            f"Unknown schema '{schema_name}'. "
            f"Choose from: {sorted(_SCHEMAS)}"
        )

    try:
        schema.validate(df, lazy=True)
        return True, []
    except SchemaErrors as exc:
        rows = exc.failure_cases.to_dicts()
        errors = [
            f"[{r.get('schema_context', '?')}] column={r.get('column', '?')} "
            f"check={r.get('check', '?')} failure={r.get('failure_case', '?')}"
            for r in rows
        ]
        return False, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a manifest parquet file against its pandera schema."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to manifest parquet file (manifest.parquet, study_manifest.parquet, splits.parquet).",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Schema name override (manifest | study_manifest | splits). Inferred from filename if omitted.",
    )
    args = parser.parse_args()

    path = Path(args.manifest)
    schema_name = args.schema or path.name

    df = pl.read_parquet(path)
    valid, errors = validate_manifest(df, schema_name)

    if valid:
        print(f"OK: {path} passed validation ({len(df)} rows).")
    else:
        print(f"FAILED: {path} has {len(errors)} violation(s):")
        for err in errors:
            print(f"  {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

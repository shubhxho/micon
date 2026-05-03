"""
Cohort query examples for the Speall MRI dataset.

Run any of these examples after downloading the manifest parquet files:

    manifest.parquet       -- series-level (one row per series)
    study_manifest.parquet -- study-level  (one row per study)
    splits.parquet         -- split assignments (study_id, split)

Usage::

    python examples/cohort_queries.py --manifest-dir /path/to/manifests

Each example loads the manifests, applies composable cohort filters from
``src.manifest.cohorts``, and prints the row count plus the first 5 rows.

The examples demonstrate common ML buyer queries and can be run before
downloading the full dataset to verify coverage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

# Make src importable when running from the repo root
_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.manifest.cohorts import (  # noqa: E402
    complete_protocol,
    dwi_only,
    grade_at_least,
    single_field,
    single_vendor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(manifest_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load manifest, study_manifest, and splits parquets from manifest_dir."""
    series_df = pl.read_parquet(manifest_dir / "manifest.parquet")
    study_df = pl.read_parquet(manifest_dir / "study_manifest.parquet")

    splits_path = manifest_dir / "splits.parquet"
    if splits_path.exists():
        splits_df = pl.read_parquet(splits_path)
        # Attach split to series-level manifest
        series_df = series_df.join(splits_df, on="study_id", how="left")
        study_df = study_df.join(splits_df, on="study_id", how="left")
    else:
        print(
            "WARNING: splits.parquet not found. "
            "Re-run builder with --with-splits to generate splits.\n"
            "Proceeding without split column.",
            file=sys.stderr,
        )

    return series_df, study_df, splits_df if splits_path.exists() else pl.DataFrame()


def _show(label: str, df: pl.DataFrame) -> None:
    """Print query label, row count, and first 5 rows."""
    sep = "-" * 72
    print(f"\n{sep}")
    print(f"QUERY: {label}")
    print(f"Rows: {len(df):,}")
    print(sep)
    if len(df) > 0:
        print(df.head(5))
    else:
        print("(no rows)")


# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------


def example_1_train_grade_a_dwi_ge_3t(series_df: pl.DataFrame) -> None:
    """All training-set Grade-A DWI series from GE 3T scanners."""
    df = series_df

    # Filter to training split if column present
    if "split" in df.columns:
        df = df.filter(pl.col("split") == "train")

    df = grade_at_least(df, "A")   # quality_grade == A
    df = dwi_only(df)              # sequence_type contains 'dwi'
    df = single_vendor(df, "GE")   # vendor == GE
    df = single_field(df, 3.0)     # field_strength_T ~ 3.0 T

    _show("Training-set Grade-A DWI series from GE 3T", df)


def example_2_complete_protocol_philips_15t(study_df: pl.DataFrame) -> None:
    """Studies with a complete T1+T2+FLAIR+DWI protocol on Philips 1.5T."""
    df = study_df
    df = single_vendor(df, "Philips")
    df = single_field(df, 1.5)
    df = complete_protocol(df, required=("T1-weighted", "T2-weighted", "FLAIR", "DWI"))

    _show("Complete T1+T2+FLAIR+DWI protocol on Philips 1.5T", df)


def example_3_100_random_val_stratified(study_df: pl.DataFrame) -> None:
    """100 random val-set studies stratified by vendor.

    Stratified sampling: draw up to 100 studies from the val split,
    preserving the vendor distribution of the full val set.
    """
    df = study_df

    if "split" not in df.columns:
        print("\nSKIPPED: splits column not present (run --with-splits).")
        return

    val_df = df.filter(pl.col("split") == "val")

    vendor_col = "vendor" if "vendor" in val_df.columns else None

    if vendor_col is None or len(val_df) == 0:
        # Fall back to simple random sample
        sample = val_df.sample(min(100, len(val_df)), seed=42)
    else:
        # Proportional stratified sample: compute per-vendor quota
        total = len(val_df)
        target = min(100, total)
        vendor_counts = val_df.group_by(vendor_col).agg(pl.len().alias("n"))
        vendor_counts = vendor_counts.with_columns(
            (pl.col("n") / total * target).round(0).cast(pl.Int64).alias("quota")
        )
        parts: list[pl.DataFrame] = []
        for row in vendor_counts.to_dicts():
            v = row[vendor_col]
            q = max(1, row["quota"])
            subset = val_df.filter(pl.col(vendor_col) == v)
            parts.append(subset.sample(min(q, len(subset)), seed=42))
        sample = pl.concat(parts)

    _show("100 random val-set studies stratified by vendor", sample)


def example_4_adc_and_dwi_across_vendors(series_df: pl.DataFrame) -> None:
    """Every study that has both ADC and DWI series (any vendor).

    Identifies study_ids where at least one series is DWI and at least one
    is ADC, then returns all series for those studies.
    """
    if "sequence_type" not in series_df.columns or "study_id" not in series_df.columns:
        print("\nSKIPPED: sequence_type or study_id column not present.")
        return

    low = pl.col("sequence_type").str.to_lowercase()

    # Studies with at least one DWI series
    dwi_studies = (
        series_df.filter(low.str.contains("dwi"))
        ["study_id"].unique()
    )

    # Studies with at least one ADC series
    adc_studies = (
        series_df.filter(low.str.contains("adc"))
        ["study_id"].unique()
    )

    both_studies = dwi_studies.filter(dwi_studies.is_in(adc_studies))
    df = series_df.filter(pl.col("study_id").is_in(both_studies))

    _show("All series for studies that have both ADC and DWI", df)


def example_5_post_2024_grade_b_plus(study_df: pl.DataFrame, series_df: pl.DataFrame) -> None:
    """All studies acquired after 2024-01-01 with grade >= B.

    Filters study manifest by acquisition date, then returns matching series.
    NOTE: The current manifest schema does not include an acquisition date
    column from DICOM -- this example shows the pattern for when that column
    is added.  If the column is absent, it proceeds with only the grade filter.
    """
    df = study_df
    df = grade_at_least(df, "B")   # dominant_grade in {A, B}

    if "acquisition_date" in df.columns:
        df = df.filter(pl.col("acquisition_date") >= pl.lit("2024-01-01"))
    else:
        print(
            "  NOTE: 'acquisition_date' column not present in study manifest; "
            "skipping date filter. Add acquisition_date extraction to builder.py "
            "to enable this filter."
        )

    _show("Studies acquired after 2024-01-01 with grade >= B", df)

    # Cross back to series level
    if len(df) > 0 and "study_id" in series_df.columns:
        series_subset = series_df.filter(pl.col("study_id").is_in(df["study_id"]))
        _show("  -> Series from above studies", series_subset)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run example cohort queries against Speall MRI manifest parquets."
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("."),
        help="Directory containing manifest.parquet, study_manifest.parquet, splits.parquet",
    )
    args = parser.parse_args()

    manifest_dir = args.manifest_dir
    if not manifest_dir.exists():
        print(f"ERROR: manifest-dir does not exist: {manifest_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading manifests from: {manifest_dir.resolve()}")
    series_df, study_df, _ = _load(manifest_dir)
    print(f"Series rows:  {len(series_df):,}")
    print(f"Study rows:   {len(study_df):,}")

    example_1_train_grade_a_dwi_ge_3t(series_df)
    example_2_complete_protocol_philips_15t(study_df)
    example_3_100_random_val_stratified(study_df)
    example_4_adc_and_dwi_across_vendors(series_df)
    example_5_post_2024_grade_b_plus(study_df, series_df)

    print("\nDone.")


if __name__ == "__main__":
    main()

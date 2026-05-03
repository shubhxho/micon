"""
Composable cohort filter helpers for the Speall MRI manifest.

Every function accepts a ``polars.DataFrame`` and returns a filtered
``polars.DataFrame`` so filters can be chained:

    df = (
        manifest
        | dwi_only
        | grade_a_only
        | single_vendor(vendor="GE")
    )

Or more explicitly::

    df = grade_a_only(dwi_only(manifest))

Sequence-type filters match against the ``sequence_type`` column (series-level
manifest).  Study-level filters match against study-level columns.  The helpers
work on both DataFrames -- they simply return 0 rows when the target column is
absent rather than raising.

Grade ordering convention (best to worst): A > B > C > D > F.
``grade_at_least("B")`` returns rows with grade in {A, B}.
"""

from __future__ import annotations

import warnings

import polars as pl

# ---------------------------------------------------------------------------
# Grade ordering (best to worst)
# ---------------------------------------------------------------------------

_GRADE_ORDER: list[str] = ["A", "B", "C", "D", "F"]


def _grades_at_least(min_grade: str) -> list[str]:
    """Return all grades >= min_grade (i.e. at-least-as-good-as)."""
    if min_grade not in _GRADE_ORDER:
        raise ValueError(f"Unknown grade '{min_grade}'. Valid grades: {_GRADE_ORDER}")
    cutoff = _GRADE_ORDER.index(min_grade)
    return _GRADE_ORDER[: cutoff + 1]


# ---------------------------------------------------------------------------
# Sequence-type filters (series-level manifest)
# ---------------------------------------------------------------------------


def _seq_filter(df: pl.DataFrame, keyword: str) -> pl.DataFrame:
    """Generic case-insensitive substring filter on sequence_type."""
    if "sequence_type" not in df.columns:
        warnings.warn(
            "Column 'sequence_type' not found; returning empty DataFrame.",
            UserWarning,
            stacklevel=3,
        )
        return df.head(0)
    return df.filter(pl.col("sequence_type").str.to_lowercase().str.contains(keyword))


def dwi_only(df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows where ``sequence_type`` contains 'dwi' (case-insensitive)."""
    return _seq_filter(df, "dwi")


def flair_only(df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows where ``sequence_type`` contains 'flair' (case-insensitive)."""
    return _seq_filter(df, "flair")


def t1_only(df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows where ``sequence_type`` contains 't1' (case-insensitive)."""
    return _seq_filter(df, "t1")


def t2_only(df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows where ``sequence_type`` contains 't2' (case-insensitive)."""
    return _seq_filter(df, "t2")


def swan_only(df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows where ``sequence_type`` contains 'swan' or 'swi'."""
    if "sequence_type" not in df.columns:
        warnings.warn(
            "Column 'sequence_type' not found; returning empty DataFrame.",
            UserWarning,
            stacklevel=2,
        )
        return df.head(0)
    low = pl.col("sequence_type").str.to_lowercase()
    return df.filter(low.str.contains("swan") | low.str.contains("swi"))


def tof_only(df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows where ``sequence_type`` contains 'tof' (time-of-flight)."""
    return _seq_filter(df, "tof")


# ---------------------------------------------------------------------------
# Grade filters
# ---------------------------------------------------------------------------


def grade_a_only(df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows where quality grade == 'A' (premium tier, score >= 80)."""
    return grade_at_least(df, "A")


def grade_at_least(df: pl.DataFrame, min_grade: str) -> pl.DataFrame:
    """Keep rows with quality grade >= ``min_grade`` (best-to-worst: A>B>C>D>F).

    ``grade_at_least(df, "B")`` returns rows graded A or B.
    ``grade_at_least(df, "A")`` returns only A rows (strictest subset).

    Checks both ``quality_grade`` (series-level) and ``dominant_grade``
    (study-level).  If neither column exists, warns and returns df unchanged.
    """
    allowed = _grades_at_least(min_grade)
    grade_col = None
    if "quality_grade" in df.columns:
        grade_col = "quality_grade"
    elif "dominant_grade" in df.columns:
        grade_col = "dominant_grade"

    if grade_col is None:
        warnings.warn(
            "Neither 'quality_grade' nor 'dominant_grade' column found; "
            "returning DataFrame unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return df

    return df.filter(pl.col(grade_col).is_in(allowed))


# ---------------------------------------------------------------------------
# Scanner / acquisition filters
# ---------------------------------------------------------------------------


def single_vendor(df: pl.DataFrame, vendor: str) -> pl.DataFrame:
    """Keep rows from a single scanner vendor (case-insensitive exact match).

    Checks column ``vendor`` if present; otherwise falls back to
    ``manufacturer`` (common DICOM column name).
    """
    vendor_col = None
    for candidate in ("vendor", "manufacturer"):
        if candidate in df.columns:
            vendor_col = candidate
            break

    if vendor_col is None:
        warnings.warn(
            "No 'vendor' or 'manufacturer' column found; returning DataFrame unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return df

    return df.filter(pl.col(vendor_col).str.to_lowercase() == vendor.lower())


def single_field(df: pl.DataFrame, field_T: float) -> pl.DataFrame:
    """Keep rows acquired at the specified field strength (in Tesla).

    Matches ``field_strength_T`` column within 0.05 T tolerance to handle
    floating-point imprecision in DICOM metadata.
    """
    col = "field_strength_T"
    if col not in df.columns:
        warnings.warn(
            f"Column '{col}' not found; returning DataFrame unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return df
    return df.filter((pl.col(col) - field_T).abs() <= 0.05)


# ---------------------------------------------------------------------------
# Pathology flag (placeholder)
# ---------------------------------------------------------------------------


def with_pathology(df: pl.DataFrame) -> pl.DataFrame:
    """Add a ``pathology_flag`` column (placeholder -- always 'unknown').

    This is a structural placeholder for future integration with radiology
    report NLP or structured finding fields.  The column marks where
    pathology annotation will live once the pipeline produces it.
    """
    return df.with_columns(pl.lit("unknown").cast(pl.Utf8).alias("pathology_flag"))


# ---------------------------------------------------------------------------
# Protocol completeness filter (study-level)
# ---------------------------------------------------------------------------


def complete_protocol(
    df: pl.DataFrame,
    required: tuple[str, ...] = ("DWI", "FLAIR", "T1-weighted", "T2-weighted"),
) -> pl.DataFrame:
    """Keep studies that have ALL sequences listed in ``required``.

    Expects a ``sequences_present`` column of type ``List[Utf8]`` (as produced
    by :func:`src.manifest.builder.build_study_manifest`).  Each entry in
    ``required`` is matched case-insensitively as a substring of any element
    in ``sequences_present``.

    For series-level DataFrames, groups by ``study_id``, checks coverage, then
    filters.  Falls back gracefully if the column is absent.
    """
    if "sequences_present" in df.columns:
        return _complete_protocol_study_level(df, required)

    if "study_id" in df.columns and "sequence_type" in df.columns:
        return _complete_protocol_series_level(df, required)

    warnings.warn(
        "Cannot determine protocol completeness: need 'sequences_present' "
        "(study-level) or 'sequence_type' + 'study_id' (series-level).",
        UserWarning,
        stacklevel=2,
    )
    return df


def _complete_protocol_study_level(df: pl.DataFrame, required: tuple[str, ...]) -> pl.DataFrame:
    """Study-level completeness via sequences_present List column."""

    def _has_all(seq_list: list[str] | None) -> bool:
        if not seq_list:
            return False
        lower_seqs = [s.lower() for s in seq_list]
        return all(any(req.lower() in s for s in lower_seqs) for req in required)

    mask = [_has_all(row) for row in df["sequences_present"].to_list()]
    return df.filter(pl.Series(mask))


def _complete_protocol_series_level(df: pl.DataFrame, required: tuple[str, ...]) -> pl.DataFrame:
    """Series-level: group by study_id, check coverage, return matching series."""
    study_seq = df.group_by("study_id").agg(pl.col("sequence_type").drop_nulls().alias("seqs"))

    def _has_all(seq_list: list[str] | None) -> bool:
        if not seq_list:
            return False
        lower_seqs = [s.lower() for s in seq_list]
        return all(any(req.lower() in s for s in lower_seqs) for req in required)

    mask = [_has_all(row) for row in study_seq["seqs"].to_list()]
    complete_studies = study_seq.filter(pl.Series(mask))["study_id"]
    return df.filter(pl.col("study_id").is_in(complete_studies))

"""
Patient-level (study-level) train/val/test split assignment for Speall MRI.

Splits are computed at the STUDY level so that no patient's data leaks across
train / val / test -- the most common bug in medical imaging ML pipelines.

Stratification is applied across the Cartesian product of ``stratify_by``
columns (e.g. vendor x dominant_grade).  Columns that are absent from the
DataFrame are silently dropped with a warning, so the function degrades
gracefully on partial schemas.

Tiny strata (< 2 members) cannot be split by StratifiedShuffleSplit; they are
folded into a synthetic "__other__" stratum and split together with the rest.
"""

from __future__ import annotations

import sys
import warnings
from typing import Sequence

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedShuffleSplit

# Sentinel used for null values in the combined stratification label
_NULL_SENTINEL = "__null__"
# Sentinel for tiny-strata merge
_OTHER_SENTINEL = "__other__"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assign_splits(
    series_df: pl.DataFrame,
    study_df: pl.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    stratify_by: Sequence[str] = ("vendor", "dominant_grade"),
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Assign train / val / test splits at the STUDY level.

    Parameters
    ----------
    series_df:
        Series-level manifest (one row per series).  Must contain ``study_id``.
    study_df:
        Study-level manifest (one row per study).  Must contain ``study_id``.
    ratios:
        (train, val, test) fractions.  Must sum to 1.0 within floating-point
        tolerance.
    stratify_by:
        Column names in *study_df* to stratify over.  Missing columns are
        dropped with a warning.  If no valid columns remain, stratification is
        skipped and a plain shuffle split is used.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (series_df_with_split, study_df_with_split)
        Both DataFrames with a new ``split`` column ("train" | "val" | "test").
        The series-level column is derived by joining study-level assignments on
        ``study_id`` -- series are never assigned splits independently.

    Notes
    -----
    - Idempotent: if ``split`` already exists in either DataFrame, a warning is
      emitted and both are returned unchanged.
    - Empty input: returns both DataFrames with an empty ``split`` column of
      type ``Utf8``.
    """
    # Validate ratios
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios):.6f}")
    if len(ratios) != 3:
        raise ValueError("ratios must be a 3-tuple (train, val, test)")
    for r in ratios:
        if r <= 0:
            raise ValueError(f"All ratios must be positive, got {r}")

    # Idempotency check
    if "split" in study_df.columns or "split" in series_df.columns:
        warnings.warn(
            "'split' column already present -- skipping split assignment.",
            UserWarning,
            stacklevel=2,
        )
        return series_df, study_df

    # Empty input guard
    if len(study_df) == 0:
        empty_study = study_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("split"))
        empty_series = series_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("split"))
        return empty_series, empty_study

    train_ratio, val_ratio, test_ratio = ratios

    # Resolve stratification columns
    valid_strat_cols = _resolve_stratify_cols(study_df, list(stratify_by))

    # Build study-level split assignment
    study_ids = study_df["study_id"].to_list()
    n = len(study_ids)
    idx_all = np.arange(n)

    # Build stratification labels
    if valid_strat_cols:
        raw_labels = _build_strat_labels(study_df, valid_strat_cols)
        labels = _merge_tiny_strata(raw_labels)
    else:
        labels = None

    # Two-pass split: first split out test, then split val from train+val
    test_size = test_ratio
    # After removing test, val fraction of the remainder
    val_of_remainder = val_ratio / (train_ratio + val_ratio)

    idx_trainval, idx_test = _do_split(idx_all, labels, test_size, seed)

    # Labels for second split
    labels_trainval = labels[idx_trainval] if labels is not None else None
    idx_train, idx_val = _do_split(idx_trainval, labels_trainval, val_of_remainder, seed + 1)

    # Build split map: study_id -> split label
    split_map: dict[str, str] = {}
    for i in idx_train:
        split_map[study_ids[i]] = "train"
    for i in idx_val:
        split_map[study_ids[i]] = "val"
    for i in idx_test:
        split_map[study_ids[i]] = "test"

    # Apply to study_df
    study_df_out = study_df.with_columns(
        pl.col("study_id").replace_strict(split_map, return_dtype=pl.Utf8).alias("split")
    )

    # Propagate to series_df via join (never re-randomise per series)
    split_series = study_df_out.select(["study_id", "split"])
    series_df_out = series_df.join(split_series, on="study_id", how="left")

    return series_df_out, study_df_out


def summarize_splits(study_df: pl.DataFrame) -> dict:
    """Return counts per split per stratum as a nested dict.

    Expects ``study_df`` to have a ``split`` column (produced by
    :func:`assign_splits`).  Also reports dominant_grade distribution if
    present.

    Returns
    -------
    dict with keys: "totals" (per split), "by_dominant_grade" (optional),
    "by_vendor" (optional), "total_studies" (int).
    """
    if "split" not in study_df.columns:
        raise ValueError("study_df does not have a 'split' column; run assign_splits first.")

    result: dict = {"total_studies": len(study_df)}

    # Per-split totals
    result["totals"] = (
        study_df.group_by("split")
        .agg(pl.len().alias("count"))
        .sort("split")
        .to_dicts()
    )

    # Per-stratum breakdown
    for col in ("dominant_grade", "vendor"):
        if col in study_df.columns:
            result[f"by_{col}"] = (
                study_df.group_by(["split", col])
                .agg(pl.len().alias("count"))
                .sort(["split", col])
                .to_dicts()
            )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_stratify_cols(df: pl.DataFrame, cols: list[str]) -> list[str]:
    """Return only those cols present in df; warn about missing ones."""
    valid = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warnings.warn(
            f"stratify_by columns not found in study_df and will be ignored: {missing}",
            UserWarning,
            stacklevel=3,
        )
    return valid


def _build_strat_labels(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Build a 1-D array of combined string labels (e.g. 'GE|A')."""
    parts = [
        df[c].fill_null(_NULL_SENTINEL).cast(pl.Utf8).to_list() for c in cols
    ]
    return np.array(["|".join(vals) for vals in zip(*parts)])


def _merge_tiny_strata(labels: np.ndarray, min_count: int = 2) -> np.ndarray:
    """Fold strata with fewer than min_count members into '__other__'."""
    unique, counts = np.unique(labels, return_counts=True)
    tiny = set(unique[counts < min_count])
    if not tiny:
        return labels
    warnings.warn(
        f"Folding {len(tiny)} tiny strata (< {min_count} members) into "
        f"'{_OTHER_SENTINEL}': {sorted(tiny)[:5]}{'...' if len(tiny) > 5 else ''}",
        UserWarning,
        stacklevel=3,
    )
    result = labels.copy()
    for t in tiny:
        result[result == t] = _OTHER_SENTINEL
    return result


def _do_split(
    idx: np.ndarray,
    labels: np.ndarray | None,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split idx into (remainder, held-out) using StratifiedShuffleSplit.

    Falls back to a plain random split if labels are None or stratification
    fails (e.g. test_size produces < 1 sample).
    """
    if len(idx) == 0:
        return idx, np.array([], dtype=int)

    # Guard: test_size must yield at least 1 sample
    n_test = max(1, int(round(len(idx) * test_size)))
    if n_test >= len(idx):
        n_test = len(idx) - 1

    if labels is not None:
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
            sub_labels = labels  # already aligned with idx positionally
            train_rel, test_rel = next(sss.split(idx, sub_labels))
            return idx[train_rel], idx[test_rel]
        except ValueError:
            # Fall back to plain random split
            warnings.warn(
                "StratifiedShuffleSplit failed (likely too few samples per stratum); "
                "falling back to unstratified random split.",
                UserWarning,
                stacklevel=3,
            )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(idx))
    test_rel = shuffled[:n_test]
    train_rel = shuffled[n_test:]
    return idx[train_rel], idx[test_rel]

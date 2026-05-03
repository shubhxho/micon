"""
Tests for src.manifest.splits and src.manifest.cohorts.

Covers:
- assign_splits is deterministic with seed=42
- Same study NEVER appears in two splits (no patient leakage)
- Stratification within +/- 5% per stratum
- Empty input -> empty output (no crash)
- cohorts.grade_at_least is monotonic (>= "A" is strictest subset)
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from src.manifest.splits import assign_splits, summarize_splits
from src.manifest.cohorts import (
    complete_protocol,
    dwi_only,
    flair_only,
    grade_a_only,
    grade_at_least,
    single_field,
    single_vendor,
    swan_only,
    t1_only,
    t2_only,
    tof_only,
    with_pathology,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_study_df(n: int = 300, seed: int = 0) -> pl.DataFrame:
    """Synthetic study-level DataFrame with vendor + dominant_grade columns."""
    import random

    rng = random.Random(seed)
    vendors = ["GE", "Philips", "Siemens", "Toshiba"]
    grades = ["A", "B", "C", "D", "F"]

    rows = []
    for i in range(n):
        rows.append(
            {
                "study_id": f"study_{i:04d}",
                "dominant_grade": rng.choice(grades),
                "vendor": rng.choice(vendors),
                "n_series": rng.randint(1, 20),
                "sequences_present": rng.sample(
                    ["DWI", "FLAIR", "T1-weighted", "T2-weighted", "SWAN", "TOF"],
                    k=rng.randint(1, 4),
                ),
            }
        )
    return pl.DataFrame(rows)


def _make_series_df(study_df: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Synthetic series-level DataFrame with one row per series per study."""
    import random

    rng = random.Random(seed)
    seq_types = ["DWI", "FLAIR", "T1-weighted", "T2-weighted", "SWAN", "TOF", "ADC"]
    grades = ["A", "B", "C", "D", "F"]

    rows = []
    for row in study_df.to_dicts():
        n = rng.randint(1, 5)
        for j in range(n):
            rows.append(
                {
                    "study_id": row["study_id"],
                    "series_uid": f"{row['study_id']}_s{j}",
                    "sequence_type": rng.choice(seq_types),
                    "quality_grade": rng.choice(grades),
                    "field_strength_T": rng.choice([1.5, 3.0]),
                    "vendor": row["vendor"],
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# splits: determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        study_df = _make_study_df(n=200, seed=7)
        series_df = _make_series_df(study_df, seed=7)

        _, study1 = assign_splits(series_df, study_df, seed=42)
        # Must call on fresh copies (no 'split' column yet)
        study_df2 = _make_study_df(n=200, seed=7)
        series_df2 = _make_series_df(study_df2, seed=7)
        _, study2 = assign_splits(series_df2, study_df2, seed=42)

        assert study1["split"].to_list() == study2["split"].to_list()

    def test_different_seeds_differ(self) -> None:
        study_df = _make_study_df(n=200, seed=7)
        series_df = _make_series_df(study_df, seed=7)

        study_df2 = _make_study_df(n=200, seed=7)
        series_df2 = _make_series_df(study_df2, seed=7)

        _, study1 = assign_splits(series_df, study_df, seed=42)
        _, study2 = assign_splits(series_df2, study_df2, seed=99)

        # With 200 studies it's astronomically unlikely they match
        assert study1["split"].to_list() != study2["split"].to_list()


# ---------------------------------------------------------------------------
# splits: no patient leakage
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_each_study_in_exactly_one_split(self) -> None:
        study_df = _make_study_df(n=300)
        series_df = _make_series_df(study_df)

        series_out, study_out = assign_splits(series_df, study_df)

        # Collect sets per split at study level
        splits = ["train", "val", "test"]
        split_sets = {
            s: set(study_out.filter(pl.col("split") == s)["study_id"].to_list())
            for s in splits
        }

        # No study appears in more than one split
        train_val = split_sets["train"] & split_sets["val"]
        train_test = split_sets["train"] & split_sets["test"]
        val_test = split_sets["val"] & split_sets["test"]

        assert len(train_val) == 0, f"train/val overlap: {train_val}"
        assert len(train_test) == 0, f"train/test overlap: {train_test}"
        assert len(val_test) == 0, f"val/test overlap: {val_test}"

    def test_all_studies_assigned(self) -> None:
        study_df = _make_study_df(n=300)
        series_df = _make_series_df(study_df)

        _, study_out = assign_splits(series_df, study_df)

        assert study_out["split"].null_count() == 0
        assert set(study_out["split"].to_list()).issubset({"train", "val", "test"})

    def test_series_split_matches_study_split(self) -> None:
        """Series belonging to the same study must all be in the same split."""
        study_df = _make_study_df(n=100)
        series_df = _make_series_df(study_df)

        series_out, study_out = assign_splits(series_df, study_df)

        # Build expected study->split map
        study_split_map = dict(
            zip(study_out["study_id"].to_list(), study_out["split"].to_list())
        )

        for row in series_out.to_dicts():
            expected = study_split_map[row["study_id"]]
            assert row["split"] == expected, (
                f"Series {row['series_uid']} study {row['study_id']}: "
                f"expected split '{expected}', got '{row['split']}'"
            )


# ---------------------------------------------------------------------------
# splits: ratios
# ---------------------------------------------------------------------------


class TestRatios:
    def test_approximate_ratios(self) -> None:
        study_df = _make_study_df(n=500)
        series_df = _make_series_df(study_df)

        _, study_out = assign_splits(
            series_df, study_df, ratios=(0.8, 0.1, 0.1), seed=42
        )

        n = len(study_out)
        for split, expected_frac in [("train", 0.8), ("val", 0.1), ("test", 0.1)]:
            actual = len(study_out.filter(pl.col("split") == split)) / n
            assert abs(actual - expected_frac) <= 0.05, (
                f"{split}: expected ~{expected_frac:.0%}, got {actual:.0%}"
            )

    def test_invalid_ratios_raise(self) -> None:
        study_df = _make_study_df(n=50)
        series_df = _make_series_df(study_df)

        with pytest.raises(ValueError, match="sum to 1.0"):
            assign_splits(series_df, study_df, ratios=(0.7, 0.1, 0.1))


# ---------------------------------------------------------------------------
# splits: stratification balance
# ---------------------------------------------------------------------------


class TestStratification:
    def test_stratification_within_5_percent(self) -> None:
        """Each stratum's proportion in train should be within 5% of overall."""
        study_df = _make_study_df(n=400, seed=3)
        series_df = _make_series_df(study_df, seed=3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, study_out = assign_splits(
                series_df,
                study_df,
                ratios=(0.8, 0.1, 0.1),
                stratify_by=("vendor", "dominant_grade"),
                seed=42,
            )

        train_df = study_out.filter(pl.col("split") == "train")
        n_train = len(train_df)
        n_total = len(study_out)

        # Check vendor balance
        for vendor_row in study_out.group_by("vendor").agg(pl.len().alias("n")).to_dicts():
            vendor = vendor_row["vendor"]
            overall_frac = vendor_row["n"] / n_total
            train_count = len(train_df.filter(pl.col("vendor") == vendor))
            train_frac = train_count / n_train if n_train > 0 else 0.0
            assert abs(train_frac - overall_frac) <= 0.05, (
                f"Vendor {vendor}: overall {overall_frac:.1%}, train {train_frac:.1%}"
            )

    def test_missing_stratify_col_warns(self) -> None:
        study_df = _make_study_df(n=100)
        series_df = _make_series_df(study_df)

        with pytest.warns(UserWarning, match="not found"):
            assign_splits(
                series_df,
                study_df,
                stratify_by=("nonexistent_col",),
                seed=42,
            )


# ---------------------------------------------------------------------------
# splits: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_study_df_returns_empty_with_split_col(self) -> None:
        empty_study = pl.DataFrame(
            {"study_id": pl.Series([], dtype=pl.Utf8), "dominant_grade": pl.Series([], dtype=pl.Utf8)}
        )
        empty_series = pl.DataFrame(
            {"study_id": pl.Series([], dtype=pl.Utf8), "series_uid": pl.Series([], dtype=pl.Utf8)}
        )
        s_out, st_out = assign_splits(empty_series, empty_study)
        assert "split" in s_out.columns
        assert "split" in st_out.columns
        assert len(s_out) == 0
        assert len(st_out) == 0

    def test_idempotent_warns_and_skips(self) -> None:
        study_df = _make_study_df(n=50).with_columns(pl.lit("train").alias("split"))
        series_df = _make_series_df(_make_study_df(n=50))

        with pytest.warns(UserWarning, match="already present"):
            s_out, st_out = assign_splits(series_df, study_df)

        # Should return originals unchanged
        assert "split" in st_out.columns
        assert all(v == "train" for v in st_out["split"].to_list())

    def test_small_dataset_no_crash(self) -> None:
        study_df = _make_study_df(n=10)
        series_df = _make_series_df(study_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_out, st_out = assign_splits(series_df, study_df, seed=42)
        assert "split" in st_out.columns
        assert len(st_out) == 10


# ---------------------------------------------------------------------------
# splits: summarize_splits
# ---------------------------------------------------------------------------


class TestSummarizeSplits:
    def test_summarize_totals(self) -> None:
        study_df = _make_study_df(n=300)
        series_df = _make_series_df(study_df)
        _, study_out = assign_splits(series_df, study_df)

        summary = summarize_splits(study_out)
        total_in_summary = sum(row["count"] for row in summary["totals"])
        assert total_in_summary == len(study_out)

    def test_summarize_raises_without_split_col(self) -> None:
        study_df = _make_study_df(n=50)
        with pytest.raises(ValueError, match="split"):
            summarize_splits(study_df)


# ---------------------------------------------------------------------------
# cohorts: grade_at_least monotonicity
# ---------------------------------------------------------------------------


class TestGradeAtLeast:
    def _grade_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "quality_grade": ["A", "A", "B", "B", "B", "C", "C", "D", "F", "F"],
                "study_id": [f"s{i}" for i in range(10)],
            }
        )

    def test_monotonic_subset(self) -> None:
        df = self._grade_df()
        # grade_at_least("A") subset of grade_at_least("B") subset of ...
        a_set = set(grade_at_least(df, "A")["study_id"].to_list())
        b_set = set(grade_at_least(df, "B")["study_id"].to_list())
        c_set = set(grade_at_least(df, "C")["study_id"].to_list())
        d_set = set(grade_at_least(df, "D")["study_id"].to_list())
        f_set = set(grade_at_least(df, "F")["study_id"].to_list())

        assert a_set <= b_set <= c_set <= d_set <= f_set

    def test_grade_a_only_is_subset_of_grade_b(self) -> None:
        df = self._grade_df()
        a_ids = set(grade_a_only(df)["study_id"].to_list())
        b_ids = set(grade_at_least(df, "B")["study_id"].to_list())
        assert a_ids <= b_ids

    def test_grade_f_returns_all(self) -> None:
        df = self._grade_df()
        f_df = grade_at_least(df, "F")
        assert len(f_df) == len(df)

    def test_grade_a_returns_only_a(self) -> None:
        df = self._grade_df()
        a_df = grade_at_least(df, "A")
        assert set(a_df["quality_grade"].to_list()) == {"A"}

    def test_invalid_grade_raises(self) -> None:
        df = self._grade_df()
        with pytest.raises(ValueError, match="Unknown grade"):
            grade_at_least(df, "Z")


# ---------------------------------------------------------------------------
# cohorts: sequence type filters
# ---------------------------------------------------------------------------


class TestSequenceFilters:
    def _seq_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "sequence_type": [
                    "DWI", "DWI b1000", "FLAIR", "T1-weighted", "T2-weighted",
                    "SWAN", "SWI", "TOF", "ADC", None,
                ],
                "study_id": [f"s{i}" for i in range(10)],
            }
        )

    def test_dwi_only(self) -> None:
        df = self._seq_df()
        out = dwi_only(df)
        assert all("dwi" in (v or "").lower() for v in out["sequence_type"].to_list())
        assert len(out) >= 2

    def test_flair_only(self) -> None:
        out = flair_only(self._seq_df())
        assert len(out) == 1

    def test_t1_only(self) -> None:
        out = t1_only(self._seq_df())
        assert len(out) == 1

    def test_t2_only(self) -> None:
        out = t2_only(self._seq_df())
        assert len(out) == 1

    def test_swan_only(self) -> None:
        out = swan_only(self._seq_df())
        assert len(out) == 2  # SWAN + SWI

    def test_tof_only(self) -> None:
        out = tof_only(self._seq_df())
        assert len(out) == 1


# ---------------------------------------------------------------------------
# cohorts: single_vendor, single_field
# ---------------------------------------------------------------------------


class TestScannerFilters:
    def _scanner_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "study_id": ["s1", "s2", "s3", "s4"],
                "vendor": ["GE", "Philips", "GE", "Siemens"],
                "field_strength_T": [3.0, 1.5, 1.5, 3.0],
            }
        )

    def test_single_vendor(self) -> None:
        out = single_vendor(self._scanner_df(), "GE")
        assert len(out) == 2
        assert all(v == "GE" for v in out["vendor"].to_list())

    def test_single_vendor_case_insensitive(self) -> None:
        out = single_vendor(self._scanner_df(), "ge")
        assert len(out) == 2

    def test_single_field(self) -> None:
        out = single_field(self._scanner_df(), 1.5)
        assert len(out) == 2

    def test_missing_vendor_col_warns(self) -> None:
        df = pl.DataFrame({"study_id": ["s1", "s2"]})
        with pytest.warns(UserWarning):
            out = single_vendor(df, "GE")
        assert len(out) == len(df)


# ---------------------------------------------------------------------------
# cohorts: complete_protocol
# ---------------------------------------------------------------------------


class TestCompleteProtocol:
    def test_complete_protocol_study_level(self) -> None:
        df = pl.DataFrame(
            {
                "study_id": ["complete", "partial", "empty"],
                "sequences_present": [
                    ["DWI", "FLAIR", "T1-weighted", "T2-weighted"],
                    ["DWI", "FLAIR"],
                    [],
                ],
            }
        )
        out = complete_protocol(df, required=("DWI", "FLAIR", "T1-weighted", "T2-weighted"))
        assert len(out) == 1
        assert out["study_id"][0] == "complete"

    def test_partial_required_passes(self) -> None:
        df = pl.DataFrame(
            {
                "study_id": ["s1", "s2"],
                "sequences_present": [
                    ["DWI", "FLAIR"],
                    ["DWI"],
                ],
            }
        )
        out = complete_protocol(df, required=("DWI", "FLAIR"))
        assert len(out) == 1
        assert out["study_id"][0] == "s1"


# ---------------------------------------------------------------------------
# cohorts: with_pathology placeholder
# ---------------------------------------------------------------------------


class TestWithPathology:
    def test_adds_pathology_flag_column(self) -> None:
        df = pl.DataFrame({"study_id": ["s1", "s2"]})
        out = with_pathology(df)
        assert "pathology_flag" in out.columns
        assert all(v == "unknown" for v in out["pathology_flag"].to_list())

    def test_original_columns_preserved(self) -> None:
        df = pl.DataFrame({"study_id": ["s1"], "vendor": ["GE"]})
        out = with_pathology(df)
        assert "study_id" in out.columns
        assert "vendor" in out.columns

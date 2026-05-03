"""
Tests for src.manifest.confidence_summary.study_confidence_rollup.

TDD London School: each test drives one observable behaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.manifest.confidence_summary import study_confidence_rollup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_annotation(directory: Path, stem: str, consensus: dict) -> None:
    """Write a minimal annotation JSON with a consensus block."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(json.dumps({"consensus": consensus}), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmptyDir:
    def test_nonexistent_dir_returns_safe_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_dir"
        result = study_confidence_rollup(missing)
        assert result["n_series"] == 0
        assert result["mean_confidence"] == 0.0
        assert result["min_confidence"] == 0.0
        assert result["pct_low_confidence"] == 0.0
        assert result["n_needs_escalation"] == 0
        assert result["pct_premium_used"] == 0.0
        assert result["series_breakdown"] == []

    def test_empty_dir_returns_safe_defaults(self, tmp_path: Path) -> None:
        empty = tmp_path / "annotations"
        empty.mkdir()
        result = study_confidence_rollup(empty)
        assert result["n_series"] == 0
        assert result["series_breakdown"] == []

    def test_dir_with_no_valid_json_returns_safe_defaults(self, tmp_path: Path) -> None:
        d = tmp_path / "annotations"
        d.mkdir()
        (d / "series.json").write_text('{"no_consensus": true}', encoding="utf-8")
        result = study_confidence_rollup(d)
        assert result["n_series"] == 0


class TestTwoSeriesRollup:
    """One series with confidence 0.9 (no escalation, no premium),
    one with confidence 0.4 (needs escalation, premium used).
    Expected: mean=0.65, pct_low=0.5, n_needs_escalation=1, pct_premium_used=0.5.
    """

    @pytest.fixture()
    def two_series_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "annotations"
        _write_annotation(
            d,
            "s0001_T1",
            {
                "confidence": 0.9,
                "needs_escalation": False,
                "tiers_used": ["cheap"],
                "premium_used": False,
            },
        )
        _write_annotation(
            d,
            "s0002_DWI",
            {
                "confidence": 0.4,
                "needs_escalation": True,
                "tiers_used": ["cheap", "premium"],
                "premium_used": True,
            },
        )
        return d

    def test_n_series(self, two_series_dir: Path) -> None:
        result = study_confidence_rollup(two_series_dir)
        assert result["n_series"] == 2

    def test_mean_confidence(self, two_series_dir: Path) -> None:
        result = study_confidence_rollup(two_series_dir)
        assert abs(result["mean_confidence"] - 0.65) < 1e-9

    def test_min_confidence(self, two_series_dir: Path) -> None:
        result = study_confidence_rollup(two_series_dir)
        assert abs(result["min_confidence"] - 0.4) < 1e-9

    def test_pct_low_confidence(self, two_series_dir: Path) -> None:
        # Only s0002 has confidence 0.4 < 0.6 threshold; s0001 = 0.9 is above.
        result = study_confidence_rollup(two_series_dir)
        assert abs(result["pct_low_confidence"] - 0.5) < 1e-9

    def test_threshold_excludes_exactly_06(self, tmp_path: Path) -> None:
        # Confidence == 0.6 must NOT be counted as low (strict < 0.6).
        d = tmp_path / "annotations"
        _write_annotation(
            d,
            "edge",
            {"confidence": 0.6, "needs_escalation": False, "tiers_used": [], "premium_used": False},
        )
        result = study_confidence_rollup(d)
        assert result["pct_low_confidence"] == 0.0

    def test_n_needs_escalation(self, two_series_dir: Path) -> None:
        result = study_confidence_rollup(two_series_dir)
        assert result["n_needs_escalation"] == 1

    def test_pct_premium_used(self, two_series_dir: Path) -> None:
        result = study_confidence_rollup(two_series_dir)
        assert abs(result["pct_premium_used"] - 0.5) < 1e-9

    def test_series_breakdown_length(self, two_series_dir: Path) -> None:
        result = study_confidence_rollup(two_series_dir)
        assert len(result["series_breakdown"]) == 2

    def test_series_breakdown_fields(self, two_series_dir: Path) -> None:
        result = study_confidence_rollup(two_series_dir)
        for entry in result["series_breakdown"]:
            assert "series_label" in entry
            assert "confidence" in entry
            assert "needs_escalation" in entry
            assert "tiers_used" in entry

    def test_series_breakdown_values(self, two_series_dir: Path) -> None:
        result = study_confidence_rollup(two_series_dir)
        # Sort by series_label for determinism
        entries = sorted(result["series_breakdown"], key=lambda e: e["series_label"])
        assert entries[0]["series_label"] == "s0001_T1"
        assert abs(entries[0]["confidence"] - 0.9) < 1e-9
        assert entries[0]["needs_escalation"] is False
        assert entries[1]["series_label"] == "s0002_DWI"
        assert abs(entries[1]["confidence"] - 0.4) < 1e-9
        assert entries[1]["needs_escalation"] is True


class TestEdgeCases:
    def test_single_series_above_threshold(self, tmp_path: Path) -> None:
        d = tmp_path / "annotations"
        _write_annotation(
            d,
            "s0001",
            {
                "confidence": 0.85,
                "needs_escalation": False,
                "tiers_used": ["cheap"],
                "premium_used": False,
            },
        )
        result = study_confidence_rollup(d)
        assert result["n_series"] == 1
        assert result["pct_low_confidence"] == 0.0
        assert result["n_needs_escalation"] == 0

    def test_all_premium(self, tmp_path: Path) -> None:
        d = tmp_path / "annotations"
        for i in range(3):
            _write_annotation(
                d,
                f"s{i:04d}",
                {
                    "confidence": 0.7,
                    "needs_escalation": False,
                    "tiers_used": ["premium"],
                    "premium_used": True,
                },
            )
        result = study_confidence_rollup(d)
        assert result["pct_premium_used"] == 1.0

    def test_malformed_json_is_skipped(self, tmp_path: Path) -> None:
        d = tmp_path / "annotations"
        d.mkdir()
        (d / "broken.json").write_text("{not valid json", encoding="utf-8")
        _write_annotation(
            d,
            "good",
            {"confidence": 0.8, "needs_escalation": False, "tiers_used": [], "premium_used": False},
        )
        result = study_confidence_rollup(d)
        assert result["n_series"] == 1

    def test_missing_confidence_key_is_skipped(self, tmp_path: Path) -> None:
        d = tmp_path / "annotations"
        _write_annotation(d, "no_conf", {"needs_escalation": False, "tiers_used": []})
        result = study_confidence_rollup(d)
        assert result["n_series"] == 0

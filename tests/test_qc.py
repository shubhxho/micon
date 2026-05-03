"""Tests for src.qc modules.

Covers:
- per-study QC report generation on Speall_MRI_Samples
- dataset QC report generation from Speall_MRI_Dataset_Info.json
- SVG badge generation + XML validity
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SAMPLES_DIR = REPO_ROOT / "Speall_MRI_Samples"
DATASET_INFO = REPO_ROOT / "Speall_MRI_Dataset_Info.json"


# ---------------------------------------------------------------------------
# per_study
# ---------------------------------------------------------------------------


class TestPerStudyReport:
    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        from src.qc.per_study import build_study_qc_report

        out = tmp_path / "study_qc.html"
        result = build_study_qc_report(SAMPLES_DIR, out)

        assert set(result.keys()) == {
            "n_series",
            "grade_counts",
            "anomalies_count",
            "conformance_issues_count",
        }
        assert isinstance(result["n_series"], int)
        assert result["n_series"] > 0
        assert isinstance(result["grade_counts"], dict)
        assert all(g in result["grade_counts"] for g in "ABCDF")

    def test_output_html_non_empty(self, tmp_path: Path) -> None:
        from src.qc.per_study import build_study_qc_report

        out = tmp_path / "study_qc.html"
        build_study_qc_report(SAMPLES_DIR, out)

        assert out.exists()
        assert out.stat().st_size > 1024  # at least 1 KB

    def test_output_is_parseable_html5(self, tmp_path: Path) -> None:
        from src.qc.per_study import build_study_qc_report

        out = tmp_path / "study_qc.html"
        build_study_qc_report(SAMPLES_DIR, out)

        content = out.read_text(encoding="utf-8")
        # Must start with <!DOCTYPE html> (case-insensitive)
        assert content.strip().lower().startswith("<!doctype html>")
        # Must contain the opening and closing html tags
        assert "<html" in content
        assert "</html>" in content

    def test_output_contains_study_metadata(self, tmp_path: Path) -> None:
        from src.qc.per_study import build_study_qc_report

        out = tmp_path / "study_qc.html"
        build_study_qc_report(SAMPLES_DIR, out)

        content = out.read_text(encoding="utf-8")
        # Study description from study_summary.json
        assert "BRAIN" in content
        # Grade labels
        assert "Grade A" in content or "grade-a" in content

    def test_conformance_section_present(self, tmp_path: Path) -> None:
        from src.qc.per_study import build_study_qc_report

        out = tmp_path / "study_qc.html"
        build_study_qc_report(SAMPLES_DIR, out)

        content = out.read_text(encoding="utf-8")
        assert "Conformance" in content

    def test_output_under_5mb(self, tmp_path: Path) -> None:
        from src.qc.per_study import build_study_qc_report

        out = tmp_path / "study_qc.html"
        build_study_qc_report(SAMPLES_DIR, out)

        size_mb = out.stat().st_size / (1024 * 1024)
        assert size_mb < 5.0, f"Report is {size_mb:.2f} MB, exceeds 5 MB limit"

    def test_custom_template(self, tmp_path: Path) -> None:
        from src.qc.per_study import build_study_qc_report

        custom = "<!DOCTYPE html><html><body>CUSTOM {{ kpis.n_series }}</body></html>"
        out = tmp_path / "custom.html"
        result = build_study_qc_report(SAMPLES_DIR, out, template_str=custom)

        content = out.read_text()
        assert "CUSTOM" in content
        assert str(result["n_series"]) in content


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


class TestDatasetReport:
    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        from src.qc.dataset import build_dataset_qc_report

        out = tmp_path / "dataset_qc.html"
        result = build_dataset_qc_report(REPO_ROOT, DATASET_INFO, None, out)

        assert "studies" in result
        assert "series" in result
        assert "dicom_files" in result
        assert "vendors" in result
        assert result["studies"] > 0

    def test_output_html_non_empty(self, tmp_path: Path) -> None:
        from src.qc.dataset import build_dataset_qc_report

        out = tmp_path / "dataset_qc.html"
        build_dataset_qc_report(REPO_ROOT, DATASET_INFO, None, out)

        assert out.exists()
        assert out.stat().st_size > 1024

    def test_contains_vendor_mix(self, tmp_path: Path) -> None:
        from src.qc.dataset import build_dataset_qc_report

        out = tmp_path / "dataset_qc.html"
        build_dataset_qc_report(REPO_ROOT, DATASET_INFO, None, out)

        content = out.read_text(encoding="utf-8")
        assert "Vendor Mix" in content

    def test_contains_embedded_chart(self, tmp_path: Path) -> None:
        from src.qc.dataset import build_dataset_qc_report

        out = tmp_path / "dataset_qc.html"
        build_dataset_qc_report(REPO_ROOT, DATASET_INFO, None, out)

        content = out.read_text(encoding="utf-8")
        # Charts are embedded as base64 PNGs
        assert "data:image/png;base64," in content

    def test_output_under_5mb(self, tmp_path: Path) -> None:
        from src.qc.dataset import build_dataset_qc_report

        out = tmp_path / "dataset_qc.html"
        build_dataset_qc_report(REPO_ROOT, DATASET_INFO, None, out)

        size_mb = out.stat().st_size / (1024 * 1024)
        assert size_mb < 5.0, f"Report is {size_mb:.2f} MB, exceeds 5 MB limit"

    def test_contains_sequence_coverage(self, tmp_path: Path) -> None:
        from src.qc.dataset import build_dataset_qc_report

        out = tmp_path / "dataset_qc.html"
        build_dataset_qc_report(REPO_ROOT, DATASET_INFO, None, out)

        content = out.read_text(encoding="utf-8")
        assert "Sequence Coverage" in content

    def test_contains_grade_distribution(self, tmp_path: Path) -> None:
        from src.qc.dataset import build_dataset_qc_report

        out = tmp_path / "dataset_qc.html"
        build_dataset_qc_report(REPO_ROOT, DATASET_INFO, None, out)

        content = out.read_text(encoding="utf-8")
        assert "Grade Distribution" in content

    def test_contains_conformance_section(self, tmp_path: Path) -> None:
        from src.qc.dataset import build_dataset_qc_report

        out = tmp_path / "dataset_qc.html"
        build_dataset_qc_report(REPO_ROOT, DATASET_INFO, None, out)

        content = out.read_text(encoding="utf-8")
        assert "Conformance Pass Rate" in content


# ---------------------------------------------------------------------------
# badge
# ---------------------------------------------------------------------------


class TestQualityBadge:
    @pytest.mark.parametrize("grade", ["A", "B", "C", "D", "F"])
    def test_produces_valid_xml(self, grade: str) -> None:
        from src.qc.badge import build_quality_badge

        svg = build_quality_badge(grade)
        # Must parse as XML without exceptions
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")

    @pytest.mark.parametrize("grade", ["A", "B", "C", "D", "F"])
    def test_contains_grade_text(self, grade: str) -> None:
        from src.qc.badge import build_quality_badge

        svg = build_quality_badge(grade)
        assert grade in svg

    def test_lowercase_grade_normalised(self) -> None:
        from src.qc.badge import build_quality_badge

        svg = build_quality_badge("a")
        assert "A" in svg

    def test_unknown_grade_produces_svg(self) -> None:
        from src.qc.badge import build_quality_badge

        svg = build_quality_badge("Z")
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")


# ---------------------------------------------------------------------------
# __init__ exports
# ---------------------------------------------------------------------------


class TestQcInit:
    def test_exports_all_three_symbols(self) -> None:
        from src import qc

        assert callable(qc.build_study_qc_report)
        assert callable(qc.build_dataset_qc_report)
        assert callable(qc.build_quality_badge)

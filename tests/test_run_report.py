"""Pure pytest tests for src.run_report and src.dry_run.

These tests exercise the public API and JSON round-trip without any Modal
imports, external services, or heavy dependencies.  All 28 original
assertions are preserved; unittest.TestCase has been replaced with plain
pytest functions and the tmp_output_dir fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.run_report import RunReport
from src.dry_run import plan, _safe_name, _is_derivative


# ---------------------------------------------------------------------------
# RunReport -- initialisation
# ---------------------------------------------------------------------------

def test_run_id_generated_when_omitted(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir)
    assert len(r.run_id) > 0
    assert "T" in r.run_id  # timestamp format includes T


def test_explicit_run_id_preserved(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="test-run-001")
    assert r.run_id == "test-run-001"


def test_started_at_is_iso(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir)
    assert "T" in r.started_at


# ---------------------------------------------------------------------------
# RunReport -- record_stage
# ---------------------------------------------------------------------------

def test_record_appends_stage(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="r1")
    r.record_stage("quality", ok=10, skipped=2, failed=1, elapsed_s=5.5)
    assert len(r.stages) == 1
    stage = r.stages[0]
    assert stage["stage"] == "quality"
    assert stage["ok"] == 10
    assert stage["skipped"] == 2
    assert stage["failed"] == 1
    assert stage["elapsed_s"] == pytest.approx(5.5)


def test_errors_capped_at_20(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="r1")
    errors = [f"err{i}" for i in range(50)]
    r.record_stage("quality", ok=0, skipped=0, failed=50, elapsed_s=1.0, errors=errors)
    assert len(r.stages[0]["errors"]) == 20


def test_extras_stored(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="r1")
    r.record_stage("quality", ok=5, skipped=0, failed=0, elapsed_s=1.0, total_slices=200)
    assert r.stages[0]["total_slices"] == 200


def test_multiple_stages(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="r1")
    r.record_stage("quality", ok=5, skipped=0, failed=0, elapsed_s=1.0)
    r.record_stage("annotation", ok=3, skipped=0, failed=0, elapsed_s=2.0)
    assert len(r.stages) == 2
    assert r.stages[1]["stage"] == "annotation"


# ---------------------------------------------------------------------------
# RunReport -- cost tracking
# ---------------------------------------------------------------------------

def test_set_estimated_cost(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="cost-test")
    r.set_estimated_cost(modal_dollars=1.5, openrouter_dollars=0.25)
    assert r.cost_estimated["modal_dollars"] == 1.5
    assert r.cost_estimated["openrouter_dollars"] == 0.25


def test_set_actual_cost(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="cost-test")
    r.set_actual_cost(modal_dollars=1.2, openrouter_dollars=0.18)
    assert r.cost_actual["modal_dollars"] == 1.2
    assert r.cost_actual["openrouter_dollars"] == 0.18


def test_default_cost_none(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="cost-test")
    assert r.cost_estimated["modal_dollars"] is None
    assert r.cost_actual["openrouter_dollars"] is None


# ---------------------------------------------------------------------------
# RunReport -- write / JSON round-trip
# ---------------------------------------------------------------------------

def test_write_creates_json(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="write-test")
    r.record_stage("quality", ok=7, skipped=1, failed=0, elapsed_s=3.2)
    r.set_estimated_cost(modal_dollars=0.05)
    path = r.write()
    assert path.exists()
    assert path.name == "write-test.json"
    assert path.parent.name == "runs"


def test_write_json_round_trip(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="round-trip")
    r.record_stage("quality", ok=3, skipped=0, failed=0, elapsed_s=1.0)
    r.set_actual_cost(modal_dollars=0.01, openrouter_dollars=0.02)
    path = r.write()

    data = json.loads(path.read_text())
    assert data["run_id"] == "round-trip"
    assert "started_at" in data
    assert "finished_at" in data
    assert data["finished_at"] is not None
    assert len(data["stages"]) == 1
    assert data["stages"][0]["ok"] == 3
    assert data["cost_actual"]["modal_dollars"] == 0.01


def test_write_sets_finished_at(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="ts-test")
    assert r.finished_at is None
    r.write()
    assert r.finished_at is not None


def test_schema_fields_present(tmp_output_dir: Path) -> None:
    """All schema fields must be present in the written JSON."""
    r = RunReport(tmp_output_dir, run_id="schema-test")
    path = r.write()
    data = json.loads(path.read_text())
    required = {
        "run_id", "started_at", "finished_at", "stages",
        "cost_estimated", "cost_actual", "modal_app", "git_sha",
    }
    assert required == required & data.keys()


# ---------------------------------------------------------------------------
# RunReport -- format_text
# ---------------------------------------------------------------------------

def test_format_text_nonempty(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="fmt-test")
    r.record_stage("quality", ok=10, skipped=2, failed=1, elapsed_s=4.5)
    text = r.format_text()
    assert "fmt-test" in text
    assert isinstance(text, str)
    assert len(text) > 0


def test_format_text_contains_stage(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="fmt2")
    r.record_stage("annotation", ok=5, skipped=0, failed=2, elapsed_s=9.1)
    text = r.format_text()
    assert "annotation" in text
    assert "fail=" in text


def test_format_text_shows_cost(tmp_output_dir: Path) -> None:
    r = RunReport(tmp_output_dir, run_id="fmt3")
    r.set_estimated_cost(modal_dollars=1.23, openrouter_dollars=0.45)
    text = r.format_text()
    assert "$1.2300" in text
    assert "$0.4500" in text


# ---------------------------------------------------------------------------
# dry_run.plan
# ---------------------------------------------------------------------------

def _make_detail(path: Path, has_quality: bool = False) -> None:
    data: dict = {"series_description": "T1"}
    if has_quality:
        data["advanced_quality"] = {"snr": 30.0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_plan_empty_dir(tmp_path: Path) -> None:
    # Use a truly empty directory (tmp_output_dir pre-creates _pipeline_state/).
    result = plan(tmp_path)
    assert result["n_studies"] == 0
    assert result["n_series_total"] == 0
    assert result["n_series_quality_pending"] == 0


def test_plan_nonexistent_dir() -> None:
    result = plan(Path("/nonexistent/path/xyz"))
    assert result["n_studies"] == 0
    assert result["est_modal_dollars"] == 0.0


def test_pending_quality_counts(tmp_path: Path) -> None:
    output_dir = tmp_path
    study = output_dir / "study_001"
    _make_detail(study / "s1" / "s1_detail.json", has_quality=False)
    _make_detail(study / "s2" / "s2_detail.json", has_quality=False)
    _make_detail(study / "s3" / "s3_detail.json", has_quality=True)

    result = plan(output_dir)
    assert result["n_series_total"] == 3
    assert result["n_series_quality_pending"] == 2
    assert result["n_studies"] == 1


def test_cost_nonzero_when_pending(tmp_path: Path) -> None:
    output_dir = tmp_path
    study = output_dir / "study_001"
    _make_detail(study / "s1" / "s1_detail.json", has_quality=False)

    result = plan(output_dir)
    assert result["est_modal_dollars"] > 0.0


def test_pack_pending_counts(tmp_path: Path) -> None:
    output_dir = tmp_path
    study = output_dir / "study_001"
    (study / "slices").mkdir(parents=True)
    (study / "slices" / "dummy.png").write_bytes(b"")

    result = plan(output_dir)
    assert result["n_studies_pack_pending"] == 1


def test_pack_done_when_tar_exists(tmp_path: Path) -> None:
    output_dir = tmp_path
    study = output_dir / "study_001"
    (study / "slices").mkdir(parents=True)
    tar_path = study / "study_001.slices.tar"
    tar_path.write_bytes(b"fake tar data")

    result = plan(output_dir)
    assert result["n_studies_pack_pending"] == 0


def test_plan_return_keys(tmp_path: Path) -> None:
    result = plan(tmp_path)
    expected_keys = {
        "n_studies", "n_series_total",
        "n_series_quality_pending", "n_series_annotation_pending",
        "n_studies_pack_pending",
        "est_modal_dollars", "est_openrouter_dollars", "est_wall_minutes",
    }
    assert expected_keys == expected_keys & result.keys()


# ---------------------------------------------------------------------------
# Helper functions: _safe_name / _is_derivative
# ---------------------------------------------------------------------------

def test_safe_name_replaces_spaces() -> None:
    assert _safe_name("Series 10 \u2014 T1") == "Series_10___T1"


def test_is_derivative_adc() -> None:
    assert _is_derivative("Series 5 \u2014 ADC map")


def test_is_derivative_mip() -> None:
    assert _is_derivative("Series 9 \u2014 MIP")


def test_not_derivative_t1() -> None:
    assert not _is_derivative("Series 3 \u2014 T1 MPRAGE")

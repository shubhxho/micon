"""Smoke tests for src.run_report and src.dry_run.

Run with:
    python -m unittest tests.test_run_report

These tests exercise the public API and JSON round-trip without any Modal
imports, external services, or heavy dependencies.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.run_report import RunReport
from src.dry_run import plan, _safe_name, _is_derivative


class TestRunReportInit(unittest.TestCase):
    def test_run_id_generated_when_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(Path(tmp))
            self.assertTrue(len(r.run_id) > 0)
            self.assertIn("T", r.run_id)  # timestamp format includes T

    def test_explicit_run_id_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(Path(tmp), run_id="test-run-001")
            self.assertEqual(r.run_id, "test-run-001")

    def test_started_at_is_iso(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(Path(tmp))
            # Should be parseable as ISO datetime (contains T and + or Z)
            self.assertTrue("T" in r.started_at)


class TestRunReportRecordStage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.r = RunReport(Path(self._tmp.name), run_id="r1")

    def tearDown(self):
        self._tmp.cleanup()

    def test_record_appends_stage(self):
        self.r.record_stage("quality", ok=10, skipped=2, failed=1, elapsed_s=5.5)
        self.assertEqual(len(self.r.stages), 1)
        stage = self.r.stages[0]
        self.assertEqual(stage["stage"], "quality")
        self.assertEqual(stage["ok"], 10)
        self.assertEqual(stage["skipped"], 2)
        self.assertEqual(stage["failed"], 1)
        self.assertAlmostEqual(stage["elapsed_s"], 5.5)

    def test_errors_capped_at_20(self):
        errors = [f"err{i}" for i in range(50)]
        self.r.record_stage("quality", ok=0, skipped=0, failed=50,
                             elapsed_s=1.0, errors=errors)
        self.assertEqual(len(self.r.stages[0]["errors"]), 20)

    def test_extras_stored(self):
        self.r.record_stage("quality", ok=5, skipped=0, failed=0,
                             elapsed_s=1.0, total_slices=200)
        self.assertEqual(self.r.stages[0]["total_slices"], 200)

    def test_multiple_stages(self):
        self.r.record_stage("quality", ok=5, skipped=0, failed=0, elapsed_s=1.0)
        self.r.record_stage("annotation", ok=3, skipped=0, failed=0, elapsed_s=2.0)
        self.assertEqual(len(self.r.stages), 2)
        self.assertEqual(self.r.stages[1]["stage"], "annotation")


class TestRunReportCost(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.r = RunReport(Path(self._tmp.name), run_id="cost-test")

    def tearDown(self):
        self._tmp.cleanup()

    def test_set_estimated_cost(self):
        self.r.set_estimated_cost(modal_dollars=1.5, openrouter_dollars=0.25)
        self.assertEqual(self.r.cost_estimated["modal_dollars"], 1.5)
        self.assertEqual(self.r.cost_estimated["openrouter_dollars"], 0.25)

    def test_set_actual_cost(self):
        self.r.set_actual_cost(modal_dollars=1.2, openrouter_dollars=0.18)
        self.assertEqual(self.r.cost_actual["modal_dollars"], 1.2)
        self.assertEqual(self.r.cost_actual["openrouter_dollars"], 0.18)

    def test_default_cost_none(self):
        self.assertIsNone(self.r.cost_estimated["modal_dollars"])
        self.assertIsNone(self.r.cost_actual["openrouter_dollars"])


class TestRunReportWrite(unittest.TestCase):
    def test_write_creates_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(Path(tmp), run_id="write-test")
            r.record_stage("quality", ok=7, skipped=1, failed=0, elapsed_s=3.2)
            r.set_estimated_cost(modal_dollars=0.05)
            path = r.write()
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "write-test.json")
            self.assertTrue(path.parent.name == "runs")

    def test_write_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(Path(tmp), run_id="round-trip")
            r.record_stage("quality", ok=3, skipped=0, failed=0, elapsed_s=1.0)
            r.set_actual_cost(modal_dollars=0.01, openrouter_dollars=0.02)
            path = r.write()

            data = json.loads(path.read_text())
            self.assertEqual(data["run_id"], "round-trip")
            self.assertIn("started_at", data)
            self.assertIn("finished_at", data)
            self.assertIsNotNone(data["finished_at"])
            self.assertEqual(len(data["stages"]), 1)
            self.assertEqual(data["stages"][0]["ok"], 3)
            self.assertEqual(data["cost_actual"]["modal_dollars"], 0.01)

    def test_write_sets_finished_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(Path(tmp), run_id="ts-test")
            self.assertIsNone(r.finished_at)
            r.write()
            self.assertIsNotNone(r.finished_at)

    def test_schema_fields_present(self):
        """All schema fields must be present in the written JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(Path(tmp), run_id="schema-test")
            path = r.write()
            data = json.loads(path.read_text())
            required = {
                "run_id", "started_at", "finished_at", "stages",
                "cost_estimated", "cost_actual", "modal_app", "git_sha",
            }
            self.assertEqual(required, required & data.keys())


class TestRunReportFormatText(unittest.TestCase):
    def test_format_text_nonempty(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(Path(tmp), run_id="fmt-test")
            r.record_stage("quality", ok=10, skipped=2, failed=1, elapsed_s=4.5)
            text = r.format_text()
            self.assertIn("fmt-test", text)
            self.assertIsInstance(text, str)
            self.assertTrue(len(text) > 0)

    def test_format_text_contains_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(Path(tmp), run_id="fmt2")
            r.record_stage("annotation", ok=5, skipped=0, failed=2, elapsed_s=9.1)
            text = r.format_text()
            self.assertIn("annotation", text)
            self.assertIn("fail=", text)

    def test_format_text_shows_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = RunReport(Path(tmp), run_id="fmt3")
            r.set_estimated_cost(modal_dollars=1.23, openrouter_dollars=0.45)
            text = r.format_text()
            self.assertIn("$1.2300", text)
            self.assertIn("$0.4500", text)


class TestDryRunPlan(unittest.TestCase):
    def _make_detail(self, path: Path, has_quality: bool = False) -> None:
        data: dict = {"series_description": "T1"}
        if has_quality:
            data["advanced_quality"] = {"snr": 30.0}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = plan(Path(tmp))
            self.assertEqual(result["n_studies"], 0)
            self.assertEqual(result["n_series_total"], 0)
            self.assertEqual(result["n_series_quality_pending"], 0)

    def test_nonexistent_dir(self):
        result = plan(Path("/nonexistent/path/xyz"))
        self.assertEqual(result["n_studies"], 0)
        self.assertEqual(result["est_modal_dollars"], 0.0)

    def test_pending_quality_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            study = output_dir / "study_001"
            # 2 series without quality, 1 with
            self._make_detail(study / "s1" / "s1_detail.json", has_quality=False)
            self._make_detail(study / "s2" / "s2_detail.json", has_quality=False)
            self._make_detail(study / "s3" / "s3_detail.json", has_quality=True)

            result = plan(output_dir)
            self.assertEqual(result["n_series_total"], 3)
            self.assertEqual(result["n_series_quality_pending"], 2)
            self.assertEqual(result["n_studies"], 1)

    def test_cost_nonzero_when_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            study = output_dir / "study_001"
            self._make_detail(study / "s1" / "s1_detail.json", has_quality=False)

            result = plan(output_dir)
            self.assertGreater(result["est_modal_dollars"], 0.0)

    def test_pack_pending_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            study = output_dir / "study_001"
            # Create a slices dir (pack-pending)
            (study / "slices").mkdir(parents=True)
            (study / "slices" / "dummy.png").write_bytes(b"")

            result = plan(output_dir)
            self.assertEqual(result["n_studies_pack_pending"], 1)

    def test_pack_done_when_tar_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            study = output_dir / "study_001"
            (study / "slices").mkdir(parents=True)
            tar_path = study / "study_001.slices.tar"
            tar_path.write_bytes(b"fake tar data")

            result = plan(output_dir)
            self.assertEqual(result["n_studies_pack_pending"], 0)

    def test_return_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = plan(Path(tmp))
            expected_keys = {
                "n_studies", "n_series_total",
                "n_series_quality_pending", "n_series_annotation_pending",
                "n_studies_pack_pending",
                "est_modal_dollars", "est_openrouter_dollars", "est_wall_minutes",
            }
            self.assertEqual(expected_keys, expected_keys & result.keys())


class TestHelpers(unittest.TestCase):
    def test_safe_name_replaces_spaces(self):
        self.assertEqual(_safe_name("Series 10 — T1"), "Series_10___T1")

    def test_is_derivative_adc(self):
        self.assertTrue(_is_derivative("Series 5 — ADC map"))

    def test_is_derivative_mip(self):
        self.assertTrue(_is_derivative("Series 9 — MIP"))

    def test_not_derivative_t1(self):
        self.assertFalse(_is_derivative("Series 3 — T1 MPRAGE"))


if __name__ == "__main__":
    unittest.main()

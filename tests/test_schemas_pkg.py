"""
Tests for the src/schemas package.

Covers:
- Backwards compatibility: existing imports from src.schemas still work
- export_jsonschema produces one file per model
- Each generated schema is valid JSON Schema Draft 2020-12
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

# ---------------------------------------------------------------------------
# 1. Backwards compatibility: existing imports must work
# ---------------------------------------------------------------------------


def test_backwards_compat_import_series_detail() -> None:
    """from src.schemas import SeriesDetail must keep working."""
    from src.schemas import SeriesDetail  # noqa: PLC0415

    assert SeriesDetail is not None
    detail = SeriesDetail.model_validate(
        {
            "series": {
                "uid": "1.2.3",
                "number": 1,
                "description": "Test",
            }
        }
    )
    assert detail.series.uid == "1.2.3"


def test_backwards_compat_import_all_models() -> None:
    """All 12 public models must be importable from src.schemas."""
    from src.schemas import (  # noqa: PLC0415
        AnnotationConsensus,
        AnnotationRecord,
        MLTrainingScore,
        ManifestRow,
        PipelineVersion,
        QualityAnalysis,
        SequenceClassification,
        SequenceParams,
        SeriesDetail,
        SeriesIdentity,
        StudyManifestRow,
        VolumeStats,
    )

    models = [
        AnnotationConsensus,
        AnnotationRecord,
        MLTrainingScore,
        ManifestRow,
        PipelineVersion,
        QualityAnalysis,
        SequenceClassification,
        SequenceParams,
        SeriesDetail,
        SeriesIdentity,
        StudyManifestRow,
        VolumeStats,
    ]
    assert len(models) == 12
    for m in models:
        assert m is not None


# ---------------------------------------------------------------------------
# 2. export_jsonschema produces one file per model + dataset.schema.json
# ---------------------------------------------------------------------------

_EXPECTED_MODEL_FILES = [
    "AnnotationConsensus.schema.json",
    "AnnotationRecord.schema.json",
    "MLTrainingScore.schema.json",
    "ManifestRow.schema.json",
    "PipelineVersion.schema.json",
    "QualityAnalysis.schema.json",
    "SequenceClassification.schema.json",
    "SequenceParams.schema.json",
    "SeriesDetail.schema.json",
    "SeriesIdentity.schema.json",
    "StudyManifestRow.schema.json",
    "VolumeStats.schema.json",
]


def test_export_produces_file_per_model(tmp_path: Path) -> None:
    """export_all writes one .schema.json per model."""
    from src.schemas.export_jsonschema import export_all  # noqa: PLC0415

    written = export_all(tmp_path)
    filenames = {p.name for p in written}

    for expected in _EXPECTED_MODEL_FILES:
        assert expected in filenames, f"Missing schema file: {expected}"


def test_export_produces_dataset_schema(tmp_path: Path) -> None:
    """export_all writes dataset.schema.json referencing all models."""
    from src.schemas.export_jsonschema import export_all  # noqa: PLC0415

    export_all(tmp_path)
    dataset_file = tmp_path / "dataset.schema.json"
    assert dataset_file.exists()

    schema = json.loads(dataset_file.read_text(encoding="utf-8"))
    assert "$schema" in schema
    assert "properties" in schema
    # All 12 model names referenced
    for name in [f.replace(".schema.json", "") for f in _EXPECTED_MODEL_FILES]:
        assert name in schema["properties"], f"{name} missing from dataset.schema.json"


def test_export_is_deterministic(tmp_path: Path) -> None:
    """Running export_all twice produces identical output."""
    from src.schemas.export_jsonschema import export_all  # noqa: PLC0415

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    paths1 = export_all(out1)
    paths2 = export_all(out2)

    for p1, p2 in zip(paths1, paths2):
        assert p1.read_text() == p2.read_text(), f"Non-deterministic output: {p1.name}"


# ---------------------------------------------------------------------------
# 3. Each generated schema is valid JSON Schema Draft 2020-12
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", _EXPECTED_MODEL_FILES + ["dataset.schema.json"])
def test_schema_is_valid_draft_2020_12(tmp_path: Path, filename: str) -> None:
    """Every generated schema must pass Draft202012Validator.check_schema."""
    from src.schemas.export_jsonschema import export_all  # noqa: PLC0415

    export_all(tmp_path)
    schema_path = tmp_path / filename
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    # check_schema raises jsonschema.SchemaError on invalid meta-schema
    Draft202012Validator.check_schema(schema)
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"

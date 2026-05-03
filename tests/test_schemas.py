"""
pytest suite for src/schemas.py and src/schema_utils.py.

Validates all sample detail.json files, study_full_series_stats.json series
entries, round-trip behaviour, and a Hypothesis property test for SequenceParams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.schemas import (
    AnnotationConsensus,
    AnnotationRecord,
    ManifestRow,
    MLTrainingScore,
    PipelineVersion,
    QualityAnalysis,
    SequenceClassification,
    SequenceParams,
    SeriesDetail,
    SeriesIdentity,
    StudyManifestRow,
    VolumeStats,
)
from src.schema_utils import load_detail, validate_directory

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent
_SAMPLES = _REPO / "Speall_MRI_Samples"
_SERIES_DIR = _SAMPLES / "series"
_STUDY_STATS = _SAMPLES / "study_full_series_stats.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_series_json() -> list[Path]:
    return sorted(_SERIES_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# 1. Every detail.json under Speall_MRI_Samples/series/ must parse cleanly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _all_series_json(), ids=lambda p: p.name)
def test_detail_json_validates(path: Path) -> None:
    """Each series detail.json must parse cleanly as SeriesDetail."""
    detail = load_detail(path)
    assert detail.series is not None


# ---------------------------------------------------------------------------
# 2. study_full_series_stats.json series array must parse cleanly
# ---------------------------------------------------------------------------


def test_study_stats_series_validate() -> None:
    """All series entries in study_full_series_stats.json validate as SeriesDetail."""
    data = json.loads(_STUDY_STATS.read_text(encoding="utf-8"))
    series_list = data.get("series", [])
    assert len(series_list) > 0, "Expected non-empty series list"

    for entry in series_list:
        detail = SeriesDetail.model_validate(entry)
        assert detail.series is not None


# ---------------------------------------------------------------------------
# 3. Round-trip test: dict -> model_validate -> model_dump() -> equal-ish
# ---------------------------------------------------------------------------


def _intersect_keys(dumped: dict, original: dict) -> dict:
    """Return ``dumped`` restricted to keys that exist in ``original``.

    ``model_dump()`` with extra="allow" produces a superset of the input
    (None-defaulted schema fields are included even when absent in the source).
    This helper strips those phantom keys so the comparison is "equal-ish".
    """
    result = {}
    for k, v_orig in original.items():
        if k not in dumped:
            continue
        v_dump = dumped[k]
        if isinstance(v_orig, dict) and isinstance(v_dump, dict):
            result[k] = _intersect_keys(v_dump, v_orig)
        else:
            result[k] = v_dump
    return result


def test_round_trip_detail_json() -> None:
    """Round-trip: every field from the original survives model_validate -> model_dump.

    model_dump produces a *superset* (None-defaulted schema fields added) so we
    intersect to the original's key set.  The only value transform is the
    intentional ``pixel_spacing: "" -> None`` coercion.
    """
    path = _SERIES_DIR / "s0005_Ax_DWI.json"
    original = json.loads(path.read_text(encoding="utf-8"))

    detail = SeriesDetail.model_validate(original)
    dumped = detail.model_dump(mode="python")

    # Build expected: original with the one known transformation
    expected = json.loads(json.dumps(original))
    expected["sequence_params"]["pixel_spacing"] = None

    # Restrict dumped to keys present in original so None-defaulted schema
    # fields don't cause a spurious mismatch.
    dumped_intersected = _intersect_keys(dumped, expected)

    assert dumped_intersected == expected, (
        "Round-trip lost or mutated a field from the original. "
        "A nested model may be missing extra='allow'."
    )


def test_round_trip_flat_study_entry() -> None:
    """Round-trip a flat study-stats series entry (no nested 'series' key)."""
    data = json.loads(_STUDY_STATS.read_text(encoding="utf-8"))
    entry = data["series"][0]

    detail = SeriesDetail.model_validate(entry)
    dumped = detail.model_dump(mode="python")

    # Identity was hoisted correctly
    assert detail.series.uid is not None
    assert detail.series.number == entry["series_number"]


# ---------------------------------------------------------------------------
# 4. Hypothesis property test: SequenceParams with any subset of fields
# ---------------------------------------------------------------------------


_float_or_none = st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False))
_int_or_none = st.one_of(st.none(), st.integers(min_value=0, max_value=65535))
_pixel_spacing = st.one_of(
    st.none(),
    st.just(""),
    st.lists(st.floats(min_value=0.01, max_value=10.0, allow_nan=False), min_size=2, max_size=2),
)


@given(
    tr=_float_or_none,
    te=_float_or_none,
    ti=_float_or_none,
    fa=_float_or_none,
    b_value=_float_or_none,
    slice_thickness=_float_or_none,
    spacing_between_slices=_float_or_none,
    rows=_int_or_none,
    columns=_int_or_none,
    field_strength=_float_or_none,
    pixel_spacing=_pixel_spacing,
)
@settings(max_examples=200)
def test_sequence_params_any_subset(
    tr: Optional[float],
    te: Optional[float],
    ti: Optional[float],
    fa: Optional[float],
    b_value: Optional[float],
    slice_thickness: Optional[float],
    spacing_between_slices: Optional[float],
    rows: Optional[int],
    columns: Optional[int],
    field_strength: Optional[float],
    pixel_spacing: object,
) -> None:
    """Any subset of SequenceParams fields (including none) should validate."""
    data = {
        k: v
        for k, v in {
            "tr": tr,
            "te": te,
            "ti": ti,
            "fa": fa,
            "b_value": b_value,
            "slice_thickness": slice_thickness,
            "spacing_between_slices": spacing_between_slices,
            "rows": rows,
            "columns": columns,
            "field_strength": field_strength,
            "pixel_spacing": pixel_spacing,
        }.items()
        if v is not None or k == "pixel_spacing"
    }
    params = SequenceParams.model_validate(data)
    # pixel_spacing="" should be coerced to None
    if data.get("pixel_spacing") == "":
        assert params.pixel_spacing is None


# ---------------------------------------------------------------------------
# 5. Smoke tests for other models
# ---------------------------------------------------------------------------


def test_sequence_params_empty() -> None:
    params = SequenceParams.model_validate({})
    assert params.tr is None
    assert params.pixel_spacing is None


def test_sequence_params_pixel_spacing_empty_string() -> None:
    params = SequenceParams.model_validate({"pixel_spacing": ""})
    assert params.pixel_spacing is None


def test_sequence_classification_validates() -> None:
    sc = SequenceClassification.model_validate(
        {
            "sequence_type": "DWI",
            "confidence": "high",
            "reasoning": ["Name matches DWI"],
            "parameters": {"TR_ms": 6000.0},
        }
    )
    assert sc.confidence == "high"
    assert sc.parameters == {"TR_ms": 6000.0}


def test_volume_stats_extra_fields() -> None:
    """Extra fields in VolumeStats must survive (extra='allow')."""
    vs = VolumeStats.model_validate(
        {
            "volume_shape": [50, 256, 256],
            "spacing_mm": [1.0, 1.0, 2.0],
            "fov_mm": [100.0, 256.0, 256.0],
            "quality_grade": "D",
            "quality_score": 44.5,
            "volume_skewness": 5.7,  # extra field
            "otsu_threshold": 1736.3,  # extra field
        }
    )
    assert vs.quality_grade == "D"
    assert vs.model_extra is not None and "volume_skewness" in vs.model_extra


def test_ml_training_score_optional() -> None:
    ml = MLTrainingScore.model_validate({})
    assert ml.score is None


def test_pipeline_version() -> None:
    pv = PipelineVersion.model_validate({"timestamp": "2026-04-30", "git_sha": "abc123"})
    assert pv.git_sha == "abc123"


def test_manifest_row_all_optional() -> None:
    row = ManifestRow.model_validate({})
    assert row.study_id is None


def test_study_manifest_row_all_optional() -> None:
    row = StudyManifestRow.model_validate({})
    assert row.n_series is None


def test_annotation_consensus() -> None:
    ac = AnnotationConsensus.model_validate(
        {
            "sequence_type": "T2 FLAIR",
            "sequence_agreement": 0.95,
            "anatomical_structures": ["brain"],
            "pathology": {"white_matter_lesions": False},
            "quality_grade": "B",
            "notable": [],
            "disagreements": [],
        }
    )
    assert ac.sequence_agreement == pytest.approx(0.95)


def test_annotation_record() -> None:
    ar = AnnotationRecord.model_validate(
        {
            "series_label": "s0007_Ax_T2_FLAIR",
            "models_called": 3,
            "models_succeeded": 3,
            "per_model": {},
            "consensus": {
                "sequence_type": "FLAIR",
                "sequence_agreement": 1.0,
                "anatomical_structures": [],
                "pathology": {},
                "quality_grade": "A",
                "notable": [],
                "disagreements": [],
            },
        }
    )
    assert ar.models_called == 3
    assert ar.consensus is not None
    assert ar.consensus.sequence_type == "FLAIR"


# ---------------------------------------------------------------------------
# 6. validate_directory helper (uses a temp directory with a bad file)
# ---------------------------------------------------------------------------


def test_validate_directory_no_detail_files(tmp_path: Path) -> None:
    """Empty dir should return no errors."""
    errors = validate_directory(tmp_path)
    assert errors == []


def test_validate_directory_valid_file(tmp_path: Path) -> None:
    """A valid detail.json should produce no errors."""
    src = _SERIES_DIR / "s0005_Ax_DWI.json"
    dest = tmp_path / "s0005_detail.json"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    errors = validate_directory(tmp_path)
    assert errors == []


def test_validate_directory_invalid_file(tmp_path: Path) -> None:
    """A detail.json missing a required ``series`` key should be reported."""
    bad = tmp_path / "bad_detail.json"
    # Write something that has no ``series`` key at all and no identity fields
    bad.write_text(json.dumps({"broken": True}), encoding="utf-8")
    errors = validate_directory(tmp_path)
    assert len(errors) == 1
    assert bad == errors[0][0]

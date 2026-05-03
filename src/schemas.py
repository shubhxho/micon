"""
Pydantic v2 schemas for the Speall MRI pipeline.

Backwards-compatible: existing dict-based code continues to work unchanged.
New code can opt in via e.g. ``SeriesDetail.from_json_file(path)``.

Key design decisions
--------------------
* ``extra="allow"`` on all models that touch real data so unknown fields from
  real detail JSONs never break validation.
* ``pixel_spacing`` in the wild is always ``""`` (empty string) -- a validator
  coerces that to ``None`` so typing stays ``list[float] | None``.
* ``SeriesIdentity`` accepts both naming conventions via ``AliasChoices``:
  - detail.json: ``uid / number / description``
  - study_full_series_stats.json: ``series_uid / series_number / series_description``
* ``SeriesDetail`` has a ``model_validator(mode='before')`` that detects the
  flat shape (no ``series`` key) and hoists the identity fields up so both
  formats validate to the same model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class _JsonFileMixin(BaseModel):
    """Shared ``from_json_file`` classmethod."""

    @classmethod
    def from_json_file(cls, path: Path) -> Self:
        """Load and validate a JSON file, returning a validated model instance."""
        path = Path(path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# SequenceParams
# ---------------------------------------------------------------------------


class SequenceParams(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    tr: float | None = Field(default=None)
    te: float | None = Field(default=None)
    ti: float | None = Field(default=None)
    fa: float | None = Field(default=None)
    b_value: float | None = Field(default=None)
    slice_thickness: float | None = Field(default=None)
    spacing_between_slices: float | None = Field(default=None)
    rows: int | None = Field(default=None)
    columns: int | None = Field(default=None)
    field_strength: float | None = Field(default=None)
    pixel_spacing: list[float] | None = Field(default=None)

    @field_validator("pixel_spacing", mode="before")
    @classmethod
    def _coerce_pixel_spacing(cls, v: Any) -> Any:
        """Coerce empty string (real-world value) to None."""
        if v == "" or v is None:
            return None
        return v


# ---------------------------------------------------------------------------
# SequenceClassification
# ---------------------------------------------------------------------------


class SequenceClassification(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    sequence_type: str
    confidence: Literal["high", "medium", "low", "unknown"]
    reasoning: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] | None = Field(default=None)


# ---------------------------------------------------------------------------
# VolumeStats
# ---------------------------------------------------------------------------


class VolumeStats(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    volume_shape: list[int] = Field(default_factory=list)
    spacing_mm: list[float] = Field(default_factory=list)
    fov_mm: list[float] = Field(default_factory=list)
    volume_min: float | None = Field(default=None)
    volume_max: float | None = Field(default=None)
    volume_mean: float | None = Field(default=None)
    volume_std: float | None = Field(default=None)
    volume_snr_estimate: float | None = Field(default=None)
    volume_cnr: float | None = Field(default=None)
    volume_entropy: float | None = Field(default=None)
    quality_grade: Literal["A", "B", "C", "D", "F"] | None = Field(default=None)
    quality_score: float | None = Field(default=None)


# ---------------------------------------------------------------------------
# QualityAnalysis
# ---------------------------------------------------------------------------


class QualityAnalysis(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    quality_grade: dict[str, Any] = Field(default_factory=dict)
    motion_analysis: dict[str, Any] = Field(default_factory=dict)
    sharpness_analysis: dict[str, Any] = Field(default_factory=dict)
    anomaly_detection: dict[str, Any] = Field(default_factory=dict)
    symmetry_analysis: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# MLTrainingScore
# ---------------------------------------------------------------------------


class MLTrainingScore(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    score: float | None = Field(default=None)
    grade: str | None = Field(default=None)
    commercial_tier: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# PipelineVersion
# ---------------------------------------------------------------------------


class PipelineVersion(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    timestamp: str | None = Field(default=None)
    git_sha: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# SeriesIdentity
# ---------------------------------------------------------------------------


class SeriesIdentity(_JsonFileMixin):
    """Accepts both ``uid`` (detail.json) and ``series_uid`` (study stats) naming."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    uid: str | None = Field(
        default=None,
        validation_alias=AliasChoices("uid", "series_uid"),
    )
    number: int | None = Field(
        default=None,
        validation_alias=AliasChoices("number", "series_number"),
    )
    description: str | None = Field(
        default=None,
        validation_alias=AliasChoices("description", "series_description"),
    )
    modality: str | None = Field(default=None)
    sop_class: str | None = Field(default=None)
    sop_class_uid: str | None = Field(default=None)
    file_count: int | None = Field(default=None)
    has_pixels: bool | None = Field(default=None)
    source_subdir: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# SeriesDetail  (top-level; handles both detail.json and study stats shapes)
# ---------------------------------------------------------------------------

# Identity field names that appear at the top level in the flat shape
_IDENTITY_KEYS = frozenset(
    {
        "series_uid",
        "series_number",
        "series_description",
        "modality",
        "sop_class",
        "sop_class_uid",
        "file_count",
        "has_pixels",
        "source_subdir",
        # also the non-prefixed variants in case they ever appear at top level
        "uid",
        "number",
        "description",
    }
)


class SeriesDetail(_JsonFileMixin):
    """Top-level container for a single series.

    Validates both:
    - ``*_detail.json`` files: nested ``series`` dict
    - ``study_full_series_stats.json`` entries: flat shape with ``series_uid`` etc.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    series: SeriesIdentity
    sequence_classification: SequenceClassification | None = Field(default=None)
    sequence_params: SequenceParams | None = Field(default=None)
    volume_stats: VolumeStats | None = Field(default=None)
    quality_analysis: QualityAnalysis | None = Field(default=None)
    advanced_quality: dict[str, Any] | None = Field(default=None)
    ml_training_score: MLTrainingScore | None = Field(default=None)
    study_id: str | None = Field(default=None)
    pipeline_version: PipelineVersion | None = Field(default=None)
    files: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    conformance_issues: list[dict[str, Any]] = Field(default_factory=list)
    conformance_summary: dict[str, Any] | None = Field(default=None)
    per_file_stats: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalise_flat_shape(cls, data: Any) -> Any:
        """Hoist flat identity fields into a ``series`` sub-dict when missing.

        study_full_series_stats.json entries have ``series_uid``, ``series_number``
        etc. at the top level rather than inside a nested ``series`` key.
        """
        if not isinstance(data, dict):
            return data
        if "series" in data:
            return data

        # Detect flat shape: has identity-style keys but no ``series`` sub-dict
        identity = {k: data[k] for k in _IDENTITY_KEYS if k in data}
        if not identity:
            return data

        remainder = {k: v for k, v in data.items() if k not in _IDENTITY_KEYS}
        remainder["series"] = identity
        return remainder


# ---------------------------------------------------------------------------
# ManifestRow  (mirrors _SERIES_SCHEMA in src/build_manifest.py)
# ---------------------------------------------------------------------------


class ManifestRow(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    study_id: str | None = None
    series_uid: str | None = None
    series_number: int | None = None
    series_description: str | None = None
    sequence_type: str | None = None
    sequence_confidence: str | None = None
    modality: str | None = None
    file_count: int | None = None
    tr_ms: float | None = None
    te_ms: float | None = None
    ti_ms: float | None = None
    fa_deg: float | None = None
    b_value: float | None = None
    field_strength_T: float | None = None
    plane: str | None = None
    volume_shape: list[int] | None = None
    spacing_mm: list[float] | None = None
    fov_mm: list[float] | None = None
    volume_snr: float | None = None
    volume_cnr: float | None = None
    volume_entropy: float | None = None
    quality_grade: str | None = None
    quality_score: float | None = None
    ml_score: float | None = None
    ml_grade: str | None = None
    commercial_tier: str | None = None
    detail_path: str | None = None
    montage_path: str | None = None
    has_tar_shard: bool | None = None


# ---------------------------------------------------------------------------
# StudyManifestRow  (mirrors _STUDY_SCHEMA in src/build_manifest.py)
# ---------------------------------------------------------------------------


class StudyManifestRow(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    study_id: str | None = None
    n_series: int | None = None
    sequences_present: list[str] | None = None
    total_files: int | None = None
    dominant_grade: str | None = None
    mean_ml_score: float | None = None
    has_dwi: bool | None = None
    has_flair: bool | None = None
    has_swan: bool | None = None
    has_tof: bool | None = None
    has_t1: bool | None = None
    has_t2: bool | None = None
    total_size_mb: float | None = None
    tar_shard_path: str | None = None


# ---------------------------------------------------------------------------
# AnnotationConsensus / AnnotationRecord
# ---------------------------------------------------------------------------


class AnnotationConsensus(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    sequence_type: str | None = None
    sequence_agreement: float | None = None
    anatomical_structures: list[str] = Field(default_factory=list)
    pathology: dict[str, Any] = Field(default_factory=dict)
    quality_grade: str | None = None
    notable: list[str] = Field(default_factory=list)
    disagreements: list[dict[str, Any]] = Field(default_factory=list)


class AnnotationRecord(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    series_label: str | None = None
    models_called: int | None = None
    models_succeeded: int | None = None
    per_model: dict[str, Any] = Field(default_factory=dict)
    consensus: AnnotationConsensus | None = None
    tissue_analysis: dict[str, Any] | None = None

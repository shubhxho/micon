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

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self


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

    tr: Optional[float] = Field(default=None)
    te: Optional[float] = Field(default=None)
    ti: Optional[float] = Field(default=None)
    fa: Optional[float] = Field(default=None)
    b_value: Optional[float] = Field(default=None)
    slice_thickness: Optional[float] = Field(default=None)
    spacing_between_slices: Optional[float] = Field(default=None)
    rows: Optional[int] = Field(default=None)
    columns: Optional[int] = Field(default=None)
    field_strength: Optional[float] = Field(default=None)
    pixel_spacing: Optional[list[float]] = Field(default=None)

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
    parameters: Optional[dict[str, Any]] = Field(default=None)


# ---------------------------------------------------------------------------
# VolumeStats
# ---------------------------------------------------------------------------


class VolumeStats(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    volume_shape: list[int] = Field(default_factory=list)
    spacing_mm: list[float] = Field(default_factory=list)
    fov_mm: list[float] = Field(default_factory=list)
    volume_min: Optional[float] = Field(default=None)
    volume_max: Optional[float] = Field(default=None)
    volume_mean: Optional[float] = Field(default=None)
    volume_std: Optional[float] = Field(default=None)
    volume_snr_estimate: Optional[float] = Field(default=None)
    volume_cnr: Optional[float] = Field(default=None)
    volume_entropy: Optional[float] = Field(default=None)
    quality_grade: Optional[Literal["A", "B", "C", "D", "F"]] = Field(default=None)
    quality_score: Optional[float] = Field(default=None)


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

    score: Optional[float] = Field(default=None)
    grade: Optional[str] = Field(default=None)
    commercial_tier: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# PipelineVersion
# ---------------------------------------------------------------------------


class PipelineVersion(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    timestamp: Optional[str] = Field(default=None)
    git_sha: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# SeriesIdentity
# ---------------------------------------------------------------------------


class SeriesIdentity(_JsonFileMixin):
    """Accepts both ``uid`` (detail.json) and ``series_uid`` (study stats) naming."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    uid: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("uid", "series_uid"),
    )
    number: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("number", "series_number"),
    )
    description: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("description", "series_description"),
    )
    modality: Optional[str] = Field(default=None)
    sop_class: Optional[str] = Field(default=None)
    sop_class_uid: Optional[str] = Field(default=None)
    file_count: Optional[int] = Field(default=None)
    has_pixels: Optional[bool] = Field(default=None)
    source_subdir: Optional[str] = Field(default=None)


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
    sequence_classification: Optional[SequenceClassification] = Field(default=None)
    sequence_params: Optional[SequenceParams] = Field(default=None)
    volume_stats: Optional[VolumeStats] = Field(default=None)
    quality_analysis: Optional[QualityAnalysis] = Field(default=None)
    advanced_quality: Optional[dict[str, Any]] = Field(default=None)
    ml_training_score: Optional[MLTrainingScore] = Field(default=None)
    study_id: Optional[str] = Field(default=None)
    pipeline_version: Optional[PipelineVersion] = Field(default=None)
    files: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    conformance_issues: list[dict[str, Any]] = Field(default_factory=list)
    conformance_summary: Optional[dict[str, Any]] = Field(default=None)
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

    study_id: Optional[str] = None
    series_uid: Optional[str] = None
    series_number: Optional[int] = None
    series_description: Optional[str] = None
    sequence_type: Optional[str] = None
    sequence_confidence: Optional[str] = None
    modality: Optional[str] = None
    file_count: Optional[int] = None
    tr_ms: Optional[float] = None
    te_ms: Optional[float] = None
    ti_ms: Optional[float] = None
    fa_deg: Optional[float] = None
    b_value: Optional[float] = None
    field_strength_T: Optional[float] = None
    plane: Optional[str] = None
    volume_shape: Optional[list[int]] = None
    spacing_mm: Optional[list[float]] = None
    fov_mm: Optional[list[float]] = None
    volume_snr: Optional[float] = None
    volume_cnr: Optional[float] = None
    volume_entropy: Optional[float] = None
    quality_grade: Optional[str] = None
    quality_score: Optional[float] = None
    ml_score: Optional[float] = None
    ml_grade: Optional[str] = None
    commercial_tier: Optional[str] = None
    detail_path: Optional[str] = None
    montage_path: Optional[str] = None
    has_tar_shard: Optional[bool] = None


# ---------------------------------------------------------------------------
# StudyManifestRow  (mirrors _STUDY_SCHEMA in src/build_manifest.py)
# ---------------------------------------------------------------------------


class StudyManifestRow(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    study_id: Optional[str] = None
    n_series: Optional[int] = None
    sequences_present: Optional[list[str]] = None
    total_files: Optional[int] = None
    dominant_grade: Optional[str] = None
    mean_ml_score: Optional[float] = None
    has_dwi: Optional[bool] = None
    has_flair: Optional[bool] = None
    has_swan: Optional[bool] = None
    has_tof: Optional[bool] = None
    has_t1: Optional[bool] = None
    has_t2: Optional[bool] = None
    total_size_mb: Optional[float] = None
    tar_shard_path: Optional[str] = None


# ---------------------------------------------------------------------------
# AnnotationConsensus / AnnotationRecord
# ---------------------------------------------------------------------------


class AnnotationConsensus(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    sequence_type: Optional[str] = None
    sequence_agreement: Optional[float] = None
    anatomical_structures: list[str] = Field(default_factory=list)
    pathology: dict[str, Any] = Field(default_factory=dict)
    quality_grade: Optional[str] = None
    notable: list[str] = Field(default_factory=list)
    disagreements: list[dict[str, Any]] = Field(default_factory=list)


class AnnotationRecord(_JsonFileMixin):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    series_label: Optional[str] = None
    models_called: Optional[int] = None
    models_succeeded: Optional[int] = None
    per_model: dict[str, Any] = Field(default_factory=dict)
    consensus: Optional[AnnotationConsensus] = None
    tissue_analysis: Optional[dict[str, Any]] = None

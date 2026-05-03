"""
Pydantic v2 schemas for the Speall MRI pipeline.

Backwards-compatible: existing dict-based code continues to work unchanged.
New code can opt in via e.g. ``SeriesDetail.from_json_file(path)``.

All model classes are defined in ``src.schemas.models``; this __init__ re-exports
them so that ``from src.schemas import SeriesDetail`` continues to work.
"""

from __future__ import annotations

from src.schemas.models import (
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
    _JsonFileMixin,
)

__all__ = [
    "AnnotationConsensus",
    "AnnotationRecord",
    "MLTrainingScore",
    "ManifestRow",
    "PipelineVersion",
    "QualityAnalysis",
    "SequenceClassification",
    "SequenceParams",
    "SeriesDetail",
    "SeriesIdentity",
    "StudyManifestRow",
    "VolumeStats",
    "_JsonFileMixin",
]

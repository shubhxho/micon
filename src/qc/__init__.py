"""QC report generators for the Speall MRI dataset.

Public API
----------
build_study_qc_report   -- per-study self-contained HTML QC report
build_dataset_qc_report -- dataset-wide statistical HTML report
build_quality_badge     -- SVG grade badge for embedding in dataset cards
"""

from .badge import build_quality_badge
from .dataset import build_dataset_qc_report
from .per_study import build_study_qc_report

__all__ = [
    "build_quality_badge",
    "build_dataset_qc_report",
    "build_study_qc_report",
]

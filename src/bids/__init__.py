"""BIDS (Brain Imaging Data Structure) converter package for the Speall MRI pipeline.

Converts Speall pipeline study output into BIDS 1.10.0-compliant layout.
"""

from src.bids.converter import convert_dataset, convert_study
from src.bids.mappings import (
    SEQUENCE_TO_BIDS,
    bids_filename,
    infer_acquisition_label,
)

__all__ = [
    "SEQUENCE_TO_BIDS",
    "bids_filename",
    "convert_dataset",
    "convert_study",
    "infer_acquisition_label",
]

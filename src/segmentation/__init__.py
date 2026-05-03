"""Segmentation pre-computation package for the Speall MRI pipeline.

Adds free segmentation labels to BIDS datasets using open-weights medical
imaging models:

  - synthstrip:  brain extraction mask (skull stripping)
  - synthseg:    brain anatomy parcellation (95+ regions)
  - monai_brain_lesion: white-matter/lesion mask via MONAI Bundle zoo

All model loading is lazy and cached; importing this package never triggers
downloads.  Output lands in BIDS derivatives layout:

  derivatives/speall-<model>/sub-XXX/ses-YY/anat/
    sub-XXX_ses-YY_space-orig_desc-<task>_dseg.nii.gz

Usage::

    from src.segmentation.pipeline import segment_dataset
    results = segment_dataset(bids_root=Path("/data/bids"),
                              models=["synthstrip", "synthseg"])
"""

from src.segmentation.pipeline import segment_dataset, segment_one_series, segment_study

__all__ = ["segment_dataset", "segment_one_series", "segment_study"]

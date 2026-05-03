"""OME-Zarr cloud-native export package.

Converts NIfTI volumes (produced by the BIDS converter) to OME-Zarr 0.5
multiscale groups suitable for streaming from S3, GCS, or local disk.

Lazy imports: zarr and ome_zarr are NOT imported at module level so this
package can be imported even when those libraries are not installed.

Quick start::

    from src.zarr_export.converter import nifti_to_omezarr
    stats = nifti_to_omezarr(
        nifti_path=Path("sub-001_ses-01_T1w.nii.gz"),
        zarr_path=Path("sub-001_ses-01_T1w.ome.zarr"),
    )
"""

from __future__ import annotations

__all__ = ["nifti_to_omezarr", "study_to_omezarr", "build_pyramid", "series_volume_to_omezarr"]

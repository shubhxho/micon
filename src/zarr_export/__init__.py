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

# NOTE: Symbols are intentionally not re-exported here -- callers import them
# directly from submodules (zarr_export.converter, .series_writer, etc.) so
# importing this package does not trigger the heavy `zarr` / `ome_zarr` deps.

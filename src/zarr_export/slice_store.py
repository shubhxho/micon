"""Zarr-backed slice store — writes a 3-D volume's slices as a single chunked
Zarr array, replacing the per-slice PNG + tar shard pipeline.

Buyers can stream individual slices directly from `slices.zarr` without
having to untar `*.slices.tar`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

# Import the canonical windowing function so Zarr and PNG paths stay in sync.
# slice_export imports slice_store lazily (inside a function body), so there
# is no circular-import risk here.
from src.export.slice_export import _normalize_slice  # noqa: E402

# Target chunk size in bytes (512 KB).
_CHUNK_TARGET_BYTES = 512 * 1024


def _compute_chunk_shape(
    n_slices: int,
    height: int,
    width: int,
    dtype_itemsize: int = 1,
) -> tuple[int, int, int]:
    """Return (n, H, W) chunk shape targeting ~512 KB per chunk."""
    slice_bytes = height * width * dtype_itemsize
    if slice_bytes == 0:
        return (1, height, width)
    n = max(1, math.floor(_CHUNK_TARGET_BYTES / slice_bytes))
    n = min(n, max(n_slices, 1))
    return (n, height, width)


def slices_to_zarr(
    volume: np.ndarray,
    zarr_path: Path,
    *,
    series_label: str,
    dtype: str = "uint8",
) -> dict[str, Any]:
    """Write a 3-D volume as a chunked Zarr array of windowed uint8 slices.

    Each axial slice is normalised with 1st/99th-percentile clipping (the
    same windowing used by the PNG exporter) so the stored byte values match
    the existing PNG pipeline.

    Args:
        volume:       3-D numpy array (n_slices, height, width), any dtype.
        zarr_path:    Destination ``.zarr`` directory (created if absent).
        series_label: Human-readable series name embedded in Zarr attrs.
        dtype:        Output dtype (default "uint8").

    Returns:
        Dict with keys:
        ``path`` (str), ``n_slices`` (int), ``n_chunks`` (int),
        ``total_bytes`` (int), ``compression_ratio`` (float).
    """
    import zarr
    from zarr.codecs import BloscCodec, BytesCodec

    zarr_path = Path(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    if volume.ndim < 3:
        volume = volume[np.newaxis, ...]

    n_slices, height, width = volume.shape[:3]
    itemsize = np.dtype(dtype).itemsize

    # Handle empty volume gracefully.
    if n_slices == 0:
        return {
            "path": str(zarr_path),
            "n_slices": 0,
            "n_chunks": 0,
            "total_bytes": 0,
            "compression_ratio": 0.0,
        }

    chunk_shape = _compute_chunk_shape(n_slices, height, width, itemsize)

    # Build uint8 array with per-slice percentile normalisation.
    windowed = np.stack([_normalize_slice(volume[i]) for i in range(n_slices)])

    # Record volume-level stats for attrs.
    vol_min = float(volume.min())
    vol_max = float(volume.max())
    original_dtype = str(volume.dtype)

    store = zarr.storage.LocalStore(str(zarr_path))
    arr = zarr.create(
        store=store,
        shape=(n_slices, height, width),
        chunks=chunk_shape,
        dtype=dtype,
        codecs=[BytesCodec(), BloscCodec(cname="zstd", clevel=3)],
        overwrite=True,
    )
    arr[:] = windowed

    arr.attrs.update({
        "series_label": series_label,
        "n_slices": n_slices,
        "original_dtype": original_dtype,
        "window": {"min": vol_min, "max": vol_max},
    })

    total_bytes = arr.nbytes_stored()
    uncompressed = n_slices * height * width * itemsize
    compression_ratio = total_bytes / uncompressed if uncompressed > 0 else 0.0

    return {
        "path": str(zarr_path),
        "n_slices": n_slices,
        "n_chunks": arr.nchunks,
        "total_bytes": total_bytes,
        "compression_ratio": compression_ratio,
    }

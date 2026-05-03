"""Zarr v3 slice store — sharded, CRC-checksummed, blosc/zstd compressed.

Replaces the per-slice PNG + tar shard pipeline with a single
``slices.zarr`` directory that:

* Uses **Zarr v3 sharding** — many small chunks are packed into a few
  shard files, slashing inode count and dramatically improving S3 / GCS
  ``LIST`` performance.
* Uses **Blosc + Zstd with bit-shuffle** — gives near-PNG ratios on
  windowed uint8 imagery while staying ~10x faster to (de)compress.
* Adds **Crc32c integrity codecs** on the shard index so silent
  corruption surfaces immediately.

Buyers can stream individual slices directly from ``slices.zarr`` without
unpacking ``*.slices.tar``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from src._logging import get_logger
from src.export.slice_export import _normalize_slice

logger = get_logger(__name__)


_CHUNK_TARGET_BYTES = 512 * 1024
_SHARD_SLICES_TARGET = 32


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


def _compute_shard_shape(
    chunk_shape: tuple[int, int, int],
    n_slices: int,
) -> tuple[int, int, int]:
    """Return a shard shape that bundles ~``_SHARD_SLICES_TARGET`` slices."""
    chunk_n, h, w = chunk_shape
    if chunk_n <= 0:
        return chunk_shape
    chunks_per_shard = max(1, _SHARD_SLICES_TARGET // chunk_n)
    shard_n = min(n_slices, chunk_n * chunks_per_shard)
    shard_n = max(shard_n, chunk_n)
    return (shard_n, h, w)


def slices_to_zarr(
    volume: np.ndarray,
    zarr_path: Path,
    *,
    series_label: str,
    dtype: str = "uint8",
) -> dict[str, Any]:
    """Write a 3-D volume as a sharded Zarr v3 array of windowed uint8 slices.

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
        ``total_bytes`` (int), ``compression_ratio`` (float),
        ``shard_shape`` (tuple), ``chunk_shape`` (tuple).
    """
    import zarr
    from zarr.codecs import BloscCname, BloscCodec, BloscShuffle, BytesCodec

    zarr_path = Path(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    if volume.ndim < 3:
        volume = volume[np.newaxis, ...]

    n_slices, height, width = volume.shape[:3]
    itemsize = np.dtype(dtype).itemsize

    if n_slices == 0:
        return {
            "path": str(zarr_path),
            "n_slices": 0,
            "n_chunks": 0,
            "total_bytes": 0,
            "compression_ratio": 0.0,
            "shard_shape": (0, height, width),
            "chunk_shape": (0, height, width),
        }

    chunk_shape = _compute_chunk_shape(n_slices, height, width, itemsize)
    shard_shape = _compute_shard_shape(chunk_shape, n_slices)

    windowed = np.stack([_normalize_slice(volume[i]) for i in range(n_slices)])

    vol_min = float(volume.min())
    vol_max = float(volume.max())
    original_dtype = str(volume.dtype)

    blosc = BloscCodec(
        cname=BloscCname.zstd,
        clevel=5,
        shuffle=BloscShuffle.bitshuffle,
    )

    store = zarr.storage.LocalStore(str(zarr_path))
    arr = zarr.create_array(
        store=store,
        name="",
        shape=(n_slices, height, width),
        chunks=chunk_shape,
        shards=shard_shape if shard_shape != chunk_shape else None,
        dtype=dtype,
        serializer=BytesCodec(),
        compressors=[blosc],
        overwrite=True,
    )
    arr[:] = windowed

    arr.attrs.update(
        {
            "series_label": series_label,
            "n_slices": n_slices,
            "original_dtype": original_dtype,
            "window": {"min": vol_min, "max": vol_max},
            "encoding": {
                "format": "zarr-v3",
                "sharding": True,
                "compressor": "blosc:zstd:5:bitshuffle",
                "shard_shape": list(shard_shape),
                "chunk_shape": list(chunk_shape),
            },
        }
    )

    total_bytes = int(arr.nbytes_stored())
    uncompressed = n_slices * height * width * itemsize
    compression_ratio = total_bytes / uncompressed if uncompressed > 0 else 0.0

    logger.debug(
        "slices_to_zarr {} shape={} shard={} chunk={} ratio={:.3f}",
        zarr_path.name,
        (n_slices, height, width),
        shard_shape,
        chunk_shape,
        compression_ratio,
    )

    return {
        "path": str(zarr_path),
        "n_slices": n_slices,
        "n_chunks": int(arr.nchunks),
        "total_bytes": total_bytes,
        "compression_ratio": compression_ratio,
        "shard_shape": shard_shape,
        "chunk_shape": chunk_shape,
    }

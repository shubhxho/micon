"""OME-Zarr writer for in-memory MRI volumes from Stage 1 DICOM extraction.

Takes a numpy volume (already in memory during series processing) and writes
a chunked OME-Zarr 0.5 multiscale group alongside the existing per-series
outputs (detail.json, multiplane.png, etc.).

Chunk-shape policy
------------------
Target ~1 MB of raw float32 data per chunk.  For a (Z, Y, X) volume the
full YX plane is kept intact so viewers can render a single axial slice with
one chunk read.  Z-chunk size is derived as::

    z_chunk = max(1, min(Z, 1_048_576 // (Y * X * 4)))

For a 256x256 float32 plane that gives z_chunk = 4 (1 048 576 / 262 144 = 4),
~1 MB per chunk.  Resulting chunk shape is (z_chunk, Y, X) capped by array
shape at each pyramid level via ``_clip_chunk``.

Lazy imports
------------
``zarr`` and ``ome_zarr`` are imported only when ``series_volume_to_omezarr``
is called, matching the pattern used in ``converter.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.zarr_export.multiscale import build_pyramid, coordinate_transformations

logger = logging.getLogger(__name__)

_TARGET_CHUNK_BYTES: int = 1_048_576  # 1 MB


def _compute_chunk_shape(
    shape: tuple[int, int, int],
    itemsize: int = 4,
) -> tuple[int, int, int]:
    """Derive a chunk shape targeting ~1 MB per chunk.

    Keeps Y and X full so a single axial slice costs one read.
    Z dimension is chosen to hit the byte target.

    Args:
        shape: Volume shape in (Z, Y, X) order.
        itemsize: Bytes per element (default 4 for float32).

    Returns:
        Chunk shape (z_chunk, Y, X) — each dimension is at least 1.
    """
    z_dim, y_dim, x_dim = shape
    plane_bytes = y_dim * x_dim * itemsize
    z_chunk = max(1, min(z_dim, _TARGET_CHUNK_BYTES // max(plane_bytes, 1)))
    return (z_chunk, y_dim, x_dim)


def _clip_chunk(
    chunk: tuple[int, int, int],
    shape: tuple[int, ...],
) -> tuple[int, int, int]:
    """Clip chunk so no dimension exceeds the corresponding array shape."""
    return tuple(min(c, s) for c, s in zip(chunk, shape))  # type: ignore[return-value]


def _collect_stats(grp: Any) -> tuple[int, int]:
    """Return (total_chunks, total_bytes) across all arrays in grp."""
    total_chunks = 0
    total_bytes = 0
    for key in grp.keys():
        try:
            arr = grp[key]
            if hasattr(arr, "nchunks"):
                total_chunks += arr.nchunks
            if hasattr(arr, "nbytes"):
                total_bytes += arr.nbytes
        except Exception:
            pass
    return total_chunks, total_bytes


def series_volume_to_omezarr(
    volume: np.ndarray,
    zarr_path: Path,
    *,
    voxel_spacing_mm: tuple[float, float, float],
    series_uid: str,
    sequence_type: str | None,
) -> dict[str, Any]:
    """Write an in-memory MRI volume as an OME-Zarr 0.5 multiscale group.

    The volume is expected in **ZYX** axis order (as produced by
    ``sitk.GetArrayFromImage`` or the pydicom stacker in ``series.py``).
    ``voxel_spacing_mm`` must also be in **(Z, Y, X)** order.

    4-level pyramid is built via :func:`src.zarr_export.multiscale.build_pyramid`.
    Chunk shape targets ~1 MB per chunk (see module docstring).

    Args:
        volume: 3-D float32-compatible array in ZYX order.
        zarr_path: Destination directory for the Zarr group (local path).
        voxel_spacing_mm: Physical voxel size (sz, sy, sx) in mm.
        series_uid: DICOM SeriesInstanceUID — stored in group attributes.
        sequence_type: Human-readable sequence label (e.g. "T1-weighted").
                       ``None`` is allowed; stored as empty string.

    Returns:
        Dict with keys::

            {
                "path": str,
                "n_levels": int,
                "total_bytes": int,
                "chunk_shape": tuple[int, int, int],
            }

    Raises:
        ImportError: If zarr or ome-zarr are not installed.
        ValueError: If volume is not 3-D.
    """
    try:
        import zarr  # lazy import
        from ome_zarr.writer import write_multiscale  # lazy import
    except ImportError as exc:
        raise ImportError("zarr and ome-zarr are required for series_volume_to_omezarr") from exc

    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D (ZYX), got ndim={volume.ndim}")

    n_levels = 4
    vol_f32 = volume.astype(np.float32, copy=False)

    pyramid = build_pyramid(vol_f32, n_levels=n_levels)
    coord_transforms = coordinate_transformations(voxel_spacing_mm, n_levels=n_levels)

    axes = [
        {"name": "z", "type": "space", "unit": "millimeter"},
        {"name": "y", "type": "space", "unit": "millimeter"},
        {"name": "x", "type": "space", "unit": "millimeter"},
    ]

    base_chunk = _compute_chunk_shape(vol_f32.shape)
    per_level_storage = [
        {"chunks": _clip_chunk(base_chunk, arr.shape)}
        for arr in pyramid
    ]

    zarr_path = Path(zarr_path)
    store = zarr.storage.LocalStore(str(zarr_path))
    grp = zarr.open_group(store=store, mode="w")

    write_multiscale(
        pyramid,
        grp,
        axes=axes,
        coordinate_transformations=coord_transforms,
        storage_options=per_level_storage,
    )

    # Embed extra series metadata for downstream consumers
    existing_attrs = dict(grp.attrs)
    existing_attrs["series_uid"] = series_uid
    existing_attrs["sequence_type"] = sequence_type or ""
    grp.attrs.update(existing_attrs)

    _, total_bytes = _collect_stats(grp)

    logger.info(
        "series_volume_to_omezarr: %s  shape=%s  levels=%d  bytes=%d  chunk=%s",
        zarr_path.name,
        vol_f32.shape,
        n_levels,
        total_bytes,
        base_chunk,
    )

    return {
        "path": str(zarr_path),
        "n_levels": n_levels,
        "total_bytes": total_bytes,
        "chunk_shape": base_chunk,
    }

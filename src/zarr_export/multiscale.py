"""Multiscale pyramid builder for OME-Zarr export.

Provides :func:`build_pyramid` which takes a 3-D numpy array in ZYX order
and returns a list of progressively downsampled arrays (full, /2, /4, /8, …).

Lazy imports: ``skimage`` is imported only when this function is called so
the module itself loads without heavy ML deps installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass  # only for type checkers -- no runtime imports


def build_pyramid(volume: np.ndarray, n_levels: int = 4) -> list[np.ndarray]:
    """Build a multiscale pyramid by 2x downsampling per level.

    Args:
        volume: 3-D array in **ZYX** axis order, any dtype.
        n_levels: Total number of levels including full resolution.
                  ``n_levels=4`` yields shapes like (Z,Y,X), (Z/2,Y/2,X/2),
                  (Z/4,Y/4,X/4), (Z/8,Y/8,X/8).

    Returns:
        List of ``n_levels`` arrays from full to most-downsampled.
        Each element is a contiguous float32 ndarray.

    Raises:
        ValueError: If ``n_levels < 1`` or ``volume.ndim != 3``.
    """
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D (ZYX), got ndim={volume.ndim}")

    pyramid: list[np.ndarray] = [volume.astype(np.float32, copy=False)]

    for _ in range(n_levels - 1):
        prev = pyramid[-1]
        next_level = _downsample_2x(prev)
        pyramid.append(next_level)

    return pyramid


def _downsample_2x(arr: np.ndarray) -> np.ndarray:
    """Downsample a 3-D ZYX array by 2x using local mean (anti-aliasing).

    Falls back to sliced sub-sampling when skimage is unavailable.
    """
    try:
        from skimage.transform import downscale_local_mean  # lazy import
        return downscale_local_mean(arr, (2, 2, 2)).astype(np.float32)
    except ImportError:
        # Fallback: simple 2x sub-sampling (no anti-aliasing)
        return arr[::2, ::2, ::2].astype(np.float32)


def coordinate_transformations(
    zooms_zyx: tuple[float, float, float],
    n_levels: int,
) -> list[list[dict]]:
    """Build OME-Zarr ``coordinateTransformations`` for each pyramid level.

    Each level's scale is the voxel size multiplied by 2^level.

    Args:
        zooms_zyx: Physical voxel size in mm, ordered (Z, Y, X).
        n_levels: Number of pyramid levels.

    Returns:
        List of per-level transform lists, e.g.:
        ``[[{"type": "scale", "scale": [sz, sy, sx]}], ...]``
    """
    sz, sy, sx = zooms_zyx
    result: list[list[dict]] = []
    for level in range(n_levels):
        factor = 2.0 ** level
        result.append([{
            "type": "scale",
            "scale": [sz * factor, sy * factor, sx * factor],
        }])
    return result

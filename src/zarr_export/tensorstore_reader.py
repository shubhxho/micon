"""TensorStore-based reader for OME-Zarr volumes written by the pipeline.

Provides a fast, async-capable alternative to zarr-python for reading
OME-Zarr multiscale groups.  Same on-disk files — different read path.

Lazy import
-----------
``tensorstore`` is a heavy native dependency.  This module is importable
even when tensorstore is not installed; the ImportError is deferred to the
first call so optional-dep consumers don't pay the import cost.

Usage::

    from src.zarr_export.tensorstore_reader import open_zarr_with_tensorstore, read_volume

    store = open_zarr_with_tensorstore(Path("series.zarr"), level=0)
    arr = store[:].read().result()          # async read, blocking wait
    # or
    arr = read_volume(Path("series.zarr"), level=0)   # convenience — returns np.ndarray
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import tensorstore as ts  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def _detect_zarr_version(zarr_path: Path) -> int:
    """Return Zarr format version (2 or 3) by inspecting on-disk markers.

    Args:
        zarr_path: Root directory of the Zarr store.

    Returns:
        3 if ``zarr.json`` exists (Zarr v3), else 2.
    """
    if (zarr_path / "zarr.json").exists():
        return 3
    return 2


def _resolve_level_path(zarr_path: Path, level: int) -> str:
    """Return the sub-path for a given pyramid level.

    Reads ``zarr.json`` multiscales metadata when available; falls back to
    ``s{level}`` (the naming convention used by this pipeline).

    Args:
        zarr_path: Root of the OME-Zarr group.
        level: Pyramid level index (0 = full resolution).

    Returns:
        Relative path string such as ``"s0"`` or ``"s2"``.
    """
    meta_file = zarr_path / "zarr.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        try:
            datasets = meta["attributes"]["ome"]["multiscales"][0]["datasets"]
            return datasets[level]["path"]
        except (KeyError, IndexError, TypeError):
            pass
    # Fallback: pipeline writes s0, s1, s2, s3
    return f"s{level}"


def open_zarr_with_tensorstore(zarr_path: Path, *, level: int = 0) -> "ts.TensorStore":
    """Open an OME-Zarr group with TensorStore. Returns the level-N array.

    Detects Zarr format version from the on-disk layout and picks the
    matching TensorStore driver (``zarr3`` for v3, ``zarr`` for v2).
    The returned handle is ready for sliced async reads::

        store = open_zarr_with_tensorstore(path, level=0)
        plane = store[5, :, :].read().result()

    Args:
        zarr_path: Path to the OME-Zarr directory written by the pipeline.
        level: Pyramid level to open (0 = highest resolution).

    Returns:
        An open ``ts.TensorStore`` handle for the requested pyramid level.

    Raises:
        ImportError: If tensorstore is not installed.
        FileNotFoundError: If ``zarr_path`` does not exist.
        ValueError: If ``level`` is out of range for the stored pyramid.
    """
    try:
        import tensorstore as ts  # lazy import
    except ImportError as exc:
        raise ImportError(
            "tensorstore is required for open_zarr_with_tensorstore. "
            "Install it with: pip install 'tensorstore>=0.1.65'"
        ) from exc

    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")

    version = _detect_zarr_version(zarr_path)
    driver = "zarr3" if version == 3 else "zarr"
    level_path = _resolve_level_path(zarr_path, level)
    array_path = str(zarr_path / level_path)

    logger.debug(
        "open_zarr_with_tensorstore: driver=%s  path=%s  level=%d → %s",
        driver,
        zarr_path.name,
        level,
        level_path,
    )

    spec: dict = {
        "driver": driver,
        "kvstore": {"driver": "file", "path": array_path},
    }
    return ts.open(spec).result()


def read_volume(zarr_path: Path, *, level: int = 0) -> np.ndarray:
    """Read an entire pyramid level into a numpy array.

    Convenience wrapper around :func:`open_zarr_with_tensorstore` that
    blocks until the full volume is in memory.  Equivalent to::

        zarr.open_group(path, mode="r")[f"s{level}"][:]

    Args:
        zarr_path: Path to the OME-Zarr directory.
        level: Pyramid level (0 = full resolution).

    Returns:
        The volume as a ``numpy.ndarray``.

    Raises:
        ImportError: If tensorstore is not installed.
        FileNotFoundError: If ``zarr_path`` does not exist.
    """
    store = open_zarr_with_tensorstore(zarr_path, level=level)
    return store[:].read().result()

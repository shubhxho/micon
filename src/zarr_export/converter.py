"""OME-Zarr converter for NIfTI volumes.

Converts NIfTI-1/2 files (produced by the BIDS pipeline) into OME-Zarr 0.5
multiscale groups with proper axes metadata and coordinate transformations.

The output is fsspec-friendly: ``zarr_path`` can be a local path string/Path
or any fsspec URL (``s3://bucket/path.ome.zarr``, ``gcs://...``).

Lazy imports: ``zarr``, ``ome_zarr``, and ``nibabel`` are imported only at
call time so this module loads without those libraries installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src._logging import get_logger
from src.zarr_export.multiscale import build_pyramid, coordinate_transformations

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ZarrPath = Path | str  # local Path or fsspec URL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nifti_to_omezarr(
    nifti_path: Path,
    zarr_path: ZarrPath,
    scales: int = 4,
    chunk_size: tuple[int, int, int] = (16, 256, 256),
) -> dict[str, Any]:
    """Convert a NIfTI file to an OME-Zarr 0.5 multiscale group.

    The volume is read via nibabel, transposed from nibabel's XYZ order to
    ZYX (the OME-Zarr convention for volumetric data), and written as a
    chunked Zarr array with a multiscale pyramid.

    Args:
        nifti_path: Source ``.nii`` or ``.nii.gz`` file.
        zarr_path:  Destination Zarr group path -- a local Path or any
                    fsspec-compatible URL (``s3://...``, ``gcs://...``).
        scales:     Total pyramid levels including full resolution.
                    ``scales=4`` yields arrays at 1x, 1/2x, 1/4x, 1/8x.
        chunk_size: Chunk shape in **ZYX** order.  Chunks are clipped per
                    level so they never exceed the array shape.

    Returns:
        Dict with keys:
        ``ok`` (bool), ``n_levels`` (int), ``total_chunks`` (int),
        ``total_bytes`` (int).

    Raises:
        ImportError: If nibabel or zarr are not installed.
        FileNotFoundError: If ``nifti_path`` does not exist.
    """
    try:
        import nibabel as nib  # lazy import
    except ImportError as exc:
        raise ImportError("nibabel is required for nifti_to_omezarr") from exc

    try:
        import zarr  # lazy import
        from ome_zarr.writer import write_multiscale  # lazy import
    except ImportError as exc:
        raise ImportError("zarr and ome-zarr are required for nifti_to_omezarr") from exc

    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    # ------------------------------------------------------------------
    # Load volume -- nibabel returns XYZ; we need ZYX for OME-Zarr
    # ------------------------------------------------------------------
    img = nib.load(str(nifti_path))
    data_xyz = img.get_fdata(dtype="float32")  # pyright: ignore[reportAttributeAccessIssue]  # (X, Y, Z)
    data_zyx = data_xyz.transpose(2, 1, 0).copy()  # (Z, Y, X)

    # Voxel sizes: nibabel get_zooms() -> (sx, sy, sz) in mm
    zooms_xyz = img.header.get_zooms()[:3]  # pyright: ignore[reportAttributeAccessIssue]
    zooms_zyx = (float(zooms_xyz[2]), float(zooms_xyz[1]), float(zooms_xyz[0]))

    # ------------------------------------------------------------------
    # Build pyramid
    # ------------------------------------------------------------------
    pyramid = build_pyramid(data_zyx, n_levels=scales)
    coord_transforms = coordinate_transformations(zooms_zyx, n_levels=scales)

    # ------------------------------------------------------------------
    # OME-Zarr axes metadata (NGFF 0.5 dict format)
    # ------------------------------------------------------------------
    axes = [
        {"name": "z", "type": "space", "unit": "millimeter"},
        {"name": "y", "type": "space", "unit": "millimeter"},
        {"name": "x", "type": "space", "unit": "millimeter"},
    ]

    # ------------------------------------------------------------------
    # Open Zarr group (works for local paths and fsspec URLs)
    # ------------------------------------------------------------------
    store = _open_store(zarr_path)
    grp = zarr.open_group(store=store, mode="w")

    # ------------------------------------------------------------------
    # Write pyramid -- pass per-level storage_options to clip chunk size
    # so chunks never exceed the array shape at any level.
    # ------------------------------------------------------------------
    per_level_storage = [{"chunks": _chunk_for_shape(chunk_size, arr.shape)} for arr in pyramid]

    write_multiscale(
        pyramid,
        grp,
        axes=axes,
        coordinate_transformations=coord_transforms,
        storage_options=per_level_storage,
    )

    # ------------------------------------------------------------------
    # Collect statistics
    # ------------------------------------------------------------------
    total_chunks, total_bytes = _compute_stats(grp)

    logger.info(
        "nifti_to_omezarr: {} -> {} levels={} chunks={} bytes={}",
        nifti_path.name,
        zarr_path,
        scales,
        total_chunks,
        total_bytes,
    )

    return {
        "ok": True,
        "n_levels": scales,
        "total_chunks": total_chunks,
        "total_bytes": total_bytes,
    }


def study_to_omezarr(
    study_dir: Path,
    out_root: Path,
) -> dict[str, Any]:
    """Convert all NIfTI files under a BIDS-layout study to OME-Zarr.

    Walks ``study_dir`` recursively for ``*.nii.gz`` / ``*.nii`` files.
    Each series is written as a sub-group of a top-level
    ``<study_name>.ome.zarr`` Zarr group.

    Args:
        study_dir: A BIDS subject/session directory (or any directory tree
                   containing NIfTI files).
        out_root:  Output root directory.  A ``<study_name>.ome.zarr/``
                   group is created here.

    Returns:
        Dict with ``ok`` (bool), ``series_converted`` (int),
        ``series_failed`` (int), ``study_zarr`` (str), ``series`` (list).
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    study_name = _sanitize_name(study_dir.name)
    study_zarr = out_root / f"{study_name}.ome.zarr"

    nifti_files = sorted(list(study_dir.rglob("*.nii.gz")) + list(study_dir.rglob("*.nii")))

    if not nifti_files:
        logger.warning("No NIfTI files found under {}", study_dir)

    results: list[dict[str, Any]] = []
    failed = 0

    for nifti_path in nifti_files:
        rel = nifti_path.relative_to(study_dir)
        # Build sub-group path: replace path separators and strip extension(s)
        subgroup_name = _rel_path_to_group_name(rel)
        series_zarr = study_zarr / subgroup_name

        try:
            stats = nifti_to_omezarr(nifti_path, series_zarr)
            stats["series"] = str(rel)
            results.append(stats)
            logger.info("  converted: {}", rel)
        except Exception as exc:
            logger.warning("  FAILED {}: {}", rel, exc)
            results.append({"ok": False, "series": str(rel), "error": str(exc)})
            failed += 1

    return {
        "ok": failed == 0,
        "series_converted": len(nifti_files) - failed,
        "series_failed": failed,
        "study_zarr": str(study_zarr),
        "series": results,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_store(zarr_path: ZarrPath) -> Any:
    """Return a zarr store for ``zarr_path`` (local or fsspec URL)."""
    import zarr  # already guarded by caller

    path_str = str(zarr_path)
    # fsspec URLs (s3://, gcs://, etc.)
    if "://" in path_str:
        import fsspec  # lazy import -- optional

        mapper = fsspec.get_mapper(path_str)
        return zarr.storage.FsspecStore(mapper.fs, path=mapper.root)

    # Local path -- use LocalStore for explicit zarr v3 control
    return zarr.storage.LocalStore(path_str)


def _chunk_for_shape(
    chunk_size: tuple[int, int, int],
    shape: tuple[int, ...],
) -> tuple[int, int, int]:
    """Clip chunk_size so it never exceeds the array shape on any axis."""
    return tuple(min(c, s) for c, s in zip(chunk_size, shape, strict=False))  # type: ignore[return-value]


def _compute_stats(grp: Any) -> tuple[int, int]:
    """Return (total_chunks, total_bytes) across all arrays in the group."""
    try:
        total_chunks = 0
        total_bytes = 0
        for key in grp:
            try:
                arr = grp[key]
                if hasattr(arr, "nchunks"):
                    total_chunks += arr.nchunks
                if hasattr(arr, "nbytes"):
                    total_bytes += arr.nbytes
            except Exception:
                pass
        return total_chunks, total_bytes
    except Exception:
        return 0, 0


def _sanitize_name(name: str) -> str:
    """Strip unsafe characters from a Zarr group name."""
    import re

    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _rel_path_to_group_name(rel: Path) -> str:
    """Convert a relative Path like ``anat/sub-001_T1w.nii.gz`` to a group name."""
    # Strip .nii.gz or .nii
    stem = rel.name
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    # Re-join parent parts + stem
    parts = [*list(rel.parent.parts), stem]
    return "/".join(_sanitize_name(p) for p in parts if p not in (".", ""))

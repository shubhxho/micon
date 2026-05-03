"""Per-slice PNG export — renders every slice of every series as individual images.

For 355K DICOM files across ~3,000 series, this generates:
  - Per-slice axial PNGs (full resolution, normalized)
  - Per-slice windowed views (brain, bone, subdural windows for applicable sequences)
  - Per-series multi-plane montages at 2x resolution

Output structure:
  slices/<series_name>/
    axial_000.png ... axial_NNN.png     # every axial slice
    coronal_000.png ... coronal_NNN.png # mid-range coronal slices
    sagittal_000.png ... sagittal_NNN.png
    brain_window_000.png ...            # brain window (W:80 L:40)
    bone_window_000.png ...             # bone window (W:2800 L:600)
    montage_hires.png                   # 2x resolution montage

At ~300KB per slice, 355K slices = ~100GB of imagery.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

# Radiological windows: (center, width)
WINDOWS = {
    "brain": (40, 80),
    "subdural": (75, 215),
    "bone": (600, 2800),
    "stroke": (32, 8),
    "soft_tissue": (50, 400),
}


def _normalize_slice(slc: np.ndarray, pct_low: float = 1, pct_high: float = 99) -> np.ndarray:
    """Normalize a 2D slice to 0-255 uint8."""
    while slc.ndim > 2:
        slc = slc[0]
    vmin, vmax = np.percentile(slc, [pct_low, pct_high])
    if vmax - vmin < 1e-6:
        return np.zeros(slc.shape, dtype=np.uint8)
    norm = np.clip((slc - vmin) / (vmax - vmin), 0, 1)
    return (norm * 255).astype(np.uint8)


def _apply_window(slc: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply radiological window to a 2D slice."""
    while slc.ndim > 2:
        slc = slc[0]
    low = center - width / 2
    high = center + width / 2
    norm = np.clip((slc - low) / (high - low + 1e-6), 0, 1)
    return (norm * 255).astype(np.uint8)


def _save_png(arr: np.ndarray, path: Path, resize_to: int = 0) -> int:
    """Save a 2D uint8 array as PNG. Returns file size in bytes."""
    img = Image.fromarray(arr, mode="L")
    if resize_to and max(img.size) > resize_to:
        ratio = resize_to / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), "PNG", optimize=True)
    return path.stat().st_size


def export_all_slices(
    vol: np.ndarray,
    series_name: str,
    out_dir: str | Path,
    windows: list[str] | None = None,
    max_size: int = 512,
    n_workers: int = 4,
    backend: Literal["png", "zarr", "both", "lance", "all"] = "png",
) -> dict:
    """Export every slice of a volume as individual PNGs and/or a Zarr store.

    Args:
        vol: 3D numpy array (slices, height, width)
        series_name: name for the output subfolder
        out_dir: base output directory
        windows: list of window names to apply (default: brain + bone)
        max_size: max dimension for output PNGs (0 = original resolution)
        n_workers: parallel PNG encoding threads
        backend: "png" (default) writes per-slice PNGs; "zarr" writes a
            chunked Zarr store at <out_dir>/slices/<series_name>.slices.zarr;
            "both" produces both outputs; "lance" writes a Lance columnar
            dataset at <out_dir>/slices/<series_name>.slices.lance;
            "all" produces PNG + Zarr + Lance.

    Returns: {"total_slices": N, "total_bytes": N, "files": [...]}
    """
    base_dir = Path(out_dir)
    result: dict = {}

    if backend in ("png", "both", "all"):
        result = _export_png(vol, series_name, base_dir, windows, max_size, n_workers)

    if backend in ("zarr", "both", "all"):
        from src.zarr_export.slice_store import slices_to_zarr
        zarr_path = base_dir / "slices" / f"{series_name}.slices.zarr"
        zarr_result = slices_to_zarr(vol, zarr_path, series_label=series_name)
        result["zarr"] = zarr_result

    if backend in ("lance", "all"):
        from src.lance_export.slice_dataset import slices_to_lance
        lance_path = base_dir / "slices" / f"{series_name}.slices.lance"
        lance_result = slices_to_lance(vol, lance_path, series_label=series_name)
        result["lance"] = lance_result

    return result


def _export_png(
    vol: np.ndarray,
    series_name: str,
    out_dir: Path,
    windows: list[str] | None,
    max_size: int,
    n_workers: int,
) -> dict:
    """Write per-slice PNGs (the original behaviour of export_all_slices)."""
    slice_dir = out_dir / "slices" / series_name
    slice_dir.mkdir(parents=True, exist_ok=True)

    if vol.ndim < 3:
        vol = vol[np.newaxis, ...]

    nz, ny, nx = vol.shape[:3]
    if windows is None:
        windows = ["brain", "bone"]

    tasks = []  # (array, path)

    # Axial slices (every slice)
    for i in range(nz):
        slc = vol[i]
        tasks.append((_normalize_slice(slc), slice_dir / f"axial_{i:04d}.png"))

    # Coronal slices (sample 20% evenly)
    n_cor = max(ny // 5, 1)
    cor_indices = np.linspace(ny // 6, ny - ny // 6, n_cor, dtype=int)
    for i in cor_indices:
        slc = vol[:, min(i, ny - 1), :]
        tasks.append((_normalize_slice(slc), slice_dir / f"coronal_{i:04d}.png"))

    # Sagittal slices (sample 20% evenly)
    n_sag = max(nx // 5, 1)
    sag_indices = np.linspace(nx // 6, nx - nx // 6, n_sag, dtype=int)
    for i in sag_indices:
        slc = vol[:, :, min(i, nx - 1)]
        tasks.append((_normalize_slice(slc), slice_dir / f"sagittal_{i:04d}.png"))

    # Windowed views (axial only, every slice)
    for win_name in windows:
        if win_name not in WINDOWS:
            continue
        center, width = WINDOWS[win_name]
        win_dir = slice_dir / win_name
        win_dir.mkdir(parents=True, exist_ok=True)
        for i in range(nz):
            slc = vol[i]
            tasks.append((_apply_window(slc, center, width), win_dir / f"{i:04d}.png"))

    # Write PNGs in parallel
    total_bytes = 0
    files = []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_save_png, arr, path, max_size): path for arr, path in tasks}
        for fut in as_completed(futures):
            sz = fut.result()
            total_bytes += sz
            files.append(str(futures[fut]))

    return {
        "series": series_name,
        "total_slices": len(tasks),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024**2, 1),
    }

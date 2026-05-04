# pyright: reportPossiblyUnboundVariable=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false
"""Apple Metal GPU acceleration via MLX.

All MLX ops go through _mlx_lock because MLX's Metal command buffers
are NOT thread-safe — concurrent submissions crash the GPU.
The lock serializes GPU work but each op is still faster than numpy
on large arrays because Metal's shader throughput is much higher.

Pyright suppressions: every public function guards on ``MLX_AVAILABLE``
before touching ``mx``, but pyright cannot follow the module-level
flag through call sites — so we silence the resulting "possibly unbound"
and Optional-access noise file-wide rather than scattering ignores.
"""

from __future__ import annotations

import threading

import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

_MIN_GPU_ELEMENTS = 16_384
_mlx_lock = threading.Lock()


def _to_mlx(arr: np.ndarray) -> mx.array:
    if arr.dtype == np.float64:
        return mx.array(arr.astype(np.float32))
    return mx.array(arr)


def gpu_available() -> bool:
    return MLX_AVAILABLE


def gpu_percentile(arr: np.ndarray, pcts: list[float]) -> np.ndarray:
    if not MLX_AVAILABLE or arr.size < _MIN_GPU_ELEMENTS:
        return np.percentile(arr, pcts)

    with _mlx_lock:
        flat = _to_mlx(arr.ravel())
        sorted_arr = mx.sort(flat)
        mx.eval(sorted_arr)
        n = sorted_arr.size
        results = []
        for p in pcts:
            idx = min(int(p / 100.0 * (n - 1)), n - 1)
            results.append(float(sorted_arr[idx].item()))
    return np.array(results)


def gpu_normalize(vol: np.ndarray, pct_low: float = 1, pct_high: float = 99) -> np.ndarray:
    if not MLX_AVAILABLE or vol.size < _MIN_GPU_ELEMENTS:
        vmin, vmax = np.percentile(vol, [pct_low, pct_high])
        return np.clip((vol - vmin) / (vmax - vmin + 1e-6), 0, 1).astype(np.float32)

    with _mlx_lock:
        g = _to_mlx(vol)
        flat = mx.reshape(g, (-1,))
        sorted_flat = mx.sort(flat)
        mx.eval(sorted_flat)
        n = sorted_flat.size
        vmin = float(sorted_flat[int(pct_low / 100.0 * (n - 1))].item())
        vmax = float(sorted_flat[int(pct_high / 100.0 * (n - 1))].item())
        result = mx.clip((g - vmin) / (vmax - vmin + 1e-6), 0, 1)
        mx.eval(result)
        out = np.array(result, dtype=np.float32)
    return out


def gpu_stats(vol: np.ndarray) -> dict:
    if not MLX_AVAILABLE or vol.size < _MIN_GPU_ELEMENTS:
        return _numpy_stats(vol)

    with _mlx_lock:
        g = _to_mlx(vol)
        vmean = mx.mean(g)
        vstd = mx.sqrt(mx.mean((g - vmean) ** 2))
        vmin = mx.min(g)
        vmax = mx.max(g)
        nonzero = mx.mean((g != 0).astype(mx.float32))
        mx.eval(vmean, vstd, vmin, vmax, nonzero)

        mean_f = float(vmean.item())
        std_f = float(vstd.item())
        min_f = float(vmin.item())
        max_f = float(vmax.item())

        if std_f > 1e-10:
            d = g - vmean
            d2 = d * d
            m3 = mx.mean(d2 * d)
            m4 = mx.mean(d2 * d2)
            mx.eval(m3, m4)
            skew = float(m3.item()) / std_f**3
            kurt = float(m4.item()) / std_f**4 - 3.0
        else:
            skew = kurt = 0.0

    # Entropy uses numpy histogram (MLX doesn't have histogram)
    flat_np = vol.ravel()
    hist, _ = np.histogram(flat_np, bins=256, range=(min_f, max_f))
    hist = hist[hist > 0]
    p = hist / hist.sum()
    entropy = float(-np.sum(p * np.log2(p)))

    return {
        "mean": mean_f,
        "std": std_f,
        "min": min_f,
        "max": max_f,
        "nonzero_pct": float(nonzero.item()) * 100,
        "dynamic_range": max_f - min_f,
        "skewness": skew,
        "kurtosis": kurt,
        "entropy": entropy,
    }


def gpu_stats_and_percentiles(
    vol: np.ndarray,
    pcts: list[float],
) -> tuple[dict, np.ndarray]:
    """Combined stats + percentiles in a SINGLE lock acquisition + GPU upload.

    Instead of gpu_stats() + gpu_percentile() separately (2 locks, 2 uploads,
    2 sorts), this does everything in one pass. The sort for percentiles is
    the most expensive op — sharing it halves total GPU work for volume_stats.
    """
    if not MLX_AVAILABLE or vol.size < _MIN_GPU_ELEMENTS:
        stats = _numpy_stats(vol)
        pct_vals = np.percentile(vol, pcts)
        return stats, pct_vals

    with _mlx_lock:
        g = _to_mlx(vol)

        # Basic stats
        vmean = mx.mean(g)
        vstd = mx.sqrt(mx.mean((g - vmean) ** 2))
        vmin = mx.min(g)
        vmax = mx.max(g)
        nonzero = mx.mean((g != 0).astype(mx.float32))
        mx.eval(vmean, vstd, vmin, vmax, nonzero)

        mean_f = float(vmean.item())
        std_f = float(vstd.item())
        min_f = float(vmin.item())
        max_f = float(vmax.item())

        # Higher moments
        if std_f > 1e-10:
            d = g - vmean
            d2 = d * d
            m3 = mx.mean(d2 * d)
            m4 = mx.mean(d2 * d2)
            mx.eval(m3, m4)
            skew = float(m3.item()) / std_f**3
            kurt = float(m4.item()) / std_f**4 - 3.0
        else:
            skew = kurt = 0.0

        # Percentiles from sorted array — reuse the GPU upload
        flat = mx.reshape(g, (-1,))
        sorted_arr = mx.sort(flat)
        mx.eval(sorted_arr)
        n = sorted_arr.size
        pct_vals = np.array(
            [float(sorted_arr[min(int(p / 100.0 * (n - 1)), n - 1)].item()) for p in pcts]
        )

    # Entropy on CPU
    flat_np = vol.ravel()
    hist, _ = np.histogram(flat_np, bins=256, range=(min_f, max_f))
    hist = hist[hist > 0]
    p = hist / hist.sum()
    entropy = float(-np.sum(p * np.log2(p)))

    stats = {
        "mean": mean_f,
        "std": std_f,
        "min": min_f,
        "max": max_f,
        "nonzero_pct": float(nonzero.item()) * 100,
        "dynamic_range": max_f - min_f,
        "skewness": skew,
        "kurtosis": kurt,
        "entropy": entropy,
    }
    return stats, pct_vals


def gpu_slice_stats(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not MLX_AVAILABLE or vol.size < _MIN_GPU_ELEMENTS or vol.ndim < 3:
        axes = tuple(range(1, vol.ndim)) if vol.ndim >= 3 else None
        if axes:
            return vol.mean(axis=axes), vol.std(axis=axes)
        return np.array([vol.mean()]), np.array([vol.std()])

    with _mlx_lock:
        g = _to_mlx(vol)
        n = vol.shape[0]
        flat = mx.reshape(g, (n, -1))
        means = mx.mean(flat, axis=1)
        stds = mx.sqrt(mx.mean((flat - mx.expand_dims(means, 1)) ** 2, axis=1))
        mx.eval(means, stds)
        return np.array(means), np.array(stds)


def gpu_histogram(
    arr: np.ndarray, bins: int = 200, range_: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated histogram via sort + searchsorted binning.

    MLX has no native histogram, so we:
      1. Sort the flat array on GPU — O(N log N), Metal parallel sort
      2. Build bin edges on CPU — O(B), trivial
      3. searchsorted on sorted GPU array for bin boundaries — O(B log N)
      4. Diff indices → counts — O(B)

    ~3-5x faster than numpy on large arrays (Metal sort dominance).
    """
    if not MLX_AVAILABLE or arr.size < _MIN_GPU_ELEMENTS:
        return np.histogram(arr.ravel(), bins=bins, range=range_)

    flat = arr.ravel()
    if range_ is None:
        range_ = (float(flat.min()), float(flat.max()))

    edges = np.linspace(range_[0], range_[1], bins + 1, dtype=np.float32)

    with _mlx_lock:
        g = _to_mlx(flat)
        sorted_arr = mx.sort(g)
        mx.eval(sorted_arr)
        # Transfer sorted array back to CPU for searchsorted
        # (MLX lacks searchsorted, but the GPU sort is the expensive part)
        sorted_np = np.array(sorted_arr)

    idx_np = np.searchsorted(sorted_np, edges)
    counts = np.diff(idx_np)
    return counts, edges


def gpu_histogram_and_percentiles(
    arr: np.ndarray,
    bins: int = 200,
    pcts: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Combined GPU histogram + percentiles in a SINGLE lock acquisition + sort.

    Sorts once on GPU, reuses the sorted array for both binning and percentile
    lookup. Avoids 2 separate lock acquisitions + 2 sorts.

    Returns: (counts, edges, percentile_values) or (counts, edges, None).
    """
    if not MLX_AVAILABLE or arr.size < _MIN_GPU_ELEMENTS:
        flat = arr.ravel()
        counts, edges = np.histogram(flat, bins=bins)
        pct_vals = np.percentile(flat, pcts) if pcts else None
        return counts, edges, pct_vals

    flat = arr.ravel()
    rng = (float(flat.min()), float(flat.max()))
    edges = np.linspace(rng[0], rng[1], bins + 1, dtype=np.float32)

    with _mlx_lock:
        g = _to_mlx(flat)
        sorted_arr = mx.sort(g)
        mx.eval(sorted_arr)
        sorted_np = np.array(sorted_arr)

        # Percentiles from the same sorted array — zero extra work
        pct_vals = None
        if pcts:
            n = sorted_np.size
            pct_vals = np.array([sorted_np[min(int(p / 100.0 * (n - 1)), n - 1)] for p in pcts])

    idx_np = np.searchsorted(sorted_np, edges)
    counts = np.diff(idx_np)
    return counts, edges, pct_vals


def gpu_tissue_pct(vol: np.ndarray, threshold: float) -> float:
    """GPU-accelerated tissue percentage — fraction of voxels > threshold."""
    if not MLX_AVAILABLE or vol.size < _MIN_GPU_ELEMENTS:
        return float((vol > threshold).mean() * 100)

    with _mlx_lock:
        g = _to_mlx(vol)
        pct = mx.mean((g > threshold).astype(mx.float32))
        mx.eval(pct)
        return float(pct.item()) * 100


def gpu_pixel_stats(arr: np.ndarray) -> dict | None:
    """GPU-accelerated per-file pixel stats — all expensive ops in one GPU session.

    Combines percentiles, skewness, kurtosis, and basic stats into a single
    transfer to minimize CPU↔GPU overhead. Returns None if GPU unavailable
    (caller falls back to numpy).
    """
    if not MLX_AVAILABLE or arr.size < _MIN_GPU_ELEMENTS:
        return None

    with _mlx_lock:
        g = _to_mlx(arr)

        # Sort once → reuse for all percentiles (O(N log N) on Metal)
        flat = mx.reshape(g, (-1,))
        sorted_arr = mx.sort(flat)
        mx.eval(sorted_arr)
        n = sorted_arr.size

        pct_vals = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pcts = []
        for p in pct_vals:
            idx = min(int(p / 100.0 * (n - 1)), n - 1)
            pcts.append(float(sorted_arr[idx].item()))

        vmean = mx.mean(g)
        vstd = mx.sqrt(mx.mean((g - vmean) ** 2))
        vmin = mx.min(g)
        vmax = mx.max(g)
        nonzero = mx.mean((g != 0).astype(mx.float32))
        mx.eval(vmean, vstd, vmin, vmax, nonzero)

        mean_f = float(vmean.item())
        std_f = float(vstd.item())

        if std_f > 1e-10:
            d = g - vmean
            d2 = d * d
            m3 = mx.mean(d2 * d)
            m4 = mx.mean(d2 * d2)
            mx.eval(m3, m4)
            skew = float(m3.item()) / std_f**3
            kurt = float(m4.item()) / std_f**4 - 3.0
        else:
            skew = kurt = 0.0

    # Entropy on CPU (histogram bins)
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(float(vmin.item()), float(vmax.item())))
    hist = hist[hist > 0]
    p = hist / hist.sum()
    ent = float(-np.sum(p * np.log2(p)))

    return {
        "pixel_min": float(vmin.item()),
        "pixel_max": float(vmax.item()),
        "pixel_mean": mean_f,
        "pixel_std": std_f,
        "pixel_median": pcts[4],
        "pixel_p1": pcts[0],
        "pixel_p5": pcts[1],
        "pixel_p10": pcts[2],
        "pixel_p25": pcts[3],
        "pixel_p50": pcts[4],
        "pixel_p75": pcts[5],
        "pixel_p90": pcts[6],
        "pixel_p95": pcts[7],
        "pixel_p99": pcts[8],
        "pixel_iqr": pcts[5] - pcts[3],
        "nonzero_ratio": float(nonzero.item()),
        "pixel_entropy": ent,
        "pixel_skewness": skew,
        "pixel_kurtosis": kurt,
    }


# ── GPU rendering functions for enhanced VLM analysis ─────────────────────────


def gpu_mip(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximum Intensity Projection along all 3 axes on Metal GPU.

    MIP collapses a 3D volume by taking the max along each axis,
    producing 3 projection images that reveal vascular structures,
    bright lesions, and overall anatomy — ideal for VLM interpretation.

    Returns: (axial_mip, coronal_mip, sagittal_mip) as float32 arrays.
    """
    if not MLX_AVAILABLE or vol.size < _MIN_GPU_ELEMENTS or vol.ndim < 3:
        return (
            vol.max(axis=0).astype(np.float32),
            vol.max(axis=1).astype(np.float32),
            vol.max(axis=2).astype(np.float32),
        )

    with _mlx_lock:
        g = _to_mlx(vol)
        axial = mx.max(g, axis=0)
        coronal = mx.max(g, axis=1)
        sagittal = mx.max(g, axis=2)
        mx.eval(axial, coronal, sagittal)
        return (
            np.array(axial, dtype=np.float32),
            np.array(coronal, dtype=np.float32),
            np.array(sagittal, dtype=np.float32),
        )


def gpu_minip(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum Intensity Projection — highlights CSF spaces, cysts, dark lesions."""
    if not MLX_AVAILABLE or vol.size < _MIN_GPU_ELEMENTS or vol.ndim < 3:
        return (
            vol.min(axis=0).astype(np.float32),
            vol.min(axis=1).astype(np.float32),
            vol.min(axis=2).astype(np.float32),
        )

    with _mlx_lock:
        g = _to_mlx(vol)
        axial = mx.min(g, axis=0)
        coronal = mx.min(g, axis=1)
        sagittal = mx.min(g, axis=2)
        mx.eval(axial, coronal, sagittal)
        return (
            np.array(axial, dtype=np.float32),
            np.array(coronal, dtype=np.float32),
            np.array(sagittal, dtype=np.float32),
        )


def gpu_window(vol: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply radiological windowing on GPU.

    Standard CT/MRI windowing: maps [center-width/2, center+width/2] → [0, 1].
    Useful for brain/bone/subdural windows that reveal different pathology.
    """
    low = center - width / 2
    high = center + width / 2

    if not MLX_AVAILABLE or vol.size < _MIN_GPU_ELEMENTS:
        return np.clip((vol - low) / (high - low + 1e-6), 0, 1).astype(np.float32)

    with _mlx_lock:
        g = _to_mlx(vol)
        result = mx.clip((g - low) / (high - low + 1e-6), 0, 1)
        mx.eval(result)
        return np.array(result, dtype=np.float32)


def gpu_tissue_mask(vol: np.ndarray, low_pct: float = 15, high_pct: float = 99) -> np.ndarray:
    """GPU-accelerated tissue segmentation mask via intensity thresholding.

    Returns a float32 mask [0, 1] where tissue voxels = 1.
    Uses percentile-based thresholding to adapt to different sequences.
    """
    if not MLX_AVAILABLE or vol.size < _MIN_GPU_ELEMENTS:
        low, high = np.percentile(vol, [low_pct, high_pct])
        return ((vol > low) & (vol < high)).astype(np.float32)

    with _mlx_lock:
        g = _to_mlx(vol)
        flat = mx.reshape(g, (-1,))
        sorted_arr = mx.sort(flat)
        mx.eval(sorted_arr)
        n = sorted_arr.size
        low = float(sorted_arr[int(low_pct / 100.0 * (n - 1))].item())
        high = float(sorted_arr[int(high_pct / 100.0 * (n - 1))].item())
        mask = ((g > low) & (g < high)).astype(mx.float32)
        mx.eval(mask)
        return np.array(mask, dtype=np.float32)


def gpu_batch_enhanced(
    vol: np.ndarray,
    low_pct: float = 1,
    high_pct: float = 99,
    tissue_low: float = 15,
    tissue_high: float = 99,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],  # MIP (ax, cor, sag)
    tuple[np.ndarray, np.ndarray, np.ndarray],  # MinIP (ax, cor, sag)
    np.ndarray,  # tissue mask
    np.ndarray,  # normalized volume
]:
    """Batch all enhanced-view GPU ops in a single lock + upload.

    Combines MIP, MinIP, tissue mask, and normalization into one GPU
    session — avoids 4 separate lock acquisitions and 4 separate
    numpy→MLX uploads of the same volume.
    """
    if not MLX_AVAILABLE or vol.size < _MIN_GPU_ELEMENTS or vol.ndim < 3:
        # CPU fallback
        vmin, vmax = np.percentile(vol, [low_pct, high_pct])
        vol_norm = np.clip((vol - vmin) / (vmax - vmin + 1e-6), 0, 1).astype(np.float32)
        tlow, thigh = np.percentile(vol, [tissue_low, tissue_high])
        mask = ((vol > tlow) & (vol < thigh)).astype(np.float32)
        mip = (
            vol.max(0).astype(np.float32),
            vol.max(1).astype(np.float32),
            vol.max(2).astype(np.float32),
        )
        minip = (
            vol.min(0).astype(np.float32),
            vol.min(1).astype(np.float32),
            vol.min(2).astype(np.float32),
        )
        return mip, minip, mask, vol_norm

    with _mlx_lock:
        g = _to_mlx(vol)

        # MIP
        mip_ax = mx.max(g, axis=0)
        mip_cor = mx.max(g, axis=1)
        mip_sag = mx.max(g, axis=2)

        # MinIP
        minip_ax = mx.min(g, axis=0)
        minip_cor = mx.min(g, axis=1)
        minip_sag = mx.min(g, axis=2)

        # Normalize
        flat = mx.reshape(g, (-1,))
        sorted_arr = mx.sort(flat)
        mx.eval(sorted_arr)
        n = sorted_arr.size
        vmin = float(sorted_arr[int(low_pct / 100.0 * (n - 1))].item())
        vmax = float(sorted_arr[int(high_pct / 100.0 * (n - 1))].item())
        norm = mx.clip((g - vmin) / (vmax - vmin + 1e-6), 0, 1)

        # Tissue mask (reuse sorted array)
        tlow = float(sorted_arr[int(tissue_low / 100.0 * (n - 1))].item())
        thigh = float(sorted_arr[int(tissue_high / 100.0 * (n - 1))].item())
        mask = ((g > tlow) & (g < thigh)).astype(mx.float32)

        # Evaluate all at once
        mx.eval(mip_ax, mip_cor, mip_sag, minip_ax, minip_cor, minip_sag, norm, mask)

        return (
            (
                np.array(mip_ax, dtype=np.float32),
                np.array(mip_cor, dtype=np.float32),
                np.array(mip_sag, dtype=np.float32),
            ),
            (
                np.array(minip_ax, dtype=np.float32),
                np.array(minip_cor, dtype=np.float32),
                np.array(minip_sag, dtype=np.float32),
            ),
            np.array(mask, dtype=np.float32),
            np.array(norm, dtype=np.float32),
        )


def _numpy_stats(vol: np.ndarray) -> dict:
    m, s = float(vol.mean()), float(vol.std())
    if s > 1e-10:
        d = vol - m
        d2 = d * d
        skew = float((d2 * d).mean() / s**3)
        kurt = float((d2 * d2).mean() / s**4 - 3.0)
    else:
        skew = kurt = 0.0

    hist, _ = np.histogram(vol.ravel(), bins=256)
    hist = hist[hist > 0]
    p = hist / hist.sum()
    entropy = float(-np.sum(p * np.log2(p)))

    return {
        "mean": m,
        "std": s,
        "min": float(vol.min()),
        "max": float(vol.max()),
        "nonzero_pct": float((vol != 0).mean() * 100),
        "dynamic_range": float(vol.max() - vol.min()),
        "skewness": skew,
        "kurtosis": kurt,
        "entropy": entropy,
    }

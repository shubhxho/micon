"""Advanced quality metrics — numpy + scipy for commercial-grade data valuation.

Adds metrics that dataset buyers specifically look for:
  - Contrast-to-noise ratio (CNR) between tissue classes
  - Noise floor estimation (Rician noise model)
  - Spatial resolution assessment
  - Signal homogeneity / bias field severity
  - Inter-slice consistency (for training data reliability)
  - Foreground/background segmentation quality
  - Histogram-based tissue class separation
  - Edge sharpness via Laplacian variance
  - Frequency domain noise analysis

All functions are pure numpy/scipy — no ML dependencies, fast, deterministic.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np


def compute_cnr(vol: np.ndarray) -> dict:
    """Contrast-to-noise ratio between tissue classes.

    Uses Otsu's method to separate foreground (tissue) from background,
    then computes CNR = |mean_tissue - mean_bg| / std_bg.
    High CNR (>10) means clear tissue boundaries — better for segmentation training.
    """
    flat = vol.ravel()
    nonzero = flat[flat > 0]
    if len(nonzero) < 100:
        return {"cnr": 0, "interpretation": "insufficient data"}

    # Otsu threshold
    hist, edges = np.histogram(nonzero, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = total - w0
    sum_total = (hist * centers).sum()
    sum0 = np.cumsum(hist * centers).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = sum0 / w0
        mu1 = (sum_total - sum0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2

    between = np.nan_to_num(between, 0)
    threshold = float(centers[np.argmax(between)])

    tissue = nonzero[nonzero > threshold]
    background = nonzero[nonzero <= threshold]

    if len(tissue) < 10 or len(background) < 10:
        return {"cnr": 0, "interpretation": "insufficient separation"}

    mean_t = float(tissue.mean())
    mean_b = float(background.mean())
    std_b = float(background.std())

    cnr = abs(mean_t - mean_b) / max(std_b, 1e-6)

    return {
        "cnr": round(cnr, 2),
        "tissue_mean": round(mean_t, 1),
        "background_mean": round(mean_b, 1),
        "background_std": round(std_b, 1),
        "threshold": round(threshold, 1),
        "interpretation": (
            "excellent"
            if cnr > 15
            else "good"
            if cnr > 10
            else "adequate"
            if cnr > 5
            else "poor"
            if cnr > 2
            else "very poor"
        ),
    }


def compute_noise_floor(vol: np.ndarray) -> dict:
    """Estimate noise floor using Rician noise model.

    In MRI, noise follows a Rician distribution. The noise floor is estimated
    from the background (air) regions. Lower noise floor = cleaner data.
    """
    if vol.ndim < 3:
        return {"noise_floor": 0, "noise_model": "insufficient"}

    # Sample corners of the middle slice (should be air/background)
    mid = vol.shape[0] // 2
    slc = vol[mid]
    while slc.ndim > 2:
        slc = slc[0]

    h, _w = slc.shape
    corner_size = max(h // 10, 5)

    corners = np.concatenate(
        [
            slc[:corner_size, :corner_size].ravel(),
            slc[:corner_size, -corner_size:].ravel(),
            slc[-corner_size:, :corner_size].ravel(),
            slc[-corner_size:, -corner_size:].ravel(),
        ]
    )

    # Remove exact zeros (padding)
    nonzero_corners = corners[corners > 0]

    if len(nonzero_corners) < 20:
        # Fallback: use lowest 5% of volume
        flat = vol.ravel()
        nonzero = flat[flat > 0]
        if len(nonzero) < 100:
            return {"noise_floor": 0, "noise_model": "insufficient"}
        p5 = np.percentile(nonzero, 5)
        background = nonzero[nonzero <= p5]
        noise_std = float(background.std())
    else:
        noise_std = float(nonzero_corners.std())

    # Rician correction: sigma = mode / sqrt(2/pi) for Rician noise
    # For magnitude images, the noise floor is ~1.526 * sigma
    rician_sigma = noise_std / 1.526 if noise_std > 0 else 0

    return {
        "noise_floor": round(noise_std, 3),
        "rician_sigma": round(rician_sigma, 3),
        "noise_model": "rician",
        "interpretation": (
            "very low noise"
            if noise_std < 5
            else "low noise"
            if noise_std < 15
            else "moderate noise"
            if noise_std < 30
            else "high noise"
        ),
    }


def compute_bias_field_severity(vol: np.ndarray) -> dict:
    """Estimate B1 inhomogeneity (bias field) severity.

    Bias field causes smooth intensity variation across the image.
    Computed as coefficient of variation of local means in a grid.
    High CoV = severe bias field = needs N4 correction before training.
    """
    if vol.ndim < 3 or vol.shape[0] < 3:
        return {"bias_severity": 0, "interpretation": "N/A"}

    mid = vol.shape[0] // 2
    slc = vol[mid].astype(np.float64)
    while slc.ndim > 2:
        slc = slc[0]

    h, w = slc.shape
    grid_size = 4
    gh, gw = h // grid_size, w // grid_size

    if gh < 5 or gw < 5:
        return {"bias_severity": 0, "interpretation": "N/A (too small)"}

    # Compute mean intensity in each grid cell
    local_means = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = slc[i * gh : (i + 1) * gh, j * gw : (j + 1) * gw]
            nonzero = patch[patch > 0]
            if len(nonzero) > 10:
                local_means.append(float(nonzero.mean()))

    if len(local_means) < 4:
        return {"bias_severity": 0, "interpretation": "insufficient tissue"}

    means = np.array(local_means)
    cov = float(means.std() / max(means.mean(), 1e-6))

    return {
        "bias_severity": round(cov, 4),
        "local_means_std": round(float(means.std()), 1),
        "local_means_avg": round(float(means.mean()), 1),
        "n_patches": len(local_means),
        "interpretation": (
            "minimal (no correction needed)"
            if cov < 0.05
            else "mild (N4 optional)"
            if cov < 0.10
            else "moderate (N4 recommended)"
            if cov < 0.20
            else "severe (N4 required)"
        ),
        "n4_recommended": cov > 0.10,
    }


def compute_edge_sharpness(vol: np.ndarray) -> dict:
    """Laplacian variance — measures focus/sharpness across the volume.

    Higher variance = sharper edges = better spatial resolution.
    Computed on multiple slices and averaged.
    """
    if vol.ndim < 2:
        return {"laplacian_variance": 0}

    if vol.ndim >= 3:
        # Sample 5 evenly spaced slices
        n = vol.shape[0]
        indices = np.linspace(n // 6, n - n // 6, min(5, n), dtype=int)
        slices = [vol[i].astype(np.float64) for i in indices]
    else:
        slices = [vol.astype(np.float64)]

    variances = []
    for slc in slices:
        while slc.ndim > 2:
            slc = slc[0]
        # Laplacian via convolution
        laplacian = (
            np.roll(slc, 1, 0)
            + np.roll(slc, -1, 0)
            + np.roll(slc, 1, 1)
            + np.roll(slc, -1, 1)
            - 4 * slc
        )
        variances.append(float(laplacian.var()))

    avg_var = np.mean(variances)

    return {
        "laplacian_variance": round(float(avg_var), 2),
        "per_slice_variance": [round(v, 2) for v in variances],
        "interpretation": (
            "very sharp"
            if avg_var > 500
            else "sharp"
            if avg_var > 100
            else "moderate"
            if avg_var > 30
            else "soft/blurry"
        ),
    }


def compute_histogram_separation(vol: np.ndarray) -> dict:
    """Tissue class separation via histogram peak analysis.

    Good MRI data shows distinct peaks for CSF, gray matter, white matter.
    Clear separation = easier segmentation = higher ML training value.
    """
    flat = vol.ravel()
    nonzero = flat[flat > 0]
    if len(nonzero) < 100:
        return {"n_peaks": 0, "separation": "insufficient"}

    hist, edges = np.histogram(nonzero, bins=128)
    centers = (edges[:-1] + edges[1:]) / 2

    # Smooth histogram
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(hist.astype(float), kernel, mode="same")

    # Find peaks: local maxima above 5% of max
    threshold = smoothed.max() * 0.05
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if (
            smoothed[i] > smoothed[i - 1]
            and smoothed[i] > smoothed[i + 1]
            and smoothed[i] > threshold
        ):
            peaks.append(
                {"index": i, "intensity": round(float(centers[i]), 1), "count": int(hist[i])}
            )

    # Separation: distance between peaks relative to histogram width
    separation = 0
    if len(peaks) >= 2:
        separations = []
        for i in range(len(peaks) - 1):
            sep = abs(peaks[i + 1]["intensity"] - peaks[i]["intensity"])
            separations.append(sep)
        separation = min(separations) / max(float(centers[-1] - centers[0]), 1e-6)

    return {
        "n_peaks": len(peaks),
        "peaks": peaks[:5],
        "min_peak_separation": round(separation, 4),
        "interpretation": (
            "excellent tissue separation"
            if len(peaks) >= 3 and separation > 0.1
            else "good separation"
            if len(peaks) >= 2 and separation > 0.05
            else "moderate separation"
            if len(peaks) >= 2
            else "poor separation — tissue classes overlap"
        ),
        "segmentation_difficulty": (
            "easy"
            if len(peaks) >= 3 and separation > 0.1
            else "medium"
            if len(peaks) >= 2
            else "hard"
        ),
    }


def compute_inter_slice_consistency(vol: np.ndarray) -> dict:
    """Measure how consistent adjacent slices are.

    Inconsistent slices = motion, misregistration, or acquisition errors.
    Computed as pairwise correlation between adjacent slices.
    Low consistency = unreliable for 3D training = lower data value.
    """
    if vol.ndim < 3 or vol.shape[0] < 3:
        return {"mean_correlation": 1.0, "interpretation": "N/A"}

    n_slices = vol.shape[0]
    # Sample up to 20 adjacent pairs
    step = max(n_slices // 20, 1)
    pairs = list(range(0, n_slices - 1, step))[:20]

    correlations = []
    for i in pairs:
        s1 = vol[i].ravel().astype(np.float64)
        s2 = vol[i + 1].ravel().astype(np.float64)
        if s1.std() > 1e-6 and s2.std() > 1e-6:
            corr = float(np.corrcoef(s1, s2)[0, 1])
            correlations.append(corr)

    if not correlations:
        return {"mean_correlation": 1.0, "interpretation": "N/A"}

    mean_corr = float(np.mean(correlations))
    min_corr = float(np.min(correlations))
    std_corr = float(np.std(correlations))

    return {
        "mean_correlation": round(mean_corr, 4),
        "min_correlation": round(min_corr, 4),
        "std_correlation": round(std_corr, 4),
        "n_pairs_checked": len(correlations),
        "interpretation": (
            "excellent consistency"
            if mean_corr > 0.95
            else "good consistency"
            if mean_corr > 0.85
            else "moderate (check for motion)"
            if mean_corr > 0.70
            else "poor (likely motion/errors)"
        ),
        "suitable_for_3d_training": mean_corr > 0.85,
    }


def full_quality_assessment(vol: np.ndarray, series_desc: str = "") -> dict:
    """Run ALL advanced quality metrics in parallel. Returns combined report.

    This is the commercial-grade quality assessment that dataset buyers
    use to decide pricing tier and inclusion/exclusion.
    """
    with ThreadPoolExecutor(max_workers=6) as pool:
        cnr_fut = pool.submit(compute_cnr, vol)
        noise_fut = pool.submit(compute_noise_floor, vol)
        bias_fut = pool.submit(compute_bias_field_severity, vol)
        edge_fut = pool.submit(compute_edge_sharpness, vol)
        hist_fut = pool.submit(compute_histogram_separation, vol)
        slice_fut = pool.submit(compute_inter_slice_consistency, vol)

    result = {
        "cnr": cnr_fut.result(),
        "noise_floor": noise_fut.result(),
        "bias_field": bias_fut.result(),
        "edge_sharpness": edge_fut.result(),
        "histogram_separation": hist_fut.result(),
        "inter_slice_consistency": slice_fut.result(),
    }

    # Overall ML training value score (0-100)
    scores = []
    cnr_val = result["cnr"].get("cnr", 0)
    scores.append(min(cnr_val / 15, 1.0) * 25)  # CNR: 0-25

    noise = result["noise_floor"].get("noise_floor", 50)
    scores.append(max(0, 1 - noise / 50) * 20)  # Noise: 0-20

    bias = result["bias_field"].get("bias_severity", 0.5)
    scores.append(max(0, 1 - bias / 0.2) * 15)  # Bias: 0-15

    edge = result["edge_sharpness"].get("laplacian_variance", 0)
    scores.append(min(edge / 200, 1.0) * 15)  # Sharpness: 0-15

    n_peaks = result["histogram_separation"].get("n_peaks", 0)
    scores.append(min(n_peaks / 3, 1.0) * 10)  # Histogram: 0-10

    consistency = result["inter_slice_consistency"].get("mean_correlation", 0)
    scores.append(min(consistency, 1.0) * 15)  # Consistency: 0-15

    ml_score = sum(scores)

    result["ml_training_score"] = {
        "score": round(ml_score, 1),
        "grade": (
            "A"
            if ml_score >= 80
            else "B"
            if ml_score >= 65
            else "C"
            if ml_score >= 50
            else "D"
            if ml_score >= 35
            else "F"
        ),
        "breakdown": {
            "cnr": round(scores[0], 1),
            "noise": round(scores[1], 1),
            "bias": round(scores[2], 1),
            "sharpness": round(scores[3], 1),
            "histogram": round(scores[4], 1),
            "consistency": round(scores[5], 1),
        },
        "commercial_tier": (
            "premium"
            if ml_score >= 80
            else "standard"
            if ml_score >= 60
            else "discount"
            if ml_score >= 40
            else "exclude"
        ),
    }

    return result

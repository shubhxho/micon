"""Quality analysis — scoring, anomaly detection, symmetry, sharpness, motion artifacts.

All functions are pure (no state) and safe for threaded/multiprocessed use.
Heavy numpy ops release the GIL so they parallelize well.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ── Quality grade ────────────────────────────────────────────────────────────


def grade_series(vstats: dict, series_desc: str = "") -> dict:
    """Compute a quality grade (A/B/C/D/F) for a single series.

    Scoring rubric (out of 100):
      SNR:         0-25 pts  (based on volume_snr_estimate)
      CNR:         0-10 pts  (contrast-to-noise ratio)
      Uniformity:  0-20 pts  (slice_intensity_uniformity)
      Tissue %:    0-15 pts  (volume_tissue_pct)
      Dynamic range: 0-12 pts
      Entropy:     0-10 pts  (information content)
      Non-zero %:  0-8 pts   (coverage)

    Sequence-aware: SNR thresholds adapt to sequence type (DWI has inherently
    lower SNR than structural sequences, spectroscopy is graded differently).

    Grade thresholds: A>=80, B>=65, C>=50, D>=35, F<35
    """
    snr = vstats.get("volume_snr_estimate", 0)
    cnr = vstats.get("volume_cnr", snr)  # fallback to SNR if no CNR
    uniformity = vstats.get("slice_intensity_uniformity", 0)
    tissue = vstats.get("volume_tissue_pct", 0)
    dyn_range = vstats.get("volume_dynamic_range", 0)
    ent = vstats.get("volume_entropy", 0)
    nonzero = vstats.get("volume_nonzero_pct", 0)

    # Sequence-aware SNR thresholds
    desc_up = series_desc.upper()
    if any(k in desc_up for k in ("DWI", "DIFF", "DTI", "ADC", "TRACE")):
        snr_target = 8.0  # DWI inherently has lower SNR
        cnr_target = 5.0
    elif any(k in desc_up for k in ("MRS", "SPECTRO", "PRESS", "STEAM")):
        snr_target = 5.0  # Spectroscopy SNR is very different
        cnr_target = 3.0
    elif any(k in desc_up for k in ("SWI", "SWAN", "PHASE")):
        snr_target = 10.0  # SWI/phase are magnitude-dependent
        cnr_target = 8.0
    elif any(k in desc_up for k in ("PERFUS", "DSC", "DCE", "ASL", "DYN")):
        snr_target = 10.0  # Dynamic/perfusion series
        cnr_target = 6.0
    else:
        snr_target = 20.0  # Structural sequences (T1, T2, FLAIR)
        cnr_target = 12.0

    # SNR score: 0-25 (sequence-adaptive target)
    snr_score = min(snr / snr_target, 1.0) * 25

    # CNR score: 0-10 (tissue vs background contrast quality)
    cnr_score = min(cnr / cnr_target, 1.0) * 10

    # Uniformity: 0-20 (1.0 = perfect, negative = very non-uniform)
    uni_score = max(0, min(uniformity, 1.0)) * 20

    # Tissue coverage: 0-15 (60%+ = full marks)
    tissue_score = min(tissue / 60.0, 1.0) * 15

    # Dynamic range: 0-12 (normalized, 500+ = full marks)
    dr_score = min(dyn_range / 500.0, 1.0) * 12

    # Entropy: 0-10 (5+ bits = full marks)
    ent_score = min(ent / 5.0, 1.0) * 10

    # Non-zero: 0-8 (50%+ = full marks)
    nz_score = min(nonzero / 50.0, 1.0) * 8

    total = snr_score + cnr_score + uni_score + tissue_score + dr_score + ent_score + nz_score

    if total >= 80:
        grade = "A"
    elif total >= 65:
        grade = "B"
    elif total >= 50:
        grade = "C"
    elif total >= 35:
        grade = "D"
    else:
        grade = "F"

    return {
        "grade": grade,
        "score": round(total, 1),
        "breakdown": {
            "snr": round(snr_score, 1),
            "cnr": round(cnr_score, 1),
            "uniformity": round(uni_score, 1),
            "tissue_coverage": round(tissue_score, 1),
            "dynamic_range": round(dr_score, 1),
            "entropy": round(ent_score, 1),
            "nonzero_coverage": round(nz_score, 1),
        },
        "snr_target": snr_target,
    }


def grade_study(series_grades: list[dict]) -> dict:
    """Compute an overall study quality grade from per-series grades."""
    if not series_grades:
        return {"grade": "N/A", "score": 0, "n_series": 0}

    scores = [g["score"] for g in series_grades]
    avg = sum(scores) / len(scores)
    worst = min(scores)
    best = max(scores)

    # Study grade is weighted: 70% average + 30% worst series
    study_score = avg * 0.7 + worst * 0.3

    if study_score >= 80:
        grade = "A"
    elif study_score >= 65:
        grade = "B"
    elif study_score >= 50:
        grade = "C"
    elif study_score >= 35:
        grade = "D"
    else:
        grade = "F"

    grade_dist = {}
    for g in series_grades:
        grade_dist[g["grade"]] = grade_dist.get(g["grade"], 0) + 1

    return {
        "grade": grade,
        "score": round(study_score, 1),
        "avg_score": round(avg, 1),
        "worst_score": round(worst, 1),
        "best_score": round(best, 1),
        "n_series": len(series_grades),
        "grade_distribution": grade_dist,
    }


# ── Anomaly detection ────────────────────────────────────────────────────────


def detect_anomalous_slices(vol: np.ndarray, z_threshold: float = 3.0) -> dict:
    """Detect outlier slices based on mean intensity z-score.

    A slice is flagged if its mean intensity deviates > z_threshold
    standard deviations from the volume-wide slice mean distribution.
    Common causes: motion, RF interference, coil dropout, wrong slice.
    """
    if vol.ndim < 3 or vol.shape[0] < 3:
        return {"n_anomalous": 0, "anomalous_slices": [], "slice_z_scores": []}

    axes = tuple(range(1, vol.ndim))
    slice_means = vol.mean(axis=axes)
    global_mean = slice_means.mean()
    global_std = slice_means.std()

    if global_std < 1e-6:
        return {"n_anomalous": 0, "anomalous_slices": [], "slice_z_scores": []}

    z_scores = (slice_means - global_mean) / global_std
    anomalous = []
    for i, z in enumerate(z_scores):
        if abs(z) > z_threshold:
            anomalous.append(
                {
                    "slice_index": int(i),
                    "z_score": round(float(z), 2),
                    "mean_intensity": round(float(slice_means[i]), 2),
                    "direction": "bright" if z > 0 else "dark",
                }
            )

    return {
        "n_anomalous": len(anomalous),
        "anomalous_slices": anomalous,
        "slice_z_scores": [round(float(z), 3) for z in z_scores],
        "global_slice_mean": round(float(global_mean), 2),
        "global_slice_std": round(float(global_std), 2),
    }


# ── Symmetry analysis ───────────────────────────────────────────────────────


def compute_symmetry(vol: np.ndarray, series_desc: str = "") -> dict:
    """Compute left-right hemisphere symmetry index.

    Only meaningful for axial acquisitions where the last axis is left-right.
    Sagittal/coronal series are skipped (returns neutral values).

    Symmetry index: 1.0 = perfect symmetry, 0.0 = completely asymmetric.
    Low symmetry can indicate: stroke, tumor, atrophy, mass effect, artifact.
    """
    if vol.ndim < 3:
        return {"symmetry_index": 1.0, "asymmetry_map_mean": 0.0, "interpretation": "N/A (2D)"}

    # Skip symmetry for non-axial acquisitions — L-R split is meaningless
    desc_up = series_desc.upper()
    if any(k in desc_up for k in ("SAG", "COR", "SAGITTAL", "CORONAL")):
        return {
            "symmetry_index": 1.0,
            "asymmetry_map_mean": 0.0,
            "interpretation": "N/A (non-axial)",
        }

    # Also skip if very few slices (single-slice MIP, projection)
    if vol.shape[0] < 3:
        return {
            "symmetry_index": 1.0,
            "asymmetry_map_mean": 0.0,
            "interpretation": "N/A (single slice)",
        }

    # Assume the last axis is left-right (sagittal)
    nx = vol.shape[-1]
    mid = nx // 2

    left = vol[..., :mid]
    right = vol[..., -mid:]
    right_flipped = np.flip(right, axis=-1)

    # Trim to same size if odd width
    min_w = min(left.shape[-1], right_flipped.shape[-1])
    left = left[..., :min_w]
    right_flipped = right_flipped[..., :min_w]

    # Normalized cross-correlation
    l_centered = left - left.mean()
    r_centered = right_flipped - right_flipped.mean()
    l_norm = np.sqrt((l_centered**2).sum())
    r_norm = np.sqrt((r_centered**2).sum())

    if l_norm < 1e-6 or r_norm < 1e-6:
        return {"symmetry_index": 0.0, "asymmetry_map_mean": 0.0}

    ncc = float((l_centered * r_centered).sum() / (l_norm * r_norm))

    # Asymmetry map: absolute difference normalized by average
    avg_intensity = (left + right_flipped) / 2.0 + 1e-6
    asymmetry_map = np.abs(left - right_flipped) / avg_intensity
    asym_mean = float(asymmetry_map.mean())

    return {
        "symmetry_index": round(max(0.0, ncc), 4),
        "asymmetry_map_mean": round(asym_mean, 4),
        "interpretation": (
            "symmetric"
            if ncc > 0.90
            else "normal"
            if ncc > 0.80
            else "mild asymmetry"
            if ncc > 0.65
            else "significant asymmetry"
        ),
    }


# ── Sharpness scoring ───────────────────────────────────────────────────────


def compute_sharpness(vol: np.ndarray) -> dict:
    """Estimate image sharpness via gradient magnitude.

    Computes the Sobel-like gradient magnitude across the mid-slice,
    then summarizes as mean/std/p95 of the gradient. Higher values = sharper.
    Low sharpness indicates: motion blur, low resolution, poor SNR.
    """
    if vol.ndim < 2:
        return {"sharpness_mean": 0.0, "sharpness_p95": 0.0}

    # Use mid-slice for 3D volumes
    if vol.ndim >= 3:
        mid = vol.shape[0] // 2
        slc = vol[mid].astype(np.float64)
    else:
        slc = vol.astype(np.float64)

    while slc.ndim > 2:
        slc = slc[0]

    # Gradient magnitude via finite differences (fast, no scipy needed)
    gy = np.diff(slc, axis=0)
    gx = np.diff(slc, axis=1)
    # Trim to common size
    min_h = min(gy.shape[0], gx.shape[0])
    min_w = min(gy.shape[1], gx.shape[1])
    grad_mag = np.sqrt(gy[:min_h, :min_w] ** 2 + gx[:min_h, :min_w] ** 2)

    gmean = float(grad_mag.mean())
    gstd = float(grad_mag.std())
    gp95 = float(np.percentile(grad_mag, 95))

    return {
        "sharpness_mean": round(gmean, 3),
        "sharpness_std": round(gstd, 3),
        "sharpness_p95": round(gp95, 3),
        "interpretation": (
            "very sharp"
            if gmean > 20
            else "sharp"
            if gmean > 10
            else "moderate"
            if gmean > 5
            else "soft/blurry"
        ),
    }


# ── Motion artifact detection ────────────────────────────────────────────────


def detect_motion_artifacts(vol: np.ndarray) -> dict:
    """Detect motion/ghosting artifacts via directional energy analysis.

    Motion artifacts in MRI appear as ghosting along the phase-encode
    direction. This manifests as higher energy in one direction of the
    2D FFT compared to the orthogonal direction.

    Ghosting ratio > 1.3 suggests significant motion artifacts.
    """
    if vol.ndim < 2:
        return {"ghosting_ratio": 1.0, "motion_detected": False}

    # Use mid-slice
    if vol.ndim >= 3:
        mid = vol.shape[0] // 2
        slc = vol[mid].astype(np.float64)
    else:
        slc = vol.astype(np.float64)

    while slc.ndim > 2:
        slc = slc[0]

    # 2D FFT magnitude spectrum
    fft2 = np.fft.fft2(slc)
    fft_shift = np.fft.fftshift(fft2)
    magnitude = np.abs(fft_shift)

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2

    # Exclude DC component (center pixel)
    magnitude[center_h, center_w] = 0

    # Energy in horizontal vs vertical bands (excluding center)
    band_w = max(w // 8, 2)
    band_h = max(h // 8, 2)

    # Horizontal energy (phase encode direction for axial scans)
    horiz_energy = magnitude[center_h - band_h : center_h + band_h, :].sum()
    # Vertical energy
    vert_energy = magnitude[:, center_w - band_w : center_w + band_w].sum()

    if min(horiz_energy, vert_energy) < 1e-6:
        ratio = 1.0
    else:
        ratio = max(horiz_energy, vert_energy) / min(horiz_energy, vert_energy)

    # Slice-to-slice correlation for motion detection
    slice_corr = 1.0
    if vol.ndim >= 3 and vol.shape[0] > 2:
        mid = vol.shape[0] // 2
        s1 = vol[mid].astype(np.float64).ravel()
        s2 = vol[mid + 1].astype(np.float64).ravel()
        if s1.std() > 1e-6 and s2.std() > 1e-6:
            slice_corr = float(np.corrcoef(s1, s2)[0, 1])

    # Combined scoring: weight both ghosting ratio and inter-slice correlation.
    # MRI images are naturally anisotropic (rectangular FOV, different phase/freq
    # encode sizes) so ratios up to ~1.8 are normal.
    motion_detected = ratio > 2.5 and slice_corr < 0.80

    # Motion severity score: 0-100 (higher = worse motion)
    ratio_severity = min((ratio - 1.0) / 4.0, 1.0) * 60  # ratio contribution
    corr_severity = max(0, (1.0 - slice_corr) / 0.3) * 40  # correlation contribution
    motion_score = round(min(ratio_severity + corr_severity, 100), 1)

    return {
        "ghosting_ratio": round(float(ratio), 3),
        "directional_energy_h": round(float(horiz_energy), 1),
        "directional_energy_v": round(float(vert_energy), 1),
        "adjacent_slice_correlation": round(slice_corr, 4),
        "motion_detected": motion_detected,
        "motion_severity_score": motion_score,
        "interpretation": (
            "no motion"
            if ratio < 1.5 and slice_corr > 0.90
            else "minimal"
            if ratio < 2.0 and slice_corr > 0.80
            else "moderate motion"
            if ratio < 3.0
            else "significant motion/ghosting"
        ),
    }


# ── Combined analysis ────────────────────────────────────────────────────────


def analyze_volume_quality(vol: np.ndarray, vstats: dict, series_desc: str = "") -> dict:
    """Run all quality analyses in parallel threads on a single volume.

    Returns combined dict with grade, anomalies, symmetry, sharpness, motion.
    """
    with ThreadPoolExecutor(max_workers=5) as pool:
        grade_fut = pool.submit(grade_series, vstats, series_desc)
        anomaly_fut = pool.submit(detect_anomalous_slices, vol)
        symmetry_fut = pool.submit(compute_symmetry, vol, series_desc)
        sharpness_fut = pool.submit(compute_sharpness, vol)
        motion_fut = pool.submit(detect_motion_artifacts, vol)

        return {
            "quality_grade": grade_fut.result(),
            "anomaly_detection": anomaly_fut.result(),
            "symmetry_analysis": symmetry_fut.result(),
            "sharpness_analysis": sharpness_fut.result(),
            "motion_analysis": motion_fut.result(),
        }

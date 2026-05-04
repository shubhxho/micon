"""Per-file DICOM extraction, sequence classification, volume stats, conformance."""

from __future__ import annotations

import multiprocessing
import multiprocessing.shared_memory
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from ._logging import get_logger
from .constants import REQUIRED_MR_TAGS, TRANSFER_SYNTAX_NAMES
from .helpers import entropy, safe_getfloat, safe_value, skewness_kurtosis

logger = get_logger(__name__)

# ── Shared memory helpers (zero-copy pixel transfer between processes) ────────


def _shm_create(arr: np.ndarray) -> multiprocessing.shared_memory.SharedMemory | None:
    """Write a numpy array into a new shared memory block. Returns the shm object."""
    from multiprocessing.shared_memory import SharedMemory

    try:
        shm = SharedMemory(create=True, size=arr.nbytes)
        buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        buf[:] = arr
        shm.close()  # close local handle — block persists until unlinked
        return shm
    except Exception:
        return None


def shm_read(name: str, shape: list[int], dtype: str) -> np.ndarray | None:
    """Read a numpy array from shared memory by name. Returns None on failure."""
    from multiprocessing.shared_memory import SharedMemory

    try:
        shm = SharedMemory(name=name, create=False)
        arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf).copy()
        shm.close()
        return arr
    except Exception:
        return None


def shm_cleanup(names: list[str]) -> None:
    """Unlink shared memory blocks by name. Safe to call if already unlinked."""
    from multiprocessing.shared_memory import SharedMemory

    for name in names:
        try:
            shm = SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except Exception:
            pass


# ── Per-file extraction (runs in worker processes) ────────────────────────────


def _compute_pixel_stats(arr: np.ndarray, raw_dtype: str) -> dict:
    """Compute all pixel-level stats — Metal GPU if available, else threaded numpy."""
    from .metal import gpu_pixel_stats

    # Try GPU path first (single GPU session for all stats)
    gpu_result = gpu_pixel_stats(arr)
    if gpu_result is not None:
        hist_counts, hist_edges = np.histogram(arr.ravel(), 50)
        gpu_result["pixel_shape"] = list(arr.shape)
        gpu_result["pixel_dtype"] = raw_dtype
        gpu_result["histogram_counts"] = hist_counts.tolist()
        gpu_result["histogram_edges"] = hist_edges.tolist()
        return gpu_result

    # CPU fallback — threaded numpy ops (GIL-releasing)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pcts_fut = pool.submit(np.percentile, arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        hist_fut = pool.submit(np.histogram, arr.ravel(), 50)
        entropy_fut = pool.submit(entropy, arr)
        sk_fut = pool.submit(skewness_kurtosis, arr)

        pcts = pcts_fut.result()
        hist_counts, hist_edges = hist_fut.result()
        px_entropy = entropy_fut.result()
        px_skewness, px_kurtosis = sk_fut.result()

    return {
        "pixel_shape": list(arr.shape),
        "pixel_dtype": raw_dtype,
        "pixel_min": float(arr.min()),
        "pixel_max": float(arr.max()),
        "pixel_mean": float(arr.mean()),
        "pixel_std": float(arr.std()),
        "pixel_median": float(pcts[4]),
        "pixel_p1": float(pcts[0]),
        "pixel_p5": float(pcts[1]),
        "pixel_p10": float(pcts[2]),
        "pixel_p25": float(pcts[3]),
        "pixel_p50": float(pcts[4]),
        "pixel_p75": float(pcts[5]),
        "pixel_p90": float(pcts[6]),
        "pixel_p95": float(pcts[7]),
        "pixel_p99": float(pcts[8]),
        "pixel_iqr": float(pcts[5] - pcts[3]),
        "nonzero_ratio": float((arr != 0).mean()),
        "pixel_entropy": float(px_entropy),
        "pixel_skewness": float(px_skewness),
        "pixel_kurtosis": float(px_kurtosis),
        "histogram_counts": hist_counts.tolist(),
        "histogram_edges": hist_edges.tolist(),
    }


def extract_single_file(fpath: str, skip_pixels: bool = True) -> dict:
    """Extract DICOM metadata from one file. Runs in subprocess.

    When skip_pixels=True (default), only reads headers — no pixel decode.
    This is ~700x faster per file (0.8ms vs 574ms) because pixel stats are
    redundant: volume_stats in stage 4 computes them on the assembled volume.

    When skip_pixels=False, also decodes pixels, computes per-file stats,
    and writes to shared memory for stage 4 volume assembly.
    """
    f = Path(fpath)

    if skip_pixels:
        ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
    else:
        ds = pydicom.dcmread(fpath, force=True)

    record = {"_filename": f.name, "_filepath": fpath, "_file_size_bytes": f.stat().st_size}

    # All tags
    for elem in ds:
        if elem.tag.group == 0x7FE0:
            record["_has_pixel_data"] = True
            continue
        kw = elem.keyword or f"Tag_{elem.tag.group:04X}_{elem.tag.element:04X}"
        record[kw] = safe_value(elem)
    record.setdefault("_has_pixel_data", False)

    # File meta
    if hasattr(ds, "file_meta"):
        fm = ds.file_meta
        ts_uid = str(getattr(fm, "TransferSyntaxUID", ""))
        record["_transfer_syntax_uid"] = ts_uid
        record["_transfer_syntax_name"] = TRANSFER_SYNTAX_NAMES.get(ts_uid, ts_uid)
        record["_media_storage_sop_class"] = str(getattr(fm, "MediaStorageSOPClassUID", ""))

    # Pixel stats + shared memory (only when skip_pixels=False)
    if not skip_pixels and record.get("_has_pixel_data"):
        try:
            arr = ds.pixel_array.astype(np.float64)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            offset = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + offset
            record.update(_compute_pixel_stats(arr, str(ds.pixel_array.dtype)))

            arr32 = arr.astype(np.float32)
            shm = _shm_create(arr32)
            if shm is not None:
                record["_shm_name"] = shm.name
                record["_shm_shape"] = list(arr32.shape)
                record["_shm_dtype"] = str(arr32.dtype)
        except Exception as e:
            record["pixel_error"] = str(e)

    # Series grouping info
    record["_series_uid"] = str(getattr(ds, "SeriesInstanceUID", "unknown"))
    record["_series_number"] = getattr(ds, "SeriesNumber", "")
    record["_series_description"] = str(getattr(ds, "SeriesDescription", ""))
    record["_modality"] = str(getattr(ds, "Modality", ""))
    record["_sop_class_uid"] = str(getattr(ds, "SOPClassUID", ""))

    # Patient info
    record["_patient_id"] = str(getattr(ds, "PatientID", ""))
    record["_patient_name"] = str(getattr(ds, "PatientName", ""))
    record["_patient_sex"] = str(getattr(ds, "PatientSex", ""))
    record["_patient_birth_date"] = str(getattr(ds, "PatientBirthDate", ""))
    record["_patient_weight"] = str(getattr(ds, "PatientWeight", ""))
    record["_study_date"] = str(getattr(ds, "StudyDate", ""))
    record["_study_description"] = str(getattr(ds, "StudyDescription", ""))
    record["_institution"] = str(getattr(ds, "InstitutionName", ""))
    record["_manufacturer"] = str(getattr(ds, "Manufacturer", ""))
    record["_model"] = str(getattr(ds, "ManufacturerModelName", ""))
    record["_field_strength"] = str(getattr(ds, "MagneticFieldStrength", ""))
    record["_software_versions"] = str(getattr(ds, "SoftwareVersions", ""))
    record["_station_name"] = str(getattr(ds, "StationName", ""))

    # Sequence params
    record["_tr"] = safe_getfloat(ds, "RepetitionTime")
    record["_te"] = safe_getfloat(ds, "EchoTime")
    record["_ti"] = safe_getfloat(ds, "InversionTime")
    record["_fa"] = safe_getfloat(ds, "FlipAngle")
    record["_b_value"] = safe_getfloat(ds, "DiffusionBValue")

    # Spatial ordering — pre-extract so stage 4 doesn't re-read headers
    pos = getattr(ds, "ImagePositionPatient", None)
    if pos is not None and len(pos) >= 3:
        record["_z_position"] = float(pos[2])
    else:
        record["_z_position"] = None
    record["_instance_number"] = int(getattr(ds, "InstanceNumber", 0) or 0)

    return record


# ── Sequence classification ─────────────────────────────────────────────────


def _normalize_series_name(desc: str) -> str:
    """Normalize series description for robust pattern matching.

    Strips whitespace, collapses separators to spaces, uppercases,
    and removes common vendor prefixes to handle naming inconsistencies
    like "T1W", "T1-WEIGHTED", "T1_3D", "t1w_axial".
    """
    s = desc.strip().upper()
    s = _RE_SEPARATORS.sub(" ", s)
    s = _RE_WHITESPACE.sub(" ", s)
    for prefix in ("SE ", "MS ", "WIP ", "REF ", "PH ", "MAG "):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s


import re as _re

# Pre-compiled regex patterns for sequence classification — compiled ONCE at
# module load instead of recompiling 50+ patterns per series call.
_SEQ_PATTERNS: list[tuple[_re.Pattern[str], str]] = [
    (_re.compile(p), t)
    for p, t in [
        (r"\bE?ADC\b", "ADC Map"),
        (r"\bFA\s*MAP\b", "FA Map"),
        (r"\bTRACE\b", "Trace DWI"),
        (r"\bISO.*B\b", "Isotropic DWI"),
        (r"\bREG.*DWI\b", "Registered DWI"),
        (r"\bMPRAGE\b", "MPRAGE (3D T1)"),
        (r"\bMP2RAGE\b", "MP2RAGE"),
        (r"\bSPACE\b", "SPACE (3D TSE)"),
        (r"\bCUBE\b", "CUBE (3D FSE)"),
        (r"\bVISTA\b", "VISTA (3D TSE)"),
        (r"\b3D\s*T1\b", "3D T1-weighted"),
        (r"\b3D\s*T2\b", "3D T2-weighted"),
        (r"\b3D\s*FLAIR\b", "3D FLAIR"),
        (r"\bFLAIR\b", "FLAIR"),
        (r"\bSTIR\b", "STIR"),
        (r"\bDIR\b", "DIR (Double Inversion Recovery)"),
        (r"\bDWI\b", "DWI"),
        (r"\bDIFF\b", "DWI"),
        (r"\bDTI\b", "DTI"),
        (r"\bSWAN\b", "SWI (SWAN)"),
        (r"\bSWI\b", "SWI"),
        (r"\bFILT.?PHA\b", "Phase Image"),
        (r"\bPHASE\b", "Phase Image"),
        (r"\bPROCESSED\s*IMAGE", "Processed/MIP"),
        (r"\bCOL:", "Color MIP"),
        (r"\bPJN:", "Projection MIP"),
        (r"\bB.?FFE\b", "Balanced FFE (bSSFP)"),
        (r"\bBSSFP\b", "Balanced FFE (bSSFP)"),
        (r"\bFIESTA\b", "FIESTA (bSSFP)"),
        (r"\bTRUE.?FISP\b", "TrueFISP (bSSFP)"),
        (r"\bGRE\b", "GRE (Gradient Echo)"),
        (r"\bFFE\b", "FFE (Gradient Echo)"),
        (r"\bSPGR\b", "SPGR (Spoiled GRE)"),
        (r"\bFLASH\b", "FLASH (Spoiled GRE)"),
        (r"\bTOF\b", "TOF MRA"),
        (r"\bMRA\b", "MRA"),
        (r"\bMRV\b", "MRV"),
        (r"\bASL\b", "ASL Perfusion"),
        (r"\bDSC\b", "DSC Perfusion"),
        (r"\bDCE\b", "DCE Perfusion"),
        (r"\bPERFUS", "Perfusion"),
        (r"\bMRS\b", "MR Spectroscopy"),
        (r"\bT1\s*W", "T1-weighted"),
        (r"\bT1\b", "T1-weighted"),
        (r"\bT2\s*W", "T2-weighted"),
        (r"\bT2\b", "T2-weighted"),
        (r"\bPD\s*W?\b", "Proton Density"),
        (r"\bTSE\b", "TSE (Turbo Spin Echo)"),
        (r"\bFSE\b", "FSE (Fast Spin Echo)"),
        (r"\bSURVEY\b", "Survey/Localizer"),
        (r"\bLOCAL", "Survey/Localizer"),
        (r"\bSCOUT\b", "Survey/Localizer"),
        (r"\bLOC\b", "Survey/Localizer"),
    ]
]

# Pre-compiled regex for _normalize_series_name
_RE_SEPARATORS = _re.compile(r"[_\-/\\]+")
_RE_WHITESPACE = _re.compile(r"\s+")


def classify_sequence(
    desc: str,
    tr: float | None,
    te: float | None,
    ti: float | None,
    fa: float | None,
    b_value: float | None,
) -> dict:
    desc_norm = _normalize_series_name(desc)
    result = {"sequence_type": "Unknown", "confidence": "low", "reasoning": []}

    for pattern, seq_type in _SEQ_PATTERNS:
        if pattern.search(desc_norm):
            result["sequence_type"] = seq_type
            result["confidence"] = "high"
            result["reasoning"].append(f"Name matches '{pattern.pattern}'")
            break

    # Refine diffusion with b-value
    if b_value is not None and result["sequence_type"] in (
        "DWI",
        "DTI",
        "Trace DWI",
        "Isotropic DWI",
    ):
        if b_value > 500:
            result["reasoning"].append(f"b-value={b_value} confirms diffusion weighting")
        elif b_value == 0:
            result["sequence_type"] = "DWI (b=0)"
            result["reasoning"].append("b=0 reference image")

    # Timing-based fallback with multi-parameter heuristics
    if result["confidence"] == "low":
        if ti is not None:
            if ti > 1800:
                result["sequence_type"] = "FLAIR"
                result["confidence"] = "high"
                result["reasoning"].append(f"TI={ti}ms -> CSF nulling (FLAIR)")
            elif 1200 < ti <= 1800:
                result["sequence_type"] = "FLAIR (probable)"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TI={ti}ms -> possible FLAIR")
            elif 100 < ti < 300:
                result["sequence_type"] = "STIR"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TI={ti}ms -> fat nulling (STIR)")
            elif 400 < ti < 1200 and tr is not None and tr < 3000:
                result["sequence_type"] = "T1-weighted (IR)"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TI={ti}ms TR={tr}ms -> T1 inversion recovery")

        if result["confidence"] == "low" and tr is not None and te is not None:
            if tr > 4000 and te > 80:
                result["sequence_type"] = "T2-weighted"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TR={tr}ms TE={te}ms -> T2")
            elif tr > 4000 and te < 30:
                result["sequence_type"] = "Proton Density"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TR={tr}ms TE={te}ms -> long TR + short TE (PD)")
            elif 2000 < tr <= 4000 and te > 80:
                result["sequence_type"] = "T2-weighted"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TR={tr}ms TE={te}ms -> T2")
            elif tr < 800 and te < 30:
                result["sequence_type"] = "T1-weighted"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TR={tr}ms TE={te}ms -> short TR + short TE (T1)")
            elif tr < 800 and te < 10 and fa is not None and fa < 30:
                result["sequence_type"] = "GRE (Gradient Echo)"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TR={tr}ms TE={te}ms FA={fa}° -> GRE")
            elif tr < 10 and te < 5:
                result["sequence_type"] = "Balanced FFE (bSSFP)"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TR={tr}ms TE={te}ms -> very short TR/TE (bSSFP)")

        if result["confidence"] == "low" and b_value is not None and b_value > 0:
            result["sequence_type"] = "DWI"
            result["confidence"] = "medium"
            result["reasoning"].append(f"b-value={b_value} -> diffusion-weighted")

    result["parameters"] = {
        "TR_ms": tr,
        "TE_ms": te,
        "TI_ms": ti,
        "flip_angle_deg": fa,
        "b_value": b_value,
    }
    return result


# ── Volume stats ─────────────────────────────────────────────────────────────


def _recursive_slice_stats(vol: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Recursively compute per-slice mean and std via divide-and-conquer.

    Splits the slice range in half, recurses on each half, and concatenates.
    """
    if end - start <= 1:
        slc = vol[start]
        return np.array([slc.mean()]), np.array([slc.std()])

    mid = (start + end) // 2
    left_means, left_stds = _recursive_slice_stats(vol, start, mid)
    right_means, right_stds = _recursive_slice_stats(vol, mid, end)
    return np.concatenate([left_means, right_means]), np.concatenate([left_stds, right_stds])


def _estimate_background_noise(vol: np.ndarray) -> float:
    """Estimate noise from background air regions using multi-slice sampling.

    Strategy (NEMA-inspired):
      1. Sample corners from multiple slices (not just mid-slice) for robustness
      2. Validate that corners contain air (low mean vs tissue)
      3. Reject outlier corners via IQR filtering (removes partial-tissue corners)
      4. Fall back to lowest-intensity 10% if corners fail
    """
    if vol.ndim < 2:
        return max(float(vol.std()), 1e-6)

    vol_mean = float(vol.mean())

    # Sample multiple slices for more robust noise estimation
    if vol.ndim >= 3 and vol.shape[0] > 4:
        n_slices = vol.shape[0]
        # Sample 5 slices spread across the volume (skip first/last 10%)
        start = max(n_slices // 10, 1)
        end = n_slices - start
        sample_indices = np.linspace(start, end - 1, min(5, end - start), dtype=int)
    elif vol.ndim >= 3:
        sample_indices = [vol.shape[0] // 2]
    else:
        sample_indices = None  # 2D

    corner_stds = []
    if sample_indices is not None:
        for si in sample_indices:
            slc = vol[si]
            while slc.ndim > 2:
                slc = slc[0]
            h, w = slc.shape[:2]
            ch, cw = max(h // 8, 2), max(w // 8, 2)
            corners = np.concatenate(
                [
                    slc[:ch, :cw].ravel(),
                    slc[:ch, -cw:].ravel(),
                    slc[-ch:, :cw].ravel(),
                    slc[-ch:, -cw:].ravel(),
                ]
            )
            nonzero_corners = corners[corners != 0]
            if len(nonzero_corners) > 10:
                corner_mean = float(nonzero_corners.mean())
                if corner_mean < vol_mean * 0.3:
                    corner_stds.append(float(nonzero_corners.std()))
    else:
        # 2D case
        slc = vol
        while slc.ndim > 2:
            slc = slc[0]
        h, w = slc.shape[:2]
        ch, cw = max(h // 8, 2), max(w // 8, 2)
        corners = np.concatenate(
            [
                slc[:ch, :cw].ravel(),
                slc[:ch, -cw:].ravel(),
                slc[-ch:, :cw].ravel(),
                slc[-ch:, -cw:].ravel(),
            ]
        )
        nonzero_corners = corners[corners != 0]
        if len(nonzero_corners) > 10:
            corner_mean = float(nonzero_corners.mean())
            if corner_mean < vol_mean * 0.3:
                corner_stds.append(float(nonzero_corners.std()))

    # Use median of per-slice corner noise (robust to outliers)
    if len(corner_stds) >= 2:
        noise = float(np.median(corner_stds))
        if noise > 1e-6:
            return noise
    elif len(corner_stds) == 1 and corner_stds[0] > 1e-6:
        return corner_stds[0]

    # Fallback: noise from lowest-intensity 10% of the volume
    flat = vol.ravel()
    nonzero = flat[flat != 0]
    if len(nonzero) < 10:
        return max(float(vol.std()), 1e-6)
    p10 = float(np.percentile(nonzero, 10))
    background = nonzero[nonzero <= p10]
    if len(background) > 10:
        noise = float(background.std())
        if noise > 1e-6:
            return noise

    return max(float(vol.std()), 1e-6)


def _otsu_threshold(vol: np.ndarray) -> float:
    """Compute Otsu's threshold for foreground/background separation.

    Finds the threshold that minimizes intra-class variance across
    a 256-bin histogram, providing adaptive tissue segmentation.
    """
    hist, edges = np.histogram(vol.ravel(), bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return float(vol.mean())

    w0 = np.cumsum(hist).astype(np.float64)
    w1 = total - w0
    sum_total = (hist * centers).sum()
    sum0 = np.cumsum(hist * centers).astype(np.float64)
    sum1 = sum_total - sum0

    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = sum0 / w0
        mu1 = sum1 / w1
        between = w0 * w1 * (mu0 - mu1) ** 2

    between = np.nan_to_num(between, 0.0)
    best_idx = int(np.argmax(between))
    return float(centers[best_idx])


def volume_stats(vol: np.ndarray, sitk_img: sitk.Image | None = None) -> dict:
    from .metal import gpu_available, gpu_slice_stats, gpu_stats_and_percentiles, gpu_tissue_pct

    ndim = vol.ndim
    if sitk_img:
        spacing = list(sitk_img.GetSpacing())
        origin = list(sitk_img.GetOrigin())
        direction = list(sitk_img.GetDirection())
    else:
        spacing = [1.0] * ndim
        origin = [0.0] * ndim
        direction = []

    sp3 = ([*spacing, 1.0, 1.0, 1.0])[:3]
    voxel_vol_mm3 = float(np.prod(sp3))
    total_vol_mm3 = voxel_vol_mm3 * int(np.prod(vol.shape[:3]))
    fov = [float(vol.shape[i]) * float(sp3[min(2 - i, len(sp3) - 1)]) for i in range(min(3, ndim))]

    # Background noise estimation for proper SNR (signal / noise_floor)
    bg_noise = _estimate_background_noise(vol)

    # Otsu's threshold for adaptive tissue segmentation
    otsu_thresh = _otsu_threshold(vol)

    if gpu_available() and vol.size >= 16_384:
        # ── Metal GPU path — single lock for stats + percentiles ──
        gs, pcts = gpu_stats_and_percentiles(vol, [1, 5, 15, 25, 50, 75, 95, 99])
        tissue_pct = gpu_tissue_pct(vol, otsu_thresh)

        if ndim >= 3 and vol.shape[0] > 1:
            slice_means, slice_stds = gpu_slice_stats(vol)
        else:
            slice_means = np.array([gs["mean"]])
            slice_stds = np.array([gs["std"]])
        slice_snrs = slice_means / (slice_stds + 1e-6)

        vol_mean, vol_std = gs["mean"], gs["std"]
        vol_entropy = gs["entropy"]
        vol_skewness, vol_kurtosis = gs["skewness"], gs["kurtosis"]
    else:
        # ── CPU path — only expensive O(V) ops in threads ──
        # percentile, entropy, skewness/kurtosis are expensive and GIL-releasing.
        # mean/std are cheap O(V) single-pass ops — no thread overhead needed.
        vol_mean = float(vol.mean())
        vol_std = float(vol.std())

        with ThreadPoolExecutor(max_workers=3) as pool:
            pcts_fut = pool.submit(np.percentile, vol, [1, 5, 15, 25, 50, 75, 95, 99])
            entropy_fut = pool.submit(entropy, vol)
            sk_fut = pool.submit(skewness_kurtosis, vol)

            pcts = pcts_fut.result()
            vol_entropy = float(entropy_fut.result())
            vol_skewness, vol_kurtosis = sk_fut.result()

        # Compute tissue mask ONCE and reuse for tissue_pct + SNR
        tissue_mask = vol > otsu_thresh
        tissue_pct = float(tissue_mask.mean() * 100)

        if ndim >= 3 and vol.shape[0] > 1:
            axes = tuple(range(1, ndim))
            slice_means = vol.mean(axis=axes)
            slice_stds = vol.std(axis=axes)
        else:
            slice_means = np.array([vol_mean])
            slice_stds = np.array([vol_std])
        slice_snrs = slice_means / (slice_stds + 1e-6)

    # SNR: signal (tissue mean) / background noise — much more meaningful
    # than naive mean/std which is dominated by tissue contrast.
    # Compute tissue mask once (GPU path didn't compute it yet).
    if gpu_available() and vol.size >= 16_384:
        tissue_mask = vol > otsu_thresh
    tissue_mean = float(vol[tissue_mask].mean()) if tissue_mask.any() else vol_mean
    tissue_std = (
        float(vol[tissue_mask].std()) if tissue_mask.any() and tissue_mask.sum() > 100 else vol_std
    )

    # NEMA-like dual SNR estimation with automatic selection
    snr_from_bg = float(tissue_mean / (bg_noise + 1e-6))
    snr_from_tissue = float(tissue_mean / (tissue_std + 1e-6))

    # If bg_noise estimate gives implausible SNR (>100), fall back to tissue-based
    snr_estimate = snr_from_tissue if snr_from_bg > 100 else snr_from_bg

    # Contrast-to-noise ratio: tissue vs background separation
    bg_mask = ~tissue_mask & (vol > 0)
    if bg_mask.any() and bg_mask.sum() > 100:
        bg_mean = float(vol[bg_mask].mean())
        cnr = float(abs(tissue_mean - bg_mean) / (bg_noise + 1e-6))
    else:
        cnr = snr_estimate  # fallback: CNR ~ SNR when no clear background

    naive_snr = float(vol_mean / (vol_std + 1e-6))

    return {
        "volume_shape": list(vol.shape),
        "volume_voxel_count": int(np.prod(vol.shape)),
        "spacing_mm": sp3,
        "origin_mm": origin,
        "direction_cosines": direction,
        "voxel_volume_mm3": voxel_vol_mm3,
        "total_volume_mm3": total_vol_mm3,
        "total_volume_cc": total_vol_mm3 / 1000.0,
        "fov_mm": fov,
        "volume_min": float(vol.min()),
        "volume_max": float(vol.max()),
        "volume_mean": vol_mean,
        "volume_std": vol_std,
        "volume_median": float(pcts[4]),
        "volume_p1": float(pcts[0]),
        "volume_p5": float(pcts[1]),
        "volume_p25": float(pcts[3]),
        "volume_p75": float(pcts[5]),
        "volume_p95": float(pcts[6]),
        "volume_p99": float(pcts[7]),
        "volume_iqr": float(pcts[5] - pcts[3]),
        "volume_dynamic_range": float(vol.max() - vol.min()),
        "volume_snr_estimate": snr_estimate,
        "volume_snr_naive": naive_snr,
        "volume_cnr": cnr,
        "background_noise_std": bg_noise,
        "otsu_threshold": otsu_thresh,
        "volume_nonzero_pct": float((vol != 0).mean() * 100),
        "volume_tissue_pct": tissue_pct,
        "volume_entropy": vol_entropy,
        "volume_skewness": vol_skewness,
        "volume_kurtosis": vol_kurtosis,
        "slice_snr_mean": float(slice_snrs.mean()),
        "slice_snr_std": float(slice_snrs.std()),
        "slice_snr_min": float(slice_snrs.min()),
        "slice_snr_max": float(slice_snrs.max()),
        "slice_intensity_uniformity": 1.0 - float(slice_means.std() / (slice_means.mean() + 1e-6)),
    }


# ── Conformance check ────────────────────────────────────────────────────────


def _check_one_record(r: dict) -> dict | None:
    """Check conformance for a single record. Returns issue dict or None."""
    missing = [tag for tag in REQUIRED_MR_TAGS if r.get(tag) in (None, "", "None")]
    if missing:
        return {
            "filename": r.get("_filename", "?"),
            "missing_tags": missing,
            "missing_count": len(missing),
            "completeness_pct": round(100 * (1 - len(missing) / len(REQUIRED_MR_TAGS)), 1),
        }
    return None


def check_conformance(records: list[dict]) -> list[dict]:
    """Check conformance across all records using threaded parallel checking."""
    if not records:
        return []

    from concurrent.futures import as_completed

    issues = []
    with ThreadPoolExecutor(max_workers=min(len(records), 8)) as pool:
        futures = [pool.submit(_check_one_record, r) for r in records]
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                issues.append(result)
    return issues

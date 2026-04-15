#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydicom>=3.0",
#   "SimpleITK>=2.4",
#   "nibabel>=5.3",
#   "polars>=1.17",
#   "numpy>=2.1",
#   "rich>=13.9",
#   "typer>=0.15",
#   "openai>=1.50",
#   "matplotlib>=3.10",
#   "scipy>=1.14",
#   "pillow>=11.0",
# ]
# ///

"""
dcm_extract.py — Comprehensive DICOM brain scan extraction + Claude analysis

Extracts EVERY tag, groups by series, auto-classifies MRI sequences,
computes per-series volume stats, generates multi-plane montages,
cross-series comparisons, DICOM conformance checks, and a full HTML report.

Usage:  uv run main.py <folder> [--claude] [--export-nii] [--out-dir ./output]
"""

from __future__ import annotations

import base64
import datetime
import html as html_mod
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Optional

import openai
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
import pydicom
import polars as pl
import SimpleITK as sitk
import typer
from pydicom.uid import UID
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree

app = typer.Typer(rich_markup_mode="rich")
console = Console()

# ── UID lookups ──────────────────────────────────────────────────────────────

SOP_CLASS_NAMES = {
    "1.2.840.10008.5.1.4.1.1.11.1": "Grayscale Softcopy PS",
    "1.2.840.10008.5.1.4.1.1.11.2": "Color Softcopy PS",
    "1.2.840.10008.5.1.4.1.1.4":     "MR Image Storage",
    "1.2.840.10008.5.1.4.1.1.2":     "CT Image Storage",
    "1.2.840.10008.5.1.4.1.1.7":     "Secondary Capture",
    "1.2.840.10008.5.1.4.1.1.4.1":   "Enhanced MR Image Storage",
    "1.2.840.10008.5.1.4.1.1.66":    "Raw Data Storage",
    "1.2.840.10008.5.1.4.1.1.66.4":  "Segmentation Storage",
    "1.2.840.10008.5.1.1.1":          "Basic Film Session",
}

TRANSFER_SYNTAX_NAMES = {
    "1.2.840.10008.1.2":       "Implicit VR LE",
    "1.2.840.10008.1.2.1":     "Explicit VR LE",
    "1.2.840.10008.1.2.2":     "Explicit VR BE",
    "1.2.840.10008.1.2.4.50":  "JPEG Baseline",
    "1.2.840.10008.1.2.4.70":  "JPEG Lossless",
    "1.2.840.10008.1.2.4.80":  "JPEG-LS Lossless",
    "1.2.840.10008.1.2.4.90":  "JPEG 2000 Lossless",
    "1.2.840.10008.1.2.4.91":  "JPEG 2000",
    "1.2.840.10008.1.2.5":     "RLE Lossless",
}

NON_IMAGE_SOP = {
    "1.2.840.10008.5.1.4.1.1.11.1",
    "1.2.840.10008.5.1.4.1.1.11.2",
    "1.2.840.10008.5.1.1.1",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_value(elem) -> str | int | float | list | None:
    if elem is None:
        return None
    val = elem.value
    if isinstance(val, pydicom.sequence.Sequence):
        return f"[Sequence: {len(val)} items]"
    if isinstance(val, pydicom.valuerep.PersonName):
        return str(val)
    if isinstance(val, pydicom.uid.UID):
        return str(val)
    if isinstance(val, bytes):
        return val.hex() if len(val) <= 64 else f"[bytes: {len(val)}]"
    if isinstance(val, pydicom.multival.MultiValue):
        return [float(v) if isinstance(v, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal)) else str(v) for v in val]
    if isinstance(val, (int, float)):
        return val
    return str(val)


def _to_json(v):
    """Make any value JSON-serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _entropy(arr: np.ndarray) -> float:
    hist, _ = np.histogram(arr.ravel(), bins=256)
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def _skewness(arr: np.ndarray) -> float:
    m, s = arr.mean(), arr.std()
    return float(((arr - m) ** 3).mean() / s ** 3) if s > 1e-10 else 0.0


def _kurtosis(arr: np.ndarray) -> float:
    m, s = arr.mean(), arr.std()
    return float(((arr - m) ** 4).mean() / s ** 4 - 3.0) if s > 1e-10 else 0.0


def _safe_squeeze(arr: np.ndarray) -> np.ndarray:
    """Squeeze leading singleton dims down to 3D."""
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    return arr


def _get_2d_slice(vol: np.ndarray, idx: int) -> np.ndarray:
    """Get a 2D slice from a volume, handling extra dims."""
    slc = vol[idx]
    while slc.ndim > 2:
        slc = slc[0]
    return slc


# ── Sequence classification ─────────────────────────────────────────────────

def classify_sequence(desc: str, tr: float | None, te: float | None,
                      ti: float | None, fa: float | None,
                      b_value: float | None) -> dict:
    """Auto-classify MRI sequence from metadata."""
    desc_up = desc.upper()
    result = {"sequence_type": "Unknown", "confidence": "low", "reasoning": []}

    # Direct name matching (high confidence)
    name_map = {
        "T1W": "T1-weighted", "T1": "T1-weighted",
        "T2W": "T2-weighted", "T2": "T2-weighted",
        "FLAIR": "FLAIR", "DWI": "DWI", "ADC": "ADC",
        "GRE": "GRE (Gradient Echo)", "B_FFE": "Balanced FFE (bSSFP)",
        "B-FFE": "Balanced FFE (bSSFP)", "BFFE": "Balanced FFE (bSSFP)",
        "SWI": "SWI", "SURVEY": "Survey/Localizer",
        "LOCALIZER": "Survey/Localizer", "LOC": "Survey/Localizer",
    }
    for key, seq_type in name_map.items():
        if key in desc_up:
            result["sequence_type"] = seq_type
            result["confidence"] = "high"
            result["reasoning"].append(f"Series name contains '{key}'")
            break

    # Refine with parameters
    if "DWI" in desc_up and b_value is not None:
        if b_value > 500:
            result["reasoning"].append(f"b-value={b_value} confirms diffusion weighting")
        elif b_value == 0:
            result["sequence_type"] = "DWI (b=0)"
            result["reasoning"].append("b=0 reference image")

    if "ADC" in desc_up:
        result["sequence_type"] = "ADC Map"
        result["reasoning"].append("Apparent Diffusion Coefficient map (derived)")

    if "ISO" in desc_up and "B" in desc_up:
        result["sequence_type"] = "Isotropic DWI"
        result["reasoning"].append("Isotropic trace-weighted diffusion image")

    if "REG" in desc_up and "DWI" in desc_up:
        result["sequence_type"] = "Registered DWI"
        result["reasoning"].append("Motion-corrected/registered diffusion series")

    # Parameter-based classification if still unknown
    if result["confidence"] == "low":
        if tr is not None and te is not None:
            if tr > 2000 and te > 80:
                result["sequence_type"] = "T2-weighted"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TR={tr}ms TE={te}ms → long TR + long TE")
            elif tr > 2000 and te < 30:
                result["sequence_type"] = "T1-weighted (or FLAIR)"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TR={tr}ms TE={te}ms → long TR + short TE")
            elif tr < 800 and te < 30:
                result["sequence_type"] = "T1-weighted"
                result["confidence"] = "medium"
                result["reasoning"].append(f"TR={tr}ms TE={te}ms → short TR + short TE")

        if ti is not None and ti > 1500:
            result["sequence_type"] = "FLAIR"
            result["confidence"] = "high"
            result["reasoning"].append(f"TI={ti}ms → CSF nulling inversion time")

    result["parameters"] = {
        "TR_ms": tr, "TE_ms": te, "TI_ms": ti,
        "flip_angle_deg": fa, "b_value": b_value,
    }
    return result


# ── DICOM conformance checks ────────────────────────────────────────────────

REQUIRED_MR_TAGS = [
    "PatientID", "PatientName", "PatientSex", "PatientBirthDate",
    "StudyDate", "StudyTime", "StudyDescription", "StudyInstanceUID",
    "SeriesNumber", "SeriesDescription", "SeriesInstanceUID",
    "Modality", "Manufacturer", "MagneticFieldStrength",
    "RepetitionTime", "EchoTime", "FlipAngle",
    "SliceThickness", "SpacingBetweenSlices", "PixelSpacing",
    "Rows", "Columns", "BitsAllocated",
    "ImageOrientationPatient", "ImagePositionPatient",
    "PhotometricInterpretation",
]


def check_conformance(records: list[dict]) -> list[dict]:
    """Check DICOM conformance — missing/empty required tags per file."""
    issues = []
    for r in records:
        missing = []
        for tag in REQUIRED_MR_TAGS:
            val = r.get(tag)
            if val is None or val == "" or val == "None":
                missing.append(tag)
        if missing:
            issues.append({
                "filename": r.get("_filename", "?"),
                "missing_tags": missing,
                "missing_count": len(missing),
                "completeness_pct": round(100 * (1 - len(missing) / len(REQUIRED_MR_TAGS)), 1),
            })
    return issues


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_all_tags(ds: pydicom.Dataset) -> dict:
    """Extract every non-pixel-data tag from a DICOM dataset."""
    record = {}
    for elem in ds:
        if elem.tag.group == 0x7FE0:  # All pixel data variants
            record["_has_pixel_data"] = True
            continue
        kw = elem.keyword or f"Tag_{elem.tag.group:04X}_{elem.tag.element:04X}"
        record[kw] = safe_value(elem)
    record.setdefault("_has_pixel_data", False)

    # File meta info (transfer syntax, etc.)
    if hasattr(ds, "file_meta"):
        fm = ds.file_meta
        ts_uid = str(getattr(fm, "TransferSyntaxUID", ""))
        record["_transfer_syntax_uid"] = ts_uid
        record["_transfer_syntax_name"] = TRANSFER_SYNTAX_NAMES.get(ts_uid, ts_uid)
        record["_media_storage_sop_class"] = str(getattr(fm, "MediaStorageSOPClassUID", ""))
        record["_implementation_class_uid"] = str(getattr(fm, "ImplementationClassUID", ""))

    # File size
    return record


def extract_pixel_stats(ds: pydicom.Dataset) -> dict:
    try:
        arr = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        offset = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + offset
        pcts = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        hist_counts, hist_edges = np.histogram(arr.ravel(), bins=50)
        return {
            "pixel_shape": list(arr.shape),
            "pixel_dtype": str(ds.pixel_array.dtype),
            "pixel_min": float(arr.min()),
            "pixel_max": float(arr.max()),
            "pixel_mean": float(arr.mean()),
            "pixel_std": float(arr.std()),
            "pixel_median": float(pcts[4]),
            "pixel_p1": float(pcts[0]), "pixel_p5": float(pcts[1]),
            "pixel_p10": float(pcts[2]), "pixel_p25": float(pcts[3]),
            "pixel_p50": float(pcts[4]), "pixel_p75": float(pcts[5]),
            "pixel_p90": float(pcts[6]), "pixel_p95": float(pcts[7]),
            "pixel_p99": float(pcts[8]),
            "pixel_iqr": float(pcts[5] - pcts[3]),
            "nonzero_ratio": float((arr != 0).mean()),
            "pixel_entropy": float(_entropy(arr)),
            "pixel_skewness": float(_skewness(arr)),
            "pixel_kurtosis": float(_kurtosis(arr)),
            "histogram_counts": hist_counts.tolist(),
            "histogram_edges": hist_edges.tolist(),
        }
    except Exception as e:
        return {"pixel_error": str(e)}


def extract_series_params(ds: pydicom.Dataset) -> dict:
    """Extract sequence-relevant params from a dataset."""
    def _f(attr):
        try:
            v = getattr(ds, attr, None)
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            pass
        return None

    return {
        "tr": _f("RepetitionTime"),
        "te": _f("EchoTime"),
        "ti": _f("InversionTime"),
        "fa": _f("FlipAngle"),
        "slice_thickness": _f("SliceThickness"),
        "spacing_between_slices": _f("SpacingBetweenSlices"),
        "rows": _f("Rows"),
        "columns": _f("Columns"),
        "field_strength": _f("MagneticFieldStrength"),
        "pixel_spacing": str(getattr(ds, "PixelSpacing", "")),
        "b_value": _f("DiffusionBValue"),
    }


# ── Volume loading ───────────────────────────────────────────────────────────

def load_series_as_volume(dcm_files: list[Path]) -> tuple[np.ndarray | None, sitk.Image | None]:
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(f) for f in dcm_files])
        image = reader.Execute()
        return sitk.GetArrayFromImage(image), image
    except Exception:
        return None, None


def load_volume_from_file(f: Path) -> tuple[np.ndarray | None, dict]:
    """Try loading pixel data from a single DICOM file."""
    try:
        ds = pydicom.dcmread(str(f), force=True)
        arr = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        offset = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + offset
        arr = _safe_squeeze(arr)
        return arr, extract_series_params(ds)
    except Exception:
        return None, {}


def volume_stats(vol: np.ndarray, sitk_img: sitk.Image | None = None) -> dict:
    ndim = vol.ndim
    if sitk_img:
        spacing = list(sitk_img.GetSpacing())
        origin = list(sitk_img.GetOrigin())
        direction = list(sitk_img.GetDirection())
    else:
        spacing = [1.0] * ndim
        origin = [0.0] * ndim
        direction = []

    voxel_vol_mm3 = float(np.prod(spacing[:3]))
    total_vol_mm3 = voxel_vol_mm3 * int(np.prod(vol.shape[:3]))
    total_vol_cc = total_vol_mm3 / 1000.0

    # FOV calculation (Z, Y, X shape × X, Y, Z spacing)
    sp3 = (spacing + [1.0, 1.0, 1.0])[:3]
    fov = [float(vol.shape[i]) * float(sp3[min(2 - i, len(sp3) - 1)]) for i in range(min(3, ndim))]

    pcts = np.percentile(vol, [1, 5, 25, 50, 75, 95, 99])
    threshold = float(np.percentile(vol, 15))
    tissue_mask = vol > threshold
    tissue_pct = float(tissue_mask.mean() * 100)

    # Per-slice SNR variation (useful for quality assessment)
    if ndim >= 3 and vol.shape[0] > 1:
        slice_means = [float(vol[i].mean()) for i in range(vol.shape[0])]
        slice_stds = [float(vol[i].std()) for i in range(vol.shape[0])]
        slice_snrs = [m / (s + 1e-6) for m, s in zip(slice_means, slice_stds)]
    else:
        slice_means = [float(vol.mean())]
        slice_stds = [float(vol.std())]
        slice_snrs = [float(vol.mean() / (vol.std() + 1e-6))]

    return {
        "volume_shape": list(vol.shape),
        "volume_voxel_count": int(np.prod(vol.shape)),
        "spacing_mm": sp3,
        "origin_mm": origin,
        "direction_cosines": direction,
        "voxel_volume_mm3": voxel_vol_mm3,
        "total_volume_mm3": total_vol_mm3,
        "total_volume_cc": total_vol_cc,
        "fov_mm": fov,
        "volume_min": float(vol.min()),
        "volume_max": float(vol.max()),
        "volume_mean": float(vol.mean()),
        "volume_std": float(vol.std()),
        "volume_median": float(pcts[3]),
        "volume_p1": float(pcts[0]), "volume_p5": float(pcts[1]),
        "volume_p25": float(pcts[2]), "volume_p75": float(pcts[4]),
        "volume_p95": float(pcts[5]), "volume_p99": float(pcts[6]),
        "volume_iqr": float(pcts[4] - pcts[2]),
        "volume_dynamic_range": float(vol.max() - vol.min()),
        "volume_snr_estimate": float(vol.mean() / (vol.std() + 1e-6)),
        "volume_nonzero_pct": float((vol != 0).mean() * 100),
        "volume_tissue_pct": tissue_pct,
        "volume_entropy": float(_entropy(vol)),
        "volume_skewness": float(_skewness(vol)),
        "volume_kurtosis": float(_kurtosis(vol)),
        "slice_snr_mean": float(np.mean(slice_snrs)),
        "slice_snr_std": float(np.std(slice_snrs)),
        "slice_snr_min": float(np.min(slice_snrs)),
        "slice_snr_max": float(np.max(slice_snrs)),
        "slice_intensity_uniformity": 1.0 - float(np.std(slice_means) / (np.mean(slice_means) + 1e-6)),
    }


# ── Export functions ─────────────────────────────────────────────────────────

def export_nifti(dcm_files: list[Path], out_path: Path) -> None:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(f) for f in dcm_files])
    image = reader.Execute()
    sitk.WriteImage(image, str(out_path))


def export_multiplane_montage(vol: np.ndarray, series_name: str, out_dir: Path,
                               n_per_plane: int = 6) -> Path:
    """Export a 3-plane montage: axial, sagittal, coronal."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vol = _safe_squeeze(vol)

    vmin, vmax = np.percentile(vol, [1, 99])
    vol_norm = np.clip((vol - vmin) / (vmax - vmin + 1e-6), 0, 1)

    if vol_norm.ndim < 3:
        vol_norm = vol_norm[np.newaxis, ...]

    nz, ny, nx = vol_norm.shape[:3]
    planes = {
        "Axial": lambda i: _get_2d_slice(vol_norm, i) if i < nz else np.zeros((ny, nx)),
        "Coronal": lambda i: vol_norm[:, min(i, ny - 1), :] if vol_norm.ndim >= 3 else vol_norm[0],
        "Sagittal": lambda i: vol_norm[:, :, min(i, nx - 1)] if vol_norm.ndim >= 3 else vol_norm[0],
    }
    plane_sizes = {"Axial": nz, "Coronal": ny, "Sagittal": nx}

    fig = plt.figure(figsize=(4 * n_per_plane, 4 * 3), facecolor="black")
    gs = gridspec.GridSpec(3, n_per_plane, figure=fig, hspace=0.05, wspace=0.02)

    for row, (plane_name, get_slice) in enumerate(planes.items()):
        n = plane_sizes[plane_name]
        indices = np.linspace(max(n // 8, 0), min(n - n // 8, n - 1), n_per_plane, dtype=int) if n > 1 else [0]
        for col, idx in enumerate(indices[:n_per_plane]):
            ax = fig.add_subplot(gs[row, col])
            slc = get_slice(idx)
            while slc.ndim > 2:
                slc = slc[0]
            ax.imshow(slc, cmap="gray", interpolation="bilinear", aspect="auto")
            if col == 0:
                ax.set_ylabel(plane_name, fontsize=10, color="cyan", fontweight="bold")
            ax.set_title(f"{idx}", fontsize=7, color="white", pad=2)
            ax.axis("off")

    p = out_dir / f"{series_name}_multiplane.png"
    fig.savefig(p, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return p


def export_histogram(vol: np.ndarray, series_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Linear scale
    ax1.hist(vol.ravel(), bins=200, color="#2196F3", alpha=0.8, edgecolor="none")
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Voxel Count")
    ax1.set_title(f"{series_name} — Linear")

    # Log scale
    ax2.hist(vol.ravel(), bins=200, color="#FF9800", alpha=0.8, edgecolor="none")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Voxel Count (log)")
    ax2.set_title(f"{series_name} — Log Scale")
    ax2.set_yscale("log")

    fig.tight_layout()
    p = out_dir / f"{series_name}_histogram.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def export_cross_series_comparison(series_data: dict[str, dict], out_dir: Path) -> Path:
    """Bar chart comparing key metrics across all image series."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    snrs, entropies, tissue_pcts, dyn_ranges = [], [], [], []

    for name, info in sorted(series_data.items()):
        vs = info.get("vstats")
        if vs is None:
            continue
        names.append(info.get("label", name[:20]))
        snrs.append(vs.get("volume_snr_estimate", 0))
        entropies.append(vs.get("volume_entropy", 0))
        tissue_pcts.append(vs.get("volume_tissue_pct", 0))
        dyn_ranges.append(vs.get("volume_dynamic_range", 0))

    if not names:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    x = np.arange(len(names))
    w = 0.6

    for ax, vals, title, color in [
        (axes[0, 0], snrs, "SNR Estimate", "#4CAF50"),
        (axes[0, 1], entropies, "Entropy (bits)", "#2196F3"),
        (axes[1, 0], tissue_pcts, "Tissue Coverage %", "#FF9800"),
        (axes[1, 1], dyn_ranges, "Dynamic Range", "#9C27B0"),
    ]:
        ax.barh(x, vals, w, color=color, alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.01, i, f"{v:.1f}", va="center", fontsize=7)

    fig.suptitle("Cross-Series Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "cross_series_comparison.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


# ── HTML Report ──────────────────────────────────────────────────────────────

def _img_to_b64(path: Path) -> str:
    if path and path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return ""


def generate_html_report(
    patient_info: dict,
    series_info: list[dict],
    conformance_issues: list[dict],
    image_paths: dict[str, dict],
    cross_series_path: Path | None,
    out_dir: Path,
) -> Path:
    """Generate a self-contained HTML report with embedded images."""

    parts = ["""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>DICOM Brain Study Report</title>
<style>
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:20px}
h1{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:10px}
h2{color:#79c0ff;margin-top:30px}
h3{color:#d2a8ff}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.metric{text-align:center;padding:12px;background:#0d1117;border-radius:6px}
.metric .val{font-size:1.8em;font-weight:bold;color:#58a6ff}
.metric .label{font-size:0.8em;color:#8b949e;margin-top:4px}
table{border-collapse:collapse;width:100%;margin:10px 0}
th{background:#21262d;color:#58a6ff;padding:8px 12px;text-align:left;font-size:0.85em}
td{padding:6px 12px;border-bottom:1px solid #21262d;font-size:0.85em}
tr:hover{background:#161b22}
img{max-width:100%;border-radius:6px;margin:8px 0}
.tag-high{color:#3fb950} .tag-med{color:#d29922} .tag-low{color:#f85149}
.seq-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:bold;margin:2px}
.seq-T1{background:#1a4731;color:#3fb950} .seq-T2{background:#1a3148;color:#58a6ff}
.seq-FLAIR{background:#3d1f00;color:#d29922} .seq-DWI{background:#3d1a3d;color:#d2a8ff}
.seq-ADC{background:#1a1a3d;color:#79c0ff} .seq-GRE{background:#3d1a1a;color:#f85149}
.seq-other{background:#21262d;color:#8b949e}
.warn{background:#3d2200;border-left:3px solid #d29922;padding:8px 12px;margin:8px 0;border-radius:4px}
.ok{background:#0d2818;border-left:3px solid #3fb950;padding:8px 12px;margin:8px 0;border-radius:4px}
</style></head><body>"""]

    # Header
    p = patient_info
    parts.append(f"""
<h1>DICOM Brain Study Report</h1>
<div class="card"><div class="grid">
<div class="metric"><div class="val">{html_mod.escape(p.get('patient_name','?'))}</div><div class="label">Patient</div></div>
<div class="metric"><div class="val">{html_mod.escape(p.get('patient_sex','?'))}</div><div class="label">Sex</div></div>
<div class="metric"><div class="val">{html_mod.escape(p.get('study_date','?'))}</div><div class="label">Study Date</div></div>
<div class="metric"><div class="val">{html_mod.escape(p.get('manufacturer','?'))} {html_mod.escape(p.get('model',''))}</div><div class="label">Scanner</div></div>
<div class="metric"><div class="val">{html_mod.escape(p.get('field_strength','?'))}T</div><div class="label">Field Strength</div></div>
<div class="metric"><div class="val">{html_mod.escape(p.get('institution','?'))}</div><div class="label">Institution</div></div>
</div></div>""")

    # Overview metrics
    n_series = len(series_info)
    n_image = sum(1 for s in series_info if s.get("has_pixels"))
    n_files = sum(s.get("file_count", 0) for s in series_info)
    parts.append(f"""
<h2>Study Overview</h2>
<div class="card"><div class="grid">
<div class="metric"><div class="val">{n_files}</div><div class="label">Total Files</div></div>
<div class="metric"><div class="val">{n_series}</div><div class="label">Total Series</div></div>
<div class="metric"><div class="val">{n_image}</div><div class="label">Image Series</div></div>
<div class="metric"><div class="val">{n_series - n_image}</div><div class="label">Non-Image (PS)</div></div>
</div></div>""")

    # Cross-series comparison
    if cross_series_path and cross_series_path.exists():
        b64 = _img_to_b64(cross_series_path)
        parts.append(f'<h2>Cross-Series Comparison</h2><div class="card"><img src="data:image/png;base64,{b64}"></div>')

    # Per-series detail
    parts.append("<h2>Series Detail</h2>")
    for s in series_info:
        snum = s.get("series_number", "?")
        desc = s.get("series_description", "?")
        seq_class = s.get("sequence_classification", {})
        seq_type = seq_class.get("sequence_type", "Unknown")
        confidence = seq_class.get("confidence", "low")

        # Badge color
        badge_cls = "seq-other"
        for key, cls in [("T1", "seq-T1"), ("T2", "seq-T2"), ("FLAIR", "seq-FLAIR"),
                          ("DWI", "seq-DWI"), ("ADC", "seq-ADC"), ("GRE", "seq-GRE")]:
            if key in seq_type.upper():
                badge_cls = cls
                break
        conf_cls = {"high": "tag-high", "medium": "tag-med", "low": "tag-low"}.get(confidence, "tag-low")

        parts.append(f"""
<div class="card">
<h3>Series {snum} — {html_mod.escape(desc)}
<span class="seq-badge {badge_cls}">{html_mod.escape(seq_type)}</span>
<span class="{conf_cls}" style="font-size:0.7em">({confidence})</span>
</h3>""")

        # Parameters table
        params = s.get("sequence_params", {})
        vs = s.get("volume_stats", {})
        if params or vs:
            parts.append('<table><tr><th>Parameter</th><th>Value</th><th>Metric</th><th>Value</th></tr>')
            param_rows = [(k, v) for k, v in params.items() if v is not None and v != "" and v != "None"]
            metric_rows = [
                ("Shape", vs.get("volume_shape", "")),
                ("SNR", f"{vs.get('volume_snr_estimate', 0):.2f}"),
                ("Entropy", f"{vs.get('volume_entropy', 0):.1f} bits"),
                ("Tissue %", f"{vs.get('volume_tissue_pct', 0):.1f}%"),
                ("Dynamic Range", f"{vs.get('volume_dynamic_range', 0):.0f}"),
                ("Uniformity", f"{vs.get('slice_intensity_uniformity', 0):.3f}"),
                ("Volume (cc)", f"{vs.get('total_volume_cc', 0):.1f}"),
            ]
            max_rows = max(len(param_rows), len(metric_rows))
            for i in range(max_rows):
                pk, pv = param_rows[i] if i < len(param_rows) else ("", "")
                mk, mv = metric_rows[i] if i < len(metric_rows) else ("", "")
                parts.append(f"<tr><td>{pk}</td><td>{pv}</td><td>{mk}</td><td>{mv}</td></tr>")
            parts.append("</table>")

        # Reasoning
        reasoning = seq_class.get("reasoning", [])
        if reasoning:
            parts.append('<div style="font-size:0.85em;color:#8b949e;margin:4px 0">')
            parts.append(" | ".join(reasoning))
            parts.append("</div>")

        # Images
        img_info = image_paths.get(str(snum) + "_" + desc, {})
        mp = img_info.get("montage")
        if mp:
            b64 = _img_to_b64(mp)
            if b64:
                parts.append(f'<img src="data:image/png;base64,{b64}">')
        hp = img_info.get("histogram")
        if hp:
            b64 = _img_to_b64(hp)
            if b64:
                parts.append(f'<img src="data:image/png;base64,{b64}" style="max-height:250px">')

        parts.append("</div>")

    # Conformance
    parts.append("<h2>DICOM Conformance Check</h2>")
    if conformance_issues:
        worst = min(i["completeness_pct"] for i in conformance_issues)
        best = max(i["completeness_pct"] for i in conformance_issues)
        parts.append(f'<div class="warn">Tag completeness ranges from {worst}% to {best}% across files.</div>')
        parts.append('<div class="card"><table><tr><th>File</th><th>Missing Tags</th><th>Completeness</th></tr>')
        for issue in sorted(conformance_issues, key=lambda x: x["completeness_pct"])[:20]:
            missing_str = ", ".join(issue["missing_tags"][:8])
            if len(issue["missing_tags"]) > 8:
                missing_str += f" (+{len(issue['missing_tags']) - 8} more)"
            parts.append(f'<tr><td>{html_mod.escape(issue["filename"])}</td>'
                        f'<td style="font-size:0.8em">{missing_str}</td>'
                        f'<td>{issue["completeness_pct"]}%</td></tr>')
        parts.append("</table></div>")
    else:
        parts.append('<div class="ok">All files pass DICOM MR conformance checks.</div>')

    parts.append(f"""
<div style="text-align:center;color:#484f58;margin-top:40px;font-size:0.8em">
Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by dcm_extract.py
</div></body></html>""")

    html_path = out_dir / "report.html"
    html_path.write_text("\n".join(parts))
    return html_path


# ── Series grouping ──────────────────────────────────────────────────────────

def group_files_by_series(dcm_files: list[Path]) -> tuple[dict[str, list[Path]], dict[str, dict]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    series_meta: dict[str, dict] = {}

    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
            groups[uid].append(f)
            if uid not in series_meta:
                series_meta[uid] = {
                    "series_number": getattr(ds, "SeriesNumber", ""),
                    "series_description": str(getattr(ds, "SeriesDescription", "")),
                    "modality": str(getattr(ds, "Modality", "")),
                    "sop_class_uid": str(getattr(ds, "SOPClassUID", "")),
                }
        except Exception:
            groups["_unreadable"].append(f)

    for uid in groups:
        groups[uid] = sorted(groups[uid])
    return groups, series_meta


# ── Rich display ─────────────────────────────────────────────────────────────

def render_series_tree(series_info: list[dict]) -> None:
    tree = Tree("[bold cyan]DICOM Series[/bold cyan]")
    for s in sorted(series_info, key=lambda x: str(x.get("series_number", ""))):
        snum = s.get("series_number", "?")
        desc = s.get("series_description", "?")
        seq = s.get("sequence_classification", {})
        seq_type = seq.get("sequence_type", "")
        conf = seq.get("confidence", "")
        vs = s.get("volume_stats", {})

        label = f"[bold]{snum}[/bold] — {desc}"
        if seq_type:
            label += f"  [magenta]({seq_type})[/magenta]"
        branch = tree.add(label)

        mod = s.get("modality", "")
        n = s.get("file_count", 0)
        sop = s.get("sop_class", "")
        branch.add(f"[dim]{mod}[/dim]  |  {n} files  |  {sop}  |  confidence: {conf}")

        if vs:
            shape = vs.get("volume_shape", "?")
            sp = vs.get("spacing_mm", [])
            sp_str = f"[{', '.join(f'{x:.2f}' for x in sp[:3])}]" if sp else "?"
            branch.add(
                f"[dim]Shape:[/dim] {shape}  |  [dim]Spacing:[/dim] {sp_str} mm  |  "
                f"[dim]SNR:[/dim] {vs.get('volume_snr_estimate', 0):.2f}  |  "
                f"[dim]Tissue:[/dim] {vs.get('volume_tissue_pct', 0):.1f}%  |  "
                f"[dim]Uniformity:[/dim] {vs.get('slice_intensity_uniformity', 0):.3f}"
            )
        elif not s.get("has_pixels"):
            branch.add("[yellow]Presentation state (no pixel data)[/yellow]")

    console.print(tree)


def render_protocol_table(series_info: list[dict]) -> None:
    """Display a summary protocol table."""
    table = Table(
        title="MRI Protocol Summary",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#", style="bold", width=5)
    table.add_column("Description", width=18)
    table.add_column("Sequence", style="magenta", width=16)
    table.add_column("TR", width=7)
    table.add_column("TE", width=7)
    table.add_column("TI", width=7)
    table.add_column("FA", width=5)
    table.add_column("Matrix", width=14)
    table.add_column("Slices", width=6)
    table.add_column("SNR", width=6)

    for s in sorted(series_info, key=lambda x: str(x.get("series_number", ""))):
        if not s.get("has_pixels"):
            continue
        p = s.get("sequence_params", {})
        vs = s.get("volume_stats", {})
        shape = vs.get("volume_shape", [])
        seq_type = s.get("sequence_classification", {}).get("sequence_type", "?")
        table.add_row(
            str(s.get("series_number", "")),
            s.get("series_description", "")[:18],
            seq_type[:16],
            f"{p.get('tr', '')}" if p.get("tr") else "",
            f"{p.get('te', '')}" if p.get("te") else "",
            f"{p.get('ti', '')}" if p.get("ti") else "",
            f"{p.get('fa', '')}" if p.get("fa") else "",
            f"{shape[1]}×{shape[2]}" if len(shape) >= 3 else str(shape),
            str(shape[0]) if shape else "",
            f"{vs.get('volume_snr_estimate', 0):.2f}" if vs else "",
        )
    console.print(table)


# ── Claude ───────────────────────────────────────────────────────────────────

def ask_llm(payload: dict, montage_paths: list[Path], model: str = "gpt-5.4-nano") -> str:
    """Send extracted data + montage images to OpenAI for expert analysis."""
    client = openai.OpenAI()

    content: list[dict] = []

    # Add montage images (OpenAI vision format)
    for mp in montage_paths[:6]:
        if mp and mp.exists():
            b64 = base64.b64encode(mp.read_bytes()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })

    prompt = f"""You are a neuroradiology and medical imaging AI expert.

Below is comprehensively extracted metadata and statistics from a DICOM brain MRI study.
Montage images (3-plane: axial, coronal, sagittal) are included for each image series.

Provide a detailed structured analysis:

1. **Patient & Study Summary**
2. **Series-by-Series Protocol Analysis** — sequence type, justification, resolution, FOV
3. **Image Quality Assessment** — SNR, uniformity, artifacts, tissue coverage
4. **3D Geometry** — voxel sizes, orientations, coordinate system
5. **Clinical Relevance** — structures/pathologies assessable per sequence
6. **ML Suitability** — segmentation/classification readiness, preprocessing needed
7. **Data Completeness** — missing tags and their impact
8. **Red Flags** — anomalies in stats, geometry, metadata
9. **Recommendations** — next steps

```json
{json.dumps(payload, indent=2, default=str)}
```"""

    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


# ── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    folder: Annotated[Path, typer.Argument(help="Folder containing .dcm files")],
    analyze: Annotated[bool, typer.Option("--analyze", help="Send to gpt-5.4-nano for expert analysis")] = False,
    export_nii: Annotated[bool, typer.Option("--export-nii", help="Export each series to NIfTI")] = False,
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Output directory")] = Path("output"),
):
    """Extract everything from a DICOM brain study."""
    t0 = time.time()

    console.print(Panel.fit(
        "[bold cyan]DICOM Brain Extractor v3[/bold cyan]\n"
        "[dim]All tags · sequence classification · multi-plane montages · conformance checks · HTML report[/dim]",
        border_style="cyan",
    ))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover + group ──────────────────────────────────────────────────
    dcm_files = sorted(folder.glob("*.dcm"))
    if not dcm_files:
        console.print("[red]No .dcm files found.[/red]")
        raise typer.Exit(1)
    console.print(f"Found [bold]{len(dcm_files)}[/bold] DICOM files in [dim]{folder}[/dim]")

    groups, series_meta = group_files_by_series(dcm_files)
    console.print(f"Grouped into [bold]{len(groups)}[/bold] series\n")

    # ── 2. Extract ALL tags ──────────────────────────────────────────────────
    all_records: list[dict] = []
    all_tags_seen: set[str] = set()
    patient_info: dict = {}

    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Extracting DICOM tags", total=len(dcm_files))
        for f in dcm_files:
            ds = pydicom.dcmread(str(f), force=True)
            record = {"_filename": f.name, "_file_size_bytes": f.stat().st_size}
            tags = extract_all_tags(ds)
            record.update(tags)
            all_tags_seen.update(tags.keys())

            if record.get("_has_pixel_data"):
                record.update(extract_pixel_stats(ds))

            if not patient_info:
                patient_info = {
                    "patient_id": str(getattr(ds, "PatientID", "")),
                    "patient_name": str(getattr(ds, "PatientName", "")),
                    "patient_sex": str(getattr(ds, "PatientSex", "")),
                    "patient_birth_date": str(getattr(ds, "PatientBirthDate", "")),
                    "patient_weight": str(getattr(ds, "PatientWeight", "")),
                    "study_date": str(getattr(ds, "StudyDate", "")),
                    "study_description": str(getattr(ds, "StudyDescription", "")),
                    "institution": str(getattr(ds, "InstitutionName", "")),
                    "manufacturer": str(getattr(ds, "Manufacturer", "")),
                    "model": str(getattr(ds, "ManufacturerModelName", "")),
                    "field_strength": str(getattr(ds, "MagneticFieldStrength", "")),
                    "software_versions": str(getattr(ds, "SoftwareVersions", "")),
                    "station_name": str(getattr(ds, "StationName", "")),
                }

            all_records.append(record)
            progress.advance(task)

    console.print(f"Extracted [bold]{len(all_tags_seen)}[/bold] unique tags across all files\n")

    # ── 3. DICOM Conformance ─────────────────────────────────────────────────
    conformance_issues = check_conformance(all_records)
    if conformance_issues:
        n_issues = len(conformance_issues)
        avg_pct = sum(i["completeness_pct"] for i in conformance_issues) / n_issues
        console.print(f"[yellow]⚠ Conformance:[/yellow] {n_issues}/{len(all_records)} files have missing tags "
                      f"(avg {avg_pct:.0f}% complete)")
    else:
        console.print("[green]✓[/green] All files pass DICOM MR conformance checks")

    # ── 4. Save JSON + CSV ───────────────────────────────────────────────────
    json_records = [{k: _to_json(v) for k, v in r.items()} for r in all_records]
    json_path = out_dir / "dicom_full_dump.json"
    json_path.write_text(json.dumps(json_records, indent=2, default=str))
    console.print(f"[green]✓[/green] JSON → {json_path}")

    csv_records = []
    for r in all_records:
        flat = {}
        for k, v in r.items():
            if k in ("histogram_counts", "histogram_edges"):
                continue
            flat[k] = str(v) if isinstance(v, list) else _to_json(v)
        csv_records.append(flat)
    df = pl.DataFrame(csv_records, infer_schema_length=None)
    csv_path = out_dir / "dicom_metadata.csv"
    df.write_csv(csv_path)
    console.print(f"[green]✓[/green] CSV  → {csv_path} ({len(df)} rows × {len(df.columns)} cols)")

    # ── 5. Per-series processing ─────────────────────────────────────────────
    series_info: list[dict] = []
    series_data_for_comparison: dict[str, dict] = {}
    all_montage_paths: list[Path] = []
    image_paths: dict[str, dict] = {}  # for HTML report

    console.print("\n[bold cyan]Processing series…[/bold cyan]")

    for uid, files in groups.items():
        meta = series_meta.get(uid, {})
        desc = meta.get("series_description", "unknown")
        snum = meta.get("series_number", "?")
        sop_uid = meta.get("sop_class_uid", "")
        safe_name = f"s{snum}_{desc}".replace(" ", "_").replace("/", "-").replace("*", "x")
        key = f"{snum}_{desc}"

        is_img = sop_uid not in NON_IMAGE_SOP

        info: dict = {
            "series_uid": uid,
            "series_number": snum,
            "series_description": desc,
            "modality": meta.get("modality", ""),
            "sop_class": SOP_CLASS_NAMES.get(sop_uid, sop_uid),
            "sop_class_uid": sop_uid,
            "file_count": len(files),
            "has_pixels": is_img,
        }

        if not is_img:
            info["note"] = "Presentation state — skipped"
            series_info.append(info)
            continue

        # Get sequence parameters from first available file
        seq_params = {}
        for f in files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                seq_params = extract_series_params(ds)
                if seq_params.get("tr") or seq_params.get("te"):
                    break
            except Exception:
                pass
        info["sequence_params"] = seq_params

        # Classify sequence
        info["sequence_classification"] = classify_sequence(
            desc,
            seq_params.get("tr"), seq_params.get("te"),
            seq_params.get("ti"), seq_params.get("fa"),
            seq_params.get("b_value"),
        )

        # Load volume
        vol, sitk_img = load_series_as_volume(files)
        if vol is not None:
            vol = _safe_squeeze(vol)
        else:
            # Fallback: load from individual files
            for f in files:
                vol, _ = load_volume_from_file(f)
                if vol is not None:
                    break

        if vol is not None and vol.size > 0:
            vs = volume_stats(vol, sitk_img)
            info["volume_stats"] = vs
            console.print(f"  [cyan]{snum}[/cyan] {desc}: {vol.shape}  "
                         f"SNR={vs['volume_snr_estimate']:.2f}  "
                         f"Tissue={vs['volume_tissue_pct']:.0f}%  "
                         f"Uniformity={vs['slice_intensity_uniformity']:.3f}")

            series_data_for_comparison[uid] = {
                "label": f"{snum} {desc}"[:25],
                "vstats": vs,
            }

            # Multi-plane montage
            montage_path = export_multiplane_montage(vol, safe_name, out_dir / "montages")
            all_montage_paths.append(montage_path)

            # Histogram
            hist_path = export_histogram(vol, safe_name, out_dir / "histograms")

            image_paths[key] = {"montage": montage_path, "histogram": hist_path}
            console.print(f"    [green]✓[/green] Montage + histogram exported")

            # NIfTI
            if export_nii and sitk_img:
                nii_path = out_dir / "nifti" / f"{safe_name}.nii.gz"
                nii_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    export_nifti(files, nii_path)
                    console.print(f"    [green]✓[/green] NIfTI → {nii_path}")
                except Exception as e:
                    console.print(f"    [yellow]⚠ NIfTI failed: {e}[/yellow]")
        else:
            console.print(f"  [yellow]{snum}[/yellow] {desc}: no pixel data")

        series_info.append(info)

    console.print()

    # ── 6. Cross-series comparison ───────────────────────────────────────────
    cross_path = export_cross_series_comparison(series_data_for_comparison, out_dir)
    if cross_path:
        console.print(f"[green]✓[/green] Cross-series comparison → {cross_path}")

    # ── 7. Display ───────────────────────────────────────────────────────────
    render_series_tree(series_info)
    console.print()
    render_protocol_table(series_info)

    # Patient info
    if patient_info:
        pt = Table("Field", "Value", title="Patient & Study", box=box.SIMPLE_HEAD)
        for k, v in patient_info.items():
            if v and v != "None":
                pt.add_row(k.replace("_", " ").title(), str(v))
        console.print(pt)

    # ── 8. Save stats JSON ───────────────────────────────────────────────────
    stats_path = out_dir / "series_stats.json"
    stats_path.write_text(json.dumps({
        "patient": patient_info,
        "series": series_info,
        "conformance_issues": conformance_issues,
    }, indent=2, default=str))
    console.print(f"\n[green]✓[/green] Stats → {stats_path}")

    # ── 9. HTML Report ───────────────────────────────────────────────────────
    html_path = generate_html_report(
        patient_info, series_info, conformance_issues,
        image_paths, cross_path, out_dir,
    )
    console.print(f"[green]✓[/green] HTML report → {html_path}")

    # ── 10. Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    console.print(Panel(
        f"[bold]Extraction complete[/bold] in {elapsed:.1f}s\n"
        f"  Files:        {len(dcm_files)}\n"
        f"  Series:       {len(groups)} ({sum(1 for s in series_info if s.get('has_pixels'))} image)\n"
        f"  Unique tags:  {len(all_tags_seen)}\n"
        f"  Conformance:  {len(conformance_issues)} issues\n"
        f"  Output:       {out_dir.resolve()}\n"
        f"  HTML report:  {html_path.name}",
        title="Summary", border_style="green",
    ))

    # ── 11. Claude analysis ──────────────────────────────────────────────────
    if analyze:
        if not os.environ.get("OPENAI_API_KEY"):
            console.print("\n[yellow]⚠ OPENAI_API_KEY not set — skipping AI analysis[/yellow]")
            console.print("[dim]  Set it with: export OPENAI_API_KEY=sk-...[/dim]")
        else:
            console.print("\n[cyan]Sending to gpt-5.4-nano for expert analysis…[/cyan]\n")
            sample_tags = {}
            for r in all_records:
                if r.get("_has_pixel_data"):
                    sample_tags = {k: _to_json(v) for k, v in r.items()
                                  if not k.startswith("histogram_") and not k.startswith("_")}
                    break
            payload = {
                "patient": patient_info,
                "series": series_info,
                "conformance_issues_count": len(conformance_issues),
                "sample_full_tags": sample_tags,
            }
            try:
                analysis = ask_llm(payload, all_montage_paths)
                console.print(Panel(analysis, title="[bold]gpt-5.4-nano Analysis[/bold]",
                                    border_style="green", padding=(1, 2)))
                (out_dir / "ai_analysis.md").write_text(f"# DICOM Brain Study Analysis\n\n{analysis}\n")
                console.print(f"[green]✓[/green] Analysis saved → {out_dir / 'ai_analysis.md'}")
            except Exception as e:
                console.print(f"[red]OpenAI error:[/red] {e}")


if __name__ == "__main__":
    app()

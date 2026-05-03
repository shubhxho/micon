"""Pixel PHI scanner — OCR on slice corners for burned-in text.

Uses Tesseract OCR to check corner regions of each slice for:
  - Patient names
  - MRN/accession numbers
  - Dates
  - Institution names

Only checks corner regions (top-left, top-right, bottom-left, bottom-right)
where burned-in text typically appears. Does NOT OCR the entire image.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PixelPHIFinding:
    filename: str
    slice_index: int
    corner: str  # "top_left", "top_right", "bottom_left", "bottom_right"
    detected_text: str
    confidence: float


@dataclass
class PixelPHIReport:
    total_files: int = 0
    files_scanned: int = 0
    files_with_text: int = 0
    findings: list[PixelPHIFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.findings) == 0


# Corner region: top/bottom 12% × left/right 40% of the image
CORNER_FRAC_Y = 0.12
CORNER_FRAC_X = 0.40


def _extract_corners(pixel_array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Extract the 4 corner regions from a 2D slice."""
    h, w = pixel_array.shape[:2]
    cy = max(int(h * CORNER_FRAC_Y), 10)
    cx = max(int(w * CORNER_FRAC_X), 10)

    return [
        ("top_left", pixel_array[:cy, :cx]),
        ("top_right", pixel_array[:cy, w - cx :]),
        ("bottom_left", pixel_array[h - cy :, :cx]),
        ("bottom_right", pixel_array[h - cy :, w - cx :]),
    ]


def _ocr_region(region: np.ndarray) -> list[tuple[str, float]]:
    """Run OCR on a corner region. Returns [(text, confidence), ...]."""
    try:
        import pytesseract
        from PIL import Image

        # Normalize to uint8 for OCR
        if region.dtype != np.uint8:
            mn, mx = region.min(), region.max()
            if mx > mn:
                region = ((region - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                return []

        img = Image.fromarray(region)
        # Use OSD for orientation + script detection
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        results = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            conf = float(data["conf"][i])
            if text and len(text) >= 2 and conf > 30:
                results.append((text, conf))

        return results
    except Exception:
        return []


def _is_phi_text(text: str) -> bool:
    """Check if OCR'd text looks like PHI (not just noise)."""
    # At least 3 alphanumeric chars
    alpha = re.sub(r"[^a-zA-Z0-9]", "", text)
    if len(alpha) < 3:
        return False

    # Check against PHI patterns
    phi_patterns = [
        re.compile(r"[A-Z][a-z]{2,}"),  # capitalized name
        re.compile(r"\d{2}[-/]\d{2}[-/]\d{2,4}"),  # date
        re.compile(r"\b\d{4,}\b"),  # MRN-like number
        re.compile(r"(?:Dr|Mr|Mrs|Ms|Prof)", re.IGNORECASE),
        re.compile(r"(?:hosp|clinic|med|inst)", re.IGNORECASE),
    ]
    return any(p.search(text) for p in phi_patterns)


def scan_file_pixels(filepath: str, max_slices: int = 5) -> list[PixelPHIFinding]:
    """Scan corner regions of a DICOM file for burned-in text."""
    import pydicom

    findings = []
    filename = Path(filepath).name

    try:
        ds = pydicom.dcmread(filepath, force=True)
        if not hasattr(ds, "pixel_array"):
            return findings
        arr = ds.pixel_array
    except Exception:
        return findings

    # Handle 3D volumes — check first, middle, last slices
    if arr.ndim >= 3:
        n_slices = arr.shape[0]
        indices = [0, n_slices // 2, n_slices - 1]
        if n_slices > 4:
            indices = [0, n_slices // 4, n_slices // 2, 3 * n_slices // 4, n_slices - 1]
        slices = [(i, arr[i]) for i in indices[:max_slices]]
    else:
        slices = [(0, arr)]

    for slice_idx, slc in slices:
        while slc.ndim > 2:
            slc = slc[..., 0]

        for corner_name, corner_region in _extract_corners(slc):
            texts = _ocr_region(corner_region)
            for text, conf in texts:
                if _is_phi_text(text):
                    findings.append(
                        PixelPHIFinding(
                            filename=filename,
                            slice_index=slice_idx,
                            corner=corner_name,
                            detected_text=text,
                            confidence=conf,
                        )
                    )

    return findings


def scan_pixel_phi(
    file_paths: list[str],
    n_workers: int = 4,
    max_slices: int = 5,
) -> PixelPHIReport:
    """Scan all files for burned-in pixel PHI."""
    report = PixelPHIReport(total_files=len(file_paths))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(scan_file_pixels, fp, max_slices): fp for fp in file_paths}
        for fut in as_completed(futures):
            report.files_scanned += 1
            findings = fut.result()
            if findings:
                report.files_with_text += 1
                report.findings.extend(findings)

    return report

"""PHI scanner — regex sweep over every DICOM header value.

Detects:
  - SSN patterns (US: XXX-XX-XXXX)
  - Phone numbers (US + Indian: +91XXXXXXXXXX, XXX-XXX-XXXX)
  - Email addresses
  - Name patterns (Dr./Prof. + capitalized words)
  - Indian Aadhaar numbers (XXXX XXXX XXXX)
  - Indian PAN numbers (ABCDE1234F)
  - Date patterns in text fields
  - Institution/hospital names

Returns per-file findings with tag, matched pattern, and matched text.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pydicom

# ── Patterns ────────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, re.Pattern]] = [
    # ── US identifiers (HIPAA) ──
    ("ssn", re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
    ("phone_us", re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b')),

    # ── Indian government IDs (DPDPA + Aadhaar Act §29) ──
    ("aadhaar", re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')),
    ("pan", re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')),
    ("voter_id", re.compile(r'\b[A-Z]{3}\d{7}\b')),
    ("driving_license", re.compile(r'\b[A-Z]{2}[-/]\d{10,13}\b')),
    ("passport", re.compile(r'\b[A-Z]\d{7}\b')),
    ("uhid", re.compile(r'\b(?:UHID|MRN|HRN|HOSP)[-:/]?\s*\d{4,12}\b', re.IGNORECASE)),

    # ── Indian phone numbers ──
    ("phone_india", re.compile(r'(?<!\d)(?:\+91[\s-]?|0)\d{10}(?!\d)')),
    ("phone_landline", re.compile(r'\b0\d{2,4}[-\s]?\d{6,8}\b')),

    # ── Names (Indian + general) ──
    ("name_indian", re.compile(
        r'\b(?:Shri|Smt|Kumari?|Sri|Thiru|Selvi|Dr|Prof|Mr|Mrs|Ms)\.?\s+'
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b'
    )),
    ("surname_indian", re.compile(
        r'\b(?:Kumar|Singh|Sharma|Verma|Gupta|Patel|Shah|Jain|Mishra|Pandey'
        r'|Reddy|Rao|Nair|Menon|Pillai|Iyer|Iyengar|Mukherjee|Banerjee'
        r'|Chatterjee|Das|Ghosh|Roy|Sinha|Thakur|Yadav|Tiwari|Dubey'
        r'|Agarwal|Joshi|Kulkarni|Patil|Kaur|Gill|Bhatia|Fernandes'
        r"|D'Souza|George|Thomas|Krishnan|Naidu|Choudhury)\b",
        re.IGNORECASE,
    )),

    # ── Contact info ──
    ("email", re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')),
    ("url", re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)),

    # ── Institution ──
    ("institution", re.compile(r'\b[A-Z][a-z]{2,}\s+(?:Hospital|Clinic|Institute|Centre|Center|University|Diagnostics?|Imaging)\b', re.IGNORECASE)),

    # ── Dates / timestamps ──
    ("date_text", re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')),
    ("date_named", re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE)),
    ("timestamp_ist", re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:IST|UTC|GMT)\b', re.IGNORECASE)),

    # ── MRN / accession ──
    ("mrn_like", re.compile(r'\b[A-Z]{2,4}\d{6,12}\b')),
]

# Tags to skip (binary data, known non-PHI equipment/protocol tags)
_SKIP_TAGS = frozenset({
    "PixelData", "OverlayData", "WaveformData",
    "SpectroscopyData", "EncapsulatedDocument",
    # Equipment tags — contain vendor names like "GE MEDICAL" that aren't PHI
    "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
    "ImplementationClassUID", "ImplementationVersionName",
    "TransferSyntaxUID", "SOPClassUID", "MediaStorageSOPClassUID",
    # UIDs — long numeric strings that false-positive as phone numbers
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
    "FrameOfReferenceUID", "MediaStorageSOPInstanceUID",
    "ReferencedSOPInstanceUID", "RelatedFrameOfReferenceUID",
    # Protocol/sequence tags — contain technical terms not PHI
    "SequenceName", "ScanningSequence", "SequenceVariant",
    "ImageType", "ScanOptions", "MRAcquisitionType",
    "BodyPartExamined", "Modality", "PhotometricInterpretation",
    # Spatial/geometric tags — numeric arrays that false-positive as phone numbers
    "ImageOrientationPatient", "ImagePositionPatient",
    "PixelSpacing", "SliceLocation", "SliceThickness",
    "SpacingBetweenSlices", "WindowCenter", "WindowWidth",
    "RescaleIntercept", "RescaleSlope",
    "Rows", "Columns", "BitsAllocated", "BitsStored", "HighBit",
    "SmallestImagePixelValue", "LargestImagePixelValue",
    "NumberOfFrames", "SamplesPerPixel",
})


@dataclass
class PHIMatch:
    """A single PHI pattern match in a DICOM tag."""
    filename: str
    tag_keyword: str
    pattern_name: str
    matched_text: str
    tag_value_preview: str  # first 100 chars of the full value


@dataclass
class FileScanResult:
    """Result of scanning one file."""
    filepath: str
    matches: list[PHIMatch] = field(default_factory=list)
    error: str | None = None

    @property
    def clean(self) -> bool:
        return not self.matches and not self.error


@dataclass
class ScanReport:
    """Aggregate scan report."""
    total_files: int = 0
    clean_files: int = 0
    dirty_files: int = 0
    failed_files: int = 0
    total_matches: int = 0
    matches_by_pattern: dict[str, int] = field(default_factory=dict)
    dirty_file_details: list[FileScanResult] = field(default_factory=list)


def scan_file(filepath: str) -> FileScanResult:
    """Scan a single DICOM file for PHI patterns in ALL tag values."""
    result = FileScanResult(filepath=filepath)
    filename = Path(filepath).name

    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
    except Exception as e:
        result.error = str(e)
        return result

    for elem in ds:
        kw = elem.keyword or f"Tag_{elem.tag.group:04X}_{elem.tag.element:04X}"
        if kw in _SKIP_TAGS:
            continue

        try:
            val = str(elem.value)
        except Exception:
            continue

        if not val or len(val) < 3:
            continue

        for pattern_name, pattern in _PATTERNS:
            for match in pattern.finditer(val):
                result.matches.append(PHIMatch(
                    filename=filename,
                    tag_keyword=kw,
                    pattern_name=pattern_name,
                    matched_text=match.group(),
                    tag_value_preview=val[:100],
                ))

    return result


def scan_files(
    file_paths: list[str],
    n_workers: int = 8,
    on_progress=None,
) -> ScanReport:
    """Scan all files for PHI patterns. Returns aggregate report."""
    report = ScanReport(total_files=len(file_paths))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(scan_file, fp): fp for fp in file_paths}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            done += 1

            if result.error:
                report.failed_files += 1
            elif result.clean:
                report.clean_files += 1
            else:
                report.dirty_files += 1
                report.dirty_file_details.append(result)
                for m in result.matches:
                    report.total_matches += 1
                    report.matches_by_pattern[m.pattern_name] = (
                        report.matches_by_pattern.get(m.pattern_name, 0) + 1
                    )

            if on_progress:
                on_progress(done, len(file_paths))

    return report


def scan_report_to_dict(report: ScanReport) -> dict:
    """Serialize scan report to JSON-safe dict."""
    return {
        "total_files": report.total_files,
        "clean_files": report.clean_files,
        "dirty_files": report.dirty_files,
        "failed_files": report.failed_files,
        "total_matches": report.total_matches,
        "matches_by_pattern": report.matches_by_pattern,
        "passed": report.dirty_files == 0 and report.failed_files == 0,
        "dirty_files_detail": [
            {
                "file": r.filepath,
                "matches": [
                    {"tag": m.tag_keyword, "pattern": m.pattern_name, "text": m.matched_text}
                    for m in r.matches
                ],
            }
            for r in report.dirty_file_details[:50]  # cap at 50 for readability
        ],
    }

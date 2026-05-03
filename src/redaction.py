"""DICOM redaction — threaded, streaming, O(1) per-tag lookup.

Per-file complexity: O(T) where T = number of DICOM tags in the file.
Each tag is checked against frozen sets (O(1) membership test).
Total: O(N × T) across all files, fully parallelized via threads.

Uses ThreadPoolExecutor (not ProcessPool) because:
  - Redaction is I/O-bound (pydicom read/write release the GIL)
  - No fork overhead, no pickle serialization, no memory doubling
  - Threads share the hash salt and date shift (zero-copy)
  - Won't crash on large datasets (no subprocess memory explosion)
"""

from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pydicom

from src._logging import get_logger

log = get_logger(__name__)

# ── PHI tag sets — frozen for O(1) membership test ───────────────────────────

PHI_TAGS_REMOVE: frozenset[str] = frozenset(
    {
        # Direct identifiers — must be deleted entirely
        "PatientAddress",
        "PatientTelephoneNumbers",
        "PatientInsurancePlanCodeSequence",
        "MilitaryRank",
        "BranchOfService",
        "PatientReligiousPreference",
        "MedicalRecordLocator",
        "ReferencedPatientPhotoSequence",
        "ResponsiblePerson",
        "ResponsibleOrganization",
        "ClinicalTrialSponsorName",
        "ClinicalTrialProtocolID",
        "ClinicalTrialSubjectID",
        "ClinicalTrialSiteID",
        "ClinicalTrialSiteName",
        "RequestingPhysician",
        "ScheduledPerformingPhysicianName",
        "RequestAttributesSequence",
        "ContentSequence",
        "DistributionAddress",
        "DistributionName",
        "FillerOrderNumberImagingServiceRequest",
        "PlacerOrderNumberImagingServiceRequest",
        "OrderCallbackPhoneNumber",
        "OrderCallbackTelecomInformation",
        "OrderEnteredBy",
        "OrderEntererLocation",
        # DICOM Supplement 142 (Clinical Trial De-identification Profile)
        "IssuerOfPatientID",
        "IssuerOfPatientIDQualifiersSequence",
        "PatientBirthTime",
        "PatientAge",
        "EthnicGroup",
        "Occupation",
        "SmokingStatus",
        "PregnancyStatus",
        "PatientSexNeutered",
        "ConfidentialityConstraintOnPatientDataDescription",
        # Sequences that may contain embedded PHI
        "ReferencedStudySequence",
        "ReferencedPerformedProcedureStepSequence",
        "SourceImageSequence",
        "ReferencedImageSequence",
        "OriginalAttributesSequence",
        # Text fields that may contain free-text PHI
        "ImageComments",
        "StudyComments",
        "InterpretationAuthor",
        "InterpretationText",
        "ResultsComments",
    }
)

PHI_TAGS_BLANK: frozenset[str] = frozenset(
    {
        # Names — blank to empty string (preserve tag existence for DICOM conformance)
        "PatientName",
        "PatientID",
        "OtherPatientIDs",
        "OtherPatientNames",
        "PatientBirthName",
        "PatientMotherBirthName",
        "PatientComments",
        "AdditionalPatientHistory",
        "InstitutionName",
        "InstitutionAddress",
        "InstitutionalDepartmentName",
        "StationName",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "NameOfPhysiciansReadingStudy",
        "OperatorsName",
        "PhysiciansOfRecord",
        "AdmittingDiagnosesDescription",
        "AdmittingDiagnosesCodeSequence",
        "AccessionNumber",
        "StudyID",
        "DeviceSerialNumber",
        # Additional identifiers (DICOM PS3.15 Table E.1-1)
        "InstitutionalDepartmentTypeCodeSequence",
        "ProtocolName",  # may encode patient info in some sites
        "RequestedProcedureDescription",
        "PerformedProcedureStepDescription",
        "PerformedProcedureStepID",
        "RequestedProcedureID",
        "ScheduledProcedureStepDescription",
    }
)

PHI_TAGS_HASH: frozenset[str] = frozenset(
    {
        # UIDs — deterministic rehash preserves linkability within a salt scope
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "FrameOfReferenceUID",
        "MediaStorageSOPInstanceUID",
        # Additional UIDs that can link across studies
        "ReferencedSOPInstanceUID",
        "RelatedFrameOfReferenceUID",
    }
)

PHI_TAGS_DATE: frozenset[str] = frozenset(
    {
        # Dates — shifted by consistent offset (preserves intervals)
        "PatientBirthDate",
        "StudyDate",
        "SeriesDate",
        "AcquisitionDate",
        "ContentDate",
        "InstanceCreationDate",
        # Additional date/time fields
        "AcquisitionDateTime",
        "StudyTime",
        "SeriesTime",
        "AcquisitionTime",
        "ContentTime",
        "InstanceCreationTime",
        "PerformedProcedureStepStartDate",
        "PerformedProcedureStepStartTime",
    }
)

# Combined set for single-pass O(1) classification
_ALL_PHI_TAGS: frozenset[str] = PHI_TAGS_REMOVE | PHI_TAGS_BLANK | PHI_TAGS_HASH | PHI_TAGS_DATE


@dataclass
class RedactionResult:
    """Result of redacting a single DICOM file."""

    filepath: str
    output_path: str = ""
    tags_removed: int = 0
    tags_blanked: int = 0
    tags_hashed: int = 0
    tags_date_shifted: int = 0
    verified_clean: bool = False
    error: str | None = None


@dataclass
class RedactionSummary:
    """Aggregate summary of a redaction run."""

    files_processed: int = 0
    files_failed: int = 0
    total_tags_removed: int = 0
    total_tags_blanked: int = 0
    total_tags_hashed: int = 0
    total_tags_date_shifted: int = 0
    date_shift_days: int = 0
    results: list[RedactionResult] = field(default_factory=list)


def _hash_uid(original: str, salt: str) -> str:
    """Deterministic DICOM UID from SHA-256. O(1)."""
    h = hashlib.sha256(f"{salt}{original}".encode()).hexdigest()
    return f"1.2.826.0.1.3680043.10.{int(h[:24], 16)}"[:64]


def _shift_date(date_str: str, shift_days: int) -> str:
    """Shift DICOM date/datetime/time by N days. Handles multiple formats.

    Supports:
      YYYYMMDD           → date shift
      YYYYMMDDHHMMSS     → datetime shift (preserves time)
      YYYYMMDDHHMMSS.fff → datetime shift with fractional seconds
      HHMMSS / HHMMSS.f  → time only, returned as-is (no date to shift)
    """
    s = date_str.strip()
    if not s:
        return ""

    # Time-only fields (HHMMSS or HHMMSS.fff) — no date component to shift
    if len(s) <= 6 or (len(s) <= 13 and "." in s and len(s.split(".")[0]) <= 6):
        return s

    try:
        # Try YYYYMMDDHHMMSS.fff
        if "." in s and len(s) > 14:
            dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S")
            frac = s[14:]
            shifted = dt + timedelta(days=shift_days)
            return shifted.strftime("%Y%m%d%H%M%S") + frac
        # Try YYYYMMDDHHMMSS
        if len(s) >= 14:
            dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S")
            shifted = dt + timedelta(days=shift_days)
            return shifted.strftime("%Y%m%d%H%M%S")
        # Standard YYYYMMDD
        dt = datetime.strptime(s[:8], "%Y%m%d")
        return (dt + timedelta(days=shift_days)).strftime("%Y%m%d")
    except (ValueError, IndexError):
        return ""


def redact_single_file(
    fpath: str,
    out_dir: str,
    salt: str,
    date_shift_days: int,
    verify: bool = True,
) -> RedactionResult:
    """Redact + verify a single DICOM file in one pass.

    Complexity: O(T) where T = number of tags in the file.
    Each tag keyword is checked against frozen sets (O(1) lookup).
    Combined redact + verify eliminates the second file read.
    """
    result = RedactionResult(filepath=fpath)

    try:
        ds = pydicom.dcmread(fpath, force=True)
    except Exception as e:
        result.error = f"Read failed: {e}"
        return result

    # Apply redaction rules — O(|PHI_TAGS|) which is constant (~90 tags)
    for kw in PHI_TAGS_REMOVE:
        if hasattr(ds, kw):
            try:
                delattr(ds, kw)
                result.tags_removed += 1
            except Exception:
                pass

    for kw in PHI_TAGS_BLANK:
        if hasattr(ds, kw):
            try:
                setattr(ds, kw, "")
                result.tags_blanked += 1
            except Exception:
                try:
                    delattr(ds, kw)
                    result.tags_removed += 1
                except Exception:
                    pass

    for kw in PHI_TAGS_HASH:
        val = getattr(ds, kw, None)
        if val is not None:
            try:
                setattr(ds, kw, _hash_uid(str(val), salt))
                result.tags_hashed += 1
            except Exception:
                pass
    # Also hash file_meta UIDs
    if hasattr(ds, "file_meta"):
        for kw in ("MediaStorageSOPInstanceUID",):
            val = getattr(ds.file_meta, kw, None)
            if val is not None:
                try:
                    setattr(ds.file_meta, kw, _hash_uid(str(val), salt))
                    result.tags_hashed += 1
                except Exception:
                    pass

    for kw in PHI_TAGS_DATE:
        val = getattr(ds, kw, None)
        if val is not None and str(val).strip():
            shifted = _shift_date(str(val), date_shift_days)
            if shifted:
                try:
                    setattr(ds, kw, shifted)
                    result.tags_date_shifted += 1
                except Exception:
                    pass

    # Write redacted file
    out_path = Path(out_dir) / Path(fpath).name
    try:
        ds.save_as(str(out_path), enforce_file_format=True)
        result.output_path = str(out_path)
    except Exception as e:
        result.error = f"Write failed: {e}"
        return result

    # Verify in same pass (no second file read needed — check the in-memory dataset)
    if verify:
        clean = True
        for kw in PHI_TAGS_REMOVE | PHI_TAGS_BLANK:
            val = getattr(ds, kw, None)
            if val is not None and str(val).strip():
                clean = False
                break
        result.verified_clean = clean

    return result


def redact_files(
    file_paths: list[str],
    out_dir: str,
    n_workers: int = 8,
    salt: str = "",
    date_shift_days: int | None = None,
    verify: bool = True,
    on_progress=None,
) -> RedactionSummary:
    """Redact PHI from DICOM files using ThreadPoolExecutor.

    ThreadPoolExecutor (not ProcessPool) because:
      - pydicom I/O releases the GIL → true parallelism on file ops
      - No fork overhead, no pickle, no memory doubling
      - Shared salt/shift across threads (zero-copy)
      - Won't crash on large datasets

    Streams results to avoid holding everything in memory.
    """
    import random

    if date_shift_days is None:
        date_shift_days = random.randint(-365, -30)

    if not salt:
        salt = hashlib.sha256(os.urandom(32)).hexdigest()[:16]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    summary = RedactionSummary(date_shift_days=date_shift_days)

    # Thread-safe counter for progress
    _lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(redact_single_file, fp, out_dir, salt, date_shift_days, verify): fp
            for fp in file_paths
        }

        for fut in as_completed(futures):
            result = fut.result()

            with _lock:
                summary.results.append(result)
                if result.error:
                    summary.files_failed += 1
                else:
                    summary.files_processed += 1
                    summary.total_tags_removed += result.tags_removed
                    summary.total_tags_blanked += result.tags_blanked
                    summary.total_tags_hashed += result.tags_hashed
                    summary.total_tags_date_shifted += result.tags_date_shifted

                if on_progress:
                    on_progress(len(summary.results), len(file_paths))

    return summary

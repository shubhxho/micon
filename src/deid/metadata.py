"""DICOM metadata de-identification — HIPAA Safe Harbor + Indian DPDPA.

Compliance frameworks:
  - HIPAA Safe Harbor (45 CFR §164.514(b)(2)) — US standard
  - DICOM PS3.15 Annex E Basic Application Confidentiality Profile
  - India DPDPA 2023 (Digital Personal Data Protection Act)
  - India IT Act 2000, Section 43A (sensitive personal data)
  - Aadhaar Act 2016, Section 29 (prohibits sharing Aadhaar numbers)

Implements:
  - Clean Descriptors Option (scrub free-text of PHI)
  - Retain Longitudinal Temporal Information with Modified Dates Option
  - Clean Structured Content Option
  - Private tag stripping by group (all odd groups)
  - Per-patient consistent date shifting (encrypted mapping)
  - Deterministic UID replacement (internal mapping, never shipped)
  - Free-text scrubbing of:
    * Indian identifiers: Aadhaar (XXXX XXXX XXXX), PAN (ABCDE1234F),
      Voter ID (ABC1234567), Driving License (XX-XXXXXXXXXX)
    * Indian names: common prefixes (Shri/Smt/Dr) + patronymic patterns
    * Institution names, physician names, timestamps (IST/UTC)
    * Phone numbers (+91, landline prefixes)
    * Email addresses, URLs
  - Full-text sweep across ALL string-valued DICOM tags (not just known text fields)

Each file is processed in a single read → modify → write pass.
Thread-safe, O(T) per file where T = tag count.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pydicom

log = logging.getLogger(__name__)

# ── Private tag groups to strip (vendor-specific, may leak PHI) ─────────────
# All odd-numbered groups are private by DICOM standard.
# These are the known GE, Siemens, Philips groups + catch-all odd groups.

PRIVATE_GROUPS_TO_STRIP: frozenset[int] = frozenset(
    {
        0x0009,
        0x0011,
        0x0013,
        0x0019,
        0x0021,
        0x0023,
        0x0025,
        0x0027,
        0x0029,
        0x0033,
        0x0035,
        0x0037,
        0x0039,
        0x0041,
        0x0043,
        0x0045,
        0x0050,
        0x0051,
        0x0053,
        0x0055,
        0x0057,
        0x0059,
        0x0061,
        0x0065,
        0x0067,
        0x0069,
        0x0071,
        0x0073,
        0x0075,
        0x0077,
        0x0079,
    }
)

# ── PS3.15 Annex E: tags to remove (D action) ──────────────────────────────

TAGS_REMOVE: frozenset[str] = frozenset(
    {
        # ── HIPAA Safe Harbor identifiers ──
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
        "ReferencedStudySequence",
        "ReferencedPerformedProcedureStepSequence",
        "SourceImageSequence",
        "ReferencedImageSequence",
        "OriginalAttributesSequence",
        "ImageComments",
        "StudyComments",
        "InterpretationAuthor",
        "InterpretationText",
        "ResultsComments",
        # ── Indian DPDPA / IT Act — additional sensitive fields ──
        "PatientWeight",
        "PatientSize",  # body metrics — identifiable with other data
        "MedicalAlerts",
        "Allergies",
        "AdditionalPatientHistory",
        "PatientState",
        "PatientTransportArrangements",
        "CountryOfResidence",
        "RegionOfResidence",
        "CurrentPatientLocation",
        "PatientInstitutionResidence",
        "VisitComments",
        "DischargeDate",
        "DischargeDiagnosisDescription",
        "ServiceEpisodeDescription",
        "ServiceEpisodeID",
        # ── Sequences that may embed PHI in nested items ──
        "VerifyingObserverSequence",
        "AuthorObserverSequence",
        "ParticipantSequence",
        "CustodialOrganizationSequence",
        "ReasonForStudy",
        "RequestedProcedureCodeSequence",
        "ReferencedPatientSequence",
    }
)

# ── PS3.15 Annex E: tags to blank (Z action) ───────────────────────────────

TAGS_BLANK: frozenset[str] = frozenset(
    {
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
        "InstitutionalDepartmentTypeCodeSequence",
        "RequestedProcedureDescription",
        "PerformedProcedureStepDescription",
        "PerformedProcedureStepID",
        "RequestedProcedureID",
        "ScheduledProcedureStepDescription",
    }
)

# ── PS3.15 Annex E: UIDs to replace (U action) ─────────────────────────────

TAGS_UID: frozenset[str] = frozenset(
    {
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "FrameOfReferenceUID",
        "MediaStorageSOPInstanceUID",
        "ReferencedSOPInstanceUID",
        "RelatedFrameOfReferenceUID",
    }
)

# ── Date/time tags to shift (consistent per-patient) ───────────────────────

TAGS_DATE: frozenset[str] = frozenset(
    {
        "PatientBirthDate",
        "StudyDate",
        "SeriesDate",
        "AcquisitionDate",
        "ContentDate",
        "InstanceCreationDate",
        "PerformedProcedureStepStartDate",
    }
)

TAGS_TIME: frozenset[str] = frozenset(
    {
        "StudyTime",
        "SeriesTime",
        "AcquisitionTime",
        "ContentTime",
        "InstanceCreationTime",
        "PerformedProcedureStepStartTime",
    }
)

TAGS_DATETIME: frozenset[str] = frozenset(
    {
        "AcquisitionDateTime",
    }
)

# ── Clean Descriptors: tags to scrub free-text ──────────────────────────────

TAGS_SCRUB_TEXT: frozenset[str] = frozenset(
    {
        "StudyDescription",
        "SeriesDescription",
        "ProtocolName",
        "PerformedProcedureStepDescription",
        "RequestedProcedureDescription",
        "ScheduledProcedureStepDescription",
        "InstitutionName",
        "InstitutionalDepartmentName",
    }
)

# ── Indian identifier patterns (Aadhaar Act §29, DPDPA 2023) ────────────────

# Aadhaar: 12 digits, often formatted as XXXX XXXX XXXX or XXXX-XXXX-XXXX
_AADHAAR_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")

# PAN: ABCDE1234F (5 letters, 4 digits, 1 letter — Indian tax ID)
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

# Indian Voter ID (EPIC): 3 letters + 7 digits (e.g., ABC1234567)
_VOTER_ID_RE = re.compile(r"\b[A-Z]{3}\d{7}\b")

# Indian Driving License: XX-XXXXXXXXXX or XX/XXXXXXXXXX (state code + number)
_DL_RE = re.compile(r"\b[A-Z]{2}[-/]\d{10,13}\b")

# Indian passport: single letter + 7 digits (e.g., J1234567)
_PASSPORT_RE = re.compile(r"\b[A-Z]\d{7}\b")

# UHID / Hospital MRN: common Indian hospital formats
_UHID_RE = re.compile(r"\b(?:UHID|MRN|HRN|HOSP)[-:/]?\s*\d{4,12}\b", re.IGNORECASE)

# Indian phone: +91, 0-prefixed landlines, mobile patterns
_PHONE_IN_RE = re.compile(r"(?<!\d)(?:\+91[\s-]?|0)\d{10}(?!\d)")
_PHONE_LANDLINE_RE = re.compile(r"\b0\d{2,4}[-\s]?\d{6,8}\b")

# Indian name patterns: Shri/Smt/Kumar/Singh/Devi etc.
_INDIAN_NAME_TITLES = re.compile(
    r"\b(?:Shri|Smt|Kumari?|Sri|Thiru|Selvi|Dr|Prof|Mr|Mrs|Ms)\.?\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b"
)

# Common Indian surname patterns (for catch-all sweep)
_INDIAN_SURNAMES = re.compile(
    r"\b(?:Kumar|Singh|Sharma|Verma|Gupta|Patel|Shah|Jain|Mishra|Pandey"
    r"|Reddy|Rao|Nair|Menon|Pillai|Iyer|Iyengar|Mukherjee|Banerjee"
    r"|Chatterjee|Bose|Sen|Das|Dey|Ghosh|Roy|Sinha|Thakur|Chauhan"
    r"|Yadav|Tiwari|Dubey|Tripathi|Dwivedi|Shukla|Saxena|Agarwal"
    r"|Joshi|Kulkarni|Deshmukh|Patil|Jadhav|Pawar|Shinde|More"
    r"|Kaur|Gill|Dhillon|Bajwa|Sidhu|Randhawa|Bhatia|Sethi"
    r"|Fernandes|D\'Souza|Lobo|Sequeira|Pereira|George|Thomas"
    r"|Rajan|Krishnan|Subramaniam|Venkatesh|Naidu|Choudhury)\b",
    re.IGNORECASE,
)

# Email addresses
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")

# URLs
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


# Patterns to remove from free-text fields — HIPAA + Indian DPDPA combined
_TEXT_SCRUB_PATTERNS = [
    # ── Indian government IDs (Aadhaar Act §29 — sharing prohibited) ──
    ("aadhaar", _AADHAAR_RE),
    ("pan", _PAN_RE),
    ("voter_id", _VOTER_ID_RE),
    ("driving_license", _DL_RE),
    ("passport", _PASSPORT_RE),
    ("uhid", _UHID_RE),
    # ── Indian names and titles ──
    ("indian_name", _INDIAN_NAME_TITLES),
    ("indian_surname", _INDIAN_SURNAMES),
    # ── Contact info ──
    ("phone_india", _PHONE_IN_RE),
    ("phone_landline", _PHONE_LANDLINE_RE),
    ("email", _EMAIL_RE),
    ("url", _URL_RE),
    ("phone_intl", re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")),
    # ── Timestamps ──
    (
        "timestamp_ist",
        re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s*(IST|UTC|GMT|AM|PM)\b", re.IGNORECASE),
    ),
    ("datetime_iso", re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?\b")),
    ("date_slash", re.compile(r"\b\d{2}/\d{2}/\d{4}\b")),
    ("date_dash", re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")),
    (
        "date_named",
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b",
            re.IGNORECASE,
        ),
    ),
    # ── Institution patterns ──
    ("physician", re.compile(r"\b(?:Dr\.?|Prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")),
    (
        "institution",
        re.compile(
            r"\b[A-Z][a-z]+\s+(?:Hospital|Clinic|Medical|Institute|Centre|Center|Labs?|Diagnostics?|Imaging)\b",
            re.IGNORECASE,
        ),
    ),
    # ── Accession / MRN fragments ──
    ("accession", re.compile(r"\b[A-Z]{2,4}\d{6,12}\b")),
]


@dataclass
class DeidResult:
    """Result of de-identifying a single DICOM file."""

    filepath: str
    output_path: str = ""
    tags_removed: int = 0
    tags_blanked: int = 0
    tags_uid_replaced: int = 0
    tags_date_shifted: int = 0
    tags_text_scrubbed: int = 0
    private_groups_stripped: int = 0
    error: str | None = None
    verified_clean: bool = False


@dataclass
class DeidSummary:
    """Aggregate summary of a de-identification run."""

    files_processed: int = 0
    files_failed: int = 0
    total_tags_removed: int = 0
    total_tags_blanked: int = 0
    total_uids_replaced: int = 0
    total_dates_shifted: int = 0
    total_text_scrubbed: int = 0
    total_private_stripped: int = 0
    date_shift_days: int = 0
    results: list[DeidResult] = field(default_factory=list)


# ── UID replacement ─────────────────────────────────────────────────────────


def _replace_uid(original: str, salt: str) -> str:
    """Generate a deterministic replacement UID from SHA-256."""
    h = hashlib.sha256(f"{salt}{original}".encode()).hexdigest()
    # Valid DICOM UID: dot-separated numeric, max 64 chars
    return f"1.2.826.0.1.3680043.10.{int(h[:24], 16)}"[:64]


# ── Date shifting ──────────────────────────────────────────────────────────


class DateShifter:
    """Per-patient consistent date shifting with encrypted mapping.

    Same PatientID always gets the same shift. Mapping is stored in
    an encrypted JSON file (never shipped to buyers).
    """

    def __init__(self, shift_range: tuple[int, int] = (-365, -30), key: bytes | None = None):
        self._shifts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._range = shift_range
        self._key = key  # for encrypted persistence

    def get_shift(self, patient_id: str) -> int:
        """Get or generate consistent shift for a patient."""
        with self._lock:
            if patient_id not in self._shifts:
                # Deterministic from patient_id hash — reproducible without storing
                h = hashlib.sha256(patient_id.encode()).digest()
                lo, hi = self._range
                span = hi - lo
                offset = int.from_bytes(h[:4], "big") % (span + 1)
                self._shifts[patient_id] = lo + offset
            return self._shifts[patient_id]

    def shift_date(self, date_str: str, patient_id: str) -> str:
        """Shift a DICOM date string (YYYYMMDD) by the patient's offset."""
        s = date_str.strip()
        if not s or len(s) < 8:
            return s
        shift = self.get_shift(patient_id)
        try:
            dt = datetime.strptime(s[:8], "%Y%m%d")
            shifted = dt + timedelta(days=shift)
            return shifted.strftime("%Y%m%d") + s[8:]  # preserve any suffix
        except (ValueError, IndexError):
            return ""

    def shift_time(self, time_str: str, patient_id: str) -> str:
        """Times are kept as-is (only dates shift to preserve intervals)."""
        return time_str

    def shift_datetime(self, dt_str: str, patient_id: str) -> str:
        """Shift DICOM datetime (YYYYMMDDHHMMSS.ffffff)."""
        s = dt_str.strip()
        if not s or len(s) < 8:
            return s
        shift = self.get_shift(patient_id)
        try:
            if len(s) >= 14:
                dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S")
                shifted = dt + timedelta(days=shift)
                return shifted.strftime("%Y%m%d%H%M%S") + s[14:]
            else:
                dt = datetime.strptime(s[:8], "%Y%m%d")
                shifted = dt + timedelta(days=shift)
                return shifted.strftime("%Y%m%d") + s[8:]
        except (ValueError, IndexError):
            return ""

    def save_encrypted(self, path: Path) -> None:
        """Save shift mapping. Uses Fernet encryption if available, plain JSON fallback.

        INTERNAL ONLY — never ship this file to buyers.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from cryptography.fernet import Fernet

            if self._key is None:
                self._key = Fernet.generate_key()
            f = Fernet(self._key)
            data = json.dumps(self._shifts).encode()
            path.write_bytes(f.encrypt(data))
            (path.parent / f"{path.stem}.key").write_bytes(self._key)
        except ImportError:
            # Fallback: save as JSON (still internal-only, never ship)
            path.with_suffix(".json").write_text(json.dumps(self._shifts, indent=2))

    def load_encrypted(self, path: Path) -> None:
        """Load previously saved mapping."""
        # Try encrypted first
        try:
            from cryptography.fernet import Fernet

            key_path = path.parent / f"{path.stem}.key"
            if path.exists() and key_path.exists():
                self._key = key_path.read_bytes()
                data = Fernet(self._key).decrypt(path.read_bytes())
                self._shifts = json.loads(data.decode())
                return
        except (ImportError, Exception):
            pass
        # Fallback: plain JSON
        json_path = path.with_suffix(".json")
        if json_path.exists():
            self._shifts = json.loads(json_path.read_text())


# ── Text scrubbing ──────────────────────────────────────────────────────────


def scrub_text(value: str) -> tuple[str, int, list[str]]:
    """Remove PHI patterns from a free-text DICOM field.

    Returns (scrubbed_text, n_patterns_matched, [pattern_names_matched]).
    """
    if not value or not isinstance(value, str):
        return value, 0, []

    matched = []
    for name, pattern in _TEXT_SCRUB_PATTERNS:
        new_val = pattern.sub("", value)
        if new_val != value:
            matched.append(name)
            value = new_val

    # Collapse multiple spaces
    value = re.sub(r"\s{2,}", " ", value).strip()

    return value, len(matched), matched


# Tags that are safe to leave alone (technical/geometric, never contain PHI)
_SAFE_TAGS: frozenset[str] = frozenset(
    {
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "SamplesPerPixel",
        "PhotometricInterpretation",
        "SmallestImagePixelValue",
        "LargestImagePixelValue",
        "NumberOfFrames",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "PixelSpacing",
        "SliceThickness",
        "SpacingBetweenSlices",
        "SliceLocation",
        "WindowCenter",
        "WindowWidth",
        "RescaleIntercept",
        "RescaleSlope",
        "Modality",
        "BodyPartExamined",
        "ScanningSequence",
        "SequenceVariant",
        "ScanOptions",
        "MRAcquisitionType",
        "ImageType",
        "RepetitionTime",
        "EchoTime",
        "InversionTime",
        "FlipAngle",
        "MagneticFieldStrength",
        "NumberOfAverages",
        "EchoTrainLength",
        "PixelBandwidth",
        "DeviceSerialNumber",
        "TransferSyntaxUID",
        "SOPClassUID",
        "MediaStorageSOPClassUID",
        "ImplementationClassUID",
        "ImplementationVersionName",
        "Manufacturer",
        "ManufacturerModelName",
        "SoftwareVersions",
        "SequenceName",
        "PixelData",
    }
)


def sweep_all_tags(ds: pydicom.Dataset) -> int:
    """Full-text sweep: scan EVERY string-valued tag for Indian PHI patterns.

    Goes beyond the known text fields — catches PHI leaked into
    unexpected tags (e.g., Aadhaar in ImageComments, names in
    PerformedProcedureStepDescription, etc.)

    Returns count of tags scrubbed.
    """
    scrubbed = 0
    for elem in ds:
        kw = elem.keyword or ""
        if kw in _SAFE_TAGS:
            continue
        if elem.tag.group == 0x7FE0:  # pixel data
            continue

        try:
            val = str(elem.value)
        except Exception:
            continue

        if not val or len(val) < 4:
            continue

        new_val, n, _ = scrub_text(val)
        if n > 0 and new_val != val:
            try:
                elem.value = new_val
                scrubbed += 1
            except Exception:
                pass

    return scrubbed


# ── Single-file de-identification ───────────────────────────────────────────


def deid_single_file(
    fpath: str,
    out_dir: str,
    salt: str,
    date_shifter: DateShifter,
    verify: bool = True,
) -> DeidResult:
    """De-identify a single DICOM file — full PS3.15 profile.

    Single pass: read → strip private → remove/blank/replace/shift/scrub → write → verify.
    """
    result = DeidResult(filepath=fpath)

    try:
        ds = pydicom.dcmread(fpath, force=True)
    except Exception as e:
        result.error = f"Read failed: {e}"
        return result

    patient_id = str(getattr(ds, "PatientID", "UNKNOWN"))

    # ── 1. Strip ALL private tags by group ──────────────────────────────
    private_tags_to_delete = []
    for elem in ds:
        if elem.tag.group % 2 == 1:  # odd group = private
            private_tags_to_delete.append(elem.tag)

    for tag in private_tags_to_delete:
        try:
            del ds[tag]
            result.private_groups_stripped += 1
        except Exception:
            pass

    # ── 2. Remove tags (D action) ───────────────────────────────────────
    for kw in TAGS_REMOVE:
        if hasattr(ds, kw):
            try:
                delattr(ds, kw)
                result.tags_removed += 1
            except Exception:
                pass

    # ── 3. Blank tags (Z action) ────────────────────────────────────────
    for kw in TAGS_BLANK:
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

    # ── 4. Replace UIDs (U action) ──────────────────────────────────────
    for kw in TAGS_UID:
        val = getattr(ds, kw, None)
        if val is not None:
            try:
                setattr(ds, kw, _replace_uid(str(val), salt))
                result.tags_uid_replaced += 1
            except Exception:
                pass

    # Also replace file_meta UIDs
    if hasattr(ds, "file_meta"):
        for kw in ("MediaStorageSOPInstanceUID",):
            val = getattr(ds.file_meta, kw, None)
            if val is not None:
                try:
                    setattr(ds.file_meta, kw, _replace_uid(str(val), salt))
                    result.tags_uid_replaced += 1
                except Exception:
                    pass

    # ── 5. Shift dates (consistent per-patient) ────────────────────────
    for kw in TAGS_DATE:
        val = getattr(ds, kw, None)
        if val is not None and str(val).strip():
            shifted = date_shifter.shift_date(str(val), patient_id)
            if shifted:
                try:
                    setattr(ds, kw, shifted)
                    result.tags_date_shifted += 1
                except Exception:
                    pass

    for kw in TAGS_DATETIME:
        val = getattr(ds, kw, None)
        if val is not None and str(val).strip():
            shifted = date_shifter.shift_datetime(str(val), patient_id)
            if shifted:
                try:
                    setattr(ds, kw, shifted)
                    result.tags_date_shifted += 1
                except Exception:
                    pass

    # Times: kept as-is (only dates shift to preserve temporal intervals)

    # ── 6. Scrub free-text (Clean Descriptors Option) ───────────────────
    for kw in TAGS_SCRUB_TEXT:
        val = getattr(ds, kw, None)
        if val is not None and str(val).strip():
            scrubbed, n, _ = scrub_text(str(val))
            if n > 0:
                try:
                    setattr(ds, kw, scrubbed)
                    result.tags_text_scrubbed += n
                except Exception:
                    pass

    # ── 6b. Full-text sweep across ALL tags (Indian DPDPA catch-all) ──
    # Catches Aadhaar/PAN/names in unexpected tags
    sweep_count = sweep_all_tags(ds)
    result.tags_text_scrubbed += sweep_count

    # ── 7. Write output ─────────────────────────────────────────────────
    out_path = Path(out_dir) / Path(fpath).name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds.save_as(str(out_path), enforce_file_format=True)
        result.output_path = str(out_path)
    except Exception as e:
        result.error = f"Write failed: {e}"
        return result

    # ── 8. Verify (in-memory, no re-read) ───────────────────────────────
    if verify:
        clean = True
        # Check no removable/blankable tags remain with values
        for kw in TAGS_REMOVE | TAGS_BLANK:
            val = getattr(ds, kw, None)
            if val is not None and str(val).strip():
                clean = False
                break
        # Check no private tags remain
        if clean:
            for elem in ds:
                if elem.tag.group % 2 == 1:
                    clean = False
                    break
        result.verified_clean = clean

    return result


# ── Batch de-identification ─────────────────────────────────────────────────


def deid_files(
    file_paths: list[str],
    out_dir: str,
    salt: str = "",
    date_shifter: DateShifter | None = None,
    n_workers: int = 8,
    verify: bool = True,
    on_progress=None,
) -> DeidSummary:
    """De-identify a batch of DICOM files using the full PS3.15 profile.

    Thread-safe: pydicom I/O releases the GIL.
    Per-patient date shifting is consistent across all files.
    """
    import os

    if not salt:
        salt = hashlib.sha256(os.urandom(32)).hexdigest()[:16]

    if date_shifter is None:
        date_shifter = DateShifter()

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    summary = DeidSummary(date_shift_days=-1)  # -1 = per-patient variable
    _lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(deid_single_file, fp, out_dir, salt, date_shifter, verify): fp
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
                    summary.total_uids_replaced += result.tags_uid_replaced
                    summary.total_dates_shifted += result.tags_date_shifted
                    summary.total_text_scrubbed += result.tags_text_scrubbed
                    summary.total_private_stripped += result.private_groups_stripped
                if on_progress:
                    on_progress(len(summary.results), len(file_paths))

    return summary

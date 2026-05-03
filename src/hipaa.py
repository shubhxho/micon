"""HIPAA compliance — PHI inventory, audit trail, compliance scoring.

Does NOT redact, blur, or modify any data. Instead:
  1. Scans every DICOM file for all 18 HIPAA identifiers
  2. Builds a complete PHI inventory (what exists, where, how populated)
  3. Scores compliance across the study (completeness, consistency, risk)
  4. Generates an audit trail of every file accessed and what PHI was found
  5. Produces a compliance report for the HTML dashboard

All functions are pure, thread-safe, and O(T) per file where T = tag count.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# ── HIPAA Safe Harbor: 18 identifier categories (45 CFR §164.514(b)(2)) ────
#
# These are the 18 types of information that must be addressed for
# Safe Harbor de-identification. We scan for ALL of them but modify NOTHING.

HIPAA_18_IDENTIFIERS = {
    "names": {
        "tags": [
            "PatientName",
            "OtherPatientNames",
            "PatientBirthName",
            "PatientMotherBirthName",
            "ReferringPhysicianName",
            "PerformingPhysicianName",
            "NameOfPhysiciansReadingStudy",
            "OperatorsName",
            "PhysiciansOfRecord",
            "ResponsiblePerson",
        ],
        "category": "1. Names",
        "description": "Names of patient, physicians, operators, responsible persons",
    },
    "geographic": {
        "tags": [
            "PatientAddress",
            "InstitutionAddress",
            "InstitutionName",
            "InstitutionalDepartmentName",
            "StationName",
            "DistributionAddress",
            "DistributionName",
            "OrderEntererLocation",
            "Country",
        ],
        "category": "2. Geographic data",
        "description": "All geographic subdivisions smaller than state (address, city, zip, etc.)",
    },
    "dates": {
        "tags": [
            "PatientBirthDate",
            "StudyDate",
            "SeriesDate",
            "AcquisitionDate",
            "ContentDate",
            "InstanceCreationDate",
            "PatientAge",
        ],
        "category": "3. Dates",
        "description": "All dates related to an individual (birth, admission, discharge, death, etc.)",
    },
    "phone": {
        "tags": [
            "PatientTelephoneNumbers",
            "OrderCallbackPhoneNumber",
            "OrderCallbackTelecomInformation",
        ],
        "category": "4. Phone numbers",
        "description": "Telephone numbers",
    },
    "fax": {
        "tags": [],  # rarely in DICOM
        "category": "5. Fax numbers",
        "description": "Fax numbers (not typically in DICOM)",
    },
    "email": {
        "tags": ["PatientComments"],  # sometimes contains email
        "category": "6. Email addresses",
        "description": "Electronic mail addresses",
    },
    "ssn": {
        "tags": ["PatientID", "OtherPatientIDs"],  # sometimes SSN-derived
        "category": "7. Social Security numbers",
        "description": "Social Security numbers (may appear in PatientID)",
    },
    "mrn": {
        "tags": ["PatientID", "OtherPatientIDs", "MedicalRecordLocator", "AccessionNumber"],
        "category": "8. Medical record numbers",
        "description": "Medical record numbers and accession numbers",
    },
    "health_plan": {
        "tags": ["PatientInsurancePlanCodeSequence"],
        "category": "9. Health plan beneficiary numbers",
        "description": "Health plan beneficiary numbers",
    },
    "account": {
        "tags": ["StudyID"],
        "category": "10. Account numbers",
        "description": "Account numbers",
    },
    "license": {
        "tags": [],
        "category": "11. Certificate/license numbers",
        "description": "Certificate/license numbers (not typically in DICOM)",
    },
    "vehicle": {
        "tags": [],
        "category": "12. Vehicle identifiers",
        "description": "Vehicle identifiers and serial numbers (not in DICOM)",
    },
    "device": {
        "tags": ["DeviceSerialNumber"],
        "category": "13. Device identifiers",
        "description": "Device identifiers and serial numbers",
    },
    "urls": {
        "tags": ["RetrieveURL"],
        "category": "14. Web URLs",
        "description": "Web Universal Resource Locators",
    },
    "ip": {
        "tags": [],
        "category": "15. IP addresses",
        "description": "Internet Protocol address numbers (not typically in DICOM)",
    },
    "biometric": {
        "tags": [],
        "category": "16. Biometric identifiers",
        "description": "Biometric identifiers (not typically in DICOM tags)",
    },
    "photo": {
        "tags": ["ReferencedPatientPhotoSequence"],
        "category": "17. Full face photos",
        "description": "Full face photographic images and comparable images",
    },
    "unique_ids": {
        "tags": [
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "FrameOfReferenceUID",
            "MediaStorageSOPInstanceUID",
            "ClinicalTrialSponsorName",
            "ClinicalTrialProtocolID",
            "ClinicalTrialSubjectID",
            "ClinicalTrialSiteID",
            "ClinicalTrialSiteName",
        ],
        "category": "18. Any other unique identifying number/code",
        "description": "UIDs, clinical trial IDs, and any other unique identifier",
    },
}

# Flat set of all PHI tag keywords for O(1) lookup
ALL_PHI_TAGS: frozenset[str] = frozenset(
    tag for cat in HIPAA_18_IDENTIFIERS.values() for tag in cat["tags"]
)


@dataclass
class PHIFinding:
    """A single PHI element found in a DICOM file."""

    filename: str
    filepath: str
    tag_keyword: str
    hipaa_category: str
    value_present: bool
    value_length: int
    value_preview: str  # first 50 chars, for audit (not the full value)


@dataclass
class FileAuditRecord:
    """Audit trail for a single file scan."""

    filepath: str
    filename: str
    scan_timestamp: float
    phi_tags_found: int
    phi_tags_empty: int
    phi_categories_present: list[str]
    risk_level: str  # "high", "medium", "low", "none"


@dataclass
class HIPAAComplianceReport:
    """Full HIPAA compliance report for a study."""

    study_name: str
    scan_timestamp: str
    total_files: int
    total_phi_findings: int
    phi_by_category: dict[str, int]
    phi_by_tag: dict[str, int]
    files_with_phi: int
    files_without_phi: int
    risk_summary: dict[str, int]  # {"high": N, "medium": N, "low": N, "none": N}
    compliance_score: float  # 0-100
    hipaa_categories_present: list[str]
    hipaa_categories_absent: list[str]
    recommendations: list[str]
    audit_trail: list[FileAuditRecord]
    findings: list[PHIFinding]


# ── Per-file PHI scan ──────────────────────────────────────────────────────


def _tag_to_category(tag_keyword: str) -> str:
    """Map a DICOM tag keyword to its HIPAA identifier category. O(1)."""
    for _cat_key, cat_info in HIPAA_18_IDENTIFIERS.items():
        if tag_keyword in cat_info["tags"]:
            return cat_info["category"]
    return "Unknown"


def scan_file_phi(filepath: str) -> tuple[list[PHIFinding], FileAuditRecord]:
    """Scan a single DICOM file for all HIPAA identifiers.

    Does NOT modify the file. Returns findings + audit record.
    Complexity: O(|ALL_PHI_TAGS|) per file — constant (≈70 tag checks).
    """
    import pydicom

    filename = Path(filepath).name
    findings: list[PHIFinding] = []
    categories_found: set[str] = set()
    phi_found = 0
    phi_empty = 0

    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
    except Exception:
        return findings, FileAuditRecord(
            filepath=filepath,
            filename=filename,
            scan_timestamp=time.time(),
            phi_tags_found=0,
            phi_tags_empty=0,
            phi_categories_present=[],
            risk_level="none",
        )

    # Check every PHI tag
    for _cat_key, cat_info in HIPAA_18_IDENTIFIERS.items():
        for tag_kw in cat_info["tags"]:
            val = getattr(ds, tag_kw, None)
            if val is None:
                continue

            val_str = str(val).strip()
            is_present = len(val_str) > 0 and val_str not in ("", "None", "0")

            if is_present:
                phi_found += 1
                categories_found.add(cat_info["category"])
            else:
                phi_empty += 1

            findings.append(
                PHIFinding(
                    filename=filename,
                    filepath=filepath,
                    tag_keyword=tag_kw,
                    hipaa_category=cat_info["category"],
                    value_present=is_present,
                    value_length=len(val_str),
                    value_preview=val_str[:50] if is_present else "",
                )
            )

    # Also check file_meta for UIDs
    if hasattr(ds, "file_meta"):
        fm = ds.file_meta
        for tag_kw in ("MediaStorageSOPInstanceUID",):
            val = getattr(fm, tag_kw, None)
            if val is not None:
                val_str = str(val).strip()
                if val_str:
                    phi_found += 1
                    categories_found.add("18. Any other unique identifying number/code")
                    findings.append(
                        PHIFinding(
                            filename=filename,
                            filepath=filepath,
                            tag_keyword=tag_kw,
                            hipaa_category="18. Any other unique identifying number/code",
                            value_present=True,
                            value_length=len(val_str),
                            value_preview=val_str[:50],
                        )
                    )

    # Risk level based on what categories of PHI are present
    sensitive_categories = {
        "1. Names",
        "7. Social Security numbers",
        "4. Phone numbers",
        "6. Email addresses",
        "2. Geographic data",
    }
    found_sensitive = categories_found & sensitive_categories
    if len(found_sensitive) >= 3:
        risk = "high"
    elif len(found_sensitive) >= 1:
        risk = "medium"
    elif phi_found > 0:
        risk = "low"
    else:
        risk = "none"

    audit = FileAuditRecord(
        filepath=filepath,
        filename=filename,
        scan_timestamp=time.time(),
        phi_tags_found=phi_found,
        phi_tags_empty=phi_empty,
        phi_categories_present=sorted(categories_found),
        risk_level=risk,
    )

    return findings, audit


# ── Study-level HIPAA compliance scan ──────────────────────────────────────


def run_hipaa_scan(
    file_paths: list[str],
    study_name: str = "",
    n_workers: int = 8,
    de_identified_tags: frozenset[str] | None = None,
) -> HIPAAComplianceReport:
    """Scan all files in a study for HIPAA compliance.

    Threaded scan — pydicom header reads release the GIL.
    Returns a full compliance report without modifying any data.

    de_identified_tags: tags that have been de-identified (hashed/shifted/blanked).
      Findings for these tags are kept in the audit trail but marked as
      value_present=False so they don't count against the compliance score.
    """
    all_findings: list[PHIFinding] = []
    all_audits: list[FileAuditRecord] = []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(scan_file_phi, fp): fp for fp in file_paths}
        for fut in as_completed(futures):
            findings, audit = fut.result()
            # Mark de-identified tags as compliant (present but not PHI)
            if de_identified_tags:
                for f in findings:
                    if f.tag_keyword in de_identified_tags:
                        f.value_present = False
                        f.value_preview = "(de-identified)"
                # Recompute audit risk since some tags are now compliant
                phi_found = sum(1 for f in findings if f.value_present)
                cats = {f.hipaa_category for f in findings if f.value_present}
                sensitive = cats & {
                    "1. Names",
                    "7. Social Security numbers",
                    "4. Phone numbers",
                    "6. Email addresses",
                    "2. Geographic data",
                }
                if len(sensitive) >= 3:
                    audit.risk_level = "high"
                elif len(sensitive) >= 1:
                    audit.risk_level = "medium"
                elif phi_found > 0:
                    audit.risk_level = "low"
                else:
                    audit.risk_level = "none"
                audit.phi_tags_found = phi_found
            all_findings.extend(findings)
            all_audits.append(audit)

    # Aggregate
    phi_by_category: dict[str, int] = defaultdict(int)
    phi_by_tag: dict[str, int] = defaultdict(int)
    for f in all_findings:
        if f.value_present:
            phi_by_category[f.hipaa_category] += 1
            phi_by_tag[f.tag_keyword] += 1

    files_with_phi = sum(1 for a in all_audits if a.phi_tags_found > 0)
    files_without_phi = len(all_audits) - files_with_phi

    risk_summary = defaultdict(int)
    for a in all_audits:
        risk_summary[a.risk_level] += 1

    # Categories present/absent
    all_categories = {cat["category"] for cat in HIPAA_18_IDENTIFIERS.values()}
    present_categories = set(phi_by_category.keys())
    absent_categories = all_categories - present_categories

    # Compliance score: higher = more PHI present (more risk, lower compliance)
    # 100 = no PHI found anywhere (fully de-identified)
    # 0 = all 18 categories populated across all files
    if not all_findings:
        compliance_score = 100.0
    else:
        populated = sum(1 for f in all_findings if f.value_present)
        total_possible = len(all_findings)
        phi_ratio = populated / max(total_possible, 1)
        category_ratio = len(present_categories) / len(all_categories)
        compliance_score = max(0, 100 - (phi_ratio * 50 + category_ratio * 50))

    # Recommendations
    recommendations = []
    if "1. Names" in present_categories:
        recommendations.append(
            "Patient and physician names are present — consider de-identification "
            "before sharing outside the institution"
        )
    if "7. Social Security numbers" in present_categories:
        recommendations.append(
            "PatientID may contain SSN-derived identifiers — verify ID format "
            "and consider pseudonymization for research use"
        )
    if "3. Dates" in present_categories:
        recommendations.append(
            "Dates of service are present (birth, study, acquisition) — "
            "for Safe Harbor compliance, dates must be generalized to year only "
            "or shifted by a consistent offset"
        )
    if "2. Geographic data" in present_categories:
        recommendations.append(
            "Geographic identifiers found (institution name/address, station) — "
            "these identify the facility and may need removal for external sharing"
        )
    if "18. Any other unique identifying number/code" in present_categories:
        recommendations.append(
            "DICOM UIDs are present (Study/Series/SOP Instance UIDs) — "
            "these are linkable identifiers that should be rehashed for de-identified datasets"
        )
    if "13. Device identifiers" in present_categories:
        recommendations.append(
            "Device serial numbers present — may allow re-identification "
            "via scanner inventory records"
        )
    if risk_summary.get("high", 0) > 0:
        recommendations.append(
            f"{risk_summary['high']} files have HIGH PHI risk (3+ sensitive categories) — "
            "this data should not leave institutional controls without de-identification"
        )
    if compliance_score >= 90:
        recommendations.append(
            "Study has minimal PHI exposure — suitable for research with "
            "limited additional de-identification"
        )

    return HIPAAComplianceReport(
        study_name=study_name,
        scan_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        total_files=len(file_paths),
        total_phi_findings=sum(1 for f in all_findings if f.value_present),
        phi_by_category=dict(phi_by_category),
        phi_by_tag=dict(phi_by_tag),
        files_with_phi=files_with_phi,
        files_without_phi=files_without_phi,
        risk_summary=dict(risk_summary),
        compliance_score=round(compliance_score, 1),
        hipaa_categories_present=sorted(present_categories),
        hipaa_categories_absent=sorted(absent_categories),
        recommendations=recommendations,
        audit_trail=all_audits,
        findings=all_findings,
    )


def compliance_report_to_dict(report: HIPAAComplianceReport) -> dict:
    """Serialize a compliance report to a JSON-safe dict."""
    return {
        "study_name": report.study_name,
        "scan_timestamp": report.scan_timestamp,
        "total_files": report.total_files,
        "total_phi_findings": report.total_phi_findings,
        "phi_by_category": report.phi_by_category,
        "phi_by_tag": report.phi_by_tag,
        "files_with_phi": report.files_with_phi,
        "files_without_phi": report.files_without_phi,
        "risk_summary": report.risk_summary,
        "compliance_score": report.compliance_score,
        "hipaa_categories_present": report.hipaa_categories_present,
        "hipaa_categories_absent": report.hipaa_categories_absent,
        "recommendations": report.recommendations,
        "audit_summary": {
            "total_files_scanned": len(report.audit_trail),
            "high_risk_files": sum(1 for a in report.audit_trail if a.risk_level == "high"),
            "medium_risk_files": sum(1 for a in report.audit_trail if a.risk_level == "medium"),
            "low_risk_files": sum(1 for a in report.audit_trail if a.risk_level == "low"),
            "no_risk_files": sum(1 for a in report.audit_trail if a.risk_level == "none"),
        },
    }

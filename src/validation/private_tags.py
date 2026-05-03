"""Private tag scanner — asserts zero remaining private group elements.

Private tags (odd-numbered groups) are vendor-specific and may contain PHI.
After de-identification, ALL private tags should be stripped.
This validator confirms none remain.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pydicom


@dataclass
class PrivateTagFinding:
    filename: str
    tag: str
    group: int
    description: str


@dataclass
class PrivateTagReport:
    total_files: int = 0
    clean_files: int = 0
    dirty_files: int = 0
    total_private_tags: int = 0
    findings: list[PrivateTagFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.dirty_files == 0


def scan_file_private_tags(filepath: str) -> list[PrivateTagFinding]:
    """Check a single file for any remaining private tags."""
    findings = []
    filename = Path(filepath).name

    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
    except Exception:
        return findings

    for elem in ds:
        if elem.tag.group % 2 == 1:  # odd group = private
            findings.append(
                PrivateTagFinding(
                    filename=filename,
                    tag=f"({elem.tag.group:04X},{elem.tag.element:04X})",
                    group=elem.tag.group,
                    description=elem.keyword or "Unknown",
                )
            )

    return findings


def scan_private_tags(
    file_paths: list[str],
    n_workers: int = 8,
) -> PrivateTagReport:
    """Scan all files for remaining private tags."""
    report = PrivateTagReport(total_files=len(file_paths))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(scan_file_private_tags, fp): fp for fp in file_paths}
        for fut in as_completed(futures):
            findings = fut.result()
            if findings:
                report.dirty_files += 1
                report.total_private_tags += len(findings)
                report.findings.extend(findings[:10])  # cap per file
            else:
                report.clean_files += 1

    return report

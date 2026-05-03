"""Clean DICOM export — proper study/series/instance hierarchy.

Produces a buyer-ready directory tree:
  output/
    <StudyDate>_<Modality>/
      <SeriesNumber>_<SeriesDescription>/
        <InstanceNumber>.dcm

All filenames are sanitized. No source paths leak.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pydicom


def _sanitize(name: str, max_len: int = 50) -> str:
    """Make a string safe for use as a filename."""
    s = re.sub(r"[^\w\s\-.]", "", name)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len] or "unknown"


def export_clean_dicom(
    source_dir: Path | str,
    output_dir: Path | str,
    file_paths: list[str] | None = None,
) -> dict:
    """Re-organize DICOM files into clean study/series/instance hierarchy.

    Reads each file's metadata to determine the correct folder placement.
    Source paths are never reflected in the output.

    Returns: {"files_exported": N, "studies": [...], "series": [...]}
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_paths is None:
        file_paths = [str(f) for f in sorted(source_dir.rglob("*.dcm"))]

    exported = 0
    studies_seen = set()
    series_seen = set()

    for fpath in file_paths:
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
        except Exception:
            continue

        study_date = str(getattr(ds, "StudyDate", "unknown"))
        modality = str(getattr(ds, "Modality", "unknown"))
        series_num = str(getattr(ds, "SeriesNumber", "0"))
        series_desc = str(getattr(ds, "SeriesDescription", "unknown"))
        instance_num = str(getattr(ds, "InstanceNumber", "0"))

        study_folder = _sanitize(f"{study_date}_{modality}")
        series_folder = _sanitize(f"{series_num.zfill(4)}_{series_desc}")
        instance_name = f"{instance_num.zfill(6)}.dcm"

        dest = output_dir / study_folder / series_folder / instance_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(fpath, dest)
        exported += 1
        studies_seen.add(study_folder)
        series_seen.add(f"{study_folder}/{series_folder}")

    return {
        "files_exported": exported,
        "studies": sorted(studies_seen),
        "series": sorted(series_seen),
    }

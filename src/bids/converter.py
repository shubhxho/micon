"""BIDS converter for the Speall MRI pipeline.

Converts Speall study output (series JSONs + DICOM files) into a
Brain Imaging Data Structure (BIDS) 1.10.0 compliant layout.

Usage::

    python -m src.bids.converter --root <corpus_root> --bids-out <output_dir>

Or via Python API::

    from src.bids.converter import convert_study, convert_dataset
    stats = convert_study(
        study_dir=Path("Speall_MRI_Samples"),
        bids_root=Path("/data/bids"),
        subject_id="001",
        session="01",
    )
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import struct
from pathlib import Path
from typing import Any

from src._logging import get_logger
from src.bids.mappings import (
    SEQUENCE_TO_BIDS,
    bids_filename,
    infer_acquisition_label,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports -- graceful degradation
# ---------------------------------------------------------------------------

try:
    import dicom2nifti.convert_dicom  # type: ignore[import-untyped]  # noqa: F401

    _DICOM2NIFTI = True
except ImportError:
    _DICOM2NIFTI = False

try:
    import nibabel as nib  # type: ignore[import-untyped]

    _NIBABEL = True
except ImportError:
    nib = None  # type: ignore[assignment]
    _NIBABEL = False

try:
    import SimpleITK as sitk  # type: ignore[import-untyped]

    _SITK = True
except ImportError:
    sitk = None  # type: ignore[assignment]
    _SITK = False

# ---------------------------------------------------------------------------
# NIfTI writing helpers
# ---------------------------------------------------------------------------

_EMPTY_NIFTI_GZ = b""  # produced by _write_placeholder_nifti


def _write_placeholder_nifti(dest: Path) -> None:
    """Write a minimal valid gzip-compressed NIfTI-1 placeholder.

    The file is a real gzip stream containing a zero-filled NIfTI-1
    348-byte header + 4 bytes of extension block, so nibabel can read it.
    Used when DICOMs are missing or NIfTI libraries are unavailable.
    """
    # NIfTI-1 header: 348 bytes, extension: 4 zero bytes
    header = bytearray(348)
    # sizeof_hdr = 348 (int32 little-endian at byte 0)
    struct.pack_into("<i", header, 0, 348)
    # magic = "n+1\0" at bytes 344-347
    header[344:348] = b"n+1\0"
    # datatype = 4 (int16) at byte 70
    struct.pack_into("<h", header, 70, 4)
    # bitpix = 16 at byte 72
    struct.pack_into("<h", header, 72, 16)
    # dim[0] = 3, dim[1..3] = 1
    for i, v in enumerate([3, 1, 1, 1, 1, 1, 1, 1]):
        struct.pack_into("<h", header, 40 + i * 2, v)
    # pixdim[0..3] = 1.0
    for i in range(4):
        struct.pack_into("<f", header, 76 + i * 4, 1.0)
    # vox_offset = 352.0 (tells nibabel where voxels start after ext)
    struct.pack_into("<f", header, 108, 352.0)
    # scl_slope = 1.0, scl_inter = 0.0
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<f", header, 116, 0.0)
    ext_block = bytes(4)  # no extensions
    payload = bytes(header) + ext_block
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wb") as fh:
        fh.write(payload)


def _resolve_dicom_dir(
    study_dir: Path,
    source_subdir: str,
    filenames: list[str],
) -> Path | None:
    """Return the directory containing the DICOM files, or None if not found."""
    candidates: list[Path] = []
    if source_subdir:
        candidates.append(study_dir / source_subdir)
    candidates.append(study_dir)
    for cdir in candidates:
        if cdir.is_dir() and any((cdir / fn).exists() for fn in filenames[:1]):
            return cdir
    return None


def _convert_dicom_to_nifti(dicom_dir: Path, dest: Path) -> bool:
    """Convert a DICOM series directory to NIfTI. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _DICOM2NIFTI:
        try:
            import dicom2nifti.convert_dicom as _d2n  # type: ignore[import-untyped]

            _d2n.dicom_series_to_nifti(str(dicom_dir), str(dest), reorient_nifti=True)
            return dest.exists()
        except Exception as exc:
            logger.warning("dicom2nifti failed ({}); trying SimpleITK fallback", exc)

    if _SITK and _NIBABEL and sitk is not None and nib is not None:
        try:
            reader = sitk.ImageSeriesReader()
            files = reader.GetGDCMSeriesFileNames(str(dicom_dir))
            reader.SetFileNames(files)
            img = reader.Execute()
            arr = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            affine = _sitk_affine(img)
            nii = nib.Nifti1Image(arr, affine)
            nii.header.set_zooms(spacing[:3])
            nib.save(nii, str(dest))
            return dest.exists()
        except Exception as exc:
            logger.warning("SimpleITK/nibabel fallback failed ({})", exc)

    return False


def _sitk_affine(img: Any) -> Any:
    """Build a 4x4 affine from a SimpleITK image."""
    import numpy as np  # already guarded by _NIBABEL/_SITK check

    dir_cos = np.array(img.GetDirection()).reshape(3, 3)
    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    affine = np.eye(4)
    affine[:3, :3] = dir_cos * spacing
    affine[:3, 3] = origin
    return affine


# ---------------------------------------------------------------------------
# Sidecar JSON builder
# ---------------------------------------------------------------------------


def _build_sidecar(
    series_entry: dict[str, Any],
    file_paths: list[str],
    patient: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a BIDS JSON sidecar from a study_full_series_stats.json series entry.

    Args:
        series_entry: One entry from the ``series`` array in study_full_series_stats.json.
        file_paths: List of source DICOM file paths (stored as ``_SourceFiles``).
        patient: Optional ``patient`` block from study_full_series_stats.json.
            Used to fill Manufacturer, ManufacturerModelName, MagneticFieldStrength,
            and AcquisitionTime when series-level values are missing.
    """
    sc = series_entry.get("sequence_classification", {})
    params = series_entry.get("sequence_params", {})
    seq_type = sc.get("sequence_type", "")
    p = patient or {}

    # MagneticFieldStrength: prefer series-level, fall back to patient-level
    field_strength = _coerce_float(params.get("field_strength")) or _coerce_float(
        p.get("field_strength")
    )

    # AcquisitionTime: use study_date from patient block (ISO-8601 date)
    acq_time = _format_acq_time(p.get("study_date", ""))

    sidecar: dict[str, Any] = {
        "Manufacturer": p.get("manufacturer") or None,
        "ManufacturerModelName": p.get("model") or None,
        "MagneticFieldStrength": field_strength,
        "AcquisitionTime": acq_time,
        "SeriesDescription": series_entry.get("series_description", ""),
        "ProtocolName": series_entry.get("series_description", ""),
        "RepetitionTime": _ms_to_s(params.get("tr")),
        "EchoTime": _ms_to_s(params.get("te")),
        "FlipAngle": params.get("fa"),
        "_SourceFiles": file_paths,
    }

    if seq_type == "FLAIR" and params.get("ti") is not None:
        sidecar["InversionTime"] = _ms_to_s(params.get("ti"))

    if seq_type == "DWI":
        sidecar["DiffusionBValue"] = params.get("b_value")

    # Drop None values to keep the sidecar clean
    return {k: v for k, v in sidecar.items() if v is not None}


def _format_acq_time(study_date: str) -> str | None:
    """Convert DICOM YYYYMMDD date string to ISO-8601 date string, or None."""
    if study_date and len(study_date) >= 8:
        try:
            return f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
        except (IndexError, TypeError):
            pass
    return None


def _ms_to_s(val: Any) -> float | None:
    """Convert milliseconds to seconds, returning None for missing values."""
    try:
        return float(val) / 1000.0 if val is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_float(val: Any) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# DWI ancillary files
# ---------------------------------------------------------------------------


def _write_bval_bvec(dest_stem: Path, b_value: float | None) -> None:
    """Write .bval and .bvec files for a DWI series."""
    bval = b_value if b_value is not None else 1000.0
    dest_stem.with_suffix(".bval").write_text(f"{bval:.1f}\n")
    dest_stem.with_suffix(".bvec").write_text(
        "0.0\n0.0\n1.0\n"  # single direction placeholder
    )


# ---------------------------------------------------------------------------
# Dataset-level files
# ---------------------------------------------------------------------------


def _write_dataset_description(bids_root: Path) -> None:
    desc = {
        "Name": "Speall MRI Dataset",
        "BIDSVersion": "1.10.0",
        "DatasetType": "raw",
        "License": "CC-BY-4.0",
        "Authors": ["Speall AI"],
        "Acknowledgements": (
            "Data processed by the Speall MRI pipeline. BIDS conversion via src.bids.converter."
        ),
        "HowToAcknowledge": (
            "Please cite the Speall MRI Dataset and acknowledge the Speall BIDS converter."
        ),
        "Funding": [],
        "ReferencesAndLinks": [
            "https://bids.neuroimaging.io/",
            "https://bids-specification.readthedocs.io/en/stable/",
        ],
    }
    out = bids_root / "dataset_description.json"
    out.write_text(json.dumps(desc, indent=2))


def _write_participants(bids_root: Path, rows: list[dict[str, Any]]) -> None:
    """Write participants.tsv and participants.json."""
    if not rows:
        return
    headers = ["participant_id", "age_bracket", "sex", "site"]
    tsv_lines = ["\t".join(headers)]
    for row in rows:
        tsv_lines.append("\t".join(str(row.get(h, "n/a")) for h in headers))
    (bids_root / "participants.tsv").write_text("\n".join(tsv_lines) + "\n")

    sidecar = {
        "participant_id": {
            "Description": "Unique participant identifier",
        },
        "age_bracket": {
            "Description": "Age bracket (decade) to preserve privacy",
            "Units": "years",
        },
        "sex": {
            "Description": "Biological sex as reported in DICOM header",
            "Levels": {"M": "Male", "F": "Female", "O": "Other/Unknown"},
        },
        "site": {
            "Description": "Imaging institution",
        },
    }
    (bids_root / "participants.json").write_text(json.dumps(sidecar, indent=2))


def _write_readme(bids_root: Path) -> None:
    text = (
        "Speall MRI Dataset\n"
        "==================\n\n"
        "This dataset was processed by the Speall MRI pipeline and converted\n"
        "to Brain Imaging Data Structure (BIDS) 1.10.0 format.\n\n"
        "Modalities included: DWI, FLAIR, T1w, T2w, SWI (SWAN), TOF Angio,\n"
        "ADC Maps, CUBE 3D FSE, and derivative MIP projections.\n\n"
        "For full provenance see series JSON files in the source study directories.\n\n"
        "BIDS specification: https://bids-specification.readthedocs.io/\n"
        "Validator: https://bids-standard.github.io/bids-validator/\n"
    )
    (bids_root / "README").write_text(text)


def _write_derivatives_description(derivatives_root: Path) -> None:
    """Write a minimal dataset_description.json inside derivatives/speall-mips/."""
    derivatives_root.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": "Speall MIP Derivatives",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "Speall MRI pipeline"}],
    }
    (derivatives_root / "dataset_description.json").write_text(json.dumps(desc, indent=2))


# ---------------------------------------------------------------------------
# Series-level conversion
# ---------------------------------------------------------------------------


def _convert_series(
    series_entry: dict[str, Any],
    study_dir: Path,
    sub_dir: Path,
    subject_id: str,
    session: str,
    run_counter: dict[str, int],
    bids_root: Path,
    patient: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Convert a single series entry. Returns a result dict or None on skip."""
    sc = series_entry.get("sequence_classification", {})
    seq_type = sc.get("sequence_type", "")

    if seq_type not in SEQUENCE_TO_BIDS:
        logger.warning("Unknown sequence_type {!r} — skipping series", seq_type)
        return None

    modality_dir, suffix = SEQUENCE_TO_BIDS[seq_type]
    series_desc = series_entry.get("series_description", "")
    acq = infer_acquisition_label(series_desc)

    # Determine output directory
    if modality_dir == "derivatives":
        deriv_base = bids_root / "derivatives" / "speall-mips"
        out_dir = deriv_base / sub_dir.name / f"ses-{session}" / "misc"
        _write_derivatives_description(deriv_base)
    else:
        out_dir = sub_dir / f"ses-{session}" / modality_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # Run counter per (acq, suffix) to avoid filename collisions
    run_key = f"{acq}_{suffix}"
    run_counter[run_key] = run_counter.get(run_key, 0) + 1
    run = run_counter[run_key] if run_counter[run_key] > 1 else None

    stem = bids_filename(subject_id, session, suffix, acq=acq, run=run, ext="")
    nifti_path = out_dir / (stem + ".nii.gz")
    json_path = out_dir / (stem + ".json")

    # Resolve DICOM source
    source_subdir = series_entry.get("source_subdir", "")
    filenames: list[str] = series_entry.get("files", [])
    file_paths: list[str] = series_entry.get("file_paths", [])

    dicom_dir = _resolve_dicom_dir(study_dir, source_subdir, filenames)
    converted = False
    if dicom_dir is not None:
        converted = _convert_dicom_to_nifti(dicom_dir, nifti_path)
    if not converted:
        logger.debug("DICOMs not found for {!r} — writing placeholder NIfTI", series_desc)
        _write_placeholder_nifti(nifti_path)

    # JSON sidecar
    sidecar = _build_sidecar(series_entry, file_paths, patient=patient)
    json_path.write_text(json.dumps(sidecar, indent=2))

    # DWI ancillary files
    params = series_entry.get("sequence_params", {})
    if seq_type == "DWI":
        _write_bval_bvec(out_dir / stem, params.get("b_value"))

    return {
        "series_description": series_desc,
        "sequence_type": seq_type,
        "bids_path": str(nifti_path.relative_to(bids_root)),
        "converted_from_dicom": converted,
        "placeholder": not converted,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_study(
    study_dir: Path,
    bids_root: Path,
    subject_id: str,
    session: str = "01",
) -> dict[str, Any]:
    """Convert one Speall study directory into BIDS layout.

    Reads ``study_dir/study_full_series_stats.json`` and converts each series.
    If DICOMs are absent, writes placeholder ``.nii.gz`` files so the
    directory layout is always correct.

    Args:
        study_dir: Path containing ``study_full_series_stats.json``.
        bids_root: Root of the BIDS output directory tree.
        subject_id: Subject label (will be sanitized to [A-Za-z0-9]+).
        session: Session label, default ``"01"``.

    Returns:
        Dict with keys ``subject``, ``session``, ``series_converted``,
        ``series_results``.

    Raises:
        FileNotFoundError: If ``study_full_series_stats.json`` is not found
            and no series JSON files are present in ``study_dir``.
    """
    bids_root.mkdir(parents=True, exist_ok=True)
    _write_dataset_description(bids_root)
    _write_readme(bids_root)

    sub_clean = re.sub(r"[^A-Za-z0-9]", "", subject_id) or subject_id
    sub_dir = bids_root / f"sub-{sub_clean}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    stats_path = study_dir / "study_full_series_stats.json"
    series_list: list[dict[str, Any]] = []

    if stats_path.exists():
        data = json.loads(stats_path.read_text())
        patient = data.get("patient", {})
        series_list = data.get("series", [])
    else:
        # Fall back to individual series JSON files
        patient = {}
        for json_file in sorted(study_dir.glob("s*.json")):
            try:
                entry = json.loads(json_file.read_text())
                flat = _flatten_series_json(entry)
                series_list.append(flat)
            except Exception as exc:
                logger.warning("Could not parse {}: {}", json_file, exc)

    # Participants row
    sex = patient.get("patient_sex", "n/a") or "n/a"
    site = patient.get("institution", "n/a") or "n/a"
    age_bracket = _age_bracket(patient.get("patient_birth_date", ""))
    participants_row = {
        "participant_id": f"sub-{sub_clean}",
        "age_bracket": age_bracket,
        "sex": sex,
        "site": site,
    }
    _write_participants(bids_root, [participants_row])

    run_counter: dict[str, int] = {}
    results: list[dict[str, Any]] = []

    for entry in series_list:
        result = _convert_series(
            entry,
            study_dir,
            sub_dir,
            subject_id,
            session,
            run_counter,
            bids_root,
            patient=patient,
        )
        if result:
            results.append(result)
            seq = result["sequence_type"]
            desc = result["series_description"]
            bpath = result["bids_path"]
            print(f"  [{seq}] {desc} -> {bpath}")

    return {
        "subject": sub_clean,
        "session": session,
        "series_converted": len(results),
        "series_results": results,
    }


def _flatten_series_json(detail: dict[str, Any]) -> dict[str, Any]:
    """Flatten an individual series detail JSON into study_stats series shape."""
    series = detail.get("series", {})
    return {
        "series_uid": series.get("uid", ""),
        "series_number": series.get("number"),
        "series_description": series.get("description", ""),
        "modality": series.get("modality", "MR"),
        "sop_class": series.get("sop_class", ""),
        "file_count": series.get("file_count", 0),
        "source_subdir": series.get("source_subdir", ""),
        "sequence_params": detail.get("sequence_params", {}),
        "sequence_classification": detail.get("sequence_classification", {}),
        "volume_stats": detail.get("volume_stats", {}),
        "files": detail.get("files", []),
        "file_paths": detail.get("file_paths", []),
    }


def _age_bracket(birth_date: str) -> str:
    """Return decade bracket from DICOM YYYYMMDD date, or 'n/a'."""
    if not birth_date or len(birth_date) < 4:
        return "n/a"
    try:
        birth_year = int(birth_date[:4])
        study_year = 2026  # fixed reference for reproducibility
        age = study_year - birth_year
        decade = (age // 10) * 10
        return f"{decade}s"
    except (ValueError, TypeError):
        return "n/a"


def convert_dataset(corpus_root: Path, bids_root: Path) -> dict[str, Any]:
    """Convert every study subdirectory under corpus_root to BIDS.

    Each subdirectory containing a ``study_full_series_stats.json`` is
    treated as one study. The subject_id is derived from the directory name.

    Args:
        corpus_root: Root directory with one subdirectory per study.
        bids_root: Root of the BIDS output directory.

    Returns:
        Dict with ``total_studies``, ``total_series``, ``study_results``.
    """
    bids_root.mkdir(parents=True, exist_ok=True)
    study_results = []
    subject_counter = 1

    study_dirs = sorted(
        d
        for d in corpus_root.iterdir()
        if d.is_dir() and (d / "study_full_series_stats.json").exists()
    )

    if not study_dirs:
        logger.warning(
            "No study directories with study_full_series_stats.json found in {}",
            corpus_root,
        )

    for study_dir in study_dirs:
        subject_id = f"{subject_counter:03d}"
        print(f"\nConverting study {study_dir.name} -> sub-{subject_id}")
        result = convert_study(study_dir, bids_root, subject_id)
        result["study_dir"] = str(study_dir)
        study_results.append(result)
        subject_counter += 1

    total_series = sum(r.get("series_converted", 0) for r in study_results)
    return {
        "total_studies": len(study_results),
        "total_series": total_series,
        "study_results": study_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert Speall MRI studies to BIDS format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Corpus root (directory containing per-study subdirectories).",
    )
    p.add_argument(
        "--bids-out",
        type=Path,
        required=True,
        help="BIDS output root directory (will be created if needed).",
    )
    p.add_argument(
        "--subject",
        default=None,
        help=(
            "For single-study mode: convert --root as a single study with this "
            "subject ID. If omitted, --root is treated as a corpus of multiple studies."
        ),
    )
    p.add_argument(
        "--session",
        default="01",
        help="Session label (used in single-study mode).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    p.add_argument(
        "--with-zarr",
        action="store_true",
        default=False,
        help=(
            "After BIDS conversion, also export each subject's NIfTI files "
            "to OME-Zarr multiscale format alongside the BIDS output. "
            "Requires zarr and ome-zarr to be installed."
        ),
    )
    return p


def _export_zarr_alongside(bids_root: Path) -> None:
    """Export every sub-* directory in bids_root to OME-Zarr alongside BIDS.

    The Zarr output is written to ``bids_root/zarr/`` so it lives next to
    the BIDS tree.  Requires zarr and ome-zarr to be installed.
    """
    try:
        from src.zarr_export.converter import study_to_omezarr  # lazy
    except ImportError:
        logger.warning("--with-zarr: zarr/ome-zarr not installed; skipping Zarr export.")
        return

    zarr_root = bids_root / "zarr"
    zarr_root.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting OME-Zarr alongside BIDS -> {zarr_root}")

    subject_dirs = sorted(
        d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")
    )
    for sub_dir in subject_dirs:
        print(f"  zarr: {sub_dir.name} ...")
        stats = study_to_omezarr(sub_dir, zarr_root)
        print(
            f"    -> {stats['study_zarr']}  "
            f"({stats['series_converted']} series, {stats['series_failed']} failed)"
        )


def main() -> None:
    from src._logging import configure as _configure_logging

    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(level="DEBUG" if args.verbose else "INFO", force=True)

    if args.subject is not None:
        print(f"Converting single study: {args.root} -> {args.bids_out}")
        result = convert_study(
            study_dir=args.root,
            bids_root=args.bids_out,
            subject_id=args.subject,
            session=args.session,
        )
        print(f"\nDone. Converted {result['series_converted']} series.")
    else:
        print(f"Converting corpus: {args.root} -> {args.bids_out}")
        result = convert_dataset(args.root, args.bids_out)
        print(
            f"\nDone. Converted {result['total_studies']} studies, "
            f"{result['total_series']} series total."
        )

    if getattr(args, "with_zarr", False):
        _export_zarr_alongside(args.bids_out)


if __name__ == "__main__":
    main()

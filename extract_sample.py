#!/usr/bin/env python3
"""Extract MAXIMUM information from 10 DICOM studies — all local, no Modal, no API.

Dumps everything into a single JSON per study:
  - Every DICOM tag (all groups, all elements)
  - Pixel statistics (min, max, mean, std, percentiles, entropy, skewness, kurtosis)
  - Volume geometry (shape, spacing, origin, direction, FOV, voxel volume)
  - Sequence classification (T1/T2/FLAIR/DWI etc. with confidence)
  - Quality analysis (SNR, CNR, motion, symmetry, sharpness, anomalies, grade A-F)
  - Advanced quality (noise floor, bias field, edge sharpness, histogram separation, inter-slice consistency)
  - ML training score (0-100, commercial tier)
  - Series grouping (which files belong to which series)
  - Patient demographics (age, sex, weight — from DICOM headers)
  - Scanner info (manufacturer, model, field strength, software, coil)
  - Protocol parameters (TR, TE, TI, FA, b-value, bandwidth, matrix, FOV)
  - Conformance check (missing required tags)

Usage:
    python3 extract_sample.py                              # first 10 studies from T7 Shield
    python3 extract_sample.py --input ./mcap-files --n 5   # 5 studies from local dir
"""

import argparse
import contextlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def extract_all_tags(fpath: str) -> dict:
    """Extract EVERY tag from a DICOM file — maximum information."""
    import pydicom

    try:
        ds = pydicom.dcmread(fpath, force=True)
    except Exception as e:
        return {"_error": str(e), "_filepath": fpath}

    record = {
        "_filepath": fpath,
        "_filename": Path(fpath).name,
        "_file_size_bytes": Path(fpath).stat().st_size,
    }

    # Every single tag
    for elem in ds:
        if elem.tag.group == 0x7FE0:  # pixel data — skip binary
            record["_has_pixel_data"] = True
            continue

        kw = elem.keyword or f"Tag_{elem.tag.group:04X}_{elem.tag.element:04X}"
        tag_str = f"({elem.tag.group:04X},{elem.tag.element:04X})"

        try:
            val = elem.value
            if isinstance(val, bytes):
                record[kw] = f"<bytes:{len(val)}>"
            elif isinstance(val, pydicom.sequence.Sequence):
                record[kw] = f"<sequence:{len(val)} items>"
            else:
                record[kw] = str(val)
            record[f"_tag_{kw}"] = tag_str
            record[f"_vr_{kw}"] = str(elem.VR)
        except Exception:
            record[kw] = "<unreadable>"

    # File meta
    if hasattr(ds, "file_meta"):
        for elem in ds.file_meta:
            kw = elem.keyword or f"Meta_{elem.tag.group:04X}_{elem.tag.element:04X}"
            with contextlib.suppress(Exception):
                record[f"_meta_{kw}"] = str(elem.value)

    return record


def extract_study(study_dir: Path, study_idx: int) -> dict:
    """Extract ALL information from a single study."""
    import numpy as np

    t0 = time.time()
    dcm_files = sorted(study_dir.rglob("*.dcm"))

    if not dcm_files:
        return {"study": study_dir.name, "error": "no DCM files"}

    print(f"  [{study_idx}] {study_dir.name}: {len(dcm_files)} files...", flush=True)

    # ── Extract all tags from every file ─────────────────────────────────
    all_records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(extract_all_tags, str(f)): f for f in dcm_files}
        for fut in as_completed(futs):
            all_records.append(fut.result())

    all_records.sort(key=lambda r: r.get("_filename", ""))

    # ── Group by series ──────────────────────────────────────────────────
    series_groups = defaultdict(list)
    series_meta = {}
    for r in all_records:
        uid = r.get("SeriesInstanceUID", "unknown")
        series_groups[uid].append(r)
        if uid not in series_meta:
            series_meta[uid] = {
                "series_number": r.get("SeriesNumber", ""),
                "series_description": r.get("SeriesDescription", ""),
                "modality": r.get("Modality", ""),
                "sop_class_uid": r.get("SOPClassUID", ""),
                "rows": r.get("Rows", ""),
                "columns": r.get("Columns", ""),
                "bits_allocated": r.get("BitsAllocated", ""),
                "pixel_spacing": r.get("PixelSpacing", ""),
                "slice_thickness": r.get("SliceThickness", ""),
                "spacing_between_slices": r.get("SpacingBetweenSlices", ""),
                "repetition_time": r.get("RepetitionTime", ""),
                "echo_time": r.get("EchoTime", ""),
                "inversion_time": r.get("InversionTime", ""),
                "flip_angle": r.get("FlipAngle", ""),
                "magnetic_field_strength": r.get("MagneticFieldStrength", ""),
                "image_type": r.get("ImageType", ""),
                "scanning_sequence": r.get("ScanningSequence", ""),
                "sequence_variant": r.get("SequenceVariant", ""),
                "file_count": len(series_groups[uid]),
            }

    # ── Patient info ─────────────────────────────────────────────────────
    r0 = all_records[0]
    patient_info = {
        "patient_id": r0.get("PatientID", ""),
        "patient_name": r0.get("PatientName", ""),
        "patient_sex": r0.get("PatientSex", ""),
        "patient_birth_date": r0.get("PatientBirthDate", ""),
        "patient_age": r0.get("PatientAge", ""),
        "patient_weight": r0.get("PatientWeight", ""),
        "patient_size": r0.get("PatientSize", ""),
        "study_date": r0.get("StudyDate", ""),
        "study_time": r0.get("StudyTime", ""),
        "study_description": r0.get("StudyDescription", ""),
        "accession_number": r0.get("AccessionNumber", ""),
        "referring_physician": r0.get("ReferringPhysicianName", ""),
        "institution": r0.get("InstitutionName", ""),
        "institution_address": r0.get("InstitutionAddress", ""),
        "station_name": r0.get("StationName", ""),
        "department": r0.get("InstitutionalDepartmentName", ""),
    }

    # ── Scanner info ─────────────────────────────────────────────────────
    scanner_info = {
        "manufacturer": r0.get("Manufacturer", ""),
        "manufacturer_model": r0.get("ManufacturerModelName", ""),
        "software_versions": r0.get("SoftwareVersions", ""),
        "magnetic_field_strength": r0.get("MagneticFieldStrength", ""),
        "device_serial_number": r0.get("DeviceSerialNumber", ""),
        "station_name": r0.get("StationName", ""),
        "transfer_syntax": r0.get("_meta_TransferSyntaxUID", ""),
    }

    # ── Sequence classification ──────────────────────────────────────────
    try:
        from src.extraction import classify_sequence

        for uid, meta in series_meta.items():
            tr = None
            te = None
            ti = None
            fa = None
            bval = None
            with contextlib.suppress(ValueError, TypeError):
                tr = float(meta["repetition_time"]) if meta["repetition_time"] else None
            with contextlib.suppress(ValueError, TypeError):
                te = float(meta["echo_time"]) if meta["echo_time"] else None
            with contextlib.suppress(ValueError, TypeError):
                ti = float(meta["inversion_time"]) if meta["inversion_time"] else None
            with contextlib.suppress(ValueError, TypeError):
                fa = float(meta["flip_angle"]) if meta["flip_angle"] else None

            seq_class = classify_sequence(meta.get("series_description", ""), tr, te, ti, fa, bval)
            meta["sequence_classification"] = seq_class
    except Exception as e:
        print(f"    Sequence classification failed: {e}")

    # ── Volume stats + quality per series ────────────────────────────────
    series_analysis = {}
    try:
        import SimpleITK as sitk

        from src.advanced_quality import full_quality_assessment
        from src.extraction import volume_stats
        from src.helpers import safe_squeeze
        from src.quality import (
            compute_sharpness,
            compute_symmetry,
            detect_anomalous_slices,
            detect_motion_artifacts,
            grade_series,
        )

        for uid, files in series_groups.items():
            meta = series_meta.get(uid, {})
            file_paths = [r["_filepath"] for r in files if r.get("_filepath")]

            if len(file_paths) < 2:
                continue

            try:
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(file_paths[:200])
                reader.SetGlobalWarningDisplay(False)
                img = reader.Execute()
                vol = sitk.GetArrayFromImage(img).astype(np.float32)
                vol = safe_squeeze(vol)

                if vol.ndim < 3 or vol.shape[0] < 3:
                    continue

                # Volume stats
                vs = volume_stats(vol, img)

                # Quality
                grade = grade_series(vs, meta.get("series_description", ""))
                anomalies = detect_anomalous_slices(vol)
                symmetry = compute_symmetry(vol, meta.get("series_description", ""))
                sharpness = compute_sharpness(vol)
                motion = detect_motion_artifacts(vol)

                # Advanced quality
                advanced = full_quality_assessment(vol, meta.get("series_description", ""))

                series_analysis[uid] = {
                    "volume_stats": vs,
                    "quality_grade": grade,
                    "anomaly_detection": anomalies,
                    "symmetry_analysis": symmetry,
                    "sharpness_analysis": sharpness,
                    "motion_analysis": motion,
                    "advanced_quality": advanced,
                    "ml_training_score": advanced.get("ml_training_score", {}),
                }

                snum = meta.get("series_number", "?")
                sdesc = meta.get("series_description", "?")
                g = grade.get("grade", "?")
                snr = vs.get("volume_snr_estimate", 0)
                tier = advanced.get("ml_training_score", {}).get("commercial_tier", "?")
                print(f"    s{snum} {sdesc}: grade={g} SNR={snr:.1f} tier={tier}")

            except Exception as e:
                series_analysis[uid] = {"error": str(e)}

    except Exception as e:
        print(f"    Volume analysis failed: {e}")

    # ── Conformance check ────────────────────────────────────────────────
    conformance = []
    try:
        from src.extraction import check_conformance

        issues = check_conformance(all_records)
        conformance = issues
    except Exception:
        pass

    # ── Unique tags inventory ────────────────────────────────────────────
    all_tags = set()
    private_tags = set()
    for r in all_records:
        for k in r:
            if not k.startswith("_"):
                all_tags.add(k)
                tag_key = f"_tag_{k}"
                if tag_key in r:
                    tag_str = r[tag_key]
                    group = int(tag_str[1:5], 16)
                    if group % 2 == 1:
                        private_tags.add(k)

    elapsed = time.time() - t0
    print(
        f"    Done in {elapsed:.1f}s: {len(all_records)} files, "
        f"{len(series_groups)} series, {len(all_tags)} unique tags, "
        f"{len(private_tags)} private tags"
    )

    return {
        "study_name": study_dir.name,
        "study_path": str(study_dir),
        "total_files": len(dcm_files),
        "total_series": len(series_groups),
        "total_unique_tags": len(all_tags),
        "total_private_tags": len(private_tags),
        "private_tag_names": sorted(private_tags),
        "extraction_time_s": round(elapsed, 1),
        "patient": patient_info,
        "scanner": scanner_info,
        "series": {
            uid: {
                **meta,
                "file_count": len(series_groups[uid]),
                "analysis": series_analysis.get(uid, {}),
            }
            for uid, meta in series_meta.items()
        },
        "conformance_issues": conformance[:20],
        "sample_record": all_records[0] if all_records else {},
    }


def main():
    parser = argparse.ArgumentParser(description="Extract maximum info from DICOM studies")
    parser.add_argument("--input", default="/Volumes/T7 Shield/Akai - MRI", help="Input directory")
    parser.add_argument("--n", type=int, default=10, help="Number of studies to process")
    parser.add_argument("--output", default="sample_extraction.json", help="Output JSON file")
    args = parser.parse_args()

    root = Path(args.input)
    if not root.exists():
        # Fallback to mcap-files
        root = Path("mcap-files")
    if not root.exists():
        print(f"Error: {root} not found")
        sys.exit(1)

    # Find study directories
    study_dirs = sorted(
        [
            d
            for d in root.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and d.suffix != ".zip"
            and any(d.rglob("*.dcm"))
        ]
    )[: args.n]

    print(f"Extracting {len(study_dirs)} studies from {root.name}")
    print(f"Output: {args.output}")
    print()

    t0 = time.time()
    results = []
    for i, sd in enumerate(study_dirs):
        result = extract_study(sd, i + 1)
        results.append(result)

    # Write output
    output = {
        "extraction_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(root),
        "studies_extracted": len(results),
        "total_time_s": round(time.time() - t0, 1),
        "studies": results,
    }

    Path(args.output).write_text(json.dumps(output, indent=2, default=str))
    total_size = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nDone: {len(results)} studies → {args.output} ({total_size:.1f} MB)")


if __name__ == "__main__":
    main()

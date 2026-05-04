"""Series processing — load volume, compute stats, export images (threaded)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError

# Suppress ITK "Non uniform sampling" warnings globally at import time.
# These fire on DWI, angiography, multi-echo — normal for these sequences.
sitk.ProcessObject.SetGlobalWarningDisplay(False)

from ._logging import get_logger
from .constants import NON_IMAGE_SOP, SOP_CLASS_NAMES
from .exports import (
    export_enhanced_views,
    export_histogram,
    export_multiplane_montage,
    export_nifti,
)
from .extraction import classify_sequence, shm_read, volume_stats
from .helpers import safe_getfloat, safe_squeeze, to_json
from .io.msgspec_io import write_detail
from .quality import analyze_volume_quality

log = get_logger(__name__)

# Study-level SimpleITK probe: if the first file can't be read by SimpleITK,
# skip it for all series (avoids N×failed attempts). Reset per pipeline run.
_sitk_works: bool | None = None  # None = untested, True/False = probed


def reset_sitk_probe() -> None:
    """Reset the SimpleITK probe for a new pipeline run."""
    global _sitk_works
    _sitk_works = None
    # Suppress ITK warnings globally — "Non uniform sampling" fires on DWI,
    # angiography, multi-echo sequences where slice spacing naturally varies.
    sitk.ProcessObject.SetGlobalWarningDisplay(False)


def _sort_dicom_by_position(
    file_paths: list[str],
    file_records: list[dict] | None = None,
) -> list[str]:
    """Sort DICOM files by slice position for correct spatial ordering.

    Uses pre-extracted _z_position and _instance_number from stage 2 records
    when available (zero I/O). Falls back to reading headers only when records
    are not provided.
    """
    if len(file_paths) <= 1:
        return file_paths

    # Build lookup from pre-extracted records (O(1) per file, no disk I/O)
    rec_by_path: dict[str, dict] = {}
    if file_records:
        for r in file_records:
            fp = r.get("_filepath", "")
            if fp:
                rec_by_path[fp] = r

    keyed: list[tuple[float, int, str, str]] = []
    for fp in file_paths:
        rec = rec_by_path.get(fp)
        if rec is not None:
            z = rec.get("_z_position")
            z = z if z is not None else float("inf")
            inst = rec.get("_instance_number", 0)
            keyed.append((z, inst, Path(fp).name, fp))
        else:
            # Fallback: read header (only when no records available)
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                pos = getattr(ds, "ImagePositionPatient", None)
                z = float(pos[2]) if pos is not None and len(pos) >= 3 else float("inf")
                inst = int(getattr(ds, "InstanceNumber", 0) or 0)
                keyed.append((z, inst, Path(fp).name, fp))
            except (InvalidDicomError, OSError, AttributeError, ValueError, TypeError) as e:
                log.debug("Header read failed during sort for {}: {}", fp, e)
                keyed.append((float("inf"), 0, Path(fp).name, fp))

    keyed.sort()
    return [fp for _, _, _, fp in keyed]


def _read_one_slice(fp: str, shm_rec: dict | None) -> tuple[str, np.ndarray | None]:
    """Read a single DICOM slice — from shared memory or disk. Thread-safe."""
    try:
        if shm_rec is not None:
            arr = shm_read(
                shm_rec["_shm_name"],
                shm_rec["_shm_shape"],
                shm_rec["_shm_dtype"],
            )
            return fp, arr

        ds = pydicom.dcmread(fp, force=True)
        raw = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        offset = float(getattr(ds, "RescaleIntercept", 0.0))
        return fp, raw * slope + offset
    except Exception as e:
        log.debug("Slice read failed for {}: {}", fp, e)
        return fp, None


def _stack_pydicom_volume(
    sorted_paths: list[str],
    desc: str,
    file_records: list[dict] | None = None,
    n_workers: int = 8,
) -> np.ndarray | None:
    """Stack DICOM slices into a 3D volume using threaded I/O.

    Reads all slices in parallel threads (pydicom releases GIL during
    decompression). Pre-allocates the output array and writes slices
    directly into it — avoids building a list + np.stack copy.
    """
    # Build path → shared memory lookup
    shm_by_path: dict[str, dict] = {}
    if file_records:
        for r in file_records:
            if r.get("_shm_name"):
                shm_by_path[r["_filepath"]] = r

    # Read all slices in parallel — adaptive thread count
    n_readers = min(len(sorted_paths), max(n_workers, 8))
    slice_map: dict[str, np.ndarray] = {}

    with ThreadPoolExecutor(max_workers=n_readers) as pool:
        futures = [pool.submit(_read_one_slice, fp, shm_by_path.get(fp)) for fp in sorted_paths]
        for fut in futures:
            fp, arr = fut.result()
            if arr is not None:
                slice_map[fp] = arr

    if not slice_map:
        return None

    # Determine reference shape from first valid slice
    ref_shape = next(iter(slice_map.values())).shape
    valid_indices = [
        i
        for i, fp in enumerate(sorted_paths)
        if fp in slice_map and slice_map[fp].shape == ref_shape
    ]

    if not valid_indices:
        return None
    if len(valid_indices) == 1:
        return safe_squeeze(slice_map[sorted_paths[valid_indices[0]]])

    # Pre-allocate contiguous array and fill directly — no intermediate list
    vol = np.empty((len(valid_indices), *ref_shape), dtype=np.float64)
    for out_idx, src_idx in enumerate(valid_indices):
        vol[out_idx] = slice_map[sorted_paths[src_idx]]

    return safe_squeeze(vol)


@dataclass
class SeriesResult:
    uid: str
    info: dict
    vstats: dict | None = None
    vol: np.ndarray | None = field(default=None, repr=False)
    sitk_img: sitk.Image | None = field(default=None, repr=False)
    montage_path: str | None = None
    histogram_path: str | None = None
    enhanced_path: str | None = None
    label: str = ""
    series_folder: str | None = None


def process_one_series(
    uid: str,
    file_paths: list[str],
    meta: dict,
    out_dir: str,
    do_export_nii: bool,
    seq_index: int = 0,
    source_subdir: str = "",
    file_records: list[dict] | None = None,
    conformance_issues: list[dict] | None = None,
    n_workers: int = 8,
    mcap_only: bool = False,
    enable_zarr: bool = False,
) -> SeriesResult:
    """Process a single series: load volume, compute stats, export outputs.

    When mcap_only=True, skips montages, histograms, enhanced views, and
    quality analysis — only writes per-series MCAP + detail JSON. Use this
    when images are already generated/uploaded elsewhere.

    source_subdir: relative path from root input folder to the subfolder
    where the DICOM files live — mirrors input hierarchy in output.
    """
    desc = meta.get("series_description", "unknown")
    snum = meta.get("series_number", "?")
    sop_uid = meta.get("sop_class_uid", "")
    try:
        snum_pad = f"{int(snum):04d}"
    except (ValueError, TypeError):
        snum_pad = str(snum)
    safe_name = f"s{snum_pad}_{desc}".replace(" ", "_").replace("/", "-").replace("*", "x")

    # Mirror input subfolder hierarchy in output:
    #   output/<source_subdir>/<series_name>/
    if source_subdir:
        series_folder = Path(out_dir) / source_subdir / safe_name
    else:
        series_folder = Path(out_dir) / safe_name

    is_img = sop_uid not in NON_IMAGE_SOP

    info = {
        "series_uid": uid,
        "series_number": snum,
        "series_description": desc,
        "modality": meta.get("modality", ""),
        "sop_class": SOP_CLASS_NAMES.get(sop_uid, sop_uid),
        "sop_class_uid": sop_uid,
        "file_count": len(file_paths),
        "has_pixels": is_img,
    }
    if source_subdir:
        info["source_subdir"] = source_subdir

    result = SeriesResult(uid=uid, info=info, label=f"{snum} {desc}"[:25])

    if not is_img:
        info["note"] = "Presentation state — skipped"
        return result

    # Sequence params — reuse from extraction records if available (avoids re-reading DICOM)
    seq_params = _extract_seq_params(file_records, file_paths)
    info["sequence_params"] = seq_params
    info["sequence_classification"] = classify_sequence(
        desc,
        seq_params.get("tr"),
        seq_params.get("te"),
        seq_params.get("ti"),
        seq_params.get("fa"),
        seq_params.get("b_value"),
    )

    # Sort files by slice position for correct spatial ordering.
    # Uses pre-extracted position data from stage 2 (no disk I/O).
    sorted_paths = _sort_dicom_by_position(file_paths, file_records)

    # Load volume — SimpleITK first (if it works for this study), then pydicom.
    # After the first SimpleITK failure in a study, skip it for all remaining
    # series to avoid N × failed-attempt overhead (~50ms each).
    global _sitk_works
    vol, sitk_img = None, None

    if _sitk_works is not False:
        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(sorted_paths)
            image = reader.Execute()
            vol = sitk.GetArrayFromImage(image)
            sitk_img = image
            _sitk_works = True
        except Exception as e:
            if _sitk_works is None:
                log.debug("SimpleITK probe failed — skipping for remaining series: {}", e)
                _sitk_works = False
            else:
                log.debug("SimpleITK failed for {}: {}", desc, e)

    if vol is not None:
        vol = safe_squeeze(vol)
    else:
        vol = _stack_pydicom_volume(sorted_paths, desc, file_records, n_workers)

    if vol is not None and vol.size > 0:
        # volume_stats directly — no thread pool overhead for a single task
        vs = volume_stats(vol, sitk_img)

        info["volume_stats"] = vs
        result.vstats = vs
        result.vol = vol
        result.sitk_img = sitk_img

        # Quality analysis — skip when mcap_only (expensive FFT/symmetry)
        if not mcap_only:
            n_voxels = vol.size
            if n_voxels > 10_000:
                qa = analyze_volume_quality(vol, vs, desc)
            else:
                from .quality import grade_series

                qa = {
                    "quality_grade": grade_series(vs, desc),
                    "anomaly_detection": {"n_anomalous": 0, "anomalous_slices": []},
                    "symmetry_analysis": {"symmetry_index": 1.0, "interpretation": "N/A (small)"},
                    "sharpness_analysis": {"sharpness_mean": 0, "interpretation": "N/A"},
                    "motion_analysis": {"motion_detected": False, "interpretation": "N/A"},
                }
            info["quality_analysis"] = qa
            vs["quality_grade"] = qa["quality_grade"]["grade"]
            vs["quality_score"] = qa["quality_grade"]["score"]

        series_folder.mkdir(parents=True, exist_ok=True)
        result.series_folder = str(series_folder)

        if enable_zarr and vol.ndim == 3:
            _safe_write_zarr(vol, sitk_img, series_folder, safe_name, uid, info)

        if mcap_only:
            # Lightweight path: only MCAP + detail JSON, no image exports
            with ThreadPoolExecutor(max_workers=2) as export_pool:
                detail_fut = export_pool.submit(
                    _write_series_detail,
                    series_folder,
                    safe_name,
                    info,
                    file_paths,
                    file_records or [],
                    conformance_issues or [],
                )
                mcap_fut = export_pool.submit(
                    _write_series_mcap,
                    series_folder,
                    safe_name,
                    uid,
                    info,
                    file_records or [],
                )
                detail_fut.result()
                mcap_fut.result()
        else:
            # Full path: montages, histograms, enhanced views, MCAP, detail
            do_enhanced = vol.ndim >= 3 and vol.shape[0] >= 3
            with ThreadPoolExecutor(max_workers=6) as export_pool:
                montage_fut = export_pool.submit(
                    export_multiplane_montage,
                    vol,
                    safe_name,
                    str(series_folder),
                    6,
                    vs,
                )
                hist_fut = export_pool.submit(
                    export_histogram,
                    vol,
                    safe_name,
                    str(series_folder),
                    vs,
                )
                if do_enhanced:
                    enhanced_fut = export_pool.submit(
                        export_enhanced_views,
                        vol,
                        safe_name,
                        str(series_folder),
                        vs,
                    )
                detail_fut = export_pool.submit(
                    _write_series_detail,
                    series_folder,
                    safe_name,
                    info,
                    file_paths,
                    file_records or [],
                    conformance_issues or [],
                )
                if do_export_nii and sitk_img:
                    export_pool.submit(
                        _safe_export_nifti,
                        file_paths,
                        str(series_folder / f"{safe_name}.nii.gz"),
                    )

                # Wait for images first so we can embed them in MCAP
                result.montage_path = montage_fut.result()
                result.histogram_path = hist_fut.result()
                result.enhanced_path = enhanced_fut.result() if do_enhanced else None
                detail_fut.result()

                # Write MCAP with embedded images
                image_paths = {
                    "montage": result.montage_path,
                    "histogram": result.histogram_path,
                    "enhanced": result.enhanced_path,
                }
                mcap_fut = export_pool.submit(
                    _write_series_mcap,
                    series_folder,
                    safe_name,
                    uid,
                    info,
                    file_records or [],
                    image_paths,
                )
                mcap_fut.result()

    return result


def _write_series_detail(
    series_out: Path,
    safe_name: str,
    info: dict,
    file_paths: list[str],
    file_records: list[dict],
    conformance_issues: list[dict],
) -> None:
    """Write detailed per-series JSON into the series folder.

    Each folder gets a self-contained detail file with: series metadata,
    sequence classification, volume stats, per-file pixel stats,
    conformance issues, and the full file listing.
    """
    detail = {
        "series": {
            "uid": info.get("series_uid", ""),
            "number": info.get("series_number", ""),
            "description": info.get("series_description", ""),
            "modality": info.get("modality", ""),
            "sop_class": info.get("sop_class", ""),
            "file_count": info.get("file_count", 0),
            "source_subdir": info.get("source_subdir", ""),
        },
        "sequence_classification": info.get("sequence_classification", {}),
        "sequence_params": info.get("sequence_params", {}),
        "volume_stats": info.get("volume_stats", {}),
        "files": [Path(fp).name for fp in sorted(file_paths)],
        "file_paths": sorted(file_paths),
        "conformance_issues": conformance_issues,
        "conformance_summary": {
            "total_files": len(file_paths),
            "files_with_issues": len(conformance_issues),
            "pass_rate": round(100 * (1 - len(conformance_issues) / max(len(file_paths), 1)), 1),
        },
    }

    if file_records:
        # Simple list comp — dict key lookups are O(1), no thread overhead needed
        detail["per_file_stats"] = [
            {
                "filename": r.get("_filename", ""),
                "pixel_shape": r.get("pixel_shape"),
                "pixel_min": r.get("pixel_min"),
                "pixel_max": r.get("pixel_max"),
                "pixel_mean": r.get("pixel_mean"),
                "pixel_std": r.get("pixel_std"),
                "pixel_entropy": r.get("pixel_entropy"),
                "nonzero_ratio": r.get("nonzero_ratio"),
            }
            for r in file_records
        ]

    p = series_out / f"{safe_name}_detail.json"
    write_detail(p, to_json(detail), indent=2)


def _write_series_mcap(
    series_out: Path,
    safe_name: str,
    uid: str,
    info: dict,
    file_records: list[dict],
    image_paths: dict[str, str | None] | None = None,
) -> Path | None:
    """Write a per-series MCAP file into the series folder.

    Each series gets its own self-contained .mcap with:
      - One channel for per-file records
      - One summary message
      - Image channels for montage, histogram, enhanced views (raw PNG bytes)
      - Series metadata
      - Zstd-7 compression, 512KB chunks

    Images are embedded as raw PNG binary messages — viewers can render them
    directly without needing the filesystem PNGs.
    """
    if not file_records:
        return None

    try:
        from time import time_ns

        import zstandard
        from mcap.writer import CompressionType, Writer

        mcap_path = series_out / f"{safe_name}.mcap"

        record_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "series_uid": {"type": "string"},
                    "modality": {"type": "string"},
                    "pixel_stats": {"type": "object"},
                    "sequence_params": {"type": "object"},
                    "tags": {"type": "object"},
                },
            }
        )
        summary_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "series_uid": {"type": "string"},
                    "series_description": {"type": "string"},
                    "file_count": {"type": "integer"},
                    "volume_stats": {"type": "object"},
                },
            }
        )

        _orig_compress = zstandard.compress
        _zctx = zstandard.ZstdCompressor(level=7)
        zstandard.compress = lambda data, level=7: _zctx.compress(data)

        try:
            with open(mcap_path, "wb") as f:
                writer = Writer(f, chunk_size=512 * 1024, compression=CompressionType.ZSTD)
                writer.start(profile="dicom-series", library="micom-series-mcap")

                rec_schema_id = writer.register_schema(
                    name="dicom_record",
                    encoding="jsonschema",
                    data=record_schema.encode(),
                )
                sum_schema_id = writer.register_schema(
                    name="series_summary",
                    encoding="jsonschema",
                    data=summary_schema.encode(),
                )

                snum = info.get("series_number", "?")
                desc = info.get("series_description", "unknown")
                topic = f"/dicom/series/{snum}_{desc}".replace(" ", "_")

                ch_id = writer.register_channel(
                    topic=topic,
                    message_encoding="json",
                    schema_id=rec_schema_id,
                    metadata={
                        "series_uid": uid,
                        "series_number": str(snum),
                        "series_description": desc,
                        "modality": info.get("modality", ""),
                        "file_count": str(len(file_records)),
                    },
                )
                sum_ch_id = writer.register_channel(
                    topic=f"{topic}/summary",
                    message_encoding="json",
                    schema_id=sum_schema_id,
                )

                # Per-file messages
                for seq, r in enumerate(file_records):
                    msg = {
                        "filename": r.get("_filename", ""),
                        "series_uid": uid,
                        "modality": r.get("_modality", ""),
                        "pixel_stats": {
                            "shape": r.get("pixel_shape"),
                            "min": r.get("pixel_min"),
                            "max": r.get("pixel_max"),
                            "mean": r.get("pixel_mean"),
                            "std": r.get("pixel_std"),
                        },
                        "sequence_params": {
                            "tr": r.get("_tr"),
                            "te": r.get("_te"),
                            "ti": r.get("_ti"),
                            "fa": r.get("_fa"),
                            "b_value": r.get("_b_value"),
                        },
                        "tags": {
                            k: to_json(v)
                            for k, v in r.items()
                            if not k.startswith("_")
                            and k not in ("histogram_counts", "histogram_edges")
                        },
                    }
                    now = time_ns()
                    writer.add_message(
                        channel_id=ch_id,
                        log_time=now,
                        publish_time=now,
                        data=json.dumps(msg, default=str).encode(),
                        sequence=seq,
                    )

                # Summary message with volume stats
                now = time_ns()
                writer.add_message(
                    channel_id=sum_ch_id,
                    log_time=now,
                    publish_time=now,
                    data=json.dumps(
                        {
                            "series_uid": uid,
                            "series_description": desc,
                            "file_count": len(file_records),
                            "volume_stats": info.get("volume_stats", {}),
                            "quality_analysis": info.get("quality_analysis", {}),
                        },
                        default=str,
                    ).encode(),
                    sequence=0,
                )

                # ── Embed images as raw PNG messages ──────────────────────
                if image_paths:
                    img_schema_id = writer.register_schema(
                        name="png_image",
                        encoding="octet-stream",
                        data=b"",
                    )
                    for img_type, img_path in image_paths.items():
                        if not img_path or not Path(img_path).exists():
                            continue
                        png_data = Path(img_path).read_bytes()
                        img_ch_id = writer.register_channel(
                            topic=f"{topic}/images/{img_type}",
                            message_encoding="octet-stream",
                            schema_id=img_schema_id,
                            metadata={
                                "mime_type": "image/png",
                                "image_type": img_type,
                                "size_bytes": str(len(png_data)),
                                "source_file": Path(img_path).name,
                            },
                        )
                        now = time_ns()
                        writer.add_message(
                            channel_id=img_ch_id,
                            log_time=now,
                            publish_time=now,
                            data=png_data,
                            sequence=0,
                        )

                # Metadata
                writer.add_metadata(
                    name="series_info",
                    data={
                        "series_uid": uid,
                        "series_number": str(snum),
                        "series_description": desc,
                        "modality": info.get("modality", ""),
                        "sop_class": info.get("sop_class", ""),
                        "file_count": str(len(file_records)),
                    },
                )

                writer.finish()
        finally:
            zstandard.compress = _orig_compress

        return mcap_path

    except Exception:
        log.exception("Per-series MCAP write failed for {}", safe_name)
        return None


def _extract_seq_params(file_records: list[dict] | None, file_paths: list[str]) -> dict:
    """Get sequence params from pre-extracted records (no I/O), falling back to DICOM read."""
    # Try records first — already extracted in stage 2, zero I/O
    if file_records:
        for r in file_records:
            tr = r.get("_tr")
            te = r.get("_te")
            if tr is not None or te is not None:
                return {
                    "tr": tr,
                    "te": te,
                    "ti": r.get("_ti"),
                    "fa": r.get("_fa"),
                    "b_value": r.get("_b_value"),
                    "slice_thickness": None,
                    "spacing_between_slices": None,
                    "rows": None,
                    "columns": None,
                    "field_strength": None,
                    "pixel_spacing": "",
                }

    # Fallback: read first file (only if no records available)
    for fp in file_paths[:1]:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            return {
                "tr": safe_getfloat(ds, "RepetitionTime"),
                "te": safe_getfloat(ds, "EchoTime"),
                "ti": safe_getfloat(ds, "InversionTime"),
                "fa": safe_getfloat(ds, "FlipAngle"),
                "slice_thickness": safe_getfloat(ds, "SliceThickness"),
                "spacing_between_slices": safe_getfloat(ds, "SpacingBetweenSlices"),
                "rows": safe_getfloat(ds, "Rows"),
                "columns": safe_getfloat(ds, "Columns"),
                "field_strength": safe_getfloat(ds, "MagneticFieldStrength"),
                "pixel_spacing": str(getattr(ds, "PixelSpacing", "")),
                "b_value": safe_getfloat(ds, "DiffusionBValue"),
            }
        except (InvalidDicomError, OSError, AttributeError, ValueError, TypeError) as e:
            log.debug("Failed to read seq params from {}: {}", fp, e)
    return {}


def _safe_export_nifti(file_paths: list[str], out_path: str) -> None:
    try:
        export_nifti(file_paths, out_path)
    except Exception:
        log.exception("NIfTI export failed for {}", out_path)


def _safe_write_zarr(
    vol: np.ndarray,
    sitk_img: sitk.Image | None,
    series_folder: Path,
    safe_name: str,
    uid: str,
    info: dict,
) -> None:
    """Write per-series OME-Zarr group next to detail.json; silent on failure.

    Voxel spacing is read from sitk_img if available (XYZ order from ITK,
    reversed to ZYX for OME-Zarr).  Falls back to volume_stats spacing_mm
    which is also in XYZ order.  A 1 mm isotropic default is used if neither
    is available.
    """
    try:
        from .zarr_export.series_writer import series_volume_to_omezarr

        # Derive ZYX spacing: SimpleITK GetSpacing() -> (sx, sy, sz) XYZ
        if sitk_img is not None:
            sp_xyz = list(sitk_img.GetSpacing())[:3]
            sp_zyx = (float(sp_xyz[2]), float(sp_xyz[1]), float(sp_xyz[0]))
        else:
            sp_mm = info.get("volume_stats", {}).get("spacing_mm", [1.0, 1.0, 1.0])
            sp_mm = ([*list(sp_mm), 1.0, 1.0, 1.0])[:3]
            sp_zyx = (float(sp_mm[2]), float(sp_mm[1]), float(sp_mm[0]))

        seq_type = info.get("sequence_classification", {}).get("sequence_type") or None
        zarr_path = series_folder / f"{safe_name}.zarr"
        series_volume_to_omezarr(
            vol,
            zarr_path,
            voxel_spacing_mm=sp_zyx,
            series_uid=uid,
            sequence_type=seq_type,
        )
    except Exception:
        log.exception("OME-Zarr write failed for {}", safe_name)

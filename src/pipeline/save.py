"""Stage 3 — Save JSON + CSV + MCAP exports with optional compression."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time_ns

import polars as pl
from rich.console import Console

from ..compression import (
    compress_file_multi,
    compression_ratio,
    format_size,
)
from ..helpers import to_json

console = Console()

_CSV_PRIORITY_COLS = [
    "_filename",
    "_series_number",
    "_series_description",
    "_modality",
    "PatientID",
    "PatientName",
    "PatientSex",
    "PatientBirthDate",
    "StudyDate",
    "StudyDescription",
    "Modality",
    "BodyPartExamined",
    "SeriesNumber",
    "SeriesDescription",
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturerModelName",
    "RepetitionTime",
    "EchoTime",
    "FlipAngle",
    "InversionTime",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PixelSpacing",
    "Rows",
    "Columns",
    "BitsAllocated",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PhotometricInterpretation",
    "WindowCenter",
    "WindowWidth",
    "pixel_shape",
    "pixel_min",
    "pixel_max",
    "pixel_mean",
    "pixel_std",
    "pixel_median",
    "pixel_p5",
    "pixel_p95",
    "nonzero_ratio",
    "pixel_entropy",
    "pixel_skewness",
    "pixel_kurtosis",
]

# MCAP JSON schema for per-file DICOM records
_DICOM_RECORD_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "filename": {"type": "string"},
            "filepath": {"type": "string"},
            "series_uid": {"type": "string"},
            "series_number": {"type": ["string", "number"]},
            "series_description": {"type": "string"},
            "modality": {"type": "string"},
            "sop_class_uid": {"type": "string"},
            "pixel_stats": {
                "type": "object",
                "properties": {
                    "shape": {"type": "array", "items": {"type": "integer"}},
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                    "mean": {"type": "number"},
                    "std": {"type": "number"},
                    "entropy": {"type": "number"},
                    "snr": {"type": "number"},
                },
            },
            "sequence_params": {
                "type": "object",
                "properties": {
                    "tr": {"type": ["number", "null"]},
                    "te": {"type": ["number", "null"]},
                    "ti": {"type": ["number", "null"]},
                    "fa": {"type": ["number", "null"]},
                    "b_value": {"type": ["number", "null"]},
                },
            },
            "tags": {"type": "object"},
        },
    }
)

# MCAP JSON schema for series-level summary
_SERIES_SUMMARY_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "series_uid": {"type": "string"},
            "series_number": {"type": ["string", "number"]},
            "series_description": {"type": "string"},
            "modality": {"type": "string"},
            "file_count": {"type": "integer"},
            "volume_stats": {"type": "object"},
        },
    }
)


def save_data(all_records: list[dict], out_dir: Path, compress: bool = False) -> None:
    """Write JSON dump, CSV, and MCAP, then optionally compress."""
    t_save = time.time()

    def _build_json_records() -> list[dict]:
        return [{k: to_json(v) for k, v in r.items()} for r in all_records]

    def _build_tabular_records() -> list[dict]:
        result = []
        for r in all_records:
            flat = {}
            for k, v in r.items():
                if k in ("histogram_counts", "histogram_edges"):
                    continue
                flat[k] = str(v) if isinstance(v, list) else to_json(v)
            result.append(flat)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        json_records_fut = pool.submit(_build_json_records)
        tabular_records_fut = pool.submit(_build_tabular_records)

        json_records = json_records_fut.result()
        tabular_records = tabular_records_fut.result()

        def _write_json():
            text = json.dumps(json_records, indent=2, default=str)
            p = out_dir / "dicom_full_dump.json"
            p.write_text(text)
            return p, len(text.encode())

        def _build_df():
            df = pl.DataFrame(tabular_records, infer_schema_length=None)
            existing_priority = [c for c in _CSV_PRIORITY_COLS if c in df.columns]
            remaining = sorted(c for c in df.columns if c not in existing_priority)
            return df.select(existing_priority + remaining)

        json_fut = pool.submit(_write_json)
        df_fut = pool.submit(_build_df)
        mcap_fut = pool.submit(_write_mcap, all_records, out_dir)

        json_path, json_size = json_fut.result()
        df = df_fut.result()

        def _write_csv():
            p = out_dir / "dicom_metadata.csv"
            df.write_csv(p)
            return p, p.stat().st_size

        csv_fut = pool.submit(_write_csv)
        csv_path, csv_size = csv_fut.result()
        mcap_path, mcap_size, mcap_msg_count, mcap_ch_count = mcap_fut.result()

    n_rows, n_cols = df.shape
    console.print(f"[green]✓[/green] JSON → {json_path} ({format_size(json_size)})")
    console.print(
        f"[green]✓[/green] CSV  → {csv_path} ({n_rows} rows × {n_cols} cols, {format_size(csv_size)})"
    )
    console.print(
        f"[green]✓[/green] MCAP → {mcap_path} "
        f"({mcap_msg_count} msgs, {mcap_ch_count} channels, "
        f"{format_size(mcap_size)}, zstd-7)"
    )

    if compress:
        _compress_outputs(json_path, json_size, csv_path, csv_size)

    console.print(f"[dim]  Saved in {time.time() - t_save:.1f}s[/dim]")


def _write_mcap(all_records: list[dict], out_dir: Path) -> tuple[Path, int, int, int]:
    """Write all DICOM records to an MCAP file with zstd-7 compression.

    Structure:
      - Schema: dicom_record (per-file), series_summary (per-series)
      - Channel per series (topic = /dicom/series/<series_uid_short>)
      - One message per file record
      - Metadata: patient info, study info
    """
    import zstandard  # pyright: ignore[reportMissingImports]  # optional dep
    from mcap.writer import (  # pyright: ignore[reportMissingImports]  # optional dep
        CompressionType,
        Writer,
    )

    mcap_path = out_dir / "dicom_study.mcap"

    # Patch zstandard.compress to use level 7 for this writer
    _orig_compress = zstandard.compress
    _zctx = zstandard.ZstdCompressor(level=7)

    def _zstd7_compress(data: bytes, level: int = 7) -> bytes:
        return _zctx.compress(data)

    zstandard.compress = _zstd7_compress  # type: ignore[assignment]

    try:
        with open(mcap_path, "wb") as f:
            writer = Writer(
                f,
                chunk_size=1024 * 1024,  # 1 MB chunks
                compression=CompressionType.ZSTD,
            )
            writer.start(profile="dicom", library="micom-dicom-extractor")

            # Register schemas
            record_schema_id = writer.register_schema(
                name="dicom_record",
                encoding="jsonschema",
                data=_DICOM_RECORD_SCHEMA.encode(),
            )
            summary_schema_id = writer.register_schema(
                name="series_summary",
                encoding="jsonschema",
                data=_SERIES_SUMMARY_SCHEMA.encode(),
            )

            # Group records by series, register a channel per series
            from collections import defaultdict

            series_groups: dict[str, list[dict]] = defaultdict(list)
            for r in all_records:
                uid = r.get("_series_uid", "unknown")
                series_groups[uid].append(r)

            channel_ids: dict[str, int] = {}
            summary_channel_ids: dict[str, int] = {}
            for uid in sorted(series_groups.keys()):
                recs = series_groups[uid]
                r0 = recs[0]
                desc = r0.get("_series_description", "unknown")
                snum = r0.get("_series_number", "?")
                safe_topic = f"/dicom/series/{snum}_{desc}".replace(" ", "_")

                channel_ids[uid] = writer.register_channel(
                    topic=safe_topic,
                    message_encoding="json",
                    schema_id=record_schema_id,
                    metadata={
                        "series_uid": uid,
                        "series_number": str(snum),
                        "series_description": desc,
                        "modality": r0.get("_modality", ""),
                        "file_count": str(len(recs)),
                    },
                )
                summary_channel_ids[uid] = writer.register_channel(
                    topic=f"{safe_topic}/summary",
                    message_encoding="json",
                    schema_id=summary_schema_id,
                )

            # Write per-file messages
            msg_count = 0
            for uid in sorted(series_groups.keys()):
                ch_id = channel_ids[uid]
                for seq, r in enumerate(series_groups[uid]):
                    msg = {
                        "filename": r.get("_filename", ""),
                        "filepath": r.get("_filepath", ""),
                        "series_uid": uid,
                        "series_number": r.get("_series_number", ""),
                        "series_description": r.get("_series_description", ""),
                        "modality": r.get("_modality", ""),
                        "sop_class_uid": r.get("_sop_class_uid", ""),
                        "pixel_stats": {
                            "shape": r.get("pixel_shape"),
                            "min": r.get("pixel_min"),
                            "max": r.get("pixel_max"),
                            "mean": r.get("pixel_mean"),
                            "std": r.get("pixel_std"),
                            "entropy": r.get("pixel_entropy"),
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
                    msg_count += 1

                # Write series summary message
                recs = series_groups[uid]
                r0 = recs[0]
                summary = {
                    "series_uid": uid,
                    "series_number": r0.get("_series_number", ""),
                    "series_description": r0.get("_series_description", ""),
                    "modality": r0.get("_modality", ""),
                    "file_count": len(recs),
                }
                now = time_ns()
                writer.add_message(
                    channel_id=summary_channel_ids[uid],
                    log_time=now,
                    publish_time=now,
                    data=json.dumps(summary, default=str).encode(),
                    sequence=0,
                )
                msg_count += 1

            # Store patient/study info as MCAP metadata
            if all_records:
                r0 = all_records[0]
                writer.add_metadata(
                    name="patient_info",
                    data={
                        "patient_id": str(r0.get("_patient_id", "")),
                        "patient_name": str(r0.get("_patient_name", "")),
                        "patient_sex": str(r0.get("_patient_sex", "")),
                        "patient_birth_date": str(r0.get("_patient_birth_date", "")),
                    },
                )
                writer.add_metadata(
                    name="study_info",
                    data={
                        "study_date": str(r0.get("_study_date", "")),
                        "study_description": str(r0.get("_study_description", "")),
                        "institution": str(r0.get("_institution", "")),
                        "manufacturer": str(r0.get("_manufacturer", "")),
                        "model": str(r0.get("_model", "")),
                        "field_strength": str(r0.get("_field_strength", "")),
                    },
                )
                writer.add_metadata(
                    name="extraction_info",
                    data={
                        "total_files": str(len(all_records)),
                        "total_series": str(len(series_groups)),
                        "unique_modalities": ",".join(
                            sorted(
                                {r.get("_modality", "") for r in all_records if r.get("_modality")}
                            )
                        ),
                    },
                )

            writer.finish()
    finally:
        zstandard.compress = _orig_compress

    mcap_size = mcap_path.stat().st_size
    return mcap_path, mcap_size, msg_count, len(channel_ids)


def _compress_outputs(json_path: Path, json_size: int, csv_path: Path, csv_size: int) -> None:
    """Compress JSON and CSV using centralized compression module."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        json_fut = pool.submit(compress_file_multi, json_path)
        csv_fut = pool.submit(compress_file_multi, csv_path)
        json_results = json_fut.result()
        csv_results = csv_fut.result()

    for label, orig_size, results in [
        ("JSON", json_size, json_results),
        ("CSV", csv_size, csv_results),
    ]:
        console.print(f"  [cyan]Compressed {label}[/cyan] ({format_size(orig_size)}):")
        for fmt, (_, compressed_size) in sorted(results.items()):
            console.print(
                f"    {fmt} → {format_size(compressed_size)} ({compression_ratio(orig_size, compressed_size)})"
            )

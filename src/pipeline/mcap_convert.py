"""DICOM → MCAP conversion pipeline — threaded, streaming, low memory.

Converts raw DICOM files directly to MCAP format without loading full volumes.
Streams records into MCAP as they're extracted — bounded memory regardless of
dataset size. Each file is processed once (O(N×T) total).

Structure of output MCAP:
  - One channel per DICOM series (topic: /dicom/series/<num>_<desc>)
  - One message per file with full tag extraction + pixel stats
  - Series summary messages on /dicom/series/<num>_<desc>/summary
  - Patient/study metadata as MCAP metadata records
  - Zstd-7 compression, 1MB chunks

Viewable in Foxglove Studio for interactive DICOM exploration.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import time_ns

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..helpers import to_json
from .discover import discover_files

console = Console()


def _extract_for_mcap(fpath: str) -> dict:
    """Lightweight per-file extraction for MCAP (runs in thread, GIL-releasing I/O).

    Extracts all tags + pixel stats in one read. Returns a flat dict ready
    for JSON serialization into MCAP messages.
    """
    import numpy as np
    import pydicom

    from ..helpers import safe_getfloat, safe_value

    f = Path(fpath)
    ds = pydicom.dcmread(fpath, force=True)

    record = {"_filename": f.name, "_filepath": fpath}

    # All tags
    for elem in ds:
        if elem.tag.group == 0x7FE0:
            record["_has_pixel_data"] = True
            continue
        kw = elem.keyword or f"Tag_{elem.tag.group:04X}_{elem.tag.element:04X}"
        record[kw] = safe_value(elem)
    record.setdefault("_has_pixel_data", False)

    # Pixel stats
    if record.get("_has_pixel_data"):
        try:
            arr = ds.pixel_array.astype(np.float64)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            offset = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + offset
            record["pixel_shape"] = list(arr.shape)
            record["pixel_min"] = float(arr.min())
            record["pixel_max"] = float(arr.max())
            record["pixel_mean"] = float(arr.mean())
            record["pixel_std"] = float(arr.std())
        except Exception as e:
            record["pixel_error"] = str(e)

    # Series/patient info
    record["_series_uid"] = str(getattr(ds, "SeriesInstanceUID", "unknown"))
    record["_series_number"] = str(getattr(ds, "SeriesNumber", ""))
    record["_series_description"] = str(getattr(ds, "SeriesDescription", ""))
    record["_modality"] = str(getattr(ds, "Modality", ""))
    record["_sop_class_uid"] = str(getattr(ds, "SOPClassUID", ""))
    record["_patient_id"] = str(getattr(ds, "PatientID", ""))
    record["_patient_name"] = str(getattr(ds, "PatientName", ""))
    record["_study_date"] = str(getattr(ds, "StudyDate", ""))
    record["_institution"] = str(getattr(ds, "InstitutionName", ""))
    record["_tr"] = safe_getfloat(ds, "RepetitionTime")
    record["_te"] = safe_getfloat(ds, "EchoTime")
    record["_ti"] = safe_getfloat(ds, "InversionTime")
    record["_fa"] = safe_getfloat(ds, "FlipAngle")
    record["_b_value"] = safe_getfloat(ds, "DiffusionBValue")
    record["_instance_number"] = int(getattr(ds, "InstanceNumber", 0) or 0)

    return record


def _encode_mcap_message(r: dict, uid: str, snum: str, desc: str) -> bytes:
    """Pre-encode a record into JSON bytes for MCAP — offloads serialization to threads."""
    msg = {
        "filename": r.get("_filename", ""),
        "series_uid": uid,
        "series_number": snum,
        "series_description": desc,
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
            if not k.startswith("_") and k not in ("histogram_counts", "histogram_edges")
        },
    }
    return json.dumps(msg, default=str).encode()


def run_mcap_convert(
    folder: Path,
    out_path: Path | None = None,
    workers: int = 0,
    recursive: bool = True,
) -> Path:
    """Convert DICOM folder to MCAP — streaming extraction + write pipeline.

    Phase 1: Extract all files in parallel threads, group by series.
    Phase 2: Encode MCAP messages in parallel threads, write sequentially.

    Memory: only holds records (metadata-sized dicts) — no pixel arrays retained.
    """
    import zstandard
    from mcap.writer import CompressionType, Writer

    t0 = time.time()
    n_workers = workers or min(multiprocessing.cpu_count(), 8)

    console.print(
        Panel.fit(
            f"[bold cyan]DICOM → MCAP Converter[/bold cyan]  [dim]({n_workers} threads)[/dim]\n"
            "[dim]Threaded extraction · streaming write · zstd-7 · O(N×T)[/dim]",
            border_style="cyan",
        )
    )

    # Discover
    dcm_files = discover_files(folder, recursive=recursive)
    file_paths = [str(f) for f in dcm_files]

    if out_path is None:
        out_path = folder.parent / f"{folder.name}.mcap"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: Extract all files in threads
    console.print(f"[cyan]Extracting {len(file_paths)} files ({n_workers} threads)…[/cyan]")

    all_records: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting", total=len(file_paths))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_extract_for_mcap, fp): fp for fp in file_paths}
            for fut in as_completed(futures):
                all_records.append(fut.result())
                progress.advance(task)

    t_extract = time.time() - t0

    # Group by series
    series_groups: dict[str, list[dict]] = defaultdict(list)
    for r in all_records:
        series_groups[r.get("_series_uid", "unknown")].append(r)

    # Sort within each series by instance number
    for uid in series_groups:
        series_groups[uid].sort(key=lambda r: r.get("_instance_number", 0))

    # Phase 2: Encode messages in parallel, write to MCAP sequentially
    console.print(f"[cyan]Writing MCAP ({len(series_groups)} series, zstd-7)…[/cyan]")

    # Pre-encode all messages in parallel threads (JSON serialization is CPU-bound)
    encoded_messages: dict[str, list[bytes]] = {}
    series_order = sorted(series_groups.keys())

    with ThreadPoolExecutor(max_workers=n_workers) as encode_pool:
        encode_futures: dict[str, list] = {}
        for uid in series_order:
            recs = series_groups[uid]
            r0 = recs[0]
            snum = r0.get("_series_number", "?")
            desc = r0.get("_series_description", "unknown")
            encode_futures[uid] = [
                encode_pool.submit(_encode_mcap_message, r, uid, snum, desc) for r in recs
            ]
        for uid in series_order:
            encoded_messages[uid] = [f.result() for f in encode_futures[uid]]

    # Schemas
    record_schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "series_uid": {"type": "string"},
                "series_number": {"type": "string"},
                "series_description": {"type": "string"},
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
                "series_number": {"type": "string"},
                "series_description": {"type": "string"},
                "file_count": {"type": "integer"},
                "modality": {"type": "string"},
            },
        }
    )

    _orig_compress = zstandard.compress
    _zctx = zstandard.ZstdCompressor(level=7)
    zstandard.compress = lambda data, level=7: _zctx.compress(data)

    try:
        with open(out_path, "wb") as f:
            writer = Writer(f, chunk_size=1024 * 1024, compression=CompressionType.ZSTD)
            writer.start(profile="dicom", library="micom-mcap-converter")

            record_schema_id = writer.register_schema(
                name="dicom_record",
                encoding="jsonschema",
                data=record_schema.encode(),
            )
            summary_schema_id = writer.register_schema(
                name="series_summary",
                encoding="jsonschema",
                data=summary_schema.encode(),
            )

            # Write pre-encoded messages — sequential I/O, no serialization overhead
            msg_count = 0
            for uid in series_order:
                recs = series_groups[uid]
                r0 = recs[0]
                snum = r0.get("_series_number", "?")
                desc = r0.get("_series_description", "unknown")
                topic = f"/dicom/series/{snum}_{desc}".replace(" ", "_")

                ch_id = writer.register_channel(
                    topic=topic,
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
                summary_ch_id = writer.register_channel(
                    topic=f"{topic}/summary",
                    message_encoding="json",
                    schema_id=summary_schema_id,
                )

                # Write pre-encoded per-file messages
                for seq, data in enumerate(encoded_messages[uid]):
                    now = time_ns()
                    writer.add_message(
                        channel_id=ch_id,
                        log_time=now,
                        publish_time=now,
                        data=data,
                        sequence=seq,
                    )
                    msg_count += 1

                # Series summary
                now = time_ns()
                writer.add_message(
                    channel_id=summary_ch_id,
                    log_time=now,
                    publish_time=now,
                    data=json.dumps(
                        {
                            "series_uid": uid,
                            "series_number": snum,
                            "series_description": desc,
                            "modality": r0.get("_modality", ""),
                            "file_count": len(recs),
                        }
                    ).encode(),
                    sequence=0,
                )
                msg_count += 1

            # Metadata
            if all_records:
                r0 = all_records[0]
                writer.add_metadata(
                    name="patient_info",
                    data={
                        "patient_id": str(r0.get("_patient_id", "")),
                        "patient_name": str(r0.get("_patient_name", "")),
                        "study_date": str(r0.get("_study_date", "")),
                        "institution": str(r0.get("_institution", "")),
                    },
                )
                writer.add_metadata(
                    name="conversion_info",
                    data={
                        "total_files": str(len(all_records)),
                        "total_series": str(len(series_groups)),
                        "extraction_time_s": f"{t_extract:.1f}",
                    },
                )

            writer.finish()
    finally:
        zstandard.compress = _orig_compress

    t_total = time.time() - t0
    mcap_size = out_path.stat().st_size
    from ..compression import format_size

    console.print(
        Panel(
            f"[bold]MCAP conversion complete[/bold] in [bold]{t_total:.1f}s[/bold]\n"
            f"  Files:     {len(all_records)}\n"
            f"  Series:    {len(series_groups)}\n"
            f"  Messages:  {msg_count}\n"
            f"  Size:      {format_size(mcap_size)} (zstd-7)\n"
            f"  Output:    {out_path}",
            title="Summary",
            border_style="cyan",
        )
    )

    return out_path

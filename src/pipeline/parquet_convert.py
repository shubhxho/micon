"""Universal file → Parquet conversion — any file type, threaded, columnar.

Converts an entire folder (recursively) to a single Parquet file with:
  - One row per file
  - Auto-detects file type and extracts appropriate metadata
  - Supported: DICOM (.dcm), JSON, CSV, NIfTI (.nii/.nii.gz), images (png/jpg),
    text files, and any other file (basic metadata: name, size, modified time)
  - Threaded extraction for I/O parallelism
  - Snappy compression

Usage:
  uv run main.py parquet ./my_folder -o ./output.parquet
  uv run main.py parquet ./my_folder --ext dcm,json,png
"""

from __future__ import annotations

import json
import mimetypes
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def _extract_file_metadata(fpath: str) -> dict:
    """Extract metadata from any file based on its type. Runs in a thread."""
    f = Path(fpath)
    stat = f.stat()

    # Base metadata for every file
    record = {
        "filepath": fpath,
        "filename": f.name,
        "stem": f.stem,
        "extension": f.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "parent_dir": f.parent.name,
        "relative_path": str(f),
        "mime_type": mimetypes.guess_type(fpath)[0] or "application/octet-stream",
    }

    ext = f.suffix.lower()

    # Route to type-specific extractor
    if ext == ".dcm" or ext == "":
        _extract_dicom(fpath, record)
    elif ext == ".json":
        _extract_json(fpath, record)
    elif ext == ".csv":
        _extract_csv(fpath, record)
    elif ext in (".nii", ".gz") and ".nii" in f.name:
        _extract_nifti(fpath, record)
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
        _extract_image(fpath, record)
    elif ext in (".txt", ".md", ".log", ".yaml", ".yml", ".toml", ".ini", ".cfg"):
        _extract_text(fpath, record)
    elif ext == ".parquet":
        _extract_parquet(fpath, record)

    return record


def _extract_dicom(fpath: str, record: dict) -> None:
    """Extract DICOM tags and pixel stats."""
    try:
        import numpy as np
        import pydicom

        from ..helpers import safe_getfloat

        ds = pydicom.dcmread(fpath, force=True)
        record["file_type"] = "dicom"

        # Core DICOM tags
        record["dicom_patient_id"] = str(getattr(ds, "PatientID", ""))
        record["dicom_patient_name"] = str(getattr(ds, "PatientName", ""))
        record["dicom_study_date"] = str(getattr(ds, "StudyDate", ""))
        record["dicom_modality"] = str(getattr(ds, "Modality", ""))
        record["dicom_series_uid"] = str(getattr(ds, "SeriesInstanceUID", ""))
        record["dicom_series_number"] = str(getattr(ds, "SeriesNumber", ""))
        record["dicom_series_description"] = str(getattr(ds, "SeriesDescription", ""))
        record["dicom_institution"] = str(getattr(ds, "InstitutionName", ""))
        record["dicom_manufacturer"] = str(getattr(ds, "Manufacturer", ""))
        record["dicom_rows"] = int(getattr(ds, "Rows", 0) or 0)
        record["dicom_columns"] = int(getattr(ds, "Columns", 0) or 0)
        record["dicom_instance_number"] = int(getattr(ds, "InstanceNumber", 0) or 0)
        record["dicom_tr"] = safe_getfloat(ds, "RepetitionTime")
        record["dicom_te"] = safe_getfloat(ds, "EchoTime")
        record["dicom_ti"] = safe_getfloat(ds, "InversionTime")
        record["dicom_fa"] = safe_getfloat(ds, "FlipAngle")
        record["dicom_b_value"] = safe_getfloat(ds, "DiffusionBValue")
        record["dicom_slice_thickness"] = safe_getfloat(ds, "SliceThickness")
        record["dicom_sop_class_uid"] = str(getattr(ds, "SOPClassUID", ""))

        # Pixel stats
        has_pixels = any(elem.tag.group == 0x7FE0 for elem in ds)
        record["has_pixel_data"] = has_pixels
        if has_pixels:
            try:
                arr = ds.pixel_array.astype(np.float64)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                offset = float(getattr(ds, "RescaleIntercept", 0.0))
                arr = arr * slope + offset
                record["pixel_shape"] = str(list(arr.shape))
                record["pixel_min"] = float(arr.min())
                record["pixel_max"] = float(arr.max())
                record["pixel_mean"] = float(arr.mean())
                record["pixel_std"] = float(arr.std())
            except Exception:
                pass

        # Count total tags
        record["dicom_tag_count"] = sum(1 for _ in ds)

    except Exception as e:
        record["file_type"] = "dicom_error"
        record["error"] = str(e)


def _extract_json(fpath: str, record: dict) -> None:
    """Extract JSON file metadata."""
    try:
        text = Path(fpath).read_text(errors="replace")
        data = json.loads(text)
        record["file_type"] = "json"
        record["json_type"] = type(data).__name__
        if isinstance(data, list):
            record["json_length"] = len(data)
        elif isinstance(data, dict):
            record["json_keys"] = str(list(data.keys())[:20])
            record["json_length"] = len(data)
        record["text_length"] = len(text)
    except Exception as e:
        record["file_type"] = "json_error"
        record["error"] = str(e)


def _extract_csv(fpath: str, record: dict) -> None:
    """Extract CSV file metadata."""
    try:
        df = pl.read_csv(fpath, n_rows=0, infer_schema_length=0)
        record["file_type"] = "csv"
        record["csv_columns"] = len(df.columns)
        record["csv_column_names"] = str(df.columns[:20])
        # Count rows without loading full file
        with open(fpath) as f:
            record["csv_rows"] = sum(1 for _ in f) - 1
    except Exception as e:
        record["file_type"] = "csv_error"
        record["error"] = str(e)


def _extract_nifti(fpath: str, record: dict) -> None:
    """Extract NIfTI file metadata."""
    try:
        import nibabel as nib

        img = nib.load(fpath)
        record["file_type"] = "nifti"
        record["nifti_shape"] = str(list(img.shape))  # pyright: ignore[reportAttributeAccessIssue]
        record["nifti_ndim"] = len(img.shape)  # pyright: ignore[reportAttributeAccessIssue]
        record["nifti_dtype"] = str(img.get_data_dtype())  # pyright: ignore[reportAttributeAccessIssue]
        record["nifti_voxel_size"] = str(list(img.header.get_zooms()))  # pyright: ignore[reportAttributeAccessIssue]
        affine = img.affine  # pyright: ignore[reportAttributeAccessIssue]
        if affine is not None:
            record["nifti_orientation"] = str(nib.aff2axcodes(affine))
    except Exception as e:
        record["file_type"] = "nifti_error"
        record["error"] = str(e)


def _extract_image(fpath: str, record: dict) -> None:
    """Extract image file metadata."""
    try:
        from PIL import Image

        img = Image.open(fpath)
        record["file_type"] = "image"
        record["image_width"] = img.width
        record["image_height"] = img.height
        record["image_mode"] = img.mode
        record["image_format"] = img.format or Path(fpath).suffix[1:].upper()
    except Exception as e:
        record["file_type"] = "image_error"
        record["error"] = str(e)


def _extract_text(fpath: str, record: dict) -> None:
    """Extract text file metadata."""
    try:
        text = Path(fpath).read_text(errors="replace")
        record["file_type"] = "text"
        record["text_length"] = len(text)
        record["text_lines"] = text.count("\n") + 1
        record["text_words"] = len(text.split())
    except Exception as e:
        record["file_type"] = "text_error"
        record["error"] = str(e)


def _extract_parquet(fpath: str, record: dict) -> None:
    """Extract parquet file metadata."""
    try:
        schema = pl.read_parquet_schema(fpath)
        record["file_type"] = "parquet"
        record["parquet_columns"] = len(schema)
        record["parquet_column_names"] = str(list(schema.keys())[:20])
    except Exception as e:
        record["file_type"] = "parquet_error"
        record["error"] = str(e)


def run_parquet_convert(
    folder: Path,
    out_path: Path | None = None,
    workers: int = 0,
    recursive: bool = True,
    exts: list[str] | None = None,
) -> Path:
    """Convert any folder of files to Parquet — threaded, universal.

    Args:
        folder: Root folder to scan.
        out_path: Output .parquet path (default: <folder>.parquet).
        workers: Thread count (default: CPU count).
        exts: Optional list of extensions to filter (e.g. ["dcm", "json"]).
              If None, includes ALL files.
    """
    t0 = time.time()
    n_workers = workers or min(multiprocessing.cpu_count(), 8)

    console.print(
        Panel.fit(
            f"[bold green]Universal → Parquet Converter[/bold green]  [dim]({n_workers} threads)[/dim]\n"
            "[dim]Any file type · threaded · columnar · snappy compressed[/dim]",
            border_style="green",
        )
    )

    # Discover all files recursively
    all_files: list[Path] = []
    if recursive:
        for root, _, files in os.walk(folder):
            for fname in files:
                fp = Path(root) / fname
                if exts:
                    if any(fp.name.lower().endswith(f".{e}") for e in exts):
                        all_files.append(fp)
                else:
                    all_files.append(fp)
    else:
        for entry in os.scandir(folder):
            if entry.is_file():
                fp = Path(entry.path)
                if exts:
                    if any(fp.name.lower().endswith(f".{e}") for e in exts):
                        all_files.append(fp)
                else:
                    all_files.append(fp)

    all_files.sort()

    if not all_files:
        console.print(f"[red]No files found in {folder}[/red]")
        raise SystemExit(1)

    # Show file type breakdown
    ext_counts: dict[str, int] = {}
    for f in all_files:
        e = f.suffix.lower() or "(none)"
        ext_counts[e] = ext_counts.get(e, 0) + 1
    ext_summary = ", ".join(
        f"{e}: {c}" for e, c in sorted(ext_counts.items(), key=lambda x: -x[1])[:10]
    )
    console.print(f"Found [bold]{len(all_files)}[/bold] files — {ext_summary}\n")

    if out_path is None:
        out_path = folder.parent / f"{folder.name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract metadata in parallel threads
    console.print(f"[green]Extracting metadata from {len(all_files)} files…[/green]")

    records: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[green]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting", total=len(all_files))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_extract_file_metadata, str(f)): f for f in all_files}
            for fut in as_completed(futures):
                records.append(fut.result())
                progress.advance(task)

    records.sort(key=lambda r: r.get("filepath", ""))

    # Build DataFrame and write Parquet
    console.print("[green]Writing Parquet…[/green]")
    df = pl.DataFrame(records, infer_schema_length=None)

    # Reorder: base fields first, then type-specific
    base_cols = [
        "filepath",
        "filename",
        "stem",
        "extension",
        "file_type",
        "size_bytes",
        "modified",
        "created",
        "parent_dir",
        "mime_type",
    ]
    existing_base = [c for c in base_cols if c in df.columns]
    remaining = sorted(c for c in df.columns if c not in existing_base)
    df = df.select(existing_base + remaining)

    df.write_parquet(out_path, compression="snappy")

    t_total = time.time() - t0
    file_size = out_path.stat().st_size
    from ..compression import format_size

    # Type breakdown
    type_counts = {}
    for r in records:
        ft = r.get("file_type", "unknown")
        type_counts[ft] = type_counts.get(ft, 0) + 1
    type_str = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))

    console.print(
        Panel(
            f"[bold]Parquet conversion complete[/bold] in [bold]{t_total:.1f}s[/bold]\n"
            f"  Files:    {len(records)}\n"
            f"  Types:    {type_str}\n"
            f"  Columns:  {len(df.columns)}\n"
            f"  Rows:     {len(df)}\n"
            f"  Size:     {format_size(file_size)} (snappy)\n"
            f"  Output:   {out_path}",
            title="Summary",
            border_style="green",
        )
    )

    return out_path

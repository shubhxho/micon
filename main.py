#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydicom>=3.0",
#   "SimpleITK>=2.4",
#   "nibabel>=5.3",
#   "polars>=1.17",
#   "numpy>=2.1",
#   "rich>=13.9",
#   "typer>=0.15",
#   "anthropic>=0.40",
#   "matplotlib>=3.10",
#   "scipy>=1.14",
#   "pillow>=11.0",
# ]
# ///

"""
dcm_extract.py — Full DICOM brain scan extraction + Claude analysis
Usage:  uv run dcm_extract.py <folder> [--claude] [--export-nii] [--slices]
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import anthropic
import nibabel as nib
import numpy as np
import pydicom
import polars as pl
import SimpleITK as sitk
import typer
from pydicom.uid import UID
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

app = typer.Typer(rich_markup_mode="rich")
console = Console()

# ── DICOM tag groups to extract ───────────────────────────────────────────────
TAG_GROUPS = {
    "Patient":     ["PatientID", "PatientName", "PatientBirthDate", "PatientSex", "PatientWeight"],
    "Study":       ["StudyDate", "StudyTime", "StudyDescription", "StudyInstanceUID", "AccessionNumber"],
    "Series":      ["SeriesNumber", "SeriesDescription", "SeriesInstanceUID", "Modality", "BodyPartExamined"],
    "Acquisition": [
        "MagneticFieldStrength", "ScanningSequence", "SequenceVariant",
        "RepetitionTime", "EchoTime", "FlipAngle", "InversionTime",
        "SliceThickness", "SpacingBetweenSlices", "PixelSpacing",
        "Rows", "Columns", "NumberOfTemporalPositions",
        "ImageOrientationPatient", "ImagePositionPatient",
        "PhotometricInterpretation", "BitsAllocated", "HighBit",
    ],
    "Equipment":   ["Manufacturer", "ManufacturerModelName", "SoftwareVersions",
                    "MagneticFieldStrength", "InstitutionName", "StationName"],
    "Display":     ["WindowCenter", "WindowWidth", "RescaleSlope", "RescaleIntercept"],
}


def safe_get(ds: pydicom.Dataset, tag: str) -> str:
    try:
        val = getattr(ds, tag)
        if isinstance(val, pydicom.sequence.Sequence):
            return f"[sequence, {len(val)} items]"
        return str(val)
    except AttributeError:
        return ""


def extract_pixel_stats(ds: pydicom.Dataset) -> dict:
    """Extract pixel array + compute stats."""
    try:
        arr = ds.pixel_array.astype(np.float32)
        # Apply rescale if available
        slope  = float(getattr(ds, "RescaleSlope",     1.0))
        offset = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + offset
        return {
            "pixel_shape":   str(arr.shape),
            "pixel_min":     f"{arr.min():.1f}",
            "pixel_max":     f"{arr.max():.1f}",
            "pixel_mean":    f"{arr.mean():.2f}",
            "pixel_std":     f"{arr.std():.2f}",
            "pixel_p5":      f"{np.percentile(arr, 5):.1f}",
            "pixel_p95":     f"{np.percentile(arr, 95):.1f}",
            "nonzero_ratio": f"{(arr != 0).mean():.3f}",
        }
    except Exception as e:
        return {"pixel_error": str(e)}


def load_series_as_volume(dcm_files: list[Path]) -> Optional[np.ndarray]:
    """Use SimpleITK to load a sorted DICOM series as a 3D volume."""
    try:
        reader = sitk.ImageSeriesReader()
        fnames  = [str(f) for f in dcm_files]
        reader.SetFileNames(fnames)
        image = reader.Execute()
        return sitk.GetArrayFromImage(image)  # (Z, Y, X)
    except Exception:
        return None


def volume_stats(vol: np.ndarray) -> dict:
    """Compute volume-level statistics for the meta model."""
    return {
        "volume_shape":       str(vol.shape),
        "volume_voxel_count": int(np.prod(vol.shape)),
        "volume_min":         float(vol.min()),
        "volume_max":         float(vol.max()),
        "volume_mean":        float(vol.mean()),
        "volume_std":         float(vol.std()),
        "volume_snr_estimate":float(vol.mean() / (vol.std() + 1e-6)),
        "volume_nonzero_pct": float((vol != 0).mean() * 100),
        "volume_dynamic_range": float(vol.max() - vol.min()),
    }


def export_nifti(dcm_files: list[Path], out_path: Path) -> None:
    """Export DICOM series to NIfTI using SimpleITK → nibabel."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(f) for f in dcm_files])
    image = reader.Execute()
    arr    = sitk.GetArrayFromImage(image)
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    affine[:3, 3] = origin
    nii = nib.Nifti1Image(arr, affine)
    nib.save(nii, out_path)
    console.print(f"[green]✓[/green] NIfTI saved → {out_path}")


def render_metadata_tree(records: list[dict]) -> None:
    """Display tag groups as a Rich tree for the first slice."""
    if not records:
        return
    r = records[0]
    tree = Tree("[bold]DICOM metadata[/bold] — first slice")
    for group, tags in TAG_GROUPS.items():
        branch = tree.add(f"[cyan]{group}[/cyan]")
        for tag in tags:
            val = r.get(tag, "")
            if val:
                branch.add(f"[dim]{tag}[/dim]  {val}")
    console.print(tree)


def ask_claude(records: list[dict], vol_stats: Optional[dict], model: str = "claude-opus-4-6") -> str:
    """Send extracted metadata + volume stats to Claude for expert analysis."""
    client = anthropic.Anthropic()

    # Build a compact JSON summary (avoid sending all 100+ files)
    sample = records[:5]
    payload = {
        "slice_count": len(records),
        "sample_slices": sample,
        "volume_stats": vol_stats or {},
    }

    prompt = f"""You are a medical imaging and physical-AI data expert.

Below is extracted metadata and statistics from a DICOM brain MRI series.
Provide a detailed structured analysis covering:

1. **Scan protocol** — What MRI sequence is this? (T1, T2, FLAIR, DWI, etc.)
   Justify from TR/TE/FlipAngle/InversionTime if present.
2. **Image quality indicators** — SNR estimate, dynamic range, nonzero ratio.
3. **3D volume geometry** — voxel size, FOV, slice count, orientation.
4. **Clinical / research relevance** — What brain structures/pathologies could be studied?
5. **ML suitability** — Is this suitable for training a world model / segmentation / classification?
   What preprocessing steps are recommended?
6. **Data gaps** — What metadata is missing that would be important?
7. **Red flags** — Anything anomalous in the statistics?

DICOM data:
```json
{json.dumps(payload, indent=2, default=str)}
```
"""

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    folder: Annotated[Path, typer.Argument(help="Folder containing .dcm files")],
    claude: Annotated[bool, typer.Option("--claude", help="Send to Claude for analysis")] = False,
    export_nii: Annotated[bool, typer.Option("--export-nii", help="Export series to NIfTI")] = False,
    slices: Annotated[bool, typer.Option("--slices", help="Show per-slice pixel stats table")] = False,
    out: Annotated[Path, typer.Option("--out", help="Output CSV path")] = Path("dicom_metadata.csv"),
):
    """Extract everything from a DICOM brain series."""

    console.print(Panel.fit(
        "[bold cyan]DICOM Brain Extractor[/bold cyan]\n"
        f"[dim]pydicom · SimpleITK · nibabel · polars · anthropic[/dim]",
        border_style="cyan",
    ))

    # 1. Discover files
    dcm_files = sorted(folder.glob("*.dcm"))
    if not dcm_files:
        console.print("[red]No .dcm files found.[/red]")
        raise typer.Exit(1)
    console.print(f"Found [bold]{len(dcm_files)}[/bold] DICOM files in [dim]{folder}[/dim]\n")

    # 2. Extract per-slice metadata
    records = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting metadata", total=len(dcm_files))
        for f in dcm_files:
            ds = pydicom.dcmread(str(f), force=True)
            row: dict = {"filename": f.name}
            for tags in TAG_GROUPS.values():
                for tag in tags:
                    row[tag] = safe_get(ds, tag)
            row.update(extract_pixel_stats(ds))
            records.append(row)
            progress.advance(task)

    # 3. Display metadata tree
    render_metadata_tree(records)

    # 4. Build polars dataframe + save CSV
    df = pl.DataFrame(records, infer_schema_length=None)
    df.write_csv(out)
    console.print(f"\n[green]✓[/green] Metadata CSV → {out}  "
                  f"([dim]{len(df)} rows × {len(df.columns)} cols[/dim])")

    # 5. Per-slice pixel stats table
    if slices:
        table = Table("File", "Shape", "Min", "Max", "Mean", "Std",
                      title="Per-slice pixel statistics", box=box.SIMPLE_HEAD)
        for r in records[:20]:  # cap at 20 for display
            table.add_row(
                r["filename"], r.get("pixel_shape", ""),
                r.get("pixel_min", ""), r.get("pixel_max", ""),
                r.get("pixel_mean", ""), r.get("pixel_std", ""),
            )
        console.print(table)

    # 6. Load as 3D volume → compute volume stats
    console.print("\n[cyan]Loading 3D volume via SimpleITK…[/cyan]")
    vol = load_series_as_volume(dcm_files)
    vstats = None
    if vol is not None:
        vstats = volume_stats(vol)
        vt = Table("Metric", "Value", title="3D volume statistics", box=box.SIMPLE_HEAD)
        for k, v in vstats.items():
            vt.add_row(k, str(v))
        console.print(vt)
    else:
        console.print("[yellow]⚠ Could not assemble 3D volume (single-slice or mixed series)[/yellow]")

    # 7. Export NIfTI
    if export_nii and vol is not None:
        nii_path = folder / "output_volume.nii.gz"
        export_nifti(dcm_files, nii_path)

    # 8. Claude analysis
    if claude:
        console.print("\n[cyan]Sending to Claude for expert analysis…[/cyan]\n")
        try:
            analysis = ask_claude(records, vstats)
            console.print(Panel(analysis, title="[bold]Claude Analysis[/bold]",
                                border_style="green", padding=(1, 2)))
            # Save analysis
            analysis_path = folder / "claude_analysis.md"
            analysis_path.write_text(f"# DICOM Brain Series Analysis\n\n{analysis}\n")
            console.print(f"[green]✓[/green] Analysis saved → {analysis_path}")
        except anthropic.APIError as e:
            console.print(f"[red]Anthropic API error:[/red] {e}")


if __name__ == "__main__":
    app()
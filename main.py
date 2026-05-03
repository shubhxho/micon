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
#   "mlx-vlm>=0.4.3",
#   "mlx-lm>=0.19",
#   "matplotlib>=3.10",
#   "scipy>=1.14",
#   "pillow>=11.0",
#   "python-dotenv>=1.0",
#   "mlx>=0.22",
#   "mcap>=1.1",
#   "zstandard>=0.23",
#   "huggingface_hub>=0.25",
#   "modal>=0.73",
#   "openai>=1.50",
# ]
# ///

"""
micom — MRI processing pipeline.

Just run it:  uv run main.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import sys
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.pipeline import discover_dcm_folders, run_pipeline
from src.pipeline.mcap_convert import run_mcap_convert
from src.pipeline.parquet_convert import run_parquet_convert
from src.pipeline.redact import run_redaction

console = Console()

app = typer.Typer(rich_markup_mode="rich", no_args_is_help=False)

_T7 = Path("/Volumes/T7 Shield/redacted")
DEFAULT_INPUT_DIR = _T7 if _T7.exists() else Path("mcap-files")

# ── Grade colors ────────────────────────────────────────────────────────────

GRADE_STYLE = {
    "A": "bold green",
    "B": "green",
    "C": "yellow",
    "D": "red",
    "F": "bold red",
}


def _grade_text(grade: str, score: float = 0) -> Text:
    style = GRADE_STYLE.get(grade, "dim")
    t = Text(f"{grade}", style=style)
    if score:
        t.append(f" ({score:.0f}/100)", style="dim")
    return t


def _find_input() -> Path:
    """Auto-detect DICOM input — T7 Shield first, then mcap-files/."""
    if _T7.exists():
        return _T7
    local = Path("mcap-files")
    if local.exists() and any(local.rglob("*.dcm")):
        return local
    return local


def _open_report(out_dir: Path):
    """Open the HTML report in the default browser."""
    report = out_dir / "report.html"
    if report.exists():
        webbrowser.open(f"file://{report.resolve()}")


# ── Main command ────────────────────────────────────────────────────────────


@app.command()
def main(
    folder: Annotated[
        Path | None, typer.Argument(help="DICOM folder (auto-detects T7 Shield or mcap-files/)")
    ] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Output directory")] = Path(
        "output"
    ),
    do_redact: Annotated[
        bool, typer.Option("--do-redact", "-r", help="HIPAA-redact before extraction")
    ] = False,
    upload_hf: Annotated[
        bool, typer.Option("--upload-hf", "-u", help="Upload to Hugging Face")
    ] = False,
    hf_repo: Annotated[str, typer.Option("--hf-repo", help="HF repo (auto from username)")] = "",
    hf_public: Annotated[bool, typer.Option("--hf-public", help="Make HF dataset public")] = True,
    export_nii: Annotated[bool, typer.Option("--export-nii", help="Export NIfTI volumes")] = False,
    analyze: Annotated[bool, typer.Option("--analyze", help="AI analysis (Apple Silicon)")] = False,
    compress: Annotated[bool, typer.Option("--compress", help="Compress outputs")] = False,
    local: Annotated[bool, typer.Option("--local", help="Run on this machine, not Modal")] = False,
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Don't open report in browser")
    ] = False,
):
    """Process a DICOM study — upload, extract, grade, download.

    Just run it. Auto-detects your T7 Shield or mcap-files/ folder,
    uploads to Modal, processes on auto-scaled cloud CPUs, downloads
    the results, and opens the HTML report.
    """
    # ── Resolve input ──────────────────────────────────────────────────
    if folder is None:
        folder = _find_input()
    if not folder.exists():
        console.print(f"[red]Not found:[/red] {folder}")
        console.print("[dim]Plug in your T7 Shield or put .dcm files in mcap-files/[/dim]")
        raise SystemExit(1)

    t0 = time.time()

    # ── Banner ─────────────────────────────────────────────────────────
    mode = "local" if local else "cloud"
    flags = []
    if do_redact:
        flags.append("redact")
    if upload_hf:
        flags.append("hf")
    if export_nii:
        flags.append("nifti")
    if analyze:
        flags.append("ai")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""

    console.print(
        Panel(
            f"[bold]{folder.name}[/bold]  [dim]{folder}[/dim]\n"
            f"[dim]{mode}{flag_str} → {out_dir}[/dim]",
            title="[bold cyan]micom[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    # ── Dispatch ───────────────────────────────────────────────────────
    if local:
        result = _run_local(
            folder,
            analyze,
            export_nii,
            out_dir,
            compress,
            do_redact,
            upload_hf,
            hf_repo,
            not hf_public,
        )
    else:
        result = _run_modal(
            folder,
            analyze,
            export_nii,
            out_dir,
            compress,
            do_redact,
            upload_hf,
            hf_repo,
            not hf_public,
        )

    if result is None:
        return

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column(style="bold")
    tbl.add_column()

    study = result.get("study", folder.name)
    grade = result.get("study_grade", "?")
    score = result.get("study_score", 0)
    hipaa = result.get("hipaa_score", "?")
    series_ok = result.get("series_succeeded", "?")
    series_total = result.get("image_series", "?")
    grade_dist = result.get("grade_distribution", {})

    tbl.add_row("Study", study)
    tbl.add_row("Grade", _grade_text(grade, score))

    if grade_dist:
        dist_parts = []
        for g in ("A", "B", "C", "D", "F"):
            c = grade_dist.get(g, 0)
            if c:
                dist_parts.append(Text(f"{g}:{c}", style=GRADE_STYLE.get(g, "")))
        if dist_parts:
            combined = Text(" ")
            for i, p in enumerate(dist_parts):
                if i > 0:
                    combined.append(" ")
                combined.append_text(p)
            tbl.add_row("", combined)

    tbl.add_row("HIPAA", f"{hipaa}/100")
    tbl.add_row("Series", f"{series_ok}/{series_total}")

    final_report = out_dir / "final_report.md"
    if final_report.exists():
        tbl.add_row("Report", str(final_report))

    if result.get("hf_url"):
        tbl.add_row("HF", result["hf_url"])

    tbl.add_row("Output", str(out_dir))
    tbl.add_row("Time", f"{elapsed:.0f}s")

    console.print(
        Panel(tbl, title="[bold green]Done[/bold green]", border_style="green", padding=(0, 1))
    )

    # ── Open report ────────────────────────────────────────────────────
    if not no_open:
        _open_report(out_dir)


# ── Modal pipeline ──────────────────────────────────────────────────────────


def _run_modal(
    folder: Path,
    analyze: bool,
    export_nii: bool,
    out_dir: Path,
    compress: bool,
    do_redact: bool,
    upload_hf: bool,
    hf_repo: str,
    hf_private: bool,
) -> dict | None:
    """Upload → [redact →] extract on Modal → download."""
    import modal

    from modal_app import (
        _VOL_OUTPUT,
        _ensure_uploaded,
        _parallel_download,
        run_extraction_pipeline,
        run_redaction_pipeline,
    )
    from modal_app import (
        app as modal_app,
    )

    # ── Upload ─────────────────────────────────────────────────────────
    with console.status("[cyan]Uploading to Modal volume…[/cyan]"):
        study_name, stats = _ensure_uploaded(str(folder))
    if stats.get("error"):
        console.print(f"[red]Upload failed:[/red] {stats['error']}")
        return None
    if stats.get("uploaded"):
        console.print(
            f"[green]Uploaded[/green] {stats['uploaded']} files "
            f"({stats.get('total_bytes', 0) / 1024**2:.0f} MB, "
            f"{stats.get('elapsed', 0):.0f}s)"
        )
    else:
        console.print("[dim]Already on volume[/dim]")

    extract_study = study_name
    result = {}

    with modal.enable_output(), modal_app.run():
        # ── Redact ─────────────────────────────────────────────────────
        if do_redact:
            console.print(f"[red]Redacting[/red] {study_name}…")
            redact_result = run_redaction_pipeline.remote(study_name)
            if redact_result.get("error"):
                console.print(f"[red]Redaction failed:[/red] {redact_result['error']}")
                return None
            imp = redact_result.get("improvement", {})
            console.print(
                f"[green]Redacted[/green] {imp.get('files_redacted', '?')} files → "
                f"{imp.get('post_score', '?')}/100"
            )
            extract_study = redact_result.get("redacted_study", f"{study_name}_redacted")

        # ── Extract ────────────────────────────────────────────────────
        console.print(f"[cyan]Processing[/cyan] {extract_study} on Modal…")
        result = run_extraction_pipeline.remote(
            extract_study,
            analyze,
            export_nii,
            compress,
            upload_hf=upload_hf,
            hf_repo=hf_repo,
            hf_private=hf_private,
        )
        if result.get("error"):
            console.print(f"[red]Extraction failed:[/red] {result['error']}")
            return None

        grade = result.get("study_grade", "?")
        score = result.get("study_score", 0)
        console.print(
            Text.assemble(
                ("Extracted ", ""),
                ("  ", ""),
                _grade_text(grade, score),
                ("  ", ""),
                (f"HIPAA {result.get('hipaa_score', '?')}/100", "dim"),
            )
        )

    # ── Download ───────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    with console.status("[cyan]Downloading results…[/cyan]"):
        remote_dir = str(_VOL_OUTPUT / extract_study)
        n_files, total_bytes = _parallel_download(remote_dir, out_dir)
        if do_redact:
            redact_remote = str(_VOL_OUTPUT / f"{study_name}_redacted")
            hipaa_dl = out_dir / "redaction"
            hipaa_dl.mkdir(parents=True, exist_ok=True)
            _parallel_download(redact_remote, hipaa_dl)

    console.print(f"[green]Downloaded[/green] {n_files} files ({total_bytes / 1024**2:.1f} MB)")

    result["study"] = study_name
    return result


# ── Local pipeline ──────────────────────────────────────────────────────────


def _run_local(
    folder: Path,
    analyze: bool,
    export_nii: bool,
    out_dir: Path,
    compress: bool,
    do_redact: bool,
    upload_hf: bool,
    hf_repo: str,
    hf_private: bool,
) -> dict | None:
    """Run entirely on this machine."""
    import json
    import multiprocessing

    extract_folder = folder

    if do_redact:
        redact_out = out_dir / "redacted"
        n_workers = min(multiprocessing.cpu_count(), 8)
        console.print(f"[red]Redacting[/red] {folder.name}…")
        summary = run_redaction(folder, redact_out, n_workers)
        extract_folder = redact_out
        console.print(f"[green]Redacted[/green] {summary.files_processed} files")

    run_pipeline(extract_folder, analyze, export_nii, out_dir, 0, compress)

    # Build result dict from output files
    result: dict = {"study": folder.name}
    stats_path = out_dir / "series_stats.json"
    hipaa_path = out_dir / "hipaa_compliance.json"
    patient_info = series_info = hipaa_report = None

    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        patient_info = stats.get("patient")
        series_info = stats.get("series", [])
        result["image_series"] = sum(1 for s in series_info if s.get("has_pixels"))
        result["series_succeeded"] = result["image_series"]

    if hipaa_path.exists():
        hipaa_report = json.loads(hipaa_path.read_text())
        result["hipaa_score"] = hipaa_report.get(
            "compliance_score", hipaa_report.get("post_redaction", {}).get("compliance_score", "?")
        )

    # Compute grade from series
    if series_info:
        grades = [
            s.get("quality_analysis", {}).get("quality_grade", {}).get("grade")
            for s in series_info
            if s.get("has_pixels") and s.get("quality_analysis")
        ]
        grades = [g for g in grades if g]
        if grades:
            from src.quality import grade_study

            all_grade_dicts = [
                s["quality_analysis"]["quality_grade"]
                for s in series_info
                if s.get("has_pixels") and s.get("quality_analysis", {}).get("quality_grade")
            ]
            sg = grade_study(all_grade_dicts)
            result["study_grade"] = sg.get("grade", "?")
            result["study_score"] = sg.get("score", 0)
            result["grade_distribution"] = sg.get("grade_distribution", {})

    if upload_hf:
        console.print("[cyan]Uploading to Hugging Face…[/cyan]")
        try:
            from src.hf_upload import upload_to_huggingface

            url = upload_to_huggingface(
                out_dir,
                study_name=folder.name,
                repo_id=hf_repo or "",
                patient_info=patient_info,
                series_info=series_info,
                hipaa_report=hipaa_report,
                private=hf_private,
                source_dir=folder,
            )
            result["hf_url"] = url
            console.print(f"[green]Uploaded[/green] → {url}")
        except Exception as e:
            console.print(f"[red]HF upload failed:[/red] {e}")

    return result


# ── Other commands ──────────────────────────────────────────────────────────


@app.command()
def mcap(
    folder: Annotated[Path, typer.Argument(help="Folder containing .dcm files")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output .mcap file")] = None,
    workers: Annotated[int, typer.Option("--workers", help="Parallel threads")] = 0,
):
    """Convert DICOM folder to MCAP format."""
    run_mcap_convert(folder, output, workers)


@app.command(name="mcap-export")
def mcap_export(
    folder: Annotated[Path | None, typer.Argument(help="Folder containing .dcm files")] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Output directory")] = Path("output"),
    workers: Annotated[int, typer.Option("--workers", help="Parallel workers")] = 0,
    batch: Annotated[bool, typer.Option("--batch", help="Process subfolders separately")] = False,
):
    """MCAP-only export — per-series MCAP files, no images."""
    if folder is None:
        folder = _find_input()
        if not folder.exists():
            console.print(f"[red]Not found: {folder}[/red]")
            raise SystemExit(1)

    if batch:
        t0 = time.time()
        dcm_folders = discover_dcm_folders(folder)
        succeeded, failed = [], []
        for i, dcm_folder in enumerate(dcm_folders, 1):
            rel = dcm_folder.relative_to(folder)
            sub_out = out_dir / rel
            console.print(f"[dim][{i}/{len(dcm_folders)}][/dim] {rel}")
            try:
                run_pipeline(
                    dcm_folder, out_dir=sub_out, workers=workers, recursive=False, mcap_only=True
                )
                succeeded.append(dcm_folder)
            except (SystemExit, Exception) as exc:
                console.print(f"[red]Failed: {rel} — {exc}[/red]")
                failed.append((dcm_folder, str(exc)))
        console.print(
            f"[green]{len(succeeded)} ok[/green], [red]{len(failed)} failed[/red] "
            f"in {time.time() - t0:.0f}s"
        )
    else:
        run_pipeline(folder, out_dir=out_dir, workers=workers, mcap_only=True)


@app.command()
def parquet(
    folder: Annotated[Path, typer.Argument(help="Folder to convert")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output .parquet file")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", help="Parallel threads")] = 0,
    extensions: Annotated[str, typer.Option("--ext", help="File extensions (comma-sep)")] = "",
):
    """Convert any folder to Parquet."""
    exts = [e.strip().lower() for e in extensions.split(",") if e.strip()] if extensions else None
    run_parquet_convert(folder, output, workers, exts=exts)


@app.command()
def redact(
    folder: Annotated[Path | None, typer.Argument(help="DICOM folder to redact")] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Output directory")] = Path("redacted"),
    workers: Annotated[int, typer.Option("--workers", help="Parallel workers")] = 0,
    salt: Annotated[str, typer.Option("--salt", help="UID hash salt")] = "",
    date_shift: Annotated[int, typer.Option("--date-shift", help="Date shift days")] = 0,
    no_verify: Annotated[bool, typer.Option("--no-verify", help="Skip verification")] = False,
):
    """HIPAA Safe Harbor de-identification."""
    import json
    import multiprocessing

    from src.hipaa import compliance_report_to_dict, run_hipaa_scan

    if folder is None:
        folder = _find_input()
        if not folder.exists():
            console.print(f"[red]Not found: {folder}[/red]")
            raise SystemExit(1)

    n_workers = workers or min(multiprocessing.cpu_count(), 8)

    from src.pipeline.discover import discover_files

    dcm_files = discover_files(folder, recursive=True)
    file_paths = [str(f) for f in dcm_files]

    console.print(f"[red]Scanning[/red] {len(file_paths)} files…")
    pre_report = run_hipaa_scan(file_paths, study_name=f"{folder.name}_pre", n_workers=n_workers)
    console.print(
        f"  Pre: {pre_report.compliance_score:.0f}/100, {pre_report.total_phi_findings} PHI"
    )

    console.print("[red]Redacting[/red]…")
    shift = date_shift if date_shift != 0 else None
    summary = run_redaction(folder, out_dir, n_workers, salt, shift, False, not no_verify)

    from src.redaction import PHI_TAGS_BLANK, PHI_TAGS_DATE, PHI_TAGS_HASH, PHI_TAGS_REMOVE

    de_identified = PHI_TAGS_HASH | PHI_TAGS_DATE | PHI_TAGS_BLANK | PHI_TAGS_REMOVE

    redacted_files = [str(f) for f in out_dir.rglob("*.dcm")]
    if redacted_files:
        console.print(f"[cyan]Verifying[/cyan] {len(redacted_files)} files…")
        post_report = run_hipaa_scan(
            redacted_files,
            study_name=f"{folder.name}_post",
            n_workers=n_workers,
            de_identified_tags=de_identified,
        )

        delta_phi = pre_report.total_phi_findings - post_report.total_phi_findings
        delta_score = post_report.compliance_score - pre_report.compliance_score

        console.print(
            f"[green]Done[/green]  "
            f"+{delta_score:.0f} score, -{delta_phi} PHI  "
            f"({post_report.compliance_score:.0f}/100)"
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        combined = {
            "pre_redaction": compliance_report_to_dict(pre_report),
            "post_redaction": compliance_report_to_dict(post_report),
            "improvement": {
                "phi_removed": delta_phi,
                "score_increase": round(delta_score, 1),
                "files_redacted": summary.files_processed,
                "files_failed": summary.files_failed,
                "tags_removed": summary.total_tags_removed,
                "tags_blanked": summary.total_tags_blanked,
                "tags_hashed": summary.total_tags_hashed,
                "tags_date_shifted": summary.total_tags_date_shifted,
            },
        }
        (out_dir / "hipaa_compliance.json").write_text(json.dumps(combined, indent=2))


@app.command()
def hipaa(
    folder: Annotated[Path | None, typer.Argument(help="DICOM folder to scan")] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Output directory")] = Path("output"),
    workers: Annotated[int, typer.Option("--workers", help="Parallel workers")] = 0,
):
    """HIPAA compliance scan — read-only, no modifications."""
    import json
    import multiprocessing

    from src.hipaa import compliance_report_to_dict, run_hipaa_scan
    from src.pipeline.discover import discover_files

    if folder is None:
        folder = _find_input()
        if not folder.exists():
            console.print(f"[red]Not found: {folder}[/red]")
            raise SystemExit(1)

    n_workers = workers or multiprocessing.cpu_count()
    dcm_files = discover_files(folder, recursive=True)
    file_paths = [str(f) for f in dcm_files]

    console.print(f"Scanning [bold]{len(file_paths)}[/bold] files…")
    report = run_hipaa_scan(file_paths, study_name=folder.name, n_workers=n_workers)

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column(style="bold")
    tbl.add_column()
    tbl.add_row("Score", f"{report.compliance_score:.0f}/100")
    tbl.add_row("Files", f"{report.total_files} scanned, {report.files_with_phi} with PHI")
    tbl.add_row("Findings", str(report.total_phi_findings))
    for level in ("high", "medium", "low"):
        count = report.risk_summary.get(level, 0)
        if count:
            color = {"high": "red", "medium": "yellow", "low": "cyan"}[level]
            tbl.add_row(f"  {level}", f"[{color}]{count}[/{color}]")

    console.print(Panel(tbl, title="HIPAA", border_style="cyan", padding=(0, 1)))

    if report.recommendations:
        for rec in report.recommendations:
            console.print(f"  [dim]{rec}[/dim]")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "hipaa_compliance.json"
    report_path.write_text(json.dumps(compliance_report_to_dict(report), indent=2))
    console.print(f"[green]Report[/green] → {report_path}")


@app.command()
def deid(
    folder: Annotated[Path | None, typer.Argument(help="DICOM folder to de-identify")] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Output directory")] = Path(
        "clean_output"
    ),
    workers: Annotated[int, typer.Option("--workers", help="Parallel workers")] = 0,
    skip_deface: Annotated[bool, typer.Option("--skip-deface", help="Skip pixel defacing")] = False,
    skip_ocr: Annotated[bool, typer.Option("--skip-ocr", help="Skip OCR pixel PHI scan")] = False,
):
    """Commercial de-identification pipeline — buyer-ready DICOM export.

    Runs: ingest → deface → PS3.15 metadata de-id → validate → quality → manifest → export.
    Output passes a serious de-identification audit.
    """
    import multiprocessing

    from src.pipeline.commercial import run_commercial_pipeline

    if folder is None:
        folder = _find_input()
    if not folder.exists():
        console.print(f"[red]Not found:[/red] {folder}")
        raise SystemExit(1)

    n = workers or multiprocessing.cpu_count()
    result = run_commercial_pipeline(
        folder,
        out_dir,
        n_workers=n,
        skip_deface=skip_deface,
        skip_pixel_ocr=skip_ocr,
    )

    if result.get("validation_passed"):
        console.print(f"\n[bold green]BUYER-READY[/bold green] → {out_dir}")
    else:
        console.print("\n[bold red]NOT READY[/bold red] — see validation failures above")


if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith("-")):
        sys.argv.insert(1, "main")
    app()

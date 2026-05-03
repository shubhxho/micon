"""Stage 8 — AI analysis pipeline (local MLX models).

Flow:
  1. Derivatives: lightweight VLM analysis (sequential — GPU bound)
  2. Primaries: full VLM analysis + chaining (sequential — GPU bound)
  3. Cross-series: compare findings across all sequences (text LM)
  4. Synthesize: structured clinical report (text LM)

All inference runs locally on Apple Silicon via MLX — no API calls.
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..annotation.local import (
    _is_derivative,
    analyze_and_chain,
    cross_series_comparison,
    synthesize_report,
)
from ..helpers import to_json

console = Console()


def run_ai_analysis(
    series_results: list,
    all_records: list[dict],
    patient_info: dict,
    series_info: list[dict],
    conformance_issues: list[dict],
    out_dir: Path,
    n_workers: int,
) -> None:
    """Run the full AI analysis pipeline with local MLX models."""

    # Collect all series info
    primary_items: list[tuple[str, str, str | None, dict | None]] = []
    derivative_items: list[tuple[str, str, str | None, dict | None]] = []
    series_folders: dict[str, str] = {}

    for r in series_results:
        if not r.montage_path:
            continue
        label = (
            f"Series {r.info.get('series_number', '?')} — {r.info.get('series_description', '')}"
        )
        qa = r.info.get("quality_analysis")
        item = (r.montage_path, label, r.enhanced_path, qa)

        if _is_derivative(label):
            derivative_items.append(item)
        else:
            primary_items.append(item)

        if r.series_folder:
            series_folders[label] = r.series_folder

    n_total = len(primary_items) + len(derivative_items)
    console.print(
        f"\n[bold cyan]AI Analysis (local MLX)[/bold cyan] — {n_total} series "
        f"({len(primary_items)} primary, {len(derivative_items)} derivative)"
    )

    t0 = time.time()
    image_analyses: dict[str, str] = {}

    # ── Batch 1: Derivatives (lightweight, sequential — GPU bound) ───────
    if derivative_items:
        t_deriv = time.time()
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Derivatives (lightweight)", total=len(derivative_items))
            for path, label, _, _ in derivative_items:
                try:
                    result = analyze_and_chain(path, label, None, 0, None)
                    image_analyses[label] = result["merged"]
                except Exception as e:
                    image_analyses[label] = f"Error: {e}"
                progress.advance(task)

        console.print(
            f"[green]✓[/green] {len(derivative_items)} derivatives done in "
            f"{time.time() - t_deriv:.1f}s"
        )

    # ── Batch 2: Primaries (full analysis + chaining, sequential) ────────
    if primary_items:
        t_prim = time.time()
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Primaries (full + chain)", total=len(primary_items))
            for path, label, enhanced, qa in primary_items:
                try:
                    result = analyze_and_chain(path, label, enhanced, 1, qa)
                    image_analyses[label] = result["merged"]
                    n_fu = len(result["followups"])
                    reason = result.get("chain_reason", "")
                    if result["skipped_chain"]:
                        status = f"[dim]{reason}[/dim]"
                    else:
                        status = f"[green]+{n_fu} follow-ups[/green] ({reason})"
                    progress.console.print(f"  {label}: {status}")
                except Exception as e:
                    image_analyses[label] = f"Error: {e}"
                progress.advance(task)

        n_chained = sum(1 for a in image_analyses.values() if "Follow-up" in a)
        console.print(
            f"[green]✓[/green] {len(primary_items)} primaries done in "
            f"{time.time() - t_prim:.1f}s ({n_chained} chained)"
        )

    # ── Cross-series comparison ──────────────────────────────────────────
    console.print("\n[cyan]Cross-series comparison...[/cyan]")
    t_cross = time.time()
    cross_series = cross_series_comparison(image_analyses, series_info)
    console.print(f"[green]✓[/green] Cross-series done in {time.time() - t_cross:.1f}s")

    # ── Write analyses, then synthesize (sequential — GPU bound) ──────────
    console.print("\n[cyan]Synthesizing final report...[/cyan]")
    t_synth = time.time()

    # Write per-series analyses to disk
    md = "# Per-Series Image Analysis (Local MLX VLM)\n\n"
    for label, analysis in sorted(image_analyses.items()):
        md += f"## {label}\n\n{analysis}\n\n---\n\n"
        folder = series_folders.get(label)
        if folder:
            Path(folder, "ai_analysis.md").write_text(f"# {label}\n\n{analysis}\n")
    (out_dir / "image_analyses.md").write_text(md)
    if cross_series:
        (out_dir / "cross_series_analysis.md").write_text(
            f"# Cross-Series Comparison\n\n{cross_series}\n"
        )
    console.print(f"[green]✓[/green] Analyses written to {out_dir}")

    sample_tags = {}
    for r in all_records:
        if r.get("_has_pixel_data"):
            sample_tags = {
                k: to_json(v)
                for k, v in r.items()
                if not k.startswith("histogram_") and not k.startswith("_")
            }
            break
    payload = {
        "patient": patient_info,
        "series": series_info,
        "conformance_issues_count": len(conformance_issues),
        "sample_full_tags": sample_tags,
    }

    try:
        report = synthesize_report(payload, image_analyses, cross_series)
        console.print(f"[green]✓[/green] Synthesis done in {time.time() - t_synth:.1f}s")
        console.print(
            Panel(
                report,
                title="[bold]AI Analysis Report[/bold]",
                border_style="green",
                padding=(1, 2),
            )
        )
        (out_dir / "ai_analysis.md").write_text(f"# DICOM Study Analysis\n\n{report}\n")
        console.print(f"[green]✓[/green] Full report → {out_dir / 'ai_analysis.md'}")
    except Exception as e:
        console.print(f"[red]Synthesis error:[/red] {e}")

    total = time.time() - t0
    console.print(f"\n[dim]AI analysis total: {total:.1f}s[/dim]")

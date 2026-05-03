"""CLI dry-run planner: print a rich-styled table of pending work and cost estimates.

Usage::

    python -m src.cli_planner --root /path/to/output_dir

Exits 0 on success, 1 if output_dir does not exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from src.dry_run import plan


def _build_table(data: dict) -> Table:
    table = Table(
        title="Pipeline Dry-Run Plan",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold", min_width=32)
    table.add_column("Value", justify="right", min_width=16)

    def _row(label: str, value: str, style: str = "") -> None:
        table.add_row(label, value, style=style)

    _row("Studies found", str(data["n_studies"]))
    _row("Series total (detail.json)", str(data["n_series_total"]))
    table.add_section()

    _row("Stage 2 quality pending", str(data["n_series_quality_pending"]),
         style="yellow" if data["n_series_quality_pending"] else "")
    _row("Stage 3 annotation pending", str(data["n_series_annotation_pending"]),
         style="yellow" if data["n_series_annotation_pending"] else "")
    _row("Stage 4 pack pending (studies)", str(data["n_studies_pack_pending"]),
         style="yellow" if data["n_studies_pack_pending"] else "")
    table.add_section()

    modal_val = f"${data['est_modal_dollars']:.4f}"
    or_val = f"${data['est_openrouter_dollars']:.4f}"
    total = data["est_modal_dollars"] + data["est_openrouter_dollars"]
    total_val = f"${total:.4f}"

    _row("Est. Modal cost", modal_val,
         style="red" if data["est_modal_dollars"] > 5 else "green")
    _row("Est. OpenRouter cost", or_val,
         style="red" if data["est_openrouter_dollars"] > 2 else "green")
    _row("Est. total cost", total_val,
         style="bold red" if total > 10 else "bold green")
    table.add_section()

    wall = data["est_wall_minutes"]
    wall_str = f"{wall:.1f} min" if wall < 60 else f"{wall/60:.1f} h"
    _row("Est. wall time (rough)", wall_str)

    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli_planner",
        description="Dry-run plan: how much work and cost would the next run incur?",
    )
    parser.add_argument(
        "--root", required=True, metavar="OUTPUT_DIR",
        help="Path to the pipeline output directory (same as output_dir in resume_pipeline.py)",
    )
    args = parser.parse_args()

    output_dir = Path(args.root)
    console = Console()

    if not output_dir.exists():
        console.print(f"[bold red]Error:[/bold red] output_dir does not exist: {output_dir}")
        sys.exit(1)

    console.print(f"Scanning [bold]{output_dir}[/bold] ...")
    data = plan(output_dir)

    table = _build_table(data)
    console.print(table)

    console.print(
        "\n[dim]Cost model: Modal CPU $0.000111/s/CPU, quality ~30s@2CPU, "
        "annotate ~15s@1CPU; OpenRouter Gemma 4 31B $0.30/1M in + $0.50/1M out, "
        "~3000 in + 2000 out tokens/series. Wall time assumes ~50/320/20 concurrent.[/dim]"
    )


if __name__ == "__main__":
    main()

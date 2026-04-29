"""Redaction pipeline — threaded, streaming, single-pass redact+verify."""

from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .discover import discover_files
from ..redaction import (
    redact_files, RedactionSummary,
    PHI_TAGS_REMOVE, PHI_TAGS_BLANK, PHI_TAGS_HASH, PHI_TAGS_DATE,
)

console = Console()


def run_redaction(
    folder: Path,
    out_dir: Path,
    workers: int = 0,
    salt: str = "",
    date_shift: int | None = None,
    preserve_uids: bool = False,
    verify: bool = True,
    recursive: bool = True,
) -> RedactionSummary:
    """Run the full DICOM redaction pipeline.

    Single-pass: redact + verify happen in the same file read (no second I/O).
    ThreadPoolExecutor: no fork, no pickle, no crash on large datasets.
    """
    t0 = time.time()
    n_workers = workers or min(multiprocessing.cpu_count(), 8)

    console.print(Panel.fit(
        f"[bold red]DICOM PHI Redaction Pipeline[/bold red]  [dim]({n_workers} threads)[/dim]\n"
        "[dim]HIPAA Safe Harbor · O(T) per file · threaded · single-pass verify[/dim]",
        border_style="red",
    ))

    # Stage 1 — Discover
    dcm_files = discover_files(folder, recursive=recursive)
    file_paths = [str(f) for f in dcm_files]

    n_rules = len(PHI_TAGS_REMOVE) + len(PHI_TAGS_BLANK) + len(PHI_TAGS_HASH) + len(PHI_TAGS_DATE)
    console.print(
        f"Redaction: [red]{len(PHI_TAGS_REMOVE)} remove[/red] + "
        f"[yellow]{len(PHI_TAGS_BLANK)} blank[/yellow] + "
        f"[cyan]{len(PHI_TAGS_HASH)} hash[/cyan] + "
        f"[magenta]{len(PHI_TAGS_DATE)} date-shift[/magenta] = "
        f"[bold]{n_rules}[/bold] rules\n"
    )

    # Stage 2 — Redact + verify (single pass, threaded)
    console.print(f"[bold red]Redacting {len(file_paths)} files ({n_workers} threads)…[/bold red]")

    with Progress(
        SpinnerColumn(), TextColumn("[red]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Redact + verify", total=len(file_paths))

        def _on_progress(done: int, total: int) -> None:
            progress.update(task, completed=done)

        summary = redact_files(
            file_paths, str(out_dir), n_workers,
            salt, date_shift, verify, _on_progress,
        )

    t_total = time.time() - t0

    # Results
    clean = sum(1 for r in summary.results if r.verified_clean)
    dirty = summary.files_processed - clean
    if verify:
        if dirty == 0:
            console.print(f"[green]✓[/green] All {clean} files verified clean")
        else:
            console.print(f"[red]✗[/red] {dirty} files have remaining PHI")

    # Summary table
    table = Table(title="Redaction Actions", show_lines=True)
    table.add_column("Action", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("[red]Removed[/red]", str(summary.total_tags_removed))
    table.add_row("[yellow]Blanked[/yellow]", str(summary.total_tags_blanked))
    table.add_row("[cyan]Hashed[/cyan]", str(summary.total_tags_hashed))
    table.add_row("[magenta]Date-shifted[/magenta]", str(summary.total_tags_date_shifted))
    console.print(table)

    files_per_sec = summary.files_processed / max(t_total, 0.01)
    console.print(Panel(
        f"[bold]Redaction complete[/bold] in [bold]{t_total:.1f}s[/bold] "
        f"({files_per_sec:.0f} files/s)\n"
        f"  Files:        {summary.files_processed} processed, {summary.files_failed} failed\n"
        f"  Date shift:   {summary.date_shift_days} days\n"
        f"  Verified:     {clean} clean" + (f", {dirty} dirty" if dirty else "") + "\n"
        f"  Output:       {out_dir.resolve()}",
        title="Summary", border_style="red",
    ))

    # Write log
    log_path = out_dir / "redaction_log.json"
    log_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(folder),
        "output": str(out_dir),
        "files_processed": summary.files_processed,
        "files_failed": summary.files_failed,
        "date_shift_days": summary.date_shift_days,
        "tags_removed": summary.total_tags_removed,
        "tags_blanked": summary.total_tags_blanked,
        "tags_hashed": summary.total_tags_hashed,
        "tags_date_shifted": summary.total_tags_date_shifted,
        "verified_clean": clean,
        "time_s": t_total,
        "files_per_second": files_per_sec,
    }
    log_path.write_text(json.dumps(log_data, indent=2))
    console.print(f"[green]✓[/green] Log → {log_path}")

    return summary

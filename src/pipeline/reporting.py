"""Stage 5 — Cross-series comparison, stats, and HTML report + compression."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console

from ..compression import compress_file_multi, compression_ratio, format_size
from ..exports import export_cross_series_comparison
from ..report import generate_html_report

console = Console()


def generate_reports(
    patient_info: dict,
    series_info: list[dict],
    conformance_issues: list[dict],
    series_data_for_comparison: dict,
    image_paths: dict,
    out_dir: Path,
    n_workers: int,
    compress: bool = False,
) -> Path:
    """Generate cross-series comparison, stats JSON, and HTML report. Returns HTML path."""
    t_report = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        cross_fut = pool.submit(
            export_cross_series_comparison,
            series_data_for_comparison,
            str(out_dir),
        )
        stats_text = json.dumps(
            {
                "patient": patient_info,
                "series": series_info,
                "conformance_issues": conformance_issues,
            },
            indent=2,
            default=str,
        )
        stats_path = out_dir / "series_stats.json"
        stats_fut = pool.submit(stats_path.write_text, stats_text)

        cross_path = cross_fut.result()
        if cross_path:
            console.print(f"[green]✓[/green] Cross-series comparison → {cross_path}")

        html_path = generate_html_report(
            patient_info,
            series_info,
            conformance_issues,
            image_paths,
            cross_path,
            out_dir,
            pool,
        )

        stats_fut.result()

    console.print(f"[green]✓[/green] Stats → {stats_path}")
    console.print(f"[green]✓[/green] HTML report → {html_path}")

    if compress:
        _compress_reports(html_path, stats_path, stats_text)

    console.print(f"[dim]  Reports generated in {time.time() - t_report:.1f}s[/dim]")
    return html_path


def _compress_reports(html_path: Path, stats_path: Path, stats_text: str) -> None:
    """Compress HTML and stats JSON using centralized compression module."""
    html_size = html_path.stat().st_size
    stats_size = len(stats_text.encode())

    with ThreadPoolExecutor(max_workers=2) as pool:
        html_fut = pool.submit(compress_file_multi, html_path)
        stats_fut = pool.submit(compress_file_multi, stats_path)
        html_results = html_fut.result()
        stats_results = stats_fut.result()

    for label, orig_size, results in [
        ("report.html", html_size, html_results),
        ("series_stats.json", stats_size, stats_results),
    ]:
        console.print(f"  [cyan]Compressed {label}[/cyan] ({format_size(orig_size)}):")
        for fmt, (_, compressed_size) in sorted(results.items()):
            console.print(
                f"    {fmt} → {format_size(compressed_size)} ({compression_ratio(orig_size, compressed_size)})"
            )

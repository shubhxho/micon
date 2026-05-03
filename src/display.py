"""Rich console display — series tree and protocol table (built concurrently)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from rich import box
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

console = Console()


def _build_series_tree(series_info: list[dict]) -> Tree:
    """Build the series tree widget (no printing — pure data construction)."""
    tree = Tree("[bold cyan]DICOM Series[/bold cyan]")

    image_series = [s for s in series_info if s.get("has_pixels")]
    ps_series = [s for s in series_info if not s.get("has_pixels")]

    def _sort_key(s):
        try:
            return (0, int(s.get("series_number", 0)))
        except (ValueError, TypeError):
            return (1, str(s.get("series_number", "")))

    for s in sorted(image_series, key=_sort_key):
        snum = s.get("series_number", "?")
        desc = s.get("series_description", "?")
        seq = s.get("sequence_classification", {})
        seq_type = seq.get("sequence_type", "")
        vs = s.get("volume_stats", {})

        label = f"[bold]{snum}[/bold] — {desc}"
        if seq_type:
            label += f"  [magenta]({seq_type})[/magenta]"
        branch = tree.add(label)
        branch.add(
            f"[dim]{s.get('modality', '')}[/dim]  |  {s.get('file_count', 0)} files  |  {s.get('sop_class', '')}"
        )

        if vs:
            sp = vs.get("spacing_mm", [])
            sp_str = f"[{', '.join(f'{x:.2f}' for x in sp[:3])}]" if sp else "?"
            grade = vs.get("quality_grade", "?")
            grade_color = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "red"}.get(
                grade, "dim"
            )
            branch.add(
                f"[dim]Shape:[/dim] {vs.get('volume_shape', '?')}  |  [dim]Spacing:[/dim] {sp_str} mm  |  "
                f"[dim]SNR:[/dim] {vs.get('volume_snr_estimate', 0):.2f}  |  "
                f"[dim]Tissue:[/dim] {vs.get('volume_tissue_pct', 0):.1f}%  |  "
                f"[dim]Uniformity:[/dim] {vs.get('slice_intensity_uniformity', 0):.3f}  |  "
                f"[{grade_color}]Grade: {grade}[/{grade_color}]"
            )

            # Show quality warnings inline
            qa = s.get("quality_analysis", {})
            motion = qa.get("motion_analysis", {})
            anomaly = qa.get("anomaly_detection", {})
            symmetry = qa.get("symmetry_analysis", {})
            warnings = []
            if motion.get("motion_detected"):
                warnings.append(f"[yellow]motion: {motion.get('interpretation', '?')}[/yellow]")
            if anomaly.get("n_anomalous", 0) > 0:
                warnings.append(f"[yellow]{anomaly['n_anomalous']} anomalous slices[/yellow]")
            if symmetry.get("symmetry_index", 1) < 0.70:
                warnings.append(
                    f"[yellow]asymmetry: {symmetry.get('interpretation', '?')}[/yellow]"
                )
            if warnings:
                branch.add("  ".join(warnings))

    if ps_series:
        n_ps = len(ps_series)
        n_ps_files = sum(s.get("file_count", 0) for s in ps_series)
        tree.add(
            f"[dim]{n_ps} Presentation State series ({n_ps_files} files) — no pixel data[/dim]"
        )

    return tree


def _build_protocol_table(series_info: list[dict]) -> Table:
    """Build the protocol table widget (no printing — pure data construction)."""
    table = Table(title="MRI Protocol Summary", box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="bold", width=6)
    table.add_column("Description", width=22)
    table.add_column("Sequence", style="magenta", width=18)
    table.add_column("TR", width=8)
    table.add_column("TE", width=8)
    table.add_column("TI", width=8)
    table.add_column("FA", width=5)
    table.add_column("Matrix", width=12)
    table.add_column("Slices", width=6)
    table.add_column("SNR", width=7)
    table.add_column("Grade", width=5)

    for s in sorted(series_info, key=lambda s: str(s.get("series_number", ""))):
        if not s.get("has_pixels"):
            continue
        p = s.get("sequence_params", {})
        vs = s.get("volume_stats", {})
        shape = vs.get("volume_shape", [])
        grade = vs.get("quality_grade", "")
        gc = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "red"}.get(grade, "")
        grade_str = f"[{gc}]{grade}[/{gc}]" if gc else grade
        table.add_row(
            str(s.get("series_number", "")),
            s.get("series_description", "")[:22],
            s.get("sequence_classification", {}).get("sequence_type", "?")[:18],
            f"{p['tr']:.0f}" if p.get("tr") else "",
            f"{p['te']:.1f}" if p.get("te") else "",
            f"{p['ti']:.0f}" if p.get("ti") else "",
            f"{p['fa']:.0f}" if p.get("fa") else "",
            f"{shape[1]}x{shape[2]}" if len(shape) >= 3 else "",
            str(shape[0]) if shape else "",
            f"{vs.get('volume_snr_estimate', 0):.1f}" if vs else "",
            grade_str,
        )
    return table


def render_series_tree(series_info: list[dict]) -> None:
    console.print(_build_series_tree(series_info))


def render_protocol_table(series_info: list[dict]) -> None:
    console.print(_build_protocol_table(series_info))


def build_and_render_all(series_info: list[dict]) -> None:
    """Build tree + protocol table concurrently, then print both."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        tree_fut = pool.submit(_build_series_tree, series_info)
        table_fut = pool.submit(_build_protocol_table, series_info)
        tree = tree_fut.result()
        table = table_fut.result()
    console.print()
    console.print(tree)
    console.print()
    console.print(table)

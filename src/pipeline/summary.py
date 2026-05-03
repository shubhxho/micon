"""Stage 7 — Display results and summary panel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..display import _build_protocol_table, _build_series_tree

console = Console()


def _build_patient_table(patient_info: dict) -> Table | None:
    """Build the patient info table widget (no printing)."""
    if not patient_info:
        return None
    pt = Table("Field", "Value", title="Patient & Study")
    for k, v in patient_info.items():
        if v and v != "None":
            pt.add_row(k.replace("_", " ").title(), str(v))
    return pt


def display_results(
    patient_info: dict,
    series_info: list[dict],
) -> None:
    """Build all 3 display widgets concurrently, then print them."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        tree_fut = pool.submit(_build_series_tree, series_info)
        table_fut = pool.submit(_build_protocol_table, series_info)
        patient_fut = pool.submit(_build_patient_table, patient_info)

        tree = tree_fut.result()
        table = table_fut.result()
        patient_table = patient_fut.result()

    console.print()
    console.print(tree)
    console.print()
    console.print(table)
    if patient_table:
        console.print(patient_table)


def print_summary(
    elapsed: float,
    t_extract: float,
    n_files: int,
    n_image_uids: int,
    n_with_vol: int,
    n_ps_groups: int,
    n_tags: int,
    n_conformance: int,
    n_image_records: int,
    out_dir: Path,
    html_name: str,
    n_workers: int,
    series_results: list | None = None,
) -> None:
    """Print final summary panel with study-level quality grade."""
    from ..quality import grade_series, grade_study

    # Compute study-level quality grade
    study_grade_info = ""
    if series_results:
        grades = [grade_series(r.vstats) for r in series_results if r.vstats]
        if grades:
            sg = grade_study(grades)
            grade = sg["grade"]
            gc = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "red"}.get(
                grade, "dim"
            )
            dist = " ".join(f"{g}:{n}" for g, n in sorted(sg["grade_distribution"].items()))
            study_grade_info = (
                f"\n  [bold]Study grade:  [{gc}]{grade}[/{gc}][/bold] "
                f"(score: {sg['score']}/100, {dist})"
            )

    console.print(
        Panel(
            f"[bold]Extraction complete[/bold] in [bold]{elapsed:.1f}s[/bold]  "
            f"[dim](extract: {t_extract:.1f}s, series: {elapsed - t_extract:.1f}s)[/dim]\n"
            f"  Workers:       {n_workers}\n"
            f"  Files:         {n_files}\n"
            f"  Image series:  {n_image_uids} ({n_with_vol} with volumes)\n"
            f"  PS series:     {n_ps_groups} (skipped)\n"
            f"  Unique tags:   {n_tags}\n"
            f"  Conformance:   {n_conformance} issues in {n_image_records} image files\n"
            f"  Output:        {out_dir.resolve()}\n"
            f"  HTML report:   {html_name}"
            f"{study_grade_info}",
            title="Summary",
            border_style="green",
        )
    )

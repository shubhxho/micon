#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "polars>=1.17",
#   "rich>=13.9",
#   "typer>=0.15",
# ]
# ///

"""
parquet_extract.py — Extract and display all information from a Parquet file.

Usage:  uv run parquet_extract.py <file.parquet> [--head 20] [--export-csv] [--export-json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(rich_markup_mode="rich")
console = Console()


def file_size_str(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def schema_table(df: pl.DataFrame) -> Table:
    t = Table(title="Schema", box=box.ROUNDED, show_lines=True)
    t.add_column("#", style="dim", width=4)
    t.add_column("Column", style="cyan bold")
    t.add_column("Type", style="magenta")
    t.add_column("Nulls", style="yellow", justify="right")
    t.add_column("Null %", justify="right")
    t.add_column("Unique", justify="right")
    for i, (name, dtype) in enumerate(zip(df.columns, df.dtypes)):
        null_count = df[name].null_count()
        null_pct = f"{100 * null_count / len(df):.1f}%" if len(df) > 0 else "—"
        try:
            unique = str(df[name].n_unique())
        except Exception:
            unique = "—"
        t.add_row(str(i), name, str(dtype), str(null_count), null_pct, unique)
    return t


def stats_table(df: pl.DataFrame) -> Table:
    numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    if not numeric_cols:
        return None

    t = Table(title="Numeric Statistics", box=box.ROUNDED, show_lines=True)
    t.add_column("Column", style="cyan bold")
    for stat in ("min", "max", "mean", "median", "std", "p25", "p75"):
        t.add_column(stat, justify="right")

    for col in numeric_cols:
        s = df[col].drop_nulls()
        if len(s) == 0:
            continue
        try:
            t.add_row(
                col,
                f"{s.min():.4g}",
                f"{s.max():.4g}",
                f"{s.mean():.4g}",
                f"{s.median():.4g}",
                f"{s.std():.4g}",
                f"{s.quantile(0.25):.4g}",
                f"{s.quantile(0.75):.4g}",
            )
        except Exception:
            pass
    return t


def string_stats_table(df: pl.DataFrame) -> Table:
    str_cols = [c for c in df.columns if df[c].dtype == pl.Utf8 or df[c].dtype == pl.String]
    if not str_cols:
        return None

    t = Table(title="String/Categorical Stats", box=box.ROUNDED, show_lines=True)
    t.add_column("Column", style="cyan bold")
    t.add_column("Unique", justify="right")
    t.add_column("Most Frequent", style="green")
    t.add_column("Freq", justify="right")
    t.add_column("Avg Length", justify="right")

    for col in str_cols:
        s = df[col].drop_nulls()
        if len(s) == 0:
            continue
        try:
            unique = s.n_unique()
            top = s.value_counts().sort("count", descending=True).head(1)
            top_val = str(top[col][0])[:40] if len(top) > 0 else "—"
            top_count = str(top["count"][0]) if len(top) > 0 else "—"
            avg_len = f"{s.str.len_chars().mean():.1f}"
            t.add_row(col, str(unique), top_val, top_count, avg_len)
        except Exception:
            pass
    return t


def preview_table(df: pl.DataFrame, n: int) -> Table:
    preview = df.head(n)
    t = Table(title=f"Preview (first {min(n, len(df))} rows)", box=box.ROUNDED, show_lines=True)
    for col in preview.columns:
        t.add_column(col, max_width=30, overflow="ellipsis")
    for row in preview.iter_rows():
        t.add_row(*(str(v)[:30] for v in row))
    return t


@app.command()
def main(
    file: Annotated[Path, typer.Argument(help="Path to .parquet file")],
    head: Annotated[int, typer.Option("--head", help="Number of rows to preview")] = 10,
    export_csv: Annotated[bool, typer.Option("--export-csv", help="Export to CSV")] = False,
    export_json: Annotated[bool, typer.Option("--export-json", help="Export to JSON")] = False,
):
    """Extract and display all information from a Parquet file."""
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)

    df = pl.read_parquet(file)
    fsize = file_size_str(file.stat().st_size)

    # Overview
    console.print(Panel.fit(
        f"[bold cyan]Parquet Extractor[/bold cyan]\n"
        f"[dim]File:[/dim]    {file}\n"
        f"[dim]Size:[/dim]    {fsize}\n"
        f"[dim]Rows:[/dim]    {len(df):,}\n"
        f"[dim]Columns:[/dim] {len(df.columns)}",
        border_style="cyan",
    ))

    # Schema
    console.print(schema_table(df))

    # Numeric stats
    ns = stats_table(df)
    if ns:
        console.print(ns)

    # String stats
    ss = string_stats_table(df)
    if ss:
        console.print(ss)

    # Preview
    console.print(preview_table(df, head))

    # Exports
    if export_csv:
        out = file.with_suffix(".csv")
        df.write_csv(out)
        console.print(f"[green]Exported CSV:[/green] {out}")

    if export_json:
        out = file.with_suffix(".json")
        out.write_text(json.dumps(df.to_dicts(), indent=2, default=str))
        console.print(f"[green]Exported JSON:[/green] {out}")


if __name__ == "__main__":
    app()

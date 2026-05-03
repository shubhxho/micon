"""Typer CLI for the OME-Zarr export package.

Usage::

    # Convert a BIDS root to OME-Zarr (one .ome.zarr per subject)
    python -m src.zarr_export.cli convert --bids-root /data/bids --out /data/zarr

    # Inspect an existing OME-Zarr group (prints metadata)
    python -m src.zarr_export.cli inspect /data/zarr/sub-001.ome.zarr
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="zarr-export",
    help="OME-Zarr cloud-native export for the Speall MRI pipeline.",
    add_completion=False,
)


@app.command("convert")
def cmd_convert(
    bids_root: Annotated[
        Path,
        typer.Option("--bids-root", help="Root of the BIDS dataset to convert."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for OME-Zarr groups."),
    ],
    scales: Annotated[
        int,
        typer.Option("--scales", help="Number of pyramid levels (default 4)."),
    ] = 4,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Convert every NIfTI in a BIDS tree to OME-Zarr multiscale format."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from src.zarr_export.converter import study_to_omezarr  # lazy

    if not bids_root.exists():
        typer.echo(f"ERROR: --bids-root does not exist: {bids_root}", err=True)
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)

    # Walk subject directories (BIDS sub-* pattern)
    subject_dirs = sorted(d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-"))
    if not subject_dirs:
        # Fall back: treat entire bids_root as one study
        subject_dirs = [bids_root]

    total_converted = 0
    total_failed = 0

    for sub_dir in subject_dirs:
        typer.echo(f"Converting {sub_dir.name} ...")
        stats = study_to_omezarr(sub_dir, out)
        total_converted += stats["series_converted"]
        total_failed += stats["series_failed"]
        typer.echo(
            f"  -> {stats['study_zarr']}  "
            f"({stats['series_converted']} series, {stats['series_failed']} failed)"
        )

    typer.echo(f"\nDone. Converted {total_converted} series, {total_failed} failed.")
    if total_failed:
        raise typer.Exit(code=1)


@app.command("inspect")
def cmd_inspect(
    zarr_path: Annotated[
        str,
        typer.Argument(help="Path or fsspec URL to an OME-Zarr group."),
    ],
    pretty: Annotated[
        bool,
        typer.Option("--pretty", "-p", help="Pretty-print JSON output."),
    ] = True,
) -> None:
    """Print OME-Zarr metadata for a Zarr group."""
    try:
        import zarr  # lazy
    except ImportError:
        typer.echo("ERROR: zarr is not installed. Run: uv add zarr ome-zarr", err=True)
        raise typer.Exit(code=1)

    try:
        store = _open_store_for_inspect(zarr_path)
        grp = zarr.open_group(store=store, mode="r")
    except Exception as exc:
        typer.echo(f"ERROR opening Zarr group: {exc}", err=True)
        raise typer.Exit(code=1)

    attrs = dict(grp.attrs)
    if not attrs:
        typer.echo("(no .zattrs metadata found)")
        return

    indent = 2 if pretty else None
    typer.echo(json.dumps(attrs, indent=indent))

    # Print a brief summary of arrays in the group
    typer.echo("\nArrays:")
    _print_group_tree(grp, prefix="  ")


def _open_store_for_inspect(zarr_path: str) -> object:
    """Return a zarr store suitable for read-only inspection."""
    import zarr

    if "://" in zarr_path:
        import fsspec
        mapper = fsspec.get_mapper(zarr_path)
        return zarr.storage.FsspecStore(mapper.fs, path=mapper.root, read_only=True)
    return zarr.storage.LocalStore(zarr_path, read_only=True)


def _print_group_tree(grp: object, prefix: str = "") -> None:
    """Recursively print array shapes in a Zarr group."""
    import zarr

    for key in sorted(grp.keys()):  # type: ignore[attr-defined]
        item = grp[key]  # type: ignore[index]
        if isinstance(item, zarr.Array):
            typer.echo(f"{prefix}{key}: shape={item.shape} dtype={item.dtype}")
        elif isinstance(item, zarr.Group):
            typer.echo(f"{prefix}{key}/")
            _print_group_tree(item, prefix=prefix + "  ")


if __name__ == "__main__":
    app()

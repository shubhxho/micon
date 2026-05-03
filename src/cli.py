"""Unified Speall MRI pipeline CLI.

Single entry-point that wraps every existing workflow:

  speall dev           local quality + pack on one study
  speall annotate      local Gemma annotation iteration
  speall plan          cost + wall-time preview
  speall manifest      generate manifest.parquet
  speall pdf           regenerate prospectus
  speall resume        spawn full Modal resume pipeline
  speall upload        spawn focused HF upload (auto-manifest)
  speall backfill      backfill study_id on existing detail.json
  speall version       package version + git sha

Existing scripts (dev_run.py, upload_to_hf.py, etc.) are unchanged.
The CLI is additive -- choose either the script directly or ``speall <cmd>``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="speall", help="Speall MRI pipeline CLI", no_args_is_help=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> None:
    """Print the resolved command then stream it to the terminal."""
    typer.echo("$ " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


# ---------------------------------------------------------------------------
# Local commands
# ---------------------------------------------------------------------------

@app.command()
def dev(
    study: str = typer.Option(
        "mcap-files/3D_Ax_SWAN/",
        "--study",
        help="Path to a study directory.",
    ),
    skip_quality: bool = typer.Option(False, "--skip-quality", help="Skip quality + slice export."),
    skip_pack: bool = typer.Option(False, "--skip-pack", help="Skip tarball packing."),
) -> None:
    """Run quality + pack flow locally on a single study (mirrors dev_run.py)."""
    cmd = [sys.executable, "dev_run.py", "--study", study]
    if skip_quality:
        cmd.append("--skip-quality")
    if skip_pack:
        cmd.append("--skip-pack")
    _run(cmd)


@app.command()
def annotate(
    montage: str = typer.Option(..., "--montage", help="Path to a *_multiplane.png montage."),
    label: str = typer.Option(..., "--label", help='Series label, e.g. "Series 5 -- Ax DWI".'),
) -> None:
    """Annotate a single series montage locally using the Gemma model (mirrors dev_run.py --annotate)."""
    _run([sys.executable, "dev_run.py", "--annotate", "--montage", montage, "--label", label])


@app.command()
def plan(
    root: str = typer.Option(..., "--root", help="Pipeline output directory to scan."),
) -> None:
    """Print a cost + wall-time dry-run plan for the pipeline."""
    from rich.console import Console
    from src.dry_run import plan as _plan
    from src.cli_planner import _build_table

    output_dir = Path(root)
    console = Console()

    if not output_dir.exists():
        console.print(f"[bold red]Error:[/bold red] output_dir does not exist: {output_dir}")
        raise typer.Exit(code=1)

    console.print(f"Scanning [bold]{output_dir}[/bold] ...")
    data = _plan(output_dir)
    console.print(_build_table(data))
    console.print(
        "\n[dim]Cost model: Modal CPU $0.000111/s/CPU, quality ~30s@2CPU, "
        "annotate ~15s@1CPU; OpenRouter Gemma 4 31B $0.30/1M in + $0.50/1M out, "
        "~3000 in + 2000 out tokens/series. Wall time assumes ~50/320/20 concurrent.[/dim]"
    )


@app.command()
def manifest(
    root: str = typer.Option(..., "--root", help="Root directory to crawl for detail.json files."),
    out: str = typer.Option(..., "--out", help="Output directory for parquet files."),
) -> None:
    """Generate manifest.parquet and study_manifest.parquet from pipeline output."""
    from src.build_manifest import write_manifests

    counts = write_manifests(Path(root), Path(out))
    typer.echo(f"Wrote {counts['series_rows']} series rows -> {Path(out) / 'manifest.parquet'}")
    typer.echo(f"Wrote {counts['study_rows']} study rows  -> {Path(out) / 'study_manifest.parquet'}")


@app.command()
def pdf() -> None:
    """Regenerate the Speall MRI dataset prospectus PDF."""
    # generate_pdf.py lives at the repo root; add it to sys.path if needed.
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from generate_pdf import build
    build()


@app.command()
def sweep() -> None:
    """[DEPRECATED] PHI sweep has been removed from the workflow."""
    typer.echo("PHI sweep is no longer part of the workflow.")
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the package version and current git SHA."""
    pkg_version = _get_version()
    git_sha = _get_git_sha()
    typer.echo(f"speall {pkg_version}  git:{git_sha}")


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("micom")
    except Exception:
        pass
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return "unknown"
    toml_path = Path(__file__).parent.parent / "pyproject.toml"
    if toml_path.exists():
        with toml_path.open("rb") as fh:
            data = tomllib.load(fh)
        return data.get("project", {}).get("version", "unknown")
    return "unknown"


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Modal shell-out commands
# ---------------------------------------------------------------------------

@app.command()
def resume(
    repo: str = typer.Option("shubhxho/speall-mri", "--repo", help="HuggingFace repo ID."),
    skip_quality: bool = typer.Option(False, "--skip-quality", help="Skip quality stage."),
    skip_annotation: bool = typer.Option(False, "--skip-annotation", help="Skip annotation stage."),
    skip_pack: bool = typer.Option(False, "--skip-pack", help="Skip pack stage."),
) -> None:
    """Spawn the full Modal resume pipeline (quality + annotation + pack + HF upload)."""
    cmd = ["modal", "run", "--detach", "resume_pipeline.py", "--repo", repo]
    if skip_quality:
        cmd.append("--skip-quality")
    if skip_annotation:
        cmd.append("--skip-annotation")
    if skip_pack:
        cmd.append("--skip-pack")
    _run(cmd)


@app.command()
def upload(
    repo: str = typer.Option("shubhxho/speall-mri", "--repo", help="HuggingFace repo ID."),
    skip_manifest: bool = typer.Option(False, "--skip-manifest", help="Skip manifest generation."),
    squash: bool = typer.Option(False, "--squash", help="Squash HF commit history before upload."),
) -> None:
    """Spawn a focused HF upload via Modal (auto-generates manifest by default)."""
    cmd = ["modal", "run", "--detach", "upload_to_hf.py", "--repo", repo]
    if skip_manifest:
        cmd.append("--skip-manifest")
    if squash:
        cmd.append("--squash")
    _run(cmd)


@app.command()
def backfill(
    repo_dir: str = typer.Option("akai_mri", "--repo-dir", help="Sub-directory under /vol/output/ to backfill."),
) -> None:
    """Backfill study_id and pipeline_version into existing detail.json files via Modal."""
    _run(["modal", "run", "--detach", "backfill_metadata.py", "--repo-dir", repo_dir])


if __name__ == "__main__":
    app()

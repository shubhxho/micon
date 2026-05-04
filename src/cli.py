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
from typing import Annotated

from cyclopts import App, Parameter

app = App(name="speall", help="Speall MRI pipeline CLI")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> None:
    """Print the resolved command then stream it to the terminal."""
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Local commands
# ---------------------------------------------------------------------------


@app.command
def dev(
    study: Annotated[
        str, Parameter(name="--study", help="Path to a study directory.")
    ] = "mcap-files/3D_Ax_SWAN/",
    skip_quality: Annotated[
        bool, Parameter(name="--skip-quality", help="Skip quality + slice export.")
    ] = False,
    skip_pack: Annotated[bool, Parameter(name="--skip-pack", help="Skip tarball packing.")] = False,
) -> None:
    """Run quality + pack flow locally on a single study (mirrors dev_run.py)."""
    cmd = [sys.executable, "dev_run.py", "--study", study]
    if skip_quality:
        cmd.append("--skip-quality")
    if skip_pack:
        cmd.append("--skip-pack")
    _run(cmd)


@app.command
def annotate(
    montage: Annotated[
        str, Parameter(name="--montage", help="Path to a *_multiplane.png montage.")
    ],
    label: Annotated[
        str, Parameter(name="--label", help='Series label, e.g. "Series 5 -- Ax DWI".')
    ],
) -> None:
    """Annotate a single series montage locally using the Gemma model (mirrors dev_run.py --annotate)."""
    _run([sys.executable, "dev_run.py", "--annotate", "--montage", montage, "--label", label])


@app.command
def plan(
    root: Annotated[str, Parameter(name="--root", help="Pipeline output directory to scan.")],
) -> None:
    """Print a cost + wall-time dry-run plan for the pipeline."""
    from rich.console import Console

    from src.pipeline.cli_planner import _build_table
    from src.pipeline.plan import plan as _plan

    output_dir = Path(root)
    console = Console()

    if not output_dir.exists():
        console.print(f"[bold red]Error:[/bold red] output_dir does not exist: {output_dir}")
        sys.exit(1)

    console.print(f"Scanning [bold]{output_dir}[/bold] ...")
    data = _plan(output_dir)
    console.print(_build_table(data))
    console.print(
        "\n[dim]Cost model: Modal CPU $0.000111/s/CPU, quality ~30s@2CPU, "
        "annotate ~15s@1CPU; OpenRouter Gemma 4 31B $0.30/1M in + $0.50/1M out, "
        "~3000 in + 2000 out tokens/series. Wall time assumes ~50/320/20 concurrent.[/dim]"
    )


@app.command
def manifest(
    root: Annotated[
        str, Parameter(name="--root", help="Root directory to crawl for detail.json files.")
    ],
    out: Annotated[str, Parameter(name="--out", help="Output directory for parquet files.")],
) -> None:
    """Generate manifest.parquet and study_manifest.parquet from pipeline output."""
    from src.manifest.builder import write_manifests

    counts = write_manifests(Path(root), Path(out))
    print(f"Wrote {counts['series_rows']} series rows -> {Path(out) / 'manifest.parquet'}")
    print(f"Wrote {counts['study_rows']} study rows  -> {Path(out) / 'study_manifest.parquet'}")


@app.command(name="sqlite-export")
def sqlite_export(
    output_dir: Annotated[
        str,
        Parameter(
            name="--output-dir",
            help="Directory containing manifest.parquet (and optionally study_manifest.parquet).",
        ),
    ],
    db_path: Annotated[
        str,
        Parameter(name="--db-path", help="Destination SQLite database file."),
    ] = "manifest.db",
    no_views: Annotated[
        bool,
        Parameter(name="--no-views", help="Skip creating the series_overview view."),
    ] = False,
) -> None:
    """Export manifest.parquet -> SQLite for Datasette browsing."""
    from src.manifest.sqlite_export import manifest_to_sqlite, write_datasette_metadata

    out_dir = Path(output_dir)
    db = Path(db_path)

    parquets: list[Path] = []
    for name in ("manifest.parquet", "study_manifest.parquet"):
        candidate = out_dir / name
        if candidate.exists():
            parquets.append(candidate)
    if not parquets:
        print(f"No manifest parquet files found in {out_dir}", file=sys.stderr)
        sys.exit(1)

    db_out = manifest_to_sqlite(parquets, db, with_views=not no_views)
    meta_out = write_datasette_metadata(db_out)
    print(f"Wrote SQLite database -> {db_out}")
    print(f"Wrote Datasette metadata -> {meta_out}")
    print(f"Now run: uv run datasette serve {db_out} --metadata {meta_out}")


@app.command
def datapackage(
    bundle_dir: Annotated[
        str, Parameter(name="--bundle-dir", help="Directory containing the bundle artifacts.")
    ],
    name: Annotated[
        str, Parameter(name="--name", help="Frictionless package name (slug).")
    ] = "speall-mri-bundle",
    pkg_version: Annotated[
        str,
        Parameter(name="--pkg-version", help="Semantic version of the package."),
    ] = "1.0.0",
) -> None:
    """Emit a Frictionless `datapackage.json` envelope inside BUNDLE_DIR."""
    from src.integrity.datapackage import build_datapackage, write_datapackage

    root = Path(bundle_dir)
    if not root.is_dir():
        print(f"Error: bundle_dir does not exist: {root}", file=sys.stderr)
        sys.exit(1)
    pkg = build_datapackage(root, name=name, version=pkg_version)
    out = write_datapackage(root, pkg)
    print(f"Wrote {out} ({len(pkg['resources'])} resources)")


@app.command
def pdf() -> None:
    """Regenerate the Speall MRI dataset prospectus PDF."""
    # generate_pdf.py lives at the repo root; add it to sys.path if needed.
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from generate_pdf import build

    build()


@app.command
def sweep() -> None:
    """[DEPRECATED] PHI sweep has been removed from the workflow."""
    print("PHI sweep is no longer part of the workflow.")
    sys.exit(1)


@app.command
def version() -> None:
    """Print the package version and current git SHA."""
    pkg_version = _get_version()
    git_sha = _get_git_sha()
    print(f"speall {pkg_version}  git:{git_sha}")


def _get_version() -> str:
    try:
        from importlib.metadata import version as _version

        return _version("micom")
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


@app.command
def resume(
    repo: Annotated[
        str, Parameter(name="--repo", help="HuggingFace repo ID.")
    ] = "shubhxho/speall-mri",
    skip_quality: Annotated[
        bool, Parameter(name="--skip-quality", help="Skip quality stage.")
    ] = False,
    skip_annotation: Annotated[
        bool, Parameter(name="--skip-annotation", help="Skip annotation stage.")
    ] = False,
    skip_pack: Annotated[bool, Parameter(name="--skip-pack", help="Skip pack stage.")] = False,
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


@app.command
def upload(
    repo: Annotated[
        str, Parameter(name="--repo", help="HuggingFace repo ID.")
    ] = "shubhxho/speall-mri",
    skip_manifest: Annotated[
        bool, Parameter(name="--skip-manifest", help="Skip manifest generation.")
    ] = False,
    squash: Annotated[
        bool, Parameter(name="--squash", help="Squash HF commit history before upload.")
    ] = False,
) -> None:
    """Spawn a focused HF upload via Modal (auto-generates manifest by default)."""
    cmd = ["modal", "run", "--detach", "upload_to_hf.py", "--repo", repo]
    if skip_manifest:
        cmd.append("--skip-manifest")
    if squash:
        cmd.append("--squash")
    _run(cmd)


@app.command
def backfill(
    repo_dir: Annotated[
        str, Parameter(name="--repo-dir", help="Sub-directory under /vol/output/ to backfill.")
    ] = "akai_mri",
) -> None:
    """Backfill study_id and pipeline_version into existing detail.json files via Modal."""
    _run(["modal", "run", "--detach", "backfill_metadata.py", "--repo-dir", repo_dir])


if __name__ == "__main__":
    app()

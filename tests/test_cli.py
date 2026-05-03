"""Tests for the unified speall Typer CLI (src/cli.py).

Run with:
    uv run --with typer --with rich --with pytest pytest tests/test_cli.py -v
"""

from __future__ import annotations

from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Top-level help
# ---------------------------------------------------------------------------


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    output = result.output
    expected_commands = [
        "dev",
        "annotate",
        "plan",
        "manifest",
        "pdf",
        "sweep",
        "version",
        "resume",
        "upload",
        "backfill",
    ]
    for cmd in expected_commands:
        assert cmd in output, f"command '{cmd}' not found in --help output"


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


def test_version_exits_zero():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0


def test_version_output_contains_speall():
    result = runner.invoke(app, ["version"])
    assert "speall" in result.output


# ---------------------------------------------------------------------------
# sweep (deprecated, must exit non-zero)
# ---------------------------------------------------------------------------


def test_sweep_exits_nonzero():
    result = runner.invoke(app, ["sweep"])
    assert result.exit_code != 0


def test_sweep_deprecation_message():
    result = runner.invoke(app, ["sweep"])
    assert "no longer part of the workflow" in result.output


# ---------------------------------------------------------------------------
# --help on Modal shell-out commands (never actually invokes Modal)
# ---------------------------------------------------------------------------


def test_resume_help_exits_zero():
    result = runner.invoke(app, ["resume", "--help"])
    assert result.exit_code == 0


def test_resume_help_shows_repo_option():
    result = runner.invoke(app, ["resume", "--help"])
    assert "--repo" in result.output


def test_upload_help_exits_zero():
    result = runner.invoke(app, ["upload", "--help"])
    assert result.exit_code == 0


def test_upload_help_shows_options():
    result = runner.invoke(app, ["upload", "--help"])
    assert "--repo" in result.output
    assert "--skip-manifest" in result.output
    assert "--squash" in result.output


def test_backfill_help_exits_zero():
    result = runner.invoke(app, ["backfill", "--help"])
    assert result.exit_code == 0


def test_backfill_help_shows_repo_dir():
    result = runner.invoke(app, ["backfill", "--help"])
    assert "--repo-dir" in result.output


# ---------------------------------------------------------------------------
# --help on local commands
# ---------------------------------------------------------------------------


def test_dev_help_exits_zero():
    result = runner.invoke(app, ["dev", "--help"])
    assert result.exit_code == 0


def test_dev_help_shows_options():
    result = runner.invoke(app, ["dev", "--help"])
    assert "--study" in result.output
    assert "--skip-quality" in result.output
    assert "--skip-pack" in result.output


def test_annotate_help_exits_zero():
    result = runner.invoke(app, ["annotate", "--help"])
    assert result.exit_code == 0


def test_annotate_help_shows_options():
    result = runner.invoke(app, ["annotate", "--help"])
    assert "--montage" in result.output
    assert "--label" in result.output


def test_plan_help_exits_zero():
    result = runner.invoke(app, ["plan", "--help"])
    assert result.exit_code == 0


def test_manifest_help_exits_zero():
    result = runner.invoke(app, ["manifest", "--help"])
    assert result.exit_code == 0


def test_manifest_help_shows_options():
    result = runner.invoke(app, ["manifest", "--help"])
    assert "--root" in result.output
    assert "--out" in result.output


def test_pdf_help_exits_zero():
    result = runner.invoke(app, ["pdf", "--help"])
    assert result.exit_code == 0

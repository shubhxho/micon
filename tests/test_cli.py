"""Tests for the unified speall Cyclopts CLI (src/cli.py).

Run with:
    uv run pytest tests/test_cli.py -v

Note: Previously used typer.testing.CliRunner. Migrated to direct invocation
with captured output (capsys/capfd) after migrating src/cli.py from Typer to
Cyclopts. Cyclopts raises SystemExit on --help (exit 0) and on errors.
"""

from __future__ import annotations

import contextlib
import io

import pytest
from rich.console import Console

from src.cli import app


def _invoke(args: list[str]) -> tuple[str, int]:
    """Invoke the cyclopts app and return (stdout_text, exit_code).

    Cyclopts raises SystemExit for --help and for sys.exit() calls.
    We capture all Rich console output by injecting a StringIO console.
    Plain print() calls are captured via capsys in individual tests,
    but for help output we redirect the app's console.
    """
    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=False, width=120)
    exit_code = 0
    try:
        app(args, console=console)
    except SystemExit as e:
        exit_code = int(e.code) if e.code is not None else 0
    return buf.getvalue(), exit_code


# ---------------------------------------------------------------------------
# Top-level help
# ---------------------------------------------------------------------------


def test_help_exits_zero():
    _, code = _invoke(["--help"])
    assert code == 0


def test_help_lists_all_commands():
    output, _ = _invoke(["--help"])
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


def test_version_exits_zero(capsys):
    exit_code = 0
    try:
        app(["version"])
    except SystemExit as e:
        exit_code = int(e.code) if e.code is not None else 0
    assert exit_code == 0


def test_version_output_contains_speall(capsys):
    with contextlib.suppress(SystemExit):
        app(["version"])
    captured = capsys.readouterr()
    assert "speall" in captured.out


# ---------------------------------------------------------------------------
# sweep (deprecated, must exit non-zero)
# ---------------------------------------------------------------------------


def test_sweep_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        app(["sweep"])
    assert exc_info.value.code != 0


def test_sweep_deprecation_message(capsys):
    with pytest.raises(SystemExit):
        app(["sweep"])
    captured = capsys.readouterr()
    assert "no longer part of the workflow" in captured.out


# ---------------------------------------------------------------------------
# --help on Modal shell-out commands (never actually invokes Modal)
# ---------------------------------------------------------------------------


def test_resume_help_exits_zero():
    _, code = _invoke(["resume", "--help"])
    assert code == 0


def test_resume_help_shows_repo_option():
    output, _ = _invoke(["resume", "--help"])
    assert "--repo" in output


def test_upload_help_exits_zero():
    _, code = _invoke(["upload", "--help"])
    assert code == 0


def test_upload_help_shows_options():
    output, _ = _invoke(["upload", "--help"])
    assert "--repo" in output
    assert "--skip-manifest" in output
    assert "--squash" in output


def test_backfill_help_exits_zero():
    _, code = _invoke(["backfill", "--help"])
    assert code == 0


def test_backfill_help_shows_repo_dir():
    output, _ = _invoke(["backfill", "--help"])
    assert "--repo-dir" in output


# ---------------------------------------------------------------------------
# --help on local commands
# ---------------------------------------------------------------------------


def test_dev_help_exits_zero():
    _, code = _invoke(["dev", "--help"])
    assert code == 0


def test_dev_help_shows_options():
    output, _ = _invoke(["dev", "--help"])
    assert "--study" in output
    assert "--skip-quality" in output
    assert "--skip-pack" in output


def test_annotate_help_exits_zero():
    _, code = _invoke(["annotate", "--help"])
    assert code == 0


def test_annotate_help_shows_options():
    output, _ = _invoke(["annotate", "--help"])
    assert "--montage" in output
    assert "--label" in output


def test_plan_help_exits_zero():
    _, code = _invoke(["plan", "--help"])
    assert code == 0


def test_manifest_help_exits_zero():
    _, code = _invoke(["manifest", "--help"])
    assert code == 0


def test_manifest_help_shows_options():
    output, _ = _invoke(["manifest", "--help"])
    assert "--root" in output
    assert "--out" in output


def test_pdf_help_exits_zero():
    _, code = _invoke(["pdf", "--help"])
    assert code == 0

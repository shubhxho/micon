"""Shared pytest fixtures and marker declarations for the micom test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the repo root importable (for `from src.X import ...` style) and also
# make the src/ directory itself importable (for bare `import stage_sentinels`).
_REPO_ROOT = str(Path(__file__).parent.parent)
_SRC_DIR = str(Path(__file__).parent.parent / "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Marker declarations
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (deselect with -m 'not slow')",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    """A tmp_path-based directory with a pre-created _pipeline_state sub-dir."""
    state_dir = tmp_path / "_pipeline_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def sample_detail_json() -> dict:
    """Load Speall_MRI_Samples/series/s0005_Ax_DWI.json as a dict."""
    repo_root = Path(__file__).parent.parent
    json_path = repo_root / "Speall_MRI_Samples" / "series" / "s0005_Ax_DWI.json"
    return json.loads(json_path.read_text(encoding="utf-8"))


@pytest.fixture()
def sample_study_dir() -> Path:
    """Return path to Speall_MRI_Samples/series/."""
    repo_root = Path(__file__).parent.parent
    return repo_root / "Speall_MRI_Samples" / "series"

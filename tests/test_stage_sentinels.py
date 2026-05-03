"""Pure pytest tests for src/stage_sentinels.py.

Coverage: mark_started, mark_finished, is_done, read_state, summarize, atomic writes.
Plus one Hypothesis property test: for any (ok, errors, metadata), a complete
mark_started -> mark_finished cycle produces a sentinel where is_done == ok.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stage_sentinels import (
    is_done,
    mark_finished,
    mark_started,
    read_state,
    summarize,
)


# ---------------------------------------------------------------------------
# Basic sentinel lifecycle
# ---------------------------------------------------------------------------

def test_mark_started_creates_sentinel(tmp_path: Path) -> None:
    path = mark_started(tmp_path, "quality")
    assert path.exists(), "sentinel file is created by mark_started"


def test_mark_started_state_fields(tmp_path: Path) -> None:
    mark_started(tmp_path, "quality")
    state = read_state(tmp_path, "quality")
    assert state is not None, "read_state returns a dict"
    assert state["stage"] == "quality"
    assert state["status"] == "in_progress"
    assert state["started_at"] is not None
    assert state["finished_at"] is None


def test_is_done_false_while_in_progress(tmp_path: Path) -> None:
    mark_started(tmp_path, "annotation")
    assert not is_done(tmp_path, "annotation"), "is_done False when in_progress"


def test_is_done_false_before_any_sentinel(tmp_path: Path) -> None:
    assert not is_done(tmp_path, "nonexistent"), "is_done False when no sentinel exists"


def test_mark_finished_ok(tmp_path: Path) -> None:
    mark_started(tmp_path, "pack")
    mark_finished(tmp_path, "pack", ok=True, errors=0, inputs_processed=42)
    assert is_done(tmp_path, "pack"), "is_done True after mark_finished ok=True"
    state = read_state(tmp_path, "pack")
    assert state is not None
    assert state["status"] == "ok"
    assert state["errors"] == 0
    assert state["inputs_processed"] == 42
    assert state["finished_at"] is not None


def test_started_at_preserved_across_mark_finished(tmp_path: Path) -> None:
    mark_started(tmp_path, "upload")
    started_state = read_state(tmp_path, "upload")
    assert started_state is not None
    started_at = started_state["started_at"]

    mark_finished(tmp_path, "upload", ok=True)
    finished_state = read_state(tmp_path, "upload")
    assert finished_state is not None
    assert finished_state["started_at"] == started_at, "started_at preserved through mark_finished"


def test_mark_finished_failed(tmp_path: Path) -> None:
    mark_started(tmp_path, "quality")
    mark_finished(tmp_path, "quality", ok=False, errors=3)
    assert not is_done(tmp_path, "quality"), "is_done False after mark_finished ok=False"
    state = read_state(tmp_path, "quality")
    assert state is not None
    assert state["status"] == "failed"
    assert state["errors"] == 3


def test_metadata_merge(tmp_path: Path) -> None:
    mark_started(tmp_path, "pack", metadata={"study_id": "ABC"})
    mark_finished(tmp_path, "pack", ok=True, metadata={"output_path": "/vol/pack.tar"})
    state = read_state(tmp_path, "pack")
    assert state is not None
    assert state["metadata"].get("study_id") == "ABC", "original metadata key preserved"
    assert state["metadata"].get("output_path") == "/vol/pack.tar", "new metadata key added"


def test_summarize_returns_all_stages(tmp_path: Path) -> None:
    mark_started(tmp_path, "stage_a")
    mark_started(tmp_path, "stage_b")
    mark_finished(tmp_path, "stage_b", ok=True)
    summary = summarize(tmp_path)
    assert "stage_a" in summary
    assert "stage_b" in summary
    assert summary["stage_a"]["status"] == "in_progress"
    assert summary["stage_b"]["status"] == "ok"


def test_summarize_empty_dir(tmp_path: Path) -> None:
    summary = summarize(tmp_path)
    assert summary == {}, "summarize returns empty dict when no sentinels exist"


def test_mark_finished_without_prior_mark_started(tmp_path: Path) -> None:
    """mark_finished should not crash when called without a prior mark_started."""
    mark_finished(tmp_path, "orphan", ok=True)
    state = read_state(tmp_path, "orphan")
    assert state is not None
    assert state["status"] == "ok", "orphan sentinel created with ok status"


# ---------------------------------------------------------------------------
# Atomic write: .tmp file is replaced, leaving no leftover
# ---------------------------------------------------------------------------

def test_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    mark_started(tmp_path, "quality")
    state_dir = tmp_path / "_pipeline_state"
    tmp_files = list(state_dir.glob("*.tmp"))
    assert tmp_files == [], "no leftover .tmp files after mark_started"


# ---------------------------------------------------------------------------
# Hypothesis property test
# ---------------------------------------------------------------------------

_safe_metadata: st.SearchStrategy[dict[str, Any]] = st.dictionaries(
    keys=st.text(min_size=1, max_size=20).filter(str.isidentifier),
    values=st.one_of(st.text(max_size=30), st.integers(), st.booleans()),
    max_size=5,
)


@given(
    ok=st.booleans(),
    errors=st.integers(min_value=0, max_value=100),
    metadata=_safe_metadata,
)
@settings(max_examples=50)
def test_is_done_equals_ok_after_cycle(ok: bool, errors: int, metadata: dict) -> None:
    """For any (ok, errors, metadata), mark_started then mark_finished gives is_done == ok.

    Uses an internal tempdir rather than a fixture so Hypothesis can manage
    its own per-example isolation without triggering the function_scoped_fixture
    health check.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mark_started(tmp, "prop_stage")
        mark_finished(tmp, "prop_stage", ok=ok, errors=errors, metadata=metadata)
        assert is_done(tmp, "prop_stage") == ok

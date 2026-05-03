"""Smoke tests for src/stage_sentinels.py.

Run with:
    python tests/test_stage_sentinels.py

All tests use a temporary directory so they never pollute the repo.
Exit code is 0 on success, 1 on any assertion failure.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

# Make sure the project src/ is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stage_sentinels import (  # noqa: E402
    is_done,
    mark_finished,
    mark_started,
    read_state,
    summarize,
)

_FAILURES: list[str] = []


def _assert(condition: bool, message: str) -> None:
    if not condition:
        _FAILURES.append(message)
        print(f"  FAIL: {message}")
    else:
        print(f"  ok:   {message}")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def test_mark_started_creates_sentinel() -> None:
    print("test_mark_started_creates_sentinel")
    with tempfile.TemporaryDirectory() as tmp:
        path = mark_started(tmp, "quality")
        _assert(path.exists(), "sentinel file is created by mark_started")
        state = read_state(tmp, "quality")
        _assert(state is not None, "read_state returns a dict")
        assert state is not None
        _assert(state["stage"] == "quality", "stage field matches")
        _assert(state["status"] == "in_progress", "status is in_progress after mark_started")
        _assert(state["started_at"] is not None, "started_at is populated")
        _assert(state["finished_at"] is None, "finished_at is None before finish")


def test_is_done_false_while_in_progress() -> None:
    print("test_is_done_false_while_in_progress")
    with tempfile.TemporaryDirectory() as tmp:
        mark_started(tmp, "annotation")
        _assert(not is_done(tmp, "annotation"), "is_done False when in_progress")


def test_is_done_false_before_any_sentinel() -> None:
    print("test_is_done_false_before_any_sentinel")
    with tempfile.TemporaryDirectory() as tmp:
        _assert(not is_done(tmp, "nonexistent"), "is_done False when no sentinel exists")


def test_mark_finished_ok() -> None:
    print("test_mark_finished_ok")
    with tempfile.TemporaryDirectory() as tmp:
        mark_started(tmp, "pack")
        mark_finished(tmp, "pack", ok=True, errors=0, inputs_processed=42)
        _assert(is_done(tmp, "pack"), "is_done True after mark_finished ok=True")
        state = read_state(tmp, "pack")
        assert state is not None
        _assert(state["status"] == "ok", "status is ok")
        _assert(state["errors"] == 0, "errors field is 0")
        _assert(state["inputs_processed"] == 42, "inputs_processed is 42")
        _assert(state["finished_at"] is not None, "finished_at is populated after finish")


def test_started_at_preserved_across_mark_finished() -> None:
    print("test_started_at_preserved_across_mark_finished")
    with tempfile.TemporaryDirectory() as tmp:
        mark_started(tmp, "upload")
        started_state = read_state(tmp, "upload")
        assert started_state is not None
        started_at = started_state["started_at"]

        mark_finished(tmp, "upload", ok=True)
        finished_state = read_state(tmp, "upload")
        assert finished_state is not None
        _assert(
            finished_state["started_at"] == started_at,
            "started_at is preserved through mark_finished",
        )


def test_mark_finished_failed() -> None:
    print("test_mark_finished_failed")
    with tempfile.TemporaryDirectory() as tmp:
        mark_started(tmp, "quality")
        mark_finished(tmp, "quality", ok=False, errors=3)
        _assert(not is_done(tmp, "quality"), "is_done False after mark_finished ok=False")
        state = read_state(tmp, "quality")
        assert state is not None
        _assert(state["status"] == "failed", "status is failed")
        _assert(state["errors"] == 3, "errors field is 3")


def test_metadata_merge() -> None:
    print("test_metadata_merge")
    with tempfile.TemporaryDirectory() as tmp:
        mark_started(tmp, "pack", metadata={"study_id": "ABC"})
        mark_finished(tmp, "pack", ok=True, metadata={"output_path": "/vol/pack.tar"})
        state = read_state(tmp, "pack")
        assert state is not None
        _assert(state["metadata"].get("study_id") == "ABC", "original metadata key preserved")
        _assert(
            state["metadata"].get("output_path") == "/vol/pack.tar",
            "new metadata key added by mark_finished",
        )


def test_summarize_returns_all_stages() -> None:
    print("test_summarize_returns_all_stages")
    with tempfile.TemporaryDirectory() as tmp:
        mark_started(tmp, "stage_a")
        mark_started(tmp, "stage_b")
        mark_finished(tmp, "stage_b", ok=True)
        summary = summarize(tmp)
        _assert("stage_a" in summary, "stage_a appears in summarize output")
        _assert("stage_b" in summary, "stage_b appears in summarize output")
        _assert(summary["stage_a"]["status"] == "in_progress", "stage_a still in_progress")
        _assert(summary["stage_b"]["status"] == "ok", "stage_b is ok")


def test_summarize_empty_dir() -> None:
    print("test_summarize_empty_dir")
    with tempfile.TemporaryDirectory() as tmp:
        summary = summarize(tmp)
        _assert(summary == {}, "summarize returns empty dict when no sentinels exist")


def test_mark_finished_without_prior_mark_started() -> None:
    """mark_finished should not crash when called without a prior mark_started."""
    print("test_mark_finished_without_prior_mark_started")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            mark_finished(tmp, "orphan", ok=True)
            state = read_state(tmp, "orphan")
            assert state is not None
            _assert(state["status"] == "ok", "orphan sentinel created with ok status")
        except Exception as exc:  # noqa: BLE001
            _assert(False, f"mark_finished without mark_started raised: {exc}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all() -> int:
    tests = [
        test_mark_started_creates_sentinel,
        test_is_done_false_while_in_progress,
        test_is_done_false_before_any_sentinel,
        test_mark_finished_ok,
        test_started_at_preserved_across_mark_finished,
        test_mark_finished_failed,
        test_metadata_merge,
        test_summarize_returns_all_stages,
        test_summarize_empty_dir,
        test_mark_finished_without_prior_mark_started,
    ]
    for test_fn in tests:
        try:
            test_fn()
        except Exception:  # noqa: BLE001
            _FAILURES.append(f"{test_fn.__name__} raised an unexpected exception")
            traceback.print_exc()

    print()
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)} failure(s)):")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print(f"All {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())

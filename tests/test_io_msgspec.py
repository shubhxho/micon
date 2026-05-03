"""
Tests for src.io.msgspec_io hot-path helpers.

TDD London School: each test drives one observable behaviour of the
fast_dumps / fast_loads / write_json / read_json public API.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.io.msgspec_io import fast_dumps, fast_loads, read_json, write_json

# ---------------------------------------------------------------------------
# fast_dumps
# ---------------------------------------------------------------------------


class TestFastDumps:
    def test_returns_bytes(self) -> None:
        result = fast_dumps({"key": "value"})
        assert isinstance(result, bytes)

    def test_indented_output(self) -> None:
        result = fast_dumps({"a": 1})
        assert b"\n" in result, "fast_dumps must produce indented (multi-line) output"

    def test_numpy_array_serialised(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = fast_dumps({"arr": arr})
        parsed = json.loads(result)
        assert parsed["arr"] == [1.0, 2.0, 3.0]

    def test_numpy_int_scalar_serialised(self) -> None:
        result = fast_dumps({"v": np.int32(7)})
        parsed = json.loads(result)
        assert parsed["v"] == 7

    def test_datetime_serialised(self) -> None:
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        result = fast_dumps({"ts": dt})
        parsed = json.loads(result)
        assert "2024" in parsed["ts"]

    def test_nested_list_preserved(self) -> None:
        obj = {"outer": [{"inner": [1, 2, 3]}, {"x": None}]}
        result = fast_dumps(obj)
        assert json.loads(result) == obj

    def test_valid_json_parseable_by_stdlib(self) -> None:
        obj = {"score": 0.9, "tags": ["a", "b"], "meta": {"k": True}}
        result = fast_dumps(obj)
        parsed = json.loads(result)
        assert parsed == obj

    def test_byte_identical_to_stdlib_for_simple_dict(self) -> None:
        """For all-stdlib-types dicts, orjson OPT_INDENT_2 must match json.dumps(indent=2)."""
        d = {"name": "test", "count": 42, "scores": [1, 2, 3], "nested": {"a": True}}
        stdlib_bytes = json.dumps(d, indent=2).encode()
        fast_bytes = fast_dumps(d)
        assert fast_bytes == stdlib_bytes


# ---------------------------------------------------------------------------
# fast_loads
# ---------------------------------------------------------------------------


class TestFastLoads:
    def test_parses_bytes(self) -> None:
        data = b'{"a": 1, "b": [2, 3]}'
        result = fast_loads(data)
        assert result == {"a": 1, "b": [2, 3]}

    def test_parses_str(self) -> None:
        data = '{"x": "hello"}'
        result = fast_loads(data)
        assert result["x"] == "hello"

    def test_round_trips_fast_dumps(self) -> None:
        obj = {"key": "value", "nums": [1, 2, 3], "flag": True, "nothing": None}
        assert fast_loads(fast_dumps(obj)) == obj


# ---------------------------------------------------------------------------
# write_json / read_json round-trip
# ---------------------------------------------------------------------------


class TestWriteReadJson:
    def test_round_trip_simple_dict(self, tmp_path: Path) -> None:
        obj = {"series_uid": "1.2.3", "count": 5, "tags": ["mri", "brain"]}
        p = tmp_path / "out.json"
        write_json(p, obj)
        assert read_json(p) == obj

    def test_round_trip_with_numpy_array(self, tmp_path: Path) -> None:
        obj = {"volume_shape": np.array([64, 64, 32]), "label": "T1"}
        p = tmp_path / "vol.json"
        write_json(p, obj)
        result = read_json(p)
        assert result["volume_shape"] == [64, 64, 32]
        assert result["label"] == "T1"

    def test_round_trip_with_datetime(self, tmp_path: Path) -> None:
        dt = datetime(2025, 3, 1, 10, 30, 0, tzinfo=UTC)
        obj = {"timestamp": dt, "study_id": "study_001"}
        p = tmp_path / "dt.json"
        write_json(p, obj)
        result = read_json(p)
        assert "2025" in result["timestamp"]
        assert result["study_id"] == "study_001"

    def test_round_trip_with_nested_lists(self, tmp_path: Path) -> None:
        obj = {
            "matrix": [[1, 2, 3], [4, 5, 6]],
            "deep": {"inner": [{"x": 1}, {"x": 2}]},
        }
        p = tmp_path / "nested.json"
        write_json(p, obj)
        assert read_json(p) == obj

    def test_output_is_valid_stdlib_json(self, tmp_path: Path) -> None:
        obj = {"a": 1, "b": [True, None, 3.14]}
        p = tmp_path / "valid.json"
        write_json(p, obj)
        parsed = json.loads(p.read_bytes())
        assert parsed == obj

    def test_output_file_is_indented(self, tmp_path: Path) -> None:
        obj = {"key": "value", "nested": {"inner": 1}}
        p = tmp_path / "indented.json"
        write_json(p, obj)
        raw = p.read_text()
        assert "\n" in raw, "write_json must produce indented output"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "data.json"
        write_json(p, {"ok": True})
        assert p.exists()
        assert read_json(p) == {"ok": True}

    def test_atomic_write_no_partial_file(self, tmp_path: Path) -> None:
        """write_json uses tmp+rename — file must exist and be complete after call."""
        p = tmp_path / "atomic.json"
        obj = {"rows": list(range(100))}
        write_json(p, obj)
        result = read_json(p)
        assert len(result["rows"]) == 100

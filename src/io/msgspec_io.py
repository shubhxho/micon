"""
Faster JSON I/O using msgspec for the detail.json read/write hot path.

msgspec provides 5-10x faster JSON parsing than stdlib ``json`` because it
avoids object-graph construction overhead by using typed structs internally.
When used with ``dict`` decode it still beats stdlib by 3-5x on large files.

Lazy import strategy
--------------------
msgspec is imported lazily inside each function so that the module is usable
without msgspec installed (a one-line deprecation print is emitted and the
code falls back to stdlib ``json``).

Functions
---------
read_detail(path)        -- fast JSON parse of a single detail.json file
write_detail(path, data) -- atomic write (tmp + rename) of a detail.json file
read_jsonl(path)         -- iterator over lines of a .jsonl file
write_jsonl(path, rows)  -- write rows to a .jsonl file (overwrites)
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _msgspec_encoder() -> Any:
    """Return a msgspec JSON encoder, or None if msgspec is unavailable."""
    try:
        import msgspec.json as _mj  # noqa: PLC0415

        return _mj
    except ImportError:
        print(
            "DeprecationWarning: msgspec not installed; falling back to stdlib json. "
            "Install msgspec>=0.18 for faster I/O."
        )
        return None


def read_detail(path: Path) -> dict[str, Any]:
    """Parse a detail.json file and return its contents as a dict.

    Uses msgspec for 5-10x speedup over stdlib json on large files.
    Falls back to stdlib json if msgspec is not installed.

    Parameters
    ----------
    path:
        Path to the JSON file to read.

    Returns
    -------
    dict[str, Any]
        Parsed JSON contents.
    """
    path = Path(path)
    raw = path.read_bytes()

    mj = _msgspec_encoder()
    if mj is not None:
        return mj.decode(raw, type=dict)  # type: ignore[return-value]

    return json.loads(raw)


def write_detail(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as JSON to *path* atomically (via tmp file + rename).

    The atomic write ensures that a crash mid-write never leaves a partial
    file at the target path.

    Parameters
    ----------
    path:
        Destination file path.
    data:
        Dict to serialize as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mj = _msgspec_encoder()
    if mj is not None:
        payload = mj.encode(data)
    else:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")

    # Write to a temp file in the same directory, then rename (atomic on POSIX).
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)

    os.replace(tmp_path, path)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Iterate over records in a newline-delimited JSON file.

    Empty lines and lines consisting only of whitespace are skipped.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file.

    Yields
    ------
    dict[str, Any]
        One parsed record per non-empty line.
    """
    path = Path(path)
    mj = _msgspec_encoder()

    with path.open("rb") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if mj is not None:
                yield mj.decode(stripped, type=dict)  # type: ignore[misc]
            else:
                yield json.loads(stripped)


def write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> None:
    """Write *rows* to *path* as newline-delimited JSON (overwrites existing).

    Parameters
    ----------
    path:
        Destination ``.jsonl`` file path.
    rows:
        Iterable of dicts to serialize one-per-line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mj = _msgspec_encoder()

    with path.open("wb") as fh:
        for record in rows:
            if mj is not None:
                line = mj.encode(record) + b"\n"
            else:
                line = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
            fh.write(line)

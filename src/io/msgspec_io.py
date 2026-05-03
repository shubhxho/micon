"""Fast JSON I/O — msgspec → orjson → stdlib fallback chain.

Encoding hot paths (``detail.json``, manifests, JSONL streams) go through a
two-tier accelerator:

1. ``msgspec.json``  (typed, fastest decode)
2. ``orjson``        (fastest encode, best ``numpy``/``datetime`` support)
3. stdlib ``json``   (always-available last resort)

The encoders are resolved once at import and cached. Modules that previously
called ``json.loads(path.read_text())`` / ``json.dump(data, fh)`` should call
``read_detail`` / ``write_detail`` (or ``loads`` / ``dumps``) instead — the
public API mirrors stdlib ``json`` closely enough for a 1:1 swap.

Additionally, ``fast_dumps`` / ``fast_loads`` / ``write_json`` / ``read_json``
are provided as canonical hot-path aliases used by series.py, schema_utils.py,
and manifest/builder.py.
"""

from __future__ import annotations

import json as _stdlib_json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

__all__ = [
    "dumps",
    "dumps_bytes",
    "fast_dumps",
    "fast_loads",
    "loads",
    "read_detail",
    "read_json",
    "read_jsonl",
    "write_detail",
    "write_json",
    "write_jsonl",
]


try:
    import msgspec.json as _msgspec_json  # type: ignore[import-not-found]

    _MSGSPEC_DECODER = _msgspec_json.Decoder()
    _MSGSPEC_ENCODER = _msgspec_json.Encoder()
except ImportError:
    _msgspec_json = None
    _MSGSPEC_DECODER = None
    _MSGSPEC_ENCODER = None


try:
    import orjson as _orjson  # type: ignore[import-not-found]
except ImportError:
    _orjson = None


_ORJSON_OPTS = 0
if _orjson is not None:
    _ORJSON_OPTS = (
        _orjson.OPT_SERIALIZE_NUMPY | _orjson.OPT_NON_STR_KEYS | _orjson.OPT_SERIALIZE_DATACLASS
    )

# Options for the fast_dumps canonical hot-path (includes OPT_INDENT_2)
_FAST_DUMPS_OPTS = 0
if _orjson is not None:
    _FAST_DUMPS_OPTS = (
        _orjson.OPT_INDENT_2
        | _orjson.OPT_SERIALIZE_NUMPY
        | _orjson.OPT_NON_STR_KEYS
        | _orjson.OPT_SERIALIZE_DATACLASS
    )


def loads(data: str | bytes | bytearray | memoryview) -> Any:
    """Parse a JSON document — msgspec → orjson → stdlib."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    if _MSGSPEC_DECODER is not None:
        return _MSGSPEC_DECODER.decode(data)
    if _orjson is not None:
        return _orjson.loads(data)
    return _stdlib_json.loads(data)


def dumps_bytes(obj: Any, *, indent: int | None = None) -> bytes:
    """Encode *obj* to JSON bytes — orjson → msgspec → stdlib."""
    if _orjson is not None:
        opts = _ORJSON_OPTS
        if indent == 2:
            opts |= _orjson.OPT_INDENT_2
        return _orjson.dumps(obj, option=opts)
    if _MSGSPEC_ENCODER is not None and indent is None:
        return _MSGSPEC_ENCODER.encode(obj)
    return _stdlib_json.dumps(obj, ensure_ascii=False, indent=indent).encode("utf-8")


def dumps(obj: Any, *, indent: int | None = None) -> str:
    """Encode *obj* to a JSON string."""
    return dumps_bytes(obj, indent=indent).decode("utf-8")


# ---------------------------------------------------------------------------
# Canonical hot-path aliases (fast_dumps / fast_loads / write_json / read_json)
# ---------------------------------------------------------------------------


def fast_dumps(obj: Any) -> bytes:
    """Encode *obj* to indented JSON bytes using orjson with numpy/datetime support.

    Uses ``OPT_INDENT_2 | OPT_SERIALIZE_NUMPY`` for byte-compatible output with
    ``json.dumps(indent=2, default=str)``.  Falls back to stdlib ``json.dumps``
    for objects orjson cannot handle (e.g. custom non-serialisable types).
    """
    if _orjson is not None:
        try:
            return _orjson.dumps(obj, option=_FAST_DUMPS_OPTS)
        except TypeError:
            pass
    return _stdlib_json.dumps(obj, indent=2, default=str).encode("utf-8")


def fast_loads(data: bytes | str) -> Any:
    """Parse a JSON document using orjson (fastest) -> stdlib fallback."""
    if _orjson is not None:
        return _orjson.loads(data)
    return _stdlib_json.loads(data)


def read_json(path: Path | str) -> Any:
    """Read and parse a JSON file, returning the decoded object."""
    return fast_loads(Path(path).read_bytes())


def write_json(path: Path | str, obj: Any) -> None:
    """Atomically write *obj* as indented JSON to *path* (tmp + rename).

    Equivalent to ``write_detail(path, obj, indent=2)`` -- provided as a
    public alias for call sites that write non-detail JSON (e.g. manifests).
    """
    write_detail(path, obj, indent=2)


# ---------------------------------------------------------------------------
# Core I/O helpers
# ---------------------------------------------------------------------------


def read_detail(path: Path | str) -> dict[str, Any]:
    """Parse a JSON file (typically ``detail.json``) and return a dict."""
    return loads(Path(path).read_bytes())


def write_detail(
    path: Path | str,
    data: dict[str, Any],
    *,
    indent: int | None = None,
    fsync: bool = True,
) -> None:
    """Atomically write *data* as JSON to *path* (tmp + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dumps_bytes(data, indent=indent)

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, payload)
        if fsync:
            os.fsync(fd)
    finally:
        os.close(fd)

    os.replace(tmp_path, path)


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Iterate parsed records from a newline-delimited JSON file."""
    with Path(path).open("rb") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            yield loads(stripped)


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    """Write *rows* as newline-delimited JSON, overwriting *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for record in rows:
            fh.write(dumps_bytes(record))
            fh.write(b"\n")

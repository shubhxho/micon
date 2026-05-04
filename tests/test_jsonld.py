"""Tests for src/schemas/jsonld.py + the detail.json self-description wrapper.

Covers:
  1. with_jsonld() returns the three top-level keys at the *top* of the dict.
  2. Round-trip: write detail.json with with_jsonld -> read back -> keys present.
  3. SeriesDetail.model_validate() tolerates the new top-level keys (extra="allow").
  4. load_detail() accepts both JSON-LD-wrapped and legacy detail.json shapes.
  5. Idempotency: with_jsonld(with_jsonld(d)) == with_jsonld(d).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.constants import JSONLD_CONTEXT, JSONLD_DEFAULT_TYPE, SCHEMA_BASE_URL
from src.io.msgspec_io import read_json, write_detail
from src.schema_utils import load_detail
from src.schemas import SeriesDetail
from src.schemas.jsonld import schema_url, with_jsonld

_REPO = Path(__file__).parent.parent
_SAMPLE = _REPO / "Speall_MRI_Samples" / "series" / "s0005_Ax_DWI.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_schema_url() -> str:
    return f"{SCHEMA_BASE_URL}/SeriesDetail.schema.json"


# ---------------------------------------------------------------------------
# Pure with_jsonld() behaviour
# ---------------------------------------------------------------------------


def test_with_jsonld_adds_three_top_keys() -> None:
    """All three self-description keys are present after wrapping."""
    wrapped = with_jsonld({"series": {"uid": "abc"}})
    assert wrapped["$schema"] == _expected_schema_url()
    assert wrapped["@context"] == JSONLD_CONTEXT
    assert wrapped["@type"] == JSONLD_DEFAULT_TYPE
    # Original payload survives.
    assert wrapped["series"] == {"uid": "abc"}


def test_with_jsonld_keys_appear_first() -> None:
    """$schema must be the first byte-stream key — readers may scan only the head."""
    wrapped = with_jsonld({"series": {"uid": "abc"}, "files": []})
    keys = list(wrapped.keys())
    assert keys[0] == "$schema"
    assert keys[1] == "@context"
    assert keys[2] == "@type"


def test_with_jsonld_does_not_mutate_input() -> None:
    """Input dict must be returned intact (no in-place mutation)."""
    original = {"series": {"uid": "abc"}}
    snapshot = json.dumps(original)
    with_jsonld(original)
    assert json.dumps(original) == snapshot


def test_with_jsonld_is_idempotent() -> None:
    """Re-wrapping must not duplicate or shift keys (e.g. backfill re-runs)."""
    once = with_jsonld({"series": {"uid": "abc"}})
    twice = with_jsonld(once)
    assert once == twice
    assert list(once.keys()) == list(twice.keys())


def test_with_jsonld_preserves_existing_self_description() -> None:
    """If a caller already supplied $schema/@context/@type, keep their values."""
    custom = {
        "$schema": "https://example.com/custom.json",
        "@context": {"@vocab": "https://example.com/"},
        "@type": "Dataset",
        "series": {"uid": "abc"},
    }
    wrapped = with_jsonld(custom)
    assert wrapped["$schema"] == "https://example.com/custom.json"
    assert wrapped["@context"] == {"@vocab": "https://example.com/"}
    assert wrapped["@type"] == "Dataset"


def test_schema_url_default_and_custom() -> None:
    assert schema_url() == _expected_schema_url()
    assert schema_url("VolumeStats") == f"{SCHEMA_BASE_URL}/VolumeStats.schema.json"


# ---------------------------------------------------------------------------
# Round-trip through write_detail / read_json (the actual hot path)
# ---------------------------------------------------------------------------


def test_round_trip_jsonld_detail_json(tmp_path: Path) -> None:
    """write detail.json -> read back -> keys match + SeriesDetail validates."""
    payload = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    wrapped = with_jsonld(payload)

    out = tmp_path / "s0005_detail.json"
    write_detail(out, wrapped, indent=2)

    # Read it back via the canonical fast loader.
    reread = read_json(out)

    # The three self-description keys survived a write+read cycle.
    assert reread["$schema"] == _expected_schema_url()
    assert reread["@context"] == JSONLD_CONTEXT
    assert reread["@type"] == JSONLD_DEFAULT_TYPE
    # Domain payload survived intact (sample key check).
    assert reread["series"]["uid"] == payload["series"]["uid"]


def test_load_detail_tolerates_jsonld_wrapped_file(tmp_path: Path) -> None:
    """load_detail() must accept JSON-LD-wrapped detail.json files unchanged.

    SeriesDetail has extra="allow", so unknown top-level keys flow through
    Pydantic without error and do not corrupt the typed view.
    """
    payload = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    out = tmp_path / "s0005_detail.json"
    write_detail(out, with_jsonld(payload), indent=2)

    detail = load_detail(out)
    assert detail.series is not None
    assert detail.series.uid == payload["series"]["uid"]


def test_load_detail_still_handles_legacy_files(tmp_path: Path) -> None:
    """Legacy detail.json (no JSON-LD keys) must keep loading — backward compat."""
    payload = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    out = tmp_path / "legacy_detail.json"
    out.write_text(json.dumps(payload), encoding="utf-8")

    detail = load_detail(out)
    assert detail.series is not None
    assert detail.series.uid == payload["series"]["uid"]


def test_jsonld_keys_appear_at_top_of_serialised_bytes(tmp_path: Path) -> None:
    """The on-disk byte stream must have $schema as the first key.

    This is the discriminating check that a reader scanning only the head
    of the file (e.g. a CDN preview, JSON Schema validator) sees the
    schema reference immediately.
    """
    payload = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    out = tmp_path / "s0005_detail.json"
    write_detail(out, with_jsonld(payload), indent=2)

    head = out.read_bytes()[:200]
    # Strip the leading "{" + whitespace and look for the first quoted key.
    first_key_start = head.find(b'"')
    assert first_key_start != -1
    first_key_end = head.find(b'"', first_key_start + 1)
    first_key = head[first_key_start + 1 : first_key_end].decode("ascii")
    assert first_key == "$schema"


# ---------------------------------------------------------------------------
# Cross-check: SeriesDetail directly accepts the wrapped dict
# ---------------------------------------------------------------------------


def test_series_detail_validate_with_jsonld_keys() -> None:
    """SeriesDetail.model_validate() must accept the wrapped dict directly."""
    payload = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    wrapped = with_jsonld(payload)
    detail = SeriesDetail.model_validate(wrapped)
    assert detail.series is not None

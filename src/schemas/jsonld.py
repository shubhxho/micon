"""JSON-LD / JSON Schema injection helpers for per-series detail.json.

Every detail.json the pipeline writes is wrapped with three top-level keys so
the document is self-describing:

  - ``$schema``  → URL of the canonical JSON Schema for ``SeriesDetail``
  - ``@context`` → schema.org JSON-LD context
  - ``@type``    → schema.org class for the document

These keys are *additive*: ``SeriesDetail`` already declares
``ConfigDict(extra="allow")``, so existing readers parse new outputs unchanged
and old detail.json files (without the keys) still validate.
"""

from __future__ import annotations

from typing import Any

from src.constants import JSONLD_CONTEXT, JSONLD_DEFAULT_TYPE, SCHEMA_BASE_URL

__all__ = ["with_jsonld", "schema_url"]


def schema_url(schema_name: str = "SeriesDetail") -> str:
    """Return the canonical JSON Schema URL for *schema_name*.

    >>> schema_url("SeriesDetail")
    'https://raw.githubusercontent.com/shubhxho/micon/main/schemas/SeriesDetail.schema.json'
    """
    return f"{SCHEMA_BASE_URL}/{schema_name}.schema.json"


def with_jsonld(
    detail: dict[str, Any],
    *,
    schema_name: str = "SeriesDetail",
    type_: str = JSONLD_DEFAULT_TYPE,
) -> dict[str, Any]:
    """Return a new dict with ``$schema`` / ``@context`` / ``@type`` prepended.

    The original *detail* dict is **not** mutated.  Key insertion order matters:
    Python ``dict`` (3.7+) and orjson both preserve insertion order, so the
    self-description keys land at the *top* of the serialised file where a
    streaming reader sees them first.

    If any of the three keys already exist in *detail* (e.g. an idempotent
    re-run reading and re-writing the file), the existing values are kept.
    """
    out: dict[str, Any] = {
        "$schema": detail.get("$schema", schema_url(schema_name)),
        "@context": detail.get("@context", dict(JSONLD_CONTEXT)),
        "@type": detail.get("@type", type_),
    }
    # Append the rest, skipping the three keys we just inserted so we don't
    # duplicate them or shift their position.
    for k, v in detail.items():
        if k in out:
            continue
        out[k] = v
    return out

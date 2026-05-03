"""Structural validator for Croissant 1.0 JSON-LD metadata files.

Validates:
  - Top-level required fields: @context, @type=sc:Dataset, name,
    description, url, license, distribution, recordSet
  - Each distribution entry has @id, @type, name,
    and either contentUrl or a fileObject reference
  - Each recordSet has @id, @type=cr:RecordSet, name, and a field array

Accepts both bare names (``name``) and schema.org-namespaced names
(``sc:name``) per common Croissant usage.

CLI usage:
    python -m src.integrity.validate_croissant --path croissant.json
    Exit 0 = clean, Exit 1 = errors found.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.io.msgspec_io import loads

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CROISSANT = _REPO_ROOT / "croissant.json"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _get(obj: dict, *keys: str) -> Any:
    """Return first non-None value found among *keys* in *obj*."""
    for k in keys:
        if k in obj:
            return obj[k]
    return None


def _has(obj: dict, *keys: str) -> bool:
    return any(k in obj for k in keys)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_croissant(data: dict) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []

    # --- Top-level required fields ---
    if "@context" not in data:
        errors.append("Missing required field: @context")

    typ = _get(data, "@type")
    if typ not in ("sc:Dataset", "https://schema.org/Dataset"):
        errors.append(f"@type must be 'sc:Dataset', got {typ!r}")

    for field, ns_field in [
        ("name", "sc:name"),
        ("description", "sc:description"),
        ("url", "sc:url"),
        ("license", "sc:license"),
    ]:
        if not _has(data, field, ns_field):
            errors.append(f"Missing required field: '{field}' (or '{ns_field}')")

    # --- distribution ---
    distribution = _get(data, "distribution")
    if distribution is None:
        errors.append("Missing required field: 'distribution'")
    elif not isinstance(distribution, list) or len(distribution) == 0:
        errors.append("'distribution' must be a non-empty array")
    else:
        for i, entry in enumerate(distribution):
            prefix = f"distribution[{i}]"
            if not _has(entry, "@id"):
                errors.append(f"{prefix}: missing '@id'")
            if not _has(entry, "@type"):
                errors.append(f"{prefix}: missing '@type'")
            if not _has(entry, "name", "sc:name"):
                errors.append(f"{prefix}: missing 'name' or 'sc:name'")
            has_url = _has(
                entry,
                "contentUrl",
                "sc:contentUrl",
                "cr:FileObject",
                "cr:FileSet",
            )
            has_file_ref = _has(entry, "cr:source", "source")
            if not has_url and not has_file_ref:
                errors.append(
                    f"{prefix}: must have 'contentUrl'/'sc:contentUrl' or a file-object reference"
                )

    # --- recordSet ---
    record_set = _get(data, "recordSet")
    if record_set is None:
        errors.append("Missing required field: 'recordSet'")
    elif not isinstance(record_set, list) or len(record_set) == 0:
        errors.append("'recordSet' must be a non-empty array")
    else:
        for i, rs in enumerate(record_set):
            prefix = f"recordSet[{i}]"
            if not _has(rs, "@id"):
                errors.append(f"{prefix}: missing '@id'")
            typ_rs = _get(rs, "@type")
            if typ_rs != "cr:RecordSet":
                errors.append(f"{prefix}: @type must be 'cr:RecordSet', got {typ_rs!r}")
            if not _has(rs, "name", "sc:name"):
                errors.append(f"{prefix}: missing 'name' or 'sc:name'")
            fields = _get(rs, "field", "cr:field")
            if not isinstance(fields, list) or len(fields) == 0:
                errors.append(f"{prefix}: 'field'/'cr:field' must be a non-empty array")

    return errors


def validate_croissant_file(path: Path) -> list[str]:
    """Load *path* and return validation errors."""
    path = Path(path)
    try:
        data = loads(path.read_bytes())
    except Exception as exc:
        return [f"Failed to parse JSON: {exc}"]
    return validate_croissant(data)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _cli_validate(path: Path = _DEFAULT_CROISSANT) -> None:  # pragma: no cover
    """Validate a Croissant 1.0 JSON-LD file (default: repo croissant.json)."""
    errors = validate_croissant_file(path)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {path} is valid Croissant 1.0")


def _main() -> None:  # pragma: no cover
    from cyclopts import App

    cli = App(name="validate-croissant", help="Validate Croissant 1.0 JSON-LD.")
    cli.default(_cli_validate)
    cli()


if __name__ == "__main__":  # pragma: no cover
    _main()

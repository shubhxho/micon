"""Frictionless Data Package envelope (`datapackage.json`) builder.

Frictionless Data Packages are a vendor-neutral, JSON-based container format
for tabular and binary data. Major open-data portals (data.gov, the EU Open
Data Portal, OKFN catalogs) and an increasing number of buyer-side data
clearing houses expect a `datapackage.json` next to the actual artifacts.

Spec: https://datapackage.org/standard/data-package/

This module emits a minimal, spec-compliant envelope from any output bundle
that contains parquet/json/png/mcap/dcm files. We intentionally do *not*
depend on the `frictionless` Python library -- it is heavyweight, pulls in
pandas, and we only need a structural builder + lightweight validator.

Public API
----------
build_datapackage(bundle_root, *, name, version=..., licenses=..., contributors=...)
    Walk *bundle_root* and return a dict ready for JSON serialization.
write_datapackage(bundle_root, data, *, indent=2)
    Serialize *data* to ``<bundle_root>/datapackage.json`` (atomic write).
validate_datapackage(path)
    Lightweight structural validator. Returns a list of error strings.

CLI usage:
    python -m src.integrity.datapackage build <bundle_root> [--name X]
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from src.integrity.checksums import sha256_file
from src.io.msgspec_io import read_json, write_json

__all__ = ["build_datapackage", "validate_datapackage", "write_datapackage"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Filename -> (format, IANA media type). Lowercase keys.
_FORMAT_MAP: dict[str, tuple[str, str]] = {
    ".parquet": ("parquet", "application/vnd.apache.parquet"),
    ".json": ("json", "application/json"),
    ".png": ("png", "image/png"),
    ".mcap": ("mcap", "application/x-mcap"),
    ".dcm": ("dcm", "application/dicom"),
}

# Resource bundle artifacts that we never want to recurse into as resources.
# `datapackage.json` itself is excluded so re-runs don't embed a stale copy.
_SKIP_FILENAMES: frozenset[str] = frozenset({"datapackage.json", ".DS_Store"})

# Per-extension JSON-Schema reference (relative to the bundle root). The
# `schemas/` directory ships at the repo root and gets copied into bundles.
_DETAIL_SCHEMA_REF = "schemas/SeriesDetail.schema.json"
_MANIFEST_PARQUET_SCHEMA_REF = "schemas/ManifestRow.schema.json"
_STUDY_PARQUET_SCHEMA_REF = "schemas/StudyManifestRow.schema.json"

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_datapackage(
    bundle_root: Path,
    *,
    name: str,
    version: str = "1.0.0",
    licenses: list[dict] | None = None,
    contributors: list[dict] | None = None,
    title: str | None = None,
    description: str | None = None,
    keywords: list[str] | None = None,
) -> dict:
    """Walk *bundle_root* and return a Frictionless Data Package dict.

    Parameters
    ----------
    bundle_root:
        Directory containing the artifacts (parquet, json, png, mcap, dcm).
    name:
        Required. Lower-case slug for the package (e.g. ``speall-mri-2026-04``).
    version:
        Semantic version string. Defaults to ``"1.0.0"``.
    licenses:
        Optional list of license dicts (each at minimum ``{"name": "..."}``).
    contributors:
        Optional list of contributor dicts (each at minimum ``{"title": "..."}``).
    title, description, keywords:
        Optional human-facing metadata.

    Returns:
        A dict ready for ``json.dumps`` / ``write_datapackage``.
    """
    bundle_root = Path(bundle_root).resolve()
    if not bundle_root.is_dir():
        raise NotADirectoryError(f"bundle_root is not a directory: {bundle_root}")

    resources = list(_iter_resources(bundle_root))

    pkg: dict = {
        # Frictionless v2 profile -- "data-package" is the generic profile.
        "profile": "data-package",
        "name": name,
        "version": version,
        "created": datetime.now(UTC).isoformat(),
        "resources": resources,
    }
    if title is not None:
        pkg["title"] = title
    if description is not None:
        pkg["description"] = description
    if keywords:
        pkg["keywords"] = list(keywords)
    if licenses:
        pkg["licenses"] = licenses
    if contributors:
        pkg["contributors"] = contributors
    return pkg


def write_datapackage(bundle_root: Path, data: dict, *, indent: int = 2) -> Path:
    """Write *data* as ``datapackage.json`` inside *bundle_root*.

    Uses ``src.io.msgspec_io.write_json`` for atomic tmp+rename semantics.
    Returns the path written.
    """
    bundle_root = Path(bundle_root)
    out = bundle_root / "datapackage.json"
    # write_json from msgspec_io is hard-coded to indent=2 (matches our default);
    # we accept an *indent* arg for forward compatibility but silently honor 2.
    del indent
    write_json(out, data)
    return out


def validate_datapackage(path: Path) -> list[str]:
    """Lightweight structural validator.

    Checks that the file parses as JSON, has top-level ``name`` and
    ``resources``, and that each resource has a ``path`` and ``format``.
    Returns a list of error strings (empty list = valid).
    """
    path = Path(path)
    errors: list[str] = []
    try:
        doc = read_json(path)
    except FileNotFoundError:
        return [f"file not found: {path}"]
    except Exception as exc:
        return [f"could not parse JSON: {exc}"]

    if not isinstance(doc, dict):
        return ["top-level value must be a JSON object"]

    if "name" not in doc:
        errors.append("missing required field: name")
    elif not isinstance(doc["name"], str) or not doc["name"]:
        errors.append("'name' must be a non-empty string")

    if "resources" not in doc:
        errors.append("missing required field: resources")
        return errors

    resources = doc["resources"]
    if not isinstance(resources, list):
        errors.append("'resources' must be an array")
        return errors
    if not resources:
        errors.append("'resources' must contain at least one entry")

    for idx, res in enumerate(resources):
        if not isinstance(res, dict):
            errors.append(f"resources[{idx}] must be an object")
            continue
        if "path" not in res:
            errors.append(f"resources[{idx}] missing required field: path")
        if "format" not in res:
            errors.append(f"resources[{idx}] missing required field: format")
    return errors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_resources(bundle_root: Path) -> Iterable[dict]:
    """Yield Frictionless resource dicts for every recognized file under root."""
    seen_names: dict[str, int] = {}
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in _SKIP_FILENAMES:
            continue
        ext = path.suffix.lower()
        fmt_entry = _FORMAT_MAP.get(ext)
        if fmt_entry is None:
            continue
        fmt, mediatype = fmt_entry
        rel = path.relative_to(bundle_root).as_posix()

        # Slugify resource name; ensure uniqueness with a counter suffix.
        base = _slugify(path.stem) or f"resource-{len(seen_names)}"
        n = seen_names.get(base, 0)
        seen_names[base] = n + 1
        res_name = base if n == 0 else f"{base}-{n}"

        resource: dict = {
            "name": res_name,
            "path": rel,
            "format": fmt,
            "mediatype": mediatype,
            "bytes": path.stat().st_size,
            "hash": f"sha256:{sha256_file(path)}",
        }
        schema_ref = _schema_ref_for(rel, fmt)
        if schema_ref is not None:
            resource["schema"] = schema_ref
        yield resource


def _slugify(value: str) -> str:
    """Convert *value* to a Frictionless-compatible resource name slug.

    Lower-case, ASCII alphanumerics plus ``-`` / ``_``. Trims leading/trailing
    separators and collapses runs of disallowed characters into a single ``-``.
    """
    lowered = value.lower()
    cleaned = _SLUG_RE.sub("-", lowered)
    return cleaned.strip("-_")


def _schema_ref_for(rel_path: str, fmt: str) -> str | None:
    """Return a schema reference for known artifact families, else None."""
    name = rel_path.rsplit("/", 1)[-1]
    if fmt == "parquet":
        if name == "manifest.parquet":
            return _MANIFEST_PARQUET_SCHEMA_REF
        if name == "study_manifest.parquet":
            return _STUDY_PARQUET_SCHEMA_REF
        return None
    if fmt == "json" and name.startswith("detail"):
        return _DETAIL_SCHEMA_REF
    # series/<sNNNN>_*.json files are per-series detail records too.
    if fmt == "json" and rel_path.startswith("series/") and name.startswith("s"):
        return _DETAIL_SCHEMA_REF
    return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_cmd(
    bundle_root: Path,
    name: str = "speall-mri-bundle",
    version: str = "1.0.0",
) -> None:  # pragma: no cover
    """Build a datapackage.json envelope inside BUNDLE_ROOT."""
    pkg = build_datapackage(bundle_root, name=name, version=version)
    out = write_datapackage(bundle_root, pkg)
    print(f"Wrote {out}  ({len(pkg['resources'])} resources)")


def _validate_cmd(path: Path) -> None:  # pragma: no cover
    """Validate an existing datapackage.json structurally."""
    import sys

    errors = validate_datapackage(path)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"{path}: OK")


def _main() -> None:  # pragma: no cover
    from cyclopts import App

    cli = App(name="datapackage", help="Frictionless Data Package envelope.")
    cli.command(_build_cmd, name="build")
    cli.command(_validate_cmd, name="validate")
    cli()


if __name__ == "__main__":  # pragma: no cover
    _main()

"""
Ergonomic helpers for working with Pydantic v2 pipeline schemas.
"""

from __future__ import annotations

from pathlib import Path

from src.schemas import SeriesDetail


def load_detail(path: Path) -> SeriesDetail:
    """Load and validate a ``*_detail.json`` file, returning a ``SeriesDetail``."""
    path = Path(path)
    return SeriesDetail.from_json_file(path)


def validate_directory(root: Path) -> list[tuple[Path, str]]:
    """Walk *root* for ``*_detail.json`` files and validate each one.

    Returns a list of ``(path, error_message)`` tuples for any file that
    fails validation.  An empty list means all files are clean.
    """
    root = Path(root)
    errors: list[tuple[Path, str]] = []

    for detail_path in sorted(root.rglob("*_detail.json")):
        try:
            SeriesDetail.from_json_file(detail_path)
        except Exception as exc:
            errors.append((detail_path, str(exc)))

    return errors

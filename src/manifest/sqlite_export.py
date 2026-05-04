"""Export manifest parquet files to a Datasette-compatible SQLite database.

Purpose
-------
The pipeline writes ``manifest.parquet`` and ``study_manifest.parquet`` for
buyers, but a static parquet is hard to explore interactively. This module
turns those parquet files into a single ``manifest.db`` SQLite database that
can be served with `Datasette <https://datasette.io>`_::

    pipx install datasette
    uv run datasette serve manifest.db --metadata metadata.json

Design notes
------------
* Read-only consumer of ``src.manifest.builder`` outputs -- never modifies them.
* Uses ``polars`` (already a dep) to read parquet and the stdlib ``sqlite3``
  module to write -- no new third-party deps required. ``polars.write_database``
  is intentionally avoided because it pulls in SQLAlchemy.
* List/struct columns are serialized to JSON strings; Datasette renders JSON
  cells nicely out of the box.
* Adds a foreign key from ``manifest.study_id`` to ``study_manifest.study_id``
  when both tables are present so Datasette renders cross-table links.
* Adds indexes on a small set of high-cardinality query columns.
* Optionally creates a ``series_overview`` view joining the two tables for a
  one-shot explore experience.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import polars as pl

# Columns we want indexes on, when present. Picked for typical buyer queries:
# filter by study, locate a series, narrow by sequence/modality/grade.
_INDEX_CANDIDATES: tuple[str, ...] = (
    "study_id",
    "series_uid",
    "sequence_type",
    "modality",
    "quality_grade",
)

# Polars dtype -> SQLite affinity. Anything else falls through to TEXT (with
# JSON-encoded values for list/struct cells).
_TYPE_MAP: dict[type, str] = {
    pl.Boolean: "INTEGER",
    pl.Int8: "INTEGER",
    pl.Int16: "INTEGER",
    pl.Int32: "INTEGER",
    pl.Int64: "INTEGER",
    pl.UInt8: "INTEGER",
    pl.UInt16: "INTEGER",
    pl.UInt32: "INTEGER",
    pl.UInt64: "INTEGER",
    pl.Float32: "REAL",
    pl.Float64: "REAL",
    pl.Utf8: "TEXT",
}


def _sqlite_type(dtype: pl.DataType) -> str:
    """Map a polars dtype to a SQLite column affinity."""
    return _TYPE_MAP.get(type(dtype), "TEXT")


def _is_complex(dtype: pl.DataType) -> bool:
    """Return True for columns we must JSON-encode before insertion."""
    return isinstance(dtype, (pl.List, pl.Array, pl.Struct))


def _quote_ident(name: str) -> str:
    """Quote a SQLite identifier safely."""
    return '"' + name.replace('"', '""') + '"'


def _column_defs(
    df: pl.DataFrame,
    *,
    fk_target: tuple[str, str] | None = None,
    primary_key: str | None = None,
) -> str:
    """Build the column-definition fragment for a CREATE TABLE statement.

    ``fk_target`` -- when supplied as ``(column, "table(col)")`` -- adds a
    FOREIGN KEY clause referencing the target table.
    ``primary_key`` -- when supplied, marks that column as ``PRIMARY KEY``
    inline. Required for the FK target column so SQLite accepts the FK.
    """
    parts: list[str] = []
    for name, dtype in df.schema.items():
        col_def = f"{_quote_ident(name)} {_sqlite_type(dtype)}"
        if primary_key is not None and name == primary_key:
            col_def += " PRIMARY KEY"
        parts.append(col_def)
    if fk_target is not None:
        col, ref = fk_target
        if col in df.schema:
            parts.append(f"FOREIGN KEY ({_quote_ident(col)}) REFERENCES {ref}")
    return ", ".join(parts)


def _row_to_sqlite(row: tuple[Any, ...], complex_idx: set[int]) -> tuple[Any, ...]:
    """Convert a polars row into a SQLite-friendly tuple.

    Booleans -> 0/1, list/struct -> JSON string, everything else passes
    through. ``None`` stays ``None``.
    """
    out: list[Any] = []
    for i, val in enumerate(row):
        if val is None:
            out.append(None)
        elif i in complex_idx:
            out.append(json.dumps(val, default=str))
        elif isinstance(val, bool):
            out.append(1 if val else 0)
        else:
            out.append(val)
    return tuple(out)


def _write_table(
    conn: sqlite3.Connection,
    table: str,
    df: pl.DataFrame,
    *,
    fk_target: tuple[str, str] | None = None,
    primary_key: str | None = None,
) -> None:
    """Create ``table`` from ``df`` and insert all rows."""
    qtable = _quote_ident(table)
    conn.execute(f"DROP TABLE IF EXISTS {qtable}")
    conn.execute(
        f"CREATE TABLE {qtable} ({_column_defs(df, fk_target=fk_target, primary_key=primary_key)})"
    )

    cols = list(df.schema.keys())
    complex_idx = {i for i, name in enumerate(cols) if _is_complex(df.schema[name])}
    placeholders = ", ".join("?" for _ in cols)
    qcols = ", ".join(_quote_ident(c) for c in cols)
    insert_sql = f"INSERT INTO {qtable} ({qcols}) VALUES ({placeholders})"

    rows = (_row_to_sqlite(r, complex_idx) for r in df.iter_rows())
    conn.executemany(insert_sql, rows)


def _create_indexes(conn: sqlite3.Connection, table: str, columns: list[str]) -> None:
    """Create one index per ``_INDEX_CANDIDATES`` column that exists in ``table``."""
    qtable = _quote_ident(table)
    for col in _INDEX_CANDIDATES:
        if col not in columns:
            continue
        idx_name = _quote_ident(f"idx_{table}_{col}")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {qtable} ({_quote_ident(col)})")


def _maybe_create_overview_view(conn: sqlite3.Connection, tables: dict[str, list[str]]) -> None:
    """Create ``series_overview`` joining series + study when both exist.

    The series table is detected as ``manifest`` and the study table as
    ``study_manifest`` (matching the parquet stems produced by the builder).
    Falls back gracefully if either is missing.
    """
    series_t = "manifest"
    study_t = "study_manifest"
    if series_t not in tables or study_t not in tables:
        return
    if "study_id" not in tables[series_t] or "study_id" not in tables[study_t]:
        return

    # Pick a small, useful column set if available; otherwise SELECT *.
    series_cols = [
        c
        for c in (
            "series_uid",
            "study_id",
            "series_description",
            "sequence_type",
            "modality",
            "quality_grade",
            "ml_score",
        )
        if c in tables[series_t]
    ]
    study_cols = [
        c
        for c in ("n_series", "dominant_grade", "mean_ml_score", "total_size_mb")
        if c in tables[study_t]
    ]
    if not series_cols:
        series_cols = tables[series_t]

    select_parts = [f"s.{_quote_ident(c)} AS {_quote_ident(c)}" for c in series_cols]
    select_parts += [f"st.{_quote_ident(c)} AS {_quote_ident('study_' + c)}" for c in study_cols]
    select_clause = ", ".join(select_parts)

    conn.execute("DROP VIEW IF EXISTS series_overview")
    conn.execute(
        f"CREATE VIEW series_overview AS "
        f"SELECT {select_clause} "
        f"FROM {_quote_ident(series_t)} s "
        f"LEFT JOIN {_quote_ident(study_t)} st ON s.study_id = st.study_id"
    )


def manifest_to_sqlite(
    parquet_paths: list[Path],
    out_db: Path,
    *,
    with_views: bool = True,
) -> Path:
    """Convert one or more manifest parquet files into a SQLite database.

    Parameters
    ----------
    parquet_paths:
        Parquet files to ingest. Each becomes one table named after the file
        stem (``manifest.parquet`` -> table ``manifest``).
    out_db:
        Destination SQLite database file. Overwritten if it exists.
    with_views:
        When True, create a ``series_overview`` view joining ``manifest`` and
        ``study_manifest`` if both are present.

    Returns
    -------
    Path
        Absolute path to the written ``out_db``.
    """
    out_db = Path(out_db)
    out_db.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        out_db.unlink()

    # Load all dataframes up-front so we know which tables will exist (needed
    # for the FK clause on the series table).
    dfs: dict[str, pl.DataFrame] = {}
    for p in parquet_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(p)
        dfs[p.stem] = pl.read_parquet(p)

    has_study = "study_manifest" in dfs

    with sqlite3.connect(out_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        # Write study_manifest first so the FK on manifest can reference it.
        # Mark study_id as PRIMARY KEY so SQLite accepts the FK reference.
        if has_study:
            study_pk = "study_id" if "study_id" in dfs["study_manifest"].schema else None
            _write_table(conn, "study_manifest", dfs["study_manifest"], primary_key=study_pk)

        for stem, df in dfs.items():
            if stem == "study_manifest":
                continue
            fk = None
            if has_study and stem == "manifest" and "study_id" in df.schema:
                fk = ("study_id", f"{_quote_ident('study_manifest')}(study_id)")
            _write_table(conn, stem, df, fk_target=fk)

        # Indexes
        tables: dict[str, list[str]] = {stem: list(df.schema.keys()) for stem, df in dfs.items()}
        for stem, cols in tables.items():
            _create_indexes(conn, stem, cols)

        # Optional overview view
        if with_views:
            _maybe_create_overview_view(conn, tables)

        conn.commit()

    return out_db.resolve()


def write_datasette_metadata(out_db: Path) -> Path:
    """Write a Datasette ``metadata.json`` next to ``out_db``.

    Datasette picks this up automatically when invoked with
    ``--metadata <path>``. The values here describe the Speall MRI dataset for
    prospective buyers browsing the explorer.
    """
    out_db = Path(out_db)
    meta_path = out_db.parent / "metadata.json"
    db_key = out_db.stem

    metadata: dict[str, Any] = {
        "title": "Speall MRI Dataset Explorer",
        "description": (
            "Interactive explorer for the Speall MRI dataset manifest. "
            "Browse series-level and study-level metadata, run ad-hoc SQL, "
            "and facet by sequence type, modality, and quality grade."
        ),
        "license": "See LICENSE file in source repository",
        "license_url": "https://github.com/shubhxho/speall-mri",
        "source": "Speall MRI pipeline (manifest.parquet)",
        "source_url": "https://github.com/shubhxho/speall-mri",
        "databases": {
            db_key: {
                "tables": {
                    "manifest": {
                        "title": "Series-level manifest",
                        "description": "One row per imaging series.",
                        "facets": ["sequence_type", "modality", "quality_grade", "plane"],
                        "sort_desc": "ml_score",
                    },
                    "study_manifest": {
                        "title": "Study-level manifest",
                        "description": "One row per study (aggregated from series).",
                        "facets": ["dominant_grade"],
                        "sort_desc": "mean_ml_score",
                    },
                },
            }
        },
    }

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return meta_path.resolve()

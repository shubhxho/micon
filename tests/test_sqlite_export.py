"""Tests for src.manifest.sqlite_export."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from src.manifest.sqlite_export import (
    manifest_to_sqlite,
    write_datasette_metadata,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manifest_parquet(tmp_path: Path) -> Path:
    """A tiny series-level manifest parquet with mixed dtypes (incl. lists)."""
    df = pl.DataFrame(
        {
            "study_id": ["s001", "s001", "s002"],
            "series_uid": ["uid-a", "uid-b", "uid-c"],
            "series_description": ["Ax DWI", "Sag T1", "Cor FLAIR"],
            "sequence_type": ["DWI", "T1", "FLAIR"],
            "modality": ["MR", "MR", "MR"],
            "quality_grade": ["A", "B", "A"],
            "ml_score": [0.91, 0.74, 0.88],
            "volume_shape": [[64, 256, 256], [80, 320, 320], [70, 256, 256]],
            "spacing_mm": [[5.0, 0.9, 0.9], [2.0, 0.7, 0.7], [4.0, 0.9, 0.9]],
            "has_tar_shard": [True, True, False],
        }
    )
    p = tmp_path / "manifest.parquet"
    df.write_parquet(p)
    return p


@pytest.fixture()
def study_parquet(tmp_path: Path) -> Path:
    """A tiny study-level manifest parquet."""
    df = pl.DataFrame(
        {
            "study_id": ["s001", "s002"],
            "n_series": [2, 1],
            "sequences_present": [["DWI", "T1"], ["FLAIR"]],
            "dominant_grade": ["A", "A"],
            "mean_ml_score": [0.825, 0.88],
            "total_size_mb": [123.4, 56.7],
        }
    )
    p = tmp_path / "study_manifest.parquet"
    df.write_parquet(p)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_export_creates_tables(manifest_parquet: Path, tmp_path: Path) -> None:
    """Exporting a single parquet creates one table named after the file stem."""
    db = tmp_path / "out.db"
    out = manifest_to_sqlite([manifest_parquet], db)

    assert out.exists()
    with sqlite3.connect(out) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "manifest" in names


def test_row_counts_match_parquet(manifest_parquet: Path, tmp_path: Path) -> None:
    db = tmp_path / "out.db"
    manifest_to_sqlite([manifest_parquet], db)

    with sqlite3.connect(db) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM manifest").fetchone()
    expected = pl.read_parquet(manifest_parquet).height
    assert count == expected == 3


def test_indexes_created_on_high_cardinality_cols(manifest_parquet: Path, tmp_path: Path) -> None:
    db = tmp_path / "out.db"
    manifest_to_sqlite([manifest_parquet], db)

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='manifest'"
        ).fetchall()
    idx_names = {r[0] for r in rows}

    # We expect indexes on the columns from _INDEX_CANDIDATES that are
    # actually present in the test fixture.
    for col in ("study_id", "series_uid", "sequence_type", "modality", "quality_grade"):
        assert f"idx_manifest_{col}" in idx_names, f"missing index for {col}: {idx_names}"


def test_list_columns_serialized_as_json(manifest_parquet: Path, tmp_path: Path) -> None:
    db = tmp_path / "out.db"
    manifest_to_sqlite([manifest_parquet], db)

    with sqlite3.connect(db) as conn:
        (raw,) = conn.execute(
            "SELECT volume_shape FROM manifest WHERE series_uid='uid-a'"
        ).fetchone()
    parsed = json.loads(raw)
    assert parsed == [64, 256, 256]


def test_booleans_stored_as_integers(manifest_parquet: Path, tmp_path: Path) -> None:
    db = tmp_path / "out.db"
    manifest_to_sqlite([manifest_parquet], db)

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT series_uid, has_tar_shard FROM manifest ORDER BY series_uid"
        ).fetchall()
    assert rows == [("uid-a", 1), ("uid-b", 1), ("uid-c", 0)]


def test_two_table_export_with_view_and_fk(
    manifest_parquet: Path, study_parquet: Path, tmp_path: Path
) -> None:
    db = tmp_path / "out.db"
    manifest_to_sqlite([manifest_parquet, study_parquet], db, with_views=True)

    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        assert {"manifest", "study_manifest", "series_overview"} <= names

        # Foreign key declared on manifest -> study_manifest
        fks = conn.execute("PRAGMA foreign_key_list('manifest')").fetchall()
        assert any(fk[2] == "study_manifest" and fk[3] == "study_id" for fk in fks), fks

        # The view actually returns joined rows.
        rows = conn.execute(
            "SELECT series_uid, study_n_series, study_dominant_grade "
            "FROM series_overview ORDER BY series_uid"
        ).fetchall()
        assert len(rows) == 3
        # uid-a is in study s001 -> n_series=2, dominant_grade=A
        uid_a = next(r for r in rows if r[0] == "uid-a")
        assert uid_a[1] == 2
        assert uid_a[2] == "A"


def test_with_views_false_skips_view(
    manifest_parquet: Path, study_parquet: Path, tmp_path: Path
) -> None:
    db = tmp_path / "out.db"
    manifest_to_sqlite([manifest_parquet, study_parquet], db, with_views=False)

    with sqlite3.connect(db) as conn:
        views = conn.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()
    assert views == []


def test_overwrites_existing_db(manifest_parquet: Path, tmp_path: Path) -> None:
    db = tmp_path / "out.db"
    db.write_bytes(b"not a real sqlite file")
    out = manifest_to_sqlite([manifest_parquet], db)
    # Should be a valid SQLite file now.
    with sqlite3.connect(out) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM manifest").fetchone()
    assert count == 3


def test_missing_parquet_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        manifest_to_sqlite([tmp_path / "nope.parquet"], tmp_path / "out.db")


def test_metadata_json_well_formed(manifest_parquet: Path, tmp_path: Path) -> None:
    db = tmp_path / "out.db"
    manifest_to_sqlite([manifest_parquet], db)

    meta_path = write_datasette_metadata(db)
    assert meta_path.exists()
    assert meta_path.name == "metadata.json"

    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["title"]
    assert data["description"]
    assert "out" in data["databases"]  # keyed by db stem
    assert "manifest" in data["databases"]["out"]["tables"]
    facets = data["databases"]["out"]["tables"]["manifest"]["facets"]
    assert "sequence_type" in facets

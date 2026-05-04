"""Tests for src.integrity.datapackage (Frictionless envelope)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from src.integrity.datapackage import (
    build_datapackage,
    validate_datapackage,
    write_datapackage,
)
from src.io.msgspec_io import read_json, write_json

# ---------------------------------------------------------------------------
# Fixture: a tiny fake bundle
# ---------------------------------------------------------------------------


def _write_minimal_png(path: Path) -> None:
    """Write a 1x1 transparent PNG (smallest valid PNG)."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    raw = b"\x00\x00\x00\x00\x00"  # filter byte + RGBA
    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    path.write_bytes(sig + ihdr + idat + iend)


@pytest.fixture
def fake_bundle(tmp_path: Path) -> Path:
    """Build a tiny bundle with parquet + 2 json + 1 png."""
    # Plain bytes is fine for the parquet -- format detection is by extension.
    (tmp_path / "manifest.parquet").write_bytes(b"PAR1\x00\x00\x00\x00")
    (tmp_path / "study_manifest.parquet").write_bytes(b"PAR1\x00\x00\x00\x00")

    series_dir = tmp_path / "series"
    series_dir.mkdir()
    write_json(series_dir / "s0005_Ax_DWI.json", {"series_id": "s0005"})
    write_json(tmp_path / "study_summary.json", {"summary": "ok"})

    _write_minimal_png(tmp_path / "montage.png")

    # noise files we expect to skip
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "README.txt").write_text("ignored extension")
    return tmp_path


# ---------------------------------------------------------------------------
# build_datapackage
# ---------------------------------------------------------------------------


def test_build_minimum_required_fields(fake_bundle: Path) -> None:
    pkg = build_datapackage(fake_bundle, name="speall-test")
    assert pkg["name"] == "speall-test"
    assert pkg["version"] == "1.0.0"
    assert "created" in pkg
    assert "resources" in pkg and isinstance(pkg["resources"], list)


def test_build_resource_count_and_skip_rules(fake_bundle: Path) -> None:
    pkg = build_datapackage(fake_bundle, name="speall-test")
    paths = sorted(r["path"] for r in pkg["resources"])
    # Expect: 2 parquet + 2 json + 1 png = 5 (README.txt, .DS_Store excluded).
    assert len(paths) == 5
    assert "manifest.parquet" in paths
    assert "study_manifest.parquet" in paths
    assert "montage.png" in paths
    assert "series/s0005_Ax_DWI.json" in paths
    assert "study_summary.json" in paths
    assert not any(p.endswith(".DS_Store") for p in paths)
    assert not any(p.endswith("README.txt") for p in paths)


def test_build_hashes_have_sha256_prefix(fake_bundle: Path) -> None:
    pkg = build_datapackage(fake_bundle, name="speall-test")
    for r in pkg["resources"]:
        assert r["hash"].startswith("sha256:")
        # 7-char prefix + 64 hex digits
        assert len(r["hash"]) == len("sha256:") + 64
        assert isinstance(r["bytes"], int) and r["bytes"] > 0


def test_build_format_and_mediatype(fake_bundle: Path) -> None:
    pkg = build_datapackage(fake_bundle, name="speall-test")
    by_path = {r["path"]: r for r in pkg["resources"]}
    assert by_path["manifest.parquet"]["format"] == "parquet"
    assert by_path["manifest.parquet"]["mediatype"] == "application/vnd.apache.parquet"
    assert by_path["montage.png"]["format"] == "png"
    assert by_path["montage.png"]["mediatype"] == "image/png"
    assert by_path["study_summary.json"]["format"] == "json"


def test_build_attaches_schema_for_known_artifacts(fake_bundle: Path) -> None:
    pkg = build_datapackage(fake_bundle, name="speall-test")
    by_path = {r["path"]: r for r in pkg["resources"]}
    assert by_path["manifest.parquet"]["schema"] == "schemas/ManifestRow.schema.json"
    assert by_path["study_manifest.parquet"]["schema"] == "schemas/StudyManifestRow.schema.json"
    assert by_path["series/s0005_Ax_DWI.json"]["schema"] == "schemas/SeriesDetail.schema.json"
    # study_summary.json is not a per-series detail -> no schema attached.
    assert "schema" not in by_path["study_summary.json"]
    assert "schema" not in by_path["montage.png"]


def test_build_resource_names_are_slugified(fake_bundle: Path) -> None:
    # Inject a filename with characters that violate Frictionless slug rules.
    weird = fake_bundle / "series" / "s0550_ADC_10_-6_mm²_s.json"
    write_json(weird, {"weird": True})
    pkg = build_datapackage(fake_bundle, name="speall-test")
    # All names must be lower + [a-z0-9_-]+ only.
    import re

    name_re = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
    for r in pkg["resources"]:
        assert name_re.match(r["name"]), f"bad slug: {r['name']}"
    names = [r["name"] for r in pkg["resources"]]
    assert len(names) == len(set(names)), "resource names must be unique"


def test_build_optional_fields_propagate(fake_bundle: Path) -> None:
    pkg = build_datapackage(
        fake_bundle,
        name="speall-test",
        title="Speall MRI Test",
        description="A test bundle",
        keywords=["mri", "brain"],
        licenses=[{"name": "CC-BY-4.0"}],
        contributors=[{"title": "Speall"}],
    )
    assert pkg["title"] == "Speall MRI Test"
    assert pkg["description"] == "A test bundle"
    assert pkg["keywords"] == ["mri", "brain"]
    assert pkg["licenses"] == [{"name": "CC-BY-4.0"}]
    assert pkg["contributors"] == [{"title": "Speall"}]


def test_build_skips_existing_datapackage_json(fake_bundle: Path) -> None:
    # Pre-create one to simulate a re-run; it must not appear as a resource.
    write_json(fake_bundle / "datapackage.json", {"stale": True})
    pkg = build_datapackage(fake_bundle, name="speall-test")
    assert not any(r["path"] == "datapackage.json" for r in pkg["resources"])


def test_build_rejects_non_directory(tmp_path: Path) -> None:
    f = tmp_path / "nope.txt"
    f.write_text("hi")
    with pytest.raises(NotADirectoryError):
        build_datapackage(f, name="speall-test")


# ---------------------------------------------------------------------------
# write_datapackage round-trip
# ---------------------------------------------------------------------------


def test_round_trip_via_msgspec_io(fake_bundle: Path) -> None:
    pkg = build_datapackage(fake_bundle, name="speall-test")
    out = write_datapackage(fake_bundle, pkg)
    assert out == fake_bundle / "datapackage.json"
    assert out.exists()
    reread = read_json(out)
    for key in ("name", "version", "created", "resources"):
        assert key in reread
    assert reread["name"] == "speall-test"
    assert len(reread["resources"]) == len(pkg["resources"])


# ---------------------------------------------------------------------------
# validate_datapackage
# ---------------------------------------------------------------------------


def test_validate_happy_path(fake_bundle: Path) -> None:
    pkg = build_datapackage(fake_bundle, name="speall-test")
    out = write_datapackage(fake_bundle, pkg)
    assert validate_datapackage(out) == []


def test_validate_missing_name(tmp_path: Path) -> None:
    p = tmp_path / "datapackage.json"
    write_json(p, {"resources": [{"path": "x.parquet", "format": "parquet"}]})
    errs = validate_datapackage(p)
    assert any("name" in e for e in errs)


def test_validate_missing_resources(tmp_path: Path) -> None:
    p = tmp_path / "datapackage.json"
    write_json(p, {"name": "x"})
    errs = validate_datapackage(p)
    assert any("resources" in e for e in errs)


def test_validate_resource_missing_path_and_format(tmp_path: Path) -> None:
    p = tmp_path / "datapackage.json"
    write_json(p, {"name": "x", "resources": [{"name": "r"}]})
    errs = validate_datapackage(p)
    assert any("path" in e for e in errs)
    assert any("format" in e for e in errs)


def test_validate_missing_file(tmp_path: Path) -> None:
    errs = validate_datapackage(tmp_path / "does-not-exist.json")
    assert errs and "not found" in errs[0]

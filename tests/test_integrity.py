"""Tests for src.integrity: checksums, provenance, datacite, validate_croissant."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.integrity.checksums import (
    build_checksum_manifest,
    verify_checksum_manifest,
    write_checksum_manifest,
)
from src.integrity.datacite import _REQUIRED_KEYS, build_datacite_metadata
from src.integrity.provenance import build_prov_graph
from src.integrity.validate_croissant import validate_croissant_file

_REPO_ROOT = Path(__file__).parent.parent
_SAMPLES = _REPO_ROOT / "Speall_MRI_Samples" / "series"


# ---------------------------------------------------------------------------
# checksums
# ---------------------------------------------------------------------------


class TestChecksums:
    def test_roundtrip_three_files(self, tmp_path: Path) -> None:
        """Build manifest on 3 files, then verify -- all should be OK."""
        (tmp_path / "a.txt").write_bytes(b"hello")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_bytes(b"world")
        (tmp_path / "c.json").write_text('{"x": 1}', encoding="utf-8")

        manifest_path = tmp_path / "checksums.json"
        write_checksum_manifest(tmp_path, manifest_path)

        # Manifest itself is in the directory; verify should not include it
        # (it's not excluded by default, so n_ok counts it too -- that's fine)
        n_ok, n_bad, bad = verify_checksum_manifest(manifest_path, tmp_path)
        assert n_bad == 0, f"Expected 0 mismatches, got: {bad}"
        assert n_ok >= 3

    def test_corrupted_file_detected(self, tmp_path: Path) -> None:
        """A file whose bytes change after manifest creation must be flagged."""
        target = tmp_path / "secret.txt"
        target.write_bytes(b"original content")
        manifest_path = tmp_path / "checksums.json"
        write_checksum_manifest(tmp_path, manifest_path)

        # Corrupt the file
        target.write_bytes(b"tampered!!")

        n_ok, n_bad, bad = verify_checksum_manifest(manifest_path, tmp_path)
        assert n_bad >= 1
        assert any("secret.txt" in p for p in bad)

    def test_missing_file_detected(self, tmp_path: Path) -> None:
        """A file deleted after manifest creation appears as <missing>."""
        victim = tmp_path / "gone.txt"
        victim.write_bytes(b"I will vanish")
        manifest_path = tmp_path / "checksums.json"
        write_checksum_manifest(tmp_path, manifest_path)

        victim.unlink()

        _, n_bad, bad = verify_checksum_manifest(manifest_path, tmp_path)
        assert n_bad >= 1
        assert any("<missing>" in p for p in bad)

    def test_exclude_patterns(self, tmp_path: Path) -> None:
        """DCM and tar files are excluded by default."""
        (tmp_path / "slice.dcm").write_bytes(b"\x00" * 64)
        (tmp_path / "archive.tar").write_bytes(b"\x00" * 64)
        (tmp_path / "keep.json").write_text("{}", encoding="utf-8")

        manifest = build_checksum_manifest(tmp_path)
        assert "slice.dcm" not in manifest["files"]
        assert "archive.tar" not in manifest["files"]
        assert "keep.json" in manifest["files"]

    def test_manifest_schema(self, tmp_path: Path) -> None:
        """Manifest has required top-level keys."""
        (tmp_path / "x.txt").write_bytes(b"x")
        m = build_checksum_manifest(tmp_path)
        assert m["version"] == 1
        assert m["algorithm"] == "sha256"
        assert "generated_at" in m
        assert isinstance(m["files"], dict)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_graph_non_empty(self) -> None:
        """Building a graph from Speall_MRI_Samples/series/ yields entities."""
        graph = build_prov_graph(_SAMPLES)
        assert len(graph["entity"]) > 0, "Expected at least one entity"
        assert len(graph["activity"]) > 0, "Expected at least one activity"
        assert len(graph["agent"]) > 0, "Expected at least one agent"

    def test_derived_entities_present(self) -> None:
        """Each detail JSON becomes a derived entity."""
        graph = build_prov_graph(_SAMPLES)
        n_json = len(list(_SAMPLES.glob("*.json")))
        n_detail = sum(
            1 for eid in graph["entity"] if eid.startswith("micom:detail:")
        )
        assert n_detail == n_json

    def test_was_generated_by_edges(self) -> None:
        """Every derived entity has at least one wasGeneratedBy edge."""
        graph = build_prov_graph(_SAMPLES)
        generated = {
            e["prov:entity"] for e in graph["wasGeneratedBy"].values()
        }
        derived = {
            eid for eid in graph["entity"] if eid.startswith("micom:detail:")
        }
        assert derived.issubset(generated)

    def test_prov_prefix_present(self) -> None:
        """PROV namespace must be declared."""
        graph = build_prov_graph(_SAMPLES)
        assert "prov" in graph["prefix"]
        assert "http://www.w3.org/ns/prov#" in graph["prefix"]["prov"]


# ---------------------------------------------------------------------------
# datacite
# ---------------------------------------------------------------------------


class TestDatacite:
    def test_required_keys_present(self) -> None:
        """All 8 required top-level DataCite 4.5 keys must be present."""
        metadata = build_datacite_metadata()
        for key in _REQUIRED_KEYS:
            assert key in metadata, f"Missing DataCite key: {key}"

    def test_resource_type_general_is_dataset(self) -> None:
        metadata = build_datacite_metadata()
        assert metadata["types"]["resourceTypeGeneral"] == "Dataset"

    def test_creators_non_empty(self) -> None:
        metadata = build_datacite_metadata()
        assert len(metadata["creators"]) >= 1
        assert metadata["creators"][0]["name"] == "Shubh"

    def test_identifiers_doi_placeholder(self) -> None:
        metadata = build_datacite_metadata()
        doi_entries = [
            i for i in metadata["identifiers"] if i["identifierType"] == "DOI"
        ]
        assert len(doi_entries) == 1

    def test_rights_list_mit(self) -> None:
        metadata = build_datacite_metadata()
        rights_ids = [r.get("rightsIdentifier") for r in metadata["rightsList"]]
        assert "MIT" in rights_ids

    def test_dates_include_collected(self) -> None:
        metadata = build_datacite_metadata()
        date_types = [d["dateType"] for d in metadata["dates"]]
        assert "Collected" in date_types

    def test_subjects_include_mri(self) -> None:
        metadata = build_datacite_metadata()
        subjects = [s["subject"] for s in metadata["subjects"]]
        assert "MRI" in subjects


# ---------------------------------------------------------------------------
# validate_croissant
# ---------------------------------------------------------------------------


class TestValidateCroissant:
    def test_existing_croissant_passes(self) -> None:
        """The repo's croissant.json must pass structural validation."""
        errors = validate_croissant_file(_REPO_ROOT / "croissant.json")
        assert errors == [], f"Croissant validation errors: {errors}"

    def test_minimal_valid_document(self) -> None:
        """A minimal well-formed document passes."""
        doc = {
            "@context": {"sc": "https://schema.org/"},
            "@type": "sc:Dataset",
            "sc:name": "Test",
            "sc:description": "Test dataset",
            "sc:url": "https://example.com",
            "sc:license": "MIT",
            "distribution": [
                {
                    "@id": "dist-1",
                    "@type": "cr:FileObject",
                    "sc:name": "file.parquet",
                    "sc:contentUrl": "https://example.com/file.parquet",
                }
            ],
            "recordSet": [
                {
                    "@id": "records",
                    "@type": "cr:RecordSet",
                    "sc:name": "records",
                    "cr:field": [
                        {
                            "@type": "cr:Field",
                            "@id": "records/id",
                            "sc:name": "id",
                        }
                    ],
                }
            ],
        }
        from src.integrity.validate_croissant import validate_croissant
        errors = validate_croissant(doc)
        assert errors == []

    def test_missing_record_set_fails(self) -> None:
        """A document lacking recordSet must return an error."""
        doc = {
            "@context": {},
            "@type": "sc:Dataset",
            "sc:name": "T",
            "sc:description": "D",
            "sc:url": "U",
            "sc:license": "L",
            "distribution": [
                {
                    "@id": "d",
                    "@type": "cr:FileObject",
                    "sc:name": "f",
                    "sc:contentUrl": "u",
                }
            ],
        }
        from src.integrity.validate_croissant import validate_croissant
        errors = validate_croissant(doc)
        assert any("recordSet" in e for e in errors)

    def test_bad_type_fails(self) -> None:
        """@type other than sc:Dataset must produce an error."""
        doc = {
            "@context": {},
            "@type": "schema:Nonsense",
            "sc:name": "X",
            "sc:description": "Y",
            "sc:url": "Z",
            "sc:license": "W",
            "distribution": [{"@id": "d", "@type": "cr:FileObject",
                              "sc:name": "f", "sc:contentUrl": "u"}],
            "recordSet": [{"@id": "r", "@type": "cr:RecordSet",
                           "sc:name": "r", "cr:field": [{"@id": "f"}]}],
        }
        from src.integrity.validate_croissant import validate_croissant
        errors = validate_croissant(doc)
        assert any("@type" in e for e in errors)

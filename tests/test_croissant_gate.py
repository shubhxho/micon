"""Tests for the Croissant 1.0 validation gate that fronts every HF upload.

The contract under test:

  * If ``croissant.json`` (or the file passed via ``croissant_path``) passes
    structural validation, ``upload_to_huggingface`` proceeds and the
    HF API is called.
  * If validation fails, ``CroissantValidationError`` is raised and NO HF
    API method is invoked — the upload is hard-aborted.
  * The ``skip_croissant_check=True`` escape hatch bypasses the gate even
    when the metadata is invalid.

All HF traffic is mocked — these tests do not hit the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.hf_upload import (
    CroissantValidationError,
    gate_croissant,
    upload_to_huggingface,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_VALID_CROISSANT: dict = {
    "@context": {
        "@vocab": "https://schema.org/",
        "cr": "http://mlcommons.org/croissant/",
    },
    "@type": "sc:Dataset",
    "name": "test-dataset",
    "description": "fixture",
    "url": "https://example.org/test",
    "license": "MIT",
    "distribution": [
        {
            "@id": "file-1",
            "@type": "cr:FileObject",
            "name": "data.parquet",
            "contentUrl": "https://example.org/data.parquet",
        }
    ],
    "recordSet": [
        {
            "@id": "rs-1",
            "@type": "cr:RecordSet",
            "name": "rows",
            "field": [
                {"@id": "rs-1/col", "@type": "cr:Field", "name": "col"}
            ],
        }
    ],
}


_INVALID_CROISSANT: dict = {
    # Missing @context, @type, distribution, recordSet — multiple violations.
    "name": "broken",
}


@pytest.fixture()
def valid_croissant_file(tmp_path: Path) -> Path:
    p = tmp_path / "croissant.json"
    p.write_text(json.dumps(_VALID_CROISSANT))
    return p


@pytest.fixture()
def invalid_croissant_file(tmp_path: Path) -> Path:
    p = tmp_path / "croissant.json"
    p.write_text(json.dumps(_INVALID_CROISSANT))
    return p


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    """A minimal out_dir that has at least one uploadable file (README)."""
    d = tmp_path / "out"
    d.mkdir()
    # upload_to_huggingface writes its own README.md, but we still need the
    # directory to exist before it walks for files.
    return d


# ---------------------------------------------------------------------------
# gate_croissant() — direct unit tests
# ---------------------------------------------------------------------------


class TestGateCroissant:
    def test_valid_passes_silently(self, valid_croissant_file: Path) -> None:
        # Returns None on success; should not raise.
        assert gate_croissant(valid_croissant_file) is None

    def test_invalid_raises_with_violations_listed(
        self, invalid_croissant_file: Path
    ) -> None:
        with pytest.raises(CroissantValidationError) as excinfo:
            gate_croissant(invalid_croissant_file)
        msg = str(excinfo.value)
        # The exception message includes the violation count and at least
        # one of the missing fields.
        assert "violation" in msg
        assert "@context" in msg or "@type" in msg or "distribution" in msg

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CroissantValidationError, match="not found"):
            gate_croissant(tmp_path / "does_not_exist.json")


# ---------------------------------------------------------------------------
# upload_to_huggingface() — gate integration with mocked HF API
# ---------------------------------------------------------------------------


class TestUploadGate:
    def test_invalid_croissant_blocks_upload(
        self, out_dir: Path, invalid_croissant_file: Path
    ) -> None:
        """Bad metadata: gate raises BEFORE any HF API call is made."""
        with (
            patch("huggingface_hub.HfApi") as mock_api_cls,
            patch("huggingface_hub.create_repo") as mock_create_repo,
        ):
            with pytest.raises(CroissantValidationError):
                upload_to_huggingface(
                    out_dir=out_dir,
                    study_name="test",
                    repo_id="user/test",
                    token="fake-token",
                    croissant_path=invalid_croissant_file,
                )
            # The gate fires before HfApi is even constructed.
            mock_api_cls.assert_not_called()
            mock_create_repo.assert_not_called()

    def test_valid_croissant_allows_upload(
        self, out_dir: Path, valid_croissant_file: Path
    ) -> None:
        """Good metadata: gate passes, HF API is called."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "tester"}

        with (
            patch("huggingface_hub.HfApi", return_value=mock_api) as mock_api_cls,
            patch("huggingface_hub.create_repo") as mock_create_repo,
        ):
            url = upload_to_huggingface(
                out_dir=out_dir,
                study_name="test-study",
                repo_id="user/test",
                token="fake-token",
                croissant_path=valid_croissant_file,
            )

            mock_api_cls.assert_called_once_with(token="fake-token")
            mock_create_repo.assert_called_once()
            # upload_folder is invoked via the inner _do_upload retry wrapper.
            assert mock_api.upload_folder.called
            assert url == "https://huggingface.co/datasets/user/test"

    def test_skip_flag_bypasses_gate_even_on_invalid(
        self, out_dir: Path, invalid_croissant_file: Path
    ) -> None:
        """Escape hatch: skip_croissant_check=True ships invalid metadata."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "tester"}

        with (
            patch("huggingface_hub.HfApi", return_value=mock_api),
            patch("huggingface_hub.create_repo"),
        ):
            # Should NOT raise, despite the croissant_path being invalid.
            url = upload_to_huggingface(
                out_dir=out_dir,
                study_name="test-study",
                repo_id="user/test",
                token="fake-token",
                croissant_path=invalid_croissant_file,
                skip_croissant_check=True,
            )
            assert mock_api.upload_folder.called
            assert "user/test" in url

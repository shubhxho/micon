"""Tests for src/segmentation -- open-weights segmentation pipeline.

Covers:
  1. Model registry has the three expected entries with required metadata.
  2. segment_one_series with a mocked model produces the correct BIDS
     derivatives path.
  3. extras_for_buyers.augment_manifest adds the expected columns.
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.segmentation.models import (
    MODEL_REGISTRY,
    get_model_meta,
    list_models,
    clear_cache,
)
from src.segmentation.pipeline import (
    _derivatives_filename,
    _derivatives_dir,
    segment_one_series,
    segment_study,
)
from src.segmentation.extras_for_buyers import (
    augment_manifest,
    _parse_study_id,
    compute_cohort_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_placeholder_nifti(dest: Path) -> None:
    """Write a minimal valid gzip-compressed NIfTI-1 placeholder."""
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    header[344:348] = b"n+1\0"
    struct.pack_into("<h", header, 70, 4)
    struct.pack_into("<h", header, 72, 16)
    for i, v in enumerate([3, 8, 8, 8, 1, 1, 1, 1]):
        struct.pack_into("<h", header, 40 + i * 2, v)
    for i in range(4):
        struct.pack_into("<f", header, 76 + i * 4, 1.0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<f", header, 116, 0.0)
    voxels = bytes(8 * 8 * 8 * 2)  # int16 zeros
    ext_block = bytes(4)
    payload = bytes(header) + ext_block + voxels
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wb") as fh:
        fh.write(payload)


@pytest.fixture()
def bids_root(tmp_path: Path) -> Path:
    """Minimal BIDS tree with one T1w placeholder NIfTI."""
    root = tmp_path / "bids"
    anat_dir = root / "sub-001" / "ses-01" / "anat"
    anat_dir.mkdir(parents=True)
    t1w = anat_dir / "sub-001_ses-01_T1w.nii.gz"
    _write_placeholder_nifti(t1w)
    return root


@pytest.fixture(autouse=True)
def clear_model_cache() -> None:  # type: ignore[return]
    """Clear the in-process model cache before each test."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# 1. Model registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_has_three_required_entries(self) -> None:
        for name in ("synthstrip", "synthseg", "monai_brain_lesion"):
            assert name in MODEL_REGISTRY, f"Missing {name!r} in MODEL_REGISTRY"

    def test_synthstrip_metadata(self) -> None:
        meta = MODEL_REGISTRY["synthstrip"]
        assert meta["task"] == "brain_extraction"
        assert "T1w" in meta["input_modalities"]
        assert meta["loader"] == "freesurfer/synthstrip"
        assert meta["loader_type"] == "huggingface"
        assert meta["desc_label"] == "brainmask"

    def test_synthseg_metadata(self) -> None:
        meta = MODEL_REGISTRY["synthseg"]
        assert meta["task"] == "brain_parcellation"
        assert "T1w" in meta["input_modalities"]
        assert meta["loader_type"] == "huggingface"
        assert meta["desc_label"] == "parcel"

    def test_monai_brain_lesion_metadata(self) -> None:
        meta = MODEL_REGISTRY["monai_brain_lesion"]
        assert meta["task"] == "brain_lesion"
        assert "FLAIR" in meta["input_modalities"]
        assert meta["loader_type"] == "monai_bundle"
        assert meta["desc_label"] == "lesion"

    def test_all_entries_have_required_keys(self) -> None:
        required_keys = {
            "task", "input_modalities", "loader", "loader_type",
            "desc_label", "derivative_name",
        }
        for name, meta in MODEL_REGISTRY.items():
            missing = required_keys - meta.keys()
            assert not missing, f"{name}: missing keys {missing}"

    def test_list_models_returns_all_three(self) -> None:
        names = list_models()
        assert "synthstrip" in names
        assert "synthseg" in names
        assert "monai_brain_lesion" in names

    def test_get_model_meta_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_model_meta("nonexistent_model")


# ---------------------------------------------------------------------------
# 2. segment_one_series -- mocked model
# ---------------------------------------------------------------------------

class TestSegmentOneSeries:
    def test_output_path_in_bids_derivatives(
        self, tmp_path: Path, bids_root: Path
    ) -> None:
        """Mocked inference should produce a path in derivatives layout."""
        anat_dir = bids_root / "sub-001" / "ses-01" / "anat"
        nifti_path = anat_dir / "sub-001_ses-01_T1w.nii.gz"

        out_dir = _derivatives_dir(bids_root, "speall-synthstrip", "001", "01")
        fake_mask = np.ones((8, 8, 8), dtype="uint8")

        with patch("src.segmentation.pipeline._run_model_inference", return_value=fake_mask):
            result = segment_one_series(nifti_path, "synthstrip", out_dir, "001", "01")

        assert result["ok"] is True
        mask_path: Path = result["mask_path"]

        # Must live under derivatives/speall-synthstrip/
        rel = mask_path.relative_to(bids_root)
        parts = rel.parts
        assert parts[0] == "derivatives"
        assert parts[1] == "speall-synthstrip"
        assert parts[2] == "sub-001"
        assert parts[3] == "ses-01"
        assert parts[4] == "anat"

    def test_output_filename_bids_pattern(
        self, bids_root: Path
    ) -> None:
        """Output filename must follow the BIDS derivatives naming convention."""
        anat_dir = bids_root / "sub-001" / "ses-01" / "anat"
        nifti_path = anat_dir / "sub-001_ses-01_T1w.nii.gz"
        out_dir = _derivatives_dir(bids_root, "speall-synthstrip", "001", "01")
        fake_mask = np.zeros((8, 8, 8), dtype="uint8")
        fake_mask[2:6, 2:6, 2:6] = 1

        with patch("src.segmentation.pipeline._run_model_inference", return_value=fake_mask):
            result = segment_one_series(nifti_path, "synthstrip", out_dir, "001", "01")

        assert result["ok"] is True
        name = result["mask_path"].name
        assert name == "sub-001_ses-01_space-orig_desc-brainmask_dseg.nii.gz"

    def test_n_voxels_nonzero(self, bids_root: Path) -> None:
        anat_dir = bids_root / "sub-001" / "ses-01" / "anat"
        nifti_path = anat_dir / "sub-001_ses-01_T1w.nii.gz"
        out_dir = _derivatives_dir(bids_root, "speall-synthstrip", "001", "01")
        fake_mask = np.ones((8, 8, 8), dtype="uint8")

        with patch("src.segmentation.pipeline._run_model_inference", return_value=fake_mask):
            result = segment_one_series(nifti_path, "synthstrip", out_dir, "001", "01")

        assert result["n_voxels"] == 8 * 8 * 8

    def test_model_failure_returns_ok_false(self, bids_root: Path) -> None:
        anat_dir = bids_root / "sub-001" / "ses-01" / "anat"
        nifti_path = anat_dir / "sub-001_ses-01_T1w.nii.gz"
        out_dir = _derivatives_dir(bids_root, "speall-synthstrip", "001", "01")

        with patch(
            "src.segmentation.pipeline._run_model_inference",
            side_effect=RuntimeError("model not available"),
        ):
            result = segment_one_series(nifti_path, "synthstrip", out_dir, "001", "01")

        assert result["ok"] is False
        assert result["mask_path"] is None

    def test_missing_nifti_returns_ok_false(self, tmp_path: Path) -> None:
        nifti_path = tmp_path / "nonexistent.nii.gz"
        out_dir = tmp_path / "out"
        result = segment_one_series(nifti_path, "synthstrip", out_dir, "001", "01")
        assert result["ok"] is False

    def test_derivatives_description_written(self, bids_root: Path) -> None:
        """segment_study must write a dataset_description.json in the derivative."""
        fake_mask = np.ones((8, 8, 8), dtype="uint8")
        with patch("src.segmentation.pipeline._run_model_inference", return_value=fake_mask):
            segment_study(bids_root, "001", "01", ["synthstrip"])

        desc_path = bids_root / "derivatives" / "speall-synthstrip" / "dataset_description.json"
        assert desc_path.exists()
        data = json.loads(desc_path.read_text())
        assert data["DatasetType"] == "derivative"
        assert data["BIDSVersion"] == "1.10.0"


# ---------------------------------------------------------------------------
# 3. Manifest augmentation
# ---------------------------------------------------------------------------

class TestManifestAugmentation:
    def _make_manifest(self, tmp_path: Path, study_ids: list[str]) -> Path:
        df = pl.DataFrame(
            {"study_id": study_ids, "sequence_type": ["T1-weighted"] * len(study_ids)},
            schema={"study_id": pl.Utf8, "sequence_type": pl.Utf8},
        )
        p = tmp_path / "manifest.parquet"
        df.write_parquet(str(p))
        return p

    def test_augment_adds_required_columns(self, tmp_path: Path) -> None:
        manifest_path = self._make_manifest(tmp_path, ["sub-001_ses-01"])
        bids_root = tmp_path / "bids"
        bids_root.mkdir()

        result = augment_manifest(manifest_path, bids_root)

        for col in (
            "has_brain_mask",
            "has_brain_parcellation",
            "has_lesion_mask",
            "brain_volume_cm3",
            "ventricle_volume_cm3",
            "lesion_load_cm3",
        ):
            assert col in result.columns, f"Missing column: {col}"

    def test_augment_false_when_no_derivatives(self, tmp_path: Path) -> None:
        manifest_path = self._make_manifest(tmp_path, ["sub-001_ses-01"])
        bids_root = tmp_path / "bids"
        bids_root.mkdir()

        result = augment_manifest(manifest_path, bids_root)
        row = result.row(0, named=True)

        assert row["has_brain_mask"] is False
        assert row["has_brain_parcellation"] is False
        assert row["has_lesion_mask"] is False

    def test_augment_true_when_mask_present(
        self, tmp_path: Path, bids_root: Path
    ) -> None:
        """Place a synthstrip mask in the right BIDS path; expect has_brain_mask=True."""
        mask_dir = (
            bids_root / "derivatives" / "speall-synthstrip" / "sub-001" / "ses-01" / "anat"
        )
        mask_dir.mkdir(parents=True)
        fake_mask = np.ones((4, 4, 4), dtype="uint8")
        import nibabel as _nib
        import numpy as _np
        affine = _np.eye(4)
        img = _nib.Nifti1Image(fake_mask, affine)
        mask_fname = "sub-001_ses-01_space-orig_desc-brainmask_dseg.nii.gz"
        _nib.save(img, str(mask_dir / mask_fname))

        manifest_path = self._make_manifest(tmp_path, ["sub-001_ses-01"])
        result = augment_manifest(manifest_path, bids_root)
        row = result.row(0, named=True)

        assert row["has_brain_mask"] is True
        assert row["brain_volume_cm3"] > 0.0

    def test_existing_columns_preserved(self, tmp_path: Path) -> None:
        manifest_path = self._make_manifest(tmp_path, ["sub-001_ses-01"])
        bids_root = tmp_path / "bids"
        bids_root.mkdir()

        result = augment_manifest(manifest_path, bids_root)
        assert "study_id" in result.columns
        assert "sequence_type" in result.columns

    def test_row_count_unchanged(self, tmp_path: Path) -> None:
        manifest_path = self._make_manifest(tmp_path, ["sub-001_ses-01", "sub-002_ses-01"])
        bids_root = tmp_path / "bids"
        bids_root.mkdir()

        result = augment_manifest(manifest_path, bids_root)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 4. Helper utilities
# ---------------------------------------------------------------------------

class TestParseStudyId:
    @pytest.mark.parametrize("study_id,expected_sub,expected_ses", [
        ("sub-001_ses-01", "001", "01"),
        ("sub-MEMAR01_ses-02", "MEMAR01", "02"),
        ("001", "001", "01"),
        ("sub-999", "999", "01"),
    ])
    def test_various_formats(
        self, study_id: str, expected_sub: str, expected_ses: str
    ) -> None:
        sub, ses = _parse_study_id(study_id)
        assert sub == expected_sub
        assert ses == expected_ses


class TestDerivativesFilename:
    def test_synthstrip_filename(self) -> None:
        name = _derivatives_filename("001", "01", "brainmask")
        assert name == "sub-001_ses-01_space-orig_desc-brainmask_dseg.nii.gz"

    def test_filename_no_special_chars(self) -> None:
        name = _derivatives_filename("001", "01", "parcel")
        assert " " not in name
        assert "_" not in name.split("sub-")[0]

    def test_custom_extension(self) -> None:
        name = _derivatives_filename("001", "01", "lesion", ext=".nii")
        assert name.endswith(".nii")


class TestCohortSummary:
    def test_summary_keys(self) -> None:
        df = pl.DataFrame(
            {
                "study_id": ["sub-001_ses-01"],
                "has_brain_mask": [True],
                "has_brain_parcellation": [False],
                "has_lesion_mask": [False],
                "brain_volume_cm3": [1200.0],
                "ventricle_volume_cm3": [15.0],
                "lesion_load_cm3": [0.0],
            }
        )
        summary = compute_cohort_summary(df)
        assert summary["n_studies"] == 1
        assert summary["n_with_brain_mask"] == 1
        assert summary["n_with_parcellation"] == 0
        assert summary["mean_brain_volume_cm3"] == 1200.0

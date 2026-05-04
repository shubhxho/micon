"""Tests for src/zarr_export -- OME-Zarr export package.

All heavy libraries (zarr, ome_zarr, nibabel) are skipped via
``pytest.importorskip`` if not installed so CI without those deps
does not fail.

Test layout:
  1. MultiscalePyramid         -- build_pyramid shape / dtype checks
  2. CoordinateTransformations -- multiscale.coordinate_transformations
  3. NIfTIToOmeZarr            -- round-trip with a synthetic NIfTI
  4. OmeZarrMetadata           -- axes names, units, coordinate transforms
  5. StudyToOmeZarr            -- multi-series walk
  6. CliInspect                -- inspect command prints JSON metadata
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level skip guard -- if zarr / ome_zarr not installed, skip all
# ---------------------------------------------------------------------------
zarr = pytest.importorskip("zarr", reason="zarr not installed")
ome_zarr_pkg = pytest.importorskip("ome_zarr", reason="ome_zarr not installed")
nib = pytest.importorskip("nibabel", reason="nibabel not installed")

from src.zarr_export.converter import nifti_to_omezarr, study_to_omezarr
from src.zarr_export.multiscale import build_pyramid, coordinate_transformations

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_nifti(path: Path, shape: tuple[int, int, int] = (32, 64, 64)) -> Path:
    """Write a minimal real NIfTI-1 file with random float32 data."""
    vol = np.random.default_rng(42).standard_normal(shape).astype(np.float32)
    affine = np.diag([1.5, 1.5, 2.0, 1.0])  # 1.5mm in-plane, 2mm slice
    img = nib.Nifti1Image(vol, affine)
    img.header.set_zooms((1.5, 1.5, 2.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


@pytest.fixture()
def nifti_file(tmp_path: Path) -> Path:
    return _make_synthetic_nifti(tmp_path / "T1w.nii.gz", shape=(32, 64, 64))


@pytest.fixture()
def small_nifti_file(tmp_path: Path) -> Path:
    """Very small volume to stress-test chunk-clipping logic."""
    return _make_synthetic_nifti(tmp_path / "small.nii.gz", shape=(4, 8, 8))


@pytest.fixture()
def zarr_out(tmp_path: Path) -> Path:
    return tmp_path / "T1w.ome.zarr"


# ---------------------------------------------------------------------------
# 1. MultiscalePyramid
# ---------------------------------------------------------------------------


class TestMultiscalePyramid:
    def test_single_level_returns_full_res(self) -> None:
        vol = np.zeros((32, 64, 64), dtype=np.float32)
        pyramid = build_pyramid(vol, n_levels=1)
        assert len(pyramid) == 1
        assert pyramid[0].shape == (32, 64, 64)

    def test_n_levels_matches_requested(self) -> None:
        vol = np.ones((64, 64, 64), dtype=np.float32)
        for n in [2, 3, 4]:
            pyramid = build_pyramid(vol, n_levels=n)
            assert len(pyramid) == n

    def test_each_level_is_half_the_previous(self) -> None:
        vol = np.zeros((64, 64, 32), dtype=np.float32)
        pyramid = build_pyramid(vol, n_levels=4)
        for i in range(1, len(pyramid)):
            for dim_idx in range(3):
                expected = max(1, pyramid[i - 1].shape[dim_idx] // 2)
                actual = pyramid[i].shape[dim_idx]
                # allow +-1 due to integer division in downscale_local_mean
                assert abs(actual - expected) <= 1, (
                    f"Level {i} dim {dim_idx}: expected ~{expected}, got {actual}"
                )

    def test_output_dtype_is_float32(self) -> None:
        vol = np.zeros((16, 16, 8), dtype=np.uint16)
        pyramid = build_pyramid(vol, n_levels=2)
        for arr in pyramid:
            assert arr.dtype == np.float32

    def test_invalid_n_levels_raises(self) -> None:
        vol = np.zeros((16, 16, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="n_levels"):
            build_pyramid(vol, n_levels=0)

    def test_non_3d_input_raises(self) -> None:
        vol = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="3-D"):
            build_pyramid(vol, n_levels=2)

    def test_first_level_equals_input_transposed(self) -> None:
        vol = np.arange(32 * 16 * 8, dtype=np.float32).reshape(32, 16, 8)
        pyramid = build_pyramid(vol, n_levels=3)
        np.testing.assert_array_almost_equal(pyramid[0], vol)


# ---------------------------------------------------------------------------
# 2. CoordinateTransformations
# ---------------------------------------------------------------------------


class TestCoordinateTransformations:
    def test_returns_correct_number_of_levels(self) -> None:
        ct = coordinate_transformations((2.0, 1.5, 1.5), n_levels=4)
        assert len(ct) == 4

    def test_level0_scale_matches_zooms(self) -> None:
        ct = coordinate_transformations((2.0, 1.5, 1.5), n_levels=3)
        scale0 = ct[0][0]["scale"]
        assert scale0 == [2.0, 1.5, 1.5]

    def test_each_level_doubles_previous(self) -> None:
        ct = coordinate_transformations((1.0, 1.0, 1.0), n_levels=4)
        for i in range(1, len(ct)):
            prev = ct[i - 1][0]["scale"]
            curr = ct[i][0]["scale"]
            expected = [v * 2 for v in prev]
            assert curr == pytest.approx(expected), f"Level {i}: expected {expected}, got {curr}"

    def test_each_entry_has_type_scale(self) -> None:
        ct = coordinate_transformations((1.0, 1.0, 1.0), n_levels=3)
        for entry in ct:
            assert entry[0]["type"] == "scale"


# ---------------------------------------------------------------------------
# 3. NIfTIToOmeZarr -- round-trip
# ---------------------------------------------------------------------------


class TestNiftiToOmeZarr:
    def test_returns_ok_dict(self, nifti_file: Path, zarr_out: Path) -> None:
        result = nifti_to_omezarr(nifti_file, zarr_out, scales=4)
        assert result["ok"] is True
        assert result["n_levels"] == 4
        assert result["total_chunks"] >= 0
        assert result["total_bytes"] >= 0

    def test_zarr_group_is_created(self, nifti_file: Path, zarr_out: Path) -> None:
        nifti_to_omezarr(nifti_file, zarr_out, scales=3)
        grp = zarr.open_group(str(zarr_out), mode="r")
        assert grp is not None

    def test_correct_number_of_array_keys(self, nifti_file: Path, zarr_out: Path) -> None:
        """write_multiscale writes s0, s1, s2... arrays."""
        nifti_to_omezarr(nifti_file, zarr_out, scales=4)
        grp = zarr.open_group(str(zarr_out), mode="r")
        array_keys = [k for k in grp if not k.startswith(".")]
        assert len(array_keys) == 4

    def test_missing_nifti_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            nifti_to_omezarr(tmp_path / "nonexistent.nii.gz", tmp_path / "out.zarr")

    def test_small_volume_chunk_clipping(self, small_nifti_file: Path, tmp_path: Path) -> None:
        """Chunk size should be clipped to array shape; no error on tiny volumes."""
        result = nifti_to_omezarr(
            small_nifti_file,
            tmp_path / "small.ome.zarr",
            scales=3,
            chunk_size=(16, 256, 256),  # larger than array
        )
        assert result["ok"] is True

    def test_array_data_shape_is_zyx(self, nifti_file: Path, zarr_out: Path) -> None:
        """Full-res array (s0) should have ZYX shape matching transposed NIfTI."""
        nifti_to_omezarr(nifti_file, zarr_out, scales=2)
        grp = zarr.open_group(str(zarr_out), mode="r")
        # nibabel loads (32, 64, 64) XYZ; after transpose -> ZYX (64, 64, 32)
        s0 = grp["s0"]
        assert s0.shape == (64, 64, 32)


# ---------------------------------------------------------------------------
# 4. OmeZarrMetadata
# ---------------------------------------------------------------------------


class TestOmeZarrMetadata:
    @pytest.fixture(autouse=True)
    def _run_conversion(self, nifti_file: Path, zarr_out: Path) -> None:
        nifti_to_omezarr(nifti_file, zarr_out, scales=4)
        self._grp = zarr.open_group(str(zarr_out), mode="r")
        self._attrs = dict(self._grp.attrs)

    def test_ome_key_present(self) -> None:
        assert "ome" in self._attrs

    def test_multiscales_key_present(self) -> None:
        assert "multiscales" in self._attrs["ome"]

    def test_axes_have_three_entries(self) -> None:
        axes = self._attrs["ome"]["multiscales"][0]["axes"]
        assert len(axes) == 3

    def test_axes_names_are_zyx(self) -> None:
        axes = self._attrs["ome"]["multiscales"][0]["axes"]
        names = [a["name"] for a in axes]
        assert names == ["z", "y", "x"]

    def test_axes_type_is_space(self) -> None:
        axes = self._attrs["ome"]["multiscales"][0]["axes"]
        for ax in axes:
            assert ax["type"] == "space"

    def test_axes_unit_is_millimeter(self) -> None:
        axes = self._attrs["ome"]["multiscales"][0]["axes"]
        for ax in axes:
            assert ax["unit"] == "millimeter"

    def test_datasets_has_four_entries(self) -> None:
        datasets = self._attrs["ome"]["multiscales"][0]["datasets"]
        assert len(datasets) == 4

    def test_each_dataset_has_path(self) -> None:
        datasets = self._attrs["ome"]["multiscales"][0]["datasets"]
        for ds in datasets:
            assert "path" in ds

    def test_each_dataset_has_coordinate_transformations(self) -> None:
        datasets = self._attrs["ome"]["multiscales"][0]["datasets"]
        for ds in datasets:
            assert "coordinateTransformations" in ds

    def test_scale_transform_type(self) -> None:
        datasets = self._attrs["ome"]["multiscales"][0]["datasets"]
        for ds in datasets:
            for ct in ds["coordinateTransformations"]:
                assert ct["type"] == "scale"

    def test_scale_increases_per_level(self) -> None:
        datasets = self._attrs["ome"]["multiscales"][0]["datasets"]
        level0_scale = datasets[0]["coordinateTransformations"][0]["scale"]
        level1_scale = datasets[1]["coordinateTransformations"][0]["scale"]
        # Each element of level1 should be 2x level0
        for s0, s1 in zip(level0_scale, level1_scale, strict=False):
            assert abs(s1 - s0 * 2) < 1e-5, f"Expected s1={s0 * 2}, got {s1}"

    def test_version_is_05(self) -> None:
        assert self._attrs["ome"]["version"] == "0.5"


# ---------------------------------------------------------------------------
# 5. StudyToOmeZarr
# ---------------------------------------------------------------------------


class TestStudyToOmeZarr:
    @pytest.fixture()
    def study_dir(self, tmp_path: Path) -> Path:
        """Fake BIDS study with three NIfTI files."""
        study = tmp_path / "sub-001"
        _make_synthetic_nifti(study / "ses-01" / "anat" / "T1w.nii.gz")
        _make_synthetic_nifti(study / "ses-01" / "anat" / "T2w.nii.gz")
        _make_synthetic_nifti(study / "ses-01" / "dwi" / "dwi.nii.gz")
        return study

    def test_returns_summary_dict(self, study_dir: Path, tmp_path: Path) -> None:
        result = study_to_omezarr(study_dir, tmp_path / "zarr_out")
        assert "series_converted" in result
        assert "series_failed" in result
        assert "study_zarr" in result

    def test_converts_all_series(self, study_dir: Path, tmp_path: Path) -> None:
        result = study_to_omezarr(study_dir, tmp_path / "zarr_out")
        assert result["series_converted"] == 3
        assert result["series_failed"] == 0

    def test_study_zarr_group_is_created(self, study_dir: Path, tmp_path: Path) -> None:
        result = study_to_omezarr(study_dir, tmp_path / "zarr_out")
        study_zarr = Path(result["study_zarr"])
        assert study_zarr.exists()

    def test_each_series_has_ok_true(self, study_dir: Path, tmp_path: Path) -> None:
        result = study_to_omezarr(study_dir, tmp_path / "zarr_out")
        for series in result["series"]:
            assert series.get("ok") is True, f"Series {series['series']} failed"

    def test_empty_study_dir_returns_zero_series(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_study"
        empty.mkdir()
        result = study_to_omezarr(empty, tmp_path / "zarr_out")
        assert result["series_converted"] == 0


# ---------------------------------------------------------------------------
# 6. CLI inspect
# ---------------------------------------------------------------------------


class TestCliInspect:
    def test_inspect_prints_metadata(
        self, nifti_file: Path, zarr_out: Path, capsys: pytest.CaptureFixture
    ) -> None:
        nifti_to_omezarr(nifti_file, zarr_out, scales=2)

        from typer.testing import CliRunner

        from src.zarr_export.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["inspect", str(zarr_out)])
        assert result.exit_code == 0, result.output
        output = result.output
        assert "multiscales" in output or "ome" in output

    def test_inspect_invalid_path_exits_nonzero(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from src.zarr_export.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["inspect", str(tmp_path / "nonexistent.zarr")])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 7. Sharding (Zarr v3 killer feature)
# ---------------------------------------------------------------------------


class TestNiftiSharding:
    """Regression: nifti_to_omezarr must produce sharded Zarr v3 arrays.

    Without sharding, a 100GB volume with 1MB chunks lands as 100K small
    files; with shard=8x chunk it collapses to ~12.5K files, ~10x cheaper to
    LIST on S3/GCS.  This guard ensures shards stay enabled and that the
    shard shape is always an integer multiple of the chunk shape (a Zarr v3
    constraint that previously broke at deeper pyramid levels for some
    volume sizes).
    """

    @pytest.fixture()
    def sharded_nifti_zarr(self, tmp_path: Path) -> Path:
        """Volume large enough that >1 chunk fits along Z at each level."""
        nifti_p = _make_synthetic_nifti(tmp_path / "sh.nii.gz", shape=(64, 64, 64))
        out = tmp_path / "sh.ome.zarr"
        # Smaller chunks so multiple chunks fit along Z and sharding kicks in
        nifti_to_omezarr(nifti_p, out, scales=4, chunk_size=(8, 64, 64))
        return out

    def test_level0_is_sharded(self, sharded_nifti_zarr: Path) -> None:
        arr = zarr.open_array(str(sharded_nifti_zarr / "s0"), mode="r")
        assert arr.shards is not None, "Expected Zarr v3 sharding on s0"
        assert arr.shards != arr.chunks

    def test_shard_is_multiple_of_chunk_all_levels(self, sharded_nifti_zarr: Path) -> None:
        for key in ("s0", "s1", "s2", "s3"):
            arr = zarr.open_array(str(sharded_nifti_zarr / key), mode="r")
            if arr.shards is None:
                continue
            for s, c in zip(arr.shards, arr.chunks, strict=True):
                assert s % c == 0, f"{key}: shard {arr.shards} not multiple of chunk {arr.chunks}"

    def test_zarr_format_is_v3(self, sharded_nifti_zarr: Path) -> None:
        arr = zarr.open_array(str(sharded_nifti_zarr / "s0"), mode="r")
        assert arr.metadata.zarr_format == 3

    def test_sharding_disabled_when_target_one(self, tmp_path: Path) -> None:
        """shard_chunks_target=1 must skip sharding entirely."""
        nifti_p = _make_synthetic_nifti(tmp_path / "ns.nii.gz", shape=(32, 32, 32))
        out = tmp_path / "ns.ome.zarr"
        nifti_to_omezarr(nifti_p, out, scales=2, chunk_size=(8, 32, 32), shard_chunks_target=1)
        arr = zarr.open_array(str(out / "s0"), mode="r")
        assert arr.shards is None or arr.shards == arr.chunks

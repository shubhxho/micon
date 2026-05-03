"""Tests for src/zarr_export/series_writer.py -- per-series OME-Zarr writer.

Tests:
  1. SeriesVolumeWriter   -- write → re-read, pyramid level 0 round-trips
  2. OmeNgffMetadata      -- axes, units, dataset paths, version
  3. ChunkShapePolicy     -- chunk bytes <= 2 MB, dynamic derivation
  4. SpacingEncoding      -- coordinate transforms match voxel_spacing_mm
  5. ReturnDict           -- correct keys returned
  6. ErrorHandling        -- non-3D volume raises ValueError
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip entire module if zarr / ome_zarr not installed
zarr = pytest.importorskip("zarr", reason="zarr not installed")
pytest.importorskip("ome_zarr", reason="ome_zarr not installed")

from src.zarr_export.series_writer import (
    _clip_chunk,
    _compute_chunk_shape,
    series_volume_to_omezarr,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def brain_vol() -> np.ndarray:
    """Synthetic float32 brain-like volume, shape (32, 64, 64) ZYX."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((32, 64, 64)).astype(np.float32)


@pytest.fixture()
def zarr_result(brain_vol: np.ndarray, tmp_path: Path) -> tuple[dict, Path]:
    """Write volume to a temp zarr and return (result_dict, zarr_path)."""
    zp = tmp_path / "T1w.zarr"
    result = series_volume_to_omezarr(
        brain_vol,
        zp,
        voxel_spacing_mm=(2.0, 1.5, 1.5),
        series_uid="1.2.3.4.5",
        sequence_type="T1-weighted",
    )
    return result, zp


# ---------------------------------------------------------------------------
# 1. SeriesVolumeWriter -- round-trip
# ---------------------------------------------------------------------------

class TestSeriesVolumeWriter:
    def test_zarr_group_exists(self, zarr_result: tuple[dict, Path]) -> None:
        _, zp = zarr_result
        assert zp.exists()

    def test_level0_roundtrips_data(
        self, brain_vol: np.ndarray, zarr_result: tuple[dict, Path]
    ) -> None:
        """s0 array must contain the same values as the input volume."""
        _, zp = zarr_result
        grp = zarr.open_group(str(zp), mode="r")
        s0 = grp["s0"][:]
        np.testing.assert_allclose(s0, brain_vol, rtol=1e-5)

    def test_four_pyramid_levels_written(self, zarr_result: tuple[dict, Path]) -> None:
        _, zp = zarr_result
        grp = zarr.open_group(str(zp), mode="r")
        arr_keys = [k for k in grp.keys() if not k.startswith(".")]
        assert len(arr_keys) == 4

    def test_pyramid_shapes_decrease(self, zarr_result: tuple[dict, Path]) -> None:
        """Each level's first dim should be <= the previous level's first dim."""
        _, zp = zarr_result
        grp = zarr.open_group(str(zp), mode="r")
        prev_z = None
        for key in ["s0", "s1", "s2", "s3"]:
            arr = grp[key]
            z = arr.shape[0]
            if prev_z is not None:
                assert z <= prev_z
            prev_z = z


# ---------------------------------------------------------------------------
# 2. OmeNgffMetadata
# ---------------------------------------------------------------------------

class TestOmeNgffMetadata:
    @pytest.fixture(autouse=True)
    def _load_attrs(self, zarr_result: tuple[dict, Path]) -> None:
        _, zp = zarr_result
        grp = zarr.open_group(str(zp), mode="r")
        self._attrs = dict(grp.attrs)

    def test_ome_key_present(self) -> None:
        assert "ome" in self._attrs

    def test_version_is_05(self) -> None:
        assert self._attrs["ome"]["version"] == "0.5"

    def test_multiscales_present(self) -> None:
        assert "multiscales" in self._attrs["ome"]

    def test_axes_zyx(self) -> None:
        axes = self._attrs["ome"]["multiscales"][0]["axes"]
        names = [a["name"] for a in axes]
        assert names == ["z", "y", "x"]

    def test_axes_type_space(self) -> None:
        axes = self._attrs["ome"]["multiscales"][0]["axes"]
        for ax in axes:
            assert ax["type"] == "space"

    def test_axes_unit_millimeter(self) -> None:
        axes = self._attrs["ome"]["multiscales"][0]["axes"]
        for ax in axes:
            assert ax["unit"] == "millimeter"

    def test_four_dataset_entries(self) -> None:
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

    def test_series_uid_in_attrs(self) -> None:
        assert self._attrs.get("series_uid") == "1.2.3.4.5"

    def test_sequence_type_in_attrs(self) -> None:
        assert self._attrs.get("sequence_type") == "T1-weighted"

    def test_level0_scale_matches_voxel_spacing(self) -> None:
        """s0 coordinateTransformations scale must match voxel_spacing_mm."""
        datasets = self._attrs["ome"]["multiscales"][0]["datasets"]
        s0_ct = datasets[0]["coordinateTransformations"][0]
        assert s0_ct["type"] == "scale"
        assert s0_ct["scale"] == pytest.approx([2.0, 1.5, 1.5])

    def test_level1_scale_doubles_level0(self) -> None:
        datasets = self._attrs["ome"]["multiscales"][0]["datasets"]
        s0_scale = datasets[0]["coordinateTransformations"][0]["scale"]
        s1_scale = datasets[1]["coordinateTransformations"][0]["scale"]
        for s0, s1 in zip(s0_scale, s1_scale):
            assert s1 == pytest.approx(s0 * 2)


# ---------------------------------------------------------------------------
# 3. ChunkShapePolicy
# ---------------------------------------------------------------------------

class TestChunkShapePolicy:
    def test_chunk_bytes_under_2mb(self, brain_vol: np.ndarray, tmp_path: Path) -> None:
        """Level-0 chunk must be <= 2 MB raw float32 bytes."""
        zp = tmp_path / "chunk_test.zarr"
        result = series_volume_to_omezarr(
            brain_vol,
            zp,
            voxel_spacing_mm=(2.0, 1.5, 1.5),
            series_uid="uid-chunk",
            sequence_type=None,
        )
        chunk = result["chunk_shape"]
        chunk_bytes = chunk[0] * chunk[1] * chunk[2] * 4  # float32
        assert chunk_bytes <= 2 * 1024 * 1024

    def test_compute_chunk_shape_targets_1mb(self) -> None:
        """For 256x256 plane, z_chunk should target ~1 MB."""
        shape = (50, 256, 256)
        chunk = _compute_chunk_shape(shape)
        chunk_bytes = chunk[0] * chunk[1] * chunk[2] * 4
        assert chunk_bytes <= 1_048_576 * 2  # at most 2x target (rounding)
        assert chunk[1] == 256
        assert chunk[2] == 256

    def test_compute_chunk_shape_small_z(self) -> None:
        """z_chunk must be at least 1 even for single-slice volumes."""
        shape = (1, 64, 64)
        chunk = _compute_chunk_shape(shape)
        assert chunk[0] >= 1

    def test_clip_chunk_does_not_exceed_shape(self) -> None:
        chunk = (32, 256, 256)
        shape = (8, 16, 16)
        clipped = _clip_chunk(chunk, shape)
        assert clipped == (8, 16, 16)

    def test_small_volume_no_error(self, tmp_path: Path) -> None:
        """Small volumes (tiny Z) must not error — chunk clipping handles it."""
        vol = np.ones((3, 8, 8), dtype=np.float32)
        result = series_volume_to_omezarr(
            vol,
            tmp_path / "small.zarr",
            voxel_spacing_mm=(5.0, 1.0, 1.0),
            series_uid="uid-small",
            sequence_type=None,
        )
        assert result["n_levels"] == 4


# ---------------------------------------------------------------------------
# 4. SpacingEncoding
# ---------------------------------------------------------------------------

class TestSpacingEncoding:
    def test_custom_spacing_reflected_in_transforms(self, tmp_path: Path) -> None:
        vol = np.zeros((16, 32, 32), dtype=np.float32)
        sp = (3.0, 0.5, 0.5)
        zp = tmp_path / "sp_test.zarr"
        series_volume_to_omezarr(
            vol,
            zp,
            voxel_spacing_mm=sp,
            series_uid="uid-sp",
            sequence_type=None,
        )
        grp = zarr.open_group(str(zp), mode="r")
        attrs = dict(grp.attrs)
        scale0 = attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        assert scale0 == pytest.approx([3.0, 0.5, 0.5])


# ---------------------------------------------------------------------------
# 5. ReturnDict
# ---------------------------------------------------------------------------

class TestReturnDict:
    def test_return_keys(self, zarr_result: tuple[dict, Path]) -> None:
        result, _ = zarr_result
        assert set(result.keys()) == {"path", "n_levels", "total_bytes", "chunk_shape"}

    def test_n_levels_is_4(self, zarr_result: tuple[dict, Path]) -> None:
        result, _ = zarr_result
        assert result["n_levels"] == 4

    def test_path_is_str(self, zarr_result: tuple[dict, Path]) -> None:
        result, _ = zarr_result
        assert isinstance(result["path"], str)

    def test_total_bytes_positive(self, zarr_result: tuple[dict, Path]) -> None:
        result, _ = zarr_result
        assert result["total_bytes"] > 0

    def test_chunk_shape_is_3_tuple(self, zarr_result: tuple[dict, Path]) -> None:
        result, _ = zarr_result
        assert len(result["chunk_shape"]) == 3


# ---------------------------------------------------------------------------
# 6. ErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_non_3d_raises_value_error(self, tmp_path: Path) -> None:
        vol_2d = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="3-D"):
            series_volume_to_omezarr(
                vol_2d,
                tmp_path / "bad.zarr",
                voxel_spacing_mm=(1.0, 1.0, 1.0),
                series_uid="uid-bad",
                sequence_type=None,
            )

    def test_none_sequence_type_stored_as_empty_string(self, tmp_path: Path) -> None:
        vol = np.zeros((8, 16, 16), dtype=np.float32)
        zp = tmp_path / "none_seq.zarr"
        series_volume_to_omezarr(
            vol,
            zp,
            voxel_spacing_mm=(1.0, 1.0, 1.0),
            series_uid="uid-none",
            sequence_type=None,
        )
        grp = zarr.open_group(str(zp), mode="r")
        assert dict(grp.attrs).get("sequence_type") == ""

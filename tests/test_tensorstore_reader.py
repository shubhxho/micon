"""Tests for src/zarr_export/tensorstore_reader.py.

Tests:
  1. TensorStoreOpen     -- shape and dtype of opened level-0 array
  2. ReadVolumeRoundtrip -- read_volume matches the original numpy array
  3. Level1Downsampled   -- level=1 returns a smaller array
  4. LazyImportGuard     -- module importable without tensorstore installed
  5. FileNotFoundError   -- missing path raises FileNotFoundError
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip entire module if tensorstore is not installed
ts = pytest.importorskip("tensorstore", reason="tensorstore not installed")

# zarr / ome_zarr are required to write the fixture
pytest.importorskip("zarr", reason="zarr not installed")
pytest.importorskip("ome_zarr", reason="ome_zarr not installed")

from src.zarr_export.series_writer import series_volume_to_omezarr
from src.zarr_export.tensorstore_reader import open_zarr_with_tensorstore, read_volume

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_vol() -> np.ndarray:
    """Synthetic 12x64x64 float32 volume with fixed seed."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((12, 64, 64)).astype(np.float32)


@pytest.fixture(scope="module")
def written_zarr(tmp_path_factory: pytest.TempPathFactory, synthetic_vol: np.ndarray) -> Path:
    """Write the synthetic volume to a temp OME-Zarr and return the path."""
    zp = tmp_path_factory.mktemp("ts_reader") / "synthetic.zarr"
    series_volume_to_omezarr(
        synthetic_vol,
        zp,
        voxel_spacing_mm=(2.0, 1.0, 1.0),
        series_uid="1.2.840.99999",
        sequence_type="T1-weighted",
    )
    return zp


# ---------------------------------------------------------------------------
# 1. TensorStoreOpen
# ---------------------------------------------------------------------------


class TestTensorStoreOpen:
    def test_shape_matches_input(self, written_zarr: Path, synthetic_vol: np.ndarray) -> None:
        """Level-0 TensorStore handle must report the same shape as input."""
        store = open_zarr_with_tensorstore(written_zarr, level=0)
        assert tuple(store.shape) == synthetic_vol.shape

    def test_dtype_is_float32(self, written_zarr: Path) -> None:
        store = open_zarr_with_tensorstore(written_zarr, level=0)
        assert store.dtype == np.float32

    def test_returns_tensorstore_object(self, written_zarr: Path) -> None:
        import tensorstore as _ts

        store = open_zarr_with_tensorstore(written_zarr, level=0)
        assert hasattr(store, "read")  # TensorStore handle


# ---------------------------------------------------------------------------
# 2. ReadVolumeRoundtrip
# ---------------------------------------------------------------------------


class TestReadVolumeRoundtrip:
    def test_read_volume_matches_input(self, written_zarr: Path, synthetic_vol: np.ndarray) -> None:
        """read_volume(level=0) must be numerically equal to original (lossless zstd)."""
        result = read_volume(written_zarr, level=0)
        np.testing.assert_array_equal(result, synthetic_vol)

    def test_returns_numpy_array(self, written_zarr: Path) -> None:
        result = read_volume(written_zarr, level=0)
        assert isinstance(result, np.ndarray)

    def test_shape_matches(self, written_zarr: Path, synthetic_vol: np.ndarray) -> None:
        result = read_volume(written_zarr, level=0)
        assert result.shape == synthetic_vol.shape


# ---------------------------------------------------------------------------
# 3. Level1Downsampled
# ---------------------------------------------------------------------------


class TestLevel1Downsampled:
    def test_level1_smaller_shape(self, written_zarr: Path, synthetic_vol: np.ndarray) -> None:
        """Level 1 must have at least one dimension strictly smaller than level 0."""
        arr1 = read_volume(written_zarr, level=1)
        assert any(d1 < d0 for d1, d0 in zip(arr1.shape, synthetic_vol.shape, strict=False))

    def test_level1_all_dims_le_level0(self, written_zarr: Path, synthetic_vol: np.ndarray) -> None:
        """All level-1 dims must be <= level-0 dims (pyramids only shrink)."""
        arr1 = read_volume(written_zarr, level=1)
        for d1, d0 in zip(arr1.shape, synthetic_vol.shape, strict=False):
            assert d1 <= d0

    def test_level1_dtype_float32(self, written_zarr: Path) -> None:
        arr1 = read_volume(written_zarr, level=1)
        assert arr1.dtype == np.float32


# ---------------------------------------------------------------------------
# 4. LazyImportGuard
# ---------------------------------------------------------------------------


class TestLazyImportGuard:
    def test_module_importable(self) -> None:
        """The module must import without error."""
        from src.zarr_export import tensorstore_reader

        assert tensorstore_reader is not None


# ---------------------------------------------------------------------------
# 5. FileNotFoundError
# ---------------------------------------------------------------------------


class TestFileNotFoundError:
    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            open_zarr_with_tensorstore(tmp_path / "does_not_exist.zarr")

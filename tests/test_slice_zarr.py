"""Tests for src/zarr_export/slice_store and the zarr backend of slice_export.

Coverage:
  1. SlicestoZarr       -- round-trip shape, dtype, windowing correctness
  2. CompressionRatio   -- stored bytes < uncompressed bytes
  3. BothBackend        -- both PNG and Zarr files are produced
  4. EmptyVolume        -- returns sensible dict without crashing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr", reason="zarr not installed")

from src.zarr_export.slice_store import _normalize_slice, slices_to_zarr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _smooth_volume(shape: tuple[int, int, int] = (20, 64, 64)) -> np.ndarray:
    """Return a smooth float32 gradient volume that compresses well."""
    n, h, w = shape
    base = np.linspace(0.0, 1000.0, h * w, dtype="float32").reshape(h, w)
    return np.stack([base * (0.8 + 0.4 * i / max(n - 1, 1)) for i in range(n)])


# ---------------------------------------------------------------------------
# 1. SlicesToZarr -- round-trip
# ---------------------------------------------------------------------------


class TestSlicesToZarr:
    def test_output_shape_matches_volume(self, tmp_path: Path) -> None:
        vol = _smooth_volume((12, 64, 64))
        slices_to_zarr(vol, tmp_path / "out.zarr", series_label="t1")
        arr = zarr.open_array(str(tmp_path / "out.zarr"), mode="r")
        assert arr.shape == (12, 64, 64)

    def test_output_dtype_is_uint8(self, tmp_path: Path) -> None:
        vol = _smooth_volume((8, 32, 32))
        slices_to_zarr(vol, tmp_path / "out.zarr", series_label="t1")
        arr = zarr.open_array(str(tmp_path / "out.zarr"), mode="r")
        assert arr.dtype == np.uint8

    def test_windowing_matches_manual_calculation(self, tmp_path: Path) -> None:
        """Stored pixel values must match the 1st/99th-percentile normalisation."""
        vol = _smooth_volume((4, 32, 32))
        slices_to_zarr(vol, tmp_path / "out.zarr", series_label="t1")
        arr = zarr.open_array(str(tmp_path / "out.zarr"), mode="r")[:]

        for i in range(vol.shape[0]):
            expected = _normalize_slice(vol[i])
            np.testing.assert_array_equal(
                arr[i],
                expected,
                err_msg=f"Windowing mismatch on slice {i}",
            )

    def test_returns_correct_n_slices(self, tmp_path: Path) -> None:
        vol = _smooth_volume((15, 64, 64))
        result = slices_to_zarr(vol, tmp_path / "out.zarr", series_label="s1")
        assert result["n_slices"] == 15

    def test_attrs_embedded(self, tmp_path: Path) -> None:
        vol = _smooth_volume((8, 32, 32))
        slices_to_zarr(vol, tmp_path / "out.zarr", series_label="my_series")
        arr = zarr.open_array(str(tmp_path / "out.zarr"), mode="r")
        attrs = dict(arr.attrs)
        assert attrs["series_label"] == "my_series"
        assert attrs["n_slices"] == 8
        assert "original_dtype" in attrs
        assert "window" in attrs
        assert "min" in attrs["window"]
        assert "max" in attrs["window"]

    def test_zarr_directory_is_created(self, tmp_path: Path) -> None:
        zarr_path = tmp_path / "nested" / "dir" / "out.zarr"
        vol = _smooth_volume((4, 16, 16))
        result = slices_to_zarr(vol, zarr_path, series_label="t1")
        assert zarr_path.exists()
        assert result["path"] == str(zarr_path)


# ---------------------------------------------------------------------------
# 2. CompressionRatio
# ---------------------------------------------------------------------------


class TestCompressionRatio:
    def test_compression_ratio_less_than_one(self, tmp_path: Path) -> None:
        """A smooth gradient volume should compress well below 1.0."""
        vol = _smooth_volume((20, 128, 128))
        result = slices_to_zarr(vol, tmp_path / "out.zarr", series_label="t1")
        assert result["compression_ratio"] < 1.0, (
            f"Expected compression_ratio < 1.0, got {result['compression_ratio']:.4f}"
        )

    def test_total_bytes_positive(self, tmp_path: Path) -> None:
        vol = _smooth_volume((8, 64, 64))
        result = slices_to_zarr(vol, tmp_path / "out.zarr", series_label="t1")
        assert result["total_bytes"] > 0

    def test_n_chunks_reported(self, tmp_path: Path) -> None:
        vol = _smooth_volume((20, 64, 64))
        result = slices_to_zarr(vol, tmp_path / "out.zarr", series_label="t1")
        assert result["n_chunks"] >= 1


# ---------------------------------------------------------------------------
# 3. BothBackend -- export_all_slices with backend="both"
# ---------------------------------------------------------------------------


class TestBothBackend:
    def test_both_png_and_zarr_exist(self, tmp_path: Path) -> None:
        from src.export.slice_export import export_all_slices

        vol = _smooth_volume((4, 32, 32)).astype("float32")
        result = export_all_slices(
            vol,
            series_name="test_series",
            out_dir=tmp_path,
            windows=[],
            backend="both",
        )

        # PNG directory
        png_dir = tmp_path / "slices" / "test_series"
        assert png_dir.exists(), "PNG directory should exist"
        pngs = list(png_dir.glob("axial_*.png"))
        assert len(pngs) == 4, f"Expected 4 axial PNGs, found {len(pngs)}"

        # Zarr store
        zarr_path = tmp_path / "slices" / "test_series.slices.zarr"
        assert zarr_path.exists(), "Zarr store should exist"
        assert "zarr" in result

    def test_zarr_only_no_pngs(self, tmp_path: Path) -> None:
        from src.export.slice_export import export_all_slices

        vol = _smooth_volume((4, 32, 32)).astype("float32")
        export_all_slices(
            vol,
            series_name="zarr_only",
            out_dir=tmp_path,
            windows=[],
            backend="zarr",
        )
        zarr_path = tmp_path / "slices" / "zarr_only.slices.zarr"
        assert zarr_path.exists()

        # No PNG axial files should be created
        png_dir = tmp_path / "slices" / "zarr_only"
        axial_pngs = list(png_dir.glob("axial_*.png")) if png_dir.exists() else []
        assert len(axial_pngs) == 0, "PNG files should not be written in zarr-only mode"

    def test_png_default_unchanged(self, tmp_path: Path) -> None:
        """Default backend='png' must not create any Zarr store."""
        from src.export.slice_export import export_all_slices

        vol = _smooth_volume((4, 32, 32)).astype("float32")
        export_all_slices(
            vol,
            series_name="png_only",
            out_dir=tmp_path,
            windows=[],
        )
        zarr_path = tmp_path / "slices" / "png_only.slices.zarr"
        assert not zarr_path.exists(), "Zarr store must not be created with default backend"


# ---------------------------------------------------------------------------
# 4. EmptyVolume
# ---------------------------------------------------------------------------


class TestEmptyVolume:
    def test_empty_volume_returns_zero_slices(self, tmp_path: Path) -> None:
        vol = np.zeros((0, 64, 64), dtype="float32")
        result = slices_to_zarr(vol, tmp_path / "empty.zarr", series_label="empty")
        assert result["n_slices"] == 0
        assert result["n_chunks"] == 0
        assert result["total_bytes"] == 0

    def test_empty_volume_does_not_crash(self, tmp_path: Path) -> None:
        vol = np.zeros((0, 32, 32), dtype="float32")
        # Should complete without raising
        slices_to_zarr(vol, tmp_path / "empty2.zarr", series_label="e")

    def test_empty_volume_compression_ratio_zero(self, tmp_path: Path) -> None:
        vol = np.zeros((0, 32, 32), dtype="float32")
        result = slices_to_zarr(vol, tmp_path / "empty3.zarr", series_label="e")
        assert result["compression_ratio"] == 0.0


# ---------------------------------------------------------------------------
# 5. Sharding (Zarr v3 killer feature)
# ---------------------------------------------------------------------------


class TestSlicesSharding:
    """Regression: slices_to_zarr must use Zarr v3 sharding for large stacks.

    Without sharding the per-slice store explodes into thousands of tiny
    chunk files, each one a separate S3/GCS GET on read.  This guard makes
    sure the shard codec stays wired and that shard shape is a valid
    integer multiple of the chunk shape.
    """

    def test_large_stack_is_sharded(self, tmp_path: Path) -> None:
        # 200 slices at 256x256 uint8 ≈ 12 MB raw → many chunks per shard
        vol = _smooth_volume((200, 256, 256))
        zp = tmp_path / "sh.zarr"
        slices_to_zarr(vol, zp, series_label="t1")
        arr = zarr.open_array(str(zp), mode="r")
        assert arr.shards is not None, "Expected Zarr v3 sharding"
        assert arr.shards != arr.chunks
        for s, c in zip(arr.shards, arr.chunks, strict=True):
            assert s % c == 0, f"shard {arr.shards} not multiple of chunk {arr.chunks}"

    def test_zarr_format_is_v3(self, tmp_path: Path) -> None:
        vol = _smooth_volume((40, 64, 64))
        zp = tmp_path / "fmt.zarr"
        slices_to_zarr(vol, zp, series_label="t1")
        arr = zarr.open_array(str(zp), mode="r")
        assert arr.metadata.zarr_format == 3

    def test_encoding_attrs_record_shard_shape(self, tmp_path: Path) -> None:
        vol = _smooth_volume((200, 256, 256))
        zp = tmp_path / "enc.zarr"
        slices_to_zarr(vol, zp, series_label="t1")
        arr = zarr.open_array(str(zp), mode="r")
        attrs = dict(arr.attrs)
        assert attrs["encoding"]["sharding"] is True
        assert attrs["encoding"]["format"] == "zarr-v3"

"""Tests for src/lance_export/slice_dataset and the lance backend of slice_export.

Coverage:
  1. RoundTrip      -- write Lance, reopen, assert len == 20, schema, decode slice
  2. AllBackend     -- backend="all" writes png/zarr/lance side-by-side
  3. EmptyVolume    -- returns {n_slices: 0} without crashing
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

lance = pytest.importorskip("lance", reason="lance not installed")

from src.lance_export.slice_dataset import slices_to_lance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _smooth_volume(shape: tuple[int, int, int] = (20, 128, 128)) -> np.ndarray:
    """Return a smooth float32 gradient volume."""
    n, h, w = shape
    base = np.linspace(0.0, 1000.0, h * w, dtype="float32").reshape(h, w)
    return np.stack([base * (0.8 + 0.4 * i / max(n - 1, 1)) for i in range(n)])


# ---------------------------------------------------------------------------
# 1. RoundTrip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_dataset_length_matches_n_slices(self, tmp_path: Path) -> None:
        vol = _smooth_volume((20, 128, 128))
        slices_to_lance(vol, tmp_path / "out.lance", series_label="t1")
        ds = lance.dataset(str(tmp_path / "out.lance"))
        assert len(ds) == 20

    def test_schema_columns_present(self, tmp_path: Path) -> None:
        vol = _smooth_volume((8, 64, 64))
        result = slices_to_lance(vol, tmp_path / "out.lance", series_label="t1")
        ds = lance.dataset(str(tmp_path / "out.lance"))
        col_names = set(ds.schema.names)
        assert {"slice_index", "image", "n_rows", "n_cols", "dtype", "series_label"}.issubset(
            col_names
        )

    def test_image_column_is_binary(self, tmp_path: Path) -> None:
        vol = _smooth_volume((4, 32, 32))
        slices_to_lance(vol, tmp_path / "out.lance", series_label="t1")
        ds = lance.dataset(str(tmp_path / "out.lance"))
        schema = ds.schema
        img_field = schema.field("image")
        assert pa.types.is_binary(img_field.type), f"Expected binary, got {img_field.type}"

    def test_slice_decode_gives_correct_shape(self, tmp_path: Path) -> None:
        """Decoding ds.take([5])['image'][0] with PIL must yield a 128x128 uint8 array."""
        from PIL import Image

        vol = _smooth_volume((20, 128, 128))
        slices_to_lance(vol, tmp_path / "out.lance", series_label="t1")
        ds = lance.dataset(str(tmp_path / "out.lance"))
        batch = ds.take([5]).to_pydict()
        png_bytes = batch["image"][0]
        img = Image.open(io.BytesIO(png_bytes))
        arr = np.array(img)
        assert arr.shape == (128, 128), f"Expected (128,128), got {arr.shape}"
        assert arr.dtype == np.uint8

    def test_n_rows_n_cols_stored_correctly(self, tmp_path: Path) -> None:
        vol = _smooth_volume((10, 64, 96))
        slices_to_lance(vol, tmp_path / "out.lance", series_label="s1")
        ds = lance.dataset(str(tmp_path / "out.lance"))
        row = ds.take([0]).to_pydict()
        assert row["n_rows"][0] == 64
        assert row["n_cols"][0] == 96

    def test_series_label_stored_in_every_row(self, tmp_path: Path) -> None:
        vol = _smooth_volume((5, 32, 32))
        slices_to_lance(vol, tmp_path / "out.lance", series_label="my_series")
        ds = lance.dataset(str(tmp_path / "out.lance"))
        labels = ds.to_table(columns=["series_label"]).to_pydict()["series_label"]
        assert all(lbl == "my_series" for lbl in labels)

    def test_returns_correct_metadata(self, tmp_path: Path) -> None:
        vol = _smooth_volume((12, 64, 64))
        result = slices_to_lance(vol, tmp_path / "out.lance", series_label="t1")
        assert result["n_slices"] == 12
        assert result["total_bytes"] > 0
        assert "schema" in result
        assert result["path"] == str(tmp_path / "out.lance")

    def test_windowing_byte_identical_to_normalize_slice(self, tmp_path: Path) -> None:
        """PNG bytes in Lance must decode to same pixels as _normalize_slice."""
        from PIL import Image
        from src.export.slice_export import _normalize_slice

        vol = _smooth_volume((4, 32, 32))
        slices_to_lance(vol, tmp_path / "out.lance", series_label="t1")
        ds = lance.dataset(str(tmp_path / "out.lance"))
        batch = ds.take([2]).to_pydict()
        png_bytes = batch["image"][0]
        decoded = np.array(Image.open(io.BytesIO(png_bytes)))
        expected = _normalize_slice(vol[2])
        np.testing.assert_array_equal(decoded, expected)


# ---------------------------------------------------------------------------
# 2. AllBackend
# ---------------------------------------------------------------------------


class TestAllBackend:
    def test_all_three_outputs_exist(self, tmp_path: Path) -> None:
        from src.export.slice_export import export_all_slices

        vol = _smooth_volume((4, 32, 32)).astype("float32")
        result = export_all_slices(
            vol,
            series_name="test_all",
            out_dir=tmp_path,
            windows=[],
            backend="all",
        )
        # PNG directory
        png_dir = tmp_path / "slices" / "test_all"
        assert png_dir.exists(), "PNG directory must exist"
        assert len(list(png_dir.glob("axial_*.png"))) == 4

        # Zarr store
        zarr_path = tmp_path / "slices" / "test_all.slices.zarr"
        assert zarr_path.exists(), "Zarr store must exist"

        # Lance dataset
        lance_path = tmp_path / "slices" / "test_all.slices.lance"
        assert lance_path.exists(), "Lance dataset must exist"

        assert "zarr" in result
        assert "lance" in result

    def test_lance_only_no_pngs(self, tmp_path: Path) -> None:
        from src.export.slice_export import export_all_slices

        vol = _smooth_volume((4, 32, 32)).astype("float32")
        export_all_slices(
            vol,
            series_name="lance_only",
            out_dir=tmp_path,
            windows=[],
            backend="lance",
        )
        lance_path = tmp_path / "slices" / "lance_only.slices.lance"
        assert lance_path.exists(), "Lance dataset must exist"

        png_dir = tmp_path / "slices" / "lance_only"
        axial_pngs = list(png_dir.glob("axial_*.png")) if png_dir.exists() else []
        assert len(axial_pngs) == 0, "No PNGs should be written in lance-only mode"

    def test_png_default_unchanged(self, tmp_path: Path) -> None:
        """Default backend='png' must not create any Lance dataset."""
        from src.export.slice_export import export_all_slices

        vol = _smooth_volume((4, 32, 32)).astype("float32")
        export_all_slices(
            vol,
            series_name="png_default",
            out_dir=tmp_path,
            windows=[],
        )
        lance_path = tmp_path / "slices" / "png_default.slices.lance"
        assert not lance_path.exists(), "Lance must not be created with default backend"


# ---------------------------------------------------------------------------
# 3. EmptyVolume
# ---------------------------------------------------------------------------


class TestEmptyVolume:
    def test_empty_volume_returns_zero_slices(self, tmp_path: Path) -> None:
        vol = np.zeros((0, 64, 64), dtype="float32")
        result = slices_to_lance(vol, tmp_path / "empty.lance", series_label="empty")
        assert result["n_slices"] == 0

    def test_empty_volume_total_bytes_zero(self, tmp_path: Path) -> None:
        vol = np.zeros((0, 64, 64), dtype="float32")
        result = slices_to_lance(vol, tmp_path / "empty.lance", series_label="empty")
        assert result["total_bytes"] == 0

    def test_empty_volume_does_not_crash(self, tmp_path: Path) -> None:
        vol = np.zeros((0, 32, 32), dtype="float32")
        # Should complete without raising
        slices_to_lance(vol, tmp_path / "empty2.lance", series_label="e")

    def test_empty_volume_lance_dataset_openable(self, tmp_path: Path) -> None:
        """Empty lance dataset should still be openable and have len==0."""
        vol = np.zeros((0, 64, 64), dtype="float32")
        slices_to_lance(vol, tmp_path / "empty.lance", series_label="empty")
        ds = lance.dataset(str(tmp_path / "empty.lance"))
        assert len(ds) == 0

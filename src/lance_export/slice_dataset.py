"""Lance-backed slice dataset — writes a 3-D volume's axial slices as a
Lance columnar dataset suitable for direct HuggingFace datasets consumption.

Each row stores one normalised axial slice encoded as PNG bytes so training
code can read `ds["image"]` and decode with PIL without untarring anything.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from PIL import Image

# Import canonical windowing so Lance and PNG paths stay byte-identical.
from src.export.slice_export import _normalize_slice


_SCHEMA = pa.schema(
    [
        pa.field("slice_index", pa.int32()),
        pa.field("image", pa.binary()),
        pa.field("n_rows", pa.int32()),
        pa.field("n_cols", pa.int32()),
        pa.field("dtype", pa.string()),
        pa.field("series_label", pa.string()),
    ]
)


def _encode_png(arr: np.ndarray) -> bytes:
    """Encode a 2-D uint8 array as PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def slices_to_lance(
    volume: np.ndarray,
    lance_path: Path,
    *,
    series_label: str,
) -> dict[str, Any]:
    """Write a 3-D volume as a Lance dataset of windowed PNG-encoded slices.

    Args:
        volume:       3-D numpy array (n_slices, height, width), any dtype.
        lance_path:   Destination ``.lance`` directory (created by lance).
        series_label: Human-readable series name stored in every row.

    Returns:
        Dict with keys: ``path``, ``n_slices``, ``total_bytes``, ``schema``.
    """
    import lance

    lance_path = Path(lance_path)

    if volume.ndim < 3:
        volume = volume[np.newaxis, ...]

    n_slices, height, width = volume.shape[:3]
    original_dtype = str(volume.dtype)

    if n_slices == 0:
        table = pa.table(
            {
                "slice_index": pa.array([], type=pa.int32()),
                "image": pa.array([], type=pa.binary()),
                "n_rows": pa.array([], type=pa.int32()),
                "n_cols": pa.array([], type=pa.int32()),
                "dtype": pa.array([], type=pa.string()),
                "series_label": pa.array([], type=pa.string()),
            },
            schema=_SCHEMA,
        )
        lance.write_dataset(table, str(lance_path))
        return {
            "path": str(lance_path),
            "n_slices": 0,
            "total_bytes": 0,
            "schema": str(_SCHEMA),
        }

    rows: list[dict[str, Any]] = []
    for i in range(n_slices):
        normed = _normalize_slice(volume[i])
        png_bytes = _encode_png(normed)
        rows.append(
            {
                "slice_index": i,
                "image": png_bytes,
                "n_rows": height,
                "n_cols": width,
                "dtype": original_dtype,
                "series_label": series_label,
            }
        )

    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    lance.write_dataset(table, str(lance_path))

    ds = lance.dataset(str(lance_path))
    total_bytes = (
        sum(f.stat().st_size for f in lance_path.rglob("*") if f.is_file())
        if lance_path.exists()
        else 0
    )

    return {
        "path": str(lance_path),
        "n_slices": n_slices,
        "total_bytes": total_bytes,
        "schema": str(ds.schema),
    }

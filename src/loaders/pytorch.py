"""PyTorch Dataset and DataLoader factory for the Speall MRI Brain Dataset.

Lazy torch import: this module imports cleanly without torch installed.
Install extras: pip install "micom[torch]"

Usage
-----
    from pathlib import Path
    from src.loaders.pytorch import SpeallMRIDataset, make_dataloader

    ds = SpeallMRIDataset(
        manifest_path=Path("manifest.parquet"),
        root=Path("/data/speall"),
        split="train",
    )
    loader = make_dataloader(
        manifest=Path("manifest.parquet"),
        root=Path("/data/speall"),
        split="train",
        batch_size=8,
        num_workers=4,
    )
    for batch in loader:
        print(batch["study_id"], batch["quality_grade"])
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Lazy torch import so the module imports even when torch is not installed.
try:
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as _TorchBase

    _TORCH_AVAILABLE = True
except ImportError:
    DataLoader = None  # type: ignore[assignment,misc]
    _TorchBase = object  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read a parquet file and return rows as list of dicts.

    Tries pyarrow then falls back to pandas.
    """
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(str(path))
        cols = table.to_pydict()
        n = len(table)
        return [{col: cols[col][i] for col in cols} for i in range(n)]
    except ImportError:
        pass

    try:
        import pandas as pd

        df = pd.read_parquet(str(path))
        return df.to_dict(orient="records")
    except ImportError:
        pass

    raise ImportError(
        "Either pyarrow or pandas is required to read manifest.parquet. "
        "Install with: pip install pyarrow"
    )


def _assign_split(series_uid: str, split: str) -> bool:
    """Deterministic split assignment from uid hash when splits.parquet absent."""
    bucket = abs(hash(series_uid)) % 10
    if split == "train":
        return bucket >= 2
    if split == "test":
        return bucket == 0
    if split == "validation":
        return bucket == 1
    return True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SpeallMRIDataset(_TorchBase):  # type: ignore[misc]
    """torch.utils.data.Dataset wrapper for the Speall MRI manifest.

    Parameters
    ----------
    manifest_path:
        Path to ``manifest.parquet`` produced by ``src.manifest.builder``.
    root:
        Root of the dataset tree (study_id/ directories live here).
    split:
        One of ``"train"``, ``"test"``, ``"validation"``, or ``"all"``.
    sequence_type:
        Optional filter, e.g. ``"DWI"``.  Case-insensitive.
    quality_grades:
        Optional list of accepted grades, e.g. ``["A", "B"]``.
    transform:
        Optional callable applied to the dict returned by ``__getitem__``.
    load_volume:
        When True, load the DICOM volume via SimpleITK and attach as
        ``item["volume"]`` (numpy float32 array, shape [D, H, W]).
    splits_path:
        Path to ``splits.parquet`` with columns ``series_uid`` and ``split``.
        If absent, deterministic hash-based assignment is used.
    """

    def __init__(
        self,
        manifest_path: Path,
        root: Path,
        split: str = "train",
        sequence_type: str | None = None,
        quality_grades: list[str] | None = None,
        transform: Callable[[dict[str, Any]], Any] | None = None,
        load_volume: bool = False,
        splits_path: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.load_volume = load_volume

        all_rows = _read_parquet_rows(Path(manifest_path))

        # Build split membership set
        split_uids: set[str] | None = None
        if splits_path and Path(splits_path).exists():
            split_rows = _read_parquet_rows(Path(splits_path))
            split_uids = {r["series_uid"] for r in split_rows if r.get("split") == split}

        self._rows: list[dict[str, Any]] = []
        for row in all_rows:
            uid: str = row.get("series_uid") or ""
            if split_uids is not None:
                if uid not in split_uids:
                    continue
            elif split != "all" and not _assign_split(uid, split):
                continue

            if sequence_type:
                row_seq = (row.get("sequence_type") or "").upper()
                if row_seq != sequence_type.upper():
                    continue

            if quality_grades:
                row_grade = row.get("quality_grade") or ""
                if row_grade.upper() not in [g.upper() for g in quality_grades]:
                    continue

            self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = dict(self._rows[idx])
        study_id = row.get("study_id") or ""
        detail_path_rel = row.get("detail_path") or ""

        # Resolve paths
        series_dir = self.root / study_id
        if detail_path_rel:
            series_dir = self.root / study_id / Path(detail_path_rel).parent

        row["series_dir"] = str(series_dir)
        row["detail_json"] = self._load_detail_json(self.root, study_id, detail_path_rel)

        if self.load_volume:
            row["volume"] = self._load_volume(series_dir)

        if self.transform is not None:
            row = self.transform(row)

        return row

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_detail_json(root: Path, study_id: str, detail_path_rel: str) -> dict[str, Any]:
        if not detail_path_rel:
            return {}
        p = root / study_id / detail_path_rel
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _load_volume(series_dir: Path) -> Any:
        """Load DICOM volume via SimpleITK, returns float32 numpy array or None."""
        try:
            import SimpleITK as sitk
        except ImportError:
            return None

        dcm_files = sorted(series_dir.glob("*.dcm"))
        if not dcm_files:
            # Try .nii.gz
            nii_files = list(series_dir.glob("*.nii.gz")) + list(series_dir.glob("*.nii"))
            if nii_files:
                img = sitk.ReadImage(str(nii_files[0]))
                arr: Any = sitk.GetArrayFromImage(img)
                return arr.astype("float32")
            return None

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(f) for f in dcm_files])
        try:
            img = reader.Execute()
            arr = sitk.GetArrayFromImage(img)
            return arr.astype("float32")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def make_dataloader(
    manifest: Path,
    root: Path,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 0,
    sequence_type: str | None = None,
    quality_grades: list[str] | None = None,
    transform: Callable[[dict[str, Any]], Any] | None = None,
    load_volume: bool = False,
    splits_path: Path | None = None,
) -> DataLoader:  # type: ignore[type-arg]
    """Create a DataLoader for the Speall MRI dataset.

    Parameters mirror ``SpeallMRIDataset.__init__`` with ``batch_size``
    and ``num_workers`` added for the DataLoader constructor.

    Raises
    ------
    ImportError
        If torch is not installed.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for make_dataloader. Install with: pip install 'micom[torch]'"
        )
    from torch.utils.data import DataLoader as _DL

    dataset = SpeallMRIDataset(
        manifest_path=manifest,
        root=root,
        split=split,
        sequence_type=sequence_type,
        quality_grades=quality_grades,
        transform=transform,
        load_volume=load_volume,
        splits_path=splits_path,
    )
    return _DL(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Collate a list of dicts into a dict of lists (strings/None-safe)."""
    if not batch:
        return {}
    keys = batch[0].keys()
    return {k: [item.get(k) for item in batch] for k in keys}

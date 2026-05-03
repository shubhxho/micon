"""Tests for framework loaders: PyTorch Dataset, MONAI dict dataset, HF builder.

All tests are designed to run without torch, monai, or datasets installed.
Heavy-framework tests skip gracefully when dependencies are absent.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SERIES_SCHEMA = pa.schema([
    ("study_id", pa.string()),
    ("series_uid", pa.string()),
    ("series_number", pa.int64()),
    ("series_description", pa.string()),
    ("sequence_type", pa.string()),
    ("sequence_confidence", pa.string()),
    ("modality", pa.string()),
    ("file_count", pa.int64()),
    ("tr_ms", pa.float64()),
    ("te_ms", pa.float64()),
    ("ti_ms", pa.float64()),
    ("fa_deg", pa.float64()),
    ("b_value", pa.float64()),
    ("field_strength_T", pa.float64()),
    ("plane", pa.string()),
    ("volume_shape", pa.list_(pa.int64())),
    ("spacing_mm", pa.list_(pa.float64())),
    ("fov_mm", pa.list_(pa.float64())),
    ("volume_snr", pa.float64()),
    ("volume_cnr", pa.float64()),
    ("volume_entropy", pa.float64()),
    ("quality_grade", pa.string()),
    ("quality_score", pa.float64()),
    ("ml_score", pa.float64()),
    ("ml_grade", pa.string()),
    ("commercial_tier", pa.string()),
    ("detail_path", pa.string()),
    ("montage_path", pa.string()),
    ("has_tar_shard", pa.bool_()),
])

_N = 20  # rows in synthetic manifest


def _make_manifest_rows(n: int = _N) -> dict[str, list[Any]]:
    """Generate n synthetic manifest rows."""
    seqs = ["DWI", "FLAIR", "T1", "T2", "SWAN"]
    grades = ["A", "B", "C", "D"]
    return {
        "study_id": [f"study_{i // 4:04d}" for i in range(n)],
        "series_uid": [f"uid_{i:06d}" for i in range(n)],
        "series_number": list(range(n)),
        "series_description": [seqs[i % len(seqs)] for i in range(n)],
        "sequence_type": [seqs[i % len(seqs)] for i in range(n)],
        "sequence_confidence": ["high"] * n,
        "modality": ["MR"] * n,
        "file_count": [50] * n,
        "tr_ms": [6000.0] * n,
        "te_ms": [75.0] * n,
        "ti_ms": [None] * n,
        "fa_deg": [90.0] * n,
        "b_value": [1000.0 if i % 5 == 0 else None for i in range(n)],
        "field_strength_T": [3.0] * n,
        "plane": ["axial"] * n,
        "volume_shape": [[50, 256, 256]] * n,
        "spacing_mm": [[1.09, 1.09, 2.94]] * n,
        "fov_mm": [[147.0, 280.0, 280.0]] * n,
        "volume_snr": [0.5] * n,
        "volume_cnr": [1500.0] * n,
        "volume_entropy": [1.8] * n,
        "quality_grade": [grades[i % len(grades)] for i in range(n)],
        "quality_score": [75.0] * n,
        "ml_score": [80.0] * n,
        "ml_grade": [grades[i % len(grades)] for i in range(n)],
        "commercial_tier": ["premium"] * n,
        "detail_path": [""] * n,
        "montage_path": [None] * n,
        "has_tar_shard": [False] * n,
    }


@pytest.fixture()
def synthetic_manifest(tmp_path: Path) -> Path:
    """Write a synthetic manifest.parquet to a temp directory."""
    rows = _make_manifest_rows()
    table = pa.table(rows, schema=_SERIES_SCHEMA)
    path = tmp_path / "manifest.parquet"
    pq.write_table(table, path)
    return path


# ---------------------------------------------------------------------------
# 1. PyTorch Dataset -- module imports without torch
# ---------------------------------------------------------------------------


def test_pytorch_module_imports_without_torch() -> None:
    """src.loaders.pytorch must import cleanly even when torch is not installed."""
    # Temporarily shadow torch in sys.modules to simulate absence
    _original = sys.modules.get("torch")
    _original_data = sys.modules.get("torch.utils.data")
    sys.modules["torch"] = None  # type: ignore[assignment]
    sys.modules["torch.utils.data"] = None  # type: ignore[assignment]

    # Force re-import
    if "src.loaders.pytorch" in sys.modules:
        del sys.modules["src.loaders.pytorch"]

    try:
        mod = importlib.import_module("src.loaders.pytorch")
        assert hasattr(mod, "SpeallMRIDataset"), "SpeallMRIDataset class must be exported"
        assert hasattr(mod, "make_dataloader"), "make_dataloader must be exported"
    finally:
        # Restore
        if _original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = _original
        if _original_data is None:
            sys.modules.pop("torch.utils.data", None)
        else:
            sys.modules["torch.utils.data"] = _original_data
        # Re-import with real torch (if any) for subsequent tests
        if "src.loaders.pytorch" in sys.modules:
            del sys.modules["src.loaders.pytorch"]
        importlib.import_module("src.loaders.pytorch")


def test_speall_dataset_construction(synthetic_manifest: Path, tmp_path: Path) -> None:
    """SpeallMRIDataset constructs and __len__ returns a non-negative integer."""
    from src.loaders.pytorch import SpeallMRIDataset

    ds = SpeallMRIDataset(
        manifest_path=synthetic_manifest,
        root=tmp_path,
        split="all",
    )
    assert isinstance(len(ds), int)
    assert len(ds) == _N


def test_speall_dataset_sequence_filter(synthetic_manifest: Path, tmp_path: Path) -> None:
    """Sequence type filter returns a strict subset."""
    from src.loaders.pytorch import SpeallMRIDataset

    ds_all = SpeallMRIDataset(manifest_path=synthetic_manifest, root=tmp_path, split="all")
    ds_dwi = SpeallMRIDataset(manifest_path=synthetic_manifest, root=tmp_path, split="all", sequence_type="DWI")
    assert len(ds_dwi) < len(ds_all)
    assert len(ds_dwi) > 0


def test_speall_dataset_grade_filter(synthetic_manifest: Path, tmp_path: Path) -> None:
    """Quality grade filter keeps only matching rows."""
    from src.loaders.pytorch import SpeallMRIDataset

    ds = SpeallMRIDataset(
        manifest_path=synthetic_manifest,
        root=tmp_path,
        split="all",
        quality_grades=["A"],
    )
    # All rows have grades cycling A/B/C/D -- so A appears _N/4 times
    expected = _N // 4
    assert len(ds) == expected


def test_speall_dataset_getitem_keys(synthetic_manifest: Path, tmp_path: Path) -> None:
    """__getitem__ returns a dict with required metadata keys."""
    from src.loaders.pytorch import SpeallMRIDataset

    ds = SpeallMRIDataset(manifest_path=synthetic_manifest, root=tmp_path, split="all")
    item = ds[0]
    required_keys = {"study_id", "series_uid", "sequence_type", "quality_grade", "detail_json"}
    assert required_keys.issubset(item.keys()), f"Missing keys: {required_keys - item.keys()}"


def test_make_dataloader_raises_without_torch(synthetic_manifest: Path, tmp_path: Path) -> None:
    """make_dataloader raises ImportError when torch is absent."""
    # Simulate torch absence
    _orig = sys.modules.get("torch")
    sys.modules["torch"] = None  # type: ignore[assignment]

    if "src.loaders.pytorch" in sys.modules:
        del sys.modules["src.loaders.pytorch"]
    mod = importlib.import_module("src.loaders.pytorch")

    with pytest.raises((ImportError, TypeError)):
        mod.make_dataloader(manifest=synthetic_manifest, root=tmp_path)

    # Restore
    if _orig is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = _orig
    if "src.loaders.pytorch" in sys.modules:
        del sys.modules["src.loaders.pytorch"]
    importlib.import_module("src.loaders.pytorch")


# ---------------------------------------------------------------------------
# 2. MONAI dict dataset
# ---------------------------------------------------------------------------


def test_monai_module_imports() -> None:
    """src.loaders.monai must import cleanly without monai installed."""
    if "src.loaders.monai" in sys.modules:
        del sys.modules["src.loaders.monai"]
    mod = importlib.import_module("src.loaders.monai")
    assert hasattr(mod, "to_monai_dict_dataset")
    assert hasattr(mod, "recommend_transforms")


def test_to_monai_dict_dataset_returns_list(synthetic_manifest: Path, tmp_path: Path) -> None:
    """to_monai_dict_dataset returns a list of dicts."""
    from src.loaders.monai import to_monai_dict_dataset

    result = to_monai_dict_dataset(
        manifest_path=synthetic_manifest,
        root=tmp_path,
        split="all",
    )
    assert isinstance(result, list)
    assert len(result) == _N


def test_to_monai_dict_dataset_correct_keys(synthetic_manifest: Path, tmp_path: Path) -> None:
    """Each dict has 'image', 'label', and 'metadata' keys."""
    from src.loaders.monai import to_monai_dict_dataset

    result = to_monai_dict_dataset(
        manifest_path=synthetic_manifest,
        root=tmp_path,
        split="all",
    )
    assert len(result) > 0
    for item in result:
        assert "image" in item, f"Missing 'image' key in {item.keys()}"
        assert "label" in item, f"Missing 'label' key in {item.keys()}"
        assert "metadata" in item, f"Missing 'metadata' key in {item.keys()}"
        assert isinstance(item["label"], int), "label must be an int"
        assert isinstance(item["metadata"], dict), "metadata must be a dict"


def test_to_monai_dict_dataset_label_range(synthetic_manifest: Path, tmp_path: Path) -> None:
    """Labels are integers in [0, 4]."""
    from src.loaders.monai import to_monai_dict_dataset

    result = to_monai_dict_dataset(manifest_path=synthetic_manifest, root=tmp_path, split="all")
    for item in result:
        assert 0 <= item["label"] <= 4, f"Unexpected label: {item['label']}"


def test_to_monai_dict_dataset_sequence_filter(synthetic_manifest: Path, tmp_path: Path) -> None:
    """sequence_type filter reduces result set."""
    from src.loaders.monai import to_monai_dict_dataset

    all_items = to_monai_dict_dataset(manifest_path=synthetic_manifest, root=tmp_path, split="all")
    dwi_items = to_monai_dict_dataset(manifest_path=synthetic_manifest, root=tmp_path, split="all", sequence_type="DWI")
    assert len(dwi_items) < len(all_items)
    for item in dwi_items:
        assert item["metadata"]["sequence_type"] == "DWI"


def test_recommend_transforms_returns_list() -> None:
    """recommend_transforms returns a non-empty list for every known type."""
    from src.loaders.monai import recommend_transforms

    for seq in ["DWI", "FLAIR", "T1", "T2", "SWAN", "TOF"]:
        result = recommend_transforms(seq)
        assert isinstance(result, list)
        assert len(result) > 0, f"No transforms returned for {seq}"
        assert all(isinstance(t, str) for t in result)


def test_recommend_transforms_case_insensitive() -> None:
    """recommend_transforms accepts lowercase sequence names."""
    from src.loaders.monai import recommend_transforms

    lower_result = recommend_transforms("dwi")
    upper_result = recommend_transforms("DWI")
    assert lower_result == upper_result


def test_recommend_transforms_unknown_returns_default() -> None:
    """recommend_transforms returns a non-empty fallback for unknown types."""
    from src.loaders.monai import recommend_transforms

    result = recommend_transforms("UNKNOWN_SEQUENCE_XYZ")
    assert isinstance(result, list)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 3. HuggingFace loading script _info()
# ---------------------------------------------------------------------------


def test_hf_loading_script_info() -> None:
    """SpeallMRI._info() returns valid datasets.Features when datasets is available."""
    datasets = pytest.importorskip("datasets")

    # Import the standalone loading script (not src/)
    import importlib.util
    script_path = Path(__file__).parent.parent / "speall_mri_loading_script.py"
    assert script_path.exists(), f"Loading script not found: {script_path}"

    spec = importlib.util.spec_from_file_location("speall_mri_loading_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    builder_cls = getattr(module, "SpeallMRI", None)
    assert builder_cls is not None, "SpeallMRI class not found in loading script"

    # Instantiate with minimal config
    builder = builder_cls()
    info = builder._info()

    assert isinstance(info, datasets.DatasetInfo)
    assert info.features is not None

    required_feature_keys = {
        "study_id",
        "series_uid",
        "sequence_type",
        "quality_grade",
        "ml_score",
        "multiplane_image",
        "detail_json",
    }
    missing = required_feature_keys - set(info.features.keys())
    assert not missing, f"Missing feature keys: {missing}"


def test_hf_builder_configs() -> None:
    """BUILDER_CONFIGS covers all required config names."""
    datasets = pytest.importorskip("datasets")

    import importlib.util
    script_path = Path(__file__).parent.parent / "speall_mri_loading_script.py"
    spec = importlib.util.spec_from_file_location("speall_mri_loading_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    expected_configs = {"all", "dwi", "flair", "t1", "t2", "swan", "tof", "grade_a"}
    actual_names = {cfg.name for cfg in module.BUILDER_CONFIGS}
    missing = expected_configs - actual_names
    assert not missing, f"Missing builder configs: {missing}"

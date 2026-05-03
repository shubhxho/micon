"""HuggingFace GeneratorBasedBuilder loading script for the Speall MRI Brain Dataset.

This file is uploaded to the HF dataset repo root (shubhxho/speall-mri) and
executed server-side by the ``datasets`` library.  It is intentionally
self-contained -- no imports from ``src/``.

Usage
-----
    from datasets import load_dataset

    # Stream all series
    ds = load_dataset("shubhxho/speall-mri", "all", split="train", streaming=True)

    # Only DWI series
    ds = load_dataset("shubhxho/speall-mri", "dwi", split="train")

    # Only A-grade series (premium quality)
    ds = load_dataset("shubhxho/speall-mri", "grade_a", split="train")

Each example contains:
    study_id, series_uid, sequence_type, series_description,
    field_strength_T, quality_grade, ml_score,
    multiplane_image, histogram_image, enhanced_image,
    detail_json (serialised JSON string of the full detail record)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Generator, Iterator

import datasets

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

_VERSION = datasets.Version("2026.4.0")

# ---------------------------------------------------------------------------
# Builder configs
# ---------------------------------------------------------------------------

_SEQUENCE_CONFIGS: dict[str, str | None] = {
    "all": None,
    "dwi": "DWI",
    "flair": "FLAIR",
    "t1": "T1",
    "t2": "T2",
    "swan": "SWAN",
    "tof": "TOF",
}

_GRADE_CONFIGS: dict[str, str | None] = {
    "grade_a": "A",
}

_ALL_CONFIGS = {**_SEQUENCE_CONFIGS, **_GRADE_CONFIGS}


def _make_config(name: str) -> datasets.BuilderConfig:
    descriptions = {
        "all": "All series (no filter).",
        "dwi": "Diffusion-weighted imaging (DWI/ADC) series only.",
        "flair": "T2-FLAIR series only.",
        "t1": "T1-weighted series only.",
        "t2": "T2-weighted (non-FLAIR) series only.",
        "swan": "SWAN / SWI series only.",
        "tof": "Time-of-flight angiography series only.",
        "grade_a": "Grade-A (ml_score >= 80) series only -- premium quality.",
    }
    return datasets.BuilderConfig(
        name=name,
        version=_VERSION,
        description=descriptions.get(name, f"Config: {name}"),
    )


BUILDER_CONFIGS = [_make_config(name) for name in _ALL_CONFIGS]

DEFAULT_CONFIG_NAME = "all"


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


class SpeallMRI(datasets.GeneratorBasedBuilder):
    """Speall MRI Brain Dataset -- 1,105 studies / 34,574 series from India."""

    BUILDER_CONFIGS = BUILDER_CONFIGS
    DEFAULT_CONFIG_NAME = DEFAULT_CONFIG_NAME

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "study_id": datasets.Value("string"),
                "series_uid": datasets.Value("string"),
                "series_number": datasets.Value("int64"),
                "series_description": datasets.Value("string"),
                "sequence_type": datasets.Value("string"),
                "sequence_confidence": datasets.Value("string"),
                "modality": datasets.Value("string"),
                "file_count": datasets.Value("int64"),
                "tr_ms": datasets.Value("float64"),
                "te_ms": datasets.Value("float64"),
                "ti_ms": datasets.Value("float64"),
                "fa_deg": datasets.Value("float64"),
                "b_value": datasets.Value("float64"),
                "field_strength_T": datasets.Value("float64"),
                "plane": datasets.Value("string"),
                "volume_snr": datasets.Value("float64"),
                "volume_cnr": datasets.Value("float64"),
                "volume_entropy": datasets.Value("float64"),
                "quality_grade": datasets.Value("string"),
                "quality_score": datasets.Value("float64"),
                "ml_score": datasets.Value("float64"),
                "ml_grade": datasets.Value("string"),
                "commercial_tier": datasets.Value("string"),
                # Images -- datasets resolves file paths to PIL on access
                "multiplane_image": datasets.Image(),
                "histogram_image": datasets.Image(),
                "enhanced_image": datasets.Image(),
                # Full detail JSON serialised to string
                "detail_json": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=self.__class__.__doc__ or "",
            features=features,
            homepage="https://huggingface.co/datasets/shubhxho/speall-mri",
            license="MIT",
        )

    # ------------------------------------------------------------------
    # Split generators
    # ------------------------------------------------------------------

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        """Read split parquet shards from the dataset root."""
        data_dir = Path(dl_manager.manual_dir or self.config.data_dir or ".")

        # Try splits.parquet first; fall back to HF shard naming convention
        splits_parquet = data_dir / "splits.parquet"
        if not splits_parquet.exists():
            splits_parquet = None

        manifest_parquet = data_dir / "manifest.parquet"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "data_dir": str(data_dir),
                    "manifest_path": str(manifest_parquet),
                    "splits_path": str(splits_parquet) if splits_parquet else None,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "data_dir": str(data_dir),
                    "manifest_path": str(manifest_parquet),
                    "splits_path": str(splits_parquet) if splits_parquet else None,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                    "data_dir": str(data_dir),
                    "manifest_path": str(manifest_parquet),
                    "splits_path": str(splits_parquet) if splits_parquet else None,
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Example generator
    # ------------------------------------------------------------------

    def _generate_examples(
        self,
        split: str,
        data_dir: str,
        manifest_path: str,
        splits_path: str | None,
    ) -> Generator[tuple[int, dict[str, Any]], None, None]:
        """Yield one example per series row."""
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("pyarrow is required: pip install pyarrow") from exc

        root = Path(data_dir)

        # Load manifest
        manifest_table = pq.read_table(manifest_path)
        manifest_rows: list[dict[str, Any]] = manifest_table.to_pydict()
        n_rows = len(manifest_table)

        # Load split assignments
        split_series_uids: set[str] | None = None
        if splits_path and Path(splits_path).exists():
            splits_table = pq.read_table(splits_path)
            splits_dict = splits_table.to_pydict()
            split_col = splits_dict.get("split", [])
            uid_col = splits_dict.get("series_uid", [])
            split_series_uids = {
                uid for uid, s in zip(uid_col, split_col) if s == split
            }
        else:
            # No splits file: assign deterministically by index
            split_series_uids = None  # handled per-row below

        # Determine active filter
        config_name = self.config.name
        seq_filter: str | None = _SEQUENCE_CONFIGS.get(config_name)
        grade_filter: str | None = _GRADE_CONFIGS.get(config_name)

        idx = 0
        for row_i in range(n_rows):
            row: dict[str, Any] = {col: manifest_rows[col][row_i] for col in manifest_rows}

            series_uid: str = row.get("series_uid") or ""
            study_id: str = row.get("study_id") or ""

            # Apply split filter when no splits.parquet
            if split_series_uids is None:
                row_hash = abs(hash(series_uid)) % 10
                if split == "train" and row_hash >= 2:
                    pass  # keep
                elif split == "test" and row_hash == 0:
                    pass  # keep
                elif split == "validation" and row_hash == 1:
                    pass  # keep
                else:
                    continue
            elif series_uid not in split_series_uids:
                continue

            # Apply sequence / grade filter
            seq_type: str = (row.get("sequence_type") or "").upper()
            if seq_filter and seq_type != seq_filter.upper():
                continue

            if grade_filter:
                grade: str = row.get("quality_grade") or ""
                if grade.upper() != grade_filter.upper():
                    continue

            # Resolve image paths
            detail_path_rel: str = row.get("detail_path") or ""
            series_dir = root / study_id / Path(detail_path_rel).parent if detail_path_rel else root / study_id

            stem = Path(detail_path_rel).stem.replace("_detail", "") if detail_path_rel else series_uid
            multiplane = _find_image(series_dir, stem, ["_multiplane.png", "_montage.png"])
            histogram = _find_image(series_dir, stem, ["_histogram.png"])
            enhanced = _find_image(series_dir, stem, ["_enhanced.png"])

            # Load detail JSON
            detail_json_str = "{}"
            detail_json_path = (root / study_id / detail_path_rel) if detail_path_rel else None
            if detail_json_path and detail_json_path.exists():
                try:
                    detail_json_str = detail_json_path.read_text(encoding="utf-8")
                except OSError:
                    pass

            example: dict[str, Any] = {
                "study_id": study_id,
                "series_uid": series_uid,
                "series_number": row.get("series_number"),
                "series_description": row.get("series_description"),
                "sequence_type": row.get("sequence_type"),
                "sequence_confidence": row.get("sequence_confidence"),
                "modality": row.get("modality"),
                "file_count": row.get("file_count"),
                "tr_ms": row.get("tr_ms"),
                "te_ms": row.get("te_ms"),
                "ti_ms": row.get("ti_ms"),
                "fa_deg": row.get("fa_deg"),
                "b_value": row.get("b_value"),
                "field_strength_T": row.get("field_strength_T"),
                "plane": row.get("plane"),
                "volume_snr": row.get("volume_snr"),
                "volume_cnr": row.get("volume_cnr"),
                "volume_entropy": row.get("volume_entropy"),
                "quality_grade": row.get("quality_grade"),
                "quality_score": row.get("quality_score"),
                "ml_score": row.get("ml_score"),
                "ml_grade": row.get("ml_grade"),
                "commercial_tier": row.get("commercial_tier"),
                "multiplane_image": str(multiplane) if multiplane else None,
                "histogram_image": str(histogram) if histogram else None,
                "enhanced_image": str(enhanced) if enhanced else None,
                "detail_json": detail_json_str,
            }

            yield idx, example
            idx += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_image(directory: Path, stem: str, suffixes: list[str]) -> Path | None:
    """Locate the first existing image file matching stem + suffix."""
    for suffix in suffixes:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    # Fuzzy fallback: any PNG whose name contains the stem
    if directory.exists():
        for p in directory.glob("*.png"):
            for suffix in suffixes:
                if suffix.lstrip("_").split(".")[0] in p.name:
                    return p
    return None

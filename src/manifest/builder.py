"""
Manifest parquet generator for the Speall MRI pipeline.

Crawls a root directory for *_detail.json files (or any .json containing a
top-level 'series' dict), flattens each record into a series-level row, and
aggregates those rows into a study-level summary.  Writes manifest.parquet and
study_manifest.parquet to a specified output directory.

Usage:
    python -m src.manifest.builder --root <path> --out <path>
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

import polars as pl

from src.integrity.checksums import sha256_file
from src.io.msgspec_io import read_json
from src.manifest.confidence_summary import study_confidence_rollup

# Parallelize SHA-256 hashing once we have at least this many rows.
_PARALLEL_HASH_THRESHOLD = 1000
_HASH_WORKERS = 8

# ---------------------------------------------------------------------------
# Plane inference
# ---------------------------------------------------------------------------

_PLANE_TOKENS: list[tuple[str, str]] = [
    ("sag", "sagittal"),
    ("cor", "coronal"),
    ("ax", "axial"),
]


def _infer_plane(description: str | None) -> str | None:
    if not description:
        return None
    low = description.lower()
    for token, label in _PLANE_TOKENS:
        if token in low:
            return label
    return None


# ---------------------------------------------------------------------------
# Single-file parser
# ---------------------------------------------------------------------------


def _parse_detail(path: Path, root: Path) -> dict[str, Any] | None:
    """Parse one detail JSON into a flat dict suitable for a DataFrame row."""
    try:
        data = read_json(path)
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return None

    if not isinstance(data.get("series"), dict):
        return None

    series = data.get("series") or {}
    uid = series.get("uid")
    if not uid:
        print(f"WARNING: missing series.uid in {path}, skipping", file=sys.stderr)
        return None

    seq_cls = data.get("sequence_classification") or {}
    seq_params = data.get("sequence_params") or {}
    vs = data.get("volume_stats") or {}
    ml = data.get("ml_training_score") or {}

    # Determine study_id: prefer explicit key in JSON, fall back to path.
    # Production layout: <root>/<study_id>/<seq_subdir>/<series_id>/<series_id>_detail.json
    # Relative parts from root: [study_id, seq_subdir, series_id, filename] or fewer.
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = (path.stem,)

    path_study_id = rel_parts[0] if len(rel_parts) >= 2 else path.stem
    study_id = data.get("study_id") or series.get("study_id") or path_study_id

    # detail_path: study-relative path string
    try:
        detail_path = str(path.relative_to(root / study_id))
    except ValueError:
        detail_path = path.name

    # tar shard existence check
    tar_path = root / study_id / f"{study_id}.slices.tar"
    has_tar_shard = tar_path.exists()
    tar_shard_path = str(tar_path.relative_to(root)) if has_tar_shard else None

    volume_shape = vs.get("volume_shape")
    spacing_mm = vs.get("spacing_mm")
    fov_mm_val = vs.get("fov_mm")

    # sha256 is a manifest-level integrity hash for the series' detail.json.
    # The manifest carries one row per series (not per DICOM file), so we hash
    # the detail.json that *describes* the series. If a previous pipeline stage
    # already wrote a top-level "sha256" field in detail.json, reuse it (cache
    # hit); otherwise leave it None and let build_series_manifest compute it
    # in a parallel pass.
    cached_sha256 = data.get("sha256")
    if not (isinstance(cached_sha256, str) and len(cached_sha256) == 64):
        cached_sha256 = None

    return {
        "study_id": study_id,
        "series_uid": uid,
        "series_number": series.get("number"),
        "series_description": series.get("description"),
        "sequence_type": seq_cls.get("sequence_type"),
        "sequence_confidence": seq_cls.get("confidence"),
        "modality": series.get("modality"),
        "file_count": series.get("file_count"),
        "tr_ms": seq_params.get("tr"),
        "te_ms": seq_params.get("te"),
        "ti_ms": seq_params.get("ti"),
        "fa_deg": seq_params.get("fa"),
        "b_value": seq_params.get("b_value"),
        "field_strength_T": seq_params.get("field_strength"),
        "plane": _infer_plane(series.get("description")),
        "volume_shape": volume_shape,
        "spacing_mm": spacing_mm,
        "fov_mm": fov_mm_val,
        "volume_snr": vs.get("volume_snr_estimate"),
        "volume_cnr": vs.get("volume_cnr"),
        "volume_entropy": vs.get("volume_entropy"),
        "quality_grade": vs.get("quality_grade"),
        "quality_score": vs.get("quality_score"),
        "ml_score": ml.get("score") if ml else None,
        "ml_grade": ml.get("grade") if ml else None,
        "commercial_tier": ml.get("commercial_tier") if ml else None,
        "detail_path": detail_path,
        "montage_path": data.get("montage_path"),
        "has_tar_shard": has_tar_shard,
        "sha256": cached_sha256,
        "_tar_shard_path": tar_shard_path,
        "_abs_detail_path": str(path),
    }


# ---------------------------------------------------------------------------
# Explicit schema for stable column types across sparse datasets
# ---------------------------------------------------------------------------

_SERIES_SCHEMA: dict[str, PolarsDataType] = {
    "study_id": pl.Utf8,
    "series_uid": pl.Utf8,
    "series_number": pl.Int64,
    "series_description": pl.Utf8,
    "sequence_type": pl.Utf8,
    "sequence_confidence": pl.Utf8,
    "modality": pl.Utf8,
    "file_count": pl.Int64,
    "tr_ms": pl.Float64,
    "te_ms": pl.Float64,
    "ti_ms": pl.Float64,
    "fa_deg": pl.Float64,
    "b_value": pl.Float64,
    "field_strength_T": pl.Float64,
    "plane": pl.Utf8,
    "volume_shape": pl.List(pl.Int64),
    "spacing_mm": pl.List(pl.Float64),
    "fov_mm": pl.List(pl.Float64),
    "volume_snr": pl.Float64,
    "volume_cnr": pl.Float64,
    "volume_entropy": pl.Float64,
    "quality_grade": pl.Utf8,
    "quality_score": pl.Float64,
    "ml_score": pl.Float64,
    "ml_grade": pl.Utf8,
    "commercial_tier": pl.Utf8,
    "detail_path": pl.Utf8,
    "montage_path": pl.Utf8,
    "has_tar_shard": pl.Boolean,
    # SHA-256 hex digest (lower-case) of the per-series detail.json file.
    # Fixed 64 chars when present; nullable for backwards compatibility.
    "sha256": pl.Utf8,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_series_manifest(root: Path) -> pl.DataFrame:
    """Crawl root for detail JSON files; return one row per series.

    For each row, attaches a ``sha256`` hex digest of the underlying
    detail.json file. If a previous stage embedded a ``sha256`` field in the
    JSON it is reused (cache hit); otherwise it is computed here.  Hashing is
    parallelised with a thread-pool when the row count exceeds
    ``_PARALLEL_HASH_THRESHOLD`` (file I/O is the bottleneck).
    """
    root = Path(root)
    parsed: list[dict[str, Any]] = []

    for json_path in sorted(root.rglob("*.json")):
        record = _parse_detail(json_path, root)
        if record is None:
            continue
        parsed.append(record)

    # Resolve sha256 column: cache hits stay as-is, misses get hashed.
    # Hashing is bound by file I/O, so a thread-pool gives a real speed-up.
    needs_hash: list[int] = [
        i for i, rec in enumerate(parsed) if rec.get("sha256") is None
    ]
    if needs_hash:
        paths = [Path(parsed[i]["_abs_detail_path"]) for i in needs_hash]
        if len(parsed) >= _PARALLEL_HASH_THRESHOLD:
            with ThreadPoolExecutor(max_workers=_HASH_WORKERS) as pool:
                digests = list(pool.map(sha256_file, paths))
        else:
            digests = [sha256_file(p) for p in paths]
        for idx, digest in zip(needs_hash, digests, strict=True):
            parsed[idx]["sha256"] = digest

    rows: list[dict[str, Any]] = []
    for record in parsed:
        record.pop("_tar_shard_path", None)
        record.pop("_abs_detail_path", None)
        rows.append({col: record.get(col) for col in _SERIES_SCHEMA})

    if not rows:
        return pl.DataFrame(schema=_SERIES_SCHEMA)

    return pl.DataFrame(rows, schema=_SERIES_SCHEMA)


def build_study_manifest(series_df: pl.DataFrame, root: Path) -> pl.DataFrame:
    """Aggregate series_df into one row per study."""
    root = Path(root)

    _SEQ_FLAGS = {
        "has_dwi": lambda s: bool(s and "dwi" in s.lower()),
        "has_flair": lambda s: bool(s and "flair" in s.lower()),
        "has_swan": lambda s: bool(s and ("swan" in s.lower() or "swi" in s.lower())),
        "has_tof": lambda s: bool(s and "tof" in s.lower()),
        "has_t1": lambda s: bool(s and "t1" in s.lower()),
        "has_t2": lambda s: bool(s and "t2" in s.lower()),
    }

    study_rows: list[dict[str, Any]] = []

    for study_id, group in series_df.group_by("study_id"):
        sid = study_id[0] if isinstance(study_id, tuple) else study_id
        seq_types = group["sequence_type"].drop_nulls().to_list()
        file_counts = group["file_count"].drop_nulls().cast(pl.Int64).to_list()
        ml_scores = group["ml_score"].drop_nulls().to_list()
        grades = group["quality_grade"].drop_nulls().to_list()

        # dominant grade: most frequent non-null grade
        dominant_grade: str | None = None
        if grades:
            dominant_grade = max(set(grades), key=grades.count)

        tar_path = root / sid / f"{sid}.slices.tar"
        has_tar = tar_path.exists()
        tar_shard_path_str = str(tar_path.relative_to(root)) if has_tar else None

        size_mb: float | None = None
        if has_tar:
            try:
                size_mb = os.path.getsize(tar_path) / (1024 * 1024)
            except OSError:
                size_mb = None

        # Confidence rollup from per-series annotation consensus blocks
        _annotations_dir = root / sid / "annotations"
        _rollup = study_confidence_rollup(_annotations_dir)

        row: dict[str, Any] = {
            "study_id": sid,
            "n_series": len(group),
            "sequences_present": sorted(set(seq_types)),
            "total_files": sum(file_counts) if file_counts else None,
            "dominant_grade": dominant_grade,
            "mean_ml_score": (sum(ml_scores) / len(ml_scores)) if ml_scores else None,
            "has_dwi": any(_SEQ_FLAGS["has_dwi"](s) for s in seq_types),
            "has_flair": any(_SEQ_FLAGS["has_flair"](s) for s in seq_types),
            "has_swan": any(_SEQ_FLAGS["has_swan"](s) for s in seq_types),
            "has_tof": any(_SEQ_FLAGS["has_tof"](s) for s in seq_types),
            "has_t1": any(_SEQ_FLAGS["has_t1"](s) for s in seq_types),
            "has_t2": any(_SEQ_FLAGS["has_t2"](s) for s in seq_types),
            "total_size_mb": size_mb,
            "tar_shard_path": tar_shard_path_str,
            # confidence rollup
            "confidence_n_series": _rollup["n_series"],
            "confidence_mean": _rollup["mean_confidence"],
            "confidence_min": _rollup["min_confidence"],
            "confidence_pct_low": _rollup["pct_low_confidence"],
            "confidence_n_needs_escalation": _rollup["n_needs_escalation"],
            "confidence_pct_premium_used": _rollup["pct_premium_used"],
        }
        study_rows.append(row)

    _STUDY_SCHEMA: dict[str, PolarsDataType] = {
        "study_id": pl.Utf8,
        "n_series": pl.Int64,
        "sequences_present": pl.List(pl.Utf8),
        "total_files": pl.Int64,
        "dominant_grade": pl.Utf8,
        "mean_ml_score": pl.Float64,
        "has_dwi": pl.Boolean,
        "has_flair": pl.Boolean,
        "has_swan": pl.Boolean,
        "has_tof": pl.Boolean,
        "has_t1": pl.Boolean,
        "has_t2": pl.Boolean,
        "total_size_mb": pl.Float64,
        "tar_shard_path": pl.Utf8,
        # confidence rollup columns (additive)
        "confidence_n_series": pl.Int64,
        "confidence_mean": pl.Float64,
        "confidence_min": pl.Float64,
        "confidence_pct_low": pl.Float64,
        "confidence_n_needs_escalation": pl.Int64,
        "confidence_pct_premium_used": pl.Float64,
    }

    if not study_rows:
        return pl.DataFrame(schema=_STUDY_SCHEMA)

    return pl.DataFrame(study_rows, schema=_STUDY_SCHEMA)


def write_manifests(
    root: Path,
    out_dir: Path,
    with_splits: bool = False,
) -> dict[str, int]:
    """Build both manifests and write parquet files to out_dir.

    Parameters
    ----------
    root:
        Root directory to crawl for detail JSON files.
    out_dir:
        Output directory for parquet files.
    with_splits:
        When True, calls :func:`src.manifest.splits.assign_splits` after
        building the manifests and writes ``splits.parquet`` (one row per
        study: study_id, split) alongside the other outputs.
    """
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series_df = build_series_manifest(root)
    study_df = build_study_manifest(series_df, root)

    series_df.write_parquet(out_dir / "manifest.parquet")
    study_df.write_parquet(out_dir / "study_manifest.parquet")

    counts: dict[str, int] = {
        "series_rows": len(series_df),
        "study_rows": len(study_df),
    }

    if with_splits:
        from src.manifest.splits import assign_splits  # lazy import

        _, study_df_split = assign_splits(series_df, study_df)
        splits_df = study_df_split.select(["study_id", "split"])
        splits_df.write_parquet(out_dir / "splits.parquet")
        counts["splits_rows"] = len(splits_df)

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manifest.parquet and study_manifest.parquet for Speall MRI data."
    )
    parser.add_argument("--root", required=True, type=Path, help="Root directory to crawl")
    parser.add_argument(
        "--out", required=True, type=Path, help="Output directory for parquet files"
    )
    parser.add_argument(
        "--with-splits",
        action="store_true",
        default=False,
        help=(
            "Assign patient-level train/val/test splits after building manifests "
            "and write splits.parquet (study_id, split) to --out."
        ),
    )
    args = parser.parse_args()

    counts = write_manifests(args.root, args.out, with_splits=args.with_splits)
    print(f"Wrote {counts['series_rows']} series rows -> {args.out / 'manifest.parquet'}")
    print(f"Wrote {counts['study_rows']} study rows  -> {args.out / 'study_manifest.parquet'}")
    if args.with_splits:
        print(f"Wrote {counts['splits_rows']} split rows  -> {args.out / 'splits.parquet'}")


if __name__ == "__main__":
    main()

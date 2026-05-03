"""Dry-run planner: scan output_dir and estimate what a pipeline run would cost.

Call ``plan(output_dir)`` before spawning Modal jobs to catch "you're about to
spend $50" surprises.

Cost model (documented inline below):
  - Modal CPU: $0.000111 / second / CPU  (published rate as of 2025)
  - Stage 2 quality_one_series: ~30 s @ 2 CPU per series  → $0.00666 / series
  - Stage 3 annotate_one:       ~15 s @ 1 CPU per series  → $0.00167 / series
  - OpenRouter Gemma 4 31B: ~$0.30 / 1M input tokens, ~$0.50 / 1M output tokens
    Each annotation call uses ~3 000 input + 2 000 output tokens per series
    → (3000 * 0.30 + 2000 * 0.50) / 1_000_000 = $0.0019 / series

Wall-time estimate assumes:
  - Stage 2: ~50 concurrent containers  → wall = ceil(n / 50) × 30 s
  - Stage 3: ~10 workers × 32 inputs each = 320 concurrent  → ceil(n / 320) × 15 s
  - Stage 4: ~20 concurrent, 60 s / study  → ceil(n / 20) × 60 s
  These are rough; real wall time depends on cold-start and queue depth.

Pending detection mirrors resume_pipeline.py logic exactly:
  Stage 2 pending : detail.json exists AND ``detail["advanced_quality"]`` is falsy
  Stage 3 pending : *_multiplane.png exists, no matching annotation JSON, not a derivative
  Stage 4 pending : study dir has ``slices/`` subdir AND no non-empty ``<study>.slices.tar``
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

# ── Mirrors resume_pipeline._SAFE_NAME_RE / _safe_name ───────────────────────
# Source of truth: resume_pipeline.py (inline copy to avoid importing Modal module)
_SAFE_NAME_RE = re.compile(r"[^\w\-]")


def _safe_name(label: str) -> str:
    return _SAFE_NAME_RE.sub("_", label)


# ── Mirrors src.ai_analysis._DERIVATIVE_RE / _is_derivative ──────────────────
# Source of truth: src/ai_analysis.py (inline copy to keep this module dep-free)
_DERIVATIVE_RE = re.compile(
    r"\b(d{0,2}REG\b|d{0,2}ADC\w*|d?ISO\w*|ISOTROPIC|FILT_PHA|COL:|PJN:|MIP|MinIP"
    r"|REFORMATT?ED|SUBTRACT)",
    re.IGNORECASE,
)


def _is_derivative(label: str) -> bool:
    return bool(_DERIVATIVE_RE.search(label))


# ── Cost constants ────────────────────────────────────────────────────────────

# Modal CPU rate: $0.000111 / sec / CPU
_MODAL_CPU_RATE = 0.000111

# Stage 2: quality_one_series -- 30 s @ 2 CPU
_QUALITY_SECS = 30.0
_QUALITY_CPU = 2.0
_QUALITY_COST_PER_SERIES = _QUALITY_SECS * _QUALITY_CPU * _MODAL_CPU_RATE  # ~$0.00666

# Stage 3: annotate_one -- 15 s @ 1 CPU
_ANNOTATE_SECS = 15.0
_ANNOTATE_CPU = 1.0
_ANNOTATE_COST_PER_SERIES = _ANNOTATE_SECS * _ANNOTATE_CPU * _MODAL_CPU_RATE  # ~$0.00167

# OpenRouter Gemma 4 31B token pricing
_OR_INPUT_PER_TOK = 0.30 / 1_000_000   # $0.30 / 1M input tokens
_OR_OUTPUT_PER_TOK = 0.50 / 1_000_000  # $0.50 / 1M output tokens
_OR_INPUT_TOKS = 3_000                  # estimated input tokens per series
_OR_OUTPUT_TOKS = 2_000                 # estimated output tokens per series
_OR_COST_PER_SERIES = (
    _OR_INPUT_TOKS * _OR_INPUT_PER_TOK + _OR_OUTPUT_TOKS * _OR_OUTPUT_PER_TOK
)  # ~$0.0019

# Wall-time concurrency assumptions (rough)
_QUALITY_CONCURRENCY = 50    # ~50 Modal containers for Stage 2
_ANNOTATE_CONCURRENCY = 320  # 10 workers × 32 inputs/worker for Stage 3
_PACK_CONCURRENCY = 20       # ~20 containers for Stage 4
_PACK_SECS_PER_STUDY = 60.0  # ~60 s / study for tarring


def _count_quality_pending(all_details: list[Path]) -> int:
    """Series without advanced_quality (same pre-filter as resume_pipeline Stage 2)."""
    pending = 0
    for dp in all_details:
        try:
            data = json.loads(dp.read_text())
            if not data.get("advanced_quality"):
                pending += 1
        except Exception:
            pending += 1  # unreadable → treat as pending
    return pending


def _count_annotation_pending(output_dir: Path, all_montages: list[Path]) -> int:
    """Montages without an annotation JSON that are not derivatives."""
    ann_dir = output_dir / "annotations"
    existing_anns: set[str] = set()
    if ann_dir.is_dir():
        for f in ann_dir.glob("*.json"):
            if f.name != "study_annotations.json":
                existing_anns.add(f.stem)

    pending = 0
    for montage in all_montages:
        series_dir = montage.parent
        series_name = series_dir.name
        parts = series_name.split("_", 1)
        snum = parts[0].replace("s", "") if parts else "?"
        sdesc = parts[1] if len(parts) > 1 else ""
        label = f"Series {snum} \u2014 {sdesc}"

        if _is_derivative(label):
            continue
        if _safe_name(label) in existing_anns:
            continue
        pending += 1

    return pending


def _count_pack_pending(study_dirs: list[Path]) -> int:
    """Studies with a slices/ dir but no valid tar shard."""
    pending = 0
    for study_dir in study_dirs:
        slices_root = study_dir / "slices"
        if not slices_root.is_dir():
            continue
        tar_path = study_dir / f"{study_dir.name}.slices.tar"
        if tar_path.exists() and tar_path.stat().st_size > 0:
            continue
        pending += 1
    return pending


def _wall_minutes(n: int, secs_each: float, concurrency: int) -> float:
    if n == 0:
        return 0.0
    return math.ceil(n / concurrency) * secs_each / 60.0


def plan(output_dir: Path) -> dict:
    """Scan output_dir and return a cost/work estimate dict.

    Returns
    -------
    dict with keys:
      n_studies, n_series_total,
      n_series_quality_pending, n_series_annotation_pending,
      n_studies_pack_pending,
      est_modal_dollars, est_openrouter_dollars, est_wall_minutes
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return {
            "n_studies": 0,
            "n_series_total": 0,
            "n_series_quality_pending": 0,
            "n_series_annotation_pending": 0,
            "n_studies_pack_pending": 0,
            "est_modal_dollars": 0.0,
            "est_openrouter_dollars": 0.0,
            "est_wall_minutes": 0.0,
        }

    # Studies: direct non-hidden children that are directories
    study_dirs = [
        p for p in output_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "_internal"
    ]
    n_studies = len(study_dirs)

    all_details = list(output_dir.rglob("*_detail.json"))
    n_series_total = len(all_details)

    all_montages = list(output_dir.rglob("*_multiplane.png"))

    n_quality_pending = _count_quality_pending(all_details)
    n_annotation_pending = _count_annotation_pending(output_dir, all_montages)
    n_pack_pending = _count_pack_pending(study_dirs)

    # Cost estimates
    modal_quality = n_quality_pending * _QUALITY_COST_PER_SERIES
    modal_annotate = n_annotation_pending * _ANNOTATE_COST_PER_SERIES
    est_modal = modal_quality + modal_annotate

    est_openrouter = n_annotation_pending * _OR_COST_PER_SERIES

    # Wall time: stages run sequentially in the pipeline
    wall_quality = _wall_minutes(n_quality_pending, _QUALITY_SECS, _QUALITY_CONCURRENCY)
    wall_annotate = _wall_minutes(n_annotation_pending, _ANNOTATE_SECS, _ANNOTATE_CONCURRENCY)
    wall_pack = _wall_minutes(n_pack_pending, _PACK_SECS_PER_STUDY, _PACK_CONCURRENCY)
    est_wall = wall_quality + wall_annotate + wall_pack

    return {
        "n_studies": n_studies,
        "n_series_total": n_series_total,
        "n_series_quality_pending": n_quality_pending,
        "n_series_annotation_pending": n_annotation_pending,
        "n_studies_pack_pending": n_pack_pending,
        "est_modal_dollars": round(est_modal, 4),
        "est_openrouter_dollars": round(est_openrouter, 4),
        "est_wall_minutes": round(est_wall, 1),
    }

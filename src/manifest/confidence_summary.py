"""
Per-study confidence rollup from per-series annotation consensus blocks.

Each annotated series is expected to have a JSON file under
``<annotations_dir>/<series_label>.json`` whose top-level ``consensus`` key
holds at minimum:

    {
        "confidence": float,          # [0, 1]
        "needs_escalation": bool,
        "tiers_used": list[str],
        "premium_used": bool,
    }

The public function :func:`study_confidence_rollup` reads every JSON file in
*annotations_dir*, extracts the ``consensus`` block, and returns a study-level
summary dict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_LOW_CONFIDENCE_THRESHOLD = 0.6


def _parse_consensus(path: Path) -> dict[str, Any] | None:
    """Return the consensus block from one annotation JSON, or None on failure."""
    try:
        with path.open() as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"WARNING: could not read annotation {path}: {exc}", file=sys.stderr)
        return None

    consensus = data.get("consensus")
    if not isinstance(consensus, dict):
        return None
    return consensus


def study_confidence_rollup(annotations_dir: Path) -> dict[str, Any]:
    """Aggregate per-series consensus into per-study rollup.

    Parameters
    ----------
    annotations_dir:
        Directory containing per-series JSON files with a ``consensus`` block.

    Returns
    -------
    dict with keys:
        n_series, mean_confidence, min_confidence, pct_low_confidence,
        n_needs_escalation, pct_premium_used, series_breakdown.
    """
    annotations_dir = Path(annotations_dir)

    _SAFE_DEFAULTS: dict[str, Any] = {
        "n_series": 0,
        "mean_confidence": 0.0,
        "min_confidence": 0.0,
        "pct_low_confidence": 0.0,
        "n_needs_escalation": 0,
        "pct_premium_used": 0.0,
        "series_breakdown": [],
    }

    if not annotations_dir.exists() or not annotations_dir.is_dir():
        return _SAFE_DEFAULTS.copy()

    series_breakdown: list[dict[str, Any]] = []
    for json_path in sorted(annotations_dir.glob("*.json")):
        consensus = _parse_consensus(json_path)
        if consensus is None:
            continue
        confidence = consensus.get("confidence")
        if not isinstance(confidence, (int, float)):
            continue
        series_breakdown.append(
            {
                "series_label": json_path.stem,
                "confidence": float(confidence),
                "needs_escalation": bool(consensus.get("needs_escalation", False)),
                "tiers_used": consensus.get("tiers_used", []),
                "premium_used": bool(consensus.get("premium_used", False)),
            }
        )

    n = len(series_breakdown)
    if n == 0:
        return _SAFE_DEFAULTS.copy()

    confidences = [s["confidence"] for s in series_breakdown]
    n_low = sum(1 for c in confidences if c < _LOW_CONFIDENCE_THRESHOLD)
    n_needs_escalation = sum(1 for s in series_breakdown if s["needs_escalation"])
    n_premium = sum(1 for s in series_breakdown if s["premium_used"])

    return {
        "n_series": n,
        "mean_confidence": sum(confidences) / n,
        "min_confidence": min(confidences),
        "pct_low_confidence": n_low / n,
        "n_needs_escalation": n_needs_escalation,
        "pct_premium_used": n_premium / n,
        "series_breakdown": series_breakdown,
    }

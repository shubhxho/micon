"""Multi-model consensus building and per-series fan-out.

After ``cloud.annotate_with_model`` produces N independent annotations of
the same montage, this module reduces them to a single consensus block --
majority-vote sequence type, union of anatomical structures, OR-fold for
pathology, average quality grade, and a list of inter-model disagreements
the human reviewer should look at.

Two derived signals support downstream cost optimization:
  - ``confidence`` -- blended [0, 1] score; lets buyers slice a "high-trust"
    SKU without rerunning models.
  - ``needs_escalation`` -- True when the cheap-tier models disagree enough
    that calling the premium tier on this specific series would actually
    pay off. Callers can then run a second-pass premium annotation only
    on the small subset of escalations.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .call_model import (
    MODELS,
    _client,
    _default_lineup,
    _detect_provider,
    _filter_supported,
)


def annotate_series_multi(
    montage_path: str,
    series_label: str,
    quality_ctx: str = "",
    models: list[str] | None = None,
) -> dict:
    """Run annotation across all models in parallel. Returns consensus + per-model results."""
    # Lazy import to avoid a cloud<->consensus circular at module load.
    from .cloud import annotate_with_model

    provider = _detect_provider()
    if provider is None:
        return {"error": "No API key set (OPENROUTER_API_KEY or OPENAI_API_KEY)"}

    client = _client()
    model_keys = _filter_supported(models or _default_lineup(), provider)
    if not model_keys:
        return {"error": f"No supported models on provider={provider}"}

    results = {}
    with ThreadPoolExecutor(max_workers=len(model_keys)) as pool:
        futures = {
            pool.submit(
                annotate_with_model, client, mk, montage_path, series_label, quality_ctx, provider
            ): mk
            for mk in model_keys
        }
        for fut in as_completed(futures):
            mk = futures[fut]
            results[mk] = fut.result()

    # Build consensus from successful annotations
    successful = {k: v for k, v in results.items() if v.get("annotation")}

    consensus = _build_consensus(successful)

    return {
        "series_label": series_label,
        "models_called": len(model_keys),
        "models_succeeded": len(successful),
        "per_model": results,
        "consensus": consensus,
    }


def _build_consensus(results: dict[str, dict]) -> dict:
    """Merge annotations from multiple models into a consensus."""
    if not results:
        return {"error": "no successful annotations"}

    annotations = {k: v["annotation"] for k, v in results.items()}

    # Sequence type -- majority vote
    seq_types = [a.get("sequence_type", "Unknown") for a in annotations.values()]
    seq_type = max(set(seq_types), key=seq_types.count)
    seq_agreement = seq_types.count(seq_type) / len(seq_types)

    # Pathology -- any model finding pathology flags it
    pathology_found = any(a.get("pathology", {}).get("found", False) for a in annotations.values())
    # Findings may be either strings (legacy/relaxed) or dicts (rich schema).
    # Normalize to a string key for dedup; preserve the original payload.
    all_findings = []
    seen_keys = set()
    all_differentials = []
    for a in annotations.values():
        path = a.get("pathology", {})
        for f in path.get("findings", []):
            if isinstance(f, dict):
                key = f"{f.get('location', '?')}|{f.get('signal_on_this_sequence', '')}|{f.get('size_mm', '')}"
            else:
                key = str(f)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_findings.append(f)
        all_differentials.extend(path.get("differential", []))
    unique_findings = all_findings
    unique_differentials = list(
        dict.fromkeys(
            d if isinstance(d, str) else json.dumps(d, sort_keys=True) for d in all_differentials
        )
    )

    # Quality -- average grades
    grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
    grades = []
    for a in annotations.values():
        g = a.get("quality", {}).get("grade", "")
        if g in grade_map:
            grades.append(grade_map[g])
    avg_grade = sum(grades) / max(len(grades), 1)
    reverse_map = {4: "A", 3: "B", 2: "C", 1: "D", 0: "F"}
    consensus_grade = reverse_map.get(round(avg_grade), "?")

    # Anatomical structures -- union (rich schema: anatomical_coverage.structures_visualized;
    # legacy: anatomical_structures)
    all_structures = []
    for a in annotations.values():
        coverage = a.get("anatomical_coverage", {})
        if isinstance(coverage, dict):
            all_structures.extend(coverage.get("structures_visualized", []))
        all_structures.extend(a.get("anatomical_structures", []))
    unique_structures = list(dict.fromkeys(all_structures))

    # Notable -- union
    all_notable = []
    for a in annotations.values():
        all_notable.extend(a.get("notable", []))
    unique_notable = list(dict.fromkeys(all_notable))

    # Disagreements
    disagreements = []
    if seq_agreement < 1.0:
        disagreements.append(
            {
                "field": "sequence_type",
                "votes": {st: seq_types.count(st) for st in set(seq_types)},
            }
        )

    pathology_votes = {
        k: v["annotation"].get("pathology", {}).get("found", False)
        for k, v in results.items()
        if v.get("annotation")
    }
    if len(set(pathology_votes.values())) > 1:
        disagreements.append(
            {
                "field": "pathology_found",
                "votes": {MODELS[k]["name"]: v for k, v in pathology_votes.items()},
                "flag": "NEEDS HUMAN REVIEW -- models disagree on pathology",
            }
        )

    return {
        "sequence_type": seq_type,
        "sequence_agreement": round(seq_agreement, 2),
        "anatomical_structures": unique_structures[:20],
        "pathology": {
            "found": pathology_found,
            "findings": unique_findings[:10],
            "differential": unique_differentials[:5],
            "models_agreeing": sum(1 for v in pathology_votes.values() if v == pathology_found),
            "models_total": len(pathology_votes),
        },
        "quality_grade": consensus_grade,
        "notable": unique_notable[:10],
        "disagreements": disagreements,
        "models_used": [MODELS[k]["name"] for k in results],
        # Cost-tier telemetry: lets the manifest filter on cheap-only runs
        # vs runs that escalated to premium models.
        "tiers_used": sorted({MODELS[k].get("tier", "unknown") for k in results}),
        "premium_used": any(MODELS[k].get("tier") == "premium" for k in results),
        # A2: per-series confidence -- weighted blend of sequence agreement
        # and pathology agreement. Buyers use this to filter a "high-trust"
        # subset for premium SKUs without rerunning anything.
        "confidence": _series_confidence(seq_agreement, pathology_votes),
        # A1: escalation hint. True when the cheap tier disagrees enough
        # that calling premium models on this series would actually pay
        # off. Callers run premium-tier annotation only on these.
        "needs_escalation": _should_escalate(seq_agreement, pathology_votes),
    }


def _series_confidence(seq_agreement: float, pathology_votes: dict[str, bool]) -> float:
    """Blend sequence + pathology agreement into [0, 1] confidence."""
    if not pathology_votes:
        return round(seq_agreement, 3)
    n = len(pathology_votes)
    pos = sum(1 for v in pathology_votes.values() if v)
    pathology_agreement = max(pos, n - pos) / n
    return round(0.6 * seq_agreement + 0.4 * pathology_agreement, 3)


def _should_escalate(seq_agreement: float, pathology_votes: dict[str, bool]) -> bool:
    """Cheap-tier disagreement threshold for escalation to premium models.

    Escalate when sequence is split (<=50%) or pathology is genuinely
    contested (closer than 2/3 majority). Both signals are cheap to
    compute from the existing consensus block.
    """
    if seq_agreement <= 0.5:
        return True
    if pathology_votes:
        n = len(pathology_votes)
        pos = sum(1 for v in pathology_votes.values() if v)
        majority = max(pos, n - pos)
        if majority / n < 0.67:
            return True
    return False


__all__ = [
    "_build_consensus",
    "_series_confidence",
    "_should_escalate",
    "annotate_series_multi",
]

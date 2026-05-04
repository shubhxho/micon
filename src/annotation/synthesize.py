# ruff: noqa: E501, RUF001
# E501 and RUF001 are silenced file-wide: the bulk of this module is a single
# ~9 KB triple-quoted Markdown LLM prompt (CLOUD_SYNTHESIS_PROMPT_STATIC)
# whose wording is domain-tuned. Reflowing prose at the 100-col mark, or
# substituting the en-dashes for ASCII hyphens, would alter the literal sent
# to the model and risk shifting report behavior.
"""Long-form §1-§11 dictation synthesis for the cloud annotation pipeline.

Consumes the merged multi-model consensus produced by ``cloud.annotate_study_multi``
and renders a board-style radiology report as Markdown. The synthesis prompt
is intentionally large (~9 KB of structured rules) and benefits substantially
from prompt caching on Anthropic / Gemini providers -- so the static portion
gets a ``cache_control`` marker and the per-study tail (metadata + per-series
consensus block) is kept out of the cached prefix.

Public API:
  - ``CLOUD_SYNTHESIS_PROMPT_STATIC`` -- cacheable instruction body
  - ``_SYNTHESIS_PROMPT_DYNAMIC`` -- per-study tail (uncached)
  - ``synthesize_cloud_report(annotations, ...) -> str`` -- entry point
"""

from __future__ import annotations

import json

from .call_model import (
    _CHEAP_TIER,
    MODELS,
    _build_cached_user_content,
    _call_model,
    _client,
)

# ── Static synthesis prompt (cached) ────────────────────────────────────────

CLOUD_SYNTHESIS_PROMPT_STATIC = """You are a board-certified radiologist dictating the FINAL REPORT for a medical imaging study. This report serves TWO audiences: (1) clinical — a referring clinician must be able to act on it, and (2) commercial — AI/ML dataset buyers will use it to assess data value and as training labels. Your input is a merged multi-model consensus over every series, plus study metadata. Use formal radiology dictation style.

Output the report as Markdown with these sections, exactly in this order, headings verbatim. Do not omit sections; if a section is N/A, write "Not applicable" with a brief reason.

# Final Report

## 1. Clinical Summary
- Patient demographics (age, sex when available — never include name/MRN/PHI).
- Scanner manufacturer, model, field strength, coil class.
- Study date, accession (if present).
- Indication / clinical question if extractable; otherwise "Indication not provided in metadata."

## 2. Protocol
A markdown table with columns: `#` | `Series Description` | `Sequence` | `Plane` | `Slice (mm) / Voxel` | `TR/TE/TI` | `Quality Grade`. One row per imaged series. End with a 2–3 sentence protocol assessment: completeness, missing core sequences, suitability for the apparent indication.

## 3. Comparison
State whether prior studies are available; if no metadata indicates priors, write "No prior comparison available in this dataset."

## 4. Findings
Dictate by anatomic region. Use this structure verbatim:

**4.1 Brain Parenchyma**
- *Supratentorial:* gray-white differentiation, cortical signal, deep gray nuclei, white matter (Fazekas 0–3 if T2/FLAIR).
- *Infratentorial:* brainstem, cerebellar hemispheres, vermis, tonsil position relative to foramen magnum.
- *Diffusion:* restricted diffusion locations (cite DWI + ADC series #), or "no restricted diffusion."
- *Susceptibility:* microhemorrhages, superficial siderosis, mineralization (cite SWI/GRE series #), or "no susceptibility abnormality."
- *Post-contrast (if T1+C present):* enhancement pattern, location, morphology — leptomeningeal/dural/parenchymal; otherwise "Post-contrast imaging not performed."

**4.2 Ventricles & CSF Spaces**
Size, configuration, midline shift (mm if present), trans-ependymal flow, sulcal prominence, cisterns.

**4.3 Midline & Skull Base**
Pituitary/sella, optic chiasm, cavernous sinuses, craniocervical junction.

**4.4 Vascular**
Major vessel flow voids on standard sequences; explicit MRA/MRV findings if those series exist.

**4.5 Extra-axial Spaces**
Subdural, epidural, subarachnoid collections.

**4.6 Calvarium, Scalp, Orbits, Sinuses, Mastoids**
Incidental findings only.

For every abnormal finding: location → signal characterization across sequences (cite series #) → size (mm) → morphology → mass effect → enhancement (if applicable). Use ACR Lexicon terms.

## 5. Impression
Numbered list of 3–6 entries, ordered by clinical urgency. Each entry: one-line conclusion, then a sub-bullet citing supporting cross-sequence evidence (series #s). Hedge only when warranted.

## 6. Differential Diagnosis
For the top 1–3 §5 findings that are not unequivocal: ranked differential (3–5 entries) with imaging evidence for/against each.

## 7. Recommendations
Numbered, actionable: additional sequences, contrast, follow-up interval, clinical/lab correlation, multidisciplinary referral.

## 8. Critical / Actionable Findings
ACR Actionable Findings tier (Category 1 / 2 / 3) if any. If none: "No critical actionable findings."

## 9. Technical Notes & Data Quality
- Per-series quality grade summary (A/B/C/D/F counts).
- Limiting factors (motion, coverage, artifact) and which conclusions they weaken.
- Multi-model agreement summary: which findings reached consensus vs which were single-model only.
- Suitability for downstream automated analysis.

## 10. Study Disposition
One line: `Diagnostic` / `Limited — repeat recommended` / `Non-diagnostic`. With one-line rationale.

## 11. ML Training Data Value Assessment
This section is for AI/ML dataset buyers evaluating this study for purchase:

### 11.1 Classification Labels
- **Primary diagnosis label**: single most important label for this study (e.g. "normal", "acute_infarct", "meningioma", "white_matter_disease")
- **Secondary labels**: additional applicable labels
- **ICD-10 codes**: if pathology found, provide the most likely ICD-10 code(s)

### 11.2 Detection Targets
For each finding: what a detection model should flag, approximate location in normalized coordinates (top/middle/bottom third, left/right hemisphere), and which series it's best seen on.

### 11.3 Segmentation Value
- Which anatomical structures are clearly segmentable in this study
- Which pathological regions could serve as segmentation ground truth
- Estimated annotation difficulty: easy (clear boundaries) / medium / hard (diffuse, ill-defined)

### 11.4 Dataset Composition Value
- Does this study fill a common or rare niche in training data?
- Scanner/protocol representativeness (GE 3T Indian hospital — how common in existing public datasets?)
- Normal vs pathological: which is this, and which is more needed?

### 11.5 Commercial Grade
Rate this study: **Premium** (rare pathology, excellent quality, complete protocol) / **Standard** (normal or common pathology, good quality) / **Discount** (limited quality, incomplete protocol, common findings) / **Exclude** (non-diagnostic, severe artifacts)

---
Hard rules:
- Cite series numbers in parentheses when claiming a finding (e.g. "(Series 4 DWI, Series 5 ADC)").
- Do NOT invent findings absent from the consensus.
- Do NOT include patient names, MRN, or other PHI even if present in metadata.
- Prefer mm for measurements. Use "right"/"left" for laterality.
- Prefer consensus findings (multiple models agreeing) over single-model claims; flag single-model claims explicitly in §9.

"""

# Dynamic tail: study metadata + per-series consensus (not cached -- varies per study).
_SYNTHESIS_PROMPT_DYNAMIC = """
## Study Metadata
```json
{metadata_json}
```

## Per-Series Consensus
{consensus_block}
"""


# ── Per-series consensus formatting ─────────────────────────────────────────


def _format_series_consensus(label: str, result: dict) -> str:
    """Render one series' multi-model consensus as a compact briefing block."""
    consensus = result.get("consensus", {}) or {}
    if "error" in consensus:
        return f"### {label}\n_{consensus['error']}_"

    lines = [f"### {label}"]
    seq = consensus.get("sequence_type", "?")
    seq_agree = consensus.get("sequence_agreement", 0)
    grade = consensus.get("quality_grade", "?")
    lines.append(f"- Sequence: **{seq}** (agreement {seq_agree:.0%})")
    lines.append(f"- Quality grade: **{grade}**")

    structures = consensus.get("anatomical_structures", []) or []
    if structures:
        lines.append(f"- Coverage: {', '.join(structures[:12])}")

    pathology = consensus.get("pathology", {}) or {}
    if pathology.get("found"):
        n_agree = pathology.get("models_agreeing", 0)
        n_total = pathology.get("models_total", 0)
        lines.append(f"- **Pathology** ({n_agree}/{n_total} models):")
        for f in (pathology.get("findings", []) or [])[:6]:
            if isinstance(f, dict):
                loc = f.get("location", "?")
                sig = f.get("signal_on_this_sequence", "")
                size = f.get("size_mm", "")
                conf = f.get("confidence", "")
                bits = [b for b in (loc, sig, size, conf) if b]
                lines.append(f"  - {' • '.join(bits)}")
            else:
                lines.append(f"  - {f}")
        diffs = pathology.get("differential", []) or []
        if diffs:
            rendered = []
            for d in diffs[:5]:
                if isinstance(d, dict):
                    rendered.append(json.dumps(d, default=str))
                else:
                    rendered.append(str(d))
            lines.append(f"  - Differential: {'; '.join(rendered)}")
    else:
        lines.append("- Pathology: none flagged by consensus")

    notable = consensus.get("notable", []) or []
    if notable:
        lines.append(f"- Notable: {'; '.join(str(n) for n in notable[:6])}")

    disagreements = consensus.get("disagreements", []) or []
    if disagreements:
        fields = ", ".join(d.get("field", "?") for d in disagreements)
        lines.append(f"- ⚠ Inter-model disagreement on: {fields}")

    return "\n".join(lines)


# ── Public synthesis entry point ────────────────────────────────────────────


def synthesize_cloud_report(
    annotations: dict[str, dict],
    series_info: list[dict] | None = None,
    patient_info: dict | None = None,
    model_key: str = "gemma4",
    max_tokens: int = 8192,
) -> str:
    """Run the long-form §1-§11 dictation against a single OpenRouter model.

    Pulls the merged consensus per series + study metadata, formats them into
    the synthesis prompt, returns the rendered Markdown report.
    """
    # Local import to avoid a cloud<->synthesize circular at module load time.
    from .cloud import _sanitize_patient_info

    if model_key not in MODELS:
        raise ValueError(f"Unknown synthesis model_key: {model_key}")

    consensus_block = (
        "\n\n".join(
            _format_series_consensus(label, result) for label, result in sorted(annotations.items())
        )
        or "_No annotations available._"
    )

    compact_meta = {
        "patient": _sanitize_patient_info(patient_info),
        "series": [
            {
                k: v
                for k, v in (s or {}).items()
                if k
                not in (
                    "series_uid",
                    "volume_stats",
                    "quality_analysis",
                    "patient_id",
                    "patient_name",
                    "patient_birth_date",
                    "patient_weight",
                    "accession_number",
                )
            }
            for s in (series_info or [])
        ],
    }
    metadata_json = json.dumps(compact_meta, indent=2, default=str)[:3000]

    dynamic_section = _SYNTHESIS_PROMPT_DYNAMIC.format(
        metadata_json=metadata_json,
        consensus_block=consensus_block[:10000],
    )
    synthesis_model_id = MODELS[model_key]["id"]
    msg_content = _build_cached_user_content(
        static_text=CLOUD_SYNTHESIS_PROMPT_STATIC,
        dynamic_text=dynamic_section,
        image_b64=None,
        model_id=synthesis_model_id,
    )

    client = _client()
    # OpenRouter-side fallback chain: if the primary synthesis model is
    # unavailable / rate-limited, OR auto-rolls to the next cheap-tier slug
    # without us paying a client-side retry round-trip.
    fallback_chain = [
        MODELS[k]["id"] for k in _CHEAP_TIER if k != model_key and MODELS[k].get("vision") is True
    ]
    raw = _call_model(
        client,
        synthesis_model_id,
        [{"role": "user", "content": msg_content}],
        max_tokens=max_tokens,
        fallbacks=fallback_chain,
    )
    if not raw or raw.startswith("["):
        raise RuntimeError(raw or "empty synthesis response")
    return raw


__all__ = [
    "CLOUD_SYNTHESIS_PROMPT_STATIC",
    "_SYNTHESIS_PROMPT_DYNAMIC",
    "_format_series_consensus",
    "synthesize_cloud_report",
]

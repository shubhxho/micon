"""Cloud multi-model annotation -- flagship models label in parallel, then merge.

Runs each montage through multiple vision/text models via OpenRouter:
  - Gemma 4 12B IT (Google, open) -- multimodal, strong on-device-class reasoning
  - Gemini 2.5 Flash (Google) -- fast, strong vision
  - Qwen3 VL 30B A3B (Alibaba) -- excellent medical imaging
  - GPT-4.1 mini (OpenAI) -- reliable baseline
  - Claude Sonnet 4 (Anthropic) -- strong reasoning

Each model independently labels:
  1. Sequence identification
  2. Anatomical structures
  3. Pathology findings
  4. Quality assessment
  5. Notable findings

Results are merged into a consensus annotation with per-model confidence.
Disagreements are flagged for human review.

This module is the orchestration layer -- transport (``call_model``), routing
(``router``), pricing (``cost_model``), and long-form synthesis (``synthesize``)
each live in their own sibling module. Public symbols are re-exported here so
existing ``from src.annotation.cloud import X`` callers stay unbroken.

Requires: OPENROUTER_API_KEY env var.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time

import openai
from rich.console import Console

from .call_model import (
    _PROVIDER_OPENROUTER,
    MODELS,
    _build_cached_user_content,
    _cache_provider,
    _call_model,
    _client,
    _default_lineup,
    _detect_provider,
    _encode_image,
    _err,
    _filter_supported,
    _openrouter_extras,
    _resolve_model_id,
)
from .synthesize import (
    _SYNTHESIS_PROMPT_DYNAMIC,
    CLOUD_SYNTHESIS_PROMPT_STATIC,
    _format_series_consensus,
    synthesize_cloud_report,
)

# Plain output on Modal workers (no ANSI escape soup in captured logs);
# full rich output in interactive terminals.
_ON_MODAL = bool(os.environ.get("MODAL_TASK_ID"))
console = Console(force_terminal=not _ON_MODAL, no_color=_ON_MODAL)

# Default model for the long-form report synthesis stage.
# Gemma 4 by default -- matches the project's "process through Gemma 4" intent.
# Override via MICOM_SYNTHESIS_MODEL=claude|gemini|qwen|gpt4 for fallback.
SYNTHESIS_MODEL_KEY = os.environ.get("MICOM_SYNTHESIS_MODEL", "gemma4")

# Whitelisted metadata fields that may leave the local environment for the
# synthesis prompt. Anything not in this set is dropped -- we never ship
# patient name/MRN/DOB/weight to OpenRouter, even with a redacted study.
_SAFE_METADATA_FIELDS = frozenset(
    {
        "patient_sex",
        "patient_age_bracket",
        "study_date_year",
        "study_description",
        "institution",
        "manufacturer",
        "model",
        "field_strength",
        "software_versions",
        "station_name",
    }
)


def _sanitize_patient_info(info: dict | None) -> dict:
    """Strip PHI from patient_info before it enters a third-party prompt.

    Drops patient_id, patient_name, patient_birth_date, patient_weight, and any
    field not on the allowlist. Derives a coarse age bracket and study year
    from raw values when present so the dictation has scanner/age context
    without leaking identifiers.
    """
    if not info:
        return {}

    safe: dict = {}

    sex = (info.get("patient_sex") or "").strip().upper()
    if sex in {"M", "F", "O"}:
        safe["patient_sex"] = sex

    dob = (info.get("patient_birth_date") or "").strip()
    study_date = (info.get("study_date") or "").strip()
    if len(dob) >= 4 and len(study_date) >= 4 and dob[:4].isdigit() and study_date[:4].isdigit():
        try:
            age = int(study_date[:4]) - int(dob[:4])
            if 0 <= age <= 120:
                bracket = f"{(age // 10) * 10}s"
                safe["patient_age_bracket"] = bracket
        except ValueError:
            pass

    if len(study_date) >= 4 and study_date[:4].isdigit():
        safe["study_date_year"] = study_date[:4]

    for k in (
        "study_description",
        "institution",
        "manufacturer",
        "model",
        "field_strength",
        "software_versions",
        "station_name",
    ):
        v = info.get(k)
        if v:
            safe[k] = v

    return {k: v for k, v in safe.items() if k in _SAFE_METADATA_FIELDS}


# ── Per-model annotation ────────────────────────────────────────────────────

ANNOTATION_PROMPT_STATIC = """You are a board-certified radiologist annotating a medical imaging montage for a COMMERCIAL DATASET. Your annotation will be sold to AI/ML companies training diagnostic models (Qure.ai, Aidoc, Viz.ai tier). Quality and specificity directly determine dataset value.

Your output is one of several model annotations that will be merged into a multi-model consensus. Be maximally specific, faithful to what you see, and use ACR Lexicon conventions.

Return ONLY valid JSON (no markdown fences, no commentary), conforming exactly to this schema:

{{
  "sequence_type": "T1 | T2 | FLAIR | DWI | ADC | SWI | GRE | TOF | MRA | MRV | PD | STIR | T1+C | DSC | DCE | pCASL | MRS | DTI | bSSFP | other",
  "sequence_confidence": "high | medium | low",
  "sequence_evidence": "1-2 sentence rationale citing CSF behavior, GM/WM contrast, fat signal, susceptibility, or contrast effect",
  "plane": "axial | sagittal | coronal | mixed",
  "acquisition": "2D | 3D | unknown",
  "body_region": "brain | head_neck | spine | chest | abdomen | pelvis | extremity | whole_body | other",

  "anatomical_coverage": {{
    "extent": "full | partial | limited",
    "structures_visualized": ["list every anatomical structure clearly visible — be exhaustive"],
    "structures_partially_visible": ["structures at the edge or incompletely covered"],
    "laterality_assessment": "symmetric | asymmetric (describe which side and how)"
  }},

  "pathology": {{
    "found": true,
    "normal_statement": "if no pathology, provide a specific normal description (e.g. 'Normal gray-white differentiation, no mass, no restricted diffusion') — NOT just 'normal'",
    "findings": [
      {{
        "id": 1,
        "label": "short ML-friendly label (e.g. 'white_matter_hyperintensity', 'lacunar_infarct', 'meningioma')",
        "location": "anatomic region + laterality (e.g. 'right frontal periventricular white matter')",
        "slice_range": "approximate slice numbers where visible (e.g. 'axial slices 3-5')",
        "signal_on_this_sequence": "hyper/iso/hypo relative to [reference tissue]",
        "expected_on_other_sequences": "what this should look like on T1/T2/FLAIR/DWI if available",
        "size_mm": "A x B mm (or 'punctate' / 'confluent' / 'N/A')",
        "morphology": "well-circumscribed | ill-defined | ring-enhancing | solid | cystic | linear | wedge-shaped | punctate | confluent",
        "mass_effect": "none | mild | moderate | severe + description",
        "edema": "none | vasogenic | cytotoxic + extent",
        "confidence": "high | medium | low",
        "clinical_significance": "incidental | follow-up recommended | urgent | critical"
      }}
    ],
    "differential": [
      {{
        "diagnosis": "name",
        "probability": "most likely | possible | unlikely",
        "evidence_for": "what supports this",
        "evidence_against": "what argues against this"
      }}
    ],
    "comparison_needed": "what additional sequences would help characterize findings"
  }},

  "quality": {{
    "contrast": "sharp | moderate | poor",
    "snr": "adequate | marginal | poor",
    "artifacts": {{
      "present": ["list each artifact type"],
      "severity": "none | mild (diagnostic) | moderate (partially limiting) | severe (non-diagnostic)",
      "affected_regions": ["which anatomic regions are degraded by artifacts"]
    }},
    "coverage": "complete | partial | incomplete",
    "diagnostic_adequacy": "diagnostic | limited | non-diagnostic",
    "grade": "A | B | C | D | F",
    "grade_rationale": "one line explaining the grade",
    "ml_training_suitability": "excellent | good | acceptable | poor | unusable — for training segmentation/detection models"
  }},

  "ml_labels": {{
    "classification_tags": ["normal", "abnormal", "artifact_present", "motion_degraded", "incomplete_coverage"],
    "detection_targets": ["list anything a detection model should flag: mass, hemorrhage, infarct, calcification, etc."],
    "segmentation_regions": ["list regions suitable for auto-segmentation: brain_parenchyma, ventricles, white_matter, gray_matter, csf, lesion, etc."],
    "training_value": "high | medium | low — overall value for ML model training",
    "training_value_rationale": "why this series is/isn't valuable for training"
  }},

  "notable": [
    "list ALL notable observations — be exhaustive:",
    "- hemispheric asymmetry (volume/signal/sulcal)",
    "- atrophy pattern (global / focal — specify regions; MTA grade if assessable)",
    "- small vessel disease (Fazekas 0-3 if T2/FLAIR)",
    "- microhemorrhages or superficial siderosis if SWI/GRE",
    "- vascular variants (hypoplastic A1/P1, fetal PCA, azygos ACA)",
    "- developmental variants (cavum septum pellucidum, mega cisterna magna)",
    "- incidentals (DVA, arachnoid cyst, pineal cyst, sinus disease, mastoid effusion)",
    "- age-appropriate vs premature atrophy assessment"
  ],

  "uncertainty": ["areas where another sequence, contrast, or clinical context is needed"],
  "actionable": "none | category-1 (immediate, life-threatening) | category-2 (24-48h, clinically significant) | category-3 (routine follow-up)"
}}

Hard rules:
- If no pathology, set "found": false, provide a specific "normal_statement", use "findings": [], "differential": [].
- The "ml_labels" section is CRITICAL — this is what ML buyers pay for. Be thorough.
- Do NOT invent findings the montage does not show.
- Do NOT include patient identifiers.
- Use mm for measurements, "right"/"left" for laterality.
- Every finding MUST have a "slice_range" estimate so buyers can locate it.
- Rate "training_value" honestly — a blurry, partial-coverage scan is "low" even if pathology-free."""

# Dynamic header: series label + image description + quality context.
# Kept separate so it is NOT cached -- varies per annotation call.
ANNOTATION_PROMPT_DYNAMIC = (
    "Series: {series_label}\n"
    "**Image:** Axial (top row), Coronal (middle row), Sagittal (bottom row) "
    "— 6 slices per plane.\n"
    "{quality_ctx}"
)


def annotate_with_model(
    client: openai.OpenAI,
    model_key: str,
    montage_path: str,
    series_label: str,
    quality_ctx: str = "",
    provider: str | None = None,
) -> dict:
    """Run annotation on one model. Returns {model, annotation, raw, time_s, error}."""
    model = MODELS[model_key]
    provider = provider or _detect_provider() or _PROVIDER_OPENROUTER
    model_id = _resolve_model_id(model_key, provider)
    if model_id is None:
        return {
            "model": model["name"],
            "model_key": model_key,
            "error": f"{model['name']} not available on {provider}",
            "time_s": 0,
        }

    t0 = time.time()

    b64 = _encode_image(montage_path)
    if not b64:
        return {"model": model["name"], "error": "montage not found", "time_s": 0}

    dynamic_header = ANNOTATION_PROMPT_DYNAMIC.format(
        series_label=series_label, quality_ctx=quality_ctx
    )
    msg_content = _build_cached_user_content(
        static_text=ANNOTATION_PROMPT_STATIC,
        dynamic_text=dynamic_header,
        image_b64=b64,
        model_id=model_id,
    )

    # OR-side fallback: if the primary slug 404s / rate-limits, OpenRouter
    # tries the next slug for us. Only set on cheap-tier models that have
    # an explicit `fallback_ids` block (Gemma 4 -> Gemma 3 27B -> Kimi VL).
    fb = model.get("fallback_ids") if provider == _PROVIDER_OPENROUTER else None
    raw_or_pair = _call_model(
        client,
        model_id,
        [{"role": "user", "content": msg_content}],
        model["max_tokens"],
        fallbacks=fb,
        capture_reasoning=True,
    )
    if isinstance(raw_or_pair, tuple):
        raw, reasoning = raw_or_pair
    else:
        raw, reasoning = raw_or_pair, None
    elapsed = time.time() - t0

    # Parse JSON from response
    annotation = None
    if raw and not raw.startswith("["):
        # Try direct parse
        try:
            annotation = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting from markdown
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                with contextlib.suppress(json.JSONDecodeError):
                    annotation = json.loads(match.group(1))
            if not annotation:
                # Try finding any JSON object
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    with contextlib.suppress(json.JSONDecodeError):
                        annotation = json.loads(match.group())

    return {
        "model": model["name"],
        "model_key": model_key,
        "model_id": model_id,
        "tier": model.get("tier", "unknown"),
        "annotation": annotation,
        "raw": raw,
        "reasoning": reasoning,
        "time_s": round(elapsed, 1),
        "error": None if annotation else "failed to parse JSON",
    }


# ── Deep tissue analysis (re-exported from .tissue) ─────────────────────────
# Implementation lives in tissue.py; re-imported here so legacy callers that
# do `from src.annotation.cloud import tissue_analysis_with_model,
# TISSUE_ANALYSIS_PROMPT` keep working.
# ── Multi-model parallel annotation (re-exported from .consensus) ─────
# Implementation lives in consensus.py; re-imported here so legacy callers
# do `from src.annotation.cloud import annotate_series_multi` keep working.
from .consensus import (  # noqa: E402
    _build_consensus,
    _series_confidence,
    _should_escalate,
    annotate_series_multi,
)

# ── Full study annotation (re-exported from .study) ─────────────────────
# Implementation lives in study.py; re-imported here so legacy callers that
# do `from src.annotation.cloud import annotate_study_multi` keep working.
from .study import _build_study_summary, annotate_study_multi  # noqa: E402
from .tissue import TISSUE_ANALYSIS_PROMPT, tissue_analysis_with_model  # noqa: E402

# ── Public re-exports for legacy callers ────────────────────────────────────
# Keep ``from src.annotation.cloud import X`` working for every X that was
# previously top-level here, even after the call_model / synthesize splits.
__all__ = [
    "ANNOTATION_PROMPT_DYNAMIC",
    "ANNOTATION_PROMPT_STATIC",
    "CLOUD_SYNTHESIS_PROMPT_STATIC",
    "MODELS",
    "SYNTHESIS_MODEL_KEY",
    "TISSUE_ANALYSIS_PROMPT",
    "_PROVIDER_OPENROUTER",
    "_SAFE_METADATA_FIELDS",
    "_SYNTHESIS_PROMPT_DYNAMIC",
    "_build_cached_user_content",
    "_build_consensus",
    "_build_study_summary",
    "_cache_provider",
    "_call_model",
    "_client",
    "_default_lineup",
    "_detect_provider",
    "_encode_image",
    "_err",
    "_filter_supported",
    "_format_series_consensus",
    "_openrouter_extras",
    "_resolve_model_id",
    "_sanitize_patient_info",
    "_series_confidence",
    "_should_escalate",
    "annotate_series_multi",
    "annotate_study_multi",
    "annotate_with_model",
    "console",
    "synthesize_cloud_report",
    "tissue_analysis_with_model",
]

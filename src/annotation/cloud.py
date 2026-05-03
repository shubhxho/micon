"""Cloud multi-model annotation — flagship models label in parallel, then merge.

Runs each montage through multiple vision/text models via OpenRouter:
  - Gemma 4 31B IT (Google, open) — multimodal, strong on-device-class reasoning
  - Gemini 2.5 Flash (Google) — fast, strong vision
  - Qwen 2.5 VL 72B (Alibaba) — excellent medical imaging
  - GPT-4.1 mini (OpenAI) — reliable baseline
  - Claude Sonnet 4 (Anthropic) — strong reasoning

Each model independently labels:
  1. Sequence identification
  2. Anatomical structures
  3. Pathology findings
  4. Quality assessment
  5. Notable findings

Results are merged into a consensus annotation with per-model confidence.
Disagreements are flagged for human review.

Requires: OPENROUTER_API_KEY env var.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Plain output on Modal workers (no ANSI escape soup in captured logs);
# full rich output in interactive terminals.
_ON_MODAL = bool(os.environ.get("MODAL_TASK_ID"))
console = Console(force_terminal=not _ON_MODAL, no_color=_ON_MODAL)

# Default model for the long-form report synthesis stage.
# Gemma 4 by default — matches the project's "process through Gemma 4" intent.
# Override via MICOM_SYNTHESIS_MODEL=claude|gemini|qwen|gpt4 for fallback.
SYNTHESIS_MODEL_KEY = os.environ.get("MICOM_SYNTHESIS_MODEL", "gemma4")

# Whitelisted metadata fields that may leave the local environment for the
# synthesis prompt. Anything not in this set is dropped — we never ship
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


# ── Models ──────────────────────────────────────────────────────────────────

MODELS = {
    "gemma4": {
        "id": "google/gemma-4-31b-it",
        "name": "Gemma 4 31B IT",
        "vision": True,
        "max_tokens": 4096,
        "openai_id": None,
    },
    "gemini": {
        "id": "google/gemini-2.5-flash-preview",
        "name": "Gemini 2.5 Flash",
        "vision": True,
        "max_tokens": 4096,
        "openai_id": None,
    },
    "qwen": {
        "id": "qwen/qwen-2.5-vl-72b-instruct",
        "name": "Qwen 2.5 VL 72B",
        "vision": True,
        "max_tokens": 4096,
        "openai_id": None,
    },
    "gpt4": {
        "id": "openai/gpt-4.1-mini",
        "name": "GPT-4.1 mini",
        "vision": True,
        "max_tokens": 4096,
        # Direct-OpenAI slug — used when running through api.openai.com.
        "openai_id": "gpt-4.1-mini",
    },
    "claude": {
        "id": "anthropic/claude-sonnet-4",
        "name": "Claude Sonnet 4",
        "vision": True,
        "max_tokens": 4096,
        "openai_id": None,
    },
}


# ── OpenAI SDK transport ────────────────────────────────────────────────────
# Both paths use the official openai SDK — only the base_url differs.
#   - OPENROUTER_API_KEY → routes to https://openrouter.ai/api/v1 (full stack)
#   - OPENAI_API_KEY only → falls back to https://api.openai.com/v1 (gpt4 only)

_PROVIDER_OPENROUTER = "openrouter"
_PROVIDER_OPENAI = "openai"


def _detect_provider() -> str | None:
    if os.environ.get("OPENROUTER_API_KEY"):
        return _PROVIDER_OPENROUTER
    if os.environ.get("OPENAI_API_KEY"):
        return _PROVIDER_OPENAI
    return None


def _client() -> openai.OpenAI:
    """Return an OpenAI SDK client wired to whichever provider has credentials."""
    provider = _detect_provider()
    if provider == _PROVIDER_OPENROUTER:
        return openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    if provider == _PROVIDER_OPENAI:
        return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise RuntimeError(
        "No API key configured — set OPENROUTER_API_KEY (preferred, multi-model) "
        "or OPENAI_API_KEY (single-provider fallback)."
    )


def _resolve_model_id(model_key: str, provider: str) -> str | None:
    """Map a MODELS key to the slug for the active provider, or None if unsupported."""
    spec = MODELS[model_key]
    if provider == _PROVIDER_OPENROUTER:
        return spec["id"]
    if provider == _PROVIDER_OPENAI:
        return spec.get("openai_id")
    return None


def _filter_supported(model_keys: list[str], provider: str) -> list[str]:
    """Drop model keys the active provider can't serve (e.g. claude on direct OpenAI)."""
    return [k for k in model_keys if _resolve_model_id(k, provider) is not None]


def _encode_image(path: str) -> str | None:
    p = Path(path)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None


def _call_model(
    client: openai.OpenAI,
    model_id: str,
    messages: list[dict],
    max_tokens: int = 4096,
    retries: int = 3,
    request_timeout: float = 180.0,
) -> str:
    """Call a model with per-request timeout + bounded retry/backoff.

    The per-call timeout matters: without it a hung HTTP socket ties up the
    container until Modal's task-level timeout fires, which is much longer.
    180s is plenty of headroom for a Gemma 4 vision response.
    """
    for attempt in range(retries):
        try:
            resp = client.with_options(timeout=request_timeout).chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=messages,
            )
            return resp.choices[0].message.content
        except openai.RateLimitError:
            if attempt < retries - 1:
                time.sleep(4 * (attempt + 1))
            else:
                return f"[rate limited after {retries} attempts]"
        except openai.APITimeoutError:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return f"[timeout after {retries} attempts]"
        except Exception as e:
            return f"[error: {e}]"
    return ""


# ── Per-model annotation ────────────────────────────────────────────────────

ANNOTATION_PROMPT = """You are a board-certified radiologist annotating a medical imaging montage for a COMMERCIAL DATASET. Your annotation will be sold to AI/ML companies training diagnostic models (Qure.ai, Aidoc, Viz.ai tier). Quality and specificity directly determine dataset value.

Your output is one of several model annotations that will be merged into a multi-model consensus. Be maximally specific, faithful to what you see, and use ACR Lexicon conventions.

Series: {series_label}
**Image:** Axial (top row), Coronal (middle row), Sagittal (bottom row) — 6 slices per plane.
{quality_ctx}

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

    prompt = ANNOTATION_PROMPT.format(series_label=series_label, quality_ctx=quality_ctx)

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]

    raw = _call_model(client, model_id, [{"role": "user", "content": content}], model["max_tokens"])
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
        "annotation": annotation,
        "raw": raw,
        "time_s": round(elapsed, 1),
        "error": None if annotation else "failed to parse JSON",
    }


# ── Deep tissue analysis prompt (second pass — runs on Gemma 4) ─────────────

TISSUE_ANALYSIS_PROMPT = """You are a fellowship-trained neuroradiologist performing DEEP TISSUE ANALYSIS on a medical imaging montage. This is a second-pass analysis — a structured annotation has already been generated. Your job is to go deeper on tissue characterization, subtle findings, and clinical correlation.

Series: {series_label}
**Image:** Axial (top row), Coronal (middle row), Sagittal (bottom row).
{quality_ctx}

Previous structured annotation summary:
{prior_annotation}

Now provide a DETAILED tissue analysis. Return valid JSON only:

{{
  "tissue_analysis": {{
    "gray_matter": {{
      "cortical_thickness": "normal | thinned (location) | thickened (location)",
      "cortical_signal": "normal | abnormal (describe)",
      "deep_gray_nuclei": {{
        "caudate": "normal | atrophied | signal abnormality",
        "putamen": "normal | signal abnormality",
        "thalamus": "normal | signal abnormality",
        "globus_pallidus": "normal | mineralization | signal abnormality"
      }},
      "cortical_ribbon_preserved": true
    }},
    "white_matter": {{
      "periventricular": "normal | hyperintense foci (count/confluent) | leukoaraiosis",
      "deep_white_matter": "normal | punctate foci | confluent changes",
      "subcortical_u_fibers": "involved | spared",
      "corpus_callosum": "normal | thinned | signal abnormality",
      "fazekas_score": "0 | 1 | 2 | 3 (if FLAIR/T2)",
      "white_matter_disease_pattern": "none | age-appropriate | premature | MS-like | vascular | metabolic | post-inflammatory"
    }},
    "csf_spaces": {{
      "ventricles": "normal | dilated (Evans index estimate) | slit-like | asymmetric (describe)",
      "sulci": "normal | widened (global/focal — specify regions) | effaced",
      "cisterns": "normal | effaced (which ones)",
      "midline_shift_mm": 0,
      "hydrocephalus": "none | communicating | non-communicating | ex-vacuo"
    }},
    "brainstem_cerebellum": {{
      "brainstem": "normal | atrophied | signal abnormality (describe)",
      "cerebellum": "normal | atrophied | signal abnormality",
      "tonsillar_position": "normal | low-lying (mm below foramen magnum)",
      "cp_angle": "normal | mass | prominent vessel"
    }},
    "vascular": {{
      "flow_voids": "normal | absent (which vessels) | asymmetric",
      "vessel_caliber": "normal | ectatic | narrowed (describe)",
      "aneurysm_suspected": false,
      "vascular_malformation": "none | suspected (type, location)"
    }}
  }},

  "age_assessment": {{
    "brain_age_estimate": "pediatric (<18) | young-adult (18-40) | middle-aged (40-65) | elderly (>65)",
    "age_appropriate_changes": true,
    "premature_aging_signs": ["list any signs of premature atrophy or white matter disease"],
    "mta_score": "0 | 1 | 2 | 3 | 4 (medial temporal atrophy, if assessable)"
  }},

  "clinical_correlation": {{
    "most_likely_clinical_scenario": "normal healthy | chronic small vessel disease | acute stroke | demyelination | neoplasm | infection | congenital | post-surgical | other",
    "supporting_evidence": ["list imaging evidence for the scenario"],
    "contradicting_evidence": ["anything that doesn't fit"],
    "recommended_clinical_tests": ["relevant blood work, LP, EEG, etc. that would help"],
    "follow_up_imaging": "none needed | repeat in X months | urgent CT angiogram | contrast MRI | other"
  }},

  "dataset_value_assessment": {{
    "teaching_value": "high | medium | low — is this case educationally interesting?",
    "teaching_points": ["what can be learned from this case"],
    "rarity": "common | uncommon | rare — how often is this pattern seen?",
    "pathology_complexity": "none | single-finding | multi-finding | complex",
    "demographic_value": "Indian population MRI data is underrepresented — how does this add to global datasets?",
    "suggested_price_tier": "premium ($150+/study) | standard ($50-150) | discount ($10-50) | exclude"
  }},

  "health_recommendations": {{
    "urgency": "routine | follow-up-needed | urgent | emergency",
    "summary": "one plain-language sentence a non-medical person can understand (e.g. 'Your brain scan looks normal with no signs of stroke or tumor')",
    "key_findings_plain_language": [
      "translate each significant finding into simple language — no jargon",
      "e.g. 'Small bright spots in the brain's white matter suggest mild age-related blood vessel changes'",
      "e.g. 'No signs of stroke, bleeding, tumor, or infection were found'"
    ],
    "lifestyle_recommendations": [
      "evidence-based recommendations relevant to the findings:",
      "- If normal: 'Regular exercise (150 min/week), healthy diet, adequate sleep, and routine checkups help maintain brain health'",
      "- If white matter disease: 'Control blood pressure, manage cholesterol, stop smoking, exercise regularly'",
      "- If atrophy: 'Stay mentally active, maintain social connections, exercise, consider cognitive screening'",
      "- Always include: 'Stay hydrated, manage stress, limit alcohol'"
    ],
    "when_to_see_doctor": [
      "specific symptoms that should prompt immediate medical attention based on findings:",
      "- 'Sudden weakness on one side of the body'",
      "- 'Sudden difficulty speaking or understanding speech'",
      "- 'Worst headache of your life'",
      "- 'New seizures or loss of consciousness'",
      "- 'Progressive memory loss affecting daily life'"
    ],
    "follow_up_recommendation": "no follow-up needed | routine MRI in 12 months | follow-up MRI in 6 months | see neurologist within 2 weeks | go to emergency room",
    "brain_health_score": "A (excellent) | B (good) | C (some concerns) | D (significant concerns) | F (urgent attention needed)",
    "preventive_measures": [
      "specific preventive actions based on what was seen:",
      "- cardiovascular risk reduction if vascular changes",
      "- cognitive exercises if early atrophy signs",
      "- dietary recommendations (Mediterranean diet, omega-3)",
      "- exercise type recommendations (aerobic for vascular, resistance for overall)"
    ],
    "disclaimer": "This is an AI-generated analysis for informational purposes only. It does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment."
  }}
}}

Hard rules:
- Base EVERYTHING on what you actually see in the montage. Do NOT fabricate.
- If a tissue region is not visible in the montage, say "not assessable" rather than guessing.
- Fazekas scoring only valid on T2/FLAIR sequences.
- MTA scoring only valid on coronal T1 through hippocampi.
- This analysis directly determines dataset commercial value — be thorough."""


def tissue_analysis_with_model(
    client: openai.OpenAI,
    montage_path: str,
    series_label: str,
    prior_annotation: str = "",
    quality_ctx: str = "",
    model_key: str = "gemma4",
    provider: str | None = None,
) -> dict:
    """Run deep tissue analysis on one model (default: Gemma 4)."""
    provider = provider or _detect_provider() or _PROVIDER_OPENROUTER
    model_id = _resolve_model_id(model_key, provider)
    if model_id is None:
        return {"error": f"{model_key} not available on {provider}"}

    t0 = time.time()
    b64 = _encode_image(montage_path)
    if not b64:
        return {"error": "montage not found"}

    prompt = TISSUE_ANALYSIS_PROMPT.format(
        series_label=series_label,
        quality_ctx=quality_ctx,
        prior_annotation=prior_annotation[:2000],
    )

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]

    raw = _call_model(client, model_id, [{"role": "user", "content": content}], 4096)
    elapsed = time.time() - t0

    # Parse JSON
    annotation = None
    if raw and not raw.startswith("["):
        try:
            annotation = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                with contextlib.suppress(json.JSONDecodeError):
                    annotation = json.loads(match.group())

    return {
        "model": MODELS.get(model_key, {}).get("name", model_key),
        "tissue_analysis": annotation,
        "raw": raw,
        "time_s": round(elapsed, 1),
        "error": None if annotation else "failed to parse JSON",
    }


# ── Multi-model parallel annotation ─────────────────────────────────────────


def annotate_series_multi(
    montage_path: str,
    series_label: str,
    quality_ctx: str = "",
    models: list[str] | None = None,
) -> dict:
    """Run annotation across all models in parallel. Returns consensus + per-model results."""
    provider = _detect_provider()
    if provider is None:
        return {"error": "No API key set (OPENROUTER_API_KEY or OPENAI_API_KEY)"}

    client = _client()
    model_keys = _filter_supported(models or list(MODELS.keys()), provider)
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

    # Sequence type — majority vote
    seq_types = [a.get("sequence_type", "Unknown") for a in annotations.values()]
    seq_type = max(set(seq_types), key=seq_types.count)
    seq_agreement = seq_types.count(seq_type) / len(seq_types)

    # Pathology — any model finding pathology flags it
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

    # Quality — average grades
    grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
    grades = []
    for a in annotations.values():
        g = a.get("quality", {}).get("grade", "")
        if g in grade_map:
            grades.append(grade_map[g])
    avg_grade = sum(grades) / max(len(grades), 1)
    reverse_map = {4: "A", 3: "B", 2: "C", 1: "D", 0: "F"}
    consensus_grade = reverse_map.get(round(avg_grade), "?")

    # Anatomical structures — union (rich schema: anatomical_coverage.structures_visualized;
    # legacy: anatomical_structures)
    all_structures = []
    for a in annotations.values():
        coverage = a.get("anatomical_coverage", {})
        if isinstance(coverage, dict):
            all_structures.extend(coverage.get("structures_visualized", []))
        all_structures.extend(a.get("anatomical_structures", []))
    unique_structures = list(dict.fromkeys(all_structures))

    # Notable — union
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
                "flag": "NEEDS HUMAN REVIEW — models disagree on pathology",
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
    }


# ── Full study annotation ───────────────────────────────────────────────────

_GRADE_STYLE = {
    "A": "bold green",
    "B": "green",
    "C": "yellow",
    "D": "red",
    "F": "bold red",
}


def annotate_study_multi(
    series_results: list,
    series_info: list[dict],
    out_dir: Path,
    models: list[str] | None = None,
    n_workers: int = 4,
    synthesize: bool = True,
    synthesis_model: str | None = None,
    patient_info: dict | None = None,
) -> dict:
    """Annotate every series in the study via OpenRouter, then synthesize.

    Stages:
      1. Per-series multi-model annotation (parallel across models, sequential
         across series to respect rate limits).
      2. Consensus build with disagreement flags.
      3. Long-form §1–§10 dictation via a single model (default: Claude Sonnet 4)
         using the merged consensus + study metadata.

    Writes:
      - {out_dir}/annotations/<series>.json — per-series annotation
      - {out_dir}/annotations/study_annotations.json — full report
      - {out_dir}/final_report.md — narrative dictation
    """
    provider = _detect_provider()
    if provider is None:
        console.print(
            "[yellow]No API key set — set OPENROUTER_API_KEY (preferred) or "
            "OPENAI_API_KEY. Skipping cloud annotation.[/yellow]"
        )
        return {"error": "No API key set"}

    from .local import _build_quality_context, _is_derivative

    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    requested = models or list(MODELS.keys())
    model_keys = _filter_supported(requested, provider)
    dropped = [k for k in requested if k not in model_keys]
    if not model_keys:
        console.print(
            f"[yellow]No supported models on provider={provider}. "
            "Set OPENROUTER_API_KEY for full multi-model support.[/yellow]"
        )
        return {"error": f"No supported models on {provider}"}

    syn_key = synthesis_model or SYNTHESIS_MODEL_KEY
    if synthesize and _resolve_model_id(syn_key, provider) is None:
        # Synthesis model not served by the active provider — fall back to a
        # supported one (prefer gpt4 on direct-OpenAI).
        fallback = next(
            (
                k
                for k in ("gpt4", "claude", "gemma4", "gemini", "qwen")
                if _resolve_model_id(k, provider) is not None
            ),
            None,
        )
        if fallback:
            console.print(
                f"[yellow]Synthesis model {MODELS[syn_key]['name']} not available "
                f"on {provider}; using {MODELS[fallback]['name']} instead.[/yellow]"
            )
            syn_key = fallback
        else:
            synthesize = False

    title = "OpenRouter Annotation" if provider == _PROVIDER_OPENROUTER else "OpenAI Annotation"
    parts = [
        ("Provider: ", "bold"),
        (provider, "cyan"),
        ("\nModels: ", "bold"),
        (", ".join(MODELS[k]["name"] for k in model_keys), "cyan"),
        ("\nSeries: ", "bold"),
        (str(len(series_results)), "cyan"),
        ("    ", ""),
        ("Synthesis: ", "bold"),
        (MODELS[syn_key]["name"] if synthesize else "off", "cyan" if synthesize else "dim"),
    ]
    if dropped:
        parts.extend(
            [
                ("\nSkipped (unavailable on " + provider + "): ", "dim"),
                (", ".join(MODELS[k]["name"] for k in dropped), "dim"),
            ]
        )
    console.print(
        Panel(
            Text.assemble(*parts),
            title=f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    all_annotations: dict[str, dict] = {}

    progress_tbl = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    progress_tbl.add_column("Series", style="cyan", overflow="fold")
    progress_tbl.add_column("Sequence", style="bold")
    progress_tbl.add_column("Grade", justify="center")
    progress_tbl.add_column("Models", justify="right")
    progress_tbl.add_column("Time", justify="right", style="dim")
    progress_tbl.add_column("Flags")

    # Each annotate_series_multi call fans out to N model APIs and waits on
    # network. The outer loop is embarrassingly parallel across series, so
    # run a thread pool over OpenRouter calls. Worker count is bounded by
    # _ANN_PARALLEL (env-tunable) so we stay within OpenRouter's per-key
    # concurrency budget.
    annot_jobs: list[tuple[str, str, dict]] = []
    for r in series_results:
        montage_path = (
            r.get("montage_path") if isinstance(r, dict) else getattr(r, "montage_path", None)
        )
        if not montage_path:
            continue
        info = r.get("info", r) if isinstance(r, dict) else r.info
        label = f"Series {info.get('series_number', '?')} — {info.get('series_description', '')}"
        if _is_derivative(label):
            continue
        qa = info.get("quality_analysis")
        annot_jobs.append((montage_path, label, _build_quality_context(qa)))

    def _annotate_one(job: tuple[str, str, dict]) -> tuple[str, dict]:
        montage_path, label, quality_ctx = job
        t0 = time.time()
        try:
            result = annotate_series_multi(montage_path, label, quality_ctx, model_keys)
        except Exception as e:
            result = {"error": str(e), "models_called": 0, "models_succeeded": 0, "consensus": {}}
        result["time_s"] = round(time.time() - t0, 1)
        return label, result

    parallel = max(1, int(os.environ.get("MICOM_ANNOTATION_PARALLELISM", "8")))
    with ThreadPoolExecutor(max_workers=min(parallel, len(annot_jobs) or 1)) as pool:
        futures = {pool.submit(_annotate_one, j): j[1] for j in annot_jobs}
        for fut in as_completed(futures):
            label, result = fut.result()
            all_annotations[label] = result

            safe_name = re.sub(r"[^\w\-]", "_", label)
            (ann_dir / f"{safe_name}.json").write_text(json.dumps(result, indent=2, default=str))

            consensus = result.get("consensus", {})
            n_ok = result.get("models_succeeded", 0)
            n_total = result.get("models_called", 0)
            seq = consensus.get("sequence_type", "?")
            grade = consensus.get("quality_grade", "?")
            flags = []
            if consensus.get("disagreements"):
                flags.append(Text("disagreement", style="yellow"))
            if consensus.get("pathology", {}).get("found"):
                flags.append(Text("pathology", style="red"))
            flag_text = Text(", ").join(flags) if flags else Text("—", style="dim")

            progress_tbl.add_row(
                label,
                seq,
                Text(grade, style=_GRADE_STYLE.get(grade, "dim")),
                f"{n_ok}/{n_total}",
                f"{result.get('time_s', 0):.1f}s",
                flag_text,
            )

    console.print(progress_tbl)

    # ── Second pass: Deep tissue analysis via Gemma 4 ───────────────────
    # Runs on each primary (non-derivative) series using the initial
    # annotation as context. This adds tissue characterization, clinical
    # correlation, and per-study pricing assessment.
    console.print("\n[cyan]Deep tissue analysis (Gemma 4)…[/cyan]")
    tissue_model = "gemma4"
    if _resolve_model_id(tissue_model, provider) is None:
        tissue_model = model_keys[0] if model_keys else None

    if tissue_model:
        client = _client()

        # Index series_results by label once instead of doing two O(N) scans
        # per annotation.
        by_label: dict[str, dict] = {}
        for r in series_results:
            info = r.get("info", r) if isinstance(r, dict) else r.info
            rl = f"Series {info.get('series_number', '?')} — {info.get('series_description', '')}"
            mp = r.get("montage_path") if isinstance(r, dict) else getattr(r, "montage_path", None)
            by_label[rl] = {"montage_path": mp, "qa": info.get("quality_analysis")}

        def _tissue_one(item: tuple[str, dict]) -> tuple[str, dict | None]:
            label, result = item
            entry = by_label.get(label)
            if not entry or not entry.get("montage_path"):
                return label, None
            consensus = result.get("consensus", {})
            prior_summary = json.dumps(
                {
                    "sequence": consensus.get("sequence_type", "?"),
                    "pathology_found": consensus.get("pathology", {}).get("found", False),
                    "findings": consensus.get("pathology", {}).get("findings", [])[:3],
                    "quality_grade": consensus.get("quality_grade", "?"),
                },
                default=str,
            )
            quality_ctx = _build_quality_context(entry["qa"]) if entry["qa"] else ""
            try:
                return label, tissue_analysis_with_model(
                    client,
                    entry["montage_path"],
                    label,
                    prior_annotation=prior_summary,
                    quality_ctx=quality_ctx,
                    model_key=tissue_model,
                    provider=provider,
                )
            except Exception as e:
                return label, {"error": str(e)}

        tissue_count = 0
        with ThreadPoolExecutor(max_workers=min(parallel, len(all_annotations) or 1)) as pool:
            for label, tissue_result in pool.map(_tissue_one, list(all_annotations.items())):
                if not tissue_result or not tissue_result.get("tissue_analysis"):
                    continue
                all_annotations[label]["tissue_analysis"] = tissue_result["tissue_analysis"]
                tissue_count += 1
                safe_name = re.sub(r"[^\w\-]", "_", label)
                (ann_dir / f"{safe_name}.json").write_text(
                    json.dumps(all_annotations[label], indent=2, default=str)
                )

        console.print(
            f"  [green]✓[/green] Tissue analysis: {tissue_count}/{len(all_annotations)} series"
        )

    summary = _build_study_summary(all_annotations)

    study_report = {
        "provider": provider,
        "models": {k: MODELS[k]["name"] for k in model_keys},
        "synthesis_model": MODELS[syn_key]["name"] if synthesize else None,
        "series_annotated": len(all_annotations),
        "annotations": all_annotations,
        "summary": summary,
    }

    if synthesize and all_annotations:
        console.print(f"[cyan]Synthesizing final report via {MODELS[syn_key]['name']}…[/cyan]")
        try:
            narrative = synthesize_cloud_report(
                all_annotations,
                series_info=series_info,
                patient_info=patient_info,
                model_key=syn_key,
            )
            (out_dir / "final_report.md").write_text(narrative)
            study_report["narrative_path"] = str(out_dir / "final_report.md")
            console.print(f"[green]✓[/green] Final report → {out_dir / 'final_report.md'}")
        except Exception as e:
            console.print(f"[red]Synthesis failed:[/red] {e}")
            study_report["synthesis_error"] = str(e)

    (ann_dir / "study_annotations.json").write_text(json.dumps(study_report, indent=2, default=str))

    n_path = summary.get("pathology_detected", 0)
    n_dis = summary.get("disagreements", 0)
    console.print(
        Panel(
            Text.assemble(
                ("Annotated: ", "bold"),
                (f"{len(all_annotations)} series", ""),
                ("\nPathology: ", "bold"),
                (f"{n_path} series", "red" if n_path else ""),
                ("    ", ""),
                ("Disagreements: ", "bold"),
                (f"{n_dis} series", "yellow" if n_dis else ""),
                ("\nOutput: ", "bold"),
                (str(ann_dir), "dim"),
            ),
            title="[bold green]Annotation Complete[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
    )

    return study_report


# ── Long-form synthesis (cloud) ─────────────────────────────────────────────

CLOUD_SYNTHESIS_PROMPT = """You are a board-certified radiologist dictating the FINAL REPORT for a medical imaging study. This report serves TWO audiences: (1) clinical — a referring clinician must be able to act on it, and (2) commercial — AI/ML dataset buyers will use it to assess data value and as training labels. Your input is a merged multi-model consensus over every series, plus study metadata. Use formal radiology dictation style.

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

## Study Metadata
```json
{metadata_json}
```

## Per-Series Consensus
{consensus_block}
"""


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


def synthesize_cloud_report(
    annotations: dict[str, dict],
    series_info: list[dict] | None = None,
    patient_info: dict | None = None,
    model_key: str = "claude",
    max_tokens: int = 8192,
) -> str:
    """Run the long-form §1–§10 dictation against a single OpenRouter model.

    Pulls the merged consensus per series + study metadata, formats them into
    the synthesis prompt, returns the rendered Markdown report.
    """
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

    prompt = CLOUD_SYNTHESIS_PROMPT.format(
        metadata_json=metadata_json,
        consensus_block=consensus_block[:10000],
    )

    client = _client()
    raw = _call_model(
        client,
        MODELS[model_key]["id"],
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    if not raw or raw.startswith("["):
        raise RuntimeError(raw or "empty synthesis response")
    return raw


def _build_study_summary(annotations: dict) -> dict:
    """Summarize annotation results across all series."""
    total = len(annotations)
    if not total:
        return {}

    pathology_series = sum(
        1 for a in annotations.values() if a.get("consensus", {}).get("pathology", {}).get("found")
    )
    disagreement_series = sum(
        1 for a in annotations.values() if a.get("consensus", {}).get("disagreements")
    )

    all_findings = []
    seen = set()
    for a in annotations.values():
        for f in a.get("consensus", {}).get("pathology", {}).get("findings", []):
            if isinstance(f, dict):
                key = f"{f.get('location', '?')}|{f.get('signal_on_this_sequence', '')}|{f.get('size_mm', '')}"
            else:
                key = str(f)
            if key in seen:
                continue
            seen.add(key)
            all_findings.append(f)

    return {
        "total_series": total,
        "pathology_detected": pathology_series,
        "disagreements": disagreement_series,
        "needs_review": disagreement_series > 0,
        "all_findings": all_findings[:20],
    }

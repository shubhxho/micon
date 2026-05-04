# ruff: noqa: E501
# E501 is silenced file-wide: the bulk of this module is a single large
# triple-quoted JSON-schema LLM prompt (TISSUE_ANALYSIS_PROMPT) whose wording
# is domain-tuned. Reflowing the schema/prose at the 100-col mark would alter
# the literal sent to the model and risk shifting annotation behavior.
"""Deep tissue analysis -- second-pass annotation that goes deeper on tissue
characterization, subtle findings, and clinical correlation.

Runs after the per-model annotation pass (``cloud.annotate_with_model``)
using the merged consensus as prior context. Default model is Gemma 4 IT --
biased toward verbose, schema-compliant tissue descriptions; override via
``model_key`` if a specific provider is required for a study.

The prompt is large (~120 lines of nested schema). It is NOT cached because
this pass runs once per series and per-call uniqueness from the prior
annotation summary defeats prefix caching anyway.
"""

from __future__ import annotations

import contextlib
import json
import re
import time

import openai

# Local import-time alias so the test in ``test_prompt_caching`` that mocks
# ``src.annotation.cloud._encode_image`` does not need to be aware of this
# module. Tissue analysis is not exercised by the prompt-caching tests, so
# pulling ``_encode_image`` from ``call_model`` directly is safe.
from .call_model import (
    _PROVIDER_OPENROUTER,
    MODELS,
    _call_model,
    _detect_provider,
    _encode_image,
    _resolve_model_id,
)

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


__all__ = ["TISSUE_ANALYSIS_PROMPT", "tissue_analysis_with_model"]

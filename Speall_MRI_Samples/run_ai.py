"""Run every series in the sample study through Gemma 4 via OpenRouter.

Two-stage prompting per primary series, single-stage for derivative maps.

Inputs (auto-discovered):
  - series_index.json — list of series in the study
  - ../output/<src_dir>/<name>_multiplane.png + _detail.json

Outputs:
  - ai/<series>.json
  - ai/study_ai_summary.json

Requires OPENROUTER_API_KEY in /Users/shubh/Documents/micom/.env
"""
from __future__ import annotations
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
import openai

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

INDEX = Path(__file__).resolve().parent / "series_index.json"
OUT_DIR = Path(__file__).resolve().parent / "ai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "google/gemma-4-31b-it"
MODEL_NAME = "Gemma 4 31B IT"

DERIVATIVE_RE = re.compile(
    r'\b(d{0,2}REG\b|d{0,2}ADC\w*|d?ISO\w*|ISOTROPIC|FILT_PHA|COL:|PJN:|MIP|MinIP'
    r'|REFORMATT?ED|SUBTRACT|Processed)',
    re.IGNORECASE,
)

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def encode(path: Path) -> str | None:
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else None


def build_quality_ctx(detail: dict) -> str:
    qa = detail.get("quality_analysis", {})
    if not qa:
        return ""
    parts = ["**Quantitative metrics:**"]
    qg = qa.get("quality_grade", {})
    if qg:
        parts.append(f"- Grade: {qg.get('grade','?')} ({qg.get('score',0)}/100)")
    motion = qa.get("motion_analysis", {})
    if motion:
        parts.append(f"- Motion: ghosting={motion.get('ghosting_ratio',1):.2f}, slice_corr={motion.get('adjacent_slice_correlation',1):.3f}")
    sym = qa.get("symmetry_analysis", {})
    if sym:
        parts.append(f"- Symmetry: {sym.get('symmetry_index',0):.3f}")
    return "\n".join(parts)


ANNOT_PROMPT = """You are a board-certified neuroradiologist annotating a brain MRI montage for a commercial dataset. Your annotation will be used as ML training labels.

Series: {label}
**Image:** Axial (top), Coronal (middle), Sagittal (bottom) — 6 slices per plane.
{quality_ctx}

Return ONLY valid JSON (no fences) with this exact schema:

{{
  "sequence_type": "T1 | T2 | FLAIR | DWI | ADC | SWI | GRE | TOF | MRA | MRV | PD | STIR | T1+C | DSC | DTI | bSSFP | other",
  "sequence_confidence": "high | medium | low",
  "sequence_evidence": "1-2 sentences citing CSF behavior, GM/WM contrast, fat signal, susceptibility",
  "plane": "axial | sagittal | coronal | mixed",
  "acquisition": "2D | 3D | unknown",
  "anatomical_coverage": {{
    "extent": "full | partial | limited",
    "structures_visualized": ["list every clearly visible structure"],
    "laterality_assessment": "symmetric | asymmetric (describe)"
  }},
  "pathology": {{
    "found": true,
    "normal_statement": "if no pathology, give a specific normal description",
    "findings": [
      {{ "id": 1, "label": "snake_case", "location": "...", "slice_range": "...",
         "signal_on_this_sequence": "hyper/iso/hypo", "size_mm": "AxB or punctate",
         "morphology": "...", "mass_effect": "none|mild|moderate|severe",
         "confidence": "high|medium|low", "clinical_significance": "incidental|follow-up|urgent|critical" }}
    ],
    "differential": [ {{ "diagnosis": "name", "probability": "most likely|possible|unlikely" }} ]
  }},
  "quality": {{
    "contrast": "sharp|moderate|poor", "snr": "adequate|marginal|poor",
    "artifacts": {{ "present": [], "severity": "none|mild|moderate|severe" }},
    "diagnostic_adequacy": "diagnostic|limited|non-diagnostic",
    "grade": "A|B|C|D|F", "grade_rationale": "one line",
    "ml_training_suitability": "excellent|good|acceptable|poor|unusable"
  }},
  "ml_labels": {{
    "classification_tags": [],
    "detection_targets": [],
    "segmentation_regions": [],
    "training_value": "high|medium|low",
    "training_value_rationale": "why"
  }},
  "notable": [],
  "actionable": "none | category-1 | category-2 | category-3"
}}

If no pathology: set "found": false, populate "normal_statement", leave findings/differential empty arrays.
Base everything on what you see. Use mm and right/left."""


DERIVATIVE_PROMPT = """You are a neuroradiologist briefly characterizing a derived/post-processed map.

Series: {label}
**Image:** Axial (top), Coronal (middle), Sagittal (bottom).
{quality_ctx}

Return ONLY valid JSON (no fences):

{{
  "map_type": "ADC|eADC|isotropic_DWI|registered_DWI|phase_FILT_PHA|MIP|MinIP|subtraction|color_overlay|projection|reformatted|other",
  "source_sequence_inferred": "...",
  "expected_signal_behavior": "what should be bright/dark on this map",
  "focal_abnormalities": [
    {{ "location": "...", "signal_direction": "high|low",
       "clinical_significance": "...", "confidence": "high|medium|low" }}
  ],
  "no_focal_abnormality": false,
  "quality": {{
    "registration_accuracy": "n/a|good|fair|poor",
    "artifacts_noted": [],
    "diagnostic_adequacy": "diagnostic|limited|non-diagnostic",
    "grade": "A|B|C|D|F"
  }},
  "ml_labels": {{
    "use_case": "training_aid|reference_only|exclude",
    "training_value": "high|medium|low"
  }},
  "notes": "1 sentence"
}}

If nothing abnormal, set "no_focal_abnormality": true and leave focal_abnormalities empty."""


TISSUE_PROMPT = """You are a fellowship-trained neuroradiologist performing DEEP TISSUE ANALYSIS on this brain MRI montage.

Series: {label}
{quality_ctx}

Prior annotation summary: {prior}

Return valid JSON only (no fences):

{{
  "tissue_analysis": {{
    "gray_matter": {{ "cortical_thickness": "...", "deep_gray_nuclei": {{ "caudate": "...", "putamen": "...", "thalamus": "...", "globus_pallidus": "..." }} }},
    "white_matter": {{ "periventricular": "...", "deep_white_matter": "...", "fazekas_score": "0-3 (only if T2/FLAIR)", "white_matter_disease_pattern": "..." }},
    "csf_spaces": {{ "ventricles": "...", "sulci": "...", "midline_shift_mm": 0, "hydrocephalus": "none|communicating|non-communicating|ex-vacuo" }},
    "brainstem_cerebellum": {{ "brainstem": "...", "cerebellum": "...", "tonsillar_position": "..." }},
    "vascular": {{ "flow_voids": "...", "vessel_caliber": "...", "aneurysm_suspected": false }}
  }},
  "age_assessment": {{
    "brain_age_estimate": "pediatric|young-adult|middle-aged|elderly",
    "age_appropriate_changes": true, "premature_aging_signs": []
  }},
  "clinical_correlation": {{
    "most_likely_clinical_scenario": "...",
    "supporting_evidence": [],
    "follow_up_imaging": "none|repeat in X months|urgent CTA|contrast MRI"
  }},
  "dataset_value_assessment": {{
    "teaching_value": "high|medium|low", "rarity": "common|uncommon|rare",
    "pathology_complexity": "none|single-finding|multi-finding|complex",
    "suggested_price_tier": "premium|standard|discount|exclude"
  }},
  "health_recommendations": {{
    "urgency": "routine|follow-up-needed|urgent|emergency",
    "summary": "one plain-language sentence",
    "key_findings_plain_language": [], "lifestyle_recommendations": [],
    "when_to_see_doctor": [],
    "follow_up_recommendation": "none|routine MRI in 12 months|follow-up MRI in 6 months|see neurologist within 2 weeks|ER",
    "brain_health_score": "A|B|C|D|F",
    "disclaimer": "AI-generated. Not medical advice."
  }}
}}

If a region isn't visible, say "not assessable". Fazekas only on T2/FLAIR."""


def call_with_image(prompt: str, image_b64: str, max_tokens: int = 3500) -> tuple[str, float]:
    t0 = time.time()
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return r.choices[0].message.content, time.time() - t0
        except openai.RateLimitError:
            time.sleep(4 * (attempt + 1))
        except Exception as e:
            return f"[error: {e}]", time.time() - t0
    return "[rate limited]", time.time() - t0


def parse_json(text: str) -> dict | None:
    if not text or text.startswith("["):
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def find_montage(src_dir: Path) -> Path | None:
    for p in src_dir.glob("*_multiplane.png"):
        return p
    return None


def safe_label(label: str) -> str:
    return re.sub(r'[^\w\-]+', '_', label).strip('_') or 'unknown'


def process_series(entry: dict) -> dict:
    sn = entry["series_number"]
    label = entry["series_description"]
    src_dir = ROOT / entry["source_dir"]

    montage = find_montage(src_dir)
    is_derivative = bool(DERIVATIVE_RE.search(label))

    print(f"[s{sn:04d} {label}] {'derivative' if is_derivative else 'primary'}", flush=True)

    if not montage:
        return {"series_number": sn, "series_label": label, "error": "no montage"}

    # Locate detail.json from src_dir
    details = list(src_dir.glob("*_detail.json"))
    detail = json.loads(details[0].read_text()) if details else {}
    quality_ctx = build_quality_ctx(detail)

    img_b64 = encode(montage)
    if not img_b64:
        return {"series_number": sn, "series_label": label, "error": "encode failed"}

    if is_derivative:
        prompt = DERIVATIVE_PROMPT.format(label=label, quality_ctx=quality_ctx)
        raw, t = call_with_image(prompt, img_b64, max_tokens=2000)
        annot = parse_json(raw)
        out = {
            "series_number": sn, "series_label": label,
            "model": MODEL_NAME, "model_id": MODEL, "provider": "openrouter",
            "kind": "derivative",
            "annotation": annot, "raw": raw if not annot else None,
            "time_s": round(t, 2), "ok": annot is not None,
        }
    else:
        # Fire both prompts in parallel -- the deep-tissue prompt only uses the
        # prior annotation as light context, so racing them costs nothing in
        # quality and roughly halves wall time per primary series.
        prompt1 = ANNOT_PROMPT.format(label=label, quality_ctx=quality_ctx)
        prompt2_prior_stub = json.dumps({
            "sequence_type_hint": "see montage",
            "quality_grade_hint": "see montage",
        })
        prompt2 = TISSUE_PROMPT.format(label=label, quality_ctx=quality_ctx, prior=prompt2_prior_stub)
        with ThreadPoolExecutor(max_workers=2) as inner:
            f1 = inner.submit(call_with_image, prompt1, img_b64, 3500)
            f2 = inner.submit(call_with_image, prompt2, img_b64, 3500)
            raw1, t1 = f1.result()
            raw2, t2 = f2.result()
        annot = parse_json(raw1)
        tissue = parse_json(raw2)

        out = {
            "series_number": sn, "series_label": label,
            "model": MODEL_NAME, "model_id": MODEL, "provider": "openrouter",
            "kind": "primary",
            "stage_1_annotation": {
                "annotation": annot, "raw": raw1 if not annot else None,
                "time_s": round(t1, 2), "ok": annot is not None,
            },
            "stage_2_tissue_analysis": {
                "tissue_analysis": tissue, "raw": raw2 if not tissue else None,
                "time_s": round(t2, 2), "ok": tissue is not None,
            },
        }

    out_path = OUT_DIR / f"s{sn:04d}_{safe_label(label)}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[s{sn:04d}] ✓ wrote {out_path.name}", flush=True)
    return out


def main():
    with open(INDEX) as f:
        index = json.load(f)

    series_list = index["series"]
    results = []
    # Higher fan-out -- OpenRouter handles 16+ concurrent vision requests fine
    # per account, and each series is two sequential calls so we bound at ~32
    # in-flight HTTP connections at peak.
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(process_series, s): s for s in series_list}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                s = futures[f]
                print(f"[s{s['series_number']:04d}] FAILED: {e}", flush=True)
                results.append({"series_number": s["series_number"], "error": str(e)})

    results.sort(key=lambda r: r.get("series_number", 0))

    primary = [r for r in results if r.get("kind") == "primary"]
    derivative = [r for r in results if r.get("kind") == "derivative"]

    sequence_types = []
    grades = []
    pathology_count = 0
    pricing = []
    detection_targets = set()
    classification_tags = set()
    plain_summaries = []

    for r in primary:
        a = (r.get("stage_1_annotation") or {}).get("annotation") or {}
        t = (r.get("stage_2_tissue_analysis") or {}).get("tissue_analysis") or {}
        if a.get("sequence_type"):
            sequence_types.append(a["sequence_type"])
        g = a.get("quality", {}).get("grade")
        if g:
            grades.append(g)
        if a.get("pathology", {}).get("found"):
            pathology_count += 1
        for tag in a.get("ml_labels", {}).get("detection_targets", []):
            detection_targets.add(tag)
        for tag in a.get("ml_labels", {}).get("classification_tags", []):
            classification_tags.add(tag)
        dva = t.get("dataset_value_assessment", {})
        if dva.get("suggested_price_tier"):
            pricing.append(dva["suggested_price_tier"])
        hr = t.get("health_recommendations", {})
        if hr.get("summary"):
            plain_summaries.append({
                "series": r.get("series_label"),
                "summary": hr["summary"],
                "brain_health_score": hr.get("brain_health_score"),
                "urgency": hr.get("urgency"),
            })

    bhs = [p["brain_health_score"] for p in plain_summaries if p.get("brain_health_score")]
    consensus_bhs = max(set(bhs), key=bhs.count) if bhs else None
    consensus_price = max(set(pricing), key=pricing.count) if pricing else None

    summary = {
        "model_used": MODEL_NAME,
        "model_id": MODEL,
        "provider": "openrouter",
        "study_date": index.get("study_date"),
        "study_description": index.get("study_description"),
        "totals": {
            "series_in_study": index.get("total_series"),
            "primary_series_processed": len(primary),
            "derivative_series_processed": len(derivative),
            "primary_succeeded_stage1": sum(1 for r in primary if (r.get("stage_1_annotation") or {}).get("ok")),
            "primary_succeeded_stage2": sum(1 for r in primary if (r.get("stage_2_tissue_analysis") or {}).get("ok")),
            "derivative_succeeded": sum(1 for r in derivative if r.get("ok")),
        },
        "ai_consensus": {
            "sequence_types_detected": sequence_types,
            "quality_grade_distribution": {g: grades.count(g) for g in sorted(set(grades))},
            "pathology_series_count": pathology_count,
            "consensus_brain_health_score": consensus_bhs,
            "consensus_price_tier": consensus_price,
            "all_detection_targets": sorted(detection_targets),
            "all_classification_tags": sorted(classification_tags),
        },
        "plain_language_summaries": plain_summaries,
        "per_series_files": {
            f"s{r['series_number']:04d}_{safe_label(r.get('series_label', 'unknown'))}.json": r.get("kind", "?")
            for r in results
        },
    }

    (OUT_DIR / "study_ai_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== STUDY AI SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

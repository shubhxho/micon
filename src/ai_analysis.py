"""AI analysis via local MLX models — runs entirely on Apple Silicon GPU.

Architecture:
  1. Initial analysis: Gemma 4 reads montage + enhanced views + quality metrics
  2. Targeted follow-ups: same model generates questions, then answers them
  3. Cross-series comparison: text-only Gemma 4 pass across all series
  4. Synthesis: text-only Gemma 4 pass produces full structured report

A single multimodal Gemma 4 checkpoint backs every stage — vision and text
inference share the loaded weights via `mlx_vlm` (text-only is `image=None`).

Default model: mlx-community/gemma-4-26b-a4b-it-4bit (MoE, 4B active, ~13GB).
Override via env vars MICOM_VLM_MODEL / MICOM_LM_MODEL. Setting both to the
same id (the default) keeps memory usage to one model load.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from rich.console import Console

console = Console()

# ── Model IDs (configurable) ──────────────────────────────────────────────────

_GEMMA_4_DEFAULT = "mlx-community/gemma-4-26b-a4b-it-4bit"

_VLM_MODEL_ID = os.environ.get("MICOM_VLM_MODEL", _GEMMA_4_DEFAULT)
_LM_MODEL_ID = os.environ.get("MICOM_LM_MODEL", _GEMMA_4_DEFAULT)

# ── Lazy-loaded model singletons ─────────────────────────────────────────────

_vlm_model = None
_vlm_processor = None
_vlm_config = None
_lm_model = None
_lm_tokenizer = None
_lm_config = None
_model_lock = threading.Lock()

_DERIVATIVE_RE = re.compile(
    r"\b(d{0,2}REG\b|d{0,2}ADC\w*|d?ISO\w*|ISOTROPIC|FILT_PHA|COL:|PJN:|MIP|MinIP"
    r"|REFORMATT?ED|SUBTRACT)",
    re.IGNORECASE,
)
_HEDGING_RE = re.compile(
    r"\b(possibly|potentially|may\s+be|cannot\s+(?:fully|entirely|clearly)\s+(?:determine|assess|evaluate)"
    r"|uncertain|equivocal|borderline|subtle|faint|questionable|difficult\s+to\s+assess"
    r"|limited\s+(?:by|due)|cannot\s+(?:rule\s+out|exclude)|warrants?\s+(?:further|additional))\b",
    re.IGNORECASE,
)


def _load_vlm():
    """Load the vision-language model (lazy, once)."""
    global _vlm_model, _vlm_processor, _vlm_config
    if _vlm_model is not None:
        return _vlm_model, _vlm_processor, _vlm_config

    with _model_lock:
        if _vlm_model is not None:
            return _vlm_model, _vlm_processor, _vlm_config

        from mlx_vlm import load as vlm_load
        from mlx_vlm.utils import load_config

        console.print(f"[dim]Loading VLM: {_VLM_MODEL_ID}…[/dim]")
        _vlm_model, _vlm_processor = vlm_load(_VLM_MODEL_ID)
        _vlm_config = load_config(_VLM_MODEL_ID)
        console.print("[green]✓[/green] VLM loaded")
        return _vlm_model, _vlm_processor, _vlm_config


def _load_lm():
    """Load the text-only model (lazy, once).

    When `_LM_MODEL_ID == _VLM_MODEL_ID`, the VLM is reused — text-only stages
    call `mlx_vlm.generate(..., image=None)` rather than loading a second model.
    """
    global _lm_model, _lm_tokenizer, _lm_config
    if _lm_model is not None:
        return _lm_model, _lm_tokenizer, _lm_config

    with _model_lock:
        if _lm_model is not None:
            return _lm_model, _lm_tokenizer, _lm_config

        if _LM_MODEL_ID == _VLM_MODEL_ID:
            model, processor, config = _load_vlm()
            _lm_model, _lm_tokenizer, _lm_config = model, processor, config
            return _lm_model, _lm_tokenizer, _lm_config

        from mlx_lm import load as lm_load

        console.print(f"[dim]Loading LM: {_LM_MODEL_ID}…[/dim]")
        _lm_model, _lm_tokenizer = lm_load(_LM_MODEL_ID)
        _lm_config = None
        console.print("[green]✓[/green] LM loaded")
        return _lm_model, _lm_tokenizer, _lm_config


# ── Inference helpers ────────────────────────────────────────────────────────


def _vlm_generate(prompt: str, image_paths: list[str], max_tokens: int = 4096) -> str:
    """Run vision-language inference on local GPU."""
    model, processor, config = _load_vlm()

    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    with _model_lock:
        try:
            formatted = apply_chat_template(
                processor,
                config,
                prompt,
                num_images=len(image_paths),
            )
            result = vlm_generate(
                model,
                processor,
                formatted,
                image=image_paths or None,
                max_tokens=max_tokens,
                verbose=False,
            )
            # generate() returns GenerationResult with .text attribute
            return result.text if hasattr(result, "text") else str(result)
        except Exception as e:
            return f"[error: {e}]"


def _lm_generate(prompt: str, max_tokens: int = 4096) -> str:
    """Run text-only inference on local GPU.

    Reuses the multimodal checkpoint when LM/VLM ids match (default Gemma 4)
    by calling `mlx_vlm` with no image. Falls back to `mlx_lm` only when a
    separate text-only checkpoint is configured.
    """
    model, tokenizer, config = _load_lm()

    if _LM_MODEL_ID == _VLM_MODEL_ID:
        from mlx_vlm import generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        with _model_lock:
            try:
                formatted = apply_chat_template(
                    tokenizer,
                    config,
                    prompt,
                    num_images=0,
                )
                result = vlm_generate(
                    model,
                    tokenizer,
                    formatted,
                    image=None,
                    max_tokens=max_tokens,
                    verbose=False,
                )
                return result.text if hasattr(result, "text") else str(result)
            except Exception as e:
                return f"[error: {e}]"

    from mlx_lm import generate as lm_generate

    with _model_lock:
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            output = lm_generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=max_tokens,
                temp=0.7,
                verbose=False,
            )
            return output
        except Exception as e:
            return f"[error: {e}]"


# ── Shared utilities ─────────────────────────────────────────────────────────


def _is_derivative(label: str) -> bool:
    return bool(_DERIVATIVE_RE.search(label))


def _has_uncertainty(text: str) -> bool:
    return len(_HEDGING_RE.findall(text)) >= 2


def _parse_json_array(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("["):
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(x) for x in result if x]
        except json.JSONDecodeError:
            pass

    code_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if code_match:
        try:
            return [str(x) for x in json.loads(code_match.group(1)) if x]
        except json.JSONDecodeError:
            pass

    for match in re.finditer(r"\[.*?\]", text, re.DOTALL):
        try:
            result = json.loads(match.group())
            if isinstance(result, list) and all(isinstance(x, str) for x in result):
                return result
        except json.JSONDecodeError:
            continue

    lines = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip().strip('"').strip("'")
        if len(line) > 20:
            lines.append(line)
    return lines[:3]


# ── Stage 1: Initial analysis ────────────────────────────────────────────────


def _build_quality_context(quality_analysis: dict | None) -> str:
    if not quality_analysis:
        return ""

    parts = ["\n**Quantitative quality metrics (from automated analysis):**"]

    qg = quality_analysis.get("quality_grade", {})
    if qg:
        parts.append(f"- Quality grade: {qg.get('grade', '?')} ({qg.get('score', 0)}/100)")

    motion = quality_analysis.get("motion_analysis", {})
    if motion:
        parts.append(
            f"- Motion: ghosting_ratio={motion.get('ghosting_ratio', 1):.2f}, "
            f"slice_correlation={motion.get('adjacent_slice_correlation', 1):.3f} "
            f"({motion.get('interpretation', '?')})"
        )

    sym = quality_analysis.get("symmetry_analysis", {})
    if sym:
        parts.append(
            f"- Symmetry: index={sym.get('symmetry_index', 0):.3f} "
            f"({sym.get('interpretation', '?')})"
        )

    sharp = quality_analysis.get("sharpness_analysis", {})
    if sharp:
        parts.append(
            f"- Sharpness: mean_gradient={sharp.get('sharpness_mean', 0):.1f} "
            f"({sharp.get('interpretation', '?')})"
        )

    anomaly = quality_analysis.get("anomaly_detection", {})
    if anomaly and anomaly.get("n_anomalous", 0) > 0:
        slices = anomaly["anomalous_slices"]
        desc = ", ".join(
            f"slice {s['slice_index']} ({s['direction']}, z={s['z_score']:.1f})" for s in slices[:5]
        )
        parts.append(f"- Anomalous slices: {desc}")

    return "\n".join(parts)


def analyze_montage(
    montage_path: str,
    series_label: str,
    enhanced_path: str | None = None,
    quality_analysis: dict | None = None,
) -> str:
    if not Path(montage_path).exists():
        return f"[{series_label}]: montage not found"

    image_paths = [montage_path]
    enhanced_desc = ""
    if enhanced_path and Path(enhanced_path).exists():
        image_paths.append(enhanced_path)
        enhanced_desc = (
            "\n**Image 2 (Enhanced GPU Views):** Row 1: MIP (bright structures/vessels). "
            "Row 2: MinIP (CSF/cysts/dark lesions). Row 3: Tissue segmentation overlay."
        )

    quality_ctx = _build_quality_context(quality_analysis)

    prompt = f"""You are a board-certified neuroradiologist (CAQ Neuroradiology) annotating a medical imaging montage from a brain MRI study. Output a structured per-series dictation. Be specific, anatomically precise, and use standard neuroradiology terminology (BI-RADS-style structure, ASNR conventions, ACR Lexicon where applicable).

Series: {series_label}
**Image 1:** Axial (top), Coronal (middle), Sagittal (bottom) — 6 slices per plane.{enhanced_desc}
{quality_ctx}

Produce these sections, each labeled with the heading verbatim:

1. **Sequence Identification**
   - Sequence type (T1 / T2 / FLAIR / DWI / ADC / SWI / GRE / TOF / MRA / MRV / PD / STIR / T1+C / DSC / DCE / pCASL / MRS / DTI / 3D vol vs 2D / fat-sat / inv-recovery / B-FFE / SSFP, etc.)
   - Plane and slice profile (axial 2D, sagittal 3D iso, etc.)
   - Confidence (high/medium/low) and the signal cues that confirm it (CSF behavior, GM/WM contrast, fat signal, susceptibility, contrast effect).

2. **Anatomical Coverage**
   - Cranial-caudal extent (vertex to foramen magnum? skull base included? upper cervical cord?).
   - Visualized structures by plane: cortex/sulci, deep GM (caudate, putamen, GP, thalamus), white matter (centrum semiovale, corona radiata, internal/external capsule), corpus callosum, brainstem (midbrain, pons, medulla), cerebellum (vermis, hemispheres, tonsils), ventricles (lateral, 3rd, 4th, aqueduct), cisterns (suprasellar, prepontine, CPA, quadrigeminal), pituitary/sella, optic apparatus, orbits, sinuses, mastoids, vasculature visible.

3. **Findings**
   For each abnormality: anatomic location (lobe, gyrus, vascular territory or specific tract), signal on this sequence, size (mm in 2 axes), morphology (well-circumscribed/ill-defined, ring/solid, mass effect, edema, restricted diffusion if applicable), and laterality. If multiple, number them. If unremarkable, state explicitly: "No focal parenchymal signal abnormality on this sequence."

4. **Quality Assessment**
   - GM-WM contrast: sharp / moderate / poor.
   - SNR: adequate / marginal / poor.
   - Artifacts: motion (ghosting/blur), susceptibility, chemical shift, wrap, Gibbs ringing, zipper, RF spikes, parallel-imaging g-factor, EPI distortion, banding (B-FFE), N/2 ghosting (DWI). Note severity and location.
   - Coverage gaps or truncation.
   - Diagnostic adequacy (yes / limited / non-diagnostic) with rationale.{"  The automated quantitative metrics above flagged specific issues — for each flag, explicitly confirm or refute it from what you see." if quality_ctx else ""}
   - Letter grade (A/B/C/D/F) with one-line justification.

5. **Notable Findings & Incidentals**
   - Hemispheric asymmetry (volume, signal, sulcal pattern).
   - Atrophy pattern (global, focal, regional — frontal/temporal/parietal/cerebellar/brainstem; hippocampal — Scheltens MTA grade if assessable).
   - Small vessel disease (Fazekas 0–3) if T2/FLAIR.
   - Microhemorrhages or superficial siderosis if susceptibility-weighted.
   - Incidentals (developmental venous anomaly, arachnoid cyst, pineal cyst, sinus disease, mastoid effusion).
   - Areas of uncertainty (what would a second sequence/contrast resolve?).

Keep each section tight — bullet form is fine, but every claim must reference something visible in the montage."""

    return _vlm_generate(prompt, image_paths)


def analyze_derivative(montage_path: str, series_label: str) -> str:
    if not Path(montage_path).exists():
        return f"[{series_label}]: montage not found"

    prompt = f"""You are a neuroradiologist reviewing a derived/post-processed map.
Series: {series_label}
Montage: Axial (top), Coronal (middle), Sagittal (bottom) — 6 slices per plane.

Concise structured read (≤8 sentences total):

1. **Map type** — ADC, eADC, isotropic DWI, registered series (dREG), phase (FILT_PHA), MIP/MinIP, subtraction, color overlay (COL:), projection (PJN:), reformatted (REFORMATTED), or other. Note the source sequence if inferable.
2. **Expected signal behavior** — what should be bright vs dark on this map (e.g. ADC: free water bright, restricted diffusion dark; MIP: vessels/blood bright; phase: susceptibility +/- depending on convention).
3. **Focal abnormalities** — location, signal direction (high/low relative to background), and clinical significance specific to this map (e.g. "low ADC = restricted diffusion → acute infarct vs hypercellular tumor vs abscess").
4. **Quality & utility** — registration accuracy if dREG, artifacts (EPI distortion on ADC, vascular overlap on MIP, phase wrap), and whether the map is diagnostic for its intended purpose.

If nothing abnormal: state "No focal abnormality on this derived map."""

    return _vlm_generate(prompt, [montage_path], max_tokens=1024)


# ── Stage 2: Targeted follow-ups ─────────────────────────────────────────────


def _generate_followups(
    initial: str,
    series_label: str,
    quality_analysis: dict | None = None,
) -> list[str]:
    quality_flags = []
    if quality_analysis:
        motion = quality_analysis.get("motion_analysis", {})
        if motion.get("motion_detected"):
            quality_flags.append(f"Automated motion detection: {motion.get('interpretation')}")
        anomaly = quality_analysis.get("anomaly_detection", {})
        if anomaly.get("n_anomalous", 0) > 0:
            quality_flags.append(f"{anomaly['n_anomalous']} anomalous slices detected")
        sym = quality_analysis.get("symmetry_analysis", {})
        if sym.get("symmetry_index", 1) < 0.90:
            quality_flags.append(f"Asymmetry detected: index={sym['symmetry_index']:.3f}")

    flags_text = ""
    if quality_flags:
        flags_text = "\n\nAutomated quality flags:\n" + "\n".join(f"- {f}" for f in quality_flags)

    prompt = f"""Given this MRI analysis for "{series_label}", generate 1-3 focused follow-up questions.

Each question should reference a SPECIFIC finding from the analysis and ask something it was uncertain about.

Return a JSON array of strings.
{flags_text}

## Analysis
{initial[:3000]}"""

    text = _lm_generate(prompt, max_tokens=1024)
    return _parse_json_array(text)


def _run_followups(
    montage_path: str,
    series_label: str,
    initial: str,
    prompts: list[str],
) -> list[dict]:
    results = []
    for prompt in prompts:
        full_prompt = f"""Series: {series_label}. Previous analysis:
{initial[:1500]}

Follow-up: {prompt}

Answer specifically based on what you see in the image."""

        response = _vlm_generate(full_prompt, [montage_path], max_tokens=2048)
        if response and not response.startswith("["):
            results.append({"prompt": prompt, "response": response})
        else:
            break
    return results


# ── Combined per-series pipeline ─────────────────────────────────────────────


def analyze_and_chain(
    montage_path: str,
    series_label: str,
    enhanced_path: str | None = None,
    max_depth: int = 1,
    quality_analysis: dict | None = None,
) -> dict:
    if not Path(montage_path).exists():
        return {
            "initial": f"[{series_label}]: montage not found",
            "followups": [],
            "merged": "",
            "skipped_chain": True,
            "chain_reason": "error",
        }

    if _is_derivative(series_label):
        initial = analyze_derivative(montage_path, series_label)
        return {
            "initial": initial,
            "followups": [],
            "merged": initial,
            "skipped_chain": True,
            "chain_reason": "derivative",
        }

    initial = analyze_montage(
        montage_path,
        series_label,
        enhanced_path=enhanced_path,
        quality_analysis=quality_analysis,
    )

    if initial.startswith("[") or initial.startswith("Error"):
        return {
            "initial": initial,
            "followups": [],
            "merged": initial,
            "skipped_chain": True,
            "chain_reason": "error",
        }

    has_quality_flags = False
    if quality_analysis:
        motion = quality_analysis.get("motion_analysis", {})
        anomaly = quality_analysis.get("anomaly_detection", {})
        sym = quality_analysis.get("symmetry_analysis", {})
        has_quality_flags = (
            motion.get("motion_detected", False)
            or anomaly.get("n_anomalous", 0) > 0
            or sym.get("symmetry_index", 1) < 0.90
        )

    needs_chain = _has_uncertainty(initial) or has_quality_flags

    if not needs_chain:
        return {
            "initial": initial,
            "followups": [],
            "merged": initial,
            "skipped_chain": True,
            "chain_reason": "confident",
        }

    prompts = _generate_followups(initial, series_label, quality_analysis)
    if not prompts:
        return {
            "initial": initial,
            "followups": [],
            "merged": initial,
            "skipped_chain": True,
            "chain_reason": "no_followups_generated",
        }

    followups = _run_followups(montage_path, series_label, initial, prompts)

    merged = initial
    if followups:
        merged += "\n\n---\n## Targeted Follow-up Findings\n\n"
        for i, fu in enumerate(followups, 1):
            merged += f"### Follow-up {i}\n**Q:** {fu['prompt']}\n\n**A:** {fu['response']}\n\n"

    return {
        "initial": initial,
        "followups": followups,
        "merged": merged,
        "skipped_chain": False,
        "chain_reason": "uncertainty" if _has_uncertainty(initial) else "quality_flags",
    }


# ── Cross-series comparison ──────────────────────────────────────────────────


def cross_series_comparison(analyses: dict[str, str], series_info: list[dict]) -> str:
    series_summary = []
    for s in series_info:
        if not s.get("has_pixels"):
            continue
        seq = s.get("sequence_classification", {}).get("sequence_type", "?")
        qa = s.get("quality_analysis", {})
        grade = qa.get("quality_grade", {}).get("grade", "?") if qa else "?"
        series_summary.append(
            f"- Series {s.get('series_number', '?')} ({s.get('series_description', '')}) — {seq}, Grade={grade}"
        )

    findings = []
    for label, analysis in sorted(analyses.items()):
        if analysis.startswith("[") or analysis.startswith("Error"):
            continue
        findings.append(f"### {label}\n{analysis[:1500]}")

    prompt = f"""You are a board-certified neuroradiologist correlating findings across every series in a brain MRI study. Your output feeds the final dictation, so be exhaustive but disciplined — every claim must trace back to a specific series number.

## Available Series
{chr(10).join(series_summary)}

## Per-Series Findings
{chr(10).join(findings[:10])}

Produce these sections, exactly in this order, with the headings verbatim:

1. **Protocol Adequacy**
   - Does the protocol include the standard brain-MRI core (T1, T2, FLAIR, DWI/ADC, GRE/SWI)? List missing core sequences.
   - Are the supplementary sequences appropriate for the apparent indication (post-contrast T1, MRA/MRV, MRS, perfusion, DTI)?
   - Coverage and resolution adequate for the findings described?

2. **Multi-Sequence Lesion Correlation**
   For every abnormality flagged in any series:
   - Lesion ID (#1, #2, …) with anatomic location and laterality.
   - Expected signal on T1 / T2 / FLAIR / DWI / ADC / SWI / T1+C — fill what is observed (cite series #), what is expected from physiology, and whether observation matches expectation.
   - If only one series captured it, name the sequence(s) needed to characterize further.
   - Differential implication of the signal pattern (e.g. T2 hyperintense + restricted diffusion + no enhancement → acute infarct vs early demyelination; T1 hyperintense → fat/melanin/blood/protein/calcium).

3. **Cross-Series Quality**
   - Best series for cortex, deep GM, white matter, posterior fossa, vasculature, hemorrhage detection.
   - Series degraded by motion / artifact / coverage gaps — call out which downstream conclusions are weakened.
   - If two series of the same type exist, which one to prefer and why.

4. **Integrated Impression**
   - Unified, sequence-spanning interpretation.
   - Convergent vs divergent findings across sequences.
   - Top differential (3–5 entries, ranked) with the cross-sequence evidence supporting each.
   - What would change the diagnosis (additional sequence, contrast, follow-up interval)."""

    return _lm_generate(prompt, max_tokens=6144)


# ── Synthesis ────────────────────────────────────────────────────────────────


def synthesize_report(payload: dict, image_analyses: dict[str, str], cross_series: str = "") -> str:
    image_section = "\n\n".join(
        f"### {label}\n{analysis}"
        for label, analysis in sorted(image_analyses.items())
        if not analysis.startswith("[") and not analysis.startswith("Error")
    )

    compact_payload = {
        "patient": payload.get("patient", {}),
        "series": [
            {
                k: v
                for k, v in s.items()
                if k not in ("series_uid", "volume_stats", "quality_analysis")
            }
            for s in payload.get("series", [])
        ],
        "conformance_issues_count": payload.get("conformance_issues_count", 0),
    }

    cross_section = f"\n## Cross-Series Comparison\n{cross_series}\n" if cross_series else ""

    prompt = f"""You are a board-certified neuroradiologist (CAQ Neuroradiology) dictating the final report for a brain MRI study. Synthesize the per-series reads and cross-series correlation below into a complete, attending-grade radiology report. The report must stand alone — a referring clinician should be able to act on it without seeing the per-series notes. Use formal radiology dictation style.

Output the report as Markdown with these sections, exactly in this order, headings verbatim. Do not omit sections; if a section is N/A, write "Not applicable" under it with a brief reason.

# Final Report

## 1. Clinical Summary
- Patient demographics (age, sex when available), MRN/study UID handling, referring service if known.
- Scanner manufacturer, model, field strength, coil class.
- Study date, accession.
- Indication / clinical question if extractable from metadata; otherwise state "Indication not provided in metadata."

## 2. Protocol
A markdown table with columns: `#` | `Series Description` | `Sequence` | `Plane` | `Slice (mm) / Voxel` | `TR/TE/TI` | `Quality Grade`. One row per imaged series (omit pure derivatives unless they're the only representative of a modality). End with a 2–3 sentence protocol assessment: completeness, missing sequences, suitability for the apparent indication.

## 3. Comparison
State explicitly whether prior studies are available; if no metadata indicates priors, write "No prior comparison available in this dataset."

## 4. Findings
Dictate by anatomic region. Use this structure verbatim:

**4.1 Brain Parenchyma**
- *Supratentorial:* gray-white differentiation, cortical signal, deep gray nuclei, white matter (note Fazekas if T2/FLAIR available).
- *Infratentorial:* brainstem, cerebellar hemispheres, vermis, tonsil position relative to foramen magnum.
- *Diffusion:* restricted diffusion locations (cite DWI + ADC series #), or "no restricted diffusion."
- *Susceptibility:* microhemorrhages, superficial siderosis, mineralization (cite SWI/GRE series #), or "no susceptibility abnormality."
- *Post-contrast (if T1+C present):* enhancement pattern, location, morphology, leptomeningeal/dural/parenchymal; otherwise "Post-contrast imaging not performed."

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
A numbered list of 3–6 entries, ordered by clinical urgency. Each entry: one-line conclusion, then a sub-bullet with the supporting cross-sequence evidence (series #s). Hedge language only when the data warrants it.

## 6. Differential Diagnosis
For the top 1–3 findings in §5 that are not unequivocal, give a ranked differential (3–5 entries) with the imaging evidence for/against each.

## 7. Recommendations
Numbered, actionable: additional sequences (specify which and why), contrast administration, follow-up interval, correlation with clinical/lab data, multidisciplinary referral.

## 8. Critical / Actionable Findings
ACR Actionable Findings tier (Category 1 / 2 / 3) if any. If none, state "No critical actionable findings."

## 9. Technical Notes & Data Quality
- Per-series quality grade summary (A/B/C/D/F counts).
- Limiting factors (motion, coverage, artifact) and which conclusions they weaken.
- Suitability for downstream automated analysis (segmentation, registration, ML pipelines).
- HIPAA / de-identification status if surfaced in metadata.

## 10. Study Disposition
One line: `Diagnostic` / `Limited — repeat recommended` / `Non-diagnostic`. With one-line rationale.

---
Hard rules:
- Cite series numbers in parentheses when claiming a finding (e.g. "(Series 4 DWI, Series 5 ADC)").
- Do NOT invent findings absent from the per-series analyses or cross-series comparison.
- Do NOT include patient names or other PHI even if present in the metadata block.
- Prefer measurements in mm. Use laterality (right/left), not "RH/LH".
- If the per-series analyses contain "[error:" or "montage not found" entries, list those series in §9 and exclude them from §4.

## Metadata
```json
{json.dumps(compact_payload, indent=2, default=str)[:3000]}
```

## Per-Series Analyses
{image_section[:6000]}
{cross_section[:2000]}"""

    return _lm_generate(prompt, max_tokens=8192)

"""Study-level orchestration -- run the full annotation pipeline across every
series in a study, then synthesize the long-form report.

This is the Modal-side entry point invoked by ``modal_app.py``. It wires
together:
  - ``cloud.annotate_series_multi`` -- per-series multi-model fan-out
  - ``tissue.tissue_analysis_with_model`` -- second-pass tissue characterization
  - ``synthesize.synthesize_cloud_report`` -- final §1-§11 dictation

and writes everything (per-series JSON, study_annotations.json,
final_report.md) to ``out_dir``. Lives in its own module so cloud.py can
stay focused on per-series logic without absorbing 290 lines of report-
writing scaffolding.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .call_model import (
    _PROVIDER_OPENROUTER,
    MODELS,
    _client,
    _default_lineup,
    _detect_provider,
    _filter_supported,
    _resolve_model_id,
)
from .synthesize import synthesize_cloud_report
from .tissue import tissue_analysis_with_model

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
      3. Long-form §1-§10 dictation via a single model (default: Claude Sonnet 4)
         using the merged consensus + study metadata.

    Writes:
      - {out_dir}/annotations/<series>.json -- per-series annotation
      - {out_dir}/annotations/study_annotations.json -- full report
      - {out_dir}/final_report.md -- narrative dictation
    """
    # Cloud-side imports here to avoid a study<->cloud module-import cycle.
    from .cloud import SYNTHESIS_MODEL_KEY, annotate_series_multi, console

    provider = _detect_provider()
    if provider is None:
        console.print(
            "[yellow]No API key set -- set OPENROUTER_API_KEY (preferred) or "
            "OPENAI_API_KEY. Skipping cloud annotation.[/yellow]"
        )
        return {"error": "No API key set"}

    from .local import _build_quality_context, _is_derivative

    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    requested = models or _default_lineup()
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
        # Synthesis model not served by the active provider -- fall back to a
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
    # MICOM_ANNOTATION_PARALLELISM so we stay within OpenRouter's per-key
    # concurrency budget.
    annot_jobs: list[tuple[str, str, dict]] = []
    for r in series_results:
        montage_path = (
            r.get("montage_path") if isinstance(r, dict) else getattr(r, "montage_path", None)
        )
        if not montage_path:
            continue
        info = r.get("info", r) if isinstance(r, dict) else r.info
        label = f"Series {info.get('series_number', '?')} -- {info.get('series_description', '')}"
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
            flag_text = Text(", ").join(flags) if flags else Text("--", style="dim")

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
    console.print("\n[cyan]Deep tissue analysis (Gemma 4)...[/cyan]")
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
            rl = f"Series {info.get('series_number', '?')} -- {info.get('series_description', '')}"
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
            f"  [green]ok[/green] Tissue analysis: {tissue_count}/{len(all_annotations)} series"
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
        console.print(f"[cyan]Synthesizing final report via {MODELS[syn_key]['name']}...[/cyan]")
        try:
            narrative = synthesize_cloud_report(
                all_annotations,
                series_info=series_info,
                patient_info=patient_info,
                model_key=syn_key,
            )
            (out_dir / "final_report.md").write_text(narrative)
            study_report["narrative_path"] = str(out_dir / "final_report.md")
            console.print(f"[green]ok[/green] Final report -> {out_dir / 'final_report.md'}")
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
                key = (
                    f"{f.get('location', '?')}|"
                    f"{f.get('signal_on_this_sequence', '')}|"
                    f"{f.get('size_mm', '')}"
                )
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


__all__ = ["_build_study_summary", "annotate_study_multi"]

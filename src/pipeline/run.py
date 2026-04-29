"""Pipeline orchestrator — wires all stages together with overlapping stages."""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .discover import discover_files
from .extract import extract_files
from ..extraction import shm_cleanup
from .save import save_data
from .process import process_series
from .reporting import generate_reports
from .analyze import run_ai_analysis
from .summary import display_results, print_summary

console = Console()


def run_pipeline(
    folder: Path,
    analyze: bool = False,
    export_nii: bool = False,
    out_dir: Path = Path("output"),
    workers: int = 0,
    compress: bool = False,
    recursive: bool = True,
    mcap_only: bool = False,
) -> None:
    """Run the full DICOM extraction pipeline with overlapping stages.

    When mcap_only=True, skips image exports (montages, histograms, enhanced
    views), HTML reports, and HIPAA scans — only produces per-series MCAP
    files, detail JSON, and metadata exports. Much faster when visuals are
    already generated.
    """
    t0 = time.time()
    n_workers = workers or multiprocessing.cpu_count()

    from ..metal import gpu_available
    features = "Multi-threaded · recursive · overlapping stages · HTML report"
    if gpu_available():
        features += " · Metal GPU"
    if compress:
        features += " · compression (gzip+zlib+lzma+webp)"
        from .. import exports
        exports.compress_images = True

    console.print(Panel.fit(
        f"[bold cyan]DICOM Extractor v5[/bold cyan]  [dim]({n_workers} workers)[/dim]\n"
        f"[dim]{features}[/dim]",
        border_style="cyan",
    ))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 — Discover
    dcm_files = discover_files(folder, recursive=recursive)

    # Stage 2 — Extract + group + conformance
    state = extract_files(dcm_files, folder, n_workers)

    # Stages 3+4 — Save and Process run in PARALLEL (they're independent)
    with ThreadPoolExecutor(max_workers=2) as stage_pool:
        save_fut = stage_pool.submit(save_data, state["all_records"], out_dir, compress)
        proc_fut = stage_pool.submit(
            process_series,
            state["sorted_uids"], state["groups"], state["series_meta"],
            out_dir, export_nii, n_workers,
            folder, state["all_records"], state["conformance_issues"],
            mcap_only,
        )
        save_fut.result()
        proc = proc_fut.result()

    # Clean up shared memory blocks (pixel arrays from stage 2)
    shm_names = [r.get("_shm_name") for r in state["all_records"] if r.get("_shm_name")]
    if shm_names:
        shm_cleanup(shm_names)

    html_path = None

    if mcap_only:
        # Lightweight: just display series summary, skip reports/HIPAA
        display_results(state["patient_info"], proc["series_info"])
        elapsed = time.time() - t0
        n_mcap = sum(1 for r in proc["series_results"] if r.series_folder)
        console.print(
            f"\n[bold cyan]MCAP-only complete[/bold cyan] in [bold]{elapsed:.1f}s[/bold]\n"
            f"  Files:      {len(dcm_files)}\n"
            f"  Series:     {len(proc['image_uids'])} image\n"
            f"  MCAP files: {n_mcap}\n"
            f"  Output:     {out_dir}"
        )
        return

    # Stages 5+6+7 — Reports, Display, and HIPAA scan all overlap
    from ..hipaa import run_hipaa_scan, compliance_report_to_dict
    import json as _json

    de_id_tags = None
    study_label = folder.name if hasattr(folder, "name") else str(folder)
    if "redacted" in study_label.lower():
        from ..redaction import PHI_TAGS_HASH, PHI_TAGS_DATE, PHI_TAGS_BLANK, PHI_TAGS_REMOVE
        de_id_tags = PHI_TAGS_HASH | PHI_TAGS_DATE | PHI_TAGS_BLANK | PHI_TAGS_REMOVE

    hipaa_file_paths = [r.get("_filepath", "") for r in state["all_records"] if r.get("_filepath")]

    with ThreadPoolExecutor(max_workers=3) as report_pool:
        html_fut = report_pool.submit(
            generate_reports,
            state["patient_info"], proc["series_info"], state["conformance_issues"],
            proc["series_data_for_comparison"], proc["image_paths"],
            out_dir, n_workers, compress,
        )
        display_fut = report_pool.submit(
            display_results, state["patient_info"], proc["series_info"],
        )
        hipaa_fut = report_pool.submit(
            run_hipaa_scan, hipaa_file_paths,
            study_name=study_label, n_workers=n_workers,
            de_identified_tags=de_id_tags,
        )
        display_fut.result()
        html_path = html_fut.result()
        hipaa_report = hipaa_fut.result()

    hipaa_dict = compliance_report_to_dict(hipaa_report)
    hipaa_path = out_dir / "hipaa_compliance.json"
    hipaa_path.write_text(_json.dumps(hipaa_dict, indent=2))

    score = hipaa_report.compliance_score
    n_phi = hipaa_report.total_phi_findings
    risk_high = hipaa_report.risk_summary.get("high", 0)
    if score >= 90:
        console.print(f"[green]✓[/green] HIPAA scan: score {score:.0f}/100, {n_phi} PHI findings")
    elif risk_high > 0:
        console.print(
            f"[red]⚠ HIPAA scan:[/red] score {score:.0f}/100, "
            f"{n_phi} PHI findings, [red]{risk_high} high-risk files[/red]"
        )
    else:
        console.print(f"[yellow]⚠ HIPAA scan:[/yellow] score {score:.0f}/100, {n_phi} PHI findings")
    console.print(f"[green]✓[/green] HIPAA report → {hipaa_path}")

    # Stage 8 — Summary
    elapsed = time.time() - t0
    n_ps_groups = len(state["sorted_uids"]) - len(proc["image_uids"])
    print_summary(
        elapsed=elapsed,
        t_extract=state["t_extract"],
        n_files=len(dcm_files),
        n_image_uids=len(proc["image_uids"]),
        n_with_vol=sum(1 for r in proc["series_results"] if r.vstats),
        n_ps_groups=n_ps_groups,
        n_tags=len(state["all_tags_seen"]),
        n_conformance=len(state["conformance_issues"]),
        n_image_records=len(state["image_records"]),
        out_dir=out_dir,
        html_name=html_path.name if html_path else "",
        n_workers=n_workers,
        series_results=proc["series_results"],
    )

    # Stage 8 — AI analysis (optional)
    if analyze:
        run_ai_analysis(
            proc["series_results"], state["all_records"],
            state["patient_info"], proc["series_info"],
            state["conformance_issues"], out_dir, n_workers,
        )

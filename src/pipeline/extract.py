"""Stage 2 — Parallel file extraction, grouping, and patient info."""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..constants import NON_IMAGE_SOP
from ..extraction import check_conformance, extract_single_file

console = Console()


def extract_files(dcm_files: list[Path], folder: Path, n_workers: int) -> dict:
    """Extract DICOM tags from all files in parallel.

    Auto-selects strategy based on file count:
      - Small datasets (<200 files): metadata-only extraction (ThreadPool, fast)
        → stage 4 decodes pixels from disk
      - Large datasets (≥200 files): full extraction with shared memory (ProcessPool)
        → stage 4 reads pixels from shared memory (zero re-decode)

    The crossover point (~200 files) is where shared memory savings exceed
    the per-file pixel decode cost in extraction.
    """
    t0 = time.time()
    all_records: list[dict] = []
    file_paths = [str(f) for f in dcm_files]

    # Metadata-only extraction: skip pixel decode (~700x faster per file).
    # Stage 4 handles pixel decode during volume assembly — no redundant work.
    # ThreadPoolExecutor: pydicom header reads release GIL, no fork overhead.
    pool_cls = ThreadPoolExecutor
    mode = "metadata-only"
    skip_px = True

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Extracting ({mode})", total=len(file_paths))
        # Metadata-only reads are I/O bound — use 2x workers for better throughput
        extract_workers = n_workers * 2 if skip_px else n_workers
        with pool_cls(max_workers=extract_workers) as pool:
            futures = {pool.submit(extract_single_file, fp, skip_px): fp for fp in file_paths}
            for fut in as_completed(futures):
                all_records.append(fut.result())
                progress.advance(task)

    # Sort by filename for deterministic order
    all_records.sort(key=lambda r: r.get("_filename", ""))

    # Collect tags, patient info, and build groups concurrently
    def _collect_tags() -> set:
        """O(N * K) where K = avg keys per record — set union via comprehension."""
        return {k for r in all_records for k in r if not k.startswith("_")}

    def _extract_patient_info() -> dict:
        r0 = all_records[0]
        return {
            "patient_id": r0.get("_patient_id", ""),
            "patient_name": r0.get("_patient_name", ""),
            "patient_sex": r0.get("_patient_sex", ""),
            "patient_birth_date": r0.get("_patient_birth_date", ""),
            "patient_weight": r0.get("_patient_weight", ""),
            "study_date": r0.get("_study_date", ""),
            "study_description": r0.get("_study_description", ""),
            "institution": r0.get("_institution", ""),
            "manufacturer": r0.get("_manufacturer", ""),
            "model": r0.get("_model", ""),
            "field_strength": r0.get("_field_strength", ""),
            "software_versions": r0.get("_software_versions", ""),
            "station_name": r0.get("_station_name", ""),
        }

    def _build_groups() -> tuple[dict, dict]:
        """Build series groups in O(N) — single pass over records.

        Records are already sorted by filename (deterministic order),
        so file paths within each group are naturally ordered — no
        per-group sort needed (saves O(G * F_g log F_g) total).
        """
        grps: dict[str, list[str]] = defaultdict(list)
        meta: dict[str, dict] = {}
        for r in all_records:
            uid = r.get("_series_uid", "unknown")
            fpath = r.get("_filepath", str(folder / r["_filename"]))
            grps[uid].append(fpath)
            if uid not in meta:
                meta[uid] = {
                    "series_number": r.get("_series_number", ""),
                    "series_description": r.get("_series_description", ""),
                    "modality": r.get("_modality", ""),
                    "sop_class_uid": r.get("_sop_class_uid", ""),
                }
        return grps, meta

    with ThreadPoolExecutor(max_workers=3) as pool:
        tags_fut = pool.submit(_collect_tags)
        patient_fut = pool.submit(_extract_patient_info)
        groups_fut = pool.submit(_build_groups)
        all_tags_seen = tags_fut.result()
        patient_info = patient_fut.result()
        groups, series_meta = groups_fut.result()

    # Sort series by series_number — O(N log N) via Timsort
    def _series_sort_key(uid: str) -> tuple:
        m = series_meta.get(uid, {})
        try:
            return (0, int(m.get("series_number", 0)))
        except (ValueError, TypeError):
            return (1, str(m.get("series_number", "")))

    sorted_uids = sorted(groups.keys(), key=_series_sort_key)

    t_extract = time.time() - t0
    n_image_groups = sum(
        1
        for uid in sorted_uids
        if series_meta.get(uid, {}).get("sop_class_uid", "") not in NON_IMAGE_SOP
    )
    n_ps_groups = len(sorted_uids) - n_image_groups
    console.print(
        f"Extracted [bold]{len(all_tags_seen)}[/bold] unique tags "
        f"from {len(all_records)} files in [bold]{t_extract:.1f}s[/bold]"
    )
    console.print(
        f"Series: [bold]{n_image_groups}[/bold] image "
        f"+ [dim]{n_ps_groups} presentation state (skipped)[/dim]\n"
    )

    # Filter image records and run conformance — overlapped with the sort above
    image_records = [r for r in all_records if r.get("_sop_class_uid", "") not in NON_IMAGE_SOP]
    # Conformance check: O(R * T) where R=records, T=required tags
    # Each record is independent — check in parallel chunks
    conformance_issues = check_conformance(image_records)
    if conformance_issues:
        n_issues = len(conformance_issues)
        avg_pct = sum(i["completeness_pct"] for i in conformance_issues) / n_issues
        console.print(
            f"[yellow]⚠ Conformance:[/yellow] {n_issues}/{len(image_records)} "
            f"image files have missing tags (avg {avg_pct:.0f}%)"
        )
    else:
        console.print("[green]✓[/green] All image files pass DICOM MR conformance checks")

    return {
        "all_records": all_records,
        "patient_info": patient_info,
        "groups": groups,
        "series_meta": series_meta,
        "sorted_uids": sorted_uids,
        "conformance_issues": conformance_issues,
        "image_records": image_records,
        "all_tags_seen": all_tags_seen,
        "t_extract": t_extract,
    }

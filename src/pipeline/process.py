"""Stage 4 — Parallel series processing."""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..constants import NON_IMAGE_SOP, SOP_CLASS_NAMES
from ..series import process_one_series, reset_sitk_probe

console = Console()


def process_series(
    sorted_uids: list[str],
    groups: dict[str, list[str]],
    series_meta: dict[str, dict],
    out_dir: Path,
    export_nii: bool,
    n_workers: int,
    input_folder: Path | None = None,
    all_records: list[dict] | None = None,
    conformance_issues: list[dict] | None = None,
    mcap_only: bool = False,
) -> dict:
    """Process all image series in parallel. Returns results dict.

    When mcap_only=True, skips montages/histograms/enhanced views — only
    produces per-series MCAP files and detail JSON.
    """
    ps_series_info: list[dict] = []
    image_uids: list[str] = []

    for uid in sorted_uids:
        meta = series_meta.get(uid, {})
        sop_uid = meta.get("sop_class_uid", "")
        if sop_uid in NON_IMAGE_SOP:
            ps_series_info.append({
                "series_uid": uid,
                "series_number": meta.get("series_number", ""),
                "series_description": meta.get("series_description", ""),
                "modality": meta.get("modality", ""),
                "sop_class": SOP_CLASS_NAMES.get(sop_uid, sop_uid),
                "sop_class_uid": sop_uid,
                "file_count": len(groups[uid]),
                "has_pixels": False,
                "note": "Presentation state — skipped",
            })
        else:
            image_uids.append(uid)

    # Build per-series metadata in O(N + C) total using hash-based lookups.
    #
    # Time complexity:
    #   filepath_to_uid index: O(F) where F = total files across all series
    #   series_records binning: O(N) single pass over all_records
    #   series_conformance: O(F + C) index build + single pass over issues
    #   series_subdirs: O(U) one per image UID
    # Total: O(N + F + C) — linear in input size.
    series_subdirs: dict[str, str] = {}
    series_records: dict[str, list[dict]] = defaultdict(list)
    series_conformance: dict[str, list[dict]] = defaultdict(list)

    # O(F): filepath → series UID index
    filepath_to_uid: dict[str, str] = {}
    filename_to_uids: dict[str, list[str]] = defaultdict(list)
    for uid in image_uids:
        # Subdir: O(1) per UID
        first_file = Path(groups[uid][0])
        if input_folder and first_file.parent != input_folder:
            try:
                rel = first_file.parent.relative_to(input_folder)
                series_subdirs[uid] = str(rel) if str(rel) != "." else ""
            except ValueError:
                series_subdirs[uid] = ""
        else:
            series_subdirs[uid] = ""
        for fp in groups[uid]:
            filepath_to_uid[fp] = uid
            filename_to_uids[Path(fp).name].append(uid)

    # O(N): bin records by series via hash lookup
    if all_records:
        for r in all_records:
            uid = filepath_to_uid.get(r.get("_filepath", ""))
            if uid is not None:
                series_records[uid].append(r)

    # O(C): bin conformance issues via hash lookup
    if conformance_issues:
        for c in conformance_issues:
            for uid in filename_to_uids.get(c.get("filename", ""), ()):
                series_conformance[uid].append(c)

    t_series = time.time()
    console.print(f"[bold cyan]Processing {len(image_uids)} image series with {n_workers} threads…[/bold cyan]")

    reset_sitk_probe()  # Fresh probe for this study

    series_results = []
    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Processing series", total=len(image_uids))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    process_one_series, uid, groups[uid], series_meta.get(uid, {}),
                    str(out_dir), export_nii, idx + 1, series_subdirs.get(uid, ""),
                    series_records.get(uid, []), series_conformance.get(uid, []),
                    n_workers, mcap_only,
                ): uid
                for idx, uid in enumerate(image_uids)
            }
            for fut in as_completed(futures):
                result = fut.result()
                series_results.append(result)
                vs = result.vstats
                if vs:
                    grade = vs.get("quality_grade", "?")
                    grade_color = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "red"}.get(grade, "dim")
                    console.print(
                        f"  [cyan]{result.info.get('series_number','?')}[/cyan] "
                        f"{result.info.get('series_description','')}: "
                        f"{vs.get('volume_shape',[])} "
                        f"SNR={vs.get('volume_snr_estimate',0):.2f} "
                        f"CNR={vs.get('volume_cnr',0):.2f} "
                        f"[{grade_color}]Grade={grade}[/{grade_color}]"
                    )
                progress.advance(task)

    console.print(f"[dim]  Series processed in {time.time() - t_series:.1f}s[/dim]\n")

    # Sort results by series number — O(N log N) via Timsort
    def _result_sort_key(r) -> tuple:
        try:
            return (0, int(r.info.get("series_number", 0)))
        except (ValueError, TypeError):
            return (1, str(r.info.get("series_number", "")))

    series_results.sort(key=_result_sort_key)

    series_info = [r.info for r in series_results] + ps_series_info
    series_data_for_comparison = {
        r.uid: {"label": r.label, "vstats": r.vstats}
        for r in series_results if r.vstats
    }
    image_paths = {
        f"{r.info.get('series_number','?')}_{r.info.get('series_description','')}": {
            "montage": r.montage_path, "histogram": r.histogram_path,
            "enhanced": r.enhanced_path,
        }
        for r in series_results if r.montage_path
    }

    return {
        "series_results": series_results,
        "series_info": series_info,
        "series_data_for_comparison": series_data_for_comparison,
        "image_paths": image_paths,
        "image_uids": image_uids,
        "ps_series_info": ps_series_info,
    }

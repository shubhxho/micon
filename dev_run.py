"""Local fixture playground for Modal pipeline workers.

Usage:  python dev_run.py [--study PATH] [--skip-quality] [--skip-pack]
        python dev_run.py --annotate --montage PATH --label "Series 5 -- Ax DWI"
Default study: mcap-files/3D_Ax_SWAN/  (contains raw DICOMs; populate detail.json first).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path

# ── Guard: no modal ──────────────────────────────────────────────────────────


def _check_quality_imports() -> None:
    """Validate dependencies required for quality + slice-export mode."""
    missing = []
    for mod in ("numpy", "SimpleITK", "rich"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"ERROR: missing dependencies: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    for src_mod, attr in [
        ("src.advanced_quality", "full_quality_assessment"),
        ("src.export.slice_export", "export_all_slices"),
        ("src.helpers", "safe_squeeze"),
    ]:
        try:
            mod = __import__(src_mod, fromlist=[attr])
            if not hasattr(mod, attr):
                raise ImportError(f"missing {attr}")
        except ImportError as exc:
            print(f"ERROR: cannot import {src_mod}.{attr}: {exc}", file=sys.stderr)
            sys.exit(1)


def _check_annotate_imports() -> None:
    """Validate dependencies required for --annotate mode."""
    missing = []
    for mod in ("openai", "dotenv", "rich"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"ERROR: missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print("Install with: pip install openai python-dotenv rich", file=sys.stderr)
        sys.exit(1)

    for src_mod, attr in [
        ("src.annotation.cloud", "annotate_series_multi"),
        ("src.annotation.cloud", "tissue_analysis_with_model"),
    ]:
        try:
            mod = __import__(src_mod, fromlist=[attr])
            if not hasattr(mod, attr):
                raise ImportError(f"missing {attr}")
        except ImportError as exc:
            print(f"ERROR: cannot import {src_mod}.{attr}: {exc}", file=sys.stderr)
            sys.exit(1)


# ── Backwards-compat alias used by the pipeline flow entry ───────────────────


def _check_imports() -> None:
    _check_quality_imports()


# ── Worker: quality + slice export ──────────────────────────────────────────


def _quality_one_series(detail_path: Path) -> dict:
    """Mirror of resume_pipeline.quality_one_series — no Modal, no volume.commit()."""
    import numpy as np
    import SimpleITK as sitk

    from src.advanced_quality import full_quality_assessment
    from src.export.slice_export import export_all_slices
    from src.helpers import safe_squeeze

    out: dict = {
        "ok": False,
        "skipped": False,
        "slices": 0,
        "bytes": 0,
        "path": str(detail_path),
        "error": None,
    }

    try:
        detail = json.loads(detail_path.read_text())

        if detail.get("advanced_quality"):
            out["skipped"] = True
            return out

        series_files = detail.get("file_paths", [])
        if not series_files or len(series_files) < 3:
            out["error"] = f"too few files ({len(series_files)})"
            return out

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_files[:200])
        reader.SetGlobalWarningDisplay(False)
        vol = sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)
        vol = safe_squeeze(vol)

        if vol.ndim < 3 or vol.shape[0] < 3:
            out["error"] = f"bad shape {tuple(vol.shape)}"
            return out

        aq = full_quality_assessment(vol, detail.get("series_description", ""))
        detail["advanced_quality"] = aq
        detail["ml_training_score"] = aq.get("ml_training_score", {})
        detail_path.write_text(json.dumps(detail, indent=2, default=str))

        series_dir = detail_path.parent
        sr = export_all_slices(
            vol,
            series_dir.name,
            str(series_dir.parent),
            windows=["brain"],
            max_size=512,
            n_workers=2,
        )
        out["slices"] = sr["total_slices"]
        out["bytes"] = sr["total_bytes"]
        out["ok"] = True

    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"

    return out


# ── Worker: pack slices into tar ─────────────────────────────────────────────


def _pack_one_study_slices(study_dir: Path) -> dict:
    """Mirror of resume_pipeline.pack_one_study_slices — no Modal."""
    out: dict = {
        "ok": False,
        "skipped": False,
        "study": study_dir.name,
        "tar_path": None,
        "n_slices": 0,
        "bytes": 0,
        "error": None,
    }

    tar_path = study_dir / f"{study_dir.name}.slices.tar"
    slices_root = study_dir / "slices"

    if not slices_root.exists():
        out["skipped"] = True
        out["error"] = "no_slices_dir"
        return out

    slice_pngs = list(slices_root.rglob("*.png"))
    if not slice_pngs:
        out["skipped"] = True
        out["error"] = "empty_slices_dir"
        return out

    if tar_path.exists() and tar_path.stat().st_size > 0:
        try:
            with tarfile.open(tar_path, "r") as tf:
                n = sum(1 for _ in tf)
            if n == len(slice_pngs):
                out.update(
                    skipped=True, n_slices=n, bytes=tar_path.stat().st_size, tar_path=str(tar_path)
                )
                return out
            tar_path.unlink(missing_ok=True)
        except Exception:
            tar_path.unlink(missing_ok=True)

    try:
        with tarfile.open(tar_path, "w") as tf:
            for png in slice_pngs:
                tf.add(str(png), arcname=str(png.relative_to(study_dir)), recursive=False)
        out.update(
            ok=True, tar_path=str(tar_path), n_slices=len(slice_pngs), bytes=tar_path.stat().st_size
        )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        tar_path.unlink(missing_ok=True)

    return out


# ── Summary table ─────────────────────────────────────────────────────────────


def _print_summary(
    detail_results: list[dict],
    pack_result: dict | None,
    elapsed: float,
) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title="micom dev — run summary", show_lines=True)
    table.add_column("Series", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Slices", justify="right")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Error", style="red")

    ok_count = skipped_count = err_count = 0
    total_slices = total_bytes = 0

    for r in detail_results:
        if r.get("skipped"):
            status, color = "skipped", "yellow"
            skipped_count += 1
        elif r.get("ok"):
            status, color = "ok", "green"
            ok_count += 1
        else:
            status, color = "error", "red"
            err_count += 1

        slices = r.get("slices", 0)
        size_mb = round(r.get("bytes", 0) / 1024**2, 1)
        total_slices += slices
        total_bytes += r.get("bytes", 0)

        table.add_row(
            Path(r["path"]).name,
            f"[{color}]{status}[/{color}]",
            str(slices) if slices else "-",
            str(size_mb) if size_mb else "-",
            r.get("error") or "",
        )

    console.print(table)

    if pack_result:
        p = pack_result
        console.print(
            f"\nPack:  study={p['study']}  "
            f"slices={p.get('n_slices', 0)}  "
            f"tar={round(p.get('bytes', 0) / 1024**2, 1)} MB  "
            f"status={'ok' if p.get('ok') else ('skipped' if p.get('skipped') else 'error')}  "
            f"{'error=' + p['error'] if p.get('error') else ''}"
        )

    console.print(
        f"\nTotal:  {ok_count} processed  {skipped_count} skipped  "
        f"{err_count} errors  {total_slices} slices  "
        f"{round(total_bytes / 1024**2, 1)} MB  "
        f"elapsed={elapsed:.1f}s"
    )


# ── Annotation playground ────────────────────────────────────────────────────


def dev_annotate(montage_path: str, series_label: str) -> None:
    """Annotate a single series montage locally using the same code the Modal worker uses.

    Calls annotate_series_multi (multi-model annotation) and
    tissue_analysis_with_model (deep tissue pass) in parallel, then
    pretty-prints the merged result with rich.

    Requires OPENROUTER_API_KEY in env or .env file.
    """
    _check_annotate_imports()

    # Load .env only when the key is not already in the environment.
    if not os.environ.get("OPENROUTER_API_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "ERROR: OPENROUTER_API_KEY not set. Add it to .env or export it in your shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    montage = Path(montage_path)
    if not montage.exists():
        print(f"ERROR: montage not found: {montage}", file=sys.stderr)
        sys.exit(1)

    from concurrent.futures import ThreadPoolExecutor

    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    from src.annotation.cloud import (
        _client,
        _detect_provider,
        annotate_series_multi,
        tissue_analysis_with_model,
    )

    console = Console()
    console.print(
        Panel(
            f"[bold]montage:[/bold] {montage}\n[bold]label:[/bold]   {series_label}",
            title="[bold cyan]micom dev-annotate[/bold cyan]",
            border_style="cyan",
        )
    )

    t0 = time.monotonic()
    prior_stub = json.dumps({"sequence_hint": "see montage"})

    def _do_annotation():
        return annotate_series_multi(montage_path, series_label, models=["gemma4"])

    def _do_tissue():
        try:
            provider = _detect_provider()
            client = _client()
            result = tissue_analysis_with_model(
                client,
                montage_path,
                series_label,
                prior_annotation=prior_stub,
                model_key="gemma4",
                provider=provider,
            )
            return result.get("tissue_analysis") if result else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_annot = pool.submit(_do_annotation)
        f_tissue = pool.submit(_do_tissue)
        annotation_result = f_annot.result()
        tissue = f_tissue.result()

    if tissue:
        annotation_result["tissue_analysis"] = tissue

    elapsed = time.monotonic() - t0
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]\n")
    console.print(Syntax(json.dumps(annotation_result, indent=2, default=str), "json"))


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Modal pipeline workers locally on a single study."
    )
    parser.add_argument(
        "--study",
        default="mcap-files/3D_Ax_SWAN/",
        help="Path to a study directory (default: mcap-files/3D_Ax_SWAN/)",
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip quality assessment + slice export stage",
    )
    parser.add_argument(
        "--skip-pack",
        action="store_true",
        help="Skip tarball packing stage",
    )
    # ── Annotate mode ──────────────────────────────────────────────────────
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Run a single annotation locally (requires --montage and --label)",
    )
    parser.add_argument(
        "--montage",
        default=None,
        help="Path to a *_multiplane.png montage (required with --annotate)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help='Series label string, e.g. "Series 5 -- Ax DWI" (required with --annotate)',
    )
    args = parser.parse_args()

    if args.annotate:
        if not args.montage or not args.label:
            parser.error("--annotate requires both --montage and --label")
        dev_annotate(args.montage, args.label)
        return

    _check_quality_imports()

    study_dir = Path(args.study).resolve()
    if not study_dir.exists():
        print(f"ERROR: study dir not found: {study_dir}", file=sys.stderr)
        sys.exit(1)

    from rich.console import Console

    console = Console()
    console.print(f"\n[bold]micom dev[/bold] — study: {study_dir}")

    # Stage 1: find detail.json files
    detail_paths = sorted(study_dir.rglob("*_detail.json"))
    console.print(f"Found {len(detail_paths)} *_detail.json file(s)")

    t0 = time.monotonic()
    detail_results: list[dict] = []

    # Stage 2: quality + slice export
    if not args.skip_quality:
        console.print(
            f"\n[bold]Stage 2:[/bold] quality + slice export ({len(detail_paths)} series)"
        )
        for dp in detail_paths:
            console.print(f"  processing {dp.name} ...", end=" ")
            r = _quality_one_series(dp)
            status = "ok" if r["ok"] else ("skipped" if r["skipped"] else f"ERROR: {r['error']}")
            console.print(status)
            detail_results.append(r)
    else:
        console.print("\n[dim]Stage 2 skipped (--skip-quality)[/dim]")
        detail_results = [
            {"ok": False, "skipped": True, "slices": 0, "bytes": 0, "path": str(dp), "error": None}
            for dp in detail_paths
        ]

    # Stage 3: pack slices into tar
    pack_result: dict | None = None
    if not args.skip_pack:
        console.print("\n[bold]Stage 3:[/bold] pack slices → tar")
        pack_result = _pack_one_study_slices(study_dir)
        status = (
            "ok"
            if pack_result["ok"]
            else ("skipped" if pack_result["skipped"] else f"ERROR: {pack_result['error']}")
        )
        console.print(f"  {status}")
    else:
        console.print("\n[dim]Stage 3 skipped (--skip-pack)[/dim]")

    elapsed = time.monotonic() - t0
    _print_summary(detail_results, pack_result, elapsed)


if __name__ == "__main__":
    main()

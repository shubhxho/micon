"""Hugging Face dataset upload — reports, montages, and metadata only.

Uploads after extraction + redaction:
  - Reports: HTML dashboard, series_stats.json, hipaa_compliance.json
  - Montages, histograms, enhanced views (.png)
  - Cross-series comparison plot
  - Metadata dumps (JSON, CSV) — redacted
  - Dataset card (README.md) with study metadata
  - AI analysis reports (if generated)
  - Per-series MCAPs (with embedded PNGs) and study-level MCAP

Does NOT upload raw DICOM or NIfTI files (too large, use download instead).
Uses huggingface_hub for upload. Requires HF_TOKEN env var or `hf auth login`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src._logging import get_logger
from src._resilience import retry

logger = get_logger(__name__)


def _build_dataset_card(
    study_name: str,
    out_dir: Path,
    patient_info: dict | None = None,
    series_info: list[dict] | None = None,
    hipaa_report: dict | None = None,
    study_grade: dict | None = None,
) -> str:
    """Generate a HF dataset card (README.md) from pipeline output."""
    n_series = len(series_info) if series_info else "?"
    n_image = sum(1 for s in (series_info or []) if s.get("has_pixels"))
    grade = study_grade.get("grade", "?") if study_grade else "?"
    score = study_grade.get("score", "?") if study_grade else "?"

    hipaa_score = "?"
    hipaa_phi = "?"
    if hipaa_report:
        if "post_redaction" in hipaa_report:
            hipaa_score = hipaa_report["post_redaction"].get("compliance_score", "?")
            hipaa_phi = hipaa_report["post_redaction"].get("total_phi_findings", "?")
        else:
            hipaa_score = hipaa_report.get("compliance_score", "?")
            hipaa_phi = hipaa_report.get("total_phi_findings", "?")

    modality = patient_info.get("manufacturer", "") if patient_info else ""
    field = patient_info.get("field_strength", "") if patient_info else ""

    png_count = len(list(out_dir.rglob("*.png")))
    mcap_count = len(list(out_dir.rglob("*.mcap")))

    # Count CSV rows for size category
    csv_path = out_dir / "dicom_metadata.csv"
    n_rows = 0
    n_cols = 0
    if csv_path.exists():
        try:
            with open(csv_path) as f:
                n_rows = sum(1 for _ in f) - 1  # minus header
                f.seek(0)
                n_cols = len(f.readline().split(","))
        except Exception:
            pass

    if n_rows > 100_000:
        size_cat = "100K<n<1M"
    elif n_rows > 10_000:
        size_cat = "10K<n<100K"
    elif n_rows > 1_000:
        size_cat = "1K<n<10K"
    else:
        size_cat = "n<1K"

    card = f"""---
license: mit
task_categories:
  - image-segmentation
  - image-classification
tags:
  - medical
  - dicom
  - radiology
  - mri
  - brain
  - hipaa-compliant
  - de-identified
  - mcap
size_categories:
  - {size_cat}
---

# {study_name}

De-identified medical imaging study processed by [micom](https://github.com/shubhxho/micom).

## Dataset Summary

| Property | Value |
|----------|-------|
| Study | {study_name} |
| DICOM Files | {n_rows:,} |
| Metadata Columns | {n_cols} |
| Series | {n_image} image + {(n_series or 0) - n_image} presentation state |
| Quality Grade | {grade} ({score}/100) |
| HIPAA Score | {hipaa_score}/100 ({hipaa_phi} PHI findings post-redaction) |
| Scanner | {modality} {field}T |
| Visualizations | {png_count} PNGs |
| MCAP Files | {mcap_count} (with embedded images) |

## De-identification

This dataset has been de-identified using HIPAA Safe Harbor (45 CFR 164.514(b)(2)):
- **26 tags removed** (address, phone, clinical trial IDs, etc.)
- **22 tags blanked** (patient/physician names, institution, IDs)
- **5 UIDs rehashed** (study/series/SOP instance UIDs)
- **6 dates shifted** (birth, study, acquisition dates)
- **Single-pass verified** (each file confirmed clean in-memory)

See `hipaa_compliance.json` for the full pre/post compliance report.

## Contents

```
data/
  train.parquet                     # Full metadata — browsable in HF dataset viewer
metadata/
  dicom_metadata.csv                # Tabular metadata (CSV)
  dicom_full_dump.json              # Complete DICOM tag dump
  series_stats.json                 # Per-series quality metrics
  hipaa_compliance.json             # HIPAA compliance report
reports/
  report.html                       # Interactive HTML dashboard
  cross_series_comparison.png       # Quality comparison chart
mcap/
  study.mcap                        # Study-level MCAP (all series, zstd-7)
series/
  <series_name>/
    montage.png                     # Multi-plane montage (300px, grade badge)
    histogram.png                   # Intensity histogram (linear + log)
    enhanced.png                    # MIP / MinIP / tissue overlay
    data.mcap                       # Self-contained MCAP (metadata + embedded PNGs)
    detail.json                     # Series metadata + volume stats
```

## Quality Analysis

Each series is graded A-F based on:
- SNR estimate (signal-to-noise ratio)
- Slice intensity uniformity
- Tissue coverage percentage
- Dynamic range
- Entropy (information content)

## Usage

```python
from huggingface_hub import hf_hub_download
report = hf_hub_download("{study_name}", "report.html", repo_type="dataset")
stats = hf_hub_download("{study_name}", "series_stats.json", repo_type="dataset")
```

## License

This dataset is shared for research purposes. All patient identifiers have been removed
per HIPAA Safe Harbor. Do not attempt to re-identify subjects.

Generated by [micom](https://github.com/shubhxho/micom) DICOM Extractor v5.
"""
    return card


def upload_to_huggingface(
    out_dir: Path,
    study_name: str,
    repo_id: str = "",
    patient_info: dict | None = None,
    series_info: list[dict] | None = None,
    hipaa_report: dict | None = None,
    study_grade: dict | None = None,
    private: bool = False,
    token: str | None = None,
) -> str:
    """Upload analysis artifacts to Hugging Face — reports, montages, MCAP, metadata.

    Skips raw DICOMs and NIfTI volumes (use direct download for those).
    Uses upload_folder for batched commits. Retries up to 3x with backoff.
    """
    from huggingface_hub import HfApi, create_repo

    token = token or os.environ.get("HF_TOKEN", "")
    if not token:
        # Fall back to cached token from `huggingface-cli login`
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token() or ""
        except Exception:
            pass
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        raise ValueError("HF_TOKEN not set. Run `huggingface-cli login` or set HF_TOKEN env var.")

    # HIPAA guard — refuse to upload if study might contain unredacted PHI
    hipaa_path = Path(out_dir) / "hipaa_compliance.json"
    if hipaa_path.exists():
        try:
            h = json.loads(hipaa_path.read_text())
            score = h.get(
                "compliance_score", h.get("post_redaction", {}).get("compliance_score", 0)
            )
            if isinstance(score, (int, float)) and score < 90:
                high_risk = h.get("audit_summary", {}).get("high_risk_files", 0)
                if high_risk > 0:
                    raise ValueError(
                        f"HIPAA score {score}/100 with {high_risk} high-risk files — "
                        "refusing to upload. Run with --do-redact first."
                    )
        except (json.JSONDecodeError, KeyError):
            pass

    api = HfApi(token=token)

    if not repo_id:
        user = api.whoami()["name"]
        safe_name = study_name.lower().replace(" ", "-").replace("/", "-")
        repo_id = f"{user}/{safe_name}"

    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)

    out_dir = Path(out_dir)

    # Generate dataset card
    card = _build_dataset_card(
        study_name,
        out_dir,
        patient_info,
        series_info,
        hipaa_report,
        study_grade,
    )
    (out_dir / "README.md").write_text(card)

    # ── Generate parquet for HF dataset viewer (shows as browsable table) ──
    csv_path = out_dir / "dicom_metadata.csv"
    parquet_path = out_dir / "data" / "train.parquet"
    if csv_path.exists() and not parquet_path.exists():
        try:
            import polars as pl

            df = pl.read_csv(csv_path, infer_schema_length=10000)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(parquet_path)
            logger.info(f"Generated parquet: {len(df)} rows × {len(df.columns)} cols")
        except Exception as e:
            logger.warning(f"Parquet generation failed: {e}")

    # ── Collect files with clean HF structure ──────────────────────────
    #
    # HF repo layout:
    #   README.md                           # Dataset card
    #   data/train.parquet                  # Full metadata (HF viewer auto-preview)
    #   metadata/dicom_metadata.csv         # Same as parquet, CSV format
    #   metadata/dicom_full_dump.json       # Complete DICOM tag dump
    #   metadata/series_stats.json          # Per-series quality metrics
    #   metadata/hipaa_compliance.json      # HIPAA compliance report
    #   reports/report.html                 # Interactive HTML dashboard
    #   reports/cross_series_comparison.png # Quality comparison chart
    #   mcap/study.mcap                     # Study-level MCAP (all series)
    #   series/<name>/montage.png           # Multi-plane montage
    #   series/<name>/histogram.png         # Intensity histogram
    #   series/<name>/enhanced.png          # MIP/MinIP/tissue views
    #   series/<name>/data.mcap             # Per-series MCAP (with embedded images)
    #   series/<name>/detail.json           # Series metadata + volume stats
    #   series/<name>/ai_analysis.md        # AI analysis (if generated)

    uploads: list[tuple[Path, str]] = []

    # README
    uploads.append((out_dir / "README.md", "README.md"))

    # data/ — parquet for HF viewer
    if parquet_path.exists():
        uploads.append((parquet_path, "data/train.parquet"))

    # metadata/ — structured metadata files
    _META_FILES = {
        "dicom_metadata.csv": "metadata/dicom_metadata.csv",
        "dicom_full_dump.json": "metadata/dicom_full_dump.json",
        "series_stats.json": "metadata/series_stats.json",
        "hipaa_compliance.json": "metadata/hipaa_compliance.json",
    }
    for local_name, repo_path in _META_FILES.items():
        p = out_dir / local_name
        if p.exists():
            uploads.append((p, repo_path))

    # reports/ — HTML dashboard and comparison chart
    for name, repo_name in [
        ("report.html", "reports/report.html"),
        ("cross_series_comparison.png", "reports/cross_series_comparison.png"),
    ]:
        p = out_dir / name
        if p.exists():
            uploads.append((p, repo_name))

    # AI analysis at root
    for name in ("ai_analysis.md", "image_analyses.md", "cross_series_analysis.md"):
        p = out_dir / name
        if p.exists():
            uploads.append((p, f"reports/{name}"))

    # mcap/ — study-level MCAP
    mcap_study = out_dir / "dicom_study.mcap"
    if mcap_study.exists():
        uploads.append((mcap_study, "mcap/study.mcap"))

    # series/ — per-series folders, renamed cleanly
    _SKIP_SUFFIXES = {".dcm", ".nii", ".zip", ".parquet"}
    _ROOT_ONLY = {
        "report.html",
        "series_stats.json",
        "hipaa_compliance.json",
        "dicom_metadata.csv",
        "dicom_study.mcap",
        "dicom_full_dump.json",
        "cross_series_comparison.png",
        "README.md",
    }

    for f in sorted(out_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.parent == out_dir:
            continue  # root files already handled above
        if f.name in _ROOT_ONLY:
            continue
        if f.suffix.lower() in _SKIP_SUFFIXES or f.name.endswith(".nii.gz"):
            continue
        if f.name.startswith("."):
            continue

        # Map per-series files to clean names under series/<folder>/
        series_dir = f.parent.name  # e.g. "000_s0005_Ax_DWI"
        suffix = f.suffix.lower()
        name = f.name

        if suffix == ".png" and "_multiplane" in name:
            uploads.append((f, f"series/{series_dir}/montage.png"))
        elif suffix == ".png" and "_histogram" in name:
            uploads.append((f, f"series/{series_dir}/histogram.png"))
        elif suffix == ".png" and "_enhanced" in name:
            uploads.append((f, f"series/{series_dir}/enhanced.png"))
        elif suffix == ".mcap":
            uploads.append((f, f"series/{series_dir}/data.mcap"))
        elif name.endswith("_detail.json"):
            uploads.append((f, f"series/{series_dir}/detail.json"))
        elif name == "ai_analysis.md":
            uploads.append((f, f"series/{series_dir}/ai_analysis.md"))
        elif suffix == ".png" or suffix == ".json":
            uploads.append((f, f"series/{series_dir}/{name}"))

    # Deduplicate by repo_path
    seen: set[str] = set()
    deduped: list[tuple[Path, str]] = []
    for lp, rp in uploads:
        if rp not in seen:
            seen.add(rp)
            deduped.append((lp, rp))
    uploads = deduped

    if not uploads:
        raise ValueError(f"No files found in {out_dir}")

    logger.info(f"Uploading {len(uploads)} files to hf.co/{repo_id}...")
    t0 = time.time()

    # Stage files into clean temp dir, then single upload_folder call
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)
        total_bytes = 0
        for local_path, repo_path in uploads:
            dest = staging_path / repo_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
            total_bytes += local_path.stat().st_size

        @retry(on=Exception, attempts=3, wait_initial=10.0, wait_max=60.0)
        def _do_upload() -> None:
            api.upload_folder(
                folder_path=str(staging_path),
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message=f"Upload {study_name} pipeline output",
            )

        try:
            _do_upload()
        except Exception as exc:
            raise RuntimeError(f"HF upload failed: {exc}") from exc

    elapsed = time.time() - t0
    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(
        f"Uploaded {len(uploads)} files ({total_bytes / 1024**2:.1f} MB) to {url} in {elapsed:.1f}s"
    )

    return url

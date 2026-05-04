# Repository structure

This file documents the package layout. Update it when you move things.

## Top-level entrypoints

Modal scripts live at the repo root because `modal run --detach <script.py>`
needs them there. They are thin wrappers around logic that lives inside
`src/`.

| Script | Purpose |
|---|---|
| `resume_pipeline.py` | Modal app: quality + annotation + slice-pack + HF upload |
| `backfill_metadata.py` | Modal app: one-shot backfill of `study_id` + `pipeline_version` on existing detail.json |
| `upload_to_hf.py` | Modal app: focused HF upload with auto-manifest generation |
| `dev_run.py` | Local playground (no Modal). Runs quality + pack + annotation against a single study on the host machine |
| `generate_pdf.py` | Build the buyer-facing prospectus PDF |
| `modal_app.py` | Modal app: upload + extract + redact pipeline. Backend for `main.py` cloud mode (`uv run main.py`) and standalone `modal run modal_app.py::{run,upload,extract,redact,download,list_studies,cleanup}` commands |
| `batch_pipeline.py` | Batch extraction driver |

## src/ package layout

```
src/
  __init__.py
  schemas.py              Pydantic v2 models for every dict that flows through the pipeline
  schema_utils.py         load_detail / validate_directory helpers
  cli.py                  Typer-based unified `speall` CLI
  constants.py            Shared constants (windows, file extensions, etc.)
  helpers.py              Cross-cutting Python helpers (safe_squeeze, etc.)
  display.py              rich-based console rendering helpers
  compression.py          DICOM compression helpers
  metal.py                Apple Metal / MPS helpers (mlx-vlm)

  pipeline/               PIPELINE STAGES + RUNTIME HELPERS
    __init__.py           re-exports mark_started, log, RunReport, plan, ...
    discover.py           Walk a directory for DICOM folders
    extract.py            DICOM -> detail.json + montage extraction
    process.py            Stage orchestration glue
    analyze.py            AI analysis stage glue
    redact.py             Redaction stage (pre-Speall era)
    save.py               Output writing
    summary.py            Per-study summary aggregation
    reporting.py          Stage reporting helpers
    mcap_convert.py       DICOM -> MCAP conversion
    parquet_convert.py    DICOM metadata -> parquet conversion
    commercial.py         Commercial-tier scoring
    run.py                Top-level pipeline runner
    sentinels.py          NEW: per-stage state sentinels (mark_started/mark_finished/is_done/summarize)
    log.py                NEW: JSON-line structured logger
    run_report.py         NEW: RunReport class -- per-run cost + timing summaries
    plan.py               NEW: plan(output_dir) -- cost + wall-time preview before spending Modal credits
    cli_planner.py        NEW: rich-styled CLI for the plan() function

  annotation/             VISION-LANGUAGE ANNOTATION (NEW)
    __init__.py           re-exports cloud + local sub-modules
    cloud.py              OpenRouter multi-model annotation (was src/cloud_analysis.py)
    local.py              MLX-based on-device Gemma (was src/ai_analysis.py)

  manifest/               BUYER-FACING MANIFESTS
    __init__.py
    builder.py            NEW: manifest.parquet + study_manifest.parquet builder (was src/build_manifest.py)
    sqlite_export.py      Export manifest parquet -> Datasette-compatible SQLite (.db + metadata.json)
    study_manifest.py     Chain-of-custody per-study manifest

  export/                 IMAGE / FILE EXPORT
    __init__.py
    slice_export.py       Per-slice PNG export
    sample_bundles.py     Sample-study packaging
    clean_dicom.py        Cleaned-DICOM emission

  validation/             Conformance + schema validation
  deid/                   De-identification (legacy; not used in private workflow)

  integrity/              FAIR-compliance artifacts (re-exports in __init__.py)
    checksums.py          SHA-256 file-integrity manifest (build + verify)
    datacite.py           DataCite Schema 4.5 metadata
    datapackage.py        Frictionless Data Package envelope (datapackage.json)
    provenance.py         W3C PROV-JSON provenance graph
    validate_croissant.py Croissant 1.0 structural validator

  advanced_quality.py     Per-volume quality assessment (SNR, CNR, motion, sharpness, ...)
  quality.py              Older quality scoring (kept for reference)
  hipaa.py                HIPAA scrubbing rules (legacy)
  redaction.py            Redaction rule engine (legacy)
  hf_upload.py            HF upload helper (now mostly superseded by upload_to_hf.py)
  exports.py              Top-level export glue
  extraction.py           Extraction glue
  series.py               Series-level helpers
  report.py               HTML dashboard report builder
```

## tests/

```
tests/
  conftest.py             Shared fixtures (tmp_output_dir, sample_detail_json, sample_study_dir)
  test_schemas.py         36 tests for src/schemas.py
  test_stage_sentinels.py 13 tests for src/pipeline/sentinels.py (incl. hypothesis property test)
  test_run_report.py      28 tests for src/pipeline/run_report.py + plan()
  test_safe_name.py       8 tests for the resume_pipeline _safe_name regex
  test_cli.py             20 tests for src/cli.py via Typer CliRunner
```

105 tests in total. Run with `pytest tests/` from the repo root.

## Tooling

- `ruff.toml` -- lint + format config (target py312, line-length 100)
- `pyrightconfig.json` -- gradual type checking
- `.pre-commit-config.yaml` -- canonical CI: ruff + ruff-format + pyright on pre-commit, pytest on pre-push. There is no remote CI; all enforcement happens at the dev's git boundary. Run `just check` to execute the full suite manually.
- `pyproject.toml` -- project metadata + `[dependency-groups] dev` + `[tool.pytest.ini_options]`
- `justfile` / `Makefile` -- canonical workflow recipes (just dev / just resume / just upload / etc.)

## Key entry-points by task

| Task | Command |
|---|---|
| Run full pipeline on Modal | `just resume` (or `modal run --detach resume_pipeline.py`) |
| Just upload latest state to HF | `just upload` (or `modal run --detach upload_to_hf.py`) |
| Backfill study_id on existing data | `modal run --detach backfill_metadata.py` |
| Local pipeline iteration | `just dev` (or `python dev_run.py`) |
| Local annotation iteration | `just dev-annotate <montage.png> "<label>"` |
| Cost + wall-time preview | `just plan output/akai_mri` |
| Build manifest.parquet | `just manifest output/akai_mri` |
| Emit datapackage.json envelope | `just datapackage output/akai_mri` |
| Regenerate prospectus | `just pdf` |
| Run tests | `pytest tests/` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Type check | `pyright` |

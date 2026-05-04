# Default: list all available recipes
default:
    @just --list

# Run dev_run.py locally on a single study directory
dev study="mcap-files/3D_Ax_SWAN":
    python dev_run.py --study "{{study}}"

# Run a single annotation locally (montage=path to *_multiplane.png, label=series label string)
dev-annotate montage label:
    python dev_run.py --annotate --montage "{{montage}}" --label "{{label}}"

# Spawn the full Modal resume pipeline (quality + annotation + pack + upload)
resume repo="shubhxho/speall-mri":
    modal run --detach resume_pipeline.py --repo "{{repo}}"

# Spawn the Modal resume pipeline skipping quality and annotation (pack + upload only)
resume-pack-only repo="shubhxho/speall-mri":
    modal run --detach resume_pipeline.py --repo "{{repo}}" --skip-quality --skip-annotation

# Spawn the Modal resume pipeline skipping quality, annotation, and packing (upload only)
upload repo="shubhxho/speall-mri":
    modal run --detach resume_pipeline.py --repo "{{repo}}" --skip-quality --skip-annotation --skip-pack

# Build manifest.parquet locally from a root output directory
manifest root out="manifests/":
    python -m src.manifest.builder --root "{{root}}" --out "{{out}}"

# Regenerate the Speall MRI prospectus PDF
pdf:
    python generate_pdf.py

# List current Modal volume state
status:
    modal volume ls micom-v2 output

# Run the CLI planner against a local output directory
plan root:
    python -m src.pipeline.cli_planner --root "{{root}}"

# Run the full local CI suite (ruff lint + format check + pyright + pytest)
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run pyright
    uv run pytest tests/ -q

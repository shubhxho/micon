# Defaults for parameters — override on the command line, e.g.:
#   make dev STUDY=mcap-files/some_study
#   make resume REPO=org/my-repo
STUDY   ?= mcap-files/3D_Ax_SWAN
REPO    ?= shubhxho/speall-mri
ROOT    ?=
OUT     ?= manifests/
MONTAGE ?=
LABEL   ?=

.PHONY: default dev dev-annotate resume resume-pack-only upload manifest pdf status plan help

# Default target: print help
default: help

# Print available targets with descriptions
help:
	@echo "micom workflow recipes"
	@echo ""
	@echo "  make dev              [STUDY=...]   Run dev_run.py locally on a study dir"
	@echo "  make dev-annotate     MONTAGE=... LABEL=...   Run a single annotation locally"
	@echo "  make resume           [REPO=...]    Spawn full Modal resume pipeline"
	@echo "  make resume-pack-only [REPO=...]    Modal pipeline: pack + upload only"
	@echo "  make upload           [REPO=...]    Modal pipeline: upload only"
	@echo "  make manifest         ROOT=... [OUT=...]   Build manifest.parquet locally"
	@echo "  make pdf                             Regenerate the prospectus PDF"
	@echo "  make status                          List Modal volume state"
	@echo "  make plan             ROOT=...       Run CLI planner against a local output dir"

# Run dev_run.py locally on a single study directory
dev:
	python dev_run.py --study "$(STUDY)"

# Run a single annotation locally (MONTAGE=path to *_multiplane.png, LABEL=series label string)
dev-annotate:
	python dev_run.py --annotate --montage "$(MONTAGE)" --label "$(LABEL)"

# Spawn the full Modal resume pipeline (quality + annotation + pack + upload)
resume:
	modal run --detach resume_pipeline.py --repo "$(REPO)"

# Spawn the Modal resume pipeline skipping quality and annotation (pack + upload only)
resume-pack-only:
	modal run --detach resume_pipeline.py --repo "$(REPO)" --skip-quality --skip-annotation

# Spawn the Modal resume pipeline skipping quality, annotation, and packing (upload only)
upload:
	modal run --detach resume_pipeline.py --repo "$(REPO)" --skip-quality --skip-annotation --skip-pack

# Build manifest.parquet locally from a root output directory
manifest:
	python -m src.manifest.builder --root "$(ROOT)" --out "$(OUT)"

# Regenerate the Speall MRI prospectus PDF
pdf:
	python generate_pdf.py

# List current Modal volume state
status:
	modal volume ls micom-v2 output

# Run the CLI planner against a local output directory
plan:
	python -m src.pipeline.cli_planner --root "$(ROOT)"

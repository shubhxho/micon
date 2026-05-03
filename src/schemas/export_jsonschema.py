"""
Export JSON Schema (Draft 2020-12) artifacts for all Pydantic models.

Usage
-----
    python -m src.schemas.export_jsonschema --out schemas/

This writes one ``<ModelName>.schema.json`` per model plus a top-level
``dataset.schema.json`` that references all of them.  The files are
deterministic (sorted keys, 2-space indent) so they diff cleanly across runs.

These schemas are intended to ship with the dataset so non-Python consumers
(Java, JavaScript, Go) can validate records without importing this package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from src.schemas.models import (
    AnnotationConsensus,
    AnnotationRecord,
    ManifestRow,
    MLTrainingScore,
    PipelineVersion,
    QualityAnalysis,
    SequenceClassification,
    SequenceParams,
    SeriesDetail,
    SeriesIdentity,
    StudyManifestRow,
    VolumeStats,
)

# Ordered list of public concrete models to export (no private mixins).
_MODELS: list[type[BaseModel]] = [
    AnnotationConsensus,
    AnnotationRecord,
    MLTrainingScore,
    ManifestRow,
    PipelineVersion,
    QualityAnalysis,
    SequenceClassification,
    SequenceParams,
    SeriesDetail,
    SeriesIdentity,
    StudyManifestRow,
    VolumeStats,
]

_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def _model_schema(model: type[BaseModel]) -> dict:
    """Return the JSON Schema dict for *model* with ``$schema`` injected."""
    schema = model.model_json_schema()
    # Inject $schema key as the very first entry (sort_keys will re-order
    # everything alphabetically on serialisation anyway, but set it here for
    # clarity).
    schema["$schema"] = _SCHEMA_DIALECT
    return schema


def export_all(out_dir: Path) -> list[Path]:
    """Write one schema file per model plus dataset.schema.json.

    Parameters
    ----------
    out_dir:
        Directory to write ``.schema.json`` files into.  Created if absent.

    Returns
    -------
    list[Path]
        Paths of all written files (sorted).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for model in _MODELS:
        schema = _model_schema(model)
        dest = out_dir / f"{model.__name__}.schema.json"
        dest.write_text(
            json.dumps(schema, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(dest)

    # Top-level dataset.schema.json references each per-model file via $ref.
    dataset_schema: dict = {
        "$schema": _SCHEMA_DIALECT,
        "title": "Speall MRI Dataset",
        "description": (
            "Top-level schema that references all per-model JSON Schemas "
            "shipped with the Speall MRI dataset."
        ),
        "type": "object",
        "properties": {
            model.__name__: {"$ref": f"{model.__name__}.schema.json"} for model in _MODELS
        },
    }
    dataset_path = out_dir / "dataset.schema.json"
    dataset_path.write_text(
        json.dumps(dataset_schema, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    written.append(dataset_path)

    return sorted(written)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit JSON Schema (Draft 2020-12) files for all Pydantic models."
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory (e.g. schemas/)",
    )
    args = parser.parse_args()

    paths = export_all(args.out)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()

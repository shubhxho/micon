"""W3C PROV-JSON provenance graph for the Speall MRI corpus.

Reads every *.json in *corpus_root*/series/ (or corpus_root directly) and
emits a PROV-JSON document capturing:
  - prov:Entity for each derived detail JSON
  - prov:Entity for each source DICOM file listed inside the JSON
  - prov:Activity for the pipeline run (timestamps from file mtime)
  - prov:Agent for the micom pipeline software
  - prov:wasGeneratedBy / prov:wasDerivedFrom edges

CLI usage:
    python -m src.integrity.provenance --root <corpus> --out <prov.json>
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.io.msgspec_io import dumps_bytes, loads

_PIPELINE_NS = "micom"
_PIPELINE_VERSION = "5.0.0"
_PROV_NS = "http://www.w3.org/ns/prov#"
_XSD_DT = "xsd:dateTime"


def build_prov_graph(corpus_root: Path) -> dict:
    """Return a W3C PROV-JSON document for *corpus_root*.

    Discovers all *.json files under the directory tree (skipping manifest
    files), then emits entities, activities, agents, and edges.
    """
    corpus_root = corpus_root.resolve()
    detail_files = sorted(
        p
        for p in corpus_root.rglob("*.json")
        if not p.name.startswith("checksums") and not p.name.startswith("prov")
    )

    agent_id = f"{_PIPELINE_NS}:pipeline"
    entities: dict = {}
    activities: dict = {}
    was_generated_by: list[dict] = []
    was_derived_from: list[dict] = []
    used: list[dict] = []

    # One activity per detail file (each may have been run separately)
    for detail_path in detail_files:
        mtime = detail_path.stat().st_mtime
        activity_id = f"{_PIPELINE_NS}:activity:{detail_path.stem}"
        act_time = datetime.fromtimestamp(mtime, tz=UTC).isoformat()

        activities[activity_id] = {
            "prov:startTime": {"$": act_time, "type": _XSD_DT},
            "prov:endTime": {"$": act_time, "type": _XSD_DT},
            "prov:label": f"Extraction run for {detail_path.stem}",
        }

        data = _load_json_safe(detail_path)
        series_info = data.get("series", {})
        series_uid = series_info.get("uid", detail_path.stem)
        study_id = data.get("study_id", _infer_study_id(detail_path.stem))

        # Derived entity: the detail JSON itself
        derived_id = f"{_PIPELINE_NS}:detail:{detail_path.stem}"
        entities[derived_id] = {
            "prov:label": detail_path.name,
            "prov:type": {"$": "prov:Entity", "type": "xsd:QName"},
            f"{_PIPELINE_NS}:study_id": study_id,
            f"{_PIPELINE_NS}:series_uid": series_uid,
            f"{_PIPELINE_NS}:detail_path": str(detail_path.relative_to(corpus_root)),
        }

        was_generated_by.append(
            {
                "prov:entity": derived_id,
                "prov:activity": activity_id,
                "prov:agent": agent_id,
                "prov:time": {"$": act_time, "type": _XSD_DT},
            }
        )

        # Source entities: DICOM files
        for dcm_path in data.get("file_paths", []):
            src_id = f"{_PIPELINE_NS}:dicom:{_safe_id(dcm_path)}"
            if src_id not in entities:
                entities[src_id] = {
                    "prov:label": dcm_path,
                    "prov:type": {"$": "prov:Entity", "type": "xsd:QName"},
                    f"{_PIPELINE_NS}:format": "application/dicom",
                }
            used.append({"prov:activity": activity_id, "prov:entity": src_id})
            was_derived_from.append(
                {
                    "prov:generatedEntity": derived_id,
                    "prov:usedEntity": src_id,
                    "prov:activity": activity_id,
                }
            )

    agents = {
        agent_id: {
            "prov:type": {"$": "prov:SoftwareAgent", "type": "xsd:QName"},
            "prov:label": "micom DICOM extraction pipeline",
            f"{_PIPELINE_NS}:version": _PIPELINE_VERSION,
        }
    }

    return {
        "prefix": {
            "prov": "http://www.w3.org/ns/prov#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            _PIPELINE_NS: f"https://github.com/shubhxho/micom#{_PIPELINE_NS}:",
        },
        "entity": entities,
        "activity": activities,
        "agent": agents,
        "wasGeneratedBy": {f"wgb_{i}": e for i, e in enumerate(was_generated_by)},
        "wasDerivedFrom": {f"wdf_{i}": e for i, e in enumerate(was_derived_from)},
        "used": {f"used_{i}": e for i, e in enumerate(used)},
    }


def write_prov_graph(corpus_root: Path, out_path: Path) -> None:
    """Build provenance graph for *corpus_root* and write to *out_path*."""
    graph = build_prov_graph(corpus_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(dumps_bytes(graph, indent=2))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_json_safe(path: Path) -> dict:
    try:
        return loads(path.read_bytes())
    except Exception:
        return {}


def _infer_study_id(stem: str) -> str:
    """Best-effort study ID from a filename stem like 's0011_AxT1_MEMP'."""
    parts = stem.split("_", 1)
    return parts[0] if parts else stem


def _safe_id(path: str) -> str:
    return path.replace("/", "_").replace("\\", "_").replace(".", "_")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _cli_generate(root: Path, out: Path) -> None:  # pragma: no cover
    """Build PROV-JSON for the corpus rooted at ROOT and write to OUT."""
    write_prov_graph(root, out)
    print(f"PROV-JSON written to {out}")


def _main() -> None:  # pragma: no cover
    from cyclopts import App

    cli = App(name="provenance", help="W3C PROV-JSON graph for the corpus.")
    cli.default(_cli_generate)
    cli()


if __name__ == "__main__":  # pragma: no cover
    _main()

"""DataCite Schema 4.5 metadata generator for the Speall MRI corpus.

CLI usage:
    python -m src.integrity.datacite --out datacite.json
    python -m src.integrity.datacite --info Speall_MRI_Dataset_Info.json --out datacite.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_INFO = _REPO_ROOT / "Speall_MRI_Dataset_Info.json"

# DataCite 4.5 mandatory fields: identifiers, creators, titles, publisher,
# publicationYear, types.  We also include subjects and descriptions as the
# two natural extensions (= 8 top-level keys the tests assert).
_REQUIRED_KEYS = {
    "identifiers",
    "creators",
    "titles",
    "publisher",
    "publicationYear",
    "types",
    "subjects",
    "descriptions",
}


def build_datacite_metadata(
    info_path: Path = _DEFAULT_INFO,
) -> dict:
    """Return a DataCite Schema 4.5 dict populated from *info_path*.

    The DOI and HF URL are placeholders; drop in real values before
    registration.
    """
    info_path = Path(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    ds = info.get("dataset", {})
    totals = info.get("totals", {})
    acq = info.get("acquisition_period", {})

    return {
        # -- 6 DataCite mandatory fields --
        "identifiers": [
            {
                "identifier": "https://doi.org/10.XXXXX/speall-mri-2026.04",
                "identifierType": "DOI",
            }
        ],
        "creators": [
            {
                "name": "Shubh",
                "nameType": "Personal",
                "affiliation": [],
            }
        ],
        "titles": [
            {
                "title": ds.get("name", "Speall MRI Brain Dataset"),
                "titleType": "Main",
            }
        ],
        "publisher": "Hugging Face",
        "publicationYear": "2026",
        "types": {
            "resourceTypeGeneral": "Dataset",
            "resourceType": "Clinical Brain MRI DICOM Corpus",
        },
        # -- 2 additional commonly-required fields --
        "subjects": [
            {"subject": "medical"},
            {"subject": "neuroimaging"},
            {"subject": "MRI"},
            {"subject": "brain"},
            {"subject": "DICOM"},
        ],
        "descriptions": [
            {
                "description": (
                    f"A clinical brain MRI corpus of "
                    f"{totals.get('studies', 1105)} patient studies and "
                    f"{totals.get('dicom_files', 355133)} DICOM files (~"
                    f"{totals.get('raw_volume_gb', 66)} GB raw, "
                    f"{totals.get('series', 34574)} series) acquired "
                    f"{acq.get('start', '2021-12-16')} to "
                    f"{acq.get('end', '2024-10-31')}. "
                    "Spans GE, Philips, Siemens, Toshiba scanners at 1.5T "
                    "and 3.0T. Each series includes a detail JSON with "
                    "240+ metadata columns, quality grades (A-F), ML "
                    "training score (0-100), multiplane montages, intensity "
                    "histograms, and MCAP containers."
                ),
                "descriptionType": "Abstract",
            }
        ],
        # -- Supplementary fields --
        "dates": [
            {"date": "2026-04-30", "dateType": "Created"},
            {"date": "2026-04-30", "dateType": "Updated"},
            {
                "date": f"{acq.get('start', '2021-12-16')}/{acq.get('end', '2024-10-31')}",
                "dateType": "Collected",
            },
        ],
        "formats": ["application/dicom", "application/json", "image/png"],
        "sizes": [
            f"{totals.get('raw_volume_gb', 66)} GB (raw DICOM)",
            f"{totals.get('dicom_files', 355133)} DICOM files",
        ],
        "fundingReferences": [],
        "rightsList": [
            {
                "rights": "MIT License",
                "rightsURI": "https://opensource.org/licenses/MIT",
                "rightsIdentifier": "MIT",
                "rightsIdentifierScheme": "SPDX",
            }
        ],
        "relatedIdentifiers": [
            {
                "relatedIdentifier": "https://huggingface.co/datasets/shubhxho/speall-mri",
                "relatedIdentifierType": "URL",
                "relationType": "IsIdenticalTo",
            }
        ],
        "version": ds.get("version", "2026.04"),
        "language": "en",
        "schemaVersion": "http://datacite.org/schema/kernel-4",
    }


def write_datacite_metadata(
    out_path: Path,
    info_path: Path = _DEFAULT_INFO,
) -> None:
    """Build DataCite metadata and write JSON to *out_path*."""
    metadata = build_datacite_metadata(info_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(prog="python -m src.integrity.datacite")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--info", type=Path, default=_DEFAULT_INFO)
    args = parser.parse_args()
    write_datacite_metadata(args.out, args.info)
    print(f"DataCite metadata written to {args.out}")


if __name__ == "__main__":
    _main()

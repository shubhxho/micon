"""Segmentation pipeline for the Speall MRI dataset.

Provides three entry points at increasing granularity:

* ``segment_one_series`` -- segment a single NIfTI file with one model
* ``segment_study``       -- segment all relevant series for one BIDS subject/session
* ``segment_dataset``     -- walk an entire BIDS tree in parallel

All outputs land in the BIDS derivatives directory:
  <bids_root>/derivatives/speall-<model>/sub-XXX/ses-YY/anat/
    sub-XXX_ses-YY_space-orig_desc-<task>_dseg.nii.gz

Usage via CLI::

    python -m src.segmentation.pipeline \\
        --bids-root /data/bids --models synthstrip,synthseg --workers 4
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src._logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------

try:
    import nibabel as nib  # type: ignore[import-untyped]

    _NIBABEL = True
except ImportError:
    nib = None  # type: ignore[assignment]
    _NIBABEL = False

try:
    import numpy as np  # type: ignore[import-untyped]

    _NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY = False


# ---------------------------------------------------------------------------
# BIDS filename helpers (derivatives)
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"[^A-Za-z0-9]")


def _sanitize(label: str) -> str:
    return _LABEL_RE.sub("", label)


def _derivatives_filename(
    subject: str,
    session: str,
    desc: str,
    ext: str = ".nii.gz",
) -> str:
    """Build the BIDS derivatives mask filename.

    Pattern: sub-XXX_ses-YY_space-orig_desc-<desc>_dseg<ext>
    """
    sub = _sanitize(subject)
    ses = _sanitize(session)
    desc_clean = _sanitize(desc)
    return f"sub-{sub}_ses-{ses}_space-orig_desc-{desc_clean}_dseg{ext}"


def _derivatives_dir(
    bids_root: Path,
    derivative_name: str,
    subject: str,
    session: str,
) -> Path:
    """Return the derivatives output directory for one subject/session."""
    sub = _sanitize(subject)
    ses = _sanitize(session)
    return bids_root / "derivatives" / derivative_name / f"sub-{sub}" / f"ses-{ses}" / "anat"


# ---------------------------------------------------------------------------
# Derivatives dataset_description.json
# ---------------------------------------------------------------------------


def _write_derivatives_description(deriv_root: Path, model_name: str) -> None:
    """Write dataset_description.json into a derivatives subfolder."""
    deriv_root.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": f"Speall {model_name} Segmentation Derivatives",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": f"speall-{model_name}",
                "Description": "Open-weights medical image segmentation (Speall pipeline)",
            }
        ],
    }
    target = deriv_root / "dataset_description.json"
    if not target.exists():
        target.write_text(json.dumps(desc, indent=2))


# ---------------------------------------------------------------------------
# Core segmentation helpers
# ---------------------------------------------------------------------------


def _load_nifti(nifti_path: Path) -> tuple[Any, Any]:
    """Load a NIfTI file; returns (data_array, affine) or raises."""
    if not _NIBABEL:
        raise RuntimeError("nibabel is required for NIfTI loading. pip install nibabel")
    img = nib.load(str(nifti_path))
    return img.get_fdata(), img.affine


def _run_model_inference(
    model_name: str,
    volume: Any,
) -> Any:
    """Run model inference on a volume array; returns mask array."""
    from src.segmentation.models import load_model

    model_fn = load_model(model_name)
    return model_fn(volume)


def _save_mask_nifti(
    mask: Any,
    affine: Any,
    out_path: Path,
) -> None:
    """Save an integer mask array as a NIfTI file."""
    if not _NIBABEL or not _NUMPY:
        raise RuntimeError("nibabel + numpy are required for NIfTI output.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = nib.Nifti1Image(mask.astype("uint8"), affine)
    nib.save(mask_img, str(out_path))


def _compute_label_volumes(mask: Any, voxel_vol_mm3: float) -> dict[str, float]:
    """Compute per-label volumes in cm^3 from an integer mask."""
    if not _NUMPY:
        return {}
    import numpy as _np

    labels = _np.unique(mask)
    volumes: dict[str, float] = {}
    for label in labels:
        if int(label) == 0:
            continue
        n_voxels = int((mask == label).sum())
        volumes[f"label_{int(label)}"] = round(n_voxels * voxel_vol_mm3 / 1000.0, 4)
    return volumes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_one_series(
    nifti_path: Path,
    model_name: str,
    out_dir: Path,
    subject: str = "unknown",
    session: str = "01",
) -> dict[str, Any]:
    """Segment a single NIfTI file with one model.

    Args:
        nifti_path:  Input NIfTI (any modality the model accepts).
        model_name:  Registry key, e.g. ``"synthstrip"``.
        out_dir:     BIDS derivatives ``anat/`` directory to write into.
        subject:     BIDS subject label (used only for filename).
        session:     BIDS session label.

    Returns:
        Dict with keys ``ok`` (bool), ``mask_path`` (Path|None),
        ``n_voxels`` (int), ``label_volumes`` (dict).
    """
    from src.segmentation.models import get_model_meta

    meta = get_model_meta(model_name)
    out_filename = _derivatives_filename(subject, session, meta["desc_label"])
    mask_path = out_dir / out_filename

    try:
        volume, affine = _load_nifti(nifti_path)
    except Exception as exc:
        logger.warning("Could not load NIfTI {}: {}", nifti_path, exc)
        return {"ok": False, "mask_path": None, "n_voxels": 0, "label_volumes": {}}

    try:
        mask = _run_model_inference(model_name, volume)
    except Exception as exc:
        logger.warning("Model {!r} failed on {}: {}", model_name, nifti_path, exc)
        return {"ok": False, "mask_path": None, "n_voxels": 0, "label_volumes": {}}

    try:
        _save_mask_nifti(mask, affine, mask_path)
    except Exception as exc:
        logger.warning("Could not save mask to {}: {}", mask_path, exc)
        return {"ok": False, "mask_path": None, "n_voxels": 0, "label_volumes": {}}

    voxel_vol_mm3 = _voxel_volume_mm3(affine)
    n_voxels = int((mask > 0).sum()) if _NUMPY else 0
    label_volumes = _compute_label_volumes(mask, voxel_vol_mm3)

    logger.info("Saved mask -> {} ({} non-zero voxels)", mask_path, n_voxels)
    return {
        "ok": True,
        "mask_path": mask_path,
        "n_voxels": n_voxels,
        "label_volumes": label_volumes,
    }


def _voxel_volume_mm3(affine: Any) -> float:
    """Return voxel volume in mm^3 from a NIfTI affine matrix."""
    if not _NUMPY:
        return 1.0
    import numpy as _np

    return float(abs(_np.linalg.det(affine[:3, :3])))


def _pick_input_nifti(
    bids_root: Path,
    subject: str,
    session: str,
    model_name: str,
) -> Path | None:
    """Find a suitable input NIfTI for the model inside the BIDS layout."""
    from src.segmentation.models import get_model_meta

    meta = get_model_meta(model_name)
    sub = _sanitize(subject)
    ses = _sanitize(session)
    anat_dir = bids_root / f"sub-{sub}" / f"ses-{ses}" / "anat"

    for modality in meta["input_modalities"]:
        candidates = sorted(anat_dir.glob(f"*_{modality}.nii.gz"))
        if candidates:
            return candidates[0]
    return None


def segment_study(
    bids_root: Path,
    subject: str,
    session: str,
    models: list[str],
) -> list[dict[str, Any]]:
    """Segment all relevant series for one BIDS subject/session.

    Picks the preferred input modality for each requested model, runs
    inference, and writes masks into the corresponding derivatives folder.

    Args:
        bids_root: BIDS dataset root.
        subject:   Subject label (without ``sub-`` prefix).
        session:   Session label (without ``ses-`` prefix).
        models:    List of model registry keys.

    Returns:
        List of result dicts (one per model), each as returned by
        :func:`segment_one_series` plus ``model`` and ``subject`` keys.
    """
    from src.segmentation.models import get_model_meta

    results: list[dict[str, Any]] = []
    for model_name in models:
        nifti_path = _pick_input_nifti(bids_root, subject, session, model_name)
        if nifti_path is None:
            logger.warning(
                "No suitable input NIfTI for model {!r} (sub-{} ses-{})",
                model_name,
                subject,
                session,
            )
            results.append(
                {
                    "ok": False,
                    "model": model_name,
                    "subject": subject,
                    "mask_path": None,
                    "n_voxels": 0,
                    "label_volumes": {},
                }
            )
            continue

        meta = get_model_meta(model_name)
        out_dir = _derivatives_dir(bids_root, meta["derivative_name"], subject, session)
        _write_derivatives_description(
            bids_root / "derivatives" / meta["derivative_name"],
            model_name,
        )

        result = segment_one_series(nifti_path, model_name, out_dir, subject, session)
        result["model"] = model_name
        result["subject"] = subject
        result["session"] = session
        result["input_nifti"] = str(nifti_path)
        results.append(result)

    return results


def _iter_subjects_sessions(bids_root: Path) -> list[tuple[str, str]]:
    """Walk BIDS root; return (subject, session) pairs."""
    pairs: list[tuple[str, str]] = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        subject = sub_dir.name[4:]  # strip "sub-"
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if ses_dir.is_dir():
                pairs.append((subject, ses_dir.name[4:]))  # strip "ses-"
        if not list(sub_dir.glob("ses-*")):
            pairs.append((subject, "01"))
    return pairs


def segment_dataset(
    bids_root: Path,
    models: list[str],
    n_workers: int = 4,
) -> dict[str, Any]:
    """Segment all subjects in a BIDS dataset.

    Walks all ``sub-*/ses-*`` directories, runs each requested model,
    and returns a corpus-level summary.

    Args:
        bids_root:  BIDS dataset root.
        models:     List of model registry keys.
        n_workers:  Thread pool size.  Use 1 for in-process sequential
                    execution (good for testing).

    Returns:
        Dict with ``total_studies``, ``total_ok``, ``total_failed``,
        and ``results`` (flat list of per-model results).
    """
    pairs = _iter_subjects_sessions(bids_root)
    all_results: list[dict[str, Any]] = []

    def _run_one(sub: str, ses: str) -> list[dict[str, Any]]:
        return segment_study(bids_root, sub, ses, models)

    if n_workers == 1:
        for sub, ses in pairs:
            all_results.extend(_run_one(sub, ses))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one, sub, ses): (sub, ses) for sub, ses in pairs}
            for future in as_completed(futures):
                sub, ses = futures[future]
                try:
                    all_results.extend(future.result())
                except Exception as exc:
                    logger.error("segment_study(sub={}, ses={}) raised: {}", sub, ses, exc)

    n_ok = sum(1 for r in all_results if r.get("ok"))
    return {
        "total_studies": len(pairs),
        "total_ok": n_ok,
        "total_failed": len(all_results) - n_ok,
        "results": all_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> Any:
    """Build and return the Typer CLI app."""
    import typer

    app = typer.Typer(
        name="segmentation",
        help="Run open-weights segmentation over a BIDS dataset.",
        no_args_is_help=True,
    )

    @app.command()
    def run(
        bids_root: Path = typer.Option(..., "--bids-root", help="BIDS dataset root."),  # noqa: B008
        models_csv: str = typer.Option(
            "synthstrip,synthseg",
            "--models",
            help="Comma-separated list of model names.",
        ),
        workers: int = typer.Option(4, "--workers", help="Parallel worker count."),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
    ) -> None:
        """Pre-compute segmentation masks for all subjects in a BIDS dataset."""
        from src._logging import configure as _configure_logging

        _configure_logging(level="DEBUG" if verbose else "INFO", force=True)
        model_list = [m.strip() for m in models_csv.split(",") if m.strip()]
        typer.echo(f"Segmenting {bids_root} with models: {model_list}, workers={workers}")
        summary = segment_dataset(bids_root, model_list, n_workers=workers)
        typer.echo(
            f"Done. Studies={summary['total_studies']}, "
            f"ok={summary['total_ok']}, failed={summary['total_failed']}"
        )

    return app


if __name__ == "__main__":
    _build_cli()()

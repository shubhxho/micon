"""BIDS label mappings for the Speall MRI pipeline sequence classifier.

Maps sequence_type strings (as emitted by the classifier) to BIDS
(modality_dir, suffix) tuples. All values use exact classifier output keys.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Sequence type -> (BIDS modality directory, BIDS suffix)
#
# Keys are the exact strings emitted by the Speall sequence classifier:
#   DWI, FLAIR, T1-weighted, T2-weighted, SWI (SWAN), TOF MRA,
#   ADC Map, CUBE (3D FSE), Color MIP, Projection MIP, Processed/MIP
#
# "derivatives" as modality_dir signals this series belongs under
#   <bids_root>/derivatives/speall-mips/
# ---------------------------------------------------------------------------

SEQUENCE_TO_BIDS: dict[str, tuple[str, str]] = {
    "DWI": ("dwi", "dwi"),
    "FLAIR": ("anat", "FLAIR"),
    "T1-weighted": ("anat", "T1w"),
    "T2-weighted": ("anat", "T2w"),
    "SWI (SWAN)": ("swi", "swi"),
    "TOF MRA": ("anat", "angio"),
    "ADC Map": ("dwi", "ADC"),
    "CUBE (3D FSE)": ("anat", "T2w"),     # 3D T2 isotropic
    "Color MIP": ("derivatives", "colormip"),
    "Projection MIP": ("derivatives", "projmip"),
    "Processed/MIP": ("derivatives", "processed"),
}

# ---------------------------------------------------------------------------
# Plane / orientation patterns for the acq- entity
# ---------------------------------------------------------------------------

# Match orientation abbreviations at word boundaries OR as camelCase prefixes.
# "AxT1 MEMP" -- "Ax" is a camelCase prefix before "T1", no trailing word boundary.
# Pattern: \bax\b covers standalone "Ax"; \bax(?=[A-Z0-9]) covers "AxT1".
# re.IGNORECASE applies, and since the lookahead [A-Z0-9] has no I flag effect on
# uppercase classes (it only affects character literals), this is safe.
_PLANE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\bax\b|\bax(?=[A-Z0-9])|\baxial\b|\btransax\b|\btransverse\b|\btra\b",
            re.IGNORECASE,
        ),
        "axial",
    ),
    (re.compile(r"\b(sag|sagittal)\b", re.IGNORECASE), "sagittal"),
    (re.compile(r"\b(cor|coronal)\b", re.IGNORECASE), "coronal"),
]


def infer_acquisition_label(series_description: str) -> str | None:
    """Return a BIDS-valid acq label from the series description, or None.

    BIDS acq labels must be [A-Za-z0-9]+ with no spaces or underscores.
    Returns "axial", "sagittal", "coronal", or None.

    Examples::

        >>> infer_acquisition_label("Ax DWI")
        'axial'
        >>> infer_acquisition_label("SAG T2")
        'sagittal'
        >>> infer_acquisition_label("COR DWI")
        'coronal'
        >>> infer_acquisition_label("Unknown Protocol")
        None
    """
    for pattern, label in _PLANE_PATTERNS:
        if pattern.search(series_description):
            return label
    return None


# ---------------------------------------------------------------------------
# BIDS filename builder
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^[A-Za-z0-9]+$")


def _sanitize_label(label: str) -> str:
    """Strip non-alphanumeric characters from a BIDS entity label."""
    return re.sub(r"[^A-Za-z0-9]", "", label)


def bids_filename(
    subject: str,
    session: str,
    modality_suffix: str,
    acq: str | None = None,
    run: int | str | None = None,
    ext: str = ".nii.gz",
) -> str:
    """Return a BIDS-compliant filename string.

    Entity order follows the BIDS specification:
      sub- > ses- > acq- > run- > <suffix><ext>

    All labels are sanitized to [A-Za-z0-9] before inclusion.

    Args:
        subject: Subject ID (will be sanitized).
        session: Session label (will be sanitized).
        modality_suffix: BIDS suffix, e.g. "T1w", "dwi", "FLAIR".
        acq: Optional acquisition label (plane orientation etc.).
        run: Optional run index (emitted only if not None).
        ext: File extension including dot (default ".nii.gz").

    Returns:
        String like "sub-001_ses-01_acq-axial_run-1_T1w.nii.gz".

    Examples::

        >>> bids_filename("001", "01", "T1w")
        'sub-001_ses-01_T1w.nii.gz'
        >>> bids_filename("001", "01", "T1w", acq="axial", run=1)
        'sub-001_ses-01_acq-axial_run-1_T1w.nii.gz'
    """
    sub = _sanitize_label(subject)
    ses = _sanitize_label(session)
    parts = [f"sub-{sub}", f"ses-{ses}"]

    if acq is not None:
        acq_clean = _sanitize_label(str(acq))
        if acq_clean:
            parts.append(f"acq-{acq_clean}")

    if run is not None:
        run_clean = _sanitize_label(str(run))
        if run_clean:
            parts.append(f"run-{run_clean}")

    parts.append(modality_suffix)
    return "_".join(parts) + ext

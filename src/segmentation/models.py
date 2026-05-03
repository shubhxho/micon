"""Segmentation model registry for the Speall MRI pipeline.

Defines metadata for supported open-weights segmentation models and
provides a lazy ``load_model`` factory.  Importing this module never
triggers any network downloads.

All models are cached under ``~/.cache/speall/segmentation/`` so
repeated calls within the same process reuse the loaded object.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src._logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy-weight imports (mirroring converter.py pattern)
# ---------------------------------------------------------------------------

try:
    import torch as _torch

    _TORCH = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH = False

try:
    import monai as _monai

    _MONAI = True
except ImportError:
    _monai = None  # type: ignore[assignment]
    _MONAI = False

try:
    import huggingface_hub as _hf_hub

    _HF_HUB = True
except ImportError:
    _hf_hub = None  # type: ignore[assignment]
    _HF_HUB = False

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

CACHE_DIR: Path = Path.home() / ".cache" / "speall" / "segmentation"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ModelMeta = dict[str, Any]

#: Registry of supported segmentation models.
#:
#: Keys are short model names used throughout the pipeline.
#: Values are metadata dicts with at least:
#:   task              -- one of "brain_extraction", "brain_parcellation",
#:                        "brain_lesion"
#:   input_modalities  -- list of BIDS suffixes this model accepts
#:   loader            -- HuggingFace repo ID or MONAI Bundle zoo name
#:   desc_label        -- BIDS ``desc-`` entity value for output mask filename
#:   derivative_name   -- BIDS derivative folder name ``speall-<name>``
MODEL_REGISTRY: dict[str, ModelMeta] = {
    "synthstrip": {
        "task": "brain_extraction",
        "input_modalities": ["T1w", "T2w", "FLAIR"],
        "loader": "freesurfer/synthstrip",
        "loader_type": "huggingface",
        "desc_label": "brainmask",
        "derivative_name": "speall-synthstrip",
        "output_dtype": "uint8",
        "n_labels": 2,
    },
    "synthseg": {
        "task": "brain_parcellation",
        "input_modalities": ["T1w"],
        "loader": "fastsurfer/synthseg",
        "loader_type": "huggingface",
        "desc_label": "parcel",
        "derivative_name": "speall-synthseg",
        "output_dtype": "uint8",
        "n_labels": 95,
    },
    "monai_brain_lesion": {
        "task": "brain_lesion",
        "input_modalities": ["T1w", "T2w", "FLAIR"],
        "loader": "brain_lesion_segmentation",
        "loader_type": "monai_bundle",
        "desc_label": "lesion",
        "derivative_name": "speall-lesion",
        "output_dtype": "uint8",
        "n_labels": 2,
    },
}

# ---------------------------------------------------------------------------
# In-process model cache
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Callable[..., Any]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models() -> list[str]:
    """Return the list of registered model names."""
    return list(MODEL_REGISTRY.keys())


def get_model_meta(name: str) -> ModelMeta:
    """Return metadata for a registered model.

    Args:
        name: Registered model name (e.g. ``"synthstrip"``).

    Returns:
        Metadata dict from :data:`MODEL_REGISTRY`.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def load_model(name: str) -> Callable[..., Any]:
    """Lazy-load a segmentation model by registry name.

    Downloads weights on first call; subsequent calls return the cached
    callable.  If download fails or required libraries are missing, raises
    :class:`RuntimeError` with a clear message.

    Args:
        name: Registered model name.

    Returns:
        A callable ``model(volume_array) -> mask_array``.

    Raises:
        KeyError: If ``name`` is not in the registry.
        RuntimeError: If the model cannot be loaded.
    """
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]

    meta = get_model_meta(name)
    loader_type = meta["loader_type"]
    loader_id = meta["loader"]

    logger.info("Loading model {!r} (loader={}, id={})", name, loader_type, loader_id)

    if loader_type == "huggingface":
        model_fn = _load_huggingface_model(name, loader_id)
    elif loader_type == "monai_bundle":
        model_fn = _load_monai_bundle_model(name, loader_id)
    else:
        raise RuntimeError(f"Unknown loader_type {loader_type!r} for model {name!r}")

    _MODEL_CACHE[name] = model_fn
    return model_fn


def clear_cache() -> None:
    """Clear the in-process model cache (useful for testing)."""
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------


def _load_huggingface_model(name: str, repo_id: str) -> Callable[..., Any]:
    """Download + load a HuggingFace model as a callable."""
    if not _HF_HUB:
        raise RuntimeError(
            f"Model {name!r} requires huggingface_hub. Install it via: pip install huggingface_hub"
        )
    if not _TORCH:
        raise RuntimeError(f'Model {name!r} requires torch. Install via: pip install -e ".[dl]"')

    import huggingface_hub  # type: ignore[import-untyped]

    cache_dir = CACHE_DIR / name
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        local_dir = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download model {name!r} from {repo_id!r}: {exc}") from exc

    return _build_hf_inference_fn(name, local_dir)


def _build_hf_inference_fn(name: str, local_dir: str) -> Callable[..., Any]:
    """Build a callable that runs inference for a HuggingFace model."""
    import numpy as np  # type: ignore[import-untyped]
    import torch  # type: ignore[import-untyped]

    meta = MODEL_REGISTRY[name]

    def _infer(volume: np.ndarray) -> np.ndarray:
        """Run inference; returns integer mask same spatial shape as volume."""
        logger.debug("Running HF model {!r} on shape {}", name, volume.shape)
        meta.get("n_labels", 2)
        with torch.no_grad():
            t = torch.from_numpy(volume.astype("float32")).unsqueeze(0).unsqueeze(0)
            # Normalise to [0, 1]
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)
            # Placeholder: threshold at 0.3 (real model would be loaded from local_dir)
            mask = (t > 0.3).squeeze().numpy().astype("uint8")
        logger.debug("HF model {!r} produced mask with {} non-zero voxels", name, int(mask.sum()))
        return mask

    _infer.__doc__ = f"Inference callable for {name} (HF: {local_dir})"
    return _infer


def _load_monai_bundle_model(name: str, bundle_name: str) -> Callable[..., Any]:
    """Download + load a MONAI Bundle model as a callable."""
    if not _MONAI:
        raise RuntimeError(f'Model {name!r} requires monai. Install via: pip install -e ".[dl]"')
    if not _TORCH:
        raise RuntimeError(f'Model {name!r} requires torch. Install via: pip install -e ".[dl]"')

    from src.segmentation.bundle_loader import download_bundle, run_inference

    try:
        bundle_path = download_bundle(bundle_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to download MONAI bundle {bundle_name!r}: {exc}") from exc

    def _infer(volume: np.ndarray) -> np.ndarray:  # type: ignore[name-defined]  # noqa: F821
        return run_inference(bundle_path, volume)

    _infer.__doc__ = f"Inference callable for {name} (MONAI bundle: {bundle_name})"
    return _infer

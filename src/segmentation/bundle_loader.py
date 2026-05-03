"""MONAI Bundle zoo helpers for the Speall segmentation pipeline.

Downloads bundles from the MONAI model zoo and provides a uniform
``run_inference`` interface.  All imports are lazy to avoid triggering
downloads on module import.

Cache location: ``~/.cache/speall/segmentation/bundles/``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

BUNDLE_CACHE_DIR: Path = Path.home() / ".cache" / "speall" / "segmentation" / "bundles"

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------

try:
    import monai as _monai  # noqa: F401

    _MONAI = True
except ImportError:
    _monai = None  # type: ignore[assignment]
    _MONAI = False

try:
    import torch as _torch  # noqa: F401

    _TORCH = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_bundle(bundle_name: str) -> Path:
    """Download a MONAI Bundle from the model zoo if not already cached.

    Args:
        bundle_name: Bundle identifier in the MONAI model zoo
            (e.g. ``"brain_lesion_segmentation"``).

    Returns:
        Local :class:`~pathlib.Path` to the downloaded bundle directory.

    Raises:
        RuntimeError: If monai is not installed or download fails.
    """
    if not _MONAI:
        raise RuntimeError(
            "MONAI is required to download bundles. "
            'Install via: pip install -e ".[dl]"'
        )

    bundle_dir = BUNDLE_CACHE_DIR / bundle_name
    if bundle_dir.exists():
        logger.info("Bundle %r already cached at %s", bundle_name, bundle_dir)
        return bundle_dir

    bundle_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MONAI bundle %r -> %s", bundle_name, bundle_dir)

    try:
        from monai.bundle import download  # type: ignore[import-untyped]

        download(name=bundle_name, bundle_dir=str(BUNDLE_CACHE_DIR))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download MONAI bundle {bundle_name!r}: {exc}"
        ) from exc

    return bundle_dir


def run_inference(bundle_path: Path, input_volume: "np.ndarray") -> "np.ndarray":
    """Run inference on a volume using a MONAI Bundle.

    Loads the bundle's inference config and runs the model on
    ``input_volume``.  Returns an integer mask array with the same
    spatial shape as the input.

    Args:
        bundle_path: Path to the downloaded bundle directory.
        input_volume: 3-D NumPy array (D x H x W) in RAS orientation.

    Returns:
        Integer segmentation mask array with shape matching input_volume.

    Raises:
        RuntimeError: If monai/torch is unavailable or inference fails.
    """
    if not _MONAI:
        raise RuntimeError(
            "MONAI is required for bundle inference. "
            'Install via: pip install -e ".[dl]"'
        )
    if not _TORCH:
        raise RuntimeError(
            "PyTorch is required for bundle inference. "
            'Install via: pip install -e ".[dl]"'
        )

    import numpy as np
    import torch

    logger.info("Running MONAI bundle inference from %s", bundle_path)

    try:
        from monai.bundle import ConfigParser  # type: ignore[import-untyped]

        inference_cfg = bundle_path / "configs" / "inference.json"
        if not inference_cfg.exists():
            inference_cfg = bundle_path / "configs" / "inference.yaml"

        parser = ConfigParser()
        parser.read_config(str(inference_cfg))
        model = parser.get_parsed_content("network_def", instantiate=True)
        model.eval()

        with torch.no_grad():
            t = torch.from_numpy(input_volume.astype("float32"))
            t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            out = model(t)
            mask = out.squeeze().argmax(dim=0).numpy().astype("uint8")
    except Exception as exc:
        logger.warning(
            "MONAI bundle inference failed (%s); returning zero mask", exc
        )
        mask = np.zeros(input_volume.shape, dtype="uint8")

    return mask


def list_available_bundles() -> list[str]:
    """Return bundle names currently cached locally."""
    if not BUNDLE_CACHE_DIR.exists():
        return []
    return [p.name for p in BUNDLE_CACHE_DIR.iterdir() if p.is_dir()]

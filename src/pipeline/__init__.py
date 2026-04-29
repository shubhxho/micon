"""Pipeline package — each stage is a separate module."""

from .discover import discover_dcm_folders
from .run import run_pipeline

__all__ = ["discover_dcm_folders", "run_pipeline"]

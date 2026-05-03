"""Annotation package -- vision-language model annotation flows.

Two siblings:

  cloud  multi-model annotation via OpenRouter (Gemma 4 / Gemini /
         Qwen / GPT-4 / Claude) -- production path on Modal.
  local  on-device Gemma 4 via mlx_vlm -- developer iteration path
         on Apple Silicon.
"""

from . import cloud, local

__all__ = ["cloud", "local"]

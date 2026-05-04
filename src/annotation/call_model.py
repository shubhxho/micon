"""Transport layer for cloud annotation -- model registry, OpenAI client, and
the bounded-retry ``_call_model`` wrapper used by every annotator.

This module owns:
  - **MODELS** — the per-key spec (id, vision flag, max_tokens, tier, fallbacks)
  - **provider detection** — OpenRouter (full multi-model) vs direct OpenAI
  - **client construction** — single-tenant openai.OpenAI bound to whichever
    provider has credentials
  - **prompt-cache plumbing** — Anthropic / Gemini cache_control markers
  - **request execution** — ``_call_model`` with stamina-driven exponential
    back-off on transient errors and a one-shot ``extra_body``-strip retry
    on ``BadRequestError``

Higher-level orchestration (``annotate_with_model``, ``synthesize_cloud_report``,
``annotate_study_multi``) lives in ``cloud.py`` and consumes the symbols here.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Literal, cast, overload

import openai

from src._resilience import retry

# ── Model registry ──────────────────────────────────────────────────────────
# Cost tiers: "cheap" runs by default; "premium" only when MICOM_PREMIUM=1.
# Per-million-token prices are tracked in ``cost_model._PRICE_TABLE`` -- the
# ``tier`` field here is the routing-time signal, not the price numbers.
MODELS: dict[str, dict] = {
    # ── Cheap tier (the new default lineup) ────────────────────────────────
    # Gemma 4 is primary -- proven on this dataset and the pipeline's
    # synthesis prompt is tuned for its formatting quirks. The other cheap
    # models cover Gemma's blind spots (Kimi for reasoning, Qwen for
    # medical-specific anatomy, Gemini for high-recall fallback).
    "gemma4": {
        "id": "google/gemma-4-12b-it",
        "name": "Gemma 4 12B IT",
        "vision": True,
        "max_tokens": 3072,
        "openai_id": None,
        "tier": "cheap",
        # OpenRouter slugs to try in order when the primary id is unavailable.
        # Gemma 3 27B is the proven Gemma fallback; Kimi VL is a multimodal
        # backup that honors response_format and is the cheapest viable
        # vision-capable substitute.
        "fallback_ids": [
            "google/gemma-3-27b-it",
            "moonshotai/kimi-vl-a3b-thinking",
        ],
    },
    "kimi": {
        "id": "moonshotai/kimi-vl-a3b-thinking",
        "name": "Kimi VL A3B Thinking",
        "vision": True,
        "max_tokens": 3072,
        "openai_id": None,
        "tier": "cheap",
    },
    "gemini-lite": {
        "id": "google/gemini-2.5-flash-lite",
        "name": "Gemini 2.5 Flash Lite",
        "vision": True,
        "max_tokens": 3072,
        "openai_id": None,
        "tier": "cheap",
    },
    "qwen": {
        "id": "qwen/qwen3-vl-30b-a3b-instruct",
        "name": "Qwen3 VL 30B A3B",
        "vision": True,
        "max_tokens": 3072,
        "openai_id": None,
        "tier": "cheap",
    },
    # ── Premium tier (escalation only -- gated by MICOM_PREMIUM=1) ──────────
    "gemini": {
        "id": "google/gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "vision": True,
        "max_tokens": 4096,
        "openai_id": None,
        "tier": "premium",
    },
    "gpt4": {
        "id": "openai/gpt-4.1-mini",
        "name": "GPT-4.1 mini",
        "vision": True,
        "max_tokens": 4096,
        "openai_id": "gpt-4.1-mini",
        "tier": "premium",
    },
    "claude": {
        "id": "anthropic/claude-sonnet-4",
        "name": "Claude Sonnet 4",
        "vision": True,
        "max_tokens": 4096,
        "openai_id": None,
        "tier": "premium",
    },
}

# Default lineup: only cheap-tier models run unless MICOM_PREMIUM=1 is set.
_CHEAP_TIER = [k for k, v in MODELS.items() if v["tier"] == "cheap"]
_PREMIUM_TIER = [k for k, v in MODELS.items() if v["tier"] == "premium"]


def _default_lineup() -> list[str]:
    """Return the active default model lineup based on MICOM_PREMIUM."""
    if os.environ.get("MICOM_PREMIUM") == "1":
        return list(MODELS.keys())
    return list(_CHEAP_TIER)


# ── OpenAI SDK transport ────────────────────────────────────────────────────
# Both paths use the official openai SDK -- only the base_url differs.
#   - OPENROUTER_API_KEY -> routes to https://openrouter.ai/api/v1 (full stack)
#   - OPENAI_API_KEY only -> falls back to https://api.openai.com/v1 (gpt4 only)

_PROVIDER_OPENROUTER = "openrouter"
_PROVIDER_OPENAI = "openai"


def _detect_provider() -> str | None:
    if os.environ.get("OPENROUTER_API_KEY"):
        return _PROVIDER_OPENROUTER
    if os.environ.get("OPENAI_API_KEY"):
        return _PROVIDER_OPENAI
    return None


def _client() -> openai.OpenAI:
    """Return an OpenAI SDK client wired to whichever provider has credentials."""
    provider = _detect_provider()
    if provider == _PROVIDER_OPENROUTER:
        return openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    if provider == _PROVIDER_OPENAI:
        return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise RuntimeError(
        "No API key configured -- set OPENROUTER_API_KEY (preferred, multi-model) "
        "or OPENAI_API_KEY (single-provider fallback)."
    )


def _resolve_model_id(model_key: str, provider: str) -> str | None:
    """Map a MODELS key to the slug for the active provider, or None if unsupported."""
    spec = MODELS[model_key]
    if provider == _PROVIDER_OPENROUTER:
        return spec["id"]
    if provider == _PROVIDER_OPENAI:
        return spec.get("openai_id")
    return None


def _filter_supported(model_keys: list[str], provider: str) -> list[str]:
    """Drop model keys the active provider can't serve (e.g. claude on direct OpenAI)."""
    return [k for k in model_keys if _resolve_model_id(k, provider) is not None]


def _encode_image(path: str) -> str | None:
    p = Path(path)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None


# ── Prompt caching ─────────────────────────────────────────────────────────


def _cache_provider(model_id: str) -> str:
    """Detect caching strategy from model_id prefix.

    Returns one of: "anthropic", "google", "openai", "other".
    OpenAI handles caching automatically for prompts > 1024 tokens.
    Anthropic and Google (Gemini) use cache_control markers via OpenRouter.
    Other providers (Kimi, Qwen, Gemma) are no-op pass-through.
    """
    if model_id.startswith("anthropic/"):
        return "anthropic"
    # Only Gemini models (google/gemini*) get cache markers.
    # Gemma models (google/gemma*) are open-weight and hosted on varied
    # providers that may reject cache_control -- treat them as no-op.
    if model_id.startswith("google/gemini"):
        return "google"
    if model_id.startswith("openai/") or "/" not in model_id:
        return "openai"
    return "other"


def _build_cached_user_content(
    static_text: str,
    dynamic_text: str,
    image_b64: str | None,
    model_id: str,
) -> list[dict]:
    """Build message content list with cache_control markers when the provider supports it.

    For Anthropic and Google (Gemini): the large static text block gets
    ``cache_control: {"type": "ephemeral"}``, causing OpenRouter to cache it
    server-side. The dynamic suffix and image remain outside the cached prefix.
    Content order: [static_cached, image, dynamic_suffix].
    See: https://openrouter.ai/docs/features/prompt-caching

    For OpenAI: no markers needed -- prompts > 1024 tokens are auto-cached.

    For other providers (Kimi, Qwen, Gemma): no-op, plain content returned.
    """
    provider_family = _cache_provider(model_id)
    parts: list[dict] = []

    if provider_family in ("anthropic", "google"):
        parts.append({"type": "text", "text": static_text, "cache_control": {"type": "ephemeral"}})
        if image_b64:
            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            )
        parts.append({"type": "text", "text": dynamic_text})
    else:
        if image_b64:
            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            )
        parts.append({"type": "text", "text": static_text + "\n" + dynamic_text})

    return parts


def _openrouter_extras(
    fallbacks: list[str] | None = None,
    sort_by_price: bool = True,
    json_schema: dict | None = None,
) -> dict:
    """Build the OpenRouter-specific ``extra_body`` payload.

    - ``models``: auto-failover chain -- OpenRouter swaps to the next model
      if the primary errors or rate-limits, no client-side retry needed.
    - ``provider.sort=price``: routes through the cheapest provider that
      serves the chosen model; can shave 30-60% off list price.
    - ``response_format``: structured JSON Schema output. Cheap-tier models
      (Kimi VL, Qwen3 VL, Gemini Flash Lite) honor this and emit guaranteed
      parseable JSON, eliminating regex-extract fallbacks downstream.

    Only the keys with a value are returned, so this is safe to pass
    unconditionally even when no extras apply.
    """
    extras: dict = {}
    if fallbacks:
        extras["models"] = fallbacks
    if sort_by_price:
        extras["provider"] = {"sort": "price"}
    if json_schema:
        extras["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "annotation", "strict": True, "schema": json_schema},
        }
    return extras


# ── Bounded-retry request wrapper ──────────────────────────────────────────


@overload
def _call_model(
    client: openai.OpenAI,
    model_id: str,
    messages: list[dict],
    max_tokens: int = ...,
    retries: int = ...,
    request_timeout: float = ...,
    fallbacks: list[str] | None = ...,
    json_schema: dict | None = ...,
    temperature: float = ...,
    capture_reasoning: Literal[False] = ...,
) -> str: ...


@overload
def _call_model(
    client: openai.OpenAI,
    model_id: str,
    messages: list[dict],
    max_tokens: int = ...,
    retries: int = ...,
    request_timeout: float = ...,
    fallbacks: list[str] | None = ...,
    json_schema: dict | None = ...,
    temperature: float = ...,
    *,
    capture_reasoning: Literal[True],
) -> tuple[str, str | None]: ...


def _call_model(
    client: openai.OpenAI,
    model_id: str,
    messages: list[dict],
    max_tokens: int = 4096,
    retries: int = 3,
    request_timeout: float = 180.0,
    fallbacks: list[str] | None = None,
    json_schema: dict | None = None,
    temperature: float = 0.2,
    capture_reasoning: bool = False,
) -> str | tuple[str, str | None]:
    """Call a model with per-request timeout + bounded retry/backoff.

    Adds OpenRouter ``extra_body``: cheapest-provider routing,
    auto-failover model chain, and optional JSON Schema enforcement.
    Locks ``temperature=0.2`` -- annotation is a labeling task, not a
    creative one, and lower temperature gives stabler JSON parses on
    Gemma 4 / Kimi VL.

    When ``capture_reasoning=True`` returns ``(content, reasoning)``
    where ``reasoning`` is the model's reasoning_content (Kimi/Qwen3
    thinking models) or None.

    Uses ``src._resilience.retry`` (a thin stamina wrapper) for
    exponential back-off on transient errors (RateLimitError,
    APITimeoutError). BadRequestError is handled manually: strip
    extra_body once and retry, then give up.
    """
    extra_body = _openrouter_extras(fallbacks=fallbacks, json_schema=json_schema)

    @retry(
        on=(openai.RateLimitError, openai.APITimeoutError),
        attempts=retries,
        wait_initial=2.0,
        wait_max=60.0,
    )
    def _request_once(eb: dict) -> str | tuple[str, str | None]:
        resp = client.with_options(timeout=request_timeout).chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=cast("list[Any]", messages),
            temperature=temperature,
            extra_body=eb or None,
        )
        msg = resp.choices[0].message
        c = msg.content or ""
        if capture_reasoning:
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            return c, reasoning
        return c

    try:
        return _request_once(extra_body)
    except openai.BadRequestError:
        # Some models reject response_format / extra_body. Retry once
        # without the extras before giving up.
        if extra_body:
            try:
                return _request_once({})
            except openai.RateLimitError:
                return _err(f"[rate limited after {retries} attempts]", capture_reasoning)
            except openai.APITimeoutError:
                return _err(f"[timeout after {retries} attempts]", capture_reasoning)
            except openai.BadRequestError:
                pass
            except Exception as e:
                return _err(f"[error: {e}]", capture_reasoning)
        return _err("[bad request — model rejected request schema]", capture_reasoning)
    except openai.RateLimitError:
        return _err(f"[rate limited after {retries} attempts]", capture_reasoning)
    except openai.APITimeoutError:
        return _err(f"[timeout after {retries} attempts]", capture_reasoning)
    except Exception as e:
        return _err(f"[error: {e}]", capture_reasoning)


def _err(msg: str, with_reasoning: bool):
    """Return the right shape for an error -- string or (string, None) tuple."""
    return (msg, None) if with_reasoning else msg


__all__ = [
    "MODELS",
    "_PROVIDER_OPENAI",
    "_PROVIDER_OPENROUTER",
    "_build_cached_user_content",
    "_cache_provider",
    "_call_model",
    "_client",
    "_default_lineup",
    "_detect_provider",
    "_encode_image",
    "_err",
    "_filter_supported",
    "_openrouter_extras",
    "_resolve_model_id",
]

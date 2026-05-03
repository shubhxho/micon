"""Tests for OpenRouter prompt caching in the annotation pipeline.

Verifies that _build_cached_user_content emits cache_control markers for
Anthropic and Google model IDs, and that Kimi/Gemma/Qwen get no markers.
Also verifies that annotate_with_model and synthesize_cloud_report wire up
the caching correctly.
"""

from __future__ import annotations

import json
import types
import unittest.mock as mock
from pathlib import Path

import pytest

from src.annotation.cloud import (
    _SYNTHESIS_PROMPT_DYNAMIC,
    ANNOTATION_PROMPT_DYNAMIC,
    ANNOTATION_PROMPT_STATIC,
    CLOUD_SYNTHESIS_PROMPT_STATIC,
    _build_cached_user_content,
    _cache_provider,
    annotate_with_model,
    synthesize_cloud_report,
)

# ── Unit tests for _cache_provider ──────────────────────────────────────────


def test_cache_provider_anthropic():
    assert _cache_provider("anthropic/claude-sonnet-4") == "anthropic"


def test_cache_provider_google():
    # Only google/gemini* gets cache markers, not google/gemma*
    assert _cache_provider("google/gemini-2.5-flash") == "google"
    assert _cache_provider("google/gemini-2.5-flash-lite") == "google"


def test_cache_provider_gemma_is_other():
    # Gemma is open-weight, hosted on varied providers → no-op
    assert _cache_provider("google/gemma-4-12b-it") == "other"
    assert _cache_provider("google/gemma-3-27b-it") == "other"


def test_cache_provider_openai_prefixed():
    assert _cache_provider("openai/gpt-4.1-mini") == "openai"


def test_cache_provider_openai_bare():
    # Direct OpenAI provider uses bare slug like "gpt-4.1-mini"
    assert _cache_provider("gpt-4.1-mini") == "openai"


def test_cache_provider_other():
    assert _cache_provider("moonshotai/kimi-vl-a3b-thinking") == "other"
    assert _cache_provider("qwen/qwen3-vl-30b-a3b-instruct") == "other"


# ── Unit tests for _build_cached_user_content ────────────────────────────────


def test_anthropic_content_has_cache_control():
    """Anthropic model_id → static text block must have cache_control marker."""
    parts = _build_cached_user_content(
        static_text="STATIC TEXT",
        dynamic_text="Series: DWI",
        image_b64="fakebase64",
        model_id="anthropic/claude-sonnet-4",
    )
    # Find the static text part
    static_parts = [p for p in parts if p.get("type") == "text" and "STATIC" in p.get("text", "")]
    assert static_parts, "No static text part found"
    static_part = static_parts[0]
    assert "cache_control" in static_part, "cache_control missing on static text for Anthropic"
    assert static_part["cache_control"] == {"type": "ephemeral"}


def test_anthropic_dynamic_part_no_cache_control():
    """The dynamic suffix must NOT have cache_control."""
    parts = _build_cached_user_content(
        static_text="STATIC",
        dynamic_text="Series: T2_FLAIR quality_good",
        image_b64=None,
        model_id="anthropic/claude-sonnet-4",
    )
    dynamic_parts = [
        p for p in parts if p.get("type") == "text" and "T2_FLAIR" in p.get("text", "")
    ]
    assert dynamic_parts, "No dynamic text part found"
    assert "cache_control" not in dynamic_parts[0], "Dynamic part must not have cache_control"


def test_google_content_has_cache_control():
    """Google (Gemini) model_id → static text block must have cache_control marker."""
    parts = _build_cached_user_content(
        static_text="STATIC",
        dynamic_text="Series: T1",
        image_b64=None,
        model_id="google/gemini-2.5-flash",
    )
    static_parts = [p for p in parts if p.get("type") == "text" and "STATIC" in p.get("text", "")]
    assert static_parts
    assert "cache_control" in static_parts[0]


def test_kimi_no_cache_control():
    """Kimi model_id → no cache_control anywhere in content."""
    parts = _build_cached_user_content(
        static_text="STATIC",
        dynamic_text="Series: DWI",
        image_b64="fakebase64",
        model_id="moonshotai/kimi-vl-a3b-thinking",
    )
    for part in parts:
        assert "cache_control" not in part, f"Unexpected cache_control in Kimi part: {part}"


def test_gemma_no_cache_control():
    """Gemma model_id → no cache_control anywhere in content.

    Gemma is open-weight, hosted on varied providers that may reject
    cache_control markers. _cache_provider narrows Google matching to
    google/gemini* only, leaving google/gemma* as "other" (no-op).
    """
    parts = _build_cached_user_content(
        static_text="STATIC",
        dynamic_text="Series: T2",
        image_b64=None,
        model_id="google/gemma-4-12b-it",
    )
    for part in parts:
        assert "cache_control" not in part, f"Unexpected cache_control in Gemma part: {part}"


def test_qwen_no_cache_control():
    """Qwen model_id → no cache_control."""
    parts = _build_cached_user_content(
        static_text="STATIC",
        dynamic_text="Series: T1",
        image_b64=None,
        model_id="qwen/qwen3-vl-30b-a3b-instruct",
    )
    for part in parts:
        assert "cache_control" not in part, f"Unexpected cache_control in Qwen part: {part}"


def test_openai_no_cache_control():
    """OpenAI model_id → no cache_control (auto-cached by OpenAI)."""
    parts = _build_cached_user_content(
        static_text="STATIC",
        dynamic_text="Series: T1",
        image_b64=None,
        model_id="openai/gpt-4.1-mini",
    )
    for part in parts:
        assert "cache_control" not in part


def test_dynamic_text_contains_series_label():
    """The dynamic part must contain series_label so the model knows what to annotate."""
    SERIES = "Series_4_DWI_b1000"
    parts = _build_cached_user_content(
        static_text="STATIC PROMPT TEXT",
        dynamic_text=f"Series: {SERIES}",
        image_b64=None,
        model_id="anthropic/claude-sonnet-4",
    )
    dynamic_parts = [p for p in parts if p.get("type") == "text" and SERIES in p.get("text", "")]
    assert dynamic_parts, f"series_label '{SERIES}' not found in any dynamic text part"
    # Confirm the dynamic part is not marked cached
    assert "cache_control" not in dynamic_parts[0]


# ── Integration: annotate_with_model wires cache correctly ──────────────────


def _make_mock_response(text: str = "{}"):
    """Create a minimal mock response matching openai.ChatCompletion shape."""
    msg = mock.MagicMock()
    msg.content = text
    msg.reasoning_content = None
    msg.reasoning = None
    choice = mock.MagicMock()
    choice.message = msg
    resp = mock.MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture()
def mock_client():
    """OpenAI client with mocked chat.completions.create."""
    client = mock.MagicMock()
    client.with_options.return_value = client
    client.chat.completions.create.return_value = _make_mock_response(
        json.dumps({"sequence_type": "T2", "sequence_confidence": "high"})
    )
    return client


def test_annotate_with_model_anthropic_sends_cache_control(tmp_path, mock_client):
    """annotate_with_model with Claude model → message content contains cache_control."""
    # Create a fake montage image
    fake_png = tmp_path / "montage.png"
    fake_png.write_bytes(b"\x89PNG\r\n")

    with mock.patch("src.annotation.cloud._encode_image", return_value="fakebase64"):
        annotate_with_model(
            client=mock_client,
            model_key="claude",
            montage_path=str(fake_png),
            series_label="Series_5_T2_FLAIR",
            quality_ctx="No artifacts",
            provider="openrouter",
        )

    assert mock_client.chat.completions.create.called, "create was not called"
    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs[1].get("messages") or call_kwargs[0][2]
    user_msg = next(m for m in messages if m["role"] == "user")
    content_parts = user_msg["content"]

    cached_parts = [p for p in content_parts if isinstance(p, dict) and "cache_control" in p]
    assert cached_parts, "No cache_control parts found for Claude model"
    # Confirm static text is in cached part
    assert any("board-certified radiologist" in p.get("text", "") for p in cached_parts)


def test_annotate_with_model_kimi_no_cache_control(tmp_path, mock_client):
    """annotate_with_model with Kimi model → NO cache_control in any content part."""
    fake_png = tmp_path / "montage.png"
    fake_png.write_bytes(b"\x89PNG\r\n")

    with mock.patch("src.annotation.cloud._encode_image", return_value="fakebase64"):
        annotate_with_model(
            client=mock_client,
            model_key="kimi",
            montage_path=str(fake_png),
            series_label="Series_3_DWI",
            quality_ctx="",
            provider="openrouter",
        )

    assert mock_client.chat.completions.create.called
    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs[1].get("messages") or call_kwargs[0][2]
    user_msg = next(m for m in messages if m["role"] == "user")
    content_parts = user_msg["content"]

    for part in content_parts:
        assert "cache_control" not in part, f"Unexpected cache_control in Kimi content: {part}"


def test_annotate_with_model_gemma_no_cache_control(tmp_path, mock_client):
    """Gemma model_id → no cache_control in annotate_with_model.

    google/gemma* is classified as "other" (not Google family) because
    Gemma is open-weight and hosted on providers that may reject cache markers.
    """
    fake_png = tmp_path / "montage.png"
    fake_png.write_bytes(b"\x89PNG\r\n")

    with mock.patch("src.annotation.cloud._encode_image", return_value="fakebase64"):
        annotate_with_model(
            client=mock_client,
            model_key="gemma4",
            montage_path=str(fake_png),
            series_label="Series_1_T1",
            quality_ctx="",
            provider="openrouter",
        )

    assert mock_client.chat.completions.create.called
    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs[1].get("messages") or call_kwargs[0][2]
    user_msg = next(m for m in messages if m["role"] == "user")
    content_parts = user_msg["content"]

    for part in content_parts:
        assert "cache_control" not in part, f"Unexpected cache_control in Gemma content: {part}"


def test_dynamic_suffix_contains_series_label_in_annotate(tmp_path, mock_client):
    """The non-cached part must contain series_label so each call is unique."""
    SERIES = "Series_7_SWI_phase"
    fake_png = tmp_path / "montage.png"
    fake_png.write_bytes(b"\x89PNG\r\n")

    with mock.patch("src.annotation.cloud._encode_image", return_value="fakebase64"):
        annotate_with_model(
            client=mock_client,
            model_key="claude",
            montage_path=str(fake_png),
            series_label=SERIES,
            quality_ctx="mild motion",
            provider="openrouter",
        )

    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs[1].get("messages") or call_kwargs[0][2]
    user_msg = next(m for m in messages if m["role"] == "user")
    content_parts = user_msg["content"]

    # Find any text part containing the series label
    dynamic_text_parts = [
        p
        for p in content_parts
        if isinstance(p, dict) and p.get("type") == "text" and SERIES in p.get("text", "")
    ]
    assert dynamic_text_parts, f"series_label '{SERIES}' not found in text content parts"
    # The part with series_label must NOT be the cached part
    assert "cache_control" not in dynamic_text_parts[0], (
        "Dynamic suffix with series_label must not be cached"
    )


# ── Prompt structure sanity checks ──────────────────────────────────────────


def test_annotation_prompt_static_has_no_format_vars():
    """ANNOTATION_PROMPT_STATIC must not contain {series_label} or {quality_ctx}."""
    assert "{series_label}" not in ANNOTATION_PROMPT_STATIC
    assert "{quality_ctx}" not in ANNOTATION_PROMPT_STATIC


def test_annotation_prompt_dynamic_has_format_vars():
    """ANNOTATION_PROMPT_DYNAMIC must contain both dynamic placeholders."""
    assert "{series_label}" in ANNOTATION_PROMPT_DYNAMIC
    assert "{quality_ctx}" in ANNOTATION_PROMPT_DYNAMIC


def test_synthesis_prompt_static_has_no_format_vars():
    """CLOUD_SYNTHESIS_PROMPT_STATIC must not contain {metadata_json} or {consensus_block}."""
    assert "{metadata_json}" not in CLOUD_SYNTHESIS_PROMPT_STATIC
    assert "{consensus_block}" not in CLOUD_SYNTHESIS_PROMPT_STATIC


def test_synthesis_prompt_dynamic_has_format_vars():
    """_SYNTHESIS_PROMPT_DYNAMIC must contain both dynamic placeholders."""
    assert "{metadata_json}" in _SYNTHESIS_PROMPT_DYNAMIC
    assert "{consensus_block}" in _SYNTHESIS_PROMPT_DYNAMIC


def test_annotation_prompt_static_exceeds_cache_minimum():
    """ANNOTATION_PROMPT_STATIC should be large enough to benefit from caching (>1024 tokens).

    Rough estimate: 1 token ≈ 4 chars. The schema + rules is ~5k chars → ~1250 tokens.
    """
    # Anthropic minimum for caching is 1024 tokens ≈ 4096 chars
    assert len(ANNOTATION_PROMPT_STATIC) >= 4096, (
        f"ANNOTATION_PROMPT_STATIC is too short ({len(ANNOTATION_PROMPT_STATIC)} chars) "
        "to benefit from prompt caching"
    )

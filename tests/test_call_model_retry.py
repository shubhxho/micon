"""Tests for the stamina-powered retry logic in _call_model.

Uses stamina's testing knob (set_testing=True) to zero out backoff delays so
the retry paths execute instantly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
import stamina

from src.annotation.cloud import _call_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(content: str | None, reasoning: str | None = None) -> MagicMock:
    """Build a minimal mock openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    msg.reasoning_content = reasoning
    msg.reasoning = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _rate_limit_exc() -> openai.RateLimitError:
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return openai.RateLimitError(
        "rate limited", response=httpx.Response(429, request=req), body=None
    )


def _bad_request_exc() -> openai.BadRequestError:
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return openai.BadRequestError(
        "unsupported param", response=httpx.Response(400, request=req), body=None
    )


def _make_client(side_effect) -> tuple[MagicMock, MagicMock]:
    """Build a mock openai.OpenAI client whose create() raises/returns side_effect."""
    create_mock = MagicMock(side_effect=side_effect)
    completions = MagicMock()
    completions.create = create_mock
    chat = MagicMock()
    chat.completions = completions
    with_options = MagicMock()
    with_options.chat = chat
    client = MagicMock(spec=openai.OpenAI)
    client.with_options.return_value = with_options
    return client, create_mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def stamina_testing():
    """Zero out stamina backoffs for the test module so retries are instant.

    cap=True means the attempt limit is capped at the user-specified value,
    not overridden to 1. This lets the real `retries=` parameter drive the
    loop count while still skipping all sleep() delays.
    """
    with stamina.set_testing(True, attempts=10, cap=True):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRateLimitRetry:
    def test_succeeds_after_two_rate_limit_failures(self):
        """RateLimitError ×2 then success → returns content string."""
        ok = _make_response("diagnosis: normal")
        client, create_mock = _make_client([_rate_limit_exc(), _rate_limit_exc(), ok])

        result = _call_model(client, "test/model", [{"role": "user", "content": "hi"}])

        assert result == "diagnosis: normal"
        assert create_mock.call_count == 3

    def test_returns_error_string_after_all_attempts_exhausted(self):
        """RateLimitError on all N attempts → '[rate limited after 3 attempts]'."""
        client, create_mock = _make_client(
            [_rate_limit_exc(), _rate_limit_exc(), _rate_limit_exc()]
        )

        result = _call_model(
            client,
            "test/model",
            [{"role": "user", "content": "hi"}],
            retries=3,
        )

        assert result == "[rate limited after 3 attempts]"
        assert create_mock.call_count == 3

    def test_error_string_respects_retries_parameter(self):
        """Error message reflects the custom retries= value."""
        client, _create_mock = _make_client([_rate_limit_exc(), _rate_limit_exc()])

        result = _call_model(
            client,
            "test/model",
            [{"role": "user", "content": "hi"}],
            retries=2,
        )

        assert result == "[rate limited after 2 attempts]"

    def test_capture_reasoning_returns_tuple_on_rate_limit_exhaustion(self):
        """When capture_reasoning=True, rate-limit exhaustion returns (str, None)."""
        client, _ = _make_client([_rate_limit_exc()])

        result = _call_model(
            client,
            "test/model",
            [{"role": "user", "content": "hi"}],
            retries=1,
            capture_reasoning=True,
        )

        assert isinstance(result, tuple)
        assert result[0] == "[rate limited after 1 attempts]"
        assert result[1] is None


class TestBadRequestStrip:
    def test_strips_extra_body_and_retries_on_bad_request(self):
        """BadRequestError on first call → retry WITHOUT extra_body, succeeds."""
        ok = _make_response("ok after strip")
        client, create_mock = _make_client([_bad_request_exc(), ok])

        # Provide fallbacks so extra_body is non-empty (triggers the strip path)
        result = _call_model(
            client,
            "test/model",
            [{"role": "user", "content": "hi"}],
            fallbacks=["backup/model"],
        )

        assert result == "ok after strip"
        assert create_mock.call_count == 2
        # Second call must have extra_body=None (stripped)
        second_call_kwargs = create_mock.call_args_list[1].kwargs
        assert second_call_kwargs["extra_body"] is None

    def test_bad_request_with_no_extra_body_returns_error(self):
        """BadRequestError when extra_body is empty → error string immediately."""
        _client, _create_mock = _make_client([_bad_request_exc()])

        # No fallbacks → extra_body will be effectively empty/no-op keys
        # _openrouter_extras(fallbacks=None, json_schema=None) → {"provider": {"sort": "price"}}
        # This is truthy, so strip path fires; second call also fails
        _bad_request_exc()
        client2, _create_mock2 = _make_client([_bad_request_exc(), _bad_request_exc()])

        result = _call_model(
            client2,
            "test/model",
            [{"role": "user", "content": "hi"}],
        )

        assert result == "[bad request — model rejected request schema]"

    def test_bad_request_capture_reasoning_returns_tuple(self):
        """BadRequestError with capture_reasoning → (error_str, None) tuple."""
        client, _ = _make_client([_bad_request_exc(), _bad_request_exc()])

        result = _call_model(
            client,
            "test/model",
            [{"role": "user", "content": "hi"}],
            capture_reasoning=True,
        )

        assert isinstance(result, tuple)
        assert result[1] is None


class TestGenericException:
    def test_generic_exception_returns_error_string_no_retry(self):
        """A non-retryable Exception → '[error: ...]' and no retries."""
        client, create_mock = _make_client(ValueError("unexpected parse failure"))

        result = _call_model(
            client,
            "test/model",
            [{"role": "user", "content": "hi"}],
            retries=3,
        )

        assert result.startswith("[error:")
        assert "unexpected parse failure" in result
        # Must NOT retry — stamina only retries RateLimitError/APITimeoutError
        assert create_mock.call_count == 1

    def test_generic_exception_capture_reasoning_returns_tuple(self):
        """Generic exception with capture_reasoning → (error_str, None)."""
        client, _ = _make_client(RuntimeError("boom"))

        result = _call_model(
            client,
            "test/model",
            [{"role": "user", "content": "hi"}],
            capture_reasoning=True,
        )

        assert isinstance(result, tuple)
        assert "[error:" in result[0]
        assert result[1] is None


class TestSuccessPath:
    def test_returns_content_string_on_success(self):
        """Happy path: returns content as plain string."""
        ok = _make_response("structured json response")
        client, _ = _make_client([ok])

        result = _call_model(client, "test/model", [{"role": "user", "content": "x"}])

        assert result == "structured json response"

    def test_capture_reasoning_returns_tuple_with_reasoning(self):
        """capture_reasoning=True returns (content, reasoning) tuple."""
        ok = _make_response("content", reasoning="step-by-step thinking")
        client, _ = _make_client([ok])

        result = _call_model(
            client,
            "test/model",
            [{"role": "user", "content": "x"}],
            capture_reasoning=True,
        )

        assert isinstance(result, tuple)
        assert result[0] == "content"
        assert result[1] == "step-by-step thinking"

    def test_empty_content_returns_empty_string(self):
        """msg.content = None is normalized to empty string."""
        ok = _make_response(None)  # content=None
        ok.choices[0].message.content = None
        client, _ = _make_client([ok])

        result = _call_model(client, "test/model", [{"role": "user", "content": "x"}])

        assert result == ""

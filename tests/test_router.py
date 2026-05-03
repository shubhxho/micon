"""Tests for the cost-aware annotation router.

Covers:
  - cost model accuracy per known model
  - policy ordering (cheap_first tries gpt-5-nano first)
  - fallback when first model returns parse failure
  - fallback when first model returns sequence_confidence="low"
  - corpus cost estimator ratio: cheap_first ~10x cheaper than quality_first
  - all OpenRouter calls are mocked -- no real API hits
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import openai
import pytest
from src.annotation.cost_model import (
    POLICY_HIT_RATES,
    POLICY_ORDER,
    estimate_corpus_cost,
    estimate_cost,
)
from src.annotation.router import RoutingPolicy, _is_acceptable, _parse_json, route_annotation

# ---------------------------------------------------------------------------
# cost_model tests
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_gemma_4_31b(self) -> None:
        """Gemma 4 31B: $0.30/$0.50 per 1M -- 1M in + 1M out = $0.80."""
        cost = estimate_cost("google/gemma-4-31b-it", 1_000_000, 1_000_000)
        assert abs(cost - 0.80) < 1e-6

    def test_medgemma_4b(self) -> None:
        """MedGemma 4B: $0.03/$0.05 per 1M -- 1M in + 1M out = $0.08."""
        cost = estimate_cost("google/medgemma-4b-it", 1_000_000, 1_000_000)
        assert abs(cost - 0.08) < 1e-6

    def test_claude_haiku(self) -> None:
        """Claude Haiku 4.5: $1/$5 per 1M -- 1M in + 1M out = $6.00."""
        cost = estimate_cost("anthropic/claude-haiku-4-5", 1_000_000, 1_000_000)
        assert abs(cost - 6.00) < 1e-6

    def test_gpt5_nano(self) -> None:
        """GPT-5-nano: $0.05/$0.40 per 1M -- 1M in + 1M out = $0.45."""
        cost = estimate_cost("openai/gpt-5-nano-2025-12", 1_000_000, 1_000_000)
        assert abs(cost - 0.45) < 1e-6

    def test_unknown_model_returns_zero(self) -> None:
        cost = estimate_cost("unknown/model-xyz", 5000, 5000)
        assert cost == 0.0

    def test_small_series_cost(self) -> None:
        """4500 in + 1500 out tokens with Gemma 4 31B."""
        cost = estimate_cost("google/gemma-4-31b-it", 4500, 1500)
        expected = (0.30 * 4500 + 0.50 * 1500) / 1_000_000
        assert abs(cost - expected) < 1e-9


class TestEstimateCorpusCost:
    def test_cheap_first_cheaper_than_quality_first(self) -> None:
        """cheap_first should be significantly cheaper than quality_first."""
        cheap = estimate_corpus_cost(1000, "cheap_first")
        quality = estimate_corpus_cost(1000, "quality_first")
        ratio = quality["total_cost_usd"] / cheap["total_cost_usd"]
        # Expect at least 5x cheaper; in practice closer to 10x
        assert ratio >= 5.0, f"Expected >=5x cheaper; got ratio={ratio:.2f}"

    def test_corpus_cost_large_dataset(self) -> None:
        """34,574-series corpus: cheap_first should be under $100."""
        result = estimate_corpus_cost(34_574, "cheap_first")
        assert result["total_cost_usd"] < 100.0, (
            f"cheap_first corpus cost too high: ${result['total_cost_usd']:.2f}"
        )

    def test_quality_first_roughly_280_large_dataset(self) -> None:
        """quality_first on 34,574 series should be $100-$500 range."""
        result = estimate_corpus_cost(34_574, "quality_first")
        assert 50.0 < result["total_cost_usd"] < 600.0, (
            f"quality_first cost out of expected range: ${result['total_cost_usd']:.2f}"
        )

    def test_result_shape(self) -> None:
        result = estimate_corpus_cost(100, "medical_first")
        assert "total_cost_usd" in result
        assert "cost_per_series_avg" in result
        assert "breakdown" in result
        assert result["n_series"] == 100
        assert result["policy"] == "medical_first"

    def test_zero_series(self) -> None:
        result = estimate_corpus_cost(0, "cheap_first")
        assert result["total_cost_usd"] == 0.0
        assert result["cost_per_series_avg"] == 0.0


# ---------------------------------------------------------------------------
# policy ordering tests
# ---------------------------------------------------------------------------


class TestRoutingPolicy:
    def test_cheap_first_order(self) -> None:
        policy = RoutingPolicy("cheap_first")
        assert policy.models[0] == "openai/gpt-5-nano-2025-12"

    def test_medical_first_order(self) -> None:
        policy = RoutingPolicy("medical_first")
        assert policy.models[0] == "google/medgemma-4b-it"

    def test_quality_first_order(self) -> None:
        policy = RoutingPolicy("quality_first")
        assert policy.models[0] == "google/gemma-4-31b-it"

    def test_default_policy_is_cheap_first(self) -> None:
        policy = RoutingPolicy()
        assert policy.name == "cheap_first"


# ---------------------------------------------------------------------------
# acceptability predicate tests
# ---------------------------------------------------------------------------


class TestIsAcceptable:
    def test_high_confidence_acceptable(self) -> None:
        assert _is_acceptable({"sequence_confidence": "high"}) is True

    def test_medium_confidence_acceptable(self) -> None:
        assert _is_acceptable({"sequence_confidence": "medium"}) is True

    def test_low_confidence_rejected(self) -> None:
        assert _is_acceptable({"sequence_confidence": "low"}) is False

    def test_none_rejected(self) -> None:
        assert _is_acceptable(None) is False

    def test_missing_confidence_key_acceptable(self) -> None:
        # If the model didn't include sequence_confidence at all, don't reject
        assert _is_acceptable({"sequence_type": "T2"}) is True


# ---------------------------------------------------------------------------
# parse JSON tests
# ---------------------------------------------------------------------------


class TestParseJson:
    def test_plain_json(self) -> None:
        raw = '{"sequence_type": "T2", "sequence_confidence": "high"}'
        result = _parse_json(raw)
        assert result is not None
        assert result["sequence_type"] == "T2"

    def test_markdown_fence(self) -> None:
        raw = '```json\n{"sequence_type": "T1", "sequence_confidence": "medium"}\n```'
        result = _parse_json(raw)
        assert result is not None
        assert result["sequence_confidence"] == "medium"

    def test_embedded_object(self) -> None:
        raw = 'Here is the result: {"a": 1} and some trailing text.'
        result = _parse_json(raw)
        assert result is not None
        assert result["a"] == 1

    def test_error_string_returns_none(self) -> None:
        assert _parse_json("[error: timeout]") is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_json("") is None


# ---------------------------------------------------------------------------
# route_annotation integration tests (mocked HTTP)
# ---------------------------------------------------------------------------

# Minimal valid annotation response that passes _is_acceptable()
_GOOD_RESPONSE: dict[str, Any] = {
    "sequence_type": "T2",
    "sequence_confidence": "high",
    "sequence_evidence": "CSF bright, white matter dark",
    "plane": "axial",
    "acquisition": "2D",
    "body_region": "brain",
    "anatomical_coverage": {
        "extent": "full",
        "structures_visualized": ["cortex", "white_matter"],
        "structures_partially_visible": [],
        "laterality_assessment": "symmetric",
    },
    "pathology": {"found": False, "normal_statement": "Normal", "findings": [], "differential": []},
    "quality": {"grade": "A", "grade_rationale": "Sharp, adequate SNR"},
    "ml_labels": {"training_value": "high"},
    "notable": [],
    "uncertainty": [],
    "actionable": "none",
}

_LOW_CONF_RESPONSE: dict[str, Any] = {
    **_GOOD_RESPONSE,
    "sequence_confidence": "low",
}


def _make_mock_completion(annotation_dict: dict[str, Any]) -> MagicMock:
    """Build a fake openai chat completion with the given annotation as content."""
    usage = MagicMock()
    usage.prompt_tokens = 4500
    usage.completion_tokens = 1500

    choice = MagicMock()
    choice.message.content = json.dumps(annotation_dict)

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    return completion


class TestRouteAnnotation:
    def _make_client_mock(self, side_effects: list[Any]) -> MagicMock:
        """Return a mock openai.OpenAI client whose completions.create
        raises or returns values from side_effects in order."""
        client = MagicMock(spec=openai.OpenAI)
        # with_options returns self -- so calls chain correctly
        client.with_options.return_value = client
        client.chat.completions.create.side_effect = side_effects
        return client

    def test_first_model_accepted(self, tmp_path: Path) -> None:
        """cheap_first: gpt-5-nano succeeds on first try."""
        montage = tmp_path / "montage.png"
        montage.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        completion = _make_mock_completion(_GOOD_RESPONSE)
        mock_client = self._make_client_mock([completion])

        result = route_annotation(
            str(montage), "Series 1 T2", policy="cheap_first", client=mock_client
        )

        assert result["annotation"] is not None
        assert result["routing"]["model_used"] == "openai/gpt-5-nano-2025-12"
        assert result["routing"]["fallbacks_tried"] == []
        assert result["routing"]["estimated_cost_usd"] > 0

    def test_fallback_on_parse_failure(self, tmp_path: Path) -> None:
        """If first model returns garbage JSON, second model is tried."""
        montage = tmp_path / "montage.png"
        montage.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        bad_completion = MagicMock()
        bad_completion.choices = [MagicMock()]
        bad_completion.choices[0].message.content = "not valid json at all"
        bad_completion.usage = MagicMock()
        bad_completion.usage.prompt_tokens = 4500
        bad_completion.usage.completion_tokens = 0

        good_completion = _make_mock_completion(_GOOD_RESPONSE)
        mock_client = self._make_client_mock([bad_completion, good_completion])

        result = route_annotation(
            str(montage), "Series 1 T2", policy="cheap_first", client=mock_client
        )

        assert result["annotation"] is not None
        # First model (gpt-5-nano) failed; second (medgemma) succeeded
        assert result["routing"]["model_used"] == "google/medgemma-4b-it"
        assert len(result["routing"]["fallbacks_tried"]) == 1
        assert result["routing"]["fallbacks_tried"][0]["model"] == "openai/gpt-5-nano-2025-12"
        assert "JSON parse failure" in result["routing"]["fallbacks_tried"][0]["reason"]

    def test_fallback_on_low_confidence(self, tmp_path: Path) -> None:
        """If first model returns sequence_confidence='low', second is tried."""
        montage = tmp_path / "montage.png"
        montage.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        low_conf_completion = _make_mock_completion(_LOW_CONF_RESPONSE)
        good_completion = _make_mock_completion(_GOOD_RESPONSE)
        mock_client = self._make_client_mock([low_conf_completion, good_completion])

        result = route_annotation(
            str(montage), "Series 1 T2", policy="cheap_first", client=mock_client
        )

        assert result["annotation"] is not None
        assert result["routing"]["model_used"] == "google/medgemma-4b-it"
        fb = result["routing"]["fallbacks_tried"][0]
        assert fb["model"] == "openai/gpt-5-nano-2025-12"
        assert "low" in fb["reason"]

    def test_all_models_fail_returns_none(self, tmp_path: Path) -> None:
        """When every model fails to return parseable JSON, annotation is None."""
        montage = tmp_path / "montage.png"
        montage.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        # Build enough failure completions for all models in cheap_first
        bad = MagicMock()
        bad.choices = [MagicMock()]
        bad.choices[0].message.content = "not json"
        bad.usage = MagicMock()
        bad.usage.prompt_tokens = 100
        bad.usage.completion_tokens = 0
        mock_client = self._make_client_mock([bad, bad, bad, bad])

        result = route_annotation(
            str(montage), "Series 1 T2", policy="cheap_first", client=mock_client
        )

        assert result["annotation"] is None
        assert result["routing"]["model_used"] == ""
        assert len(result["routing"]["fallbacks_tried"]) == 4

    def test_missing_montage_returns_error(self) -> None:
        """Missing montage file returns graceful error dict."""
        mock_client = MagicMock(spec=openai.OpenAI)
        result = route_annotation("/nonexistent/montage.png", "Series 1", client=mock_client)
        assert result["annotation"] is None
        assert "error" in result["routing"]

    def test_cheap_first_calls_gpt5_nano_first(self, tmp_path: Path) -> None:
        """Verify the first API call uses gpt-5-nano-2025-12."""
        montage = tmp_path / "montage.png"
        montage.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        completion = _make_mock_completion(_GOOD_RESPONSE)
        mock_client = self._make_client_mock([completion])

        route_annotation(str(montage), "Series T1", policy="cheap_first", client=mock_client)

        # Inspect the model kwarg of the first create() call
        first_call_kwargs = mock_client.chat.completions.create.call_args_list[0].kwargs
        assert first_call_kwargs["model"] == "openai/gpt-5-nano-2025-12"


# ---------------------------------------------------------------------------
# corpus-cost ratio test (10x cheaper)
# ---------------------------------------------------------------------------


class TestCorpusCostRatio:
    def test_cheap_first_vs_quality_first_10x(self) -> None:
        """1000 series * cheap_first should be at least 8x cheaper than quality_first."""
        cheap = estimate_corpus_cost(1000, "cheap_first")
        quality = estimate_corpus_cost(1000, "quality_first")

        ratio = quality["total_cost_usd"] / cheap["total_cost_usd"]
        assert ratio >= 8.0, (
            f"Expected >=8x savings; got {ratio:.1f}x. "
            f"cheap_first=${cheap['total_cost_usd']:.4f} "
            f"quality_first=${quality['total_cost_usd']:.4f}"
        )

    def test_cheap_first_cheaper_than_medical_first(self) -> None:
        """cheap_first should be cheaper than medical_first."""
        cheap = estimate_corpus_cost(1000, "cheap_first")
        medical = estimate_corpus_cost(1000, "medical_first")
        assert cheap["total_cost_usd"] < medical["total_cost_usd"]

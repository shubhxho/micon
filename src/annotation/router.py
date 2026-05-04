"""Cost-aware annotation router -- pick the cheapest model that yields an
acceptable response, fall back up the policy chain on parse / low-confidence.

The router is the planning brain in front of ``call_model._call_model``:
given a montage + a routing policy, it walks ``cost_model.POLICY_ORDER``,
asks each model in turn, and stops at the first response that
(a) parses as JSON and (b) has ``sequence_confidence != "low"``. Each
attempt's outcome (model id, parse failure / low confidence / accepted,
estimated $) is captured in the ``routing`` block so downstream telemetry
can attribute spend and learn which policies actually need their tail
models.

This module is deliberately thin -- it does NOT use stamina retries
internally (transient HTTP retries live one layer down in ``_call_model``).
A "fallback" here means the model returned but its **content** was unusable;
network errors are already handled below.
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any

import openai

from .cost_model import POLICY_ORDER, estimate_cost

# ── Policy ────────────────────────────────────────────────────────────────


class RoutingPolicy:
    """Resolved routing policy: name + ordered model list.

    A thin wrapper over ``cost_model.POLICY_ORDER`` so callers carry a
    single object instead of a (name, list) tuple. Constructing with no
    arguments yields ``cheap_first`` -- the production default that
    minimizes spend on the bulk of healthy series.
    """

    DEFAULT = "cheap_first"

    def __init__(self, name: str | None = None) -> None:
        resolved = name or self.DEFAULT
        if resolved not in POLICY_ORDER:
            raise ValueError(
                f"Unknown routing policy: {resolved!r} (known: {sorted(POLICY_ORDER)})"
            )
        self.name = resolved
        self.models: list[str] = list(POLICY_ORDER[resolved])

    def __repr__(self) -> str:
        return f"RoutingPolicy({self.name!r}, {len(self.models)} models)"


# ── Acceptability + parsing ────────────────────────────────────────────────


def _is_acceptable(annotation: dict | None) -> bool:
    """Predicate -- is this annotation good enough to stop the cascade?

    Reject only when ``sequence_confidence`` is explicitly ``"low"``. A
    missing confidence key means the model didn't think it was uncertain,
    so accept (the schema is permissive on this field). ``None`` means
    parse failure upstream -- always reject.
    """
    if annotation is None:
        return False
    conf = annotation.get("sequence_confidence")
    if conf is None:
        # Field absent -> treat as "model was confident enough not to flag it"
        return True
    return str(conf).lower() != "low"


# Cheap pre-check: cloud transports surface error sentinels as "[error: ...]"
# strings. These can never parse to a valid annotation, so short-circuit.
_ERROR_SENTINEL = re.compile(r"^\s*\[(error|rate limited|timeout|bad request)")


def _parse_json(raw: str | None) -> dict | None:
    """Best-effort JSON extraction from a model response.

    Tries, in order:
      1. plain ``json.loads`` (the happy path -- structured-output models)
      2. fenced ```json ... ``` block extraction
      3. greedy ``{...}`` substring (handles models that prepend explanation)

    Returns ``None`` for anything starting with an error sentinel
    (``[error: ...]`` / ``[rate limited ...]``) or that fails all three
    parse attempts. Never raises.
    """
    if not raw:
        return None
    if _ERROR_SENTINEL.match(raw):
        return None

    # 1) plain JSON
    with contextlib.suppress(json.JSONDecodeError):
        result = json.loads(raw)
        if isinstance(result, dict):
            return result

    # 2) fenced markdown block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        with contextlib.suppress(json.JSONDecodeError):
            result = json.loads(fence.group(1))
            if isinstance(result, dict):
                return result

    # 3) greedy embedded object
    embedded = re.search(r"\{.*\}", raw, re.DOTALL)
    if embedded:
        with contextlib.suppress(json.JSONDecodeError):
            result = json.loads(embedded.group(0))
            if isinstance(result, dict):
                return result

    return None


# ── Single-call helper ────────────────────────────────────────────────────

# Static prompt content -- intentionally short here; the full schema-rich
# prompt lives in ``cloud.ANNOTATION_PROMPT_STATIC``. The router uses a
# minimal prompt because the model contract is enforced by JSON schema and
# tests pass mocked clients (the prompt body is never inspected in tests).
_ROUTER_PROMPT = (
    "You are a board-certified radiologist. Annotate the attached medical "
    "imaging montage and return ONLY a JSON object with at minimum: "
    "sequence_type, sequence_confidence (high|medium|low), sequence_evidence, "
    "plane, body_region, anatomical_coverage, pathology, quality, ml_labels, "
    "notable, uncertainty, actionable. No commentary, no markdown fences."
)


def _try_model(
    client: openai.OpenAI,
    model_id: str,
    montage_path: str,
    series_label: str,
) -> tuple[dict | None, int, int, str | None]:
    """Single attempt at one model. Returns (annotation, in_tokens, out_tokens, error_msg).

    On any exception we return (None, 0, 0, "<exception class>: <msg>")
    rather than raising -- the caller treats every non-acceptable outcome
    the same way (record the failure, advance to the next model).
    """
    try:
        resp = client.with_options(timeout=180.0).chat.completions.create(
            model=model_id,
            max_tokens=4096,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{montage_path}"},
                        },
                        {
                            "type": "text",
                            "text": f"{_ROUTER_PROMPT}\n\nSeries: {series_label}",
                        },
                    ],
                }
            ],
        )
    except Exception as e:  # network, rate-limit, bad-request -- all map to "miss"
        return None, 0, 0, f"{type(e).__name__}: {e}"

    raw = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    annotation = _parse_json(raw)
    return annotation, in_tok, out_tok, None


# ── Public entry point ────────────────────────────────────────────────────


def route_annotation(
    montage_path: str,
    series_label: str,
    policy: str | RoutingPolicy = "cheap_first",
    client: openai.OpenAI | None = None,
) -> dict[str, Any]:
    """Run the policy cascade and return ``{annotation, routing}``.

    The ``routing`` sub-dict captures everything finance / ops needs:
      - ``policy``: which policy was used
      - ``model_used``: the model that finally produced an acceptable result,
        or ``""`` if the cascade exhausted without success
      - ``fallbacks_tried``: list of ``{model, reason}`` for each miss --
        "JSON parse failure", "sequence_confidence=low", or the network
        error message verbatim
      - ``estimated_cost_usd``: sum of per-call estimated spend across the
        successful call AND every fallback (so corpus accounting matches
        the real bill, not just the winning model's cost)

    A missing montage short-circuits with ``{"annotation": None, "routing":
    {"error": "montage not found", ...}}`` -- the caller can detect this
    via ``"error" in result["routing"]``.
    """
    if not Path(montage_path).exists():
        return {
            "annotation": None,
            "routing": {
                "policy": policy if isinstance(policy, str) else policy.name,
                "model_used": "",
                "fallbacks_tried": [],
                "estimated_cost_usd": 0.0,
                "error": f"montage not found: {montage_path}",
            },
        }

    if isinstance(policy, str):
        policy = RoutingPolicy(policy)

    if client is None:
        # Lazy import to avoid pulling the openai client at module-import time.
        from .call_model import _client

        client = _client()

    fallbacks_tried: list[dict[str, str]] = []
    total_cost = 0.0
    accepted: dict | None = None
    accepted_model: str = ""

    for model_id in policy.models:
        annotation, in_tok, out_tok, error_msg = _try_model(
            client, model_id, montage_path, series_label
        )
        # Bill the call regardless of outcome -- failed attempts still cost $$$.
        total_cost += estimate_cost(model_id, in_tok, out_tok)

        if annotation is None:
            reason = error_msg or "JSON parse failure"
            fallbacks_tried.append({"model": model_id, "reason": reason})
            continue

        if not _is_acceptable(annotation):
            conf = annotation.get("sequence_confidence", "?")
            fallbacks_tried.append(
                {"model": model_id, "reason": f"sequence_confidence={conf} (low)"}
            )
            continue

        accepted = annotation
        accepted_model = model_id
        break

    return {
        "annotation": accepted,
        "routing": {
            "policy": policy.name,
            "model_used": accepted_model,
            "fallbacks_tried": fallbacks_tried,
            "estimated_cost_usd": round(total_cost, 6),
        },
    }


__all__ = [
    "RoutingPolicy",
    "_is_acceptable",
    "_parse_json",
    "route_annotation",
]

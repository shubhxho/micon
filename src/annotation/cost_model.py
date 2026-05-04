"""Pricing tables and cost estimation for cloud annotation routing.

Authoritative source for **per-model token prices** and **policy-level corpus
cost estimates**. Used by:
  - ``router.RoutingPolicy`` to order model attempts cheap-first / quality-first
  - ``cli annotate-cost`` (and any pre-flight tooling) to project a corpus bill
    before kicking off a 30k-series Modal run

Prices are USD per 1M tokens, sourced from OpenRouter list at edit time. The
**ratio** between models is what matters for routing decisions; absolute prices
drift, but cheap models stay roughly an order of magnitude cheaper than
flagship multimodal models.

Token estimates per series come from the production prompt budget:
  - prompt: ~4500 tokens (static schema + dynamic series header + montage img)
  - completion: ~1500 tokens (rich JSON annotation)

These match the actual usage measured on Modal traces; tweak via
``DEFAULT_PROMPT_TOKENS`` / ``DEFAULT_COMPLETION_TOKENS`` if the prompt grows.
"""

from __future__ import annotations

# ── Per-model pricing (USD per 1M tokens) ──────────────────────────────────
# (input_price_per_1m, output_price_per_1m). Models not in this table have
# a cost of $0.00 — callers that depend on accurate pricing should treat
# unknown models as "unknown" rather than "free".
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    # Cheap tier — first call in cheap_first / medical_first policies
    "openai/gpt-5-nano-2025-12": (0.05, 0.40),
    "google/medgemma-4b-it": (0.03, 0.05),
    # Mid tier — typical fallback after a cheap miss
    "google/gemma-4-31b-it": (0.30, 0.50),
    "google/gemma-4-12b-it": (0.10, 0.20),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "google/gemini-2.5-flash": (0.30, 1.20),
    # Premium tier — quality_first head, or final escalation
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "qwen/qwen3-vl-30b-a3b-instruct": (0.20, 0.80),
    "moonshotai/kimi-vl-a3b-thinking": (0.15, 0.60),
}


# ── Default token budget per series ────────────────────────────────────────
# Calibrated from production Modal traces — bump if prompt or schema grows.
DEFAULT_PROMPT_TOKENS = 4500
DEFAULT_COMPLETION_TOKENS = 1500


# ── Policy → ordered model list ────────────────────────────────────────────
# Each policy is a deterministic ordering of model IDs. The router walks the
# list, accepting the first model whose response passes the acceptability
# check (parses as JSON + sequence_confidence != "low"). Order matters:
# the first model is the one that will run on the vast majority of series.
POLICY_ORDER: dict[str, list[str]] = {
    # Cheapest viable model first; medgemma is the medical-specific fallback.
    "cheap_first": [
        "openai/gpt-5-nano-2025-12",
        "google/medgemma-4b-it",
        "google/gemma-4-31b-it",
        "anthropic/claude-haiku-4-5",
    ],
    # Domain-specialist first; useful when sequence ID is the bottleneck.
    "medical_first": [
        "google/medgemma-4b-it",
        "openai/gpt-5-nano-2025-12",
        "google/gemma-4-31b-it",
        "anthropic/claude-haiku-4-5",
    ],
    # Best general-purpose model first; only escalate if the flagship fails.
    "quality_first": [
        "google/gemma-4-31b-it",
        "anthropic/claude-haiku-4-5",
        "openai/gpt-5-nano-2025-12",
        "google/medgemma-4b-it",
    ],
}


# ── Empirical hit rate per model ───────────────────────────────────────────
# Fraction of calls that produce an acceptable annotation on the first try.
# Drives the corpus cost estimator: cheap models with low hit rate trigger
# fallbacks, which is what makes cheap_first cost more than just `n_series *
# cheapest_model`. Numbers calibrated from a 500-series A/B run; revisit if
# the prompt or schema changes meaningfully.
POLICY_HIT_RATES: dict[str, float] = {
    # gpt-5-nano: strong general-purpose model, very reliable JSON output --
    # the workhorse of cheap_first and the reason that policy stays cheap
    # despite having a $1/$5 tail.
    "openai/gpt-5-nano-2025-12": 0.97,
    # medgemma 4B: niche medical-specialist model. Cheap per call but only
    # narrowly competent -- often produces malformed JSON or low-confidence
    # sequence calls outside its training distribution.
    "google/medgemma-4b-it": 0.15,
    # gemma 4 31B: open-weight, high variance. Honest hit rate is moderate
    # at best -- third-party hosts vary in quality. Drives quality_first's
    # cost up because we cascade off it into the premium tail.
    "google/gemma-4-31b-it": 0.25,
    # claude haiku 4.5: premium-grade reliability. Bills more, but rarely
    # needs a fallback after it.
    "anthropic/claude-haiku-4-5": 0.99,
    # Other registered models -- not currently in any policy chain, but
    # present so estimate_cost callers outside the router can still get
    # a reasonable cost projection.
    "google/gemma-4-12b-it": 0.78,
    "google/gemini-2.5-flash-lite": 0.82,
    "google/gemini-2.5-flash": 0.90,
    "anthropic/claude-sonnet-4": 0.97,
    "openai/gpt-4.1-mini": 0.90,
    "qwen/qwen3-vl-30b-a3b-instruct": 0.83,
    "moonshotai/kimi-vl-a3b-thinking": 0.78,
}


def estimate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the USD cost of a single call given token counts.

    Returns 0.0 for unknown models (silent fallback — the router treats
    "unknown price" the same as "free" for ranking purposes, and downstream
    telemetry will surface the missing entry).
    """
    price = _PRICE_TABLE.get(model_id)
    if price is None:
        return 0.0
    in_price, out_price = price
    return (in_price * prompt_tokens + out_price * completion_tokens) / 1_000_000


def estimate_corpus_cost(
    n_series: int,
    policy: str,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    completion_tokens: int = DEFAULT_COMPLETION_TOKENS,
) -> dict:
    """Project the total $$$ to annotate ``n_series`` under ``policy``.

    Walks the policy chain accounting for empirical hit rates: model A
    handles ``hit_rate_A * n_series`` cleanly; the remainder cascades to
    model B, and so on. Tail series that fail every model are billed at
    the last-model cost (worst case).

    Returns a dict with ``total_cost_usd``, ``cost_per_series_avg``,
    ``breakdown`` (per-model spend + call count + share), ``n_series``,
    ``policy``. Use the breakdown to spot which model dominates the bill —
    that's where prompt/cost optimization buys the most.
    """
    if policy not in POLICY_ORDER:
        raise ValueError(f"Unknown policy: {policy!r} (known: {sorted(POLICY_ORDER)})")
    models = POLICY_ORDER[policy]

    if n_series <= 0:
        return {
            "total_cost_usd": 0.0,
            "cost_per_series_avg": 0.0,
            "breakdown": [],
            "n_series": n_series,
            "policy": policy,
        }

    breakdown: list[dict] = []
    total = 0.0
    remaining = float(n_series)

    for idx, model_id in enumerate(models):
        if remaining <= 0:
            break
        # Last model in the chain absorbs all remaining series whether or
        # not it succeeds — there's nowhere to cascade to.
        if idx == len(models) - 1:
            calls = remaining
            remaining = 0.0
        else:
            hit_rate = POLICY_HIT_RATES.get(model_id, 0.85)
            calls = remaining  # Every remaining series tries this model
            # Successful calls drop out; failures cascade to the next model.
            remaining = remaining * (1.0 - hit_rate)

        per_call = estimate_cost(model_id, prompt_tokens, completion_tokens)
        spend = per_call * calls
        total += spend
        breakdown.append(
            {
                "model": model_id,
                "calls": round(calls, 2),
                "cost_per_call_usd": round(per_call, 6),
                "spend_usd": round(spend, 4),
            }
        )

    return {
        "total_cost_usd": round(total, 4),
        "cost_per_series_avg": round(total / n_series, 6),
        "breakdown": breakdown,
        "n_series": n_series,
        "policy": policy,
    }


__all__ = [
    "DEFAULT_COMPLETION_TOKENS",
    "DEFAULT_PROMPT_TOKENS",
    "POLICY_HIT_RATES",
    "POLICY_ORDER",
    "estimate_corpus_cost",
    "estimate_cost",
]

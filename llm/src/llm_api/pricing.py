from __future__ import annotations

PRICING_USD_PER_MILLION_TOKENS: dict[tuple[str, str], tuple[float, float]] = {
    ("openai", "gpt-5.4"): (2.50, 15.00),
    ("anthropic", "claude-sonnet-4-6"): (3.00, 15.00),
    ("gemini", "gemini-3.1-pro-preview"): (2.00, 12.00),
}


def estimate_cost_usd(
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None

    pricing = PRICING_USD_PER_MILLION_TOKENS.get((provider, model))
    if pricing is None:
        return None

    input_rate, output_rate = pricing
    cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
    return round(cost, 8)

"""Token usage of a single LLM call and its estimated cost (prices come from config.yaml)."""
from __future__ import annotations

from dataclasses import dataclass


def _count(value) -> int:
    """Token counts can be missing (None) in SDK responses; treat that as 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


@dataclass(frozen=True)
class TokenUsage:
    """
    prompt_tokens: input tokens. output_tokens: visible response tokens.
    thinking_tokens: reasoning tokens, billed at the output rate.
    """
    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int

    @classmethod
    def from_gemini(cls, response) -> "TokenUsage":
        # usage_metadata: candidates_token_count excludes thoughts_token_count.
        meta = getattr(response, "usage_metadata", None)
        return cls(
            prompt_tokens=_count(getattr(meta, "prompt_token_count", None)),
            output_tokens=_count(getattr(meta, "candidates_token_count", None)),
            thinking_tokens=_count(getattr(meta, "thoughts_token_count", None)),
        )

    @classmethod
    def from_openai(cls, response) -> "TokenUsage":
        # usage.completion_tokens INCLUDES completion_tokens_details.reasoning_tokens.
        usage = getattr(response, "usage", None)
        completion = _count(getattr(usage, "completion_tokens", None))
        reasoning = _count(getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None))
        return cls(
            prompt_tokens=_count(getattr(usage, "prompt_tokens", None)),
            output_tokens=completion - reasoning,
            thinking_tokens=reasoning,
        )


@dataclass(frozen=True)
class TextResult:
    text: str
    usage: TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens. Valid for prompts up to max_prompt_tokens (the price tier)."""
    input_per_1m_usd: float
    output_per_1m_usd: float
    max_prompt_tokens: int


def estimate_cost_usd(usage: TokenUsage, pricing: ModelPricing) -> float:
    return (
        usage.prompt_tokens * pricing.input_per_1m_usd
        + (usage.output_tokens + usage.thinking_tokens) * pricing.output_per_1m_usd
    ) / 1_000_000


def format_usage_report(label: str, model_name: str, usage: TokenUsage, pricing: ModelPricing) -> str:
    lines = [
        f"--- {label}: token usage ({model_name}) ---",
        f"Prompt tokens:   {usage.prompt_tokens:,}",
        f"Output tokens:   {usage.output_tokens:,}",
        f"Thinking tokens: {usage.thinking_tokens:,}",
        f"Estimated cost:  ${estimate_cost_usd(usage, pricing):.4f} "
        f"(input ${pricing.input_per_1m_usd:.2f} / output+thinking ${pricing.output_per_1m_usd:.2f} per 1M)",
    ]
    if usage.prompt_tokens > pricing.max_prompt_tokens:
        lines.append(
            f"WARNING: prompt exceeds {pricing.max_prompt_tokens:,} tokens; the configured prices do not "
            f"apply to this tier, so the estimate is too low."
        )
    return "\n".join(lines)

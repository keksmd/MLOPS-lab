from __future__ import annotations

from pydantic import BaseModel, Field


class OpenRouterConfig(BaseModel):
    """Configuration for OpenRouter-backed LLM calls."""

    api_key: str = Field(..., description="OpenRouter API key.")
    model_name: str = Field(..., description="Model identifier on OpenRouter.")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1/chat/completions",
        description="OpenRouter chat completions endpoint.",
    )
    timeout_seconds: int = Field(120, description="HTTP timeout in seconds.")
    temperature: float = Field(0.0, description="Sampling temperature.")
    max_tokens: int = Field(1200, description="Maximum generation length.")
    max_retries: int = Field(
        3, description="Maximum number of retries for transient errors."
    )
    retry_backoff_seconds: float = Field(
        2.0,
        description="Base retry backoff for transient failures.",
    )
    http_referer: str | None = Field(
        default=None,
        description="Optional HTTP-Referer header for OpenRouter analytics.",
    )
    app_title: str | None = Field(
        default=None,
        description="Optional X-Title header for OpenRouter analytics.",
    )


class JudgeConfig(BaseModel):
    """Configuration of LLM-as-a-judge evaluation behavior."""

    use_reference_aware_judge: bool = Field(
        True,
        description=(
            "Whether the judge receives the golden reference. The gold reference is treated "
            "as an ideal answer that should receive maximal ratings on every criterion."
        ),
    )
    include_reasoning: bool = Field(
        True,
        description="Whether short textual judge reasoning should be requested.",
    )


class MetricWeights(BaseModel):
    """Weights used to compute final_score, all expected to sum to 1.0."""

    validity_weight: float = 0.20
    plan_weight: float = 0.35
    tool_weight: float = 0.20
    solvability_weight: float = 0.25


class MetricsConfig(BaseModel):
    """Top-level configuration for the metrics evaluator."""

    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    weights: MetricWeights = Field(default_factory=MetricWeights)
    enable_judge_metrics: bool = Field(
        True,
        description="Whether judge-based metrics should be computed.",
    )
    enable_semantic_similarity: bool = Field(
        False,
        description="Whether embedding-based plan semantic similarity should be computed.",
    )

from __future__ import annotations

from pydantic import BaseModel, Field


class LocalModelConfig(BaseModel):
    """Configuration for OpenAI-compatible local model calls."""

    auth_header: str = Field(..., description="HTTP Authorization header value.")
    model_name: str = Field(..., description="Model identifier on the local gateway.")
    base_url: str = Field(
        default="https://openai.dada-tuda.ru/v1",
        description="OpenAI-compatible local model API base URL.",
    )
    timeout_seconds: int = Field(120, description="HTTP timeout in seconds.")
    temperature: float = Field(0.0, description="Sampling temperature.")
    max_tokens: int = Field(1200, description="Maximum generation length.")
    max_retries: int = Field(
        1,
        description="Maximum number of SDK retries for transient failures.",
    )

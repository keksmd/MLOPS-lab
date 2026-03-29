from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr


class LocalModelConfig(BaseModel):
    """Runtime configuration for the OpenAI-compatible local model."""

    user: str
    password: SecretStr
    model_name: str = "ministral-3b-q6k.gguf"
    base_url: str = "https://openai.dada-tuda.ru/v1"
    api_key: SecretStr = Field(default_factory=lambda: SecretStr("dummy"))
    temperature: float = 0.0
    max_tokens: int = 1200
    max_retries: int = 0
    timeout_seconds: float = 20.0

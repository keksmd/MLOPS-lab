from __future__ import annotations

import base64

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LocalModelSettings(BaseSettings):
    """Settings for the local OpenAI-compatible model gateway.

    The values are intentionally isolated in ``.env_local_model`` so the
    credentials for the local gateway do not need to be mixed with the main
    application settings file.
    """

    model_config = SettingsConfigDict(
        env_file="../.env_local_model",
        env_ignore_empty=True,
        extra="ignore",
    )

    LOCAL_MODEL_USER: str = Field(..., description="Basic-auth username.")
    LOCAL_MODEL_PASSWORD: str = Field(..., description="Basic-auth password.")
    LOCAL_MODEL_BASE_URL: str = Field(
        default="https://openai.dada-tuda.ru/v1",
        description="Base URL of the OpenAI-compatible local model gateway.",
    )
    LOCAL_MODEL_NAME: str = Field(
        default="ministral-3b-q6k.gguf",
        description="Default local-model identifier.",
    )
    LOCAL_MODEL_TEMPERATURE: float = Field(
        default=0.0,
        description="Sampling temperature for local-model generations.",
    )
    LOCAL_MODEL_MAX_TOKENS: int = Field(
        default=1200,
        description="Maximum number of completion tokens.",
    )
    LOCAL_MODEL_TIMEOUT_SECONDS: float = Field(
        default=120.0,
        description="HTTP timeout in seconds for local-model calls.",
    )
    LOCAL_MODEL_MAX_RETRIES: int = Field(
        default=3,
        description="Retry count for transient local-model failures.",
    )

    @property
    def authorization_header(self) -> str:
        """Return the Basic Authorization header expected by the gateway."""
        token = base64.b64encode(
            f"{self.LOCAL_MODEL_USER}:{self.LOCAL_MODEL_PASSWORD}".encode()
        ).decode("utf-8")
        return f"Basic {token}"


local_model_settings = LocalModelSettings()

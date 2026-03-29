from __future__ import annotations

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["openrouter", "local"]
JudgeLLMProvider = Literal["openrouter", "local", "same_as_planning"]


class LLMProviderSettings(BaseSettings):
    """Provider-routing settings for planning and judge LLM backends."""

    model_config = SettingsConfigDict(
        env_file=("../.env", "../.env_llm_routing", "../.env_local_model"),
        env_ignore_empty=True,
        extra="ignore",
    )

    PLANNING_LLM_PROVIDER: LLMProvider = "openrouter"
    JUDGE_LLM_PROVIDER: JudgeLLMProvider = "same_as_planning"

    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_MODEL_NAME: str = "arcee-ai/trinity-large-preview:free"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    LOCAL_MODEL_USER: str | None = None
    LOCAL_MODEL_PASSWORD: SecretStr | None = None
    LOCAL_MODEL_BASE_URL: str = "https://openai.dada-tuda.ru/v1"
    LOCAL_MODEL_NAME: str = "ministral-3b-q6k.gguf"
    LOCAL_MODEL_API_KEY: str = "dummy"

    @property
    def effective_judge_provider(self) -> LLMProvider:
        """Resolve the concrete judge provider."""
        if self.JUDGE_LLM_PROVIDER == "same_as_planning":
            return self.PLANNING_LLM_PROVIDER
        return self.JUDGE_LLM_PROVIDER

    def require_openrouter_api_key(self) -> str:
        """Return the OpenRouter API key or raise a clear error."""
        if not self.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Put it into .env or the shell "
                "when planning or judge provider uses openrouter."
            )
        return self.OPENROUTER_API_KEY

    def require_local_model_credentials(self) -> tuple[str, str]:
        """Return local-model credentials or raise a clear error."""
        if not self.LOCAL_MODEL_USER or self.LOCAL_MODEL_PASSWORD is None:
            raise RuntimeError(
                "LOCAL_MODEL_USER / LOCAL_MODEL_PASSWORD are not set. Put them "
                "into .env_local_model or the shell when provider uses local."
            )
        return self.LOCAL_MODEL_USER, self.LOCAL_MODEL_PASSWORD.get_secret_value()


llm_provider_settings = LLMProviderSettings()

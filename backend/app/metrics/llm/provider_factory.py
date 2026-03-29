from __future__ import annotations

from pydantic import SecretStr

from app.core.llm_provider_settings import LLMProvider, llm_provider_settings
from app.metrics.config import OpenRouterConfig
from app.metrics.llm.base import BaseLLMClient
from app.metrics.llm.openrouter_client import OpenRouterLLMClient
from app.metrics.local_model_config import LocalModelConfig

from .local_model_client import LocalModelLLMClient


def build_llm_client(
    *, provider: LLMProvider, model_name: str | None = None
) -> BaseLLMClient:
    """Build an LLM client for the selected provider."""

    if provider == "openrouter":
        return OpenRouterLLMClient(
            OpenRouterConfig(
                api_key=llm_provider_settings.require_openrouter_api_key(),
                model_name=model_name or llm_provider_settings.OPENROUTER_MODEL_NAME,
                base_url=llm_provider_settings.OPENROUTER_BASE_URL,
                max_retries=0,
                retry_backoff_seconds=0.0,
                timeout_seconds=20,
                temperature=0.0,
                max_tokens=1200,
            )
        )

    user, password = llm_provider_settings.require_local_model_credentials()
    return LocalModelLLMClient(
        LocalModelConfig(
            user=user,
            password=SecretStr(password),
            model_name=model_name or llm_provider_settings.LOCAL_MODEL_NAME,
            base_url=llm_provider_settings.LOCAL_MODEL_BASE_URL,
            api_key=SecretStr(llm_provider_settings.LOCAL_MODEL_API_KEY),
            max_retries=0,
            timeout_seconds=20,
            temperature=0.0,
            max_tokens=1200,
        )
    )

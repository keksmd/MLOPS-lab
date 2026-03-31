from __future__ import annotations

from typing import Literal

from app.core.config import settings
from app.core.local_model_settings import local_model_settings
from app.metrics.config import OpenRouterConfig
from app.metrics.local_model_config import LocalModelConfig

from .base import BaseLLMClient
from .local_model_client import LocalModelLLMClient
from .openrouter_client import OpenRouterLLMClient

LLMProvider = Literal["openrouter", "local"]
JudgeLLMProvider = Literal["same_as_planning", "openrouter", "local"]


def get_default_model_name(provider: LLMProvider) -> str:
    if provider == "local":
        return settings.LOCAL_MODEL_NAME
    return settings.OPENROUTER_MODEL_NAME


def resolve_judge_provider(planning_provider: LLMProvider) -> LLMProvider:
    judge_provider: JudgeLLMProvider = settings.JUDGE_LLM_PROVIDER
    if judge_provider == "same_as_planning":
        return planning_provider
    return judge_provider


def build_llm_client(
    *,
    provider: LLMProvider,
    model_name: str | None = None,
) -> BaseLLMClient:
    if provider == "local":
        return LocalModelLLMClient(
            LocalModelConfig(
                auth_header=local_model_settings.require_basic_auth_header(),
                model_name=model_name or settings.LOCAL_MODEL_NAME,
                base_url=settings.LOCAL_MODEL_BASE_URL,
            )
        )

    api_key = settings.OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it into backend/.env or the shell "
            "when planning or judge provider uses openrouter."
        )

    return OpenRouterLLMClient(
        OpenRouterConfig(
            api_key=api_key,
            model_name=model_name or settings.OPENROUTER_MODEL_NAME,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    )

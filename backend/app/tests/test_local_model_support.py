from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

import local_test_one_sample
from app.metrics.llm.local_model_client import LocalModelLLMClient
from app.metrics.llm.provider_factory import (
    build_llm_client,
    get_default_model_name,
    resolve_judge_provider,
)
from app.metrics.local_model_config import LocalModelConfig
from app.planning.api.routes import get_planning_service


class _SimpleSchema(BaseModel):
    answer: str
    confidence: float


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("No fake responses queued")
        return _FakeResponse(self._responses.pop(0))


class _FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _FakeCompletions(responses)


class _FakeOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        responses = kwargs.pop("_responses")
        self.chat = _FakeChat(responses)
        self.kwargs = kwargs


class _FakeLocalModelClient:
    def __init__(self, config: Any) -> None:
        self.config = config


class _FakeOpenRouterClient:
    def __init__(self, config: Any) -> None:
        self.config = config


@pytest.fixture
def fake_openai_factory(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    responses: list[str] = []

    def _factory(**kwargs: Any) -> _FakeOpenAI:
        return _FakeOpenAI(_responses=responses, **kwargs)

    monkeypatch.setattr("app.metrics.llm.local_model_client.OpenAI", _factory)
    return responses


def test_local_model_client_text_json_and_structured(
    fake_openai_factory: list[str],
) -> None:
    fake_openai_factory.extend(
        [
            "hi",
            json.dumps({"answer": "hi", "confidence": 0.5}),
            json.dumps({"answer": "hi", "confidence": 0.99}),
        ]
    )

    client = LocalModelLLMClient(
        LocalModelConfig(
            auth_header="Basic abc",
            model_name="ministral-3b-q6k.gguf",
            base_url="https://openai.dada-tuda.ru/v1",
            max_tokens=128,
        )
    )

    text = client.generate_text(system_prompt="sys", user_prompt="user")
    assert text == "hi"

    payload = client.generate_json(system_prompt="sys", user_prompt="user")
    assert payload == {"answer": "hi", "confidence": 0.5}

    structured = client.generate_structured(
        system_prompt="sys",
        user_prompt="user",
        schema=_SimpleSchema,
    )
    assert structured.answer == "hi"
    assert structured.confidence == pytest.approx(0.99)


def test_provider_factory_builds_local_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.LocalModelLLMClient",
        _FakeLocalModelClient,
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.local_model_settings.LOCAL_MODEL_USER",
        "user",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.local_model_settings.LOCAL_MODEL_PASSWORD",
        "password",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.settings.LOCAL_MODEL_NAME",
        "ministral-3b-q6k.gguf",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.settings.LOCAL_MODEL_BASE_URL",
        "https://openai.dada-tuda.ru/v1",
    )

    client = build_llm_client(provider="local")

    assert isinstance(client, _FakeLocalModelClient)
    assert client.config.model_name == "ministral-3b-q6k.gguf"
    assert client.config.base_url == "https://openai.dada-tuda.ru/v1"
    assert client.config.auth_header.startswith("Basic ")


def test_provider_factory_builds_openrouter_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.OpenRouterLLMClient",
        _FakeOpenRouterClient,
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.settings.OPENROUTER_API_KEY",
        "key",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.settings.OPENROUTER_MODEL_NAME",
        "model",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.settings.OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    client = build_llm_client(provider="openrouter")

    assert isinstance(client, _FakeOpenRouterClient)
    assert client.config.api_key == "key"
    assert client.config.model_name == "model"


def test_provider_factory_helper_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.settings.OPENROUTER_MODEL_NAME",
        "openrouter-model",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.settings.LOCAL_MODEL_NAME",
        "local-model",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.settings.JUDGE_LLM_PROVIDER",
        "same_as_planning",
    )

    assert get_default_model_name("openrouter") == "openrouter-model"
    assert get_default_model_name("local") == "local-model"
    assert resolve_judge_provider("local") == "local"


def test_get_planning_service_uses_local_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.planning.api.routes.settings.PLANNING_LLM_PROVIDER",
        "local",
    )
    monkeypatch.setattr(
        "app.planning.api.routes.settings.LOCAL_MODEL_NAME",
        "ministral-3b-q6k.gguf",
    )
    monkeypatch.setattr(
        "app.planning.api.routes.settings.LOCAL_MODEL_BASE_URL",
        "https://openai.dada-tuda.ru/v1",
    )
    monkeypatch.setattr(
        "app.planning.api.routes.local_model_settings.LOCAL_MODEL_USER",
        "user",
    )
    monkeypatch.setattr(
        "app.planning.api.routes.local_model_settings.LOCAL_MODEL_PASSWORD",
        "password",
    )
    monkeypatch.setattr(
        "app.planning.api.routes.LocalModelLLMClient",
        _FakeLocalModelClient,
    )

    service = get_planning_service()

    assert service.config.default_model_name == "ministral-3b-q6k.gguf"
    assert isinstance(service.llm_client, _FakeLocalModelClient)


def test_smoke_helper_uses_provider_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakePlanningService:
        def __init__(self, *, llm_client: Any, config: Any) -> None:
            captured["llm_client"] = llm_client
            captured["config"] = config

    monkeypatch.setattr(
        "app.core.config.settings.PLANNING_LLM_PROVIDER",
        "local",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.settings.LOCAL_MODEL_NAME",
        "ministral-3b-q6k.gguf",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.settings.LOCAL_MODEL_BASE_URL",
        "https://openai.dada-tuda.ru/v1",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.local_model_settings.LOCAL_MODEL_USER",
        "user",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.local_model_settings.LOCAL_MODEL_PASSWORD",
        "password",
    )
    monkeypatch.setattr(
        "app.metrics.llm.provider_factory.LocalModelLLMClient",
        _FakeLocalModelClient,
    )
    monkeypatch.setattr(
        "app.planning.service.PlanningService",
        _FakePlanningService,
    )

    service, provider, model_name = local_test_one_sample._build_planning_service(
        show_prompts=True,
    )

    assert service is not None
    assert provider == "local"
    assert model_name == "ministral-3b-q6k.gguf"
    assert captured["config"].default_model_name == "ministral-3b-q6k.gguf"

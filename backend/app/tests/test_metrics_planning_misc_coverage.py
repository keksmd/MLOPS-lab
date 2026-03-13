
from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest


def test_import_metrics_and_planning_modules() -> None:
    module_names = [
        "app.metrics",
        "app.metrics.aggregation",
        "app.metrics.config",
        "app.metrics.evaluator",
        "app.metrics.exceptions",
        "app.metrics.heuristics",
        "app.metrics.llm",
        "app.metrics.llm.base",
        "app.metrics.llm.openrouter_client",
        "app.metrics.parsers",
        "app.metrics.prompts",
        "app.metrics.schemas",
        "app.planning",
        "app.planning.api",
        "app.planning.api.routes",
        "app.planning.config",
        "app.planning.data",
        "app.planning.data.loaders",
        "app.planning.data.taskcraft",
        "app.planning.exceptions",
        "app.planning.few_shot",
        "app.planning.normalizers",
        "app.planning.parsers",
        "app.planning.prompting",
        "app.planning.schemas",
        "app.planning.service",
    ]
    imported = [importlib.import_module(name) for name in module_names]
    assert all(module is not None for module in imported)


def test_metrics_init_exports() -> None:
    metrics_module = importlib.import_module("app.metrics")
    assert hasattr(metrics_module, "MetricsConfig")
    assert hasattr(metrics_module, "MetricsEvaluator")


def test_planning_init_exports() -> None:
    planning_module = importlib.import_module("app.planning")
    assert hasattr(planning_module, "FewShotSelector")
    assert hasattr(planning_module, "PlanningConfig")
    assert hasattr(planning_module, "PlanningService")


def test_initial_data_init_creates_user_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    initial_data = importlib.import_module("app.initial_data")

    created: dict[str, Any] = {}

    def fake_get_user_by_email(*, session: object, email: str) -> None:
        return None

    def fake_create_user(*, session: object, user_create: object) -> object:
        created["email"] = user_create.email
        created["is_superuser"] = user_create.is_superuser
        return SimpleNamespace(email=user_create.email)

    monkeypatch.setattr(initial_data.crud, "get_user_by_email", fake_get_user_by_email)
    monkeypatch.setattr(initial_data.crud, "create_user", fake_create_user)
    monkeypatch.setattr(initial_data.settings, "FIRST_SUPERUSER", "admin@example.com")
    monkeypatch.setattr(initial_data.settings, "FIRST_SUPERUSER_PASSWORD", "password")

    initial_data.init(object())

    assert created["email"] == "admin@example.com"
    assert created["is_superuser"] is True


def test_initial_data_init_skips_existing_user(monkeypatch: pytest.MonkeyPatch) -> None:
    initial_data = importlib.import_module("app.initial_data")

    called = {"create_user": False}

    def fake_get_user_by_email(*, session: object, email: str) -> object:
        return object()

    def fake_create_user(*, session: object, user_create: object) -> object:
        called["create_user"] = True
        return object()

    monkeypatch.setattr(initial_data.crud, "get_user_by_email", fake_get_user_by_email)
    monkeypatch.setattr(initial_data.crud, "create_user", fake_create_user)
    monkeypatch.setattr(initial_data.settings, "FIRST_SUPERUSER", "admin@example.com")
    monkeypatch.setattr(initial_data.settings, "FIRST_SUPERUSER_PASSWORD", "password")

    initial_data.init(object())

    assert called["create_user"] is False


def test_metrics_and_planning_exceptions_are_subclasses() -> None:
    metrics_exceptions = importlib.import_module("app.metrics.exceptions")
    planning_exceptions = importlib.import_module("app.planning.exceptions")

    for name in (
        "LLMClientError",
        "JudgeResponseParseError",
    ):
        assert issubclass(getattr(metrics_exceptions, name), Exception)

    for name in (
        "PromptBuildError",
        "PredictionParseError",
    ):
        assert issubclass(getattr(planning_exceptions, name), Exception)


def test_configs_and_schema_defaults() -> None:
    metrics_config = importlib.import_module("app.metrics.config")
    planning_config = importlib.import_module("app.planning.config")
    metrics_schemas = importlib.import_module("app.metrics.schemas")
    planning_schemas = importlib.import_module("app.planning.schemas")

    judge_cfg = metrics_config.JudgeConfig()
    openrouter_cfg = metrics_config.OpenRouterConfig(api_key="key", model_name="model")
    metrics_cfg = metrics_config.MetricsConfig()
    planning_cfg = planning_config.PlanningConfig()

    assert judge_cfg.include_reasoning in (True, False)
    assert openrouter_cfg.api_key == "key"
    assert metrics_cfg.enable_judge_metrics in (True, False)
    assert isinstance(planning_cfg.default_model_name, str)

    prompt_artifacts = planning_schemas.PromptArtifacts(
        system_prompt="sys",
        user_prompt="usr",
    )
    assert prompt_artifacts.system_prompt == "sys"

    planner_example = planning_schemas.PlannerExample(
        task="task",
        output=metrics_schemas.PlannerOutput(plan=["one"], actions=[]),
    )
    assert planner_example.task == "task"


def test_llm_base_is_abstract() -> None:
    llm_base = importlib.import_module("app.metrics.llm.base")
    with pytest.raises(TypeError):
        llm_base.BaseLLMClient()

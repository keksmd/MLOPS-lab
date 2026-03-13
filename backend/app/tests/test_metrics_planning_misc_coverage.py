
from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.initial_data as initial_data
import app.metrics as metrics_pkg
import app.metrics.llm as metrics_llm_pkg
import app.planning as planning_pkg
import app.planning.api as planning_api_pkg
import app.planning.data as planning_data_pkg


def test_package_exports_are_importable() -> None:
    assert hasattr(metrics_pkg, "MetricsConfig")
    assert hasattr(metrics_pkg, "MetricsEvaluator")
    assert hasattr(metrics_llm_pkg, "BaseLLMClient")
    assert hasattr(metrics_llm_pkg, "OpenRouterLLMClient")
    assert hasattr(planning_pkg, "PlanningConfig")
    assert hasattr(planning_pkg, "PlanningService")
    assert hasattr(planning_api_pkg, "router")
    assert hasattr(planning_data_pkg, "FewShotDatasetLoader")


def test_initial_data_init_calls_init_db(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeSession:
        def __init__(self, engine: object) -> None:
            calls["engine"] = engine

        def __enter__(self) -> object:
            session = object()
            calls["session"] = session
            return session

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_init_db(session: object) -> None:
        calls["init_db_session"] = session

    fake_engine = object()

    monkeypatch.setattr(initial_data, "Session", _FakeSession)
    monkeypatch.setattr(initial_data, "engine", fake_engine)
    monkeypatch.setattr(initial_data, "init_db", fake_init_db)

    initial_data.init()

    assert calls["engine"] is fake_engine
    assert calls["init_db_session"] is calls["session"]


def test_initial_data_main_logs_and_calls_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(initial_data, "init", lambda: calls.append("init"))
    monkeypatch.setattr(initial_data.logger, "info", lambda msg: calls.append(msg))

    initial_data.main()

    assert calls == [
        "Creating initial data",
        "init",
        "Initial data created",
    ]


def test_planning_and_metrics_module_doc_paths() -> None:
    assert planning_pkg.__all__
    assert metrics_pkg.__all__

from __future__ import annotations

import json
from typing import TypeVar

import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.metrics.llm.base import BaseLLMClient
from app.metrics.schemas import PlannerOutput
from app.planning import FewShotSelector, PlanningConfig, PlanningService
from app.planning.api.routes import get_planning_service, router
from app.planning.data.loaders import (
    FewShotDatasetLoader,
    JsonArtifactRepository,
    ToolRegistryLoader,
)
from app.planning.data.taskcraft import (
    TOOL_DESCRIPTIONS,
    build_processed_dataset,
    build_tool_registry_from_raw,
    convert_taskcraft_row,
)
from app.planning.exceptions import PredictionParseError, PromptBuildError, RepositoryLoadError
from app.planning.normalizers import (
    clean_plan_text,
    normalize_action_arguments,
    normalize_planner_output,
    safe_to_obj,
    simplify_actions,
    split_plan_steps,
)
from app.planning.parsers import PlannerOutputParser
from app.planning.prompting import PlanningPromptBuilder
from app.planning.schemas import InferenceRequest

RETRIEVED_URL_PLACEHOLDER = "<retrieved_url>"
TARGET_URL_PLACEHOLDER = "<target_url>"
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class DummyTextLLMClient(BaseLLMClient):
    """Deterministic test client for planner service tests."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.response_text

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        return json.loads(self.response_text)

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        return schema.model_validate(json.loads(self.response_text))


def test_normalizers_taskcraft_and_loaders(tmp_json_artifacts, tmp_path) -> None:
    assert safe_to_obj('{"a": 1}') == {"a": 1}
    assert safe_to_obj([1, 2]) == [1, 2]
    assert safe_to_obj("   ") is None
    assert safe_to_obj("not-json") == "not-json"

    assert clean_plan_text(
        "Here is the plan of action that I will follow to solve the task:\n```\n1. Search\n2. Open\n```"
    ) == "1. Search\n2. Open"
    assert clean_plan_text(123) == ""

    assert split_plan_steps("1. Search\n2. Open") == ["Search", "Open"]
    assert split_plan_steps("- Search\n- Open") == ["Search", "Open"]
    assert split_plan_steps("") == []

    assert normalize_action_arguments(
        "crawl_pages",
        {"url": "https://example.com", "file_path": "/tmp/x"},
    ) == {"url": RETRIEVED_URL_PLACEHOLDER}
    assert normalize_action_arguments(
        "find_archived_url",
        {"url": "https://example.com/page", "date": "20240101", "ignored": "value"},
    ) == {
        "url": TARGET_URL_PLACEHOLDER,
        "date": "20240101",
        "ignored": "value",
    }
    assert normalize_action_arguments("web_search", "bad") == {}

    actions = simplify_actions(
        [
            {"tool_name": "web_search", "arguments": {"query": "x"}},
            {"tool_name": "web_search", "arguments": {"query": "x"}},
            {"tool_name": "final_answer", "arguments": {"answer": "y"}},
            {"tool_name": "crawl_pages", "arguments": {"url": "https://example.com"}},
            "bad-action",
        ]
    )

    assert len(actions) == 2
    assert actions[1].arguments["url"] == RETRIEVED_URL_PLACEHOLDER
    assert simplify_actions("bad") == []

    output = normalize_planner_output(
        {
            "plan": "1. Search\n2. Open",
            "actions": [action.model_dump() for action in actions],
        }
    )
    assert output.plan == ["Search", "Open"]
    assert len(output.actions) == 2
    assert normalize_planner_output("bad payload").model_dump() == {"plan": [], "actions": []}

    raw_row = {
        "query": "What is the received date?",
        "ans_from_agent": str(
            {
                "trace": {
                    "plan": "1. Search\n2. Open the page",
                    "actions": [
                        {
                            "tool_name": "web_search",
                            "arguments": {"query": "received date"},
                        },
                        {
                            "tool_name": "crawl_pages",
                            "arguments": {"url": "https://example.com"},
                        },
                        {
                            "tool_name": "final_answer",
                            "arguments": {"answer": "x"},
                        },
                    ],
                }
            }
        ),
    }

    converted = convert_taskcraft_row(raw_row)
    assert converted["task"] == "What is the received date?"
    assert converted["plan"] == ["Search", "Open the page"]
    assert converted["actions"][1]["arguments"]["url"] == RETRIEVED_URL_PLACEHOLDER
    assert convert_taskcraft_row({"query": "x", "ans_from_agent": "bad"}) == {
        "task": "x",
        "plan": [],
        "actions": [],
    }
    assert convert_taskcraft_row({"query": "x", "ans_from_agent": {"trace": "bad"}}) == {
        "task": "x",
        "plan": [],
        "actions": [],
    }

    raw_df = pd.DataFrame([raw_row])
    processed_df = build_processed_dataset(raw_df)
    assert list(processed_df.columns) == [
        "task",
        "plan",
        "actions",
        "n_plan_steps",
        "n_actions",
    ]

    registry = build_tool_registry_from_raw(raw_df)
    assert [tool.tool_name for tool in registry] == ["web_search", "crawl_pages"]
    assert TOOL_DESCRIPTIONS["find_archived_url"]["arguments"]["date"]

    repository = JsonArtifactRepository()
    records = repository.load_records(tmp_json_artifacts["dataset"])
    assert len(records) == 2

    with pytest.raises(RepositoryLoadError):
        repository.load_records(tmp_path / "missing.json")

    wrapped_path = tmp_path / "wrapped.json"
    wrapped_path.write_text(
        json.dumps({"data": [{"task": "wrapped", "plan": [], "actions": []}]}),
        encoding="utf-8",
    )
    wrapped_records = repository.load_records(wrapped_path)
    assert wrapped_records[0]["task"] == "wrapped"

    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        '\n'.join([
            json.dumps({"task": "a", "plan": [], "actions": []}),
            json.dumps({"task": "b", "plan": [], "actions": []}),
        ]),
        encoding="utf-8",
    )
    assert len(repository.load_records(jsonl_path)) == 2

    tools = ToolRegistryLoader().load(tmp_json_artifacts["tools"])
    assert tools[0].tool_name == "web_search"
    assert tools[0].arguments[0].name == "query"
    optional_arg = next(arg for arg in tools[0].arguments if arg.name == "filter_year")
    assert optional_arg.required is False

    examples = FewShotDatasetLoader().load_examples(tmp_json_artifacts["dataset"])
    assert len(examples) == 2
    assert examples[0].output.plan

    query_fallback_path = tmp_path / "query_only.json"
    query_fallback_path.write_text(
        json.dumps([{"query": "fallback task", "plan": [], "actions": []}]),
        encoding="utf-8",
    )
    query_examples = FewShotDatasetLoader().load_examples(query_fallback_path)
    assert query_examples[0].task == "fallback task"


def test_prompt_builder_parser_service_and_route(tool_specs, planner_examples) -> None:
    prompt_builder = PlanningPromptBuilder()
    request = InferenceRequest(
        task="Find the received date.",
        available_tools=tool_specs,
        few_shot_examples=planner_examples[:2],
    )

    artifacts = prompt_builder.build(request)

    assert prompt_builder.get_format_instructions() in artifacts.system_prompt
    assert "Available tools:" in artifacts.system_prompt
    assert "Example 1" in artifacts.user_prompt
    assert "Task:\nFind the received date." in artifacts.user_prompt

    request_no_examples = InferenceRequest(
        task="Find the received date.",
        available_tools=tool_specs,
        few_shot_examples=[],
    )
    artifacts_no_examples = prompt_builder.build(request_no_examples)
    assert "Example 1" not in artifacts_no_examples.user_prompt

    with pytest.raises(PromptBuildError):
        prompt_builder.build(
            InferenceRequest(
                task=" ",
                available_tools=tool_specs,
            )
        )

    parser = PlannerOutputParser()
    parsed = parser.parse(
        '```json\n{"plan": ["Search", "Open"], "actions": [{"tool_name": "web_search", "arguments": {"query": "x"}}]}\n```'
    )
    assert parsed.plan == ["Search", "Open"]
    assert parsed.actions[0].tool_name == "web_search"
    assert parser.parse({"plan": ["Search"], "actions": []}).plan == ["Search"]
    assert parser.parse(PlannerOutput(plan=["Search"], actions=[])).plan == ["Search"]
    assert "JSON schema" in parser.get_format_instructions()

    with pytest.raises(PredictionParseError):
        parser.parse("not valid json")
    with pytest.raises(PredictionParseError):
        parser.parse("")

    llm_client = DummyTextLLMClient(
        json.dumps(
            {
                "plan": ["Search", "Open"],
                "actions": [
                    {
                        "tool_name": "web_search",
                        "arguments": {"query": "received date"},
                    },
                    {
                        "tool_name": "crawl_pages",
                        "arguments": {"url": "https://example.com"},
                    },
                ],
            }
        )
    )

    service = PlanningService(
        llm_client=llm_client,
        config=PlanningConfig(
            include_prompt_debug=True,
            include_raw_response=True,
        ),
        prompt_builder=prompt_builder,
        output_parser=parser,
    )

    result = service.predict(request)
    assert result.prediction.actions[1].arguments["url"] == RETRIEVED_URL_PLACEHOLDER
    assert result.raw_response is not None
    assert result.prompt_artifacts is not None
    assert result.metadata["few_shot_count"] == 2

    result2 = service.predict_from_parts(
        task="Find the received date.",
        available_tools=tool_specs,
        few_shot_examples=planner_examples[:1],
        model_name="custom-model",
    )
    assert isinstance(result2.prediction, PlannerOutput)
    assert result2.model_name == "custom-model"

    service_minimal = PlanningService(
        llm_client=llm_client,
        config=PlanningConfig(
            include_prompt_debug=False,
            include_raw_response=False,
        ),
        prompt_builder=prompt_builder,
        output_parser=parser,
    )
    minimal_result = service_minimal.predict(request_no_examples)
    assert minimal_result.raw_response is None
    assert minimal_result.prompt_artifacts is None

    selector = FewShotSelector()
    selected = selector.select_by_indices(
        planner_examples,
        [3, 100],
        fallback_count=3,
    )
    assert len(selected) == 3
    assert selected[0].task == "Task 4"

    enough_selected = selector.select_by_indices(planner_examples, [0, 1, 2], fallback_count=2)
    assert len(enough_selected) == 2

    bad_examples = planner_examples + [
        planner_examples[0].model_copy(update={"task": "   "}),
        planner_examples[0].model_copy(update={"output": PlannerOutput(plan=[], actions=[])}),
    ]
    fallback_selected = selector.select_by_indices(bad_examples, [4, 5], fallback_count=2)
    assert all(example.task.strip() for example in fallback_selected)
    assert all(example.output.plan for example in fallback_selected)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_planning_service] = lambda: service

    client = TestClient(app)
    response = client.post(
        "/planning/predict",
        json={
            "task": "Find the received date.",
            "available_tools": [tool.model_dump() for tool in tool_specs],
            "few_shot_examples": [
                example.model_dump() for example in planner_examples[:1]
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"]["actions"][0]["tool_name"] == "web_search"


class _FakeRouteLLMClient(BaseLLMClient):
    """Minimal fake OpenRouter client used to validate route wiring."""

    def __init__(self, config: object) -> None:
        self.config = config

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        raise NotImplementedError

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        raise NotImplementedError


def test_get_planning_service_uses_central_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.planning.api.routes.settings.OPENROUTER_API_KEY",
        "test-key",
    )
    monkeypatch.setattr(
        "app.planning.api.routes.settings.OPENROUTER_MODEL_NAME",
        "test-model",
    )
    monkeypatch.setattr(
        "app.planning.api.routes.OpenRouterLLMClient",
        _FakeRouteLLMClient,
    )

    service = get_planning_service()

    assert service.config.default_model_name == "test-model"
    assert isinstance(service.llm_client, _FakeRouteLLMClient)
    assert service.llm_client.config.api_key == "test-key"
    assert service.llm_client.config.model_name == "test-model"


def test_get_planning_service_without_api_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.planning.api.routes.settings.OPENROUTER_API_KEY",
        None,
    )

    with pytest.raises(HTTPException) as exc_info:
        get_planning_service()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "OPENROUTER_API_KEY is not configured"

from __future__ import annotations

import json
from typing import Any, TypeVar

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from app.metrics import MetricsConfig, MetricsEvaluator
from app.metrics.aggregation import (
    compute_aggregate_metrics,
    compute_plan_score,
    compute_solvability_score,
    compute_tool_score,
    compute_validity_score,
)
from app.metrics.config import JudgeConfig, OpenRouterConfig
from app.metrics.exceptions import JudgeResponseParseError, LLMClientError
from app.metrics.heuristics import (
    compute_arg_name_valid_rate,
    compute_duplicate_action_rate,
    compute_duplicate_step_rate,
    compute_find_archived_url_date_format_rate,
    compute_heuristic_metrics,
    compute_placeholder_compliance_rate,
    compute_required_arg_presence_rate,
    compute_tool_set_f1,
    compute_web_search_query_nonempty_rate,
)
from app.metrics.llm.base import BaseLLMClient
from app.metrics.llm.openrouter_client import OpenRouterLLMClient
from app.metrics.parsers import parse_judge_response
from app.metrics.prompts import JudgePromptBuilder
from app.metrics.schemas import EvaluationSample

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class DummyJSONLLMClient(BaseLLMClient):
    """Deterministic schema-aware test client."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(self.payload)

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return self.payload

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        return schema.model_validate(self.payload)


def test_parse_judge_response_and_aggregation(
    tool_specs,
    predicted_output,
    gold_output,
    judge_payload,
) -> None:
    sample = EvaluationSample(
        sample_id="sample-1",
        task="Find the article received date.",
        available_tools=tool_specs,
        prediction=predicted_output,
        golden=gold_output,
        raw_prediction=json.dumps(predicted_output.model_dump()),
    )

    heuristics = compute_heuristic_metrics(sample, similarity_fn=lambda _a, _b: 0.7)
    judge = parse_judge_response(judge_payload)
    aggregate = compute_aggregate_metrics(heuristics, judge, MetricsConfig().weights)

    assert heuristics.json_valid == 1.0
    assert heuristics.schema_valid == 1.0
    assert heuristics.nonempty_plan == 1.0
    assert heuristics.allowed_tool_rate == 1.0
    assert heuristics.forbidden_tool_rate == 0.0
    assert compute_arg_name_valid_rate(sample) == 1.0
    assert compute_required_arg_presence_rate(sample) == 1.0
    assert compute_duplicate_action_rate(sample) == 0.5
    assert compute_duplicate_step_rate(sample) > 0.0
    assert compute_web_search_query_nonempty_rate(sample) == 1.0
    assert compute_find_archived_url_date_format_rate(sample) is None
    assert compute_placeholder_compliance_rate(sample) == 1.0

    precision, recall, f1 = compute_tool_set_f1(sample)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(0.8)

    assert compute_validity_score(heuristics) <= 1.0
    assert compute_plan_score(judge) == pytest.approx(
        (0.9 + 0.8 + 0.85 + 0.75 + 0.5 + 0.75) / 6
    )
    assert compute_tool_score(judge) == pytest.approx((0.9 + 0.8 + 0.9) / 3)
    assert compute_solvability_score(judge) == pytest.approx(0.8)
    assert 0.0 <= aggregate.final_score <= 1.0


def test_judge_prompt_builder_uses_external_templates_and_parser(
    tool_specs,
    predicted_output,
    gold_output,
) -> None:
    sample = EvaluationSample(
        sample_id="sample-2",
        task="Find the article received date.",
        available_tools=tool_specs,
        prediction=predicted_output,
        golden=gold_output,
        raw_prediction=json.dumps(predicted_output.model_dump()),
    )

    prompt_builder = JudgePromptBuilder(
        JudgeConfig(
            use_reference_aware_judge=True,
            include_reasoning=False,
        )
    )

    system_prompt = prompt_builder.build_system_prompt()
    user_prompt = prompt_builder.build_user_prompt(sample)

    assert prompt_builder.get_format_instructions() in system_prompt
    assert "Set reasoning to an empty string." in system_prompt
    assert "golden_reference" in user_prompt
    assert '"task": "Find the article received date."' in user_prompt


def test_judge_prompt_builder_without_reference_omits_golden_reference(
    tool_specs,
    predicted_output,
) -> None:
    sample = EvaluationSample(
        sample_id="sample-3",
        task="Find the article received date.",
        available_tools=tool_specs,
        prediction=predicted_output,
        golden=None,
        raw_prediction=json.dumps(predicted_output.model_dump()),
    )

    prompt_builder = JudgePromptBuilder(
        JudgeConfig(
            use_reference_aware_judge=False,
            include_reasoning=True,
        )
    )

    system_prompt = prompt_builder.build_system_prompt()
    user_prompt = prompt_builder.build_user_prompt(sample)

    assert "Judge the prediction on its own merits." in system_prompt
    assert "golden_reference" not in user_prompt


def test_judge_prompt_and_evaluator_dataset(
    tool_specs,
    predicted_output,
    gold_output,
    judge_payload,
) -> None:
    sample = EvaluationSample(
        sample_id="sample-4",
        task="Find the article received date.",
        available_tools=tool_specs,
        prediction=predicted_output,
        golden=gold_output,
        raw_prediction=json.dumps(predicted_output.model_dump()),
    )

    prompt_builder = JudgePromptBuilder(
        JudgeConfig(
            use_reference_aware_judge=True,
            include_reasoning=False,
        )
    )

    evaluator = MetricsEvaluator(
        config=MetricsConfig(enable_judge_metrics=True),
        llm_client=DummyJSONLLMClient(judge_payload),
        prompt_builder=prompt_builder,
    )

    sample_result = evaluator.evaluate_sample(sample)
    dataset_result = evaluator.evaluate_dataset(
        [sample, sample],
        include_per_sample=True,
    )

    assert sample_result["aggregate"]["final_score"] <= 1.0
    assert dataset_result["sample_count"] == 2
    assert dataset_result["metrics"]["plan_relevance"] == pytest.approx(
        judge_payload["plan_relevance"]
    )
    assert len(dataset_result["per_sample"]) == 2


def test_parse_judge_response_invalid_payload_raises() -> None:
    with pytest.raises(JudgeResponseParseError):
        parse_judge_response({"plan_relevance": 0.5})


class _FakeStructuredModel:
    def __init__(self, response: object) -> None:
        self._response = response

    def invoke(self, messages: list[object]) -> object:
        return self._response


class _FakeChatOpenRouter:
    def __init__(self, **_: object) -> None:
        pass

    def invoke(self, messages: list[object]) -> AIMessage:
        return AIMessage(content='{"ok": true, "value": 1}')

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **_: object,
    ) -> _FakeStructuredModel:
        return _FakeStructuredModel({"ok": True, "value": 1})


class _FakeBrokenChatOpenRouter:
    def __init__(self, **_: object) -> None:
        pass

    def invoke(self, messages: list[object]) -> AIMessage:
        return AIMessage(content="not json")

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **_: object,
    ) -> _FakeStructuredModel:
        return _FakeStructuredModel("not a valid object")


def test_openrouter_client_generate_text_and_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.metrics.llm.openrouter_client.ChatOpenRouter",
        _FakeChatOpenRouter,
    )

    client = OpenRouterLLMClient(
        config=OpenRouterConfig(
            api_key="test-key",
            model_name="test-model",
            max_retries=2,
            retry_backoff_seconds=0.0,
        )
    )

    raw = client.generate_text(system_prompt="sys", user_prompt="user")
    assert '"ok": true' in raw

    obj = client.generate_json(system_prompt="sys", user_prompt="user")
    assert obj == {"ok": True, "value": 1}


def test_openrouter_client_invalid_json_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.metrics.llm.openrouter_client.ChatOpenRouter",
        _FakeBrokenChatOpenRouter,
    )

    client = OpenRouterLLMClient(
        config=OpenRouterConfig(
            api_key="test-key",
            model_name="test-model",
            max_retries=1,
            retry_backoff_seconds=0.0,
        )
    )

    with pytest.raises(LLMClientError):
        client.generate_json(system_prompt="sys", user_prompt="user")


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


def _build_sample(
    *,
    tool_specs,
    predicted_output,
    gold_output=None,
    sample_id: str = "sample",
    raw_prediction: str | None = None,
) -> EvaluationSample:
    return EvaluationSample(
        sample_id=sample_id,
        task="Find the article received date.",
        available_tools=tool_specs,
        prediction=predicted_output,
        golden=gold_output,
        raw_prediction=(
            raw_prediction
            if raw_prediction is not None
            else json.dumps(predicted_output.model_dump())
        ),
    )


def test_parse_judge_response_and_aggregation(
    tool_specs,
    predicted_output,
    gold_output,
    judge_payload,
) -> None:
    sample = _build_sample(
        tool_specs=tool_specs,
        predicted_output=predicted_output,
        gold_output=gold_output,
        sample_id="sample-1",
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


def test_parse_judge_response_invalid_payload_raises() -> None:
    with pytest.raises(JudgeResponseParseError):
        parse_judge_response({"plan_relevance": 0.5})


def test_judge_prompt_builder_paths(
    tool_specs,
    predicted_output,
    gold_output,
) -> None:
    sample = _build_sample(
        tool_specs=tool_specs,
        predicted_output=predicted_output,
        gold_output=gold_output,
        sample_id="sample-2",
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

    prompt_builder_no_ref = JudgePromptBuilder(
        JudgeConfig(
            use_reference_aware_judge=False,
            include_reasoning=True,
        )
    )
    system_prompt_no_ref = prompt_builder_no_ref.build_system_prompt()
    user_prompt_no_ref = prompt_builder_no_ref.build_user_prompt(
        _build_sample(
            tool_specs=tool_specs,
            predicted_output=predicted_output,
            gold_output=None,
            sample_id="sample-3",
        )
    )
    assert "Judge the prediction on its own merits." in system_prompt_no_ref
    assert "golden_reference" not in user_prompt_no_ref
    assert "Provide a short reasoning string" in system_prompt_no_ref


def test_judge_prompt_and_evaluator_dataset(
    tool_specs,
    predicted_output,
    gold_output,
    judge_payload,
) -> None:
    sample = _build_sample(
        tool_specs=tool_specs,
        predicted_output=predicted_output,
        gold_output=gold_output,
        sample_id="sample-4",
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
    assert "judge_structured_response" in sample_result["debug"]
    assert "judge_system_prompt" in sample_result["debug"]
    assert "judge_user_prompt" in sample_result["debug"]
    assert dataset_result["sample_count"] == 2
    assert dataset_result["metrics"]["plan_relevance"] == pytest.approx(
        judge_payload["plan_relevance"]
    )
    assert len(dataset_result["per_sample"]) == 2


def test_evaluator_without_judge_metrics_returns_no_judge(
    tool_specs,
    predicted_output,
    gold_output,
) -> None:
    sample = _build_sample(
        tool_specs=tool_specs,
        predicted_output=predicted_output,
        gold_output=gold_output,
        sample_id="sample-5",
    )

    evaluator = MetricsEvaluator(
        config=MetricsConfig(enable_judge_metrics=False),
        llm_client=None,
    )

    result = evaluator.evaluate_sample(sample)
    dataset_result = evaluator.evaluate_dataset([sample], include_per_sample=False)

    assert result["judge"] is None
    assert result["debug"] == {}
    assert 0.0 <= result["aggregate"]["final_score"] <= 1.0
    assert dataset_result["sample_count"] == 1
    assert dataset_result["per_sample"] == []


def test_evaluator_with_judge_enabled_and_no_llm_client_raises(
    tool_specs,
    predicted_output,
    gold_output,
) -> None:
    sample = _build_sample(
        tool_specs=tool_specs,
        predicted_output=predicted_output,
        gold_output=gold_output,
        sample_id="sample-6",
    )

    evaluator = MetricsEvaluator(
        config=MetricsConfig(enable_judge_metrics=True),
        llm_client=None,
    )

    with pytest.raises(ValueError):
        evaluator.evaluate_sample(sample)


class _FakeStructuredModel:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error

    def invoke(self, messages: list[object]) -> object:
        if self._error is not None:
            raise self._error
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


class _FakeNonTextChatOpenRouter:
    def __init__(self, **_: object) -> None:
        pass

    def invoke(self, messages: list[object]) -> AIMessage:
        return AIMessage(content=[{"type": "image_url", "url": "https://example.com"}])

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **_: object,
    ) -> _FakeStructuredModel:
        return _FakeStructuredModel({"ok": True, "value": 1})


class _FallbackChatOpenRouter:
    def __init__(self, **_: object) -> None:
        self.calls = 0

    def invoke(self, messages: list[object]) -> AIMessage:
        return AIMessage(content='{"ok": true, "value": 1}')

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **_: object,
    ) -> _FakeStructuredModel:
        self.calls += 1
        if self.calls == 1:
            return _FakeStructuredModel(error=RuntimeError("first strategy failed"))
        return _FakeStructuredModel({"ok": True, "value": 1})


def test_openrouter_client_paths(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_openrouter_client_error_and_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.metrics.llm.openrouter_client.ChatOpenRouter",
        _FakeBrokenChatOpenRouter,
    )

    broken_client = OpenRouterLLMClient(
        config=OpenRouterConfig(
            api_key="test-key",
            model_name="test-model",
            max_retries=1,
            retry_backoff_seconds=0.0,
        )
    )

    with pytest.raises(LLMClientError):
        broken_client.generate_json(system_prompt="sys", user_prompt="user")

    monkeypatch.setattr(
        "app.metrics.llm.openrouter_client.ChatOpenRouter",
        _FakeNonTextChatOpenRouter,
    )

    non_text_client = OpenRouterLLMClient(
        config=OpenRouterConfig(
            api_key="test-key",
            model_name="test-model",
            max_retries=1,
            retry_backoff_seconds=0.0,
        )
    )

    with pytest.raises(LLMClientError):
        non_text_client.generate_text(system_prompt="sys", user_prompt="user")

    monkeypatch.setattr(
        "app.metrics.llm.openrouter_client.ChatOpenRouter",
        _FallbackChatOpenRouter,
    )

    class SimpleSchema(BaseModel):
        ok: bool
        value: int

    fallback_client = OpenRouterLLMClient(
        config=OpenRouterConfig(
            api_key="test-key",
            model_name="test-model",
            max_retries=1,
            retry_backoff_seconds=0.0,
        )
    )

    result = fallback_client.generate_structured(
        system_prompt="sys",
        user_prompt="user",
        schema=SimpleSchema,
    )

    assert result.ok is True
    assert result.value == 1


def test_heuristics_with_invalid_raw_prediction_still_compute(
    tool_specs,
    predicted_output,
    gold_output,
) -> None:
    sample = _build_sample(
        tool_specs=tool_specs,
        predicted_output=predicted_output,
        gold_output=gold_output,
        sample_id="sample-7",
        raw_prediction="not json at all",
    )

    heuristics = compute_heuristic_metrics(sample, similarity_fn=None)

    assert heuristics.json_valid == 0.0
    assert heuristics.nonempty_plan == 1.0

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
    compute_allowed_tool_rate,
    compute_arg_name_valid_rate,
    compute_duplicate_action_rate,
    compute_duplicate_step_rate,
    compute_find_archived_url_date_format_rate,
    compute_forbidden_tool_rate,
    compute_heuristic_metrics,
    compute_json_valid,
    compute_placeholder_compliance_rate,
    compute_plan_semantic_similarity,
    compute_required_arg_presence_rate,
    compute_schema_valid,
    compute_tool_set_f1,
    compute_web_search_query_nonempty_rate,
)
from app.metrics.llm.base import BaseLLMClient
from app.metrics.llm.openrouter_client import OpenRouterLLMClient
from app.metrics.parsers import parse_judge_response
from app.metrics.prompts import JudgePromptBuilder
from app.metrics.schemas import EvaluationSample, JudgeMetricScores, PlannerOutput

RETRIEVED_URL_PLACEHOLDER = "<retrieved_url>"
TARGET_URL_PLACEHOLDER = "<target_url>"
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


class RaisingLLMClient(BaseLLMClient):
    """Client that always fails during structured generation."""

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        raise RuntimeError("judge failed")


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
    with pytest.raises(JudgeResponseParseError) as exc_info:
        parse_judge_response({"plan_relevance": 0.5})

    assert "plan_completeness" in str(exc_info.value)


def test_judge_prompt_builder_variants(
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
    assert "Provide a short reasoning string" in system_prompt_no_ref
    assert "golden_reference" not in user_prompt_no_ref
    assert "predicted_output" in user_prompt_no_ref


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

    evaluator = MetricsEvaluator(
        config=MetricsConfig(enable_judge_metrics=True),
        llm_client=DummyJSONLLMClient(judge_payload),
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

    assert result["judge"] is None
    assert result["debug"] == {}
    assert 0.0 <= result["aggregate"]["final_score"] <= 1.0


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


class _AlwaysFailChatOpenRouter:
    def __init__(self, **_: object) -> None:
        pass

    def invoke(self, messages: list[object]) -> AIMessage:
        raise RuntimeError("provider down")

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **_: object,
    ) -> _FakeStructuredModel:
        return _FakeStructuredModel(error=RuntimeError("all failed"))


class _SchemaChatOpenRouter:
    def __init__(self, **_: object) -> None:
        pass

    def invoke(self, messages: list[object]) -> AIMessage:
        return AIMessage(content="unused")

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **_: object,
    ) -> _FakeStructuredModel:
        class SimpleSchema(BaseModel):
            ok: bool
            value: int

        return _FakeStructuredModel(SimpleSchema(ok=True, value=7))


def test_openrouter_client_text_json_and_structured_paths(
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

    class SimpleSchema(BaseModel):
        ok: bool
        value: int

    structured = client.generate_structured(
        system_prompt="sys",
        user_prompt="user",
        schema=SimpleSchema,
    )
    assert structured.value == 1


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


def test_openrouter_client_non_text_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.metrics.llm.openrouter_client.ChatOpenRouter",
        _FakeNonTextChatOpenRouter,
    )

    client = OpenRouterLLMClient(
        config=OpenRouterConfig(api_key="test-key", model_name="test-model")
    )

    with pytest.raises(LLMClientError):
        client.generate_text(system_prompt="sys", user_prompt="user")

    monkeypatch.setattr(
        "app.metrics.llm.openrouter_client.ChatOpenRouter",
        _AlwaysFailChatOpenRouter,
    )
    failing_client = OpenRouterLLMClient(
        config=OpenRouterConfig(api_key="test-key", model_name="test-model")
    )

    with pytest.raises(LLMClientError):
        failing_client.generate_text(system_prompt="sys", user_prompt="user")
    with pytest.raises(LLMClientError):
        failing_client.generate_json(system_prompt="sys", user_prompt="user")


def test_openrouter_client_structured_output_fallback_and_schema_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimpleSchema(BaseModel):
        ok: bool
        value: int

    monkeypatch.setattr(
        "app.metrics.llm.openrouter_client.ChatOpenRouter",
        _FallbackChatOpenRouter,
    )
    client = OpenRouterLLMClient(
        config=OpenRouterConfig(api_key="test-key", model_name="test-model")
    )
    result = client.generate_structured(
        system_prompt="sys",
        user_prompt="user",
        schema=SimpleSchema,
    )
    assert result.ok is True
    assert result.value == 1

    monkeypatch.setattr(
        "app.metrics.llm.openrouter_client.ChatOpenRouter",
        _SchemaChatOpenRouter,
    )
    client_schema = OpenRouterLLMClient(
        config=OpenRouterConfig(api_key="test-key", model_name="test-model")
    )
    result_schema = client_schema.generate_structured(
        system_prompt="sys",
        user_prompt="user",
        schema=SimpleSchema,
    )
    assert isinstance(result_schema, SimpleSchema)
    assert result_schema.value == 7


def test_heuristics_edge_cases(tool_specs, gold_output) -> None:
    invalid_prediction = PlannerOutput.model_construct(
        plan="not-a-list",
        actions="not-a-list",
    )
    sample_invalid = EvaluationSample.model_construct(
        sample_id="invalid",
        task="broken",
        available_tools=tool_specs,
        prediction=invalid_prediction,
        golden=gold_output,
        raw_prediction="[]",
    )
    assert compute_json_valid(sample_invalid) == 0.0
    assert compute_schema_valid(sample_invalid) == 0.0

    sample_empty = EvaluationSample(
        sample_id="empty",
        task="empty",
        available_tools=tool_specs,
        prediction=PlannerOutput(plan=[], actions=[]),
        golden=PlannerOutput(plan=[], actions=[]),
        raw_prediction=None,
    )
    assert compute_allowed_tool_rate(sample_empty) == 1.0
    assert compute_forbidden_tool_rate(sample_empty) == 0.0
    assert compute_duplicate_action_rate(sample_empty) == 0.0
    assert compute_tool_set_f1(sample_empty) == (1.0, 1.0, 1.0)
    assert compute_plan_semantic_similarity(sample_empty, lambda _a, _b: 1.2) == 1.0

    sample_bad_args = EvaluationSample(
        sample_id="bad-args",
        task="bad args",
        available_tools=tool_specs,
        prediction=PlannerOutput(
            plan=["Use tools"],
            actions=[
                tool_specs[0].__class__.model_fields  # type: ignore[attr-defined]
            ],
        ),
        golden=None,
        raw_prediction="not-json",
    )
    # Rebuild a valid model-constructed sample to hit invalid arg/placeholder branches.
    sample_bad_args = EvaluationSample(
        sample_id="bad-args",
        task="bad args",
        available_tools=tool_specs,
        prediction=PlannerOutput.model_construct(
            plan=["Use tools", "Use tools"],
            actions=[
                type(gold_output.actions[0]).model_construct(
                    tool_name="web_search",
                    arguments={"wrong": "x", "query": ""},
                ),
                type(gold_output.actions[1]).model_construct(
                    tool_name="crawl_pages",
                    arguments={"url": "https://example.com"},
                ),
                type(gold_output.actions[2]).model_construct(
                    tool_name="find_archived_url",
                    arguments={"url": TARGET_URL_PLACEHOLDER, "date": "bad"},
                ),
                type(gold_output.actions[0]).model_construct(
                    tool_name="mystery_tool",
                    arguments={},
                ),
                type(gold_output.actions[0]).model_construct(
                    tool_name="final_answer",
                    arguments={},
                ),
            ],
        ),
        golden=gold_output,
        raw_prediction="not json at all",
    )

    assert compute_json_valid(sample_bad_args) == 0.0
    assert compute_arg_name_valid_rate(sample_bad_args) < 1.0
    assert compute_required_arg_presence_rate(sample_bad_args) < 1.0
    assert compute_duplicate_step_rate(sample_bad_args) > 0.0
    assert compute_web_search_query_nonempty_rate(sample_bad_args) == 0.0
    assert compute_find_archived_url_date_format_rate(sample_bad_args) == 0.0
    assert compute_placeholder_compliance_rate(sample_bad_args) == pytest.approx(0.5)
    assert compute_plan_semantic_similarity(sample_bad_args, lambda _a, _b: -0.5) == 0.0

    heuristics = compute_heuristic_metrics(sample_bad_args, similarity_fn=lambda _a, _b: 0.4)
    assert heuristics.plan_semantic_similarity == 0.4
    assert heuristics.action_count_diff is not None


def test_evaluator_propagates_llm_failures(
    tool_specs,
    predicted_output,
    gold_output,
) -> None:
    sample = _build_sample(
        tool_specs=tool_specs,
        predicted_output=predicted_output,
        gold_output=gold_output,
        sample_id="sample-7",
    )

    evaluator = MetricsEvaluator(
        config=MetricsConfig(enable_judge_metrics=True),
        llm_client=RaisingLLMClient(),
    )

    with pytest.raises(RuntimeError):
        evaluator.evaluate_sample(sample)

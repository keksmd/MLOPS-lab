from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from app import initial_data
from app.metrics import (
    ActionCall,
    AggregateMetricScores,
    DatasetMetricResult,
    EvaluationSample,
    HeuristicMetricScores,
    JudgeMetricScores,
    MetricsConfig,
    MetricsEvaluator,
    OpenRouterConfig,
    OpenRouterLLMClient,
    PlannerOutput,
    SampleMetricResult,
    ToolArgumentSpec,
    ToolSpec,
)
from app.metrics.config import JudgeConfig, MetricWeights
from app.metrics.exceptions import JudgeResponseParseError, LLMClientError
from app.metrics.llm import BaseLLMClient
from app.planning import (
    FewShotSelector,
    InferenceRequest,
    InferenceResult,
    PlannerExample,
    PlanningConfig,
    PlanningService,
    PromptArtifacts,
)
from app.planning.api import router as planning_router
from app.planning.data import (
    FewShotDatasetLoader,
    JsonArtifactRepository,
    ToolRegistryLoader,
    build_processed_dataset,
    build_tool_registry_from_raw,
    convert_taskcraft_row,
)
from app.planning.exceptions import (
    PlanningError,
    PredictionParseError,
    PromptBuildError,
    RepositoryLoadError,
)


def test_public_imports_and_schema_models(tool_specs, predicted_output, gold_output) -> None:
    tool_arg = ToolArgumentSpec(name="query", description="desc", required=True)
    tool = ToolSpec(tool_name="web_search", description="desc", arguments=[tool_arg])
    action = ActionCall(tool_name="web_search", arguments={"query": "x"})
    planner_output = PlannerOutput(plan=["Search"], actions=[action])
    sample = EvaluationSample(
        task="task",
        available_tools=[tool],
        prediction=planner_output,
        golden=gold_output,
        raw_prediction=None,
    )
    judge = JudgeMetricScores(
        plan_relevance=1.0,
        plan_completeness=1.0,
        plan_logic=1.0,
        plan_specificity=1.0,
        plan_nonredundancy=1.0,
        plan_duplicate_penalty=0.0,
        tool_appropriateness=1.0,
        tool_sufficiency=1.0,
        argument_quality=1.0,
        overall_solvability=1.0,
        critical_failure=False,
    )
    heuristics = HeuristicMetricScores(
        json_valid=1.0,
        schema_valid=1.0,
        nonempty_plan=1.0,
        allowed_tool_rate=1.0,
        forbidden_tool_rate=0.0,
        arg_name_valid_rate=1.0,
        required_arg_presence_rate=1.0,
        duplicate_action_rate=0.0,
        duplicate_step_rate=0.0,
        avg_plan_steps=1.0,
        avg_actions=1.0,
        tool_set_precision=1.0,
        tool_set_recall=1.0,
        tool_set_f1=1.0,
        action_count_diff=0.0,
        web_search_query_nonempty_rate=1.0,
        find_archived_url_date_format_rate=None,
        placeholder_compliance_rate=None,
        plan_semantic_similarity=None,
    )
    aggregate = AggregateMetricScores(
        validity_score=1.0,
        plan_score=1.0,
        tool_score=1.0,
        solvability_score=1.0,
        final_score=1.0,
        critical_failure_rate=0.0,
    )
    sample_result = SampleMetricResult(
        sample_id="1",
        heuristics=heuristics,
        judge=judge,
        aggregate=aggregate,
    )
    dataset_result = DatasetMetricResult(sample_count=1, metrics={"x": 1.0}, per_sample=[sample_result])

    planner_example = PlannerExample(task="task", output=planner_output)
    prompt_artifacts = PromptArtifacts(system_prompt="sys", user_prompt="user")
    inference_request = InferenceRequest(task="task", available_tools=[tool])
    inference_result = InferenceResult(prediction=planner_output, metadata={"x": 1})

    assert planner_output.actions[0].tool_name == "web_search"
    assert sample.golden is gold_output
    assert judge.reasoning == ""
    assert sample_result.aggregate.final_score == 1.0
    assert dataset_result.sample_count == 1
    assert planner_example.task == "task"
    assert prompt_artifacts.system_prompt == "sys"
    assert inference_request.available_tools[0].tool_name == "web_search"
    assert inference_result.metadata["x"] == 1

    metrics_config = MetricsConfig()
    planning_config = PlanningConfig()
    openrouter_config = OpenRouterConfig(api_key="key", model_name="model")
    judge_config = JudgeConfig()
    weights = MetricWeights()

    assert metrics_config.enable_judge_metrics is True
    assert planning_config.enforce_placeholder_rules is True
    assert openrouter_config.timeout_seconds == 120
    assert judge_config.use_reference_aware_judge is True
    assert weights.validity_weight + weights.plan_weight + weights.tool_weight + weights.solvability_weight == 1.0

    assert issubclass(BaseLLMClient, object)
    assert MetricsEvaluator
    assert OpenRouterLLMClient
    assert FewShotSelector
    assert PlanningService
    assert planning_router
    assert JsonArtifactRepository
    assert ToolRegistryLoader
    assert FewShotDatasetLoader
    assert convert_taskcraft_row
    assert build_processed_dataset
    assert build_tool_registry_from_raw


def test_exception_classes_and_module_exports() -> None:
    assert issubclass(PromptBuildError, PlanningError)
    assert issubclass(PredictionParseError, PlanningError)
    assert issubclass(RepositoryLoadError, PlanningError)
    assert issubclass(JudgeResponseParseError, Exception)
    assert issubclass(LLMClientError, Exception)


class _FakeSession:
    def __init__(self, engine: object) -> None:
        self.engine = engine
        self.entered = False
        self.exited = False

    def __enter__(self) -> str:
        self.entered = True
        return "session"

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True


def test_initial_data_init_and_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    monkeypatch.setattr(initial_data, "engine", object())
    monkeypatch.setattr(initial_data, "Session", _FakeSession)
    monkeypatch.setattr(initial_data, "init_db", lambda session: calls.append(("init_db", session)))
    monkeypatch.setattr(initial_data.logger, "info", lambda message: calls.append(message))

    initial_data.init()
    initial_data.main()

    assert ("init_db", "session") in calls
    assert "Creating initial data" in calls
    assert "Initial data created" in calls

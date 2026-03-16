from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolArgumentSpec(BaseModel):
    """Schema description for a single tool argument."""

    name: str = Field(..., description="Argument name.")
    description: str = Field(..., description="Human-readable argument description.")
    required: bool = Field(True, description="Whether the argument is required.")


class ToolSpec(BaseModel):
    """Description of a single available tool exposed to the planning model."""

    tool_name: str = Field(..., description="Unique tool identifier.")
    description: str = Field(..., description="Human-readable tool description.")
    arguments: list[ToolArgumentSpec] = Field(
        default_factory=list,
        description="List of arguments supported by the tool.",
    )


class ActionCall(BaseModel):
    """Compact representation of a predicted tool invocation."""

    tool_name: str = Field(..., description="Predicted tool name.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Predicted tool arguments.",
    )


class PlannerOutput(BaseModel):
    """Structured output of the planner model."""

    plan: list[str] = Field(
        default_factory=list,
        description="Ordered list of natural-language plan steps.",
    )
    actions: list[ActionCall] = Field(
        default_factory=list,
        description="Compact list of tool calls.",
    )


class EvaluationSample(BaseModel):
    """Single sample evaluated by the metrics module."""

    sample_id: str | None = Field(None, description="Optional sample identifier.")
    task: str = Field(..., description="Original user task.")
    available_tools: list[ToolSpec] = Field(
        default_factory=list,
        description="Tool registry available to the model and the judge.",
    )
    prediction: PlannerOutput = Field(..., description="Predicted plan and actions.")
    golden: PlannerOutput | None = Field(
        None,
        description="Optional gold reference used for gold-based metrics and judge prompts.",
    )
    raw_prediction: str | None = Field(
        None,
        description="Optional raw model response used for structural validation metrics.",
    )


class JudgeMetricScores(BaseModel):
    """LLM-as-a-judge scores for one sample, normalized to [0, 1]."""

    plan_relevance: float = Field(..., ge=0.0, le=1.0)
    plan_completeness: float = Field(..., ge=0.0, le=1.0)
    plan_logic: float = Field(..., ge=0.0, le=1.0)
    plan_specificity: float = Field(..., ge=0.0, le=1.0)
    plan_nonredundancy: float = Field(..., ge=0.0, le=1.0)
    plan_duplicate_penalty: float = Field(..., ge=0.0, le=1.0)

    tool_appropriateness: float = Field(..., ge=0.0, le=1.0)
    tool_sufficiency: float = Field(..., ge=0.0, le=1.0)
    argument_quality: float = Field(..., ge=0.0, le=1.0)

    overall_solvability: float = Field(..., ge=0.0, le=1.0)
    critical_failure: bool

    critical_missing_steps: list[str] = Field(default_factory=list)
    underspecified_steps: list[str] = Field(default_factory=list)
    unnecessary_actions: list[str] = Field(default_factory=list)
    bad_arguments: list[str] = Field(default_factory=list)
    duplicate_step_notes: list[str] = Field(default_factory=list)
    reasoning: str = ""


class HeuristicMetricScores(BaseModel):
    """Rule-based metrics for one sample."""

    json_valid: float = Field(..., ge=0.0, le=1.0)
    schema_valid: float = Field(..., ge=0.0, le=1.0)
    nonempty_plan: float = Field(..., ge=0.0, le=1.0)
    allowed_tool_rate: float = Field(..., ge=0.0, le=1.0)
    forbidden_tool_rate: float = Field(..., ge=0.0, le=1.0)
    arg_name_valid_rate: float = Field(..., ge=0.0, le=1.0)
    required_arg_presence_rate: float = Field(..., ge=0.0, le=1.0)
    duplicate_action_rate: float = Field(..., ge=0.0, le=1.0)
    duplicate_step_rate: float = Field(..., ge=0.0, le=1.0)

    avg_plan_steps: float = Field(..., ge=0.0)
    avg_actions: float = Field(..., ge=0.0)

    tool_set_precision: float | None = Field(None, ge=0.0, le=1.0)
    tool_set_recall: float | None = Field(None, ge=0.0, le=1.0)
    tool_set_f1: float | None = Field(None, ge=0.0, le=1.0)
    action_count_diff: float | None = Field(None, ge=0.0)

    web_search_query_nonempty_rate: float | None = Field(None, ge=0.0, le=1.0)
    find_archived_url_date_format_rate: float | None = Field(None, ge=0.0, le=1.0)
    placeholder_compliance_rate: float | None = Field(None, ge=0.0, le=1.0)
    plan_semantic_similarity: float | None = Field(None, ge=0.0, le=1.0)


class AggregateMetricScores(BaseModel):
    """Aggregated sample-level scores, all normalized to [0, 1]."""

    validity_score: float = Field(..., ge=0.0, le=1.0)
    plan_score: float = Field(..., ge=0.0, le=1.0)
    tool_score: float = Field(..., ge=0.0, le=1.0)
    solvability_score: float = Field(..., ge=0.0, le=1.0)
    final_score: float = Field(..., ge=0.0, le=1.0)
    critical_failure_rate: float = Field(..., ge=0.0, le=1.0)


class SampleMetricResult(BaseModel):
    """Complete metric output for a single evaluated sample."""

    sample_id: str | None = None
    heuristics: HeuristicMetricScores
    judge: JudgeMetricScores | None = None
    aggregate: AggregateMetricScores
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional debug artifacts such as raw judge payloads.",
    )


class DatasetMetricResult(BaseModel):
    """Aggregated metric output for a dataset split."""

    sample_count: int
    metrics: dict[str, float]
    per_sample: list[SampleMetricResult] = Field(default_factory=list)

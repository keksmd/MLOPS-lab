from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.metrics.schemas import PlannerOutput, ToolSpec


class PlannerExample(BaseModel):
    """Few-shot example used for planner prompting."""

    task: str = Field(..., description="Input task shown to the planning model.")
    output: PlannerOutput = Field(
        ..., description="Expected planner output for the task."
    )


class PromptArtifacts(BaseModel):
    """Prompt text built for a single inference call."""

    system_prompt: str = Field(..., description="System instruction sent to the model.")
    user_prompt: str = Field(..., description="User prompt sent to the model.")


class InferenceRequest(BaseModel):
    """Input payload for the planning inference service."""

    task: str = Field(..., description="User task to decompose into plan and actions.")
    available_tools: list[ToolSpec] = Field(
        default_factory=list,
        description="Tool registry available to the planning model.",
    )
    few_shot_examples: list[PlannerExample] = Field(
        default_factory=list,
        description="Optional few-shot demonstrations.",
    )
    model_name: str | None = Field(
        default=None,
        description="Optional override of the backend model name.",
    )


class InferenceResult(BaseModel):
    """Result produced by the planning inference service."""

    prediction: PlannerOutput = Field(..., description="Structured planner prediction.")
    model_name: str | None = Field(
        default=None, description="Model used for generation."
    )
    raw_response: str | None = Field(
        default=None,
        description="Raw model response before parsing and normalization.",
    )
    prompt_artifacts: PromptArtifacts | None = Field(
        default=None,
        description="Optional prompt debug information.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional service metadata, e.g. example ids or timing.",
    )

from __future__ import annotations

from pydantic import BaseModel, Field


class PlanningConfig(BaseModel):
    """Configuration of the planning inference pipeline."""

    max_few_shot_examples: int = Field(
        3,
        description="Maximum number of few-shot demonstrations inserted into the prompt.",
    )
    default_model_name: str = Field(
        default="meta-llama/llama-3.2-3b-instruct:free",
        description="Default model identifier used for baseline inference.",
    )
    include_prompt_debug: bool = Field(
        default=False,
        description="Whether prompt text should be returned in the inference result.",
    )
    include_raw_response: bool = Field(
        default=True,
        description="Whether the raw model response should be returned for debugging.",
    )
    enforce_placeholder_rules: bool = Field(
        default=True,
        description="Whether parser should normalize URL-like arguments to project placeholders.",
    )

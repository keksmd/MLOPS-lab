from __future__ import annotations

from time import perf_counter

from app.metrics.llm.base import BaseLLMClient
from app.metrics.schemas import PlannerOutput, ToolSpec

from .config import PlanningConfig
from .parsers import PlannerOutputParser
from .prompting import PlanningPromptBuilder
from .schemas import InferenceRequest, InferenceResult, PlannerExample


class PlanningService:
    """Main facade for planner inference."""

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient,
        config: PlanningConfig | None = None,
        prompt_builder: PlanningPromptBuilder | None = None,
        output_parser: PlannerOutputParser | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or PlanningConfig()
        self.prompt_builder = prompt_builder or PlanningPromptBuilder()
        self.output_parser = output_parser or PlannerOutputParser()

    def predict(self, request: InferenceRequest) -> InferenceResult:
        """Generate a planner prediction for a single task."""
        prompt_artifacts = self.prompt_builder.build(request)

        started_at = perf_counter()
        structured_prediction = self.llm_client.generate_structured(
            system_prompt=prompt_artifacts.system_prompt,
            user_prompt=prompt_artifacts.user_prompt,
            schema=PlannerOutput,
        )
        latency_seconds = perf_counter() - started_at

        prediction = self.output_parser.parse(structured_prediction)

        raw_response: str | None = None
        if self.config.include_raw_response:
            raw_response = prediction.model_dump_json(indent=2)

        return InferenceResult(
            prediction=prediction,
            model_name=request.model_name or self.config.default_model_name,
            raw_response=raw_response,
            prompt_artifacts=(
                prompt_artifacts if self.config.include_prompt_debug else None
            ),
            metadata={
                "latency_seconds": latency_seconds,
                "few_shot_count": len(request.few_shot_examples),
            },
        )

    def predict_from_parts(
        self,
        *,
        task: str,
        available_tools: list[ToolSpec],
        few_shot_examples: list[PlannerExample] | None = None,
        model_name: str | None = None,
    ) -> InferenceResult:
        """Convenience wrapper around predict()."""
        request = InferenceRequest(
            task=task,
            available_tools=available_tools,
            few_shot_examples=few_shot_examples or [],
            model_name=model_name,
        )
        return self.predict(request)
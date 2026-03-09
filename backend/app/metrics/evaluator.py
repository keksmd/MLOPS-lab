from __future__ import annotations

from typing import Any

from .aggregation import compute_aggregate_metrics
from .config import MetricsConfig
from .heuristics import SemanticSimilarityFn, compute_heuristic_metrics
from .parsers import parse_judge_response
from .prompts import JudgePromptBuilder
from .schemas import DatasetMetricResult, EvaluationSample, SampleMetricResult
from .llm.base import BaseLLMClient


class MetricsEvaluator:
    """Main facade for computing all metrics for task decomposition outputs."""

    def __init__(
        self,
        *,
        config: MetricsConfig,
        llm_client: BaseLLMClient | None = None,
        prompt_builder: JudgePromptBuilder | None = None,
        similarity_fn: SemanticSimilarityFn | None = None,
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            config: Metrics evaluation configuration.
            llm_client: Optional LLM client used for judge-based metrics.
            prompt_builder: Optional prompt builder for judge calls.
            similarity_fn: Optional semantic similarity scorer for plan similarity.
        """
        self.config = config
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or JudgePromptBuilder(config.judge)
        self.similarity_fn = similarity_fn if config.enable_semantic_similarity else None

    def evaluate_sample(self, sample: EvaluationSample) -> dict[str, Any]:
        """
        Compute the full metric bundle for a single sample.

        Returns:
            Plain dictionary with heuristic, judge, and aggregate metrics.
        """
        heuristic_scores = compute_heuristic_metrics(sample, similarity_fn=self.similarity_fn)

        judge_scores = None
        debug: dict[str, Any] = {}
        if self.config.enable_judge_metrics:
            if self.llm_client is None:
                raise ValueError("LLM client is required for judge-based metrics.")
            system_prompt = self.prompt_builder.build_system_prompt()
            user_prompt = self.prompt_builder.build_user_prompt(sample)
            raw_judge = self.llm_client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            judge_scores = parse_judge_response(raw_judge)
            debug["judge_raw_response"] = raw_judge
            debug["judge_system_prompt"] = system_prompt
            debug["judge_user_prompt"] = user_prompt

        aggregate_scores = compute_aggregate_metrics(
            heuristic_scores,
            judge_scores,
            self.config.weights,
        )

        result = SampleMetricResult(
            sample_id=sample.sample_id,
            heuristics=heuristic_scores,
            judge=judge_scores,
            aggregate=aggregate_scores,
            debug=debug,
        )
        return result.model_dump()

    def evaluate_dataset(
        self,
        samples: list[EvaluationSample],
        *,
        include_per_sample: bool = False,
    ) -> dict[str, Any]:
        """
        Compute dataset-level metrics by averaging sample-level numeric scores.

        Args:
            samples: Samples to evaluate.
            include_per_sample: Whether to include per-sample outputs in the final payload.

        Returns:
            Plain dictionary with dataset-level aggregated metrics.
        """
        per_sample_results = [SampleMetricResult(**self.evaluate_sample(sample)) for sample in samples]
        metric_accumulator: dict[str, list[float]] = {}

        for result in per_sample_results:
            flat_metrics: dict[str, Any] = {
                **result.heuristics.model_dump(),
                **result.aggregate.model_dump(),
            }
            if result.judge is not None:
                flat_metrics.update(
                    {
                        key: value
                        for key, value in result.judge.model_dump().items()
                        if isinstance(value, (int, float, bool))
                    }
                )
            for key, value in flat_metrics.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, (int, float)):
                    metric_accumulator.setdefault(key, []).append(float(value))

        dataset_metrics = {
            key: sum(values) / len(values)
            for key, values in metric_accumulator.items()
            if values
        }

        result = DatasetMetricResult(
            sample_count=len(samples),
            metrics=dataset_metrics,
            per_sample=per_sample_results if include_per_sample else [],
        )
        return result.model_dump()

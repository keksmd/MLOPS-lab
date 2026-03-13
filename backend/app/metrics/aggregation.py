from __future__ import annotations

from .config import MetricWeights
from .schemas import AggregateMetricScores, HeuristicMetricScores, JudgeMetricScores


def compute_validity_score(heuristics: HeuristicMetricScores) -> float:
    """Aggregate structural validity metrics into a single [0, 1] score."""
    values = [
        heuristics.json_valid,
        heuristics.schema_valid,
        heuristics.nonempty_plan,
        heuristics.allowed_tool_rate,
        heuristics.arg_name_valid_rate,
        heuristics.required_arg_presence_rate,
    ]
    return sum(values) / len(values)


def compute_plan_score(judge: JudgeMetricScores) -> float:
    """Aggregate plan-quality judge metrics into a single [0, 1] score."""
    values = [
        judge.plan_relevance,
        judge.plan_completeness,
        judge.plan_logic,
        judge.plan_specificity,
        judge.plan_nonredundancy,
        1.0 - judge.plan_duplicate_penalty,
    ]
    return sum(values) / len(values)


def compute_tool_score(judge: JudgeMetricScores) -> float:
    """Aggregate tool-quality judge metrics into a single [0, 1] score."""
    values = [
        judge.tool_appropriateness,
        judge.tool_sufficiency,
        judge.argument_quality,
    ]
    return sum(values) / len(values)


def compute_solvability_score(judge: JudgeMetricScores) -> float:
    """Return the normalized overall solvability score in [0, 1]."""
    return judge.overall_solvability


def compute_aggregate_metrics(
    heuristics: HeuristicMetricScores,
    judge: JudgeMetricScores | None,
    weights: MetricWeights,
) -> AggregateMetricScores:
    """Compute aggregate metrics for a single sample."""
    validity_score = compute_validity_score(heuristics)

    if judge is None:
        plan_score = 0.0
        tool_score = 0.0
        solvability_score = 0.0
        critical_failure_rate = 0.0
    else:
        plan_score = compute_plan_score(judge)
        tool_score = compute_tool_score(judge)
        solvability_score = compute_solvability_score(judge)
        critical_failure_rate = 1.0 if judge.critical_failure else 0.0

    final_score = (
        weights.validity_weight * validity_score
        + weights.plan_weight * plan_score
        + weights.tool_weight * tool_score
        + weights.solvability_weight * solvability_score
    )

    return AggregateMetricScores(
        validity_score=validity_score,
        plan_score=plan_score,
        tool_score=tool_score,
        solvability_score=solvability_score,
        final_score=final_score,
        critical_failure_rate=critical_failure_rate,
    )

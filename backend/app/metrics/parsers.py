from __future__ import annotations

from typing import Any

from .exceptions import JudgeResponseParseError
from .schemas import JudgeMetricScores


JUDGE_SCORE_FIELDS = {
    "plan_relevance",
    "plan_completeness",
    "plan_logic",
    "plan_specificity",
    "plan_nonredundancy",
    "plan_duplicate_penalty",
    "tool_appropriateness",
    "tool_sufficiency",
    "argument_quality",
    "overall_solvability",
}


def _clamp01(value: Any) -> float:
    value = float(value)
    return max(0.0, min(1.0, value))


def parse_judge_response(payload: dict[str, Any]) -> JudgeMetricScores:
    """
    Convert raw judge JSON into the strongly typed score schema.

    The parser is permissive for numeric fields: values are cast to float and clipped to [0, 1].

    Raises:
        JudgeResponseParseError: If the payload is missing required fields or contains invalid types.
    """
    try:
        normalized = dict(payload)
        for field_name in JUDGE_SCORE_FIELDS:
            normalized[field_name] = _clamp01(normalized[field_name])
        normalized["critical_failure"] = bool(normalized.get("critical_failure", False))
        return JudgeMetricScores(**normalized)
    except Exception as exc:
        raise JudgeResponseParseError(str(exc)) from exc

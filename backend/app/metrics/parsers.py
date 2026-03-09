from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .exceptions import JudgeResponseParseError
from .schemas import JudgeMetricScores


def _format_validation_error(exc: ValidationError) -> str:
    """Render a compact pydantic validation error string."""
    parts: list[str] = []

    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"])
        message = error["msg"]
        parts.append(f"{location}: {message}")

    return "; ".join(parts)


def parse_judge_response(payload: dict[str, Any]) -> JudgeMetricScores:
    """Validate a raw judge payload against the typed score schema."""
    try:
        return JudgeMetricScores.model_validate(payload)
    except ValidationError as exc:
        message = _format_validation_error(exc)
        raise JudgeResponseParseError(message) from exc
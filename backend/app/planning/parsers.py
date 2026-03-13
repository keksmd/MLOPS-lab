from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser

from app.metrics.schemas import PlannerOutput

from .exceptions import PredictionParseError
from .normalizers import normalize_planner_output


class PlannerOutputParser:
    """Validate planner outputs using LangChain and shared normalization rules."""

    def __init__(self) -> None:
        self._legacy_text_parser = PydanticOutputParser(pydantic_object=PlannerOutput)

    def parse(
        self,
        payload: PlannerOutput | Mapping[str, Any] | str,
    ) -> PlannerOutput:
        """Convert structured or legacy text output into the shared PlannerOutput."""
        if isinstance(payload, PlannerOutput):
            return normalize_planner_output(payload.model_dump())

        if isinstance(payload, Mapping):
            return normalize_planner_output(dict(payload))

        if not isinstance(payload, str) or not payload.strip():
            raise PredictionParseError("Model returned an empty response.")

        try:
            parsed_output = self._legacy_text_parser.parse(payload)
        except Exception as exc:
            raise PredictionParseError(
                "Failed to parse planner output with LangChain parser."
            ) from exc

        return normalize_planner_output(parsed_output.model_dump())

    def get_format_instructions(self) -> str:
        """Return legacy format instructions for parser-based integrations."""
        return self._legacy_text_parser.get_format_instructions()

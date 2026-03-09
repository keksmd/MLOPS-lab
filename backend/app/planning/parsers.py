from __future__ import annotations

import json
import re
from typing import Any

from app.metrics.schemas import PlannerOutput

from .exceptions import PredictionParseError
from .normalizers import normalize_planner_output


class PlannerOutputParser:
    """Parser that converts raw model text into the shared PlannerOutput schema."""

    def parse(self, raw_text: str) -> PlannerOutput:
        """
        Parse raw model text into PlannerOutput.

        The parser accepts plain JSON and fenced JSON blocks. It also attempts to
        extract the first top-level JSON object from the response.
        """
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise PredictionParseError("Model returned an empty response.")

        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            payload = json.loads(text)
            return normalize_planner_output(payload)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                return normalize_planner_output(payload)
            except Exception as exc:
                raise PredictionParseError(f"Failed to parse JSON object from model output: {text}") from exc

        raise PredictionParseError(f"Model did not return valid planner JSON: {text}")

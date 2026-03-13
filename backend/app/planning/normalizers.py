from __future__ import annotations

import ast
import json
import re
from typing import Any

from app.metrics.schemas import ActionCall, PlannerOutput

PLACEHOLDER_RULES: dict[str, dict[str, str]] = {
    "crawl_pages": {"url": "<retrieved_url>"},
    "find_archived_url": {"url": "<target_url>"},
}


def safe_to_obj(value: Any) -> Any:
    """Parse a JSON-like string into a Python object when possible."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(value)
            except Exception:
                pass
    return value


def clean_plan_text(plan_text: str) -> str:
    """Remove boilerplate and markdown fences from a plan string."""
    if not isinstance(plan_text, str):
        return ""

    text = plan_text.strip().replace("```", "").strip()
    text = re.sub(
        r"^Here is the plan of action that I will follow to solve the task:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text


def split_plan_steps(plan_text: str) -> list[str]:
    """Split plan text into an ordered list of atomic plan steps."""
    text = clean_plan_text(plan_text)
    if not text:
        return []

    parts = re.split(r"\n?\s*\d+\.\s+", text)
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) <= 1:
        parts = [
            line.strip("-• ").strip() for line in text.splitlines() if line.strip()
        ]
    return parts


def normalize_action_arguments(
    tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Normalize raw action arguments to the canonical project format."""
    if not isinstance(arguments, dict):
        return {}

    normalized: dict[str, Any] = {}
    for key, value in arguments.items():
        if key == "file_path":
            continue
        placeholder = PLACEHOLDER_RULES.get(tool_name, {}).get(key)
        if placeholder is not None:
            normalized[key] = placeholder
        else:
            normalized[key] = value
    return normalized


def simplify_actions(raw_actions: Any) -> list[ActionCall]:
    """Convert raw action traces into compact action calls used by the planner."""
    if not isinstance(raw_actions, list):
        return []

    actions: list[ActionCall] = []
    previous: ActionCall | None = None
    for raw_action in raw_actions:
        if not isinstance(raw_action, dict):
            continue
        tool_name = raw_action.get("tool_name")
        if not tool_name or tool_name == "final_answer":
            continue
        normalized = ActionCall(
            tool_name=tool_name,
            arguments=normalize_action_arguments(
                tool_name, raw_action.get("arguments", {})
            ),
        )
        if previous is None or previous.model_dump() != normalized.model_dump():
            actions.append(normalized)
            previous = normalized
    return actions


def normalize_planner_output(payload: Any) -> PlannerOutput:
    """Normalize arbitrary planner JSON into the shared PlannerOutput schema."""
    obj = safe_to_obj(payload)
    if not isinstance(obj, dict):
        return PlannerOutput(plan=[], actions=[])

    raw_plan = obj.get("plan", [])
    raw_actions = obj.get("actions", [])

    if isinstance(raw_plan, str):
        plan = split_plan_steps(raw_plan)
    elif isinstance(raw_plan, list):
        plan = [str(step).strip() for step in raw_plan if str(step).strip()]
    else:
        plan = []

    actions = simplify_actions(raw_actions)
    return PlannerOutput(plan=plan, actions=actions)

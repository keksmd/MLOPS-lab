from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from app.metrics.schemas import ToolArgumentSpec, ToolSpec

from ..normalizers import safe_to_obj, simplify_actions, split_plan_steps


TOOL_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "web_search": {
        "description": "Search the web for relevant information using a textual query.",
        "arguments": {
            "query": "search query string",
            "filter_year": "optional year filter",
        },
    },
    "crawl_pages": {
        "description": "Open a web page by URL and read its content.",
        "arguments": {
            "url": "URL placeholder of the page to inspect",
        },
    },
    "inspect_file_as_image": {
        "description": "Inspect an image-based file or figure and answer a question about its visual content.",
        "arguments": {
            "question": "question about the image or figure",
        },
    },
    "inspect_file_as_text": {
        "description": "Inspect a text-based file or document and answer a question about its contents.",
        "arguments": {
            "question": "question about the document text",
        },
    },
    "find_archived_url": {
        "description": "Find an archived version of a web page for a given URL and date.",
        "arguments": {
            "url": "placeholder of the original target URL",
            "date": "target archive date in YYYYMMDD format",
        },
    },
}


def convert_taskcraft_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw TaskCraft row into the compact training/inference format."""
    task = row.get("query", "")
    agent = safe_to_obj(row.get("ans_from_agent"))
    if not isinstance(agent, dict):
        return {"task": task, "plan": [], "actions": []}

    trace = agent.get("trace", {})
    if not isinstance(trace, dict):
        return {"task": task, "plan": [], "actions": []}

    return {
        "task": task,
        "plan": split_plan_steps(trace.get("plan", "")),
        "actions": [action.model_dump() for action in simplify_actions(trace.get("actions", []))],
    }


def build_processed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw TaskCraft dataframe into the processed planner dataset."""
    processed_records = [convert_taskcraft_row(row) for row in df.to_dict(orient="records")]
    processed_df = pd.DataFrame(processed_records)
    processed_df["n_plan_steps"] = processed_df["plan"].apply(len)
    processed_df["n_actions"] = processed_df["actions"].apply(len)
    return processed_df


def build_tool_registry_from_raw(df: pd.DataFrame) -> list[ToolSpec]:
    """Infer the set of tools from raw TaskCraft traces and attach canonical descriptions."""
    tool_freq = Counter()
    tool_arg_names = defaultdict(set)

    for row in df.to_dict(orient="records"):
        agent = safe_to_obj(row.get("ans_from_agent"))
        if not isinstance(agent, dict):
            continue
        trace = agent.get("trace", {})
        actions = trace.get("actions", []) if isinstance(trace, dict) else []
        for action in actions:
            if not isinstance(action, dict):
                continue
            tool_name = action.get("tool_name")
            if not tool_name or tool_name == "final_answer":
                continue
            tool_freq[tool_name] += 1
            arguments = action.get("arguments", {})
            if isinstance(arguments, dict):
                for arg_name in arguments.keys():
                    if arg_name != "file_path":
                        tool_arg_names[tool_name].add(arg_name)

    registry: list[ToolSpec] = []
    for tool_name, _ in tool_freq.most_common():
        meta = TOOL_DESCRIPTIONS.get(tool_name, {"description": "", "arguments": {}})
        description = meta.get("description", "")
        arg_meta = meta.get("arguments", {})
        arguments = [
            ToolArgumentSpec(name=arg_name, description=str(arg_meta.get(arg_name, "")), required=True)
            for arg_name in sorted(tool_arg_names[tool_name])
        ]
        registry.append(ToolSpec(tool_name=tool_name, description=description, arguments=arguments))
    return registry

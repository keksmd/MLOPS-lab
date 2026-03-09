from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.metrics.schemas import ActionCall, PlannerOutput, ToolArgumentSpec, ToolSpec

from ..exceptions import RepositoryLoadError
from ..normalizers import normalize_planner_output, safe_to_obj
from ..schemas import PlannerExample

OPTIONAL_ARGUMENTS: dict[str, set[str]] = {
    "web_search": {"filter_year"},
}

class JsonArtifactRepository:
    """Generic JSON/JSONL reader used for processed dataset and registry files."""

    def load_records(self, path: str | Path) -> list[dict[str, Any]]:
        """Load a list of records from a JSON array or JSONL file."""
        file_path = Path(path)
        if not file_path.exists():
            raise RepositoryLoadError(f"Artifact file does not exist: {file_path}")

        text = file_path.read_text(encoding="utf-8").strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and isinstance(obj.get("data"), list):
                return obj["data"]
            if isinstance(obj, dict) and all(isinstance(value, dict) for value in obj.values()):
                return [{"tool_name": key, **value} for key, value in obj.items()]
        except Exception:
            pass

        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


class ToolRegistryLoader:
    """Loads the tool registry JSON produced during dataset preparation."""
    
    def load(self, path: str | Path) -> list[ToolSpec]:
        records = JsonArtifactRepository().load_records(path)
        tools: list[ToolSpec] = []
        for record in records:
            arguments = []
            for arg_name, arg_description in record.get("arguments", {}).items():
                arguments.append(
                    ToolArgumentSpec(
                        name=arg_name,
                        description=str(arg_description),
                        required=arg_name not in OPTIONAL_ARGUMENTS.get(record["tool_name"], set()),
                    )
                )
            tools.append(
                ToolSpec(
                    tool_name=record["tool_name"],
                    description=record.get("description", ""),
                    arguments=arguments,
                )
            )
        return tools


class FewShotDatasetLoader:
    """Loads processed planner examples from dataset_df.json-like artifacts."""

    def load_examples(self, path: str | Path) -> list[PlannerExample]:
        records = JsonArtifactRepository().load_records(path)
        examples: list[PlannerExample] = []
        for record in records:
            task = record.get("task", record.get("query", ""))
            output = normalize_planner_output({
                "plan": record.get("plan", []),
                "actions": record.get("actions", []),
            })
            examples.append(PlannerExample(task=task, output=output))
        return examples

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from app.metrics.llm.base import BaseLLMClient
from app.metrics.schemas import ActionCall, PlannerOutput, ToolArgumentSpec, ToolSpec
from app.planning.schemas import PlannerExample

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@pytest.fixture
def tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            tool_name="web_search",
            description="Search the web for relevant information using a textual query.",
            arguments=[
                ToolArgumentSpec(
                    name="query",
                    description="search query string",
                    required=True,
                ),
                ToolArgumentSpec(
                    name="filter_year",
                    description="optional year filter",
                    required=False,
                ),
            ],
        ),
        ToolSpec(
            tool_name="crawl_pages",
            description="Open a web page by URL and read its content.",
            arguments=[
                ToolArgumentSpec(
                    name="url",
                    description="URL placeholder",
                    required=True,
                )
            ],
        ),
        ToolSpec(
            tool_name="find_archived_url",
            description="Find an archived version of a page.",
            arguments=[
                ToolArgumentSpec(
                    name="url",
                    description="target URL",
                    required=True,
                ),
                ToolArgumentSpec(
                    name="date",
                    description="YYYYMMDD",
                    required=True,
                ),
            ],
        ),
    ]


@pytest.fixture
def gold_output() -> PlannerOutput:
    return PlannerOutput(
        plan=[
            "Search for the official article page.",
            "Open the article page and inspect the metadata.",
            "If needed, inspect archived material to confirm the date.",
        ],
        actions=[
            ActionCall(
                tool_name="web_search",
                arguments={"query": "article received date"},
            ),
            ActionCall(
                tool_name="crawl_pages",
                arguments={"url": "<retrieved_url>"},
            ),
            ActionCall(
                tool_name="find_archived_url",
                arguments={"url": "<retrieved_url>", "date": "20240101"},
            ),
        ],
    )


@pytest.fixture
def predicted_output() -> PlannerOutput:
    return PlannerOutput(
        plan=[
            "Search for the official article page.",
            "Open the article page and inspect the metadata.",
            "Open the article page and inspect the metadata.",
        ],
        actions=[
            ActionCall(
                tool_name="web_search",
                arguments={"query": "article received date"},
            ),
            ActionCall(
                tool_name="crawl_pages",
                arguments={"url": "<retrieved_url>"},
            ),
            ActionCall(
                tool_name="crawl_pages",
                arguments={"url": "<retrieved_url>"},
            ),
        ],
    )


@pytest.fixture
def planner_examples(gold_output: PlannerOutput) -> list[PlannerExample]:
    return [
        PlannerExample(task="Task 1", output=gold_output),
        PlannerExample(task="Task 2", output=gold_output),
        PlannerExample(task="Task 3", output=gold_output),
        PlannerExample(task="Task 4", output=gold_output),
    ]


class DummyJSONLLMClient(BaseLLMClient):
    """Simple deterministic client for unit tests."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(self.payload, ensure_ascii=False)

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return self.payload

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        return schema.model_validate(self.payload)


@pytest.fixture
def judge_payload() -> dict[str, Any]:
    return {
        "plan_relevance": 0.9,
        "plan_completeness": 0.8,
        "plan_logic": 0.85,
        "plan_specificity": 0.75,
        "plan_nonredundancy": 0.5,
        "plan_duplicate_penalty": 0.25,
        "tool_appropriateness": 0.9,
        "tool_sufficiency": 0.8,
        "argument_quality": 0.9,
        "overall_solvability": 0.8,
        "critical_failure": False,
        "critical_missing_steps": [],
        "underspecified_steps": [],
        "unnecessary_actions": [],
        "bad_arguments": [],
        "duplicate_step_notes": ["Two steps are very similar."],
        "reasoning": "Reasonable output with a small amount of redundancy.",
    }


@pytest.fixture
def tmp_json_artifacts(
    tmp_path: Path,
    gold_output: PlannerOutput,
) -> dict[str, Path]:
    dataset_path = tmp_path / "dataset_df.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "task": "What is the received date?",
                    "plan": gold_output.plan,
                    "actions": [action.model_dump() for action in gold_output.actions],
                },
                {
                    "task": "Who is the author?",
                    "plan": gold_output.plan,
                    "actions": [action.model_dump() for action in gold_output.actions],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tools_path = tmp_path / "tool_descriptions.json"
    tools_path.write_text(
        json.dumps(
            {
                "web_search": {
                    "description": (
                        "Search the web for relevant information using a textual query."
                    ),
                    "arguments": {
                        "query": "search query string",
                        "filter_year": "optional year filter",
                    },
                },
                "crawl_pages": {
                    "description": "Open a web page by URL and read its content.",
                    "arguments": {"url": "URL of the page to inspect"},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "dataset": dataset_path,
        "tools": tools_path,
    }

_REQUIRED_TEST_ENV = {
    "PROJECT_NAME": "test-project",
    "POSTGRES_SERVER": "localhost",
    "POSTGRES_USER": "test-user",
    "POSTGRES_PASSWORD": "test-password",
    "POSTGRES_DB": "test-db",
    "FIRST_SUPERUSER": "admin@example.com",
    "FIRST_SUPERUSER_PASSWORD": "test-password",
    "OPENROUTER_API_KEY": "test-openrouter-key",
    "OPENROUTER_MODEL_NAME": "test-model",
}


for key, value in _REQUIRED_TEST_ENV.items():
    os.environ.setdefault(key, value)
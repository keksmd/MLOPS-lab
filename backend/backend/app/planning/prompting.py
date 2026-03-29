from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from app.metrics.schemas import PlannerOutput

from .exceptions import PromptBuildError
from .schemas import InferenceRequest, PromptArtifacts

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _load_template(file_name: str) -> str:
    """Load a prompt template from the local templates directory."""
    template_path = _TEMPLATE_DIR / file_name
    return template_path.read_text(encoding="utf-8")


class PlanningPromptBuilder:
    """Build planning prompts using external templates and LangChain."""

    def __init__(self) -> None:
        self._output_parser = PydanticOutputParser(
            pydantic_object=PlannerOutput,
        )
        system_prompt = PromptTemplate.from_template(
            _load_template("planning_system_prompt.j2"),
            template_format="jinja2",
        )
        user_prompt = PromptTemplate.from_template(
            _load_template("planning_user_prompt.j2"),
            template_format="jinja2",
        )
        self._system_prompt_template = ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate(prompt=system_prompt)]
        )
        self._user_prompt_template = ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate(prompt=user_prompt)]
        )

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        """Convert a LangChain message into plain text."""
        if isinstance(message.content, str):
            return message.content
        return str(message.content)

    def get_format_instructions(self) -> str:
        """Return LangChain-generated format instructions for planner output."""
        return self._output_parser.get_format_instructions()

    @staticmethod
    def _serialize_few_shot_examples(
        request: InferenceRequest,
    ) -> list[dict[str, object]]:
        """Prepare few-shot examples for template rendering."""
        return [
            {
                "example_index": index,
                "task": example.task,
                "output_json": json.dumps(
                    example.output.model_dump(),
                    ensure_ascii=False,
                    indent=2,
                ),
            }
            for index, example in enumerate(request.few_shot_examples, start=1)
        ]

    def build(self, request: InferenceRequest) -> PromptArtifacts:
        """Build the planning system and user prompts for a single inference call."""
        if not request.task.strip():
            raise PromptBuildError("Task must be non-empty.")

        context = {
            "available_tools": [tool.model_dump() for tool in request.available_tools],
            "few_shot_examples": self._serialize_few_shot_examples(request),
            "format_instructions": self.get_format_instructions(),
            "task": request.task,
        }

        system_prompt_value = self._system_prompt_template.invoke(context)
        user_prompt_value = self._user_prompt_template.invoke(context)

        system_prompt = self._message_text(system_prompt_value.to_messages()[0])
        user_prompt = self._message_text(user_prompt_value.to_messages()[0])

        return PromptArtifacts(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

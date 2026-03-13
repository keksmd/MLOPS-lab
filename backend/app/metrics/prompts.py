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

from .config import JudgeConfig
from .schemas import EvaluationSample, JudgeMetricScores

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _load_template(file_name: str) -> str:
    """Load a prompt template from the local templates directory."""
    template_path = _TEMPLATE_DIR / file_name
    return template_path.read_text(encoding="utf-8")


class JudgePromptBuilder:
    """Build judge prompts using external Jinja templates and LangChain parsers."""

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = (
            config
            if config is not None
            else JudgeConfig(
                use_reference_aware_judge=True,
                include_reasoning=True,
            )
        )
        self._output_parser = PydanticOutputParser(pydantic_object=JudgeMetricScores)

        system_prompt = PromptTemplate.from_template(
            _load_template("judge_system_prompt.j2"),
            template_format="jinja2",
        )
        user_prompt = PromptTemplate.from_template(
            _load_template("judge_user_prompt.j2"),
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
        """Return LangChain-generated format instructions for judge output."""
        return self._output_parser.get_format_instructions()

    def _build_payload(self, sample: EvaluationSample) -> dict[str, object]:
        """Build the structured judge payload for a single sample."""
        payload: dict[str, object] = {
            "task": sample.task,
            "available_tools": [tool.model_dump() for tool in sample.available_tools],
            "predicted_output": sample.prediction.model_dump(),
        }

        if self.config.use_reference_aware_judge and sample.golden is not None:
            payload["golden_reference"] = sample.golden.model_dump()

        return payload

    def _build_system_context(self) -> dict[str, object]:
        """Build template variables for the system prompt."""
        reasoning_instruction = (
            "Provide a short reasoning string that mentions the most important "
            "strengths and weaknesses of the prediction."
            if self.config.include_reasoning
            else "Set reasoning to an empty string."
        )

        return {
            "use_reference_aware_judge": self.config.use_reference_aware_judge,
            "reasoning_instruction": reasoning_instruction,
            "format_instructions": self.get_format_instructions(),
        }

    def build_system_prompt(self) -> str:
        """Render the system prompt."""
        prompt_value = self._system_prompt_template.invoke(self._build_system_context())
        return self._message_text(prompt_value.to_messages()[0])

    def build_user_prompt(self, sample: EvaluationSample) -> str:
        """Render the user prompt for one evaluation sample."""
        prompt_value = self._user_prompt_template.invoke(
            {
                "payload_json": json.dumps(
                    self._build_payload(sample),
                    ensure_ascii=False,
                    indent=2,
                )
            }
        )
        return self._message_text(prompt_value.to_messages()[0])

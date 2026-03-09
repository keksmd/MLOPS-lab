from __future__ import annotations

import json

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from app.metrics.schemas import PlannerOutput, ToolSpec

from .exceptions import PromptBuildError
from .schemas import InferenceRequest, PromptArtifacts


class PlanningPromptBuilder:
    """Build planning prompts using LangChain prompt templates."""

    _SYSTEM_TEMPLATE = """\
You are an AI planning assistant.

Your task is to convert a user task into:
1. a step-by-step plan
2. a compact list of tool calls that may be needed

Important constraints:
- Do not answer the user question itself.
- Do not include observations, retrieved content, citations, or final answers.
- Use only the tools listed below.
- Use only tool_name and arguments.
- Return valid JSON only.
- Output schema must be exactly:
{{
  "plan": ["step 1", "step 2", "..."],
  "actions": [
    {{
      "tool_name": "name_from_available_tools",
      "arguments": {{}}
    }}
  ]
}}

Rules for planning:
- The plan must be written in English.
- The plan should be explicit, sequential, and operational.
- Use as many steps as needed for the task.
- All steps must be atomic.
- The plan may include verification if it is logically needed.
- The actions list must be compact and realistic.
- Exclude any "final_answer" tool.
- For crawl_pages, use {{"tool_name": "crawl_pages", "arguments": {{"url": ""}}}}.
- For find_archived_url, use {{"tool_name": "find_archived_url", "arguments": {{"url": "", "date": "YYYYMMDD"}}}}.
- Keep web_search queries concise but specific.

Available tools:
{tools_text}
"""

    _USER_TEMPLATE = """\
You will be given a task and must return only JSON with keys "plan" and "actions".

{few_shot_section}
Task:
{task}

Return only valid JSON with keys "plan" and "actions".
"""

    def __init__(self) -> None:
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._SYSTEM_TEMPLATE),
                ("human", self._USER_TEMPLATE),
            ]
        )

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        """Convert a LangChain message into plain text."""
        if isinstance(message.content, str):
            return message.content
        return str(message.content)

    def _render_tools(self, tools: list[ToolSpec]) -> str:
        """Render the available tool registry as readable text."""
        if not tools:
            return "No tools available."

        blocks: list[str] = []

        for tool in tools:
            arg_lines: list[str] = []
            for argument in tool.arguments:
                required_label = "required" if argument.required else "optional"
                arg_lines.append(
                    f' - "{argument.name}" ({required_label}): {argument.description}'
                )

            args_text = "\n".join(arg_lines) if arg_lines else " - no arguments"
            block = (
                f"Tool: {tool.tool_name}\n"
                f"Description: {tool.description}\n"
                f"Arguments:\n{args_text}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    def _render_example_output(self, output: PlannerOutput) -> str:
        """Render one planner output example as JSON."""
        return json.dumps(output.model_dump(), ensure_ascii=False, indent=2)

    def _render_few_shot_section(self, request: InferenceRequest) -> str:
        """Render few-shot examples for the human prompt."""
        if not request.few_shot_examples:
            return ""

        parts: list[str] = ["Here are few-shot examples:", ""]

        for index, example in enumerate(request.few_shot_examples, start=1):
            parts.append(f"Example {index} — Task:")
            parts.append(example.task)
            parts.append("")
            parts.append(f"Example {index} — Output:")
            parts.append(self._render_example_output(example.output))
            parts.append("")

        return "\n".join(parts).strip()

    def build(self, request: InferenceRequest) -> PromptArtifacts:
        """Build the planning system and user prompts for a single inference call."""
        if not request.task.strip():
            raise PromptBuildError("Task must be non-empty.")

        prompt_value = self._prompt_template.invoke(
            {
                "tools_text": self._render_tools(request.available_tools),
                "few_shot_section": self._render_few_shot_section(request),
                "task": request.task,
            }
        )

        system_prompt = self._message_text(prompt_value.messages[0])
        user_prompt = self._message_text(prompt_value.messages[1])

        return PromptArtifacts(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
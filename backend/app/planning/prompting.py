from __future__ import annotations

import json

from app.metrics.schemas import PlannerOutput, ToolSpec

from .exceptions import PromptBuildError
from .schemas import InferenceRequest, PromptArtifacts


class PlanningPromptBuilder:
    """Builds the planning prompt used for baseline few-shot inference."""

    def _render_tools(self, tools: list[ToolSpec]) -> str:
        if not tools:
            return "No tools available."

        blocks: list[str] = []
        for tool in tools:
            arg_lines: list[str] = []
            for argument in tool.arguments:
                required = "required" if argument.required else "optional"
                arg_lines.append(f'  - "{argument.name}" ({required}): {argument.description}')
            args_text = "\n".join(arg_lines) if arg_lines else "  - no arguments"
            block = f"Tool: {tool.tool_name}\nDescription: {tool.description}\nArguments:\n{args_text}"
            blocks.append(block)
        return "\n\n".join(blocks)

    def _render_example_output(self, output: PlannerOutput) -> str:
        return json.dumps(output.model_dump(), ensure_ascii=False, indent=2)

    def build(self, request: InferenceRequest) -> PromptArtifacts:
        """
        Build the system and user prompts for one inference call.

        Few-shot demonstrations are embedded into the user prompt to keep the LLM
        client interface compatible with the metrics module.
        """
        if not request.task.strip():
            raise PromptBuildError("Task must be non-empty.")

        tools_text = self._render_tools(request.available_tools)
        system_prompt = f"""
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
- For crawl_pages, use {{"tool_name": "crawl_pages", "arguments": {{"url": "<retrieved_url>"}}}}.
- For find_archived_url, use {{"tool_name": "find_archived_url", "arguments": {{"url": "<target_url>", "date": "YYYYMMDD"}}}}.
- Keep web_search queries concise but specific.

Available tools:
{tools_text}
""".strip()

        parts: list[str] = [
            "You will be given a task and must return only JSON with keys 'plan' and 'actions'.",
            "",
        ]

        if request.few_shot_examples:
            parts.append("Here are few-shot examples:")
            parts.append("")
            for index, example in enumerate(request.few_shot_examples, start=1):
                parts.append(f"Example {index} — Task:")
                parts.append(example.task)
                parts.append("")
                parts.append(f"Example {index} — Output:")
                parts.append(self._render_example_output(example.output))
                parts.append("")

        parts.append("Now solve the following task.")
        parts.append("Task:")
        parts.append(request.task)
        parts.append("")
        parts.append("Return only valid JSON with keys 'plan' and 'actions'.")

        user_prompt = "\n".join(parts).strip()
        return PromptArtifacts(system_prompt=system_prompt, user_prompt=user_prompt)

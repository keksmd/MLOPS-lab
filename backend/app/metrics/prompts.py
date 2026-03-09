from __future__ import annotations

import json

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from .config import JudgeConfig
from .schemas import EvaluationSample


class JudgePromptBuilder:
    """Build judge prompts using LangChain prompt templates."""

    _SYSTEM_TEMPLATE = """\
You are a strict evaluator of task decomposition for tool-using AI agents.

Evaluate only the quality of the predicted plan and actions.
All numeric scores must be normalized to the [0, 1] range, where 0 means very poor and 1 means excellent.
Return only valid JSON.
When a golden reference is provided, treat it as the ideal answer that would receive the maximum possible rating on every criterion.
Treat the golden reference as an ideal ceiling, not as a required wording or exact sequence of steps.
Do not require lexical similarity to the golden reference.
A prediction may be correct even if phrased differently.
Do not mark a step as missing if the predicted plan already contains a semantically equivalent step.
Use critical_missing_steps only for truly absent capabilities or operations.
If a step exists but is less specific or less detailed than the golden reference, lower plan_specificity or plan_completeness instead of listing it in critical_missing_steps.
If a step is present but under-specified, list it in underspecified_steps instead of critical_missing_steps.
Do not penalize argument_quality heavily when a predicted search query is shorter than the golden reference but still likely sufficient to retrieve the correct source.
Prefer moderate penalties over harsh penalties for valid alternative search formulations.
Do not output markdown or explanations outside JSON.
"""

    _USER_TEMPLATE = """\
Evaluate the predicted output.

Consider whether the plan is relevant, complete, logically ordered, specific, and non-redundant.
Also assess whether the chosen tools and arguments are appropriate and sufficient, and whether the output would allow the task to be solved in practice.
Use normalized scores in [0, 1].
Set critical_failure=true only when the prediction has a major flaw that makes successful execution unlikely.
Only include an item in critical_missing_steps if that capability or operation is truly absent from the predicted plan.
If the plan includes an equivalent step but in a shorter or less detailed form, reflect that through lower specificity or completeness scores instead of marking it as missing.

Payload:
{payload}
"""

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()
        self._system_prompt_template = ChatPromptTemplate.from_messages(
            [("system", self._SYSTEM_TEMPLATE)]
        )
        self._user_prompt_template = ChatPromptTemplate.from_messages(
            [("human", self._USER_TEMPLATE)]
        )

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        """Convert a LangChain message into plain text."""
        if isinstance(message.content, str):
            return message.content
        return str(message.content)

    def _build_payload(self, sample: EvaluationSample) -> dict[str, object]:
        """Build the structured judge payload for a single sample."""
        tool_payload = [tool.model_dump() for tool in sample.available_tools]

        required_output_schema: dict[str, object] = {
            "plan_relevance": 0.0,
            "plan_completeness": 0.0,
            "plan_logic": 0.0,
            "plan_specificity": 0.0,
            "plan_nonredundancy": 0.0,
            "plan_duplicate_penalty": 0.0,
            "tool_appropriateness": 0.0,
            "tool_sufficiency": 0.0,
            "argument_quality": 0.0,
            "overall_solvability": 0.0,
            "critical_failure": False,
            "critical_missing_steps": [],
            "underspecified_steps": [],
            "unnecessary_actions": [],
            "bad_arguments": [],
            "duplicate_step_notes": [],
            "reasoning": "",
        }

        if not self.config.include_reasoning:
            required_output_schema.pop("reasoning", None)

        return {
            "task": sample.task,
            "available_tools": tool_payload,
            "predicted_output": sample.prediction.model_dump(),
            "golden_reference": (
                sample.golden.model_dump()
                if self.config.use_reference_aware_judge and sample.golden is not None
                else None
            ),
            "required_output_schema": required_output_schema,
        }

    def build_system_prompt(self) -> str:
        """Render the system prompt."""
        prompt_value = self._system_prompt_template.invoke({})
        return self._message_text(prompt_value.messages[0])

    def build_user_prompt(self, sample: EvaluationSample) -> str:
        """Render the user prompt for one evaluation sample."""
        payload = self._build_payload(sample)
        prompt_value = self._user_prompt_template.invoke(
            {
                "payload": json.dumps(payload, ensure_ascii=False, indent=2),
            }
        )
        return self._message_text(prompt_value.messages[0])
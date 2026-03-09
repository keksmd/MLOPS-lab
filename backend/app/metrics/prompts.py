from __future__ import annotations

import json

from .config import JudgeConfig
from .schemas import EvaluationSample


class JudgePromptBuilder:
    """Build prompts for LLM-as-a-judge evaluation."""

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()

    def build_system_prompt(self) -> str:
        """
        Build the static system prompt for judge evaluation.

        Returns:
            Strict evaluator instruction with scoring rubric and JSON schema.
        """
        return (
            "You are a strict evaluator of task decomposition for tool-using AI agents. "
            "Evaluate only the quality of the predicted plan and actions. "
            "All numeric scores must be normalized to the [0, 1] range, where 0 means very poor and 1 means excellent. "
            "Return only valid JSON. "
            "When a golden reference is provided, treat it as the ideal answer that would receive the maximum possible rating on every criterion. "
            "Do not output markdown or explanations outside JSON."
        )

    def build_user_prompt(self, sample: EvaluationSample) -> str:
        """
        Build the user prompt for a single judge call.

        Notes:
            When a gold reference is present, it must be treated as an ideal reference
            that would receive maximal ratings on all judge criteria.
        """
        tool_payload = [tool.model_dump() for tool in sample.available_tools]
        payload = {
            "task": sample.task,
            "available_tools": tool_payload,
            "predicted_output": sample.prediction.model_dump(),
            "golden_reference": sample.golden.model_dump() if (self.config.use_reference_aware_judge and sample.golden) else None,
            "required_output_schema": {
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
                "unnecessary_actions": [],
                "bad_arguments": [],
                "duplicate_step_notes": [],
                "reasoning": "",
            },
        }
        instructions = (
            "Evaluate the predicted output. Consider whether the plan is relevant, complete, logically ordered, specific, and non-redundant. "
            "Also assess whether the chosen tools and arguments are appropriate and sufficient, and whether the output would allow the task to be solved in practice. "
            "Use normalized scores in [0, 1]. Set critical_failure=true only when the prediction has a major flaw that makes successful execution unlikely."
        )
        if not self.config.include_reasoning:
            payload["required_output_schema"].pop("reasoning", None)
        return instructions + "\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)

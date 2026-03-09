from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Protocol

from .schemas import EvaluationSample, HeuristicMetricScores, ToolArgumentSpec, ToolSpec


class SemanticSimilarityFn(Protocol):
    """Callable interface for optional embedding-based semantic similarity."""

    def __call__(self, left: str, right: str) -> float:
        """Return a similarity score normalized to [0, 1]."""
        ...


PLACEHOLDER_RULES: dict[str, dict[str, str]] = {
    "crawl_pages": {"url": "<retrieved_url>"},
    "find_archived_url": {"url": "<target_url>"},
}


def _tool_map(sample: EvaluationSample) -> dict[str, ToolSpec]:
    return {tool.tool_name: tool for tool in sample.available_tools}


def compute_json_valid(sample: EvaluationSample) -> float:
    """Return 1.0 if raw_prediction is valid JSON or a normalized prediction object is present."""
    if sample.raw_prediction is None:
        return 1.0
    try:
        obj = json.loads(sample.raw_prediction)
        return 1.0 if isinstance(obj, dict) else 0.0
    except Exception:
        return 0.0


def compute_schema_valid(sample: EvaluationSample) -> float:
    """Return 1.0 if prediction follows the expected PlannerOutput schema, else 0.0."""
    if not isinstance(sample.prediction.plan, list):
        return 0.0
    if not all(isinstance(step, str) for step in sample.prediction.plan):
        return 0.0
    if not isinstance(sample.prediction.actions, list):
        return 0.0
    for action in sample.prediction.actions:
        if not isinstance(action.tool_name, str):
            return 0.0
        if not isinstance(action.arguments, dict):
            return 0.0
    return 1.0


def compute_allowed_tool_rate(sample: EvaluationSample) -> float:
    """Compute the share of predicted actions that use tools from the available registry."""
    allowed = {tool.tool_name for tool in sample.available_tools}
    actions = sample.prediction.actions
    if not actions:
        return 1.0
    ok = sum(1 for action in actions if action.tool_name in allowed)
    return ok / len(actions)


def compute_forbidden_tool_rate(sample: EvaluationSample) -> float:
    """Compute the share of actions that use explicitly forbidden tools."""
    forbidden = {"final_answer"}
    actions = sample.prediction.actions
    if not actions:
        return 0.0
    bad = sum(1 for action in actions if action.tool_name in forbidden)
    return bad / len(actions)


def compute_arg_name_valid_rate(sample: EvaluationSample) -> float:
    """Compute the share of actions whose argument names match the tool schema."""
    tools = _tool_map(sample)
    actions = sample.prediction.actions
    if not actions:
        return 1.0

    valid_actions = 0
    for action in actions:
        spec = tools.get(action.tool_name)
        if spec is None:
            continue
        allowed_arg_names = {arg.name for arg in spec.arguments}
        if set(action.arguments.keys()).issubset(allowed_arg_names):
            valid_actions += 1
    return valid_actions / len(actions)


def compute_required_arg_presence_rate(sample: EvaluationSample) -> float:
    """Compute the share of actions that include all required arguments for the tool."""
    tools = _tool_map(sample)
    actions = sample.prediction.actions
    if not actions:
        return 1.0

    valid_actions = 0
    for action in actions:
        spec = tools.get(action.tool_name)
        if spec is None:
            continue
        required_arg_names = {arg.name for arg in spec.arguments if arg.required}
        if required_arg_names.issubset(set(action.arguments.keys())):
            valid_actions += 1
    return valid_actions / len(actions)


def compute_duplicate_action_rate(sample: EvaluationSample) -> float:
    """Compute the share of consecutive duplicated actions."""
    actions = sample.prediction.actions
    if len(actions) < 2:
        return 0.0
    dup = 0
    for prev, cur in zip(actions[:-1], actions[1:]):
        if prev.tool_name == cur.tool_name and prev.arguments == cur.arguments:
            dup += 1
    return dup / (len(actions) - 1)


def _normalize_step_text(step: str) -> str:
    text = step.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _step_similarity(left: str, right: str) -> float:
    left_n = _normalize_step_text(left)
    right_n = _normalize_step_text(right)
    if not left_n or not right_n:
        return 0.0
    seq_ratio = SequenceMatcher(None, left_n, right_n).ratio()
    left_tokens, right_tokens = set(left_n.split()), set(right_n.split())
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return max(seq_ratio, jaccard)


def compute_duplicate_step_rate(sample: EvaluationSample, similarity_threshold: float = 0.85) -> float:
    """
    Estimate near-duplicate steps using a lexical similarity heuristic.

    The heuristic compares each pair of steps using the maximum of:
    - SequenceMatcher ratio over normalized text
    - token-level Jaccard overlap

    A pair is counted as duplicated when similarity is above similarity_threshold.
    """
    steps = sample.prediction.plan
    if len(steps) < 2:
        return 0.0

    duplicate_pairs = 0
    total_pairs = 0
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            total_pairs += 1
            if _step_similarity(steps[i], steps[j]) >= similarity_threshold:
                duplicate_pairs += 1

    return duplicate_pairs / total_pairs if total_pairs else 0.0


def compute_tool_set_f1(sample: EvaluationSample) -> tuple[float | None, float | None, float | None]:
    """Compute precision, recall, and F1 of predicted tool names against gold tool names."""
    if sample.golden is None:
        return None, None, None

    pred = {action.tool_name for action in sample.prediction.actions}
    gold = {action.tool_name for action in sample.golden.actions}

    if not pred and not gold:
        return 1.0, 1.0, 1.0

    precision = len(pred & gold) / len(pred) if pred else 0.0
    recall = len(pred & gold) / len(gold) if gold else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_web_search_query_nonempty_rate(sample: EvaluationSample) -> float | None:
    """Compute the share of web_search actions with a non-empty query argument."""
    relevant = [a for a in sample.prediction.actions if a.tool_name == "web_search"]
    if not relevant:
        return None
    ok = sum(1 for action in relevant if str(action.arguments.get("query", "")).strip())
    return ok / len(relevant)


def compute_find_archived_url_date_format_rate(sample: EvaluationSample) -> float | None:
    """Compute the share of find_archived_url actions with date in YYYYMMDD format."""
    relevant = [a for a in sample.prediction.actions if a.tool_name == "find_archived_url"]
    if not relevant:
        return None
    pattern = re.compile(r"^\d{8}$")
    ok = sum(1 for action in relevant if pattern.match(str(action.arguments.get("date", ""))))
    return ok / len(relevant)


def compute_placeholder_compliance_rate(sample: EvaluationSample) -> float | None:
    """Compute compliance with placeholder conventions for placeholder-based tools."""
    checked = 0
    ok = 0
    for action in sample.prediction.actions:
        expected = PLACEHOLDER_RULES.get(action.tool_name)
        if expected is None:
            continue
        checked += 1
        if all(action.arguments.get(arg_name) == arg_value for arg_name, arg_value in expected.items()):
            ok += 1
    if checked == 0:
        return None
    return ok / checked


def compute_plan_semantic_similarity(
    sample: EvaluationSample,
    similarity_fn: SemanticSimilarityFn | None,
) -> float | None:
    """Compute semantic similarity between predicted and gold plans if a scorer is provided."""
    if sample.golden is None or similarity_fn is None:
        return None
    pred_text = "\n".join(sample.prediction.plan).strip()
    gold_text = "\n".join(sample.golden.plan).strip()
    if not pred_text and not gold_text:
        return 1.0
    if not pred_text or not gold_text:
        return 0.0
    score = float(similarity_fn(pred_text, gold_text))
    return max(0.0, min(1.0, score))


def compute_heuristic_metrics(
    sample: EvaluationSample,
    *,
    similarity_fn: SemanticSimilarityFn | None = None,
) -> HeuristicMetricScores:
    """Compute the full set of rule-based metrics for a single sample."""
    precision, recall, f1 = compute_tool_set_f1(sample)

    return HeuristicMetricScores(
        json_valid=compute_json_valid(sample),
        schema_valid=compute_schema_valid(sample),
        nonempty_plan=1.0 if sample.prediction.plan else 0.0,
        allowed_tool_rate=compute_allowed_tool_rate(sample),
        forbidden_tool_rate=compute_forbidden_tool_rate(sample),
        arg_name_valid_rate=compute_arg_name_valid_rate(sample),
        required_arg_presence_rate=compute_required_arg_presence_rate(sample),
        duplicate_action_rate=compute_duplicate_action_rate(sample),
        duplicate_step_rate=compute_duplicate_step_rate(sample),
        avg_plan_steps=float(len(sample.prediction.plan)),
        avg_actions=float(len(sample.prediction.actions)),
        tool_set_precision=precision,
        tool_set_recall=recall,
        tool_set_f1=f1,
        action_count_diff=(
            abs(len(sample.prediction.actions) - len(sample.golden.actions))
            if sample.golden is not None
            else None
        ),
        web_search_query_nonempty_rate=compute_web_search_query_nonempty_rate(sample),
        find_archived_url_date_format_rate=compute_find_archived_url_date_format_rate(sample),
        placeholder_compliance_rate=compute_placeholder_compliance_rate(sample),
        plan_semantic_similarity=compute_plan_semantic_similarity(sample, similarity_fn),
    )

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.metrics.schemas import EvaluationSample
    from app.planning.service import PlanningService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent

DEFAULT_OPENROUTER_MODEL = "arcee-ai/trinity-large-preview:free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _ensure_backend_on_path() -> None:
    """Ensure backend directory is importable as a top-level package root."""
    backend_dir_str = str(BACKEND_DIR)
    if backend_dir_str not in sys.path:
        sys.path.insert(0, backend_dir_str)


def _read_repo_env(repo_root: Path) -> dict[str, str]:
    """Read simple KEY=VALUE pairs from repo-root .env if it exists."""
    env_path = repo_root / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _resolve_openrouter_settings(repo_root: Path) -> tuple[str, str, str]:
    """Resolve OpenRouter API key, model name, and base URL."""
    file_env = _read_repo_env(repo_root)

    api_key = os.getenv("OPENROUTER_API_KEY") or file_env.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Set it in the shell or in repo-root .env."
        )

    model_name = (
        os.getenv("OPENROUTER_MODEL_NAME")
        or file_env.get("OPENROUTER_MODEL_NAME")
        or DEFAULT_OPENROUTER_MODEL
    )
    base_url = (
        os.getenv("OPENROUTER_BASE_URL")
        or file_env.get("OPENROUTER_BASE_URL")
        or DEFAULT_OPENROUTER_BASE_URL
    )
    return api_key, model_name, base_url


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one random-sample local smoke test: planning inference via "
            "OpenRouter + metrics evaluation."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to processed dataset artifact (for example dataset_df.json).",
    )
    parser.add_argument(
        "--tools",
        type=Path,
        required=True,
        help="Path to tool registry artifact (for example tool_descriptions.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to pick the target sample.",
    )
    parser.add_argument(
        "--few-shot-count",
        type=int,
        default=1,
        help="How many few-shot examples to include, excluding the target sample.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Include planning prompts in the final report.",
    )
    parser.add_argument(
        "--disable-judge-metrics",
        action="store_true",
        help="Disable LLM judge metrics and compute only heuristic metrics.",
    )
    return parser


def _select_target_and_few_shot(
    *,
    examples: list[Any],
    seed: int,
    few_shot_count: int,
) -> tuple[int, Any, list[Any]]:
    """Pick one random target example and a few-shot set without leakage."""
    if not examples:
        raise RuntimeError("Dataset is empty.")
    if few_shot_count < 0:
        raise RuntimeError("--few-shot-count must be >= 0.")

    _ensure_backend_on_path()
    from app.planning.few_shot import FewShotSelector

    rng = random.Random(seed)
    target_index = rng.randrange(len(examples))
    target_example = examples[target_index]

    pool = [example for idx, example in enumerate(examples) if idx != target_index]
    rng.shuffle(pool)

    if not pool or few_shot_count == 0:
        return target_index, target_example, []

    selector = FewShotSelector()
    k = min(few_shot_count, len(pool))
    few_shot_examples = selector.select_by_indices(
        pool,
        list(range(k)),
        fallback_count=k,
    )
    return target_index, target_example, few_shot_examples


def _build_evaluation_sample(
    *,
    target_index: int,
    target_example: Any,
    tools: list[Any],
    inference_result: Any,
) -> EvaluationSample:
    """Build EvaluationSample lazily to avoid module-level project imports."""
    _ensure_backend_on_path()
    from app.metrics.schemas import EvaluationSample

    return EvaluationSample(
        sample_id=f"random-sample-{target_index}",
        task=target_example.task,
        available_tools=tools,
        prediction=inference_result.prediction,
        golden=target_example.output,
        raw_prediction=(
            inference_result.raw_response
            or inference_result.prediction.model_dump_json(indent=2)
        ),
    )


def _render_report(
    *,
    target_index: int,
    target_example: Any,
    few_shot_examples: list[Any],
    model_name: str,
    show_prompts: bool,
    inference_result: Any,
    metric_result: dict[str, Any],
) -> str:
    """Render a readable multi-section report for local inspection."""
    lines: list[str] = []

    lines.append("=" * 100)
    lines.append("TARGET SAMPLE")
    lines.append(f"index: {target_index}")
    lines.append(f"task: {target_example.task}")

    lines.append("=" * 100)
    lines.append("FEW-SHOT")
    lines.append(f"count: {len(few_shot_examples)}")
    for idx, example in enumerate(few_shot_examples, start=1):
        lines.append(f"  {idx}. {example.task}")

    lines.append("=" * 100)
    lines.append("MODEL")
    lines.append(model_name)

    if show_prompts and inference_result.prompt_artifacts is not None:
        lines.append("=" * 100)
        lines.append("PLANNING SYSTEM PROMPT")
        lines.append(inference_result.prompt_artifacts.system_prompt)
        lines.append("=" * 100)
        lines.append("PLANNING USER PROMPT")
        lines.append(inference_result.prompt_artifacts.user_prompt)

    lines.append("=" * 100)
    lines.append("PREDICTION")
    lines.append(
        json.dumps(
            inference_result.prediction.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )

    lines.append("=" * 100)
    lines.append("GOLD")
    lines.append(
        json.dumps(
            target_example.output.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )

    lines.append("=" * 100)
    lines.append("METRICS")
    lines.append(json.dumps(metric_result, ensure_ascii=False, indent=2))

    return "\n".join(lines) + "\n"


def _build_planning_service(
    *,
    api_key: str,
    model_name: str,
    base_url: str,
    show_prompts: bool,
) -> PlanningService:
    """Create project planning service with OpenRouter-backed client."""
    _ensure_backend_on_path()
    from app.metrics.config import OpenRouterConfig
    from app.metrics.llm.openrouter_client import OpenRouterLLMClient
    from app.planning.config import PlanningConfig
    from app.planning.service import PlanningService

    llm_client = OpenRouterLLMClient(
        OpenRouterConfig(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            max_retries=0,
            retry_backoff_seconds=0.0,
            timeout_seconds=20,
            temperature=0.0,
            max_tokens=1200,
        )
    )
    logger.info("OpenRouter client initialized")

    planning_service = PlanningService(
        llm_client=llm_client,
        config=PlanningConfig(
            default_model_name=model_name,
            include_prompt_debug=show_prompts,
            include_raw_response=True,
        ),
    )
    logger.info("Planning service initialized")
    return planning_service


def main() -> None:
    _ensure_backend_on_path()
    from app.metrics.config import MetricsConfig
    from app.metrics.evaluator import MetricsEvaluator
    from app.planning.data.loaders import FewShotDatasetLoader, ToolRegistryLoader

    parser = _build_arg_parser()
    args = parser.parse_args()

    logger.info("Starting local smoke test")
    logger.info("Dataset: %s", args.dataset)
    logger.info("Tools: %s", args.tools)
    logger.info("Seed: %d", args.seed)
    logger.info("Few-shot count: %d", args.few_shot_count)

    api_key, model_name, base_url = _resolve_openrouter_settings(REPO_ROOT)
    logger.info("Model: %s", model_name)
    logger.info("Base URL: %s", base_url)

    dataset_loader = FewShotDatasetLoader()
    tool_loader = ToolRegistryLoader()

    logger.info("Loading dataset examples")
    examples = dataset_loader.load_examples(args.dataset)
    logger.info("Loaded %d examples", len(examples))

    logger.info("Loading tool registry")
    tools = tool_loader.load(args.tools)
    logger.info("Loaded %d tools", len(tools))

    target_index, target_example, few_shot_examples = _select_target_and_few_shot(
        examples=examples,
        seed=args.seed,
        few_shot_count=args.few_shot_count,
    )
    logger.info("Selected target index: %d", target_index)
    logger.info("Selected %d few-shot examples", len(few_shot_examples))

    planning_service = _build_planning_service(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        show_prompts=args.show_prompts,
    )

    logger.info("Running planning inference")
    inference_result = planning_service.predict_from_parts(
        task=target_example.task,
        available_tools=tools,
        few_shot_examples=few_shot_examples,
        model_name=model_name,
    )
    logger.info("Planning inference completed")

    sample = _build_evaluation_sample(
        target_index=target_index,
        target_example=target_example,
        tools=tools,
        inference_result=inference_result,
    )

    enable_judge_metrics = not args.disable_judge_metrics
    evaluator = MetricsEvaluator(
        config=MetricsConfig(
            enable_judge_metrics=enable_judge_metrics,
            enable_semantic_similarity=False,
        ),
        llm_client=planning_service.llm_client if enable_judge_metrics else None,
    )
    logger.info("Metrics evaluator initialized (judge=%s)", enable_judge_metrics)

    logger.info("Running metrics evaluation")
    metric_result = evaluator.evaluate_sample(sample)
    logger.info("Metrics evaluation completed")

    report = _render_report(
        target_index=target_index,
        target_example=target_example,
        few_shot_examples=few_shot_examples,
        model_name=model_name,
        show_prompts=args.show_prompts,
        inference_result=inference_result,
        metric_result=metric_result,
    )
    sys.stdout.write(report)


if __name__ == "__main__":
    main()

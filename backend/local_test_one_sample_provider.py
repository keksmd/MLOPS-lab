from __future__ import annotations

import argparse
import json
import logging
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


def _ensure_backend_on_path() -> None:
    backend_dir_str = str(BACKEND_DIR)
    if backend_dir_str not in sys.path:
        sys.path.insert(0, backend_dir_str)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one random-sample local smoke test using env-selected providers "
            "for planning and judge metrics."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few-shot-count", type=int, default=1)
    parser.add_argument("--show-prompts", action="store_true")
    parser.add_argument("--disable-judge-metrics", action="store_true")
    return parser


def _select_target_and_few_shot(
    *,
    examples: list[Any],
    seed: int,
    few_shot_count: int,
) -> tuple[int, Any, list[Any]]:
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
        pool, list(range(k)), fallback_count=k
    )
    return target_index, target_example, few_shot_examples


def _build_evaluation_sample(
    *,
    target_index: int,
    target_example: Any,
    tools: list[Any],
    inference_result: Any,
) -> EvaluationSample:
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
    planning_provider: str,
    planning_model_name: str,
    judge_provider: str | None,
    show_prompts: bool,
    inference_result: Any,
    metric_result: dict[str, Any],
) -> str:
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
    lines.append("PLANNING MODEL")
    lines.append(f"provider: {planning_provider}")
    lines.append(f"model: {planning_model_name}")

    if judge_provider is not None:
        lines.append("=" * 100)
        lines.append("JUDGE MODEL")
        lines.append(f"provider: {judge_provider}")

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
            inference_result.prediction.model_dump(), ensure_ascii=False, indent=2
        )
    )

    lines.append("=" * 100)
    lines.append("GOLD")
    lines.append(
        json.dumps(target_example.output.model_dump(), ensure_ascii=False, indent=2)
    )

    lines.append("=" * 100)
    lines.append("METRICS")
    lines.append(json.dumps(metric_result, ensure_ascii=False, indent=2))
    return "\n".join(lines) + "\n"


def _build_planning_service(*, show_prompts: bool) -> tuple[PlanningService, str, str]:
    _ensure_backend_on_path()
    from app.core.llm_provider_settings import llm_provider_settings
    from app.metrics.llm.provider_factory import build_llm_client
    from app.planning.config import PlanningConfig
    from app.planning.service import PlanningService

    provider = llm_provider_settings.PLANNING_LLM_PROVIDER
    if provider == "openrouter":
        model_name = llm_provider_settings.OPENROUTER_MODEL_NAME
    else:
        model_name = llm_provider_settings.LOCAL_MODEL_NAME

    llm_client = build_llm_client(provider=provider, model_name=model_name)
    planning_service = PlanningService(
        llm_client=llm_client,
        config=PlanningConfig(
            default_model_name=model_name,
            include_prompt_debug=show_prompts,
            include_raw_response=True,
        ),
    )
    return planning_service, provider, model_name


def _build_judge_client() -> tuple[Any | None, str | None]:
    _ensure_backend_on_path()
    from app.core.llm_provider_settings import llm_provider_settings
    from app.metrics.llm.provider_factory import build_llm_client

    provider = llm_provider_settings.effective_judge_provider
    if provider == "openrouter":
        model_name = llm_provider_settings.OPENROUTER_MODEL_NAME
    else:
        model_name = llm_provider_settings.LOCAL_MODEL_NAME

    return build_llm_client(provider=provider, model_name=model_name), provider


def main() -> None:
    _ensure_backend_on_path()
    from app.metrics.config import MetricsConfig
    from app.metrics.evaluator import MetricsEvaluator
    from app.planning.data.loaders import FewShotDatasetLoader, ToolRegistryLoader

    parser = _build_arg_parser()
    args = parser.parse_args()

    logger.info("Starting provider-aware local smoke test")
    logger.info("Dataset: %s", args.dataset)
    logger.info("Tools: %s", args.tools)
    logger.info("Seed: %d", args.seed)
    logger.info("Few-shot count: %d", args.few_shot_count)

    dataset_loader = FewShotDatasetLoader()
    tool_loader = ToolRegistryLoader()

    examples = dataset_loader.load_examples(args.dataset)
    tools = tool_loader.load(args.tools)

    target_index, target_example, few_shot_examples = _select_target_and_few_shot(
        examples=examples,
        seed=args.seed,
        few_shot_count=args.few_shot_count,
    )

    planning_service, planning_provider, planning_model_name = _build_planning_service(
        show_prompts=args.show_prompts
    )

    inference_result = planning_service.predict_from_parts(
        task=target_example.task,
        available_tools=tools,
        few_shot_examples=few_shot_examples,
        model_name=planning_model_name,
    )

    sample = _build_evaluation_sample(
        target_index=target_index,
        target_example=target_example,
        tools=tools,
        inference_result=inference_result,
    )

    enable_judge_metrics = not args.disable_judge_metrics
    judge_client = None
    judge_provider = None
    if enable_judge_metrics:
        judge_client, judge_provider = _build_judge_client()

    evaluator = MetricsEvaluator(
        config=MetricsConfig(
            enable_judge_metrics=enable_judge_metrics,
            enable_semantic_similarity=False,
        ),
        llm_client=judge_client,
    )

    metric_result = evaluator.evaluate_sample(sample)

    report = _render_report(
        target_index=target_index,
        target_example=target_example,
        few_shot_examples=few_shot_examples,
        planning_provider=planning_provider,
        planning_model_name=planning_model_name,
        judge_provider=judge_provider,
        show_prompts=args.show_prompts,
        inference_result=inference_result,
        metric_result=metric_result,
    )
    sys.stdout.write(report)


if __name__ == "__main__":
    main()

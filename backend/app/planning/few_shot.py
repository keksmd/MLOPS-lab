from __future__ import annotations

from app.metrics.schemas import PlannerOutput

from .schemas import PlannerExample


class FewShotSelector:
    """Selects few-shot examples for baseline planner prompting."""

    def select_by_indices(
        self,
        examples: list[PlannerExample],
        indices: list[int],
        *,
        fallback_count: int = 3,
    ) -> list[PlannerExample]:
        """Select examples by explicit indices, with a fallback to the first valid examples."""
        valid_examples = [ex for ex in examples if ex.task.strip() and ex.output.plan]
        selected = [examples[i] for i in indices if 0 <= i < len(examples) and examples[i].task.strip() and examples[i].output.plan]
        if len(selected) >= fallback_count:
            return selected[:fallback_count]
        for example in valid_examples:
            if example not in selected:
                selected.append(example)
            if len(selected) >= fallback_count:
                break
        return selected

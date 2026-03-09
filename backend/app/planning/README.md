# Planning inference module

This package moves the baseline planner inference logic out of the notebook and into reusable Python modules.

## Design goals
- reuse the shared `PlannerOutput`, `ActionCall`, and `ToolSpec` schemas from `app.metrics.schemas`
- keep the LLM backend compatible with the metrics module by using `BaseLLMClient`
- support few-shot prompting without changing the LLM client interface
- make the planner output directly consumable by `EvaluationSample` in the metrics module

## Main entrypoint
Use `PlanningService.predict()` or `PlanningService.predict_from_parts()`.

class PlanningError(Exception):
    """Base exception for the planning inference module."""


class PromptBuildError(PlanningError):
    """Raised when a planning prompt cannot be constructed."""


class PredictionParseError(PlanningError):
    """Raised when the model output cannot be parsed into planner JSON."""


class RepositoryLoadError(PlanningError):
    """Raised when dataset or tool registry files cannot be loaded."""

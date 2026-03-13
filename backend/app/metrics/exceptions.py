class MetricsError(Exception):
    """Base exception for the metrics module."""


class LLMClientError(MetricsError):
    """Raised when an LLM backend fails or returns an invalid response."""


class JudgeResponseParseError(MetricsError):
    """Raised when judge output cannot be parsed into the expected schema."""


class InvalidPredictionError(MetricsError):
    """Raised when the evaluated prediction cannot be normalized or validated."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Abstract interface implemented by all LLM backends used in evaluation."""

    @abstractmethod
    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Generate raw text from the underlying model.

        Args:
            system_prompt: System-level instruction.
            user_prompt: User prompt content.

        Returns:
            Raw model text output.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """
        Generate a JSON object from the underlying model.

        Args:
            system_prompt: System-level instruction.
            user_prompt: User prompt content.

        Returns:
            Parsed JSON object.
        """
        raise NotImplementedError

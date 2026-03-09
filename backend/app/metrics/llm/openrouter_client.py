from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from pydantic import RootModel

from ..config import OpenRouterConfig
from ..exceptions import LLMClientError
from .base import BaseLLMClient, SchemaT


class _JSONObject(RootModel[dict[str, Any]]):
    """Wrapper schema for arbitrary JSON-object responses."""


class OpenRouterLLMClient(BaseLLMClient):
    """LangChain-backed OpenRouter client."""

    def __init__(self, config: OpenRouterConfig) -> None:
        """Initialize the OpenRouter chat model wrapper."""
        self.config = config
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "{user_prompt}"),
            ]
        )
        self._chat_model = ChatOpenRouter(
            model=config.model_name,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
            timeout=config.timeout_seconds,
        )

    def _build_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> list[BaseMessage]:
        """Render system and user prompts into LangChain chat messages."""
        prompt_value = self._prompt_template.invoke(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return list(prompt_value.messages)

    @staticmethod
    def _message_to_text(message: BaseMessage) -> str:
        """Extract plain text content from a LangChain message."""
        content = message.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)

            rendered = "\n".join(part for part in parts if part.strip()).strip()
            if rendered:
                return rendered

        raise LLMClientError("Model returned non-text content.")

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate raw text with the underlying chat model."""
        try:
            response = self._chat_model.invoke(
                self._build_messages(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            )
        except Exception as exc:  # pragma: no cover - network/provider path
            raise LLMClientError(f"OpenRouter text generation failed: {exc}") from exc

        return self._message_to_text(response)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Generate an arbitrary JSON object using structured output."""
        result = self.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=_JSONObject,
        )
        return result.root

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        """Generate a strongly typed response using LangChain structured output."""
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        strategies: tuple[dict[str, Any], ...] = (
            {"method": "json_schema", "strict": True},
            {"method": "function_calling", "strict": True},
            {},
        )

        last_error: Exception | None = None

        for strategy in strategies:
            try:
                structured_model = self._chat_model.with_structured_output(
                    schema,
                    **strategy,
                )
                response = structured_model.invoke(messages)

                if isinstance(response, schema):
                    return response

                return schema.model_validate(response)
            except Exception as exc:  # pragma: no cover - network/provider path
                last_error = exc

        error_message = "OpenRouter structured generation failed."
        if last_error is not None:
            error_message = f"{error_message} Last error: {last_error}"

        raise LLMClientError(error_message)
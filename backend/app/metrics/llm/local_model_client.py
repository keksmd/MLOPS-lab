from __future__ import annotations

import base64
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import RootModel

from ..exceptions import LLMClientError
from ..local_model_config import LocalModelConfig
from .base import BaseLLMClient, SchemaT


class _JSONObject(RootModel[dict[str, Any]]):
    """Wrapper schema for arbitrary JSON-object responses."""


class LocalModelLLMClient(BaseLLMClient):
    """OpenAI-compatible local-model client built on top of ChatOpenAI."""

    def __init__(self, config: LocalModelConfig) -> None:
        self.config = config
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "{user_prompt}"),
            ]
        )

        basic_token = base64.b64encode(
            f"{config.user}:{config.password.get_secret_value()}".encode()
        ).decode("utf-8")
        default_headers = {"Authorization": f"Basic {basic_token}"}

        self._chat_model = ChatOpenAI(
            model=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=default_headers,
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
        prompt_value = self._prompt_template.invoke(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return list(prompt_value.to_messages())

    @staticmethod
    def _message_to_text(message: BaseMessage) -> str:
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
        try:
            response = self._chat_model.invoke(
                self._build_messages(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            )
        except Exception as exc:  # pragma: no cover - transport/provider path
            raise LLMClientError(f"Local model text generation failed: {exc}") from exc

        return self._message_to_text(response)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
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
            except Exception as exc:  # pragma: no cover - transport/provider path
                last_error = exc

        error_message = "Local model structured generation failed."
        if last_error is not None:
            error_message = f"{error_message} Last error: {last_error}"

        raise LLMClientError(error_message)

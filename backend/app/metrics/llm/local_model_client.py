from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ..exceptions import LLMClientError
from ..local_model_config import LocalModelConfig
from .base import BaseLLMClient, SchemaT


class LocalModelLLMClient(BaseLLMClient):
    """LLM client for the OpenAI-compatible local model gateway.

    This client intentionally avoids tool/function calling. Local capability
    probing showed reliable support for plain chat completions, JSON mode, and
    SDK structured parsing on small prompts, while tool calling returned a 500.
    For planning and judge calls the most stable path is: request JSON mode,
    parse JSON, and validate it with Pydantic.
    """

    def __init__(self, config: LocalModelConfig) -> None:
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key="dummy",
            default_headers={"Authorization": config.auth_header},
            max_retries=config.max_retries,
            timeout=config.timeout_seconds,
        )

    @staticmethod
    def _extract_text_content(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except Exception as exc:  # pragma: no cover - malformed provider path
            raise LLMClientError(
                "Local model returned an unexpected response shape."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("Local model returned empty text content.")
        return content.strip()

    @staticmethod
    def _build_json_mode_system_prompt(
        *,
        system_prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str:
        instructions = [
            system_prompt.strip(),
            "Return exactly one valid JSON object.",
            "Do not use markdown code fences.",
            "Do not include explanatory text before or after the JSON.",
        ]

        if schema is not None:
            schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
            instructions.append(
                f"The JSON object must conform to this JSON Schema: {schema_json}"
            )

        return "\n\n".join(part for part in instructions if part)

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as exc:  # pragma: no cover - network/provider path
            raise LLMClientError(f"Local model text generation failed: {exc}") from exc

        return self._extract_text_content(response)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        try:
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_json_mode_system_prompt(
                            system_prompt=system_prompt,
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as exc:  # pragma: no cover - network/provider path
            raise LLMClientError(f"Local model JSON generation failed: {exc}") from exc

        raw_content = self._extract_text_content(response)

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                "Local model returned invalid JSON in JSON mode. "
                f"Raw content: {raw_content}"
            ) from exc

        if not isinstance(parsed, dict):
            raise LLMClientError("Local model returned JSON that is not an object.")

        return parsed

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        try:
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_json_mode_system_prompt(
                            system_prompt=system_prompt,
                            schema=schema,
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as exc:  # pragma: no cover - network/provider path
            raise LLMClientError(
                f"Local model structured generation failed: {exc}"
            ) from exc

        raw_content = self._extract_text_content(response)

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                "Local model returned invalid structured JSON. "
                f"Raw content: {raw_content}"
            ) from exc

        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            raise LLMClientError(
                "Local model returned JSON that does not match the expected schema. "
                f"Validation error: {exc}"
            ) from exc

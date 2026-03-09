from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from ..config import OpenRouterConfig
from ..exceptions import LLMClientError
from .base import BaseLLMClient


class OpenRouterLLMClient(BaseLLMClient):
    """OpenRouter-backed implementation of the generic LLM client interface."""

    def __init__(self, config: OpenRouterConfig) -> None:
        """
        Initialize an OpenRouter client.

        Args:
            config: OpenRouter connection and generation settings.
        """
        self.config = config

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.http_referer:
            headers["HTTP-Referer"] = self.config.http_referer
        if self.config.app_title:
            headers["X-Title"] = self.config.app_title
        return headers

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Call OpenRouter chat completions and return raw text output.

        Retries transient failures and raises LLMClientError on final failure.
        """
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }

        last_error: Exception | None = None
        headers = self._build_headers()
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = requests.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]

                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(self.config.retry_backoff_seconds * attempt)
                    continue

                response.raise_for_status()
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                time.sleep(self.config.retry_backoff_seconds * attempt)

        raise LLMClientError(f"OpenRouter request failed: {last_error}")

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """
        Generate a JSON response and parse it safely.

        Raises:
            LLMClientError: If the backend returns invalid JSON.
        """
        raw_text = self.generate_text(system_prompt=system_prompt, user_prompt=user_prompt)
        try:
            return self._parse_json_text(raw_text)
        except Exception as exc:
            raise LLMClientError(f"Model returned invalid JSON: {raw_text}") from exc

    @staticmethod
    def _parse_json_text(raw_text: str) -> dict[str, Any]:
        """Parse model output into a JSON object, tolerating fenced code blocks."""
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj

        raise ValueError("Response is not a valid JSON object.")

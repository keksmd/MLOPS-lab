from __future__ import annotations

import base64
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
LOCAL_MODEL_ENV_PATH = BACKEND_DIR / ".env_local_model"
ROOT_ENV_PATH = BACKEND_DIR / ".env"


class LocalModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(LOCAL_MODEL_ENV_PATH), str(ROOT_ENV_PATH)),
        env_ignore_empty=True,
        extra="ignore",
    )

    LOCAL_MODEL_USER: str | None = None
    LOCAL_MODEL_PASSWORD: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def basic_auth_header(self) -> str | None:
        if not self.LOCAL_MODEL_USER or not self.LOCAL_MODEL_PASSWORD:
            return None

        basic = base64.b64encode(
            f"{self.LOCAL_MODEL_USER}:{self.LOCAL_MODEL_PASSWORD}".encode()
        ).decode()
        return f"Basic {basic}"

    def require_basic_auth_header(self) -> str:
        header = self.basic_auth_header
        if header is None:
            raise RuntimeError(
                "LOCAL_MODEL_USER and LOCAL_MODEL_PASSWORD are not set. "
                "Put them into backend/.env_local_model or the shell."
            )
        return header


local_model_settings = LocalModelSettings()

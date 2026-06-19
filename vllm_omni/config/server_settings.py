import os
import warnings
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_deprecated_env(old_name: str, new_name: str, value):
    old = os.environ.get(old_name)
    if old is None:
        return value
    warnings.warn(
        f"{old_name} is deprecated and will be removed in a future release; use {new_name} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value if new_name in os.environ else old


class FileBackend(BaseModel):
    type: Literal["file"] = "file"
    path: str = Field(default="/tmp/storage", validate_default=True, description="Local path to store completed files.")
    file_concurrency: int = Field(
        default=4, validate_default=True, description="Maximum number of file operations permitted at a time"
    )
    file_ttl: int | None = Field(
        default=None, description="Optional TTL (in seconds) configuration settings for locally stored files."
    )
    ttl_sweep_interval: int | None = Field(
        default=None, description="Optional frequency (in seconds) to enforce file TTLs."
    )

    @field_validator("path", mode="before")
    @classmethod
    def _migrate_path(cls, value):
        return _resolve_deprecated_env("VLLM_OMNI_STORAGE_PATH", "VLLM_OMNI_SERVER_STORAGE__PATH", value)

    @field_validator("file_concurrency", mode="before")
    @classmethod
    def _migrate_file_concurrency(cls, value):
        return _resolve_deprecated_env(
            "VLLM_OMNI_STORAGE_MAX_CONCURRENCY", "VLLM_OMNI_SERVER_STORAGE__FILE_CONCURRENCY", value
        )

    @model_validator(mode="after")
    def set_default_ttl_sweep_interval(self) -> "FileBackend":
        if self.file_ttl is not None and self.ttl_sweep_interval is None:
            self.ttl_sweep_interval = 300
        return self


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLLM_OMNI_SERVER_", env_nested_delimiter="__")
    storage: FileBackend = Field(default_factory=FileBackend)


SERVER_SETTINGS_CONFIG = ServerSettings()

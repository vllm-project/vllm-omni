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


class StorageConfig(BaseModel):
    """Storage backend configuration (``ServerSettings.storage``)."""

    type: Literal["file", "s3"] = "file"
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

    # -- s3 backend -----------------------------------------------------
    bucket: str | None = Field(default=None, description="S3 bucket that holds artifacts and job records.")
    region: str | None = Field(default=None, description="S3 region; derived from the credential chain when omitted.")
    endpoint_url: str | None = Field(
        default=None, description="S3-compatible endpoint (MinIO, Ceph, ...); AWS default when omitted."
    )
    prefix: str = Field(default="omni", description="Key prefix prepended to every stored object.")
    presign: bool = Field(
        default=True, description="Redirect /v1/videos/{id}/content to a presigned S3 URL instead of proxying bytes."
    )
    presign_expires: int = Field(default=3600, description="Lifetime (in seconds) of generated presigned URLs.")
    proxy: bool = Field(
        default=False, description="Stream artifacts through the API server even when presigning is supported."
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
    def set_default_ttl_sweep_interval(self) -> "StorageConfig":
        if self.file_ttl is not None and self.ttl_sweep_interval is None:
            self.ttl_sweep_interval = 300
        if self.type == "s3" and not self.bucket:
            raise ValueError("`storage.bucket` is required when `storage.type` is 's3'.")
        if self.presign_expires < 1:
            raise ValueError("`storage.presign_expires` must be greater than or equal to 1.")
        return self


#: Previous name of :class:`StorageConfig`, kept for existing imports.
FileBackend = StorageConfig


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLLM_OMNI_SERVER_", env_nested_delimiter="__")
    storage: StorageConfig = Field(default_factory=StorageConfig)


SERVER_SETTINGS_CONFIG = ServerSettings()

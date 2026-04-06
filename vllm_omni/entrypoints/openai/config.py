from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_PREFIX = "VLLM_OMNI_SERVER"
STORAGE_PREFIX = f"{BASE_PREFIX}_STORAGE__"


class FileBackend(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=STORAGE_PREFIX)

    type: Literal["file"] = "file"
    path: str = Field(default="/tmp/storage", description="Local path to store completed files.")
    file_concurrency: int = Field(default=4, description="Maximum number of file operations permitted at a time")
    file_ttl: int | None = Field(
        default=None, description="Optional TTL (in seconds) configuration settings for locally stored files."
    )
    ttl_sweep_interval: int | None = Field(
        default=None, description="Optional frequency (in seconds) to enforce file TTLs."
    )

    @model_validator(mode="after")
    def set_default_ttl_sweep_interval(self) -> "FileBackend":
        if self.file_ttl is not None and self.ttl_sweep_interval is None:
            self.ttl_sweep_interval = 300
        return self


STORAGE_BACKENDS: TypeAlias = Annotated[
    FileBackend,  # Should always be left last in the union list
    Field(discriminator="type"),
]


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=BASE_PREFIX)
    storage: STORAGE_BACKENDS = Field(default_factory=FileBackend)


CONFIG = ServerSettings()

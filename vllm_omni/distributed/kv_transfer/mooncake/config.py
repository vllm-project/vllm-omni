# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration for the joint Thinker KV/hidden-state Mooncake backend."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

CONNECTOR_NAME = "OmniMooncakeStoreConnector"
CONNECTOR_MODULE = "vllm_omni.distributed.kv_transfer.mooncake.connector"
EXTRA_CONFIG_KEY = "omni_mooncake"


@dataclass(frozen=True)
class MooncakeConfig:
    """Explicit cache identity and bounded temporary HS resources.

    ``model_revision`` identifies the checkpoint used to produce stored rows.
    It must be supplied even when the model is loaded from a local directory.
    ``config_file`` is the JSON configuration consumed by the upstream
    Mooncake Store KV connector, not an inter-stage connector configuration.
    """

    config_file: str
    namespace: str
    model_revision: str
    max_hs_bytes: int = 256 * 1024 * 1024
    max_pending_transfers: int = 16
    load_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in ("config_file", "namespace", "model_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"mooncake_config.{name} must be a nonempty string")
        for name in ("max_hs_bytes", "max_pending_transfers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"mooncake_config.{name} must be a positive integer")
        timeout = self.load_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("mooncake_config.load_timeout_seconds must be a finite positive number")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MooncakeConfig:
        if not isinstance(value, Mapping):
            raise ValueError("mooncake_config must be a mapping")
        unknown = value.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown mooncake_config settings: {sorted(unknown)}")
        missing = {"config_file", "namespace", "model_revision"} - value.keys()
        if missing:
            raise ValueError(f"Missing mooncake_config settings: {sorted(missing)}")
        # The constructor performs runtime validation of configuration values.
        return cls(**value)  # type: ignore[arg-type]

    @classmethod
    def from_vllm_config(cls, config: VllmConfig) -> MooncakeConfig:
        transfer = config.kv_transfer_config
        if transfer is None:
            raise ValueError("Mooncake KV/HS caching requires kv_transfer_config")
        return cls.from_dict(transfer.kv_connector_extra_config.get(EXTRA_CONFIG_KEY, {}))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

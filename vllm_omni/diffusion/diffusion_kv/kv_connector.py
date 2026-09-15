# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diffusion-side assembly facade built on the upstream Mooncake KV connector.

The configured ``kv_connector`` (currently ``MooncakeConnector``) is created
through vLLM's native ``KVConnectorFactory``. PR0 only validates configuration
and creates the Scheduler/Worker connector objects; it does not subclass or
reimplement Mooncake's transfer runtime. Real page sizing, cache registration,
admission, and remote loading belong to the follow-up landing PR.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_transfer_state import ensure_kv_transfer_initialized, ensure_kv_transfer_shutdown
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


def mint_transfer_id(request_id: str) -> str:
    """Return the stable ticket shared by one AR -> Diffusion handoff."""

    return f"xfer-{request_id}"


def build_source_kv_transfer_params(
    *,
    transfer_id: str,
    remote_engine_id: str | None,
    remote_bootstrap_addr: str | None,
) -> dict[str, Any]:
    """Build the opaque producer-side metadata bag for an AR request."""

    params: dict[str, Any] = {
        "transfer_id": transfer_id,
        "do_remote_decode": True,
        "do_remote_prefill": False,
    }
    if remote_engine_id:
        params["remote_engine_id"] = remote_engine_id
    if remote_bootstrap_addr:
        params["remote_bootstrap_addr"] = remote_bootstrap_addr
    return params


def build_target_kv_transfer_params(
    *,
    source_params: Mapping[str, Any],
    remote_engine_id: str | None,
    remote_bootstrap_addr: str | None,
) -> dict[str, Any]:
    """Build the consumer-side metadata bag without interpreting pages."""

    params = dict(source_params)
    params["do_remote_prefill"] = True
    params["do_remote_decode"] = False
    if remote_engine_id:
        params["remote_engine_id"] = remote_engine_id
    if remote_bootstrap_addr:
        params["remote_bootstrap_addr"] = remote_bootstrap_addr
    return params


def bootstrap_addr_from_kv_transfer_config(kv_transfer_config: KVTransferConfig | None) -> str | None:
    """Read an optional connector bootstrap endpoint from native config."""

    if kv_transfer_config is None:
        return None
    extra_config = kv_transfer_config.kv_connector_extra_config or {}
    for key in ("bootstrap_addr", "prefill_bootstrap_addr", "mooncake_master"):
        value = extra_config.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def parse_kv_transfer_config(value: object | None) -> KVTransferConfig | None:
    """Normalize a YAML mapping or an existing native config."""

    if value is None:
        return None
    if isinstance(value, KVTransferConfig):
        _validate_kv_transfer_config(value)
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        if not payload:
            return None
        engine_id = payload.get("engine_id")
        if not isinstance(engine_id, str) or not engine_id.strip():
            raise ValueError("Diffusion native kv_transfer_config requires a non-empty engine_id")
        config = KVTransferConfig(**payload)
        _validate_kv_transfer_config(config)
        return config
    raise TypeError(f"kv_transfer_config must be a mapping or KVTransferConfig, got {type(value)!r}")


def create_scheduler_kv_connector(
    od_config: OmniDiffusionConfig,
) -> KVConnectorBase_V1 | None:
    """Create a Scheduler-role connector when native config is present."""

    kv_transfer_config = getattr(od_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return None
    if not isinstance(kv_transfer_config, KVTransferConfig):
        raise TypeError(
            f"Diffusion native kv_transfer_config must be KVTransferConfig, got {type(kv_transfer_config)!r}"
        )

    vllm_config = _build_native_vllm_config(od_config)
    connector = KVConnectorFactory.create_connector(
        config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=_empty_kv_cache_config(),
    )
    logger.info(
        "Created KV connector stub (SCHEDULER role): connector=%s engine_id=%s",
        kv_transfer_config.kv_connector,
        kv_transfer_config.engine_id,
    )
    return connector


def init_worker_kv_connector(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
    """Initialize the Worker-role connector with its rank-local cache plan."""

    if vllm_config.kv_transfer_config is None:
        return
    ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
    logger.info(
        "Initialized KV connector stub (WORKER role): connector=%s engine_id=%s",
        vllm_config.kv_transfer_config.kv_connector,
        vllm_config.kv_transfer_config.engine_id,
    )


def shutdown_kv_connector(*, scheduler_connector: KVConnectorBase_V1 | None = None) -> None:
    """Shutdown Worker and Scheduler connector objects idempotently."""

    if scheduler_connector is not None:
        scheduler_connector.shutdown()
    ensure_kv_transfer_shutdown()


def _empty_kv_cache_config() -> KVCacheConfig:
    """Empty cache config used only for PR0 connector assembly."""

    return KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])


def _build_native_vllm_config(od_config: OmniDiffusionConfig) -> VllmConfig:
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

    kv_transfer_config = getattr(od_config, "kv_transfer_config", None)
    assert isinstance(kv_transfer_config, KVTransferConfig)
    vllm_config = create_diffusion_vllm_config(torch.device("cpu"), od_config)
    vllm_config.kv_transfer_config = kv_transfer_config
    return vllm_config


def _validate_kv_transfer_config(config: KVTransferConfig) -> None:
    engine_id = config.engine_id
    if not isinstance(engine_id, str) or not engine_id.strip():
        raise ValueError("Diffusion native kv_transfer_config requires a non-empty engine_id")
    if config.kv_connector is not None and config.kv_role is None:
        raise ValueError("Diffusion native kv_transfer_config requires kv_role when kv_connector is set")
    if config.kv_role not in (None, "kv_consumer", "kv_producer", "kv_both"):
        raise ValueError(f"Unsupported KV connector role: {config.kv_role!r}")

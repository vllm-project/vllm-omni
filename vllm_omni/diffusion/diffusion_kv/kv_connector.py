# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Assemble native vLLM KV connectors for Scheduler-owned diffusion pages."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_transfer_state import ensure_kv_transfer_initialized, ensure_kv_transfer_shutdown
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
    from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput

logger = init_logger(__name__)


async def _build_prefix_transfer_params(build_transfer_params, ready_reqs, agent_meta, local_regions, remote_regions):
    prefix_reqs = []
    for request_id, send_meta in ready_reqs:
        remote_groups = agent_meta.req_blocks[request_id][1]
        if len(send_meta.local_block_ids) == len(remote_groups):
            send_meta = replace(
                send_meta,
                local_block_ids=[
                    local[: len(remote)] for local, remote in zip(send_meta.local_block_ids, remote_groups, strict=True)
                ],
            )
        prefix_reqs.append((request_id, send_meta))
    return await build_transfer_params(prefix_reqs, agent_meta, local_regions, remote_regions)


def __getattr__(name: str) -> Any:
    if name != "MooncakeConnector":
        raise AttributeError(name)

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeConnector as NativeMooncakeConnector,
    )

    class MooncakeConnector(NativeMooncakeConnector):
        """Use source prefixes for AR-to-diffusion pulls, not decode suffixes."""

        def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig):
            super().__init__(vllm_config, role, kv_cache_config)
            worker = self.connector_worker
            if worker is not None and worker.is_kv_producer:
                worker._build_transfer_params = partial(_build_prefix_transfer_params, worker._build_transfer_params)

    globals()[name] = MooncakeConnector
    return MooncakeConnector


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
    kv_cache_config: KVCacheConfig | None = None,
    vllm_config: VllmConfig | None = None,
) -> KVConnectorBase_V1 | None:
    """Create a Scheduler-role connector when native config is present."""

    kv_transfer_config = getattr(od_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return None
    if not isinstance(kv_transfer_config, KVTransferConfig):
        raise TypeError(
            f"Diffusion native kv_transfer_config must be KVTransferConfig, got {type(kv_transfer_config)!r}"
        )

    if kv_cache_config is None or vllm_config is None:
        raise ValueError("KV transfer requires paged_scheduler and a native cache configuration")
    connector = KVConnectorFactory.create_connector(
        config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=kv_cache_config,
    )
    logger.info(
        "Created KV connector (SCHEDULER role): connector=%s engine_id=%s",
        kv_transfer_config.kv_connector,
        kv_transfer_config.engine_id,
    )
    return connector


def init_worker_kv_connector(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
    """Initialize the Worker-role connector with its rank-local cache plan."""

    if vllm_config.kv_transfer_config is None:
        return
    import vllm.distributed.parallel_state as parallel_state

    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

    tp_group = parallel_state._TP
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if getattr(parallel_config, "prefill_context_parallel_size", 1) > 1 and tp_group.world_size == 1:
        parallel_state._TP = get_sp_group()
    try:
        ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
    finally:
        parallel_state._TP = tp_group
    logger.info(
        "Initialized KV connector (WORKER role): connector=%s engine_id=%s",
        vllm_config.kv_transfer_config.kv_connector,
        vllm_config.kv_transfer_config.engine_id,
    )


def shutdown_kv_connector(*, scheduler_connector: KVConnectorBase_V1 | None = None) -> None:
    """Shutdown Worker and Scheduler connector objects idempotently."""

    if scheduler_connector is not None:
        scheduler_connector.shutdown()
    ensure_kv_transfer_shutdown()


def prepare_kv_requests(requests: tuple[DiffusionKVRequest, ...], params: Mapping[str, Any]) -> None:
    for request in requests:
        request.kv_transfer_params = dict(params)
        num_tokens = min(len(request.prompt_token_ids), params["num_transfer_tokens"])
        request.prompt_token_ids = request.prompt_token_ids[:num_tokens]
        request.num_prompt_tokens = num_tokens


def commit_kv_load(
    connector: KVConnectorBase_V1,
    manager: KVCacheManager,
    requests: tuple[DiffusionKVRequest, ...],
    matched_tokens: list[int],
) -> set[str]:
    expected = set()
    for request, num_tokens in zip(requests, matched_tokens, strict=True):
        blocks = manager.get_blocks(request.request_id)
        prefix_blocks = KVCacheBlocks(
            tuple(
                group[: (num_tokens + spec.kv_cache_spec.block_size - 1) // spec.kv_cache_spec.block_size]
                for group, spec in zip(blocks.blocks, manager.kv_cache_config.kv_cache_groups, strict=True)
            )
        )
        connector.update_state_after_alloc(request, prefix_blocks, num_tokens)
        request.num_computed_tokens = num_tokens
        if request.kv_transfer_params is not None:
            expected.add(request.request_id)
    return expected


def wait_for_kv_load(
    active_connector: ActiveKVConnector, scheduler_output: DiffusionSchedulerOutput, timeout: float
) -> KVConnectorOutput:
    connector = active_connector.kv_connector
    active_connector.pre_forward(scheduler_output)
    pending = set(scheduler_output.kv_transfer_request_ids)
    received, sent = set(), set()
    deadline = time.monotonic() + timeout
    while True:
        finished_sending, finished_recving = connector.get_finished(scheduler_output.finished_req_ids)
        sent.update(finished_sending or ())
        received.update(finished_recving or ())
        pending.difference_update(received)
        if connector.get_block_ids_with_load_errors():
            raise RuntimeError("Diffusion KV connector reported invalid remote pages")
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out receiving diffusion KV for {sorted(pending)}")
        time.sleep(0.001)
    output = active_connector.post_forward(scheduler_output.finished_req_ids)
    output.finished_sending = sent | (output.finished_sending or set())
    output.finished_recving = received | (output.finished_recving or set())
    if output.invalid_block_ids:
        raise RuntimeError("Diffusion KV connector reported invalid remote pages")
    return output


def _validate_kv_transfer_config(config: KVTransferConfig) -> None:
    engine_id = config.engine_id
    if not isinstance(engine_id, str) or not engine_id.strip():
        raise ValueError("Diffusion native kv_transfer_config requires a non-empty engine_id")
    if config.kv_connector is not None and config.kv_role is None:
        raise ValueError("Diffusion native kv_transfer_config requires kv_role when kv_connector is set")
    if config.kv_role not in (None, "kv_consumer", "kv_producer", "kv_both"):
        raise ValueError(f"Unsupported KV connector role: {config.kv_role!r}")

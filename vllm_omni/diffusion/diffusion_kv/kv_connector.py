# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Assemble native vLLM KV connectors for Scheduler-owned diffusion pages."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Mapping
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


class KVTransferRegistrationError(ValueError):
    """A failed handoff whose destination pages have not been dispatched."""


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

    # vLLM 0.29 Mooncake snapshots tp_rank/tp_size in its Worker constructor;
    # consumer receive/region mapping use those saved values, not get_tp_group.
    # Restrict this compatibility bridge to that connector and restore the
    # model's TP group even when construction fails. Never remap model TP for
    # the connector's entire lifetime (it would change model collectives).
    tp_group = parallel_state._TP
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if getattr(parallel_config, "prefill_context_parallel_size", 1) > 1 and tp_group.world_size == 1:
        if vllm_config.kv_transfer_config.kv_connector != "MooncakeConnector":
            raise ValueError("Native SP KV transfer currently requires MooncakeConnector")
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


def install_mooncake_cfg_fanout(connector: KVConnectorBase_V1) -> None:
    """Account for multiple CFG destinations on one native Mooncake ticket.

    vLLM 0.29 initializes ``need_send`` to the paired consumer rank count,
    but increments ``sent`` per consumer request. Adapt only this producer
    instance, after KV initialization and before serving requests. Keep the
    upstream address planning, transport, timeout and completion code intact.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import MooncakeConnector

    if not isinstance(connector, MooncakeConnector):
        return
    worker = connector.connector_worker
    if worker is None or worker.is_kv_consumer or getattr(worker, "_omni_cfg_fanout_installed", False):
        return
    build_transfer_params = worker._build_transfer_params

    async def build_with_fanout(ready_reqs, agent_meta, local_regions, remote_regions):
        # commit_kv_load registers all rows before build_connector_meta. Thus
        # each rank's message contains the complete row set for each ticket,
        # including empty-block notifications. Count the full message, not
        # just the subset whose producer-ready events fired in this iteration.
        rows_per_ticket = Counter(transfer_id for transfer_id, _ in agent_meta.req_blocks.values())
        tickets = {send_meta.transfer_id: send_meta for _, send_meta in ready_reqs}
        for transfer_id, send_meta in tickets.items():
            rows = rows_per_ticket[transfer_id]
            previous = getattr(send_meta, "_omni_cfg_rows", None)
            if rows < 1 or (previous is not None and previous != rows):
                # Use the upstream transfer-error path: it decrements sending
                # without incrementing sent, so a malformed rank cannot free
                # source pages still needed by another rank.
                return [], [], [], [req_id for req_id, _ in ready_reqs], "Inconsistent Mooncake CFG receive count"
        # No await between checking and updating: all handlers run on the
        # sender loop. The marker belongs to the ticket, not to a request-global
        # variable; it disappears with the upstream ticket on completion/expiry.
        for transfer_id, send_meta in tickets.items():
            if getattr(send_meta, "_omni_cfg_rows", None) is None:
                send_meta.need_send *= rows_per_ticket[transfer_id]
                send_meta._omni_cfg_rows = rows_per_ticket[transfer_id]
        return await build_transfer_params(ready_reqs, agent_meta, local_regions, remote_regions)

    worker._build_transfer_params = build_with_fanout  # type: ignore[method-assign]
    setattr(worker, "_omni_cfg_fanout_installed", True)


def commit_kv_load(
    connector: KVConnectorBase_V1,
    manager: KVCacheManager,
    requests: tuple[DiffusionKVRequest, ...],
    matched_tokens: list[int],
) -> set[str]:
    expected = set()
    allocations = []
    # All rows sharing a transfer_id must reach the same connector metadata:
    # the producer uses this complete per-rank row set for fan-out accounting.
    # Validate every CFG row before mutating any connector state.
    for request, num_tokens in zip(requests, matched_tokens, strict=True):
        blocks = manager.get_blocks(request.request_id)
        # Mooncake's producer advertises complete physical blocks. Keep every
        # CFG receiver on that same block boundary so the native connector can
        # use its normal one-to-one page mapping even when the branches have
        # different logical prefix lengths. Bytes beyond ``num_tokens`` are
        # not marked computed and are overwritten by the DiT prefill.
        transfer_tokens = num_tokens
        if num_tokens > 0 and request.kv_transfer_params is not None:
            transfer_tokens = request.kv_transfer_params["num_transfer_tokens"]
            if type(transfer_tokens) is not int or not num_tokens <= transfer_tokens <= request.num_tokens:
                raise KVTransferRegistrationError(
                    "Diffusion KV transfer boundary must cover the reusable prefix "
                    f"without exceeding the allocated sequence: reusable={num_tokens}, "
                    f"transfer={transfer_tokens!r}, allocated={request.num_tokens}"
                )
        prefix_blocks = KVCacheBlocks(
            tuple(
                group[: (transfer_tokens + spec.kv_cache_spec.block_size - 1) // spec.kv_cache_spec.block_size]
                for group, spec in zip(blocks.blocks, manager.kv_cache_config.kv_cache_groups, strict=True)
            )
        )
        allocations.append((request, prefix_blocks, num_tokens))
    try:
        for request, prefix_blocks, num_tokens in allocations:
            connector.update_state_after_alloc(request, prefix_blocks, num_tokens)
            request.num_computed_tokens = num_tokens
            if request.kv_transfer_params is not None:
                expected.add(request.request_id)
    except Exception as exc:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import MooncakeConnector
        from vllm.v1.request import RequestStatus

        if not isinstance(connector, MooncakeConnector):
            # Unknown connectors may have started I/O during registration.
            raise
        # Mooncake only queues metadata here; no Worker has seen addresses yet.
        # Re-arm its pre-scheduling abort hook to replace even partially
        # registered CFG receives with empty-block notifications to the producer.
        for request in requests:
            if request.kv_transfer_params is not None:
                request.kv_transfer_params["do_remote_prefill"] = True
                request.status = RequestStatus.FINISHED_ABORTED
                delay_free, _ = connector.request_finished(request, [])
                if delay_free:
                    raise RuntimeError("Connector could not cancel an undispatched KV load") from exc
        raise KVTransferRegistrationError(f"Could not register diffusion KV receive: {exc}") from exc
    return expected


def wait_for_kv_load(
    active_connector: ActiveKVConnector, scheduler_output: DiffusionSchedulerOutput, timeout: float
) -> KVConnectorOutput:
    connector = active_connector.kv_connector
    active_connector.pre_forward(scheduler_output)
    pending = set(scheduler_output.kv_transfer_request_ids)
    finished_ids = scheduler_output.kv_finished_request_ids
    if not pending:
        output = active_connector.post_forward(finished_ids)
        if output.invalid_block_ids:
            raise RuntimeError("Diffusion KV connector reported invalid remote pages")
        return output

    received, sent = set(), set()
    deadline = time.monotonic() + timeout
    while True:
        finished_sending, finished_recving = connector.get_finished(finished_ids)
        sent.update(finished_sending or ())
        received.update(finished_recving or ())
        pending.difference_update(received)
        if connector.get_block_ids_with_load_errors():
            raise RuntimeError("Diffusion KV connector reported invalid remote pages")
        if not pending:
            break
        if time.monotonic() >= deadline:
            # A timeout is not a cancellation acknowledgement. Return partial
            # completion; Scheduler quarantines the remaining destination pages.
            break
        time.sleep(0.001)
    output = active_connector.post_forward(finished_ids)
    output.finished_sending = sent | (output.finished_sending or set())
    output.finished_recving = received | (output.finished_recving or set())
    if output.invalid_block_ids:
        raise RuntimeError("Diffusion KV connector reported invalid remote pages")
    return output


def _validate_kv_transfer_config(config: KVTransferConfig) -> None:
    engine_id = config.engine_id
    if not isinstance(engine_id, str) or not engine_id.strip():
        raise ValueError("Diffusion native kv_transfer_config requires a non-empty engine_id")
    extra_config = config.kv_connector_extra_config or {}
    timeout = extra_config.get("transfer_timeout", 60.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Diffusion transfer_timeout must be a positive finite number of seconds")
    if config.kv_connector is not None and config.kv_role is None:
        raise ValueError("Diffusion native kv_transfer_config requires kv_role when kv_connector is set")
    if config.kv_role not in (None, "kv_consumer", "kv_producer", "kv_both"):
        raise ValueError(f"Unsupported KV connector role: {config.kv_role!r}")

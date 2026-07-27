"""Stateless request and shutdown helpers for :mod:`async_omni_engine`."""

from __future__ import annotations

import threading
from typing import Any

import janus
import torch
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.messages import EngineQueueMessage, ShutdownRequestMessage
from vllm_omni.engine.rpc_result_router import CorrelatedRpcClient
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.engine.stage_runtime import StageRuntime

logger = init_logger(__name__)

SHUTDOWN_ENQUEUE_TIMEOUT_S = 1.0
SHUTDOWN_JOIN_TIMEOUT_S = 30.0
_WEAK_SHUTDOWN_JOIN_TIMEOUT_S = 1.0


def inject_global_id(target: Any, request_id: str) -> None:
    """Inject ``global_request_id`` into a prompt's additional information."""
    if isinstance(target, dict):
        if "additional_information" not in target:
            target["additional_information"] = {}
        if target["additional_information"] is None:
            target["additional_information"] = {}
        if isinstance(target["additional_information"], dict):
            target["additional_information"]["global_request_id"] = [str(request_id)]


def upgrade_to_omni_request(
    request: EngineCoreRequest,
    raw_prompt: Any,
) -> EngineCoreRequest:
    """Restore omni-only fields omitted by the upstream input processor."""
    prompt_embeds = request.prompt_embeds
    additional_information = None
    wire_payload = None
    model_intermediate_buffer = None

    if isinstance(raw_prompt, dict):
        if prompt_embeds is None:
            raw_prompt_embeds = raw_prompt.get("prompt_embeds")
            if isinstance(raw_prompt_embeds, torch.Tensor):
                prompt_embeds = raw_prompt_embeds
        raw_info = raw_prompt.get("additional_information")
        raw_buffer = raw_prompt.get("model_intermediate_buffer")
        if isinstance(raw_info, dict):
            wire_payload = dict(raw_info)
        if isinstance(raw_buffer, dict):
            model_intermediate_buffer = raw_buffer
        additional_information = serialize_additional_information(
            wire_payload,
            log_prefix="AsyncOmniEngine",
        )

    if prompt_embeds is None and additional_information is None and model_intermediate_buffer is None:
        return request

    return OmniEngineCoreRequest.from_request(
        request,
        prompt_embeds=prompt_embeds,
        additional_information=additional_information,
        model_intermediate_buffer=model_intermediate_buffer,
    )


def apply_omni_final_stage_metadata(
    request: EngineCoreRequest,
    final_stage_id: int,
) -> EngineCoreRequest:
    """Tag a request so the AR scheduler can skip DiT KV for stage zero."""
    merged: dict[str, Any] = {}
    if isinstance(request, OmniEngineCoreRequest) and request.additional_information is not None:
        merged = deserialize_additional_information(request.additional_information)
    merged["omni_final_stage_id"] = final_stage_id
    payload = serialize_additional_information(merged)
    return OmniEngineCoreRequest.from_request(
        request,
        additional_information=payload,
    )


def weak_shutdown_async_omni_engine(
    orchestrator_thread: threading.Thread | None,
    request_queue: janus.Queue[EngineQueueMessage] | None,
    output_queue: janus.Queue[EngineQueueMessage] | None,
    rpc_output_queue: janus.Queue[EngineQueueMessage] | None,
    rpc_client: CorrelatedRpcClient | None,
) -> None:
    """Best-effort orchestrator cleanup for garbage-collection finalization."""
    request_queue_closed = False
    shutdown_enqueued = enqueue_orchestrator_shutdown(
        request_queue,
        timeout=SHUTDOWN_ENQUEUE_TIMEOUT_S,
    )
    if request_queue is not None and not shutdown_enqueued:
        try:
            request_queue.close()
            request_queue_closed = True
        except Exception:
            pass

    if rpc_client is not None:
        rpc_client.close()

    try:
        if orchestrator_thread is not None and orchestrator_thread.is_alive():
            orchestrator_thread.join(timeout=_WEAK_SHUTDOWN_JOIN_TIMEOUT_S)
    except Exception:
        pass

    for q in (request_queue, output_queue, rpc_output_queue):
        try:
            if q is not None and not (q is request_queue and request_queue_closed):
                q.close()
        except Exception:
            pass


def enqueue_orchestrator_shutdown(
    request_queue: janus.Queue[EngineQueueMessage] | None,
    *,
    timeout: float,
) -> bool:
    """Deliver shutdown without waiting forever behind a full request queue."""
    if request_queue is None:
        return False
    try:
        request_queue.sync_q.put(ShutdownRequestMessage(), timeout=timeout)
    except Exception:
        return False
    return True


def shutdown_runtime_after_orchestrator(
    orchestrator_thread: threading.Thread,
    runtime: StageRuntime,
) -> None:
    """Release a stage runtime once its orchestrator can no longer use it."""
    try:
        orchestrator_thread.join()
    except Exception:
        logger.exception("[AsyncOmniEngine] Failed while awaiting deferred Orchestrator shutdown")
    if orchestrator_thread.is_alive():
        logger.error("[AsyncOmniEngine] Orchestrator is still alive; StageRuntime cleanup remains deferred")
        return
    try:
        runtime.shutdown()
    except Exception:
        logger.exception("[AsyncOmniEngine] Failed to shutdown deferred StageRuntime")

"""
Orchestrator for vLLM-Omni multi-stage runtime.

Runs inside a background thread with its own asyncio event loop.
Owns all StageEngineCoreClient instances, input/output processors,
and handles stage-to-stage transfer logic.
"""

from __future__ import annotations

import asyncio
import copy
import time as _time
from dataclasses import dataclass, field
from typing import Any

import janus
import torch
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.distributed.omni_connectors.adapter import compute_talker_prompt_ids_length
from vllm_omni.engine import (
    OmniEngineCoreRequest,
)
from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.metrics.stats import StageStats
from vllm_omni.metrics.utils import count_tokens_from_outputs

logger = init_logger(__name__)


def build_engine_core_request_from_tokens(
    request_id: str,
    prompt: dict[str, Any],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    model_config: ModelConfig | None = None,
    resumable: bool = False,
    mm_features: list | None = None,
) -> OmniEngineCoreRequest:
    """Build an OmniEngineCoreRequest directly from an OmniTokensPrompt.

    Lightweight alternative to the full InputProcessor pipeline - skips
    tokenization, multimodal preprocessing, LoRA validation, and platform
    validation.  Intended for stage 1+ where the upstream stage has already
    produced token IDs and optional embeddings.
    """
    if arrival_time is None:
        arrival_time = _time.time()

    prompt_token_ids = prompt["prompt_token_ids"]

    # Clone params and set max_tokens if needed
    sampling_params = None
    pooling_params = None
    if isinstance(params, SamplingParams):
        sampling_params = params.clone()
        if sampling_params.max_tokens is None and model_config is not None:
            sampling_params.max_tokens = model_config.max_model_len - len(prompt_token_ids)
    else:
        pooling_params = params.clone()

    prompt_embeds: torch.Tensor | None = prompt.get("prompt_embeds")

    # Serialize additional_information if present
    additional_info_payload = serialize_additional_information(
        prompt.get("additional_information"),
        log_prefix=f"build_engine_core_request_from_tokens req={request_id}",
    )

    return OmniEngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        arrival_time=arrival_time,
        lora_request=getattr(params, "lora_request", None),
        cache_salt=None,
        data_parallel_rank=None,
        prompt_embeds=prompt_embeds,
        resumable=resumable,
        additional_information=additional_info_payload,
    )


# ============================================================
# Orchestrator internals (run inside the background thread)
# ============================================================


@dataclass
class OrchestratorRequestState:
    """Per-request bookkeeping inside the Orchestrator."""

    request_id: str
    prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1

    # Metrics: timestamp when request was submitted to each stage
    stage_submit_ts: dict[int, float] = field(default_factory=dict)
    mm_processor_kwargs: dict | None = None
    mm_features: list | None = None

    streaming: StreamingInputState = field(default_factory=lambda: StreamingInputState())


@dataclass
class StreamingInputState:
    # Flag of streaming input request
    enabled: bool = False
    # Flag of segment of streaming input finished
    segment_finished: bool = False
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None
    # Model/bridge-specific runtime states (e.g., thinker->talker)
    bridge_states: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Runs inside a background thread's asyncio event loop.

    Owns all StageEngineCoreClient instances, input/output processors,
    and handles stage-to-stage transfer logic.
    """

    def __init__(
        self,
        request_async_queue: janus.AsyncQueue[dict[str, Any]],
        output_async_queue: janus.AsyncQueue[dict[str, Any]],
        rpc_async_queue: janus.AsyncQueue[dict[str, Any]],
        stage_clients: list[Any],
        output_processors: list[Any],
        stage_vllm_configs: list[Any],
        *,
        async_chunk: bool = False,
        pd_config: dict[str, Any] | None = None,
    ) -> None:
        self.request_async_queue = request_async_queue
        self.output_async_queue = output_async_queue
        self.rpc_async_queue = rpc_async_queue

        self.num_stages = len(stage_clients)
        self.async_chunk = bool(async_chunk)

        self.stage_clients: list[Any] = stage_clients
        self.output_processors: list[Any] = output_processors
        self.stage_vllm_configs: list[Any] = stage_vllm_configs

        # PD disaggregation state
        self._pd_pair: tuple[int, int] | None = None
        self._pd_bootstrap_addr: str | None = None
        self._pd_prefill_engine_id: str | None = None
        self._pd_kv_params: dict[str, Any] = {}
        if pd_config is not None:
            self._pd_pair = pd_config.get("pd_pair")
            self._pd_bootstrap_addr = pd_config.get("bootstrap_addr")
            self._pd_prefill_engine_id = pd_config.get("prefill_engine_id")

        # Per-request state
        self.request_states: dict[str, OrchestratorRequestState] = {}

        # CFG companion tracking
        self._cfg_tracker = CfgCompanionTracker()

        # Per-stage metrics accumulators.
        self._batch_seq: list[int] = [0] * self.num_stages
        self._agg_total_tokens: list[int] = [0] * self.num_stages
        self._agg_total_gen_time_ms: list[float] = [0.0] * self.num_stages

        # Shutdown coordination
        self._shutdown_event = asyncio.Event()
        self._stages_shutdown = False
        self._fatal_error: str | None = None

    async def run(self) -> None:
        """Main entry point for the Orchestrator event loop."""
        logger.info("[Orchestrator] Starting event loop")

        request_task = asyncio.create_task(self._request_handler(), name="orchestrator-request-handler")
        output_task = asyncio.create_task(
            self._orchestration_output_handler(),
            name="orchestrator-stage-output-handler",
        )

        try:
            # Run both tasks concurrently; if either fails the other is cancelled.
            await asyncio.gather(request_task, output_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[Orchestrator] Fatal error in orchestrator tasks")
            raise
        finally:
            self._shutdown_event.set()
            for t in (request_task, output_task):
                if not t.done():
                    t.cancel()
            try:
                await asyncio.gather(request_task, output_task, return_exceptions=True)
            except Exception:
                pass

            # If a fatal error caused the shutdown, drain any pending
            # add_request messages that were never processed and broadcast
            # fatal error responses so callers are not left hanging.
            if self._fatal_error is not None:
                await self._drain_pending_requests_on_fatal()

            self._shutdown_stages()

            # Cancel any remaining tasks spawned by wait_for / gather so
            # the event loop can close cleanly without "pending task" errors.
            loop = asyncio.get_running_loop()
            pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task() and not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _request_handler(self) -> None:
        """Read messages from the main thread via request_async_queue."""
        while True:
            msg = await self.request_async_queue.get()
            msg_type = msg.get("type")

            if msg_type == "add_request":
                await self._handle_add_request(msg)
            elif msg_type == "streaming_update":
                await self._handle_streaming_update(msg)
            elif msg_type == "add_companion_request":
                await self._handle_add_companion(msg)
            elif msg_type == "abort":
                await self._handle_abort(msg)
            elif msg_type == "collective_rpc":
                await self._handle_collective_rpc(msg)
            elif msg_type == "shutdown":
                logger.info("[Orchestrator] Received shutdown signal")
                self._shutdown_event.set()
                self._shutdown_stages()
                break
            else:
                logger.warning(f"[Orchestrator] Unknown message type: {msg_type}")

    async def _orchestration_output_handler(self) -> None:
        """Poll all stages, handle transfers, send final outputs to main."""
        try:
            await self._orchestration_loop()
        except asyncio.CancelledError:
            logger.debug("[Orchestrator] _orchestration_output_handler cancelled")
            return

    async def _orchestration_loop(self) -> None:
        """Inner loop for _orchestration_output_handler (clean cancellation).

        Control flow: poll raw → process through output processor → route.
        """
        while not self._shutdown_event.is_set():
            idle = True
            for stage_id in range(self.num_stages):
                if self._shutdown_event.is_set():
                    return

                # 1) Diffusion stage: poll non-blocking queue
                # TODO (Peiqi): the output of diffusion stage is OmniRequestOutput,
                # which is different from EngineCoreOutputs (LLM stages). We may want to unify
                # the output format in the future to simplify the processing logic in Orchestrator.
                stage_client = self.stage_clients[stage_id]
                if stage_client.stage_type == "diffusion":
                    output = stage_client.get_diffusion_output_nowait()
                    if output is not None:
                        idle = False

                        if getattr(output, "error", None) is not None:
                            await self.output_async_queue.put(
                                {
                                    "type": "output",
                                    "request_id": output.request_id,
                                    "stage_id": stage_id,
                                    "engine_outputs": output,
                                    "metrics": None,
                                    "finished": True,
                                }
                            )
                            self.request_states.pop(output.request_id, None)
                            continue

                        req_state = self.request_states.get(output.request_id)
                        if req_state is not None:
                            if getattr(output, "error", None) is not None:
                                parent_id = self._companion_to_parent.get(output.request_id, output.request_id)
                                await self.output_async_queue.put(
                                    {
                                        "type": "error",
                                        "request_id": parent_id,
                                        "stage_id": stage_id,
                                        "error": output.error,
                                    }
                                )
                                role_map = self._companion_map.get(parent_id, {})
                                for cid in role_map.values():
                                    self.request_states.pop(cid, None)
                                self._cleanup_companion_state(parent_id)
                                self.request_states.pop(parent_id, None)
                                continue

                            stage_metrics = self._build_stage_metrics(stage_id, output.request_id, [output], req_state)
                            await self._route_output(stage_id, output, req_state, stage_metrics)
                    continue

                # 1) Poll raw outputs from the stage
                try:
                    raw_outputs = await asyncio.wait_for(self._poll_stage_raw(stage_id), timeout=0.001)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
                except EngineDeadError as e:
                    logger.error(
                        "[Orchestrator] Stage-%s is dead: %s",
                        stage_id,
                        e,
                    )
                    self._fatal_error = str(e)
                    for req_id, req_state in list(self.request_states.items()):
                        if stage_id in req_state.stage_submit_ts:
                            await self.output_async_queue.put(
                                {
                                    "type": "error",
                                    "error": str(e),
                                    "fatal": True,
                                    "request_id": req_id,
                                }
                            )
                            self.request_states.pop(req_id, None)
                    self._shutdown_event.set()
                    raise
                except Exception:
                    if self._shutdown_event.is_set():
                        return
                    logger.exception(
                        "[Orchestrator] _poll_stage_raw failed for stage-%s",
                        stage_id,
                    )
                    raise

                if raw_outputs is None:
                    continue
                idle = False

                # Handle prefill-finished KV-ready signals before finished outputs.
                await self._handle_kv_ready_raw_outputs(stage_id, raw_outputs)

                # 2) Process raw outputs through the output processor
                request_outputs = await self._process_stage_outputs(stage_id, raw_outputs)

                # 3) Route each processed output
                for output in request_outputs:
                    req_state = self.request_states.get(output.request_id)
                    if req_state is None:
                        logger.warning(
                            "[Orchestrator] Dropping output for unknown req %s at stage-%s (known reqs: %s)",
                            output.request_id,
                            stage_id,
                            list(self.request_states.keys()),
                        )
                        continue
                    stage_metrics = None
                    if output.finished:
                        stage_metrics = self._build_stage_metrics(
                            stage_id,
                            output.request_id,
                            [output],
                            req_state,
                        )
                    await self._route_output(stage_id, output, req_state, stage_metrics)

            if idle:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)

    async def _route_output(
        self,
        stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        stage_metrics: Any,
    ) -> None:
        """Route a processed output: send to main thread and/or forward to next stage."""
        req_id = output.request_id
        finished = output.finished
        submit_ts = req_state.stage_submit_ts.get(stage_id)
        stage_client = self.stage_clients[stage_id]

        # CFG companion handling: companions don't produce user-visible output
        # and don't forward to the next stage directly.
        if finished and self._cfg_tracker.is_companion(req_id):
            await self._handle_cfg_companion_ready(req_id)
            self.request_states.pop(req_id, None)
            return

        if stage_client.final_output:
            await self.output_async_queue.put(
                {
                    "type": "output",
                    "request_id": req_id,
                    "stage_id": stage_id,
                    "engine_outputs": output,
                    "metrics": stage_metrics,
                    "finished": finished and stage_id == req_state.final_stage_id,
                    "stage_submit_ts": submit_ts,
                }
            )
        elif stage_metrics is not None:
            await self.output_async_queue.put(
                {
                    "type": "stage_metrics",
                    "request_id": req_id,
                    "stage_id": stage_id,
                    "metrics": stage_metrics,
                    "stage_submit_ts": submit_ts,
                }
            )

        # PD disaggregation: extract KV transfer params from prefill stage output
        if self._pd_pair is not None and finished and stage_id == self._pd_pair[0]:
            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is not None:
                self._pd_kv_params[req_id] = kv_params if isinstance(kv_params, dict) else dict(kv_params)
                logger.debug(
                    "[Orchestrator][PD] stored kv_transfer_params for req=%s (keys=%s)",
                    req_id,
                    list(self._pd_kv_params[req_id].keys()),
                )
            else:
                logger.warning(
                    "[Orchestrator][PD] prefill stage output for req=%s has no kv_transfer_params; "
                    "KV transfer may fail. Ensure apply_mooncake_connector_patch() was called.",
                    req_id,
                )

        if (
            (finished or (req_state.streaming.enabled and req_state.streaming.segment_finished))
            and stage_id < req_state.final_stage_id
            and not self.async_chunk
            and (not self._next_stage_already_submitted(stage_id, req_state) or req_state.streaming.enabled)
        ):
            if (
                finished
                and self._cfg_tracker.has_companions(req_id)
                and not self._cfg_tracker.all_companions_done(req_id)
            ):
                self._cfg_tracker.defer_parent(req_id, output, stage_id)
            else:
                await self._forward_to_next_stage(
                    req_id,
                    stage_id,
                    output,
                    req_state,
                    is_streaming_session=req_state.streaming.enabled,
                    is_final_update=False,
                )
                if req_state.streaming.enabled and finished:
                    # For streaming sessions, send the terminal (resumable=False) update only on a finish
                    await self._forward_to_next_stage(
                        req_id,
                        stage_id,
                        output,
                        req_state,
                        is_streaming_session=True,
                        is_final_update=True,
                    )

        if finished and stage_id == req_state.final_stage_id:
            # PD: clean up any lingering KV params for this request
            self._pd_kv_params.pop(req_id, None)
            self._cfg_tracker.cleanup_parent(req_id)
            self.request_states.pop(req_id, None)

    def _next_stage_already_submitted(self, stage_id: int, req_state: OrchestratorRequestState) -> bool:
        return (stage_id + 1) in req_state.stage_submit_ts

    async def _handle_cfg_companion_ready(self, req_id: str) -> None:
        """Mark a CFG companion as done; if all companions are done, flush deferred parent."""
        parent_id = self._cfg_tracker.on_companion_completed(req_id)
        if parent_id is None:
            return
        deferred = self._cfg_tracker.pop_pending_parent(parent_id)
        if deferred is None:
            return
        parent_state = self.request_states.get(parent_id)
        if parent_state is not None and not self._next_stage_already_submitted(deferred["stage_id"], parent_state):
            await self._forward_to_next_stage(
                parent_id,
                deferred["stage_id"],
                deferred["engine_outputs"],
                parent_state,
            )

    async def _handle_kv_ready_raw_outputs(self, stage_id: int, raw_outputs: EngineCoreOutputs) -> None:
        """Forward split requests once stage-0 KV is ready, not only when decode fully finishes."""
        if self.async_chunk:
            return
        for raw_output in raw_outputs.outputs:
            kv_params = getattr(raw_output, "kv_transfer_params", None)
            if not (isinstance(kv_params, dict) and kv_params.get("kv_ready")):
                continue
            req_id = raw_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                continue
            if self._cfg_tracker.is_companion(req_id):
                await self._handle_cfg_companion_ready(req_id)
                continue
            if stage_id >= req_state.final_stage_id:
                continue
            if self._next_stage_already_submitted(stage_id, req_state):
                continue
            if self._cfg_tracker.has_companions(req_id) and not self._cfg_tracker.all_companions_done(req_id):
                self._cfg_tracker.defer_parent(req_id, raw_output, stage_id)
            else:
                await self._forward_to_next_stage(req_id, stage_id, raw_output, req_state)

    def _build_pd_decode_params(self, req_id: str, sp: Any) -> Any:
        """Build decode-side sampling params with KV transfer params for PD routing.

        Clones the sampling params and injects kv_transfer_params that tell the
        decode engine where to pull the KV cache from (prefill engine's bootstrap addr).
        """
        sp = sp.clone()
        if sp.extra_args is None:
            sp.extra_args = {}

        # Get KV params captured from the prefill output (must include remote_request_id).
        kv_prefill_params = self._pd_kv_params.pop(req_id, None)
        if not kv_prefill_params or "remote_request_id" not in kv_prefill_params:
            raise RuntimeError(
                f"[Orchestrator][PD] Missing prefill kv_transfer_params.remote_request_id for req={req_id}"
            )

        decode_kv_params: dict[str, Any] = {
            "transfer_id": f"xfer-{req_id}",
        }

        if self._pd_bootstrap_addr:
            decode_kv_params["remote_bootstrap_addr"] = self._pd_bootstrap_addr

        if self._pd_prefill_engine_id:
            decode_kv_params["remote_engine_id"] = self._pd_prefill_engine_id

        # Overlay params from prefill side (includes remote_request_id set by monkey patch).
        decode_kv_params.update(kv_prefill_params)

        # Ensure these flags are set correctly after any overlay.
        decode_kv_params["do_remote_prefill"] = True
        decode_kv_params["do_remote_decode"] = False
        if not decode_kv_params.get("transfer_id"):
            decode_kv_params["transfer_id"] = f"xfer-{req_id}"

        sp.extra_args["kv_transfer_params"] = decode_kv_params

        logger.debug(
            "[Orchestrator][PD] decode kv_transfer_params for req=%s: %s",
            req_id,
            decode_kv_params,
        )
        return sp

    def _build_stage_metrics(
        self,
        stage_id: int,
        req_id: str,
        request_outputs: list[RequestOutput],
        req_state: OrchestratorRequestState,
    ) -> StageRequestMetrics:
        """Build StageRequestMetrics for a finished request at a stage.

        Reuses StageRequestMetrics so OrchestratorMetrics and downstream
        metric handlers can consume a stable schema.
        """
        now = _time.time()
        submit_ts = req_state.stage_submit_ts.get(stage_id, now)
        stage_gen_time_ms = (now - submit_ts) * 1000.0

        num_tokens_out = count_tokens_from_outputs(request_outputs)
        num_tokens_in = 0
        if stage_id == 0:
            for ro in request_outputs:
                ptids = getattr(ro, "prompt_token_ids", None)
                if ptids is not None:
                    num_tokens_in += len(ptids)

        # Monotonic batch counter per stage.
        self._batch_seq[stage_id] += 1
        batch_id = self._batch_seq[stage_id]

        # Accumulate for running-average stage_stats
        self._agg_total_tokens[stage_id] += num_tokens_out
        self._agg_total_gen_time_ms[stage_id] += stage_gen_time_ms

        return StageRequestMetrics(
            num_tokens_in=num_tokens_in,
            num_tokens_out=num_tokens_out,
            stage_gen_time_ms=stage_gen_time_ms,
            batch_id=batch_id,
            batch_size=1,
            rx_decode_time_ms=0.0,
            rx_transfer_bytes=0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(
                total_token=self._agg_total_tokens[stage_id],
                total_gen_time_ms=self._agg_total_gen_time_ms[stage_id],
            ),
        )

    def _build_kv_sender_info(self, sender_stage_ids: list[int]) -> dict[int, dict[str, Any]] | None:
        """Build per-request sender info for diffusion KV-transfer receivers."""
        sender_infos: dict[int, dict[str, Any]] = {}
        for sender_stage_id in dict.fromkeys(sender_stage_ids):
            if sender_stage_id < 0 or sender_stage_id >= self.num_stages:
                continue

            sender_stage = self.stage_clients[sender_stage_id]
            get_sender_info = getattr(sender_stage, "get_kv_sender_info", None)
            if not callable(get_sender_info):
                continue

            sender_info = get_sender_info()
            if not sender_info:
                logger.warning(
                    "[Orchestrator] Stage-%s has no KV sender info available",
                    sender_stage_id,
                )
                continue

            sender_infos[sender_stage_id] = sender_info

        return sender_infos or None

    async def _forward_to_next_stage(
        self,
        req_id: str,
        stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        is_streaming_session: bool = False,
        is_final_update: bool = False,
    ) -> None:
        """Forward output from current stage to the next stage.

        Handles the full pipeline: set outputs on current stage, compute
        next-stage inputs, build lightweight requests, and submit them.
        """
        next_stage_id = stage_id + 1
        next_client = self.stage_clients[next_stage_id]
        params = req_state.sampling_params_list[next_stage_id]
        next_stage_resumable = is_streaming_session and not is_final_update

        if next_client.stage_type == "diffusion":
            self.stage_clients[stage_id].set_engine_outputs([output])
            if next_client.custom_process_input_func is not None:
                _t_ar2d = _time.perf_counter()
                diffusion_prompt = next_client.custom_process_input_func(
                    self.stage_clients,
                    next_client.engine_input_source,
                    req_state.prompt,
                    False,
                )
                _dt_ar2d = (_time.perf_counter() - _t_ar2d) * 1000
                logger.info(
                    "[Orchestrator] ar2diffusion req=%s wall_time=%.3fms stage=%d->%d",
                    req_id,
                    _dt_ar2d,
                    stage_id,
                    next_stage_id,
                )
                if isinstance(diffusion_prompt, list):
                    diffusion_prompt = diffusion_prompt[0]
            else:
                diffusion_prompt = req_state.prompt

            params = self._cfg_tracker.attach_cfg_request_ids(req_id, params)

            source_stage_ids = list(getattr(next_client, "engine_input_source", None) or [stage_id])
            kv_sender_info = self._build_kv_sender_info(sender_stage_ids=source_stage_ids)
            if isinstance(diffusion_prompt, list):
                await next_client.add_batch_request_async(
                    req_id,
                    diffusion_prompt,
                    params,
                    kv_sender_info=kv_sender_info,
                )
            else:
                await next_client.add_request_async(
                    req_id,
                    diffusion_prompt,
                    params,
                    kv_sender_info=kv_sender_info,
                )
            req_state.stage_submit_ts[next_stage_id] = _time.time()
            return

        # PD disaggregation: prefill → decode routing uses original prompt + KV transfer params
        if self._pd_pair is not None and (stage_id, next_stage_id) == self._pd_pair:
            # Save prefill stage outputs so thinker2talker can merge embeddings later
            self.stage_clients[stage_id].set_engine_outputs([output])

            params = self._build_pd_decode_params(req_id, params)

            # Use the original user prompt for the decode stage (not processed embeddings)
            original_prompt = req_state.prompt
            raw_decode_inputs = [original_prompt] if not isinstance(original_prompt, list) else original_prompt

            decode_inputs: list[dict[str, Any]] = []
            for decode_input in raw_decode_inputs:
                if isinstance(decode_input, dict):
                    decode_inputs.append(decode_input)
                    continue
                prompt_token_ids = getattr(decode_input, "prompt_token_ids", None)
                if prompt_token_ids is None:
                    raise TypeError(
                        "[Orchestrator][PD] decode input must be dict or have prompt_token_ids, "
                        f"got {type(decode_input).__name__} for req={req_id}"
                    )
                decode_inputs.append({"prompt_token_ids": list(prompt_token_ids)})

            for decode_input in decode_inputs:
                request = build_engine_core_request_from_tokens(
                    request_id=req_id,
                    prompt=decode_input,
                    params=params,
                    model_config=self.stage_vllm_configs[next_stage_id].model_config,
                    mm_features=req_state.mm_features,  # Pass mm_features for M-RoPE
                )
                request.external_req_id = request.request_id

                self.output_processors[next_stage_id].add_request(
                    request=request,
                    prompt=None,
                    parent_req=None,
                    request_index=0,
                    queue=None,
                )
                await next_client.add_request_async(request)

            req_state.stage_submit_ts[next_stage_id] = _time.time()
            return

        self.stage_clients[stage_id].set_engine_outputs([output])

        # Process inputs for next stage
        try:
            next_inputs = next_client.process_engine_inputs(
                stage_list=self.stage_clients,
                prompt=req_state.prompt,
                streaming_context=(req_state.streaming if req_state.streaming.enabled else None),
            )
        except Exception:
            logger.exception(
                "[Orchestrator] req=%s process_engine_inputs FAILED for stage-%s",
                req_id,
                next_stage_id,
            )
            raise

        # Build and submit requests for each input
        for next_input in next_inputs:
            # Only AR thinker stages consume encoder mm_features; downstream
            # (talker/code2wav/…) must not see them (avoids encoder-cache misses).
            _ms = getattr(next_client, "model_stage", None)
            _mm_features = req_state.mm_features if _ms == "thinker" else None
            request = build_engine_core_request_from_tokens(
                request_id=req_id,
                prompt=next_input,
                params=params,
                model_config=self.stage_vllm_configs[next_stage_id].model_config,
                mm_features=_mm_features,
                resumable=next_stage_resumable,
            )

            # TODO: Here we directly use the req id to assign.
            request.external_req_id = request.request_id

            self.output_processors[next_stage_id].add_request(
                request=request,
                prompt=None,
                parent_req=None,
                request_index=0,
                queue=None,
            )

            await next_client.add_request_async(request)

        # Record submit timestamp for the next stage
        req_state.stage_submit_ts[next_stage_id] = _time.time()

    async def _poll_stage_raw(self, stage_id: int) -> EngineCoreOutputs | None:
        """Pull raw EngineCoreOutputs from a stage client without processing.

        Returns the raw outputs object, or None when there is nothing
        to consume.
        """
        outputs = await self.stage_clients[stage_id].get_output_async()
        if not outputs.outputs:
            return None
        return outputs

    async def _process_stage_outputs(self, stage_id: int, raw_outputs: EngineCoreOutputs) -> list[RequestOutput]:
        """Run the output processor on raw outputs, returning RequestOutputs.

        Also handles abort forwarding and scheduler stats updates.
        """
        processor = self.output_processors[stage_id]

        processed = processor.process_outputs(
            raw_outputs.outputs,
            raw_outputs.timestamp,
            None,
        )
        for eco in raw_outputs.outputs:
            if not hasattr(eco, "request_id"):
                continue
            req_state = self.request_states.get(eco.request_id)
            if req_state:
                req_state.streaming.segment_finished = eco.is_segment_finished
                req_state.streaming.new_prompt_len_snapshot = eco.new_prompt_len_snapshot

        if processed.reqs_to_abort:
            await self.stage_clients[stage_id].abort_requests_async(processed.reqs_to_abort)

        if raw_outputs.scheduler_stats is not None:
            processor.update_scheduler_stats(raw_outputs.scheduler_stats)

        return processed.request_outputs

    async def _handle_add_request(self, msg: dict[str, Any]) -> None:
        """Handle an add_request message from the main thread."""
        stage_id = 0
        request_id = msg["request_id"]
        prompt = msg["prompt"]
        original_prompt = msg.get("original_prompt", prompt)
        sampling_params_list = msg["sampling_params_list"]
        if not sampling_params_list:
            raise ValueError(f"Missing sampling params for stage 0. Got {len(sampling_params_list)} stage params.")
        params = sampling_params_list[0]
        final_stage_id = msg["final_stage_id"]

        logger.info(
            "[Orchestrator] _handle_add_request: stage=%s req=%s "
            "prompt_type=%s original_prompt_type=%s final_stage=%s "
            "num_sampling_params=%d",
            stage_id,
            request_id,
            type(prompt).__name__,
            type(original_prompt).__name__,
            final_stage_id,
            len(sampling_params_list),
        )

        # Track request state - use original_prompt so downstream stages
        # (e.g. thinker2talker) can access the raw dict with multi_modal_data.
        request = prompt
        is_streaming = bool(getattr(request, "resumable", False))
        req_state = OrchestratorRequestState(
            request_id=request_id,
            prompt=original_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            mm_features=getattr(prompt, "mm_features", None),  # Save mm_features for PD
        )
        req_state.streaming.enabled = is_streaming
        req_state.stage_submit_ts[stage_id] = _time.time()
        self.request_states[request_id] = req_state

        # Stage-0 prompt is already a fully-formed OmniEngineCoreRequest
        # (pre-processed by AsyncOmniEngine.add_request, output processor
        # already registered there) - submit directly.
        stage_client = self.stage_clients[stage_id]
        if stage_client.stage_type == "diffusion":
            if isinstance(prompt, list):
                await stage_client.add_batch_request_async(
                    request_id,
                    prompt,
                    params,
                )
            else:
                await stage_client.add_request_async(request_id, prompt, params)
        else:
            await stage_client.add_request_async(request)

        if self.async_chunk and stage_id == 0 and final_stage_id > 0:
            await self._prewarm_async_chunk_stages(request_id, request, req_state)

    async def _handle_streaming_update(self, msg: dict[str, Any]) -> None:
        """Handle a streaming_update message for an existing request."""
        stage_id = 0
        request_id = msg["request_id"]
        request = msg["prompt"]

        req_state = self.request_states.get(request_id)
        if req_state is None:
            logger.warning(
                "[Orchestrator] streaming_update for unknown req=%s, falling back to add_request",
                request_id,
            )
            fallback_msg = dict(msg)
            fallback_msg["type"] = "add_request"
            await self._handle_add_request(fallback_msg)
            return

        if "sampling_params_list" in msg and msg["sampling_params_list"]:
            req_state.sampling_params_list = msg["sampling_params_list"]
        req_state.streaming.enabled = True

        req_state.stage_submit_ts[stage_id] = _time.time()
        stage_client = self.stage_clients[stage_id]
        if stage_client.stage_type == "diffusion":
            params = req_state.sampling_params_list[stage_id]
            await stage_client.add_request_async(request_id, request, params)
        else:
            await stage_client.add_request_async(request)

    async def _prewarm_async_chunk_stages(
        self,
        request_id: str,
        stage0_request: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Pre-submit downstream stages for async-chunk mode.

        In async-chunk mode, stages exchange data through connectors/chunk adapters,
        so downstream stages should be armed once at request start instead of waiting
        for stage-finished forwarding.
        """
        if req_state.final_stage_id <= 0:
            return

        prompt_token_ids = getattr(stage0_request, "prompt_token_ids", None)
        if prompt_token_ids is None:
            logger.warning(
                "[Orchestrator] async_chunk prewarm skipped for req=%s: stage0 prompt_token_ids missing",
                request_id,
            )
            return

        # Pre-arm stage-1+ with placeholder prompt IDs.
        try:
            next_prompt_len = max(1, compute_talker_prompt_ids_length(prompt_token_ids))
        except Exception:
            next_prompt_len = max(1, len(prompt_token_ids))
        original_prompt = req_state.prompt
        if isinstance(original_prompt, dict):
            base_input = copy.deepcopy(original_prompt)
        else:
            base_input = {}
        base_input["prompt_token_ids"] = [0] * next_prompt_len
        base_input["multi_modal_data"] = None
        base_input["mm_processor_kwargs"] = None

        for next_stage_id in range(1, req_state.final_stage_id + 1):
            next_client = self.stage_clients[next_stage_id]
            params = req_state.sampling_params_list[next_stage_id]

            if next_client.stage_type == "diffusion":
                source_stage_ids = list(getattr(next_client, "engine_input_source", None) or [next_stage_id - 1])
                kv_sender_info = self._build_kv_sender_info(sender_stage_ids=source_stage_ids)
                await next_client.add_request_async(
                    request_id,
                    req_state.prompt,
                    params,
                    kv_sender_info=kv_sender_info,
                )
                req_state.stage_submit_ts[next_stage_id] = _time.time()
                continue

            request = build_engine_core_request_from_tokens(
                request_id=request_id,
                prompt=base_input,
                params=params,
                model_config=self.stage_vllm_configs[next_stage_id].model_config,
            )
            request.external_req_id = request.request_id

            self.output_processors[next_stage_id].add_request(
                request=request,
                prompt=None,
                parent_req=None,
                request_index=0,
                queue=None,
            )
            await next_client.add_request_async(request)
            req_state.stage_submit_ts[next_stage_id] = _time.time()

    async def _handle_add_companion(self, msg: dict[str, Any]) -> None:
        """Handle an add_companion_request message: submit companion to stage 0."""
        companion_id = msg["companion_id"]
        parent_id = msg["parent_id"]
        role = msg["role"]
        companion_prompt = msg["prompt"]
        sampling_params_list = msg["sampling_params_list"]

        self._cfg_tracker.register_companion(parent_id, role, companion_id)

        companion_state = OrchestratorRequestState(
            request_id=companion_id,
            prompt=companion_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=0,
        )
        companion_state.stage_submit_ts[0] = _time.time()
        self.request_states[companion_id] = companion_state

        request = companion_prompt  # Already a processed OmniEngineCoreRequest
        stage_client = self.stage_clients[0]
        await stage_client.add_request_async(request)

        logger.info(
            "[Orchestrator] CFG companion submitted: %s (role=%s, parent=%s)",
            companion_id,
            role,
            parent_id,
        )

    async def _handle_abort(self, msg: dict[str, Any]) -> None:
        """Handle an abort message from the main thread."""
        request_ids = msg["request_ids"]
        all_ids_to_abort = self._cfg_tracker.abort_parents(request_ids)
        for stage_id in range(self.num_stages):
            await self.stage_clients[stage_id].abort_requests_async(all_ids_to_abort)
        for req_id in all_ids_to_abort:
            self.request_states.pop(req_id, None)
        logger.info("[Orchestrator] Aborted request(s) %s", request_ids)

    async def _handle_collective_rpc(self, msg: dict[str, Any]) -> None:
        """Handle a control-plane RPC request from the main thread.

        TODO(AsyncOmni): parallelize stage dispatch if control latency becomes
        noticeable. The current sequential fanout keeps the first version simple
        and deterministic.
        """
        rpc_id = msg["rpc_id"]
        method = msg["method"]
        timeout = msg.get("timeout")
        args = tuple(msg.get("args", ()))
        kwargs = dict(msg.get("kwargs") or {})
        requested_stage_ids = msg.get("stage_ids")
        stage_ids = list(range(self.num_stages)) if requested_stage_ids is None else list(requested_stage_ids)

        results: list[Any] = []
        for stage_id in stage_ids:
            if stage_id < 0 or stage_id >= self.num_stages:
                results.append(
                    {
                        "supported": False,
                        "todo": True,
                        "error": f"Invalid stage id {stage_id}",
                    }
                )
                continue

            stage_client = self.stage_clients[stage_id]
            try:
                if hasattr(stage_client, "collective_rpc_async"):
                    stage_result = await stage_client.collective_rpc_async(
                        method=method,
                        timeout=timeout,
                        args=args,
                        kwargs=kwargs,
                    )
                else:
                    stage_result = {
                        "supported": False,
                        "todo": True,
                        "reason": (f"{stage_client.__class__.__name__}.collective_rpc_async is not implemented yet"),
                    }
            except Exception as exc:
                logger.exception(
                    "[Orchestrator] collective_rpc failed: stage=%s method=%s",
                    stage_id,
                    method,
                )
                stage_result = {
                    "supported": False,
                    "error": str(exc),
                }

            results.append(stage_result)

        await self.rpc_async_queue.put(
            {
                "type": "collective_rpc_result",
                "rpc_id": rpc_id,
                "method": method,
                "stage_ids": stage_ids,
                "results": results,
            }
        )

    async def _drain_pending_requests_on_fatal(self) -> None:
        """Drain the request queue and broadcast fatal errors for any
        pending add_request messages that were never processed.

        Called from the ``run()`` finally block when a fatal error
        (e.g. ``EngineDeadError``) caused the orchestrator to shut down
        before the request handler could process all queued messages.
        Also broadcasts for any already-tracked requests still in
        ``request_states`` that were not yet notified.
        """
        assert self._fatal_error is not None

        notified: set[str] = set()

        # 1) Drain pending messages from the request queue.
        while True:
            try:
                msg = self.request_async_queue.get_nowait()
            except Exception:
                break
            if msg.get("type") == "add_request":
                req_id = msg["request_id"]
                await self.output_async_queue.put(
                    {
                        "type": "error",
                        "error": self._fatal_error,
                        "fatal": True,
                        "request_id": req_id,
                    }
                )
                notified.add(req_id)

        # 2) Broadcast for any tracked requests not already notified
        #    (e.g. request was registered but the EngineDeadError handler
        #    missed it because it wasn't submitted to the dead stage yet).
        for req_id in list(self.request_states):
            if req_id not in notified:
                await self.output_async_queue.put(
                    {
                        "type": "error",
                        "error": self._fatal_error,
                        "fatal": True,
                        "request_id": req_id,
                    }
                )
            self.request_states.pop(req_id, None)

    def _shutdown_stages(self) -> None:
        """Shutdown all stage clients."""
        if self._stages_shutdown:
            return

        self._stages_shutdown = True
        logger.info("[Orchestrator] Shutting down all stages")
        for stage_id, stage_client in enumerate(self.stage_clients):
            try:
                stage_client.shutdown()
                logger.info(f"[Orchestrator] Stage {stage_id} shut down")
            except Exception as e:
                logger.warning(f"[Orchestrator] Failed to shutdown stage {stage_id}: {e}")

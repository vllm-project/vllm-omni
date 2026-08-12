"""
Orchestrator for vLLM-Omni multi-stage runtime.

Runs inside a background thread with its own asyncio event loop.
Owns logical request progression across stage pools and handles
stage-to-stage transfer logic.

Distributed membership (replica attach/detach, hub monitoring) is
handled by :class:`MembershipController`, which is injected optionally.
"""

from __future__ import annotations

import asyncio
import time as _time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import janus
import torch
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.metrics.stats import IterationStats

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.distributed.omni_connectors.utils.config import stage_receives_chunks
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.membership_controller import MembershipController
from vllm_omni.engine.messages import (
    AbortRequestMessage,
    AddCompanionRequestMessage,
    CollectiveRPCRequestMessage,
    CollectiveRPCResultMessage,
    EngineQueueMessage,
    ErrorMessage,
    InteractionMessage,
    OutputMessage,
    RegisterRemoteReplicaMessage,
    ShutdownRequestMessage,
    StageMetricsMessage,
    StageSubmissionMessage,
    UnregisterRemoteReplicaMessage,
)
from vllm_omni.engine.orchestrator_monitor import create_orch_monitor, replica_key
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.engine.stage_pool import StagePool, StageUnavailableError
from vllm_omni.metrics.prometheus import OmniRequestCounter
from vllm_omni.metrics.stat_logger import OmniPrometheusStatLogger
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_omni.experimental.fullduplex.engine.contracts import (
        DuplexControlPlanePort,
        DuplexOutputContext,
        DuplexOutputDecision,
        DuplexRequestIdentity,
        DuplexRuntimeExtension,
        DuplexStageRequestContext,
        DuplexStageSubmission,
        DuplexStageSubmissionResult,
    )
    from vllm_omni.experimental.fullduplex.engine.duplex_session import (
        DuplexSessionRuntimeManager,
        DuplexSessionRuntimeState,
    )
    from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence


def _build_terminal_empty_output(
    request_id: str,
    *,
    final_output_type: str | None,
    audio_sample_rate: int = 24000,
) -> RequestOutput:
    """Build a terminal empty output when no downstream stage input exists."""
    completion = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    if final_output_type == "audio":
        completion.multimodal_output = {
            "audio": torch.zeros((0,), dtype=torch.float32),
            "sr": audio_sample_rate,
        }
    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )


def build_engine_core_request_from_tokens(
    request_id: str,
    prompt: dict[str, Any],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    model_config: ModelConfig | None = None,
    resumable: bool = False,
    mm_features: list | None = None,
) -> OmniEngineCoreRequest:
    """Build an OmniEngineCoreRequest directly from an OmniTokensPrompt."""
    if arrival_time is None:
        arrival_time = _time.time()

    prompt_token_ids = prompt["prompt_token_ids"]

    sampling_params = None
    pooling_params = None
    if isinstance(params, SamplingParams):
        sampling_params = params.clone()
        if sampling_params.max_tokens is None and model_config is not None:
            sampling_params.max_tokens = model_config.max_model_len - len(prompt_token_ids)
    else:
        pooling_params = params.clone()

    prompt_embeds: torch.Tensor | None = prompt.get("prompt_embeds")
    raw_additional_information = prompt.get("additional_information")
    model_intermediate_buffer = prompt.get("model_intermediate_buffer")
    wire_payload: dict[str, Any] | None = None
    if isinstance(raw_additional_information, dict):
        wire_payload = dict(raw_additional_information)
    additional_info_payload = serialize_additional_information(
        wire_payload,
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
        cache_salt=prompt.get("cache_salt"),
        data_parallel_rank=None,
        prompt_embeds=prompt_embeds,
        resumable=resumable,
        additional_information=additional_info_payload,
        model_intermediate_buffer=model_intermediate_buffer if isinstance(model_intermediate_buffer, dict) else None,
    )


@dataclass
class OrchestratorRequestState:
    """Per-request bookkeeping inside the Orchestrator."""

    request_id: str
    prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1
    final_output_stage_ids: set[int] = field(default_factory=set)
    finished_final_output_stage_ids: set[int] = field(default_factory=set)

    # Wall-clock timestamp when the client-facing engine request was accepted.
    request_timestamp: float = 0.0

    # Metrics: timestamp when request was submitted to each stage.
    stage_submit_ts: dict[int, float] = field(default_factory=dict)
    mm_processor_kwargs: dict | None = None
    mm_features: list | None = None
    pd_prefill_multimodal_output: dict[str, Any] | None = None

    streaming: StreamingInputState = field(default_factory=lambda: StreamingInputState())

    # Per-request pipeline timing accumulator (milliseconds)
    pipeline_timings: dict[str, float] = field(default_factory=dict)
    duplex_identity: DuplexRequestIdentity | None = None
    duplex_stage_fences: dict[int, DuplexFence] = field(default_factory=dict)
    duplex_config_generation: int = -1
    running_counter_registered: bool = False


@dataclass
class StreamingInputState:
    # Flag of streaming input request
    enabled: bool = False
    # Flag of segment of streaming input finished
    segment_finished: bool = False
    # Tokens from the current raw segment boundary. The vLLM output processor
    # does not guarantee that EngineCoreOutput.new_token_ids survives on the
    # processed RequestOutput used by the routing layer.
    segment_token_ids: list[int] = field(default_factory=list)
    segment_output_metadata: dict[str, Any] = field(default_factory=dict)
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None
    # Model/bridge-specific runtime states (e.g., thinker->talker)
    bridge_states: dict[str, Any] = field(default_factory=dict)
    # Synchronous stage-transition capability installed by the orchestrator
    # while the downstream input processor consumes upstream token output.
    source_token_decoder: Callable[..., str] | None = None


class _OrchestratorDuplexStagePort:
    """Adapts generic stage pools to the model-neutral duplex control plane."""

    def __init__(
        self,
        *,
        stage_pools: list[StagePool],
        request_states: dict[str, OrchestratorRequestState],
        running_counter: OmniRequestCounter | None,
        cleanup_request_ids: Callable[..., Any],
        async_chunk: bool,
        prewarm_async_chunk_stages: Callable[
            [str, Any, OrchestratorRequestState],
            Awaitable[None],
        ],
    ) -> None:
        self._stage_pools = stage_pools
        self._request_states = request_states
        self._running_counter = running_counter
        self._cleanup_request_ids = cleanup_request_ids
        self._async_chunk = async_chunk
        self._prewarm_async_chunk_stages = prewarm_async_chunk_stages

    @property
    def stage_count(self) -> int:
        return len(self._stage_pools)

    def sampling_defaults(self) -> tuple[object, ...]:
        return tuple(pool.stage_client.default_sampling_params for pool in self._stage_pools)

    @staticmethod
    def _sync_bridge_state(
        request_state: OrchestratorRequestState,
        context: DuplexStageRequestContext,
    ) -> None:
        duplex_state = request_state.streaming.bridge_states.setdefault("duplex", {})
        if not isinstance(duplex_state, dict):
            duplex_state = {}
            request_state.streaming.bridge_states["duplex"] = duplex_state
        previous_epoch = duplex_state.get("epoch")
        if not isinstance(duplex_state.get("model_turn_id"), int) or previous_epoch != context.fence.epoch:
            duplex_state["model_turn_id"] = context.fence.turn_id
        duplex_state.update(
            {
                "session_id": context.session_id,
                "fence": context.fence,
                "incarnation": context.fence.incarnation,
                "epoch": context.fence.epoch,
                "turn_id": context.fence.turn_id,
                "response_seq": context.fence.response_seq,
                "session_config": dict(context.session_config),
                "runtime_config": dict(context.runtime_config),
            }
        )

    def ensure_request(self, context: DuplexStageRequestContext) -> None:
        from vllm_omni.experimental.fullduplex.engine.contracts import DuplexRequestIdentity

        request_state = self._request_states.get(context.request_id)
        if request_state is None:
            request_state = OrchestratorRequestState(
                request_id=context.request_id,
                prompt=None,
                sampling_params_list=list(context.sampling_params),
                final_stage_id=context.final_stage_id,
                duplex_config_generation=context.config_generation,
            )
            request_state.streaming.enabled = True
            self._request_states[context.request_id] = request_state
        elif request_state.duplex_config_generation != context.config_generation:
            request_state.sampling_params_list = list(context.sampling_params)
            request_state.duplex_config_generation = context.config_generation
        request_state.duplex_identity = DuplexRequestIdentity(
            session_id=context.session_id,
            fence=context.fence,
        )
        self._sync_bridge_state(request_state, context)

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        from vllm_omni.experimental.fullduplex.engine.contracts import DuplexStageSubmissionResult

        context = submission.context
        request_state = self._request_states.get(context.request_id)
        if request_state is None:
            raise RuntimeError(f"duplex request was not preregistered: {context.request_id}")
        request = build_engine_core_request_from_tokens(
            request_id=context.request_id,
            prompt=dict(submission.prompt),
            params=context.stage_sampling_params,
            model_config=self._stage_pools[context.stage_id].stage_vllm_config.model_config,
            resumable=True,
        )
        request.external_req_id = request.request_id
        pool = self._stage_pools[context.stage_id]
        if submission.already_submitted:
            replica_id = await pool.submit_update(context.request_id, request_state, request)
        else:
            replica_id = await pool.submit_initial(context.request_id, request_state, request, prompt_text=None)
            if self._async_chunk and context.stage_id == 0:
                await self._prewarm_async_chunk_stages(
                    context.request_id,
                    request,
                    request_state,
                )
        request_state.duplex_stage_fences[context.stage_id] = context.fence
        request_state.stage_submit_ts[context.stage_id] = _time.time()
        if not request_state.running_counter_registered and self._running_counter is not None:
            self._running_counter.increment()
            request_state.running_counter_registered = True
        return DuplexStageSubmissionResult(
            request_id=context.request_id,
            stage_id=context.stage_id,
            replica_id=replica_id,
        )

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
        await self._cleanup_request_ids(request_ids, abort=abort)


class Orchestrator:
    """Runs inside a background thread's asyncio event loop."""

    # Class-level defaults so tests that bypass __init__ via object.__new__
    # don't AttributeError when transfer / counter emit paths access them.
    _running_counter: OmniRequestCounter | None = None
    _transfer_emitter: Any = None
    _stat_logger: OmniPrometheusStatLogger | None = None
    duplex_control_plane: DuplexControlPlanePort | None = None

    def __init__(
        self,
        request_async_queue: janus.AsyncQueue[EngineQueueMessage],
        output_async_queue: janus.AsyncQueue[dict[str, Any]],
        rpc_async_queue: janus.AsyncQueue[dict[str, Any]],
        stage_pools: list[StagePool],
        *,
        async_chunk: bool = False,
        pd_config: dict[str, Any] | None = None,
        membership_controller: MembershipController | None = None,
        running_counter: OmniRequestCounter | None = None,
        transfer_emitter: Any = None,
        log_stats: bool = False,
        enable_orch_monitor: bool = False,
        duplex_runtime_extension: DuplexRuntimeExtension | None = None,
        enable_duplex_control: bool = False,
        duplex_session_config: DuplexSessionRuntimeConfig | None = None,
    ) -> None:
        self.request_async_queue = request_async_queue
        self.output_async_queue = output_async_queue
        self.rpc_async_queue = rpc_async_queue

        self.async_chunk = bool(async_chunk)
        self.num_stages = len(stage_pools)
        self.stage_pools: list[StagePool] = stage_pools
        self.log_stats = log_stats
        self._orch_monitor = create_orch_monitor(
            enabled=enable_orch_monitor,
            replica_sampler=self._sample_replica_metrics,
        )
        for stage_id, pool in enumerate(self.stage_pools):
            for replica_id in pool.available_replica_ids():
                self._orch_monitor.register_replica(stage_id, replica_id)

        # PD disaggregation state
        self._pd_pair: tuple[int, int] | None = None
        self._pd_bootstrap_addr: str | None = None
        self._pd_prefill_engine_id: str | None = None
        self._pd_kv_params: dict[str, Any] = {}
        if pd_config is not None:
            self._pd_pair = pd_config.get("pd_pair")
            self._pd_bootstrap_addr = pd_config.get("bootstrap_addr")
            self._pd_prefill_engine_id = pd_config.get("prefill_engine_id")
        self.request_states: dict[str, OrchestratorRequestState] = {}
        self._init_metrics_state(stage_pools, running_counter, transfer_emitter, log_stats=log_stats)

        self._cfg_tracker = CfgCompanionTracker()
        self._stage_input_processors: dict[int, Any] = {}

        self.duplex_control_plane: DuplexControlPlanePort | None = None
        self._duplex_reaper_interval_s = 1.0
        if enable_duplex_control:
            from vllm_omni.experimental.fullduplex.engine.duplex_control_plane import DuplexControlPlane
            from vllm_omni.experimental.fullduplex.engine.lease import DuplexLeaseConfig

            runtime_session_config = duplex_session_config or DuplexSessionRuntimeConfig()
            self._duplex_reaper_interval_s = runtime_session_config.reaper_interval_s

            self.duplex_control_plane = DuplexControlPlane(
                extension=duplex_runtime_extension,
                stage_port=_OrchestratorDuplexStagePort(
                    stage_pools=self.stage_pools,
                    request_states=self.request_states,
                    running_counter=self._running_counter,
                    cleanup_request_ids=self._cleanup_request_ids,
                    async_chunk=self.async_chunk,
                    prewarm_async_chunk_stages=self._prewarm_async_chunk_stages,
                ),
                result_sink=self.rpc_async_queue,
                lifecycle_sink=self.output_async_queue,
                lease_config=DuplexLeaseConfig(
                    idle_ttl_s=runtime_session_config.idle_ttl_s,
                    disconnect_grace_s=runtime_session_config.disconnect_grace_s,
                ),
                max_sessions=runtime_session_config.max_sessions,
                completed_append_limit=runtime_session_config.completed_append_cache_size,
            )

        self._shutdown_event = asyncio.Event()
        self._stages_shutdown = False

        # Distributed membership (optional, injected by DistStageRuntime)
        self._membership = membership_controller

    def _init_metrics_state(
        self,
        stage_pools: list[StagePool],
        running_counter: OmniRequestCounter | None,
        transfer_emitter: Any,
        log_stats: bool = False,
    ) -> None:
        """Wire up all metric-related orchestrator state.

        Sets ``self._running_counter`` and ``self._transfer_emitter``
        (both optional, used by request-add / forward paths), builds the
        ``(stage_id, replica_id) ↔ engine_idx`` lookup used at record() time,
        and best-effort constructs the ``OmniPrometheusStatLogger`` wrap
        that exposes ~37 upstream ``vllm:*`` families with per-(stage,
        replica) labels. Failure to build the wrap is logged and metrics
        are simply disabled — orchestrator construction continues so unit
        tests with a minimal ``vllm_config`` still pass.

        ``log_stats=False`` short-circuits the wrap entirely so the
        ~65 upstream ``vllm:*`` families are not registered in the
        Prometheus default registry at all. The per-step record() path
        already no-ops on ``scheduler_stats is None`` (which is what
        the upstream scheduler returns when its own log_stats is False),
        so this gate is mainly to keep the ``/metrics`` surface clean
        when the user did not request stats. When stats are enabled, record()
        must still receive IterationStats even on ticks where scheduler_stats
        is None; upstream PrometheusStatLogger records token/request metrics
        from iteration_stats independently of scheduler gauges.
        """
        self._running_counter = running_counter
        self._transfer_emitter = transfer_emitter

        # Flat engine_idx ↔ (stage, replica) maps. The reverse map is
        # consulted at record() time to translate the orchestrator's
        # (stage_id, replica_id) loop variables into an engine_idx the
        # underlying PrometheusStatLogger can address.
        stage_replica_map: dict[int, tuple[str, str]] = {}
        self._stage_replica_to_engine_idx: dict[tuple[int, int], int] = {}
        flat_idx = 0
        for stage_id, pool in enumerate(stage_pools):
            for replica_id in range(pool.num_replicas):
                stage_replica_map[flat_idx] = (str(stage_id), str(replica_id))
                self._stage_replica_to_engine_idx[(stage_id, replica_id)] = flat_idx
                flat_idx += 1

        if not log_stats:
            self._stat_logger = None
            return

        vllm_config_for_stats = next(
            (p.stage_vllm_config for p in stage_pools if p.stage_vllm_config is not None),
            None,
        )
        if vllm_config_for_stats is None:
            self._stat_logger = None
            return
        try:
            self._stat_logger = OmniPrometheusStatLogger(
                vllm_config=vllm_config_for_stats,
                stage_replica_map=stage_replica_map,
            )
        except Exception:
            # Minimal vllm_config in unit-test contexts can lack fields the
            # upstream PrometheusStatLogger expects. Skip wrap rather than
            # break orchestrator construction.
            logger.exception("[Orchestrator] OmniPrometheusStatLogger init failed; metrics wrap disabled")
            self._stat_logger = None

    @property
    def duplex_sessions(self) -> DuplexSessionRuntimeManager:
        if self.duplex_control_plane is None:
            raise RuntimeError("duplex control plane is disabled")
        return self.duplex_control_plane.sessions

    def _require_duplex_control_plane(self) -> DuplexControlPlanePort:
        if self.duplex_control_plane is None:
            raise RuntimeError("duplex control plane is disabled")
        return self.duplex_control_plane

    async def run(self) -> None:
        """Main entry point for the Orchestrator event loop."""
        logger.info("[Orchestrator] Starting event loop")

        request_task = asyncio.create_task(self._request_handler(), name="orchestrator-request-handler")
        output_task = asyncio.create_task(
            self._orchestration_output_handler(),
            name="orchestrator-stage-output-handler",
        )

        # Start membership watcher if distributed mode is active.
        membership_watcher: asyncio.Task[None] | None = None
        if self._membership is not None:
            self._membership.install_unregister_handlers(
                output_queue=self.output_async_queue,
                cleanup_callback=lambda ids: self._cleanup_request_ids(ids, abort=True),
            )
            membership_watcher = self._membership.start()

        tasks = [request_task, output_task]
        if self.duplex_control_plane is not None:
            tasks.append(asyncio.create_task(self._duplex_reaper_loop(), name="orchestrator-duplex-reaper"))
        if membership_watcher is not None:
            tasks.append(membership_watcher)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            raise
        except EngineDeadError as e:
            # The orchestration loop isolates per-replica EngineDeadError
            # (evict + continue), so this only fires if one escapes outside the
            # poll handler (e.g. from output post-processing). Treat it as a
            # teardown signal and let the finally block clean up; do not re-raise.
            logger.info("[Orchestrator] Engine dead, shutting down: %s", e)
            # Unblock any collective_rpc callers waiting on rpc_async_queue
            # instead of leaving them to hit their own timeout.
            await self.rpc_async_queue.put(
                ErrorMessage(
                    error=str(e) or "Stage engine died",
                    fatal=True,
                )
            )
        except Exception:
            logger.exception("[Orchestrator] Fatal error in orchestrator tasks")
            raise
        finally:
            self._shutdown_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass
            if self.duplex_control_plane is not None:
                await self.duplex_control_plane.shutdown()

            if self._membership is not None:
                await self._membership.drain_tasks(timeout=10.0)
                self._membership.shutdown()

            self._orch_monitor.flush()
            self._shutdown_stages()

            loop = asyncio.get_running_loop()
            pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task() and not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    # ---- Request handling ----

    async def _request_handler(self) -> None:
        """Read messages from the main thread via request_async_queue."""
        while True:
            msg = await self.request_async_queue.get()
            msg_type = msg.type

            if msg_type == "add_request":
                await self._handle_add_request(msg)
            elif msg_type == "streaming_update":
                await self._handle_streaming_update(msg)
            elif msg_type == "add_companion_request":
                await self._handle_add_companion(msg)
            elif self.duplex_control_plane is not None and self.duplex_control_plane.accepts(msg):
                self.duplex_control_plane.dispatch(msg)
            elif msg_type == "abort":
                await self._handle_abort(msg)
            elif msg_type == "interaction":
                await self._handle_interaction(msg)
            elif msg_type == "collective_rpc":
                await self._handle_collective_rpc(msg)
            elif isinstance(msg, RegisterRemoteReplicaMessage):
                if self._membership is not None:
                    await self._membership.handle_register(msg.stage_id, msg.replica_id)
                    self._orch_monitor.register_replica(msg.stage_id, msg.replica_id)
            elif isinstance(msg, UnregisterRemoteReplicaMessage):
                if self._membership is not None:
                    await self._membership.handle_unregister(msg.stage_id, msg.input_addr)
            elif isinstance(msg, ShutdownRequestMessage):
                logger.info("[Orchestrator] Received shutdown signal")
                self._shutdown_event.set()
                # Pre-mark stage clients as shutting down to prevent
                # proc_monitor daemon threads from flagging normal
                # process exit as EngineDeadError during teardown.
                for pool in self.stage_pools:
                    for client in pool.clients:
                        if hasattr(client, "_shutting_down"):
                            client._shutting_down = True
                # Stage teardown runs once in run()'s finally after the
                # orchestration loop observes _shutdown_event and exits.
                break
            else:
                logger.warning("[Orchestrator] Unknown message type: %s", msg_type)

    async def _duplex_reaper_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._duplex_reaper_interval_s,
                )
            except TimeoutError:
                plane = self.duplex_control_plane
                if plane is not None:
                    try:
                        await plane.reap_expired()
                    except Exception:
                        logger.exception("[Orchestrator] Duplex expiry cleanup failed; retrying on next tick")

    async def _handle_add_request(self, msg: StageSubmissionMessage) -> None:
        """Handle an add_request message from the main thread."""
        stage_id = 0
        request_id = msg.request_id
        prompt = msg.prompt
        original_prompt = msg.original_prompt
        sampling_params_list = msg.sampling_params_list
        if not sampling_params_list:
            raise ValueError(f"Missing sampling params for stage 0. Got {len(sampling_params_list)} stage params.")
        final_stage_id = msg.final_stage_id
        final_output_stage_ids = set(msg.final_output_stage_ids or [final_stage_id])

        if not self.stage_pools[stage_id].live_replica_ids():
            # Stage 0 lost all replicas between the HTTP-layer errored check and
            # dispatch. Runs before request state / running counter registration,
            # so the helper's cleanup is a no-op here.
            await self._fail_request_dead_stage(request_id, stage_id)
            return

        logger.debug(
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

        req_state = OrchestratorRequestState(
            request_id=request_id,
            prompt=original_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            final_output_stage_ids=final_output_stage_ids,
            request_timestamp=float(msg.request_timestamp or _time.time()),
            mm_features=getattr(prompt, "mm_features", None),
        )
        self.request_states[request_id] = req_state
        self._register_running_request(req_state)
        req_state.streaming.enabled = bool(getattr(prompt, "resumable", False))
        req_state.stage_submit_ts[stage_id] = _time.time()
        enqueue_ts = msg.enqueue_ts
        if enqueue_ts > 0:
            req_state.pipeline_timings["queue_wait_ms"] = (_time.perf_counter() - enqueue_ts) * 1000.0
        preprocess_ms = msg.preprocess_ms
        if preprocess_ms > 0:
            req_state.pipeline_timings["preprocess_ms"] = preprocess_ms
        if not await self._dispatch_or_fail_request(
            lambda: self.stage_pools[stage_id].submit_initial(
                request_id,
                req_state,
                prompt,
                prompt_text=msg.output_prompt_text,
            ),
            req_id=request_id,
            stage_id=stage_id,
            operation="add_request",
        ):
            return

        if self.async_chunk and stage_id == 0 and final_stage_id > 0:
            await self._prewarm_async_chunk_stages(request_id, prompt, req_state)

    async def _handle_streaming_update(self, msg: StageSubmissionMessage) -> None:
        """Handle a streaming_update message for an existing request."""
        stage_id = 0
        request_id = msg.request_id
        request = msg.prompt
        final_stage_id = msg.final_stage_id
        req_state = self.request_states.get(request_id)
        if req_state is None:
            # Streaming updates always follow the first-chunk add_request
            # through the same ordered submission queue, so an unknown id
            # here can only mean the request already finished or was
            # aborted (e.g. the client disconnected). Re-adding it would
            # resurrect a headless session that keeps cycling through the
            # stages with nobody consuming its outputs (issue #4271).
            logger.warning(
                "[Orchestrator] streaming_update for unknown req=%s; dropping (request finished or aborted)",
                request_id,
            )
            return

        if msg.sampling_params_list:
            req_state.sampling_params_list = msg.sampling_params_list

        req_state.streaming.enabled = True
        req_state.stage_submit_ts[stage_id] = _time.time()
        if not await self._dispatch_or_fail_request(
            lambda: self.stage_pools[stage_id].submit_update(
                request_id,
                req_state,
                request,
                prompt_text=msg.output_prompt_text,
            ),
            req_id=request_id,
            stage_id=stage_id,
            operation="streaming_update",
        ):
            return

        if self.async_chunk and stage_id == 0 and final_stage_id > 0:
            await self._prewarm_async_chunk_stages(request_id, request, req_state)

    async def _handle_add_companion(self, msg: AddCompanionRequestMessage) -> None:
        """Handle an add_companion_request message: submit companion to stage 0."""
        companion_id = msg.companion_id
        parent_id = msg.parent_id
        role = msg.role
        companion_prompt = msg.prompt
        sampling_params_list = msg.sampling_params_list

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            logger.info(
                "[Orchestrator] Dropping CFG companion %s (role=%s): parent %s is no longer active",
                companion_id,
                role,
                parent_id,
            )
            return

        self._cfg_tracker.register_companion(parent_id, role, companion_id)

        companion_state = OrchestratorRequestState(
            request_id=companion_id,
            prompt=companion_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=0,
            final_output_stage_ids={0},
            request_timestamp=parent_state.request_timestamp,
        )
        self.request_states[companion_id] = companion_state
        companion_state.stage_submit_ts[0] = _time.time()
        # A dead-stage failure is attributed to the parent request; its cleanup
        # also releases this companion.
        if not await self._dispatch_or_fail_request(
            lambda: self.stage_pools[0].submit_initial(
                companion_id,
                companion_state,
                companion_prompt,
                prompt_text=msg.companion_prompt_text,
                affinity_request_id=parent_id,
            ),
            req_id=parent_id,
            stage_id=0,
            operation=f"companion {companion_id}",
            dispatch_req_id=companion_id,
        ):
            return

        logger.info(
            "[Orchestrator] CFG companion submitted: %s (role=%s, parent=%s, stage-0 replica-%s)",
            companion_id,
            role,
            parent_id,
            self.stage_pools[0].get_bound_replica_id(companion_id),
        )

    async def _handle_abort(self, msg: AbortRequestMessage) -> None:
        """Handle an abort message from the main thread."""
        request_ids = msg.request_ids
        # _cleanup_request_ids is CFG-aware: it expands aborted parents to
        # their companions and fails a deferred parent whose companion is
        # aborted before its output arrived.
        await self._cleanup_request_ids(list(request_ids), abort=True)
        logger.info("[Orchestrator] Aborted request(s) %s", request_ids)

    async def _handle_interaction(self, msg: InteractionMessage) -> None:
        """Handle a midway interaction for an active streaming diffusion request."""
        stage_id = 0
        request_id = msg.request_id
        event_id = msg.interaction.get("event_id")
        req_state = self.request_states.get(request_id)
        if req_state is None:
            logger.info("[Orchestrator] Dropping interaction for inactive req %s", request_id)
            await self.output_async_queue.put(
                ErrorMessage(
                    error=f"No active request for interaction: {request_id}",
                    fatal=False,
                    request_id=request_id,
                    event_id=event_id,
                    stage_id=stage_id,
                )
            )
            return

        try:
            await self.stage_pools[stage_id].submit_interaction(request_id, msg.interaction)
        except Exception as exc:
            logger.info(
                "[Orchestrator] Failed interaction for req %s: %s",
                request_id,
                exc,
                exc_info=True,
            )
            await self.output_async_queue.put(
                ErrorMessage(
                    error=f"Failed interaction for request {request_id}: {exc}",
                    fatal=False,
                    request_id=request_id,
                    event_id=event_id,
                    stage_id=stage_id,
                )
            )

    async def _abort_request_ids(self, request_ids: list[str]) -> None:
        """Forward abort requests to all stage pools."""
        if not request_ids:
            return
        for pool in self.stage_pools:
            await pool.abort_requests(request_ids)
            pool.release_bindings(request_ids)

    def _release_request_bindings(self, request_ids: list[str]) -> None:
        """Release all stage-local route bindings for the given request ids."""
        for pool in self.stage_pools:
            pool.release_bindings(request_ids)

    async def _handle_collective_rpc(self, msg: CollectiveRPCRequestMessage) -> None:
        """Handle a control-plane RPC request from the main thread."""
        rpc_id = msg.rpc_id
        method = msg.method
        timeout = msg.timeout
        args = tuple(msg.args)
        kwargs = dict(msg.kwargs or {})
        requested_stage_ids = msg.stage_ids

        target_pools: list[StagePool] = []
        if requested_stage_ids is None:
            target_pools.extend(self.stage_pools)
        else:
            for lid in requested_stage_ids:
                if not (0 <= lid < self.num_stages):
                    logger.warning("[Orchestrator] collective_rpc: ignoring invalid stage_id %s", lid)
                    continue
                target_pools.append(self.stage_pools[lid])

        results: list[Any] = []
        stage_ids: list[int] = []
        for pool in target_pools:
            for replica_id in pool.live_replica_ids():
                stage_result = await pool.collective_rpc(
                    replica_id=replica_id,
                    method=method,
                    timeout=timeout,
                    args=args,
                    kwargs=kwargs,
                )
                stage_ids.append(pool.stage_id)
                results.append(stage_result)

        await self.rpc_async_queue.put(
            CollectiveRPCResultMessage(
                rpc_id=rpc_id,
                method=method,
                stage_ids=stage_ids,
                results=results,
            )
        )

    # ---- Orchestration loop ----

    def _sample_replica_metrics(self) -> dict[str, tuple[int, int]]:
        samples: dict[str, tuple[int, int]] = {}
        for stage_id, pool in enumerate(self.stage_pools):
            for replica_id in pool.live_replica_ids():
                key = replica_key(stage_id, replica_id)
                samples[key] = pool.replica_monitor_sample(replica_id)
        return samples

    async def _orchestration_output_handler(self) -> None:
        """Poll all stages, handle transfers, send final outputs to main."""
        try:
            await self._orchestration_loop()
        except asyncio.CancelledError:
            logger.debug("[Orchestrator] _orchestration_output_handler cancelled")
            return

    async def _orchestration_loop(self) -> None:
        """Poll stage pools and route logical outputs."""
        while not self._shutdown_event.is_set():
            idle = True
            for stage_id in range(self.num_stages):
                pool = self.stage_pools[stage_id]
                for replica_id in pool.available_replica_ids():
                    if self._shutdown_event.is_set():
                        return

                    # Shared catch so a dead replica on either poll path is
                    # evicted rather than tearing down every stage (#4285).
                    try:
                        if pool.stage_type == "diffusion":
                            diffusion_output = pool.poll_diffusion_output(replica_id)
                            if diffusion_output is None:
                                continue

                            pool.record_output_timestamps([diffusion_output])
                            processed = [diffusion_output]
                        else:
                            raw_outputs = await pool.poll_llm_raw_output(replica_id, timeout_s=0.001)
                            if raw_outputs is None:
                                continue

                            await self._handle_kv_ready_raw_outputs(stage_id, raw_outputs)
                            for eco in raw_outputs.outputs:
                                req_state = self.request_states.get(getattr(eco, "request_id", None))
                                if req_state is None or not req_state.streaming.enabled:
                                    continue
                                req_state.streaming.segment_finished = bool(getattr(eco, "is_segment_finished", False))
                                req_state.streaming.segment_token_ids = (
                                    self._coerce_int_list(getattr(eco, "new_token_ids", None))
                                    if req_state.streaming.segment_finished
                                    else []
                                )
                                raw_mm = self._completion_multimodal_output(eco, None)
                                req_state.streaming.segment_output_metadata = (
                                    dict(raw_mm)
                                    if req_state.streaming.segment_finished and isinstance(raw_mm, dict)
                                    else {}
                                )
                                req_state.streaming.new_prompt_len_snapshot = getattr(
                                    eco,
                                    "new_prompt_len_snapshot",
                                    None,
                                )
                                if req_state.streaming.enabled:
                                    await self._apply_raw_terminal_stage_finish(stage_id, eco, req_state)
                            iteration_stats = (
                                IterationStats() if (self._stat_logger is not None and raw_outputs.outputs) else None
                            )
                            processed = await pool.process_llm_raw_outputs(
                                replica_id,
                                raw_outputs,
                                iteration_stats=iteration_stats,
                            )
                            if self._stat_logger is not None and (
                                raw_outputs.scheduler_stats is not None or iteration_stats is not None
                            ):
                                self._stat_logger.record(
                                    raw_outputs.scheduler_stats,
                                    iteration_stats,
                                    engine_idx=self._stage_replica_to_engine_idx[(stage_id, replica_id)],
                                )
                    except asyncio.CancelledError:
                        raise
                    except EngineDeadError as e:
                        await self._handle_dead_replica(stage_id, replica_id, e)
                        continue
                    except Exception:
                        if self._shutdown_event.is_set():
                            return
                        logger.exception(
                            "[Orchestrator] Stage-%s replica-%s processing failed",
                            stage_id,
                            replica_id,
                        )
                        raise

                    await self._handle_processed_outputs(stage_id, replica_id, processed)
                    idle = False

            self._orch_monitor.note_loop(idle=idle)
            if idle:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)

    async def _handle_processed_outputs(self, stage_id: int, replica_id: int, outputs: list[Any]) -> None:
        """Route processed stage outputs produced by one stage poll."""
        pool = self.stage_pools[stage_id]
        for output in outputs:
            req_state = self.request_states.get(output.request_id)
            if req_state is None:
                logger.warning(
                    "[Orchestrator] Dropping output for unknown req %s at stage-%s (known reqs: %s)",
                    output.request_id,
                    stage_id,
                    list(self.request_states.keys()),
                )
                continue

            if getattr(output, "error", None) is not None:
                await self._handle_stage_error(stage_id, output)
                continue

            stage_metrics = None
            segment_finished = req_state.streaming.enabled and req_state.streaming.segment_finished
            if output.finished or segment_finished:
                stage_metrics = pool.build_stage_metrics(
                    [output],
                    submit_ts=req_state.stage_submit_ts.get(stage_id, _time.time()),
                    request_timestamp=req_state.request_timestamp,
                    replica_id=replica_id,
                    sampling_params=req_state.sampling_params_list[stage_id],
                )
                stage_metrics.pipeline_timings = dict(req_state.pipeline_timings)

            await self._route_output(stage_id, replica_id, output, req_state, stage_metrics)

    async def _handle_stage_error(self, stage_id: int, output: Any) -> None:
        """Emit a frontend-visible error and clean up request state."""
        if self._cfg_tracker.is_companion(output.request_id):
            parent_id = self._cfg_tracker.get_parent_id(output.request_id) or output.request_id
        else:
            parent_id = output.request_id
        await self.output_async_queue.put(
            ErrorMessage(
                request_id=parent_id,
                stage_id=stage_id,
                error=output.error,
                status_code=getattr(output, "error_status_code", None),
                error_type=getattr(output, "error_type", None),
            )
        )
        await self._cleanup_request_ids(
            [parent_id, *self._cfg_tracker.cleanup_parent(parent_id)],
            abort=True,
            close_duplex_sessions=True,
        )

    async def _handle_dead_replica(self, stage_id: int, replica_id: int, error: EngineDeadError) -> None:
        """Evict a dead stage replica and fail the requests stranded on it (#4285).

        A dead replica must not tear down the whole server: evict it and keep the
        orchestrator loop alive so remaining live replicas / stages keep serving.
        If the stage still has a live replica, fail only the requests bound to the
        dead replica; requests bound to a surviving replica (or not yet bound)
        keep running and can be (re)dispatched. If the stage has no live replica
        left, every request routed through it must fail since none can be served.
        """
        pool = self.stage_pools[stage_id]
        logger.error(
            "[Orchestrator] Stage-%s replica-%s is dead; evicting it and failing its in-flight requests: %s",
            stage_id,
            replica_id,
            error,
        )
        pool.evict_replica(replica_id)
        stage_has_live = bool(pool.live_replica_ids())
        failed_ids: list[str] = []
        for req_id, req_state in list(self.request_states.items()):
            if stage_id not in req_state.stage_submit_ts:
                continue
            bound = pool.get_bound_replica_id(req_id)
            if bound == replica_id or not stage_has_live:
                await self.output_async_queue.put(
                    ErrorMessage(
                        error=str(error),
                        fatal=True,
                        request_id=req_id,
                        stage_id=stage_id,
                    )
                )
                failed_ids.append(req_id)
        # Use the shared cleanup path so the running counter, PD/CFG state, and
        # bindings across every stage pool are released (not just this pool).
        # abort=True stops the failed requests' in-flight work on surviving
        # stages (abort skips dead/evicted bindings), matching the
        # distributed-membership eviction path.
        for req_id in failed_ids:
            await self._cleanup_request_ids(
                [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                abort=True,
            )

    async def _fail_request_dead_stage(self, req_id: str, stage_id: int) -> None:
        """Fail one request whose target stage has no live replica (#4285).

        Used at dispatch sites so a fully dead stage fails just that request
        instead of raising out of the request/orchestration task and tearing
        down the server. The cleanup releases the running counter, PD/CFG state,
        and bindings across every stage pool; it is a no-op for a request that
        was never registered.
        """
        logger.error(
            "[Orchestrator] req=%s: stage-%d has no live replica; failing the request",
            req_id,
            stage_id,
        )
        await self.output_async_queue.put(
            ErrorMessage(
                error=f"Stage-{stage_id} has no live replica",
                fatal=True,
                request_id=req_id,
                stage_id=stage_id,
            )
        )
        await self._cleanup_request_ids(
            [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
            abort=True,
        )

    async def _dispatch_or_fail_request(
        self,
        dispatch: Callable[[], Awaitable[Any]],
        *,
        req_id: str,
        stage_id: int,
        operation: str,
        dispatch_req_id: str | None = None,
    ) -> bool:
        """Run a dispatch action, converting stage unavailability into a
        single-request failure.

        ``dispatch`` is a thunk rather than a bare coroutine so that replica
        selection and other argument setup run inside this guard: a synchronous
        StageUnavailableError raised while building the dispatch is caught here,
        not left to propagate out of the caller.

        Replica eviction can interleave with any dispatch, surfacing as
        StageUnavailableError (no live replica / evicted slot) or
        EngineDeadError (replica died mid-submit before the poll loop evicted
        it). Both mean "this request cannot be placed", not "the server is
        broken": fail the request and keep serving (#4285). Unrelated errors
        propagate. Returns False when the request was failed.

        ``req_id`` is the request the failure is attributed to; ``dispatch_req_id``
        is the id actually being dispatched when it differs (e.g. a CFG companion
        submitted under its parent's attribution) and is used to locate the
        replica that died.
        """
        try:
            await dispatch()
            return True
        except StageUnavailableError as e:
            # No specific replica to evict: the stage already has no live
            # replica or the chosen slot was evicted. Fail just this request.
            logger.error(
                "[Orchestrator] %s dispatch for req=%s hit unavailable stage-%s: %s",
                operation,
                req_id,
                stage_id,
                e,
            )
            await self._fail_request_dead_stage(req_id, stage_id)
            return False
        except EngineDeadError as e:
            # A live replica died mid-submit before the poll loop evicted it.
            # Evict it now (and fail every request bound to it) so a burst of
            # new requests stops round-robining onto the dead slot until the
            # poll catches up. Capture the replica before failing this request,
            # since the failure releases its binding.
            pool = self.stage_pools[stage_id]
            dead_replica = pool.get_bound_replica_id(dispatch_req_id or req_id)
            logger.error(
                "[Orchestrator] %s dispatch for req=%s hit dead stage-%s replica-%s: %s",
                operation,
                req_id,
                stage_id,
                dead_replica,
                e,
            )
            await self._fail_request_dead_stage(req_id, stage_id)
            if dead_replica is not None and dead_replica in pool.live_replica_ids():
                await self._handle_dead_replica(stage_id, dead_replica, e)
            return False

    # ---- Shared helpers ----

    async def _cleanup_request_ids(
        self,
        request_ids: list[str],
        *,
        abort: bool = False,
        close_duplex_sessions: bool = False,
    ) -> None:
        """Release pool bindings and logical request state for the given ids.

        CFG-aware: cleaning a parent releases its tracker state and pulls its
        companions into the batch; cleaning a companion whose parent is NOT in
        the batch and whose output never arrived fails that parent (its bundle
        can never complete). Every teardown path (stage error, abort, replica
        loss, membership unregister) funnels through here, so tracker state
        cannot outlive its requests.
        """
        if not request_ids:
            return

        cleanup_ids = list(dict.fromkeys(request_ids))
        batch = set(cleanup_ids)
        orphaned_parents: dict[str, str] = {}
        for rid in cleanup_ids:
            pid = self._cfg_tracker.get_parent_id(rid)
            if pid is not None and pid not in batch and not self._cfg_tracker.is_companion_done(rid):
                orphaned_parents.setdefault(pid, rid)
        for pid, cid in orphaned_parents.items():
            deferred = self._cfg_tracker.pop_pending_parent(pid)
            await self.output_async_queue.put(
                ErrorMessage(
                    request_id=pid,
                    stage_id=deferred["stage_id"] if deferred is not None else None,
                    error=f"CFG companion {cid} was aborted or lost before its outputs arrived",
                )
            )
            batch.add(pid)
            cleanup_ids.append(pid)
        for rid in list(cleanup_ids):
            for cid in self._cfg_tracker.cleanup_parent(rid):
                if cid not in batch:
                    batch.add(cid)
                    cleanup_ids.append(cid)
        closing_session_ids: list[str] = []
        if close_duplex_sessions and self.duplex_control_plane is not None:
            closed_sessions = self.duplex_control_plane.close_sessions_for_request_ids(
                cleanup_ids,
                abort=abort,
                cleanup_in_progress=True,
            )
            closing_session_ids.extend(closed_sessions)
            for session_id, stale_request_ids in closed_sessions.items():
                logger.info(
                    "[Orchestrator] closed duplex session %s while cleaning failed request ids %s",
                    session_id,
                    stale_request_ids,
                )
                cleanup_ids.extend(stale_request_ids)
            cleanup_ids = list(dict.fromkeys(cleanup_ids))

        try:
            if abort:
                await self._abort_request_ids(cleanup_ids)
            self._release_request_bindings(cleanup_ids)
            for request_id in cleanup_ids:
                self._pd_kv_params.pop(request_id, None)
                req_state = self.request_states.pop(request_id, None)
                if req_state is not None and req_state.running_counter_registered and self._running_counter is not None:
                    self._running_counter.decrement()
                    req_state.running_counter_registered = False
        except BaseException:
            if closing_session_ids and self.duplex_control_plane is not None:
                self.duplex_control_plane.defer_request_cleanups(closing_session_ids)
            raise
        if closing_session_ids and self.duplex_control_plane is not None:
            self.duplex_control_plane.finalize_closed_sessions(closing_session_ids)

    async def _apply_raw_terminal_stage_finish(
        self,
        stage_id: int,
        eco: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Record session-level finish markers dropped by the streaming output processor.

        Streaming segment stops set ``is_segment_finished=True`` and are handled
        via processed outputs. Session termination (e.g. ``finish_requests`` after
        ``resumable=False``) emits a terminal ``finish_reason`` with
        ``is_segment_finished=False``, but vLLM's output processor may remove the
        request state before that EngineCoreOutput is processed.

        Only update ``finished_final_output_stage_ids`` here. Request cleanup stays
        in ``_route_output`` so downstream async-chunk stages can still deliver
        outputs after stage-0 session end.
        """
        if getattr(eco, "finish_reason", None) is None:
            return
        if getattr(eco, "is_segment_finished", False):
            return

        final_output_stage_ids = req_state.final_output_stage_ids or {req_state.final_stage_id}
        if stage_id not in final_output_stage_ids:
            return
        req_state.finished_final_output_stage_ids.add(stage_id)

    def _maybe_clone_diffusion_params_for_cfg(self, request_id: str, params: Any) -> Any:
        """Attach CFG companion ids to diffusion sampling params when needed."""
        companion_request_ids = self._cfg_tracker.get_companion_request_ids(request_id)
        if not companion_request_ids:
            return params

        import copy

        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        if not isinstance(params, OmniDiffusionSamplingParams):
            return params

        params = copy.deepcopy(params)
        params.cfg_kv_request_ids = companion_request_ids
        return params

    def _duplex_session_for_req_state(self, req_state: OrchestratorRequestState) -> DuplexSessionRuntimeState | None:
        if self.duplex_control_plane is None:
            return None
        return self.duplex_control_plane.session_for_identity(req_state.duplex_identity)

    def _record_duplex_stage_submission(
        self,
        stage_id: int,
        request_id: str,
        replica_id: int,
        req_state: OrchestratorRequestState,
    ) -> None:
        del replica_id
        identity = req_state.duplex_identity
        session = self._duplex_session_for_req_state(req_state)
        if identity is None or session is None:
            return
        req_state.duplex_stage_fences[stage_id] = identity.fence
        session.bind_stage_request(stage_id, request_id, fence=identity.fence)
        req_state.stage_submit_ts[stage_id] = _time.time()
        self._register_running_request(req_state)

    def _register_running_request(self, req_state: OrchestratorRequestState) -> None:
        if req_state.running_counter_registered or self._running_counter is None:
            return
        self._running_counter.increment()
        req_state.running_counter_registered = True

    async def _route_output(
        self,
        stage_id: int,
        replica_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        stage_metrics: Any,
    ) -> None:
        """Route a processed output: send to frontend and/or forward."""
        req_id = output.request_id
        finished = output.finished
        submit_ts = req_state.stage_submit_ts.get(stage_id)
        # CFG companion: stash output so parent can bundle [parent, *companions]
        # into source_outputs for the bridge (e.g. thinker2imagegen).
        if finished and self._cfg_tracker.is_companion(req_id):
            self._cfg_tracker.set_companion_output(req_id, output)
            await self._handle_cfg_companion_ready(req_id)
            await self._cleanup_request_ids([req_id])
            return

        request_finished = False
        if (
            finished
            and self.stage_pools[stage_id].final_output
            and not (req_state.streaming.enabled and req_state.streaming.segment_finished)
        ):
            req_state.finished_final_output_stage_ids.add(stage_id)
            final_output_stage_ids = req_state.final_output_stage_ids or {req_state.final_stage_id}
            request_finished = final_output_stage_ids.issubset(req_state.finished_final_output_stage_ids)
        # Duplex stage-0 segment boundaries are not client-visible outputs:
        # direct decisions are emitted by the model runtime extension below,
        # while spoken content flows through the next stage. Forwarding
        # the raw stage-0 output as well injects one cumulative-text,
        # no-audio message per unit that every downstream consumer must
        # filter out again (the official implementation returns exactly one
        # result per audio chunk).
        is_duplex_stage0_segment = (
            stage_id == 0 and self._is_duplex_session_request(req_state) and req_state.streaming.segment_finished
        )
        if self.stage_pools[stage_id].final_output and not is_duplex_stage0_segment:
            await self.output_async_queue.put(
                OutputMessage(
                    request_id=req_id,
                    stage_id=stage_id,
                    replica_id=replica_id,
                    engine_outputs=output,
                    metrics=stage_metrics,
                    finished=(
                        request_finished
                        or (self._is_duplex_session_request(req_state) and req_state.streaming.segment_finished)
                    ),
                    stage_submit_ts=submit_ts,
                )
            )
        elif stage_metrics is not None:
            await self.output_async_queue.put(
                StageMetricsMessage(
                    request_id=req_id,
                    stage_id=stage_id,
                    replica_id=replica_id,
                    metrics=stage_metrics,
                    stage_submit_ts=submit_ts,
                )
            )

        if self._pd_pair is not None and finished and stage_id == self._pd_pair[0]:
            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is not None:
                self._pd_kv_params[req_id] = kv_params if isinstance(kv_params, dict) else dict(kv_params)
            req_state.pd_prefill_multimodal_output = getattr(output, "multimodal_output", None)

        duplex_output_decision = self._duplex_output_decision(stage_id, output, req_state)
        if duplex_output_decision is not None:
            await self._emit_duplex_direct_output(
                stage_id,
                req_id,
                output,
                duplex_output_decision,
                stage_metrics,
                submit_ts,
            )
            return

        if (
            (finished or (req_state.streaming.enabled and req_state.streaming.segment_finished))
            and stage_id < req_state.final_stage_id
            and (not self.async_chunk or not self._stage_receives_async_chunks(stage_id + 1))
            and (not self._next_stage_already_submitted(stage_id, req_state) or req_state.streaming.enabled)
        ):
            if (
                finished
                and self._cfg_tracker.has_companions(req_id)
                and not self._cfg_tracker.all_companions_done(req_id)
            ):
                self._cfg_tracker.defer_parent(req_id, output, stage_id)
            else:
                stage_params = req_state.sampling_params_list[stage_id]
                final_only_finished = (
                    req_state.streaming.enabled
                    and finished
                    and getattr(stage_params, "output_kind", None) == RequestOutputKind.FINAL_ONLY
                )
                await self._forward_to_next_stage(
                    req_id,
                    stage_id,
                    output,
                    req_state,
                    src_replica_id=replica_id,
                    is_streaming_session=req_state.streaming.enabled,
                    is_final_update=final_only_finished,
                )
                if (
                    req_state.streaming.enabled
                    and finished
                    and not final_only_finished
                    and not self._is_duplex_session_request(req_state)
                ):
                    # For streaming sessions, send the terminal (resumable=False) update only on a finish
                    await self._forward_to_next_stage(
                        req_id,
                        stage_id,
                        output,
                        req_state,
                        src_replica_id=replica_id,
                        is_streaming_session=True,
                        is_final_update=True,
                    )

        if request_finished and not self._is_duplex_session_request(req_state):
            await self._cleanup_request_ids([req_id, *self._cfg_tracker.cleanup_parent(req_id)])

    def _next_stage_already_submitted(self, stage_id: int, req_state: OrchestratorRequestState) -> bool:
        return (stage_id + 1) in req_state.stage_submit_ts

    def _stage_receives_async_chunks(self, stage_id: int) -> bool:
        """Whether a stage's connector supplies its runtime inputs."""
        pool = self.stage_pools[stage_id]
        model_config = getattr(pool.stage_vllm_config, "model_config", None)
        return stage_receives_chunks(model_config)

    def _get_stage_input_processor(self, stage_id: int) -> Any:
        processor = self._stage_input_processors.get(stage_id)
        if processor is None:
            from vllm_omni.engine.stage_init_utils import build_stage0_input_processor

            processor = build_stage0_input_processor(self.stage_pools[stage_id].stage_vllm_config)
            self._stage_input_processors[stage_id] = processor
        return processor

    def _upgrade_processed_stage_request(self, request: Any, raw_prompt: Any) -> Any:
        prompt_embeds = getattr(request, "prompt_embeds", None)
        additional_information = None

        if isinstance(raw_prompt, dict):
            if prompt_embeds is None:
                raw_prompt_embeds = raw_prompt.get("prompt_embeds")
                if isinstance(raw_prompt_embeds, torch.Tensor):
                    prompt_embeds = raw_prompt_embeds
            additional_information = serialize_additional_information(
                raw_prompt.get("additional_information"),
                log_prefix="Orchestrator stage input",
            )

        if prompt_embeds is None and additional_information is None:
            return request

        return OmniEngineCoreRequest.from_request(
            request,
            prompt_embeds=prompt_embeds,
            additional_information=additional_information,
        )

    def _next_stage_input_is_tokens(self, next_input: Any) -> bool:
        return isinstance(next_input, dict) and "prompt_token_ids" in next_input

    def _build_next_stage_request(
        self,
        req_id: str,
        next_stage_id: int,
        next_input: Any,
        params: SamplingParams | PoolingParams,
        *,
        mm_features: list | None = None,
        resumable: bool = False,
    ) -> Any:
        next_pool = self.stage_pools[next_stage_id]
        if self._next_stage_input_is_tokens(next_input):
            request = build_engine_core_request_from_tokens(
                request_id=req_id,
                prompt=next_input,
                params=params,
                model_config=next_pool.stage_vllm_config.model_config,
                mm_features=mm_features,
                resumable=resumable,
            )
            request.external_req_id = request.request_id
            return request

        processor = self._get_stage_input_processor(next_stage_id)
        # A pooling stage is driven by PoolingParams; vLLM validates them against
        # the stage's supported tasks, so advertise the model's pooling tasks (or
        # the configured task) instead of generate.
        if isinstance(params, PoolingParams):
            model_cfg = next_pool.stage_vllm_config.model_config
            supported_tasks = tuple(getattr(model_cfg, "supported_tasks", ()) or ()) or (params.task,)
        else:
            supported_tasks = ("generate",)
        request = processor.process_inputs(
            request_id=req_id,
            prompt=next_input,
            params=params,
            supported_tasks=supported_tasks,
            arrival_time=_time.time(),
            resumable=resumable,
        )
        request = self._upgrade_processed_stage_request(request, next_input)
        request.external_req_id = req_id
        return request

    @staticmethod
    def _duplex_output_context(
        req_state: OrchestratorRequestState,
        *,
        stage_id: int | None = None,
    ) -> DuplexOutputContext | None:
        identity = req_state.duplex_identity
        if identity is None:
            return None
        from vllm_omni.experimental.fullduplex.engine.contracts import (
            DuplexOutputContext,
            DuplexRequestIdentity,
        )

        fence = req_state.duplex_stage_fences.get(stage_id, identity.fence) if stage_id is not None else identity.fence
        return DuplexOutputContext(
            identity=DuplexRequestIdentity(
                session_id=identity.session_id,
                fence=fence,
            ),
            final_stage_id=req_state.final_stage_id,
            segment_finished=req_state.streaming.enabled and req_state.streaming.segment_finished,
            segment_token_ids=tuple(req_state.streaming.segment_token_ids),
            segment_output_metadata=req_state.streaming.segment_output_metadata,
        )

    @staticmethod
    def _is_duplex_session_request(req_state: OrchestratorRequestState) -> bool:
        return req_state.duplex_identity is not None

    @staticmethod
    def _duplex_fence_for_req_state(
        req_state: OrchestratorRequestState,
        *,
        stage_id: int | None = None,
    ) -> DuplexFence | None:
        context = Orchestrator._duplex_output_context(req_state, stage_id=stage_id)
        return context.identity.fence if context is not None else None

    def _duplex_output_decision(
        self,
        stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
    ) -> DuplexOutputDecision | None:
        if self.duplex_control_plane is None:
            return None
        context = self._duplex_output_context(req_state, stage_id=stage_id)
        decision = self.duplex_control_plane.decide_output(
            stage_id,
            output,
            context,
        )
        return decision

    async def _emit_duplex_direct_output(
        self,
        stage_id: int,
        req_id: str,
        output: Any,
        decision: DuplexOutputDecision,
        stage_metrics: Any,
        submit_ts: float | None,
    ) -> None:
        action = getattr(decision.action, "value", decision.action)
        if action != "direct_response":
            raise ValueError(f"Unsupported duplex output action: {action}")
        from vllm_omni.experimental.fullduplex.output import attach_duplex_output_decision

        engine_output = attach_duplex_output_decision(
            OmniRequestOutput(
                request_id=req_id,
                finished=True,
                stage_id=stage_id,
                final_output_type=decision.final_output_type,
                request_output=output,
            ),
            decision,
        )
        await self.output_async_queue.put(
            OutputMessage(
                request_id=req_id,
                stage_id=stage_id,
                engine_outputs=engine_output,
                metrics=stage_metrics,
                finished=True,
                stage_submit_ts=submit_ts,
            )
        )

    @staticmethod
    def _completion_multimodal_output(output: Any, completion: Any) -> dict[str, Any]:
        mm_output = getattr(output, "multimodal_output", None)
        if isinstance(mm_output, dict):
            return mm_output
        mm_output = getattr(completion, "multimodal_output", None) if completion is not None else None
        return mm_output if isinstance(mm_output, dict) else {}

    @classmethod
    def _coerce_int_list(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            try:
                value = value.detach().cpu().reshape(-1).tolist()
            except Exception:
                return []
        if not isinstance(value, (list, tuple)):
            return []
        out: list[int] = []
        for item in value:
            token_id = cls._coerce_int(item)
            if token_id is not None:
                out.append(token_id)
        return out

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if hasattr(value, "detach"):
            try:
                value = value.detach().cpu().reshape(-1)
                if value.numel() == 0:
                    return None
                value = value[0].item()
            except Exception:
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    async def _handle_cfg_companion_ready(self, req_id: str) -> None:
        """Mark a CFG companion as done; if all companions are done, flush deferred parent."""
        parent_id = self._cfg_tracker.on_companion_completed(req_id)
        if parent_id is None:
            return

        deferred = self._cfg_tracker.pop_pending_parent(parent_id)
        if deferred is None:
            return

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            return

        stage_id = deferred["stage_id"]
        if (stage_id + 1) in parent_state.stage_submit_ts:
            return

        await self._forward_to_next_stage(
            parent_id,
            stage_id,
            deferred["engine_outputs"],
            parent_state,
        )

    async def _handle_kv_ready_raw_outputs(
        self,
        stage_id: int,
        raw_outputs: EngineCoreOutputs,
    ) -> None:
        """Forward split requests once stage-0 KV is ready."""
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
                # kv_ready only says the companion's KV hit the connector; its
                # processed output has not been stashed yet. Counting it as
                # done here let the parent pass the all_companions_done gate
                # and dispatch with 0/N companion outputs (degraded CFG).
                # Done-marking now happens solely on the processed-output path
                # in _route_output, after set_companion_output.
                continue
            if stage_id >= req_state.final_stage_id:
                continue
            if (stage_id + 1) in req_state.stage_submit_ts:
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

    def _emit_tx_edge(
        self,
        *,
        from_stage: int,
        from_replica: int,
        to_stage: int,
        to_pool: StagePool,
        request_id: str,
        tx_ms: float,
    ) -> None:
        """Emit per-edge transfer_tx_s + transfer_size_bytes histograms.

        ``tx_ms`` is the orchestrator-side wall-clock spent in ``next_pool.
        submit_*`` (serialize + queue submit to the receiving worker). Best-
        effort size_bytes left at 0 — orchestrator doesn't have a cheap handle
        on the serialized payload size; a follow-up can plumb that from the
        connector adapter.
        """
        if self._transfer_emitter is None:
            return
        to_replica = to_pool.get_bound_replica_id(request_id)
        if to_replica is None:
            return
        try:
            self._transfer_emitter.observe_size(from_stage, from_replica, to_stage, to_replica, 0)
            self._transfer_emitter.observe_tx_time(from_stage, from_replica, to_stage, to_replica, tx_ms / 1000.0)
        except Exception:
            logger.debug(
                "[Orchestrator] transfer_tx emit failed for edge %d->%d req=%s",
                from_stage,
                to_stage,
                request_id,
                exc_info=True,
            )

    async def _forward_to_next_stage(
        self,
        req_id: str,
        src_stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        src_replica_id: int | None = None,
        is_streaming_session: bool = False,
        is_final_update: bool = False,
    ) -> None:
        """Forward output to the next stage; a dead stage fails only this request."""
        await self._dispatch_or_fail_request(
            lambda: self._forward_to_next_stage_unguarded(
                req_id,
                src_stage_id,
                output,
                req_state,
                src_replica_id=src_replica_id,
                is_streaming_session=is_streaming_session,
                is_final_update=is_final_update,
            ),
            req_id=req_id,
            stage_id=src_stage_id + 1,
            operation="inter-stage forward",
        )

    async def _forward_to_next_stage_unguarded(
        self,
        req_id: str,
        src_stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        src_replica_id: int | None = None,
        is_streaming_session: bool = False,
        is_final_update: bool = False,
    ) -> None:
        """Forward output from the current logical stage to the next one."""
        next_logical = src_stage_id + 1
        next_pool = self.stage_pools[next_logical]
        if not next_pool.live_replica_ids():
            # The downstream stage lost all of its replicas before this in-flight
            # request could be forwarded.
            await self._fail_request_dead_stage(req_id, next_logical)
            return
        next_client = next_pool.stage_client
        params = req_state.sampling_params_list[next_logical]
        source_outputs = [output]
        next_stage_resumable = is_streaming_session and not is_final_update
        already_submitted = self._next_stage_already_submitted(src_stage_id, req_state)
        requires_multimodal_data = getattr(next_client, "requires_multimodal_data", False)
        _t_submit_start = _time.perf_counter()

        if next_pool.stage_type == "diffusion":
            # Gate: never dispatch with an incomplete CFG bundle. Checked
            # non-destructively BEFORE popping — a pop-then-redefer would lose
            # the partial outputs. all_companions_done is trustworthy here
            # because done-marking happens only after set_companion_output.
            if self._cfg_tracker.has_companions(req_id) and not self._cfg_tracker.all_companions_done(req_id):
                self._cfg_tracker.defer_parent(req_id, output, src_stage_id)
                logger.info(
                    "[Orchestrator] req=%s: CFG companion outputs not all stashed yet; re-deferring parent",
                    req_id,
                )
                return
            # Peek, don't pop: outputs stay stashed until cleanup_parent so a
            # streaming re-submission bundles the complete set again rather
            # than an empty one.
            companion_outputs = self._cfg_tracker.get_companion_outputs(req_id)
            expected = len(self._cfg_tracker.get_companion_request_ids(req_id))
            if expected > len(companion_outputs):
                # Companions are done but outputs are missing — inconsistent
                # tracker state (should be unreachable with peek semantics).
                # Fail the request rather than degrade CFG conditioning.
                logger.error(
                    "[Orchestrator] req=%s: only %d/%d CFG companion outputs available; "
                    "failing the request instead of dispatching degraded CFG",
                    req_id,
                    len(companion_outputs),
                    expected,
                )
                await self.output_async_queue.put(
                    ErrorMessage(
                        request_id=req_id,
                        stage_id=src_stage_id,
                        error=(
                            f"CFG companion outputs incomplete ({len(companion_outputs)}/{expected}); "
                            "request aborted to avoid degraded CFG conditioning"
                        ),
                    )
                )
                await self._cleanup_request_ids(
                    [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                    abort=True,
                )
                return
            diffusion_source_outputs = [output, *companion_outputs]
            if next_client.custom_process_input_func is not None:
                _t_ar2d = _time.perf_counter()
                _fn = next_client.custom_process_input_func
                _extra_kwargs: dict[str, Any] = {}
                # TODO: replace signature probe with explicit kwarg contract.
                try:
                    import inspect as _inspect

                    if "sampling_params" in _inspect.signature(_fn).parameters:
                        _extra_kwargs["sampling_params"] = params
                except (TypeError, ValueError):
                    pass
                diffusion_prompt = _fn(
                    diffusion_source_outputs,
                    req_state.prompt,
                    requires_multimodal_data,
                    **_extra_kwargs,
                )
                _dt_ar2d = (_time.perf_counter() - _t_ar2d) * 1000
                req_state.pipeline_timings["ar2diffusion_ms"] = _dt_ar2d
                logger.info(
                    "[Orchestrator] ar2diffusion req=%s wall_time=%.3fms stage=%d->%d",
                    req_id,
                    _dt_ar2d,
                    src_stage_id,
                    next_logical,
                )
                if diffusion_prompt is None:
                    error_output = OmniRequestOutput.from_error(
                        req_id,
                        f"Stage-{src_stage_id} produced no valid inputs for diffusion stage-{next_logical}",
                    )
                    logger.warning(
                        "[Orchestrator] req=%s stage=%d produced empty diffusion inputs for stage=%d; "
                        "routing terminal error output",
                        req_id,
                        src_stage_id,
                        next_logical,
                    )
                    await self.output_async_queue.put(
                        OutputMessage(
                            request_id=req_id,
                            stage_id=next_logical,
                            engine_outputs=error_output,
                            metrics=None,
                            finished=True,
                        )
                    )
                    await self._cleanup_request_ids(
                        [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                    )
                    return
                if isinstance(diffusion_prompt, list):
                    if not diffusion_prompt:
                        error_output = OmniRequestOutput.from_error(
                            req_id,
                            f"Stage-{src_stage_id} produced no valid inputs for diffusion stage-{next_logical}",
                        )
                        logger.warning(
                            "[Orchestrator] req=%s stage=%d produced empty diffusion inputs for stage=%d; "
                            "routing terminal error output",
                            req_id,
                            src_stage_id,
                            next_logical,
                        )
                        await self.output_async_queue.put(
                            OutputMessage(
                                request_id=req_id,
                                stage_id=next_logical,
                                engine_outputs=error_output,
                                metrics=None,
                                finished=True,
                            )
                        )
                        await self._cleanup_request_ids(
                            [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                        )
                        return
                    if len(diffusion_prompt) == 1:
                        diffusion_prompt = diffusion_prompt[0]
            else:
                diffusion_prompt = req_state.prompt

            if already_submitted:
                replica_id = await next_pool.submit_update(req_id, req_state, diffusion_prompt)
            else:
                replica_id = await next_pool.submit_initial(
                    req_id,
                    req_state,
                    diffusion_prompt,
                    submit_kwargs={
                        "kv_sender_info": self._build_kv_sender_info(
                            list(getattr(next_client, "engine_input_source", None) or [src_stage_id]),
                            request_id=req_id,
                        )
                    },
                    params_override=self._maybe_clone_diffusion_params_for_cfg(req_id, params),
                )
            self._record_duplex_stage_submission(
                next_logical,
                req_id,
                replica_id,
                req_state,
            )
            req_state.stage_submit_ts[next_logical] = _time.time()
            _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
            self._emit_tx_edge(
                from_stage=src_stage_id,
                from_replica=src_replica_id if src_replica_id is not None else 0,
                to_stage=next_logical,
                to_pool=next_pool,
                request_id=req_id,
                tx_ms=_tx_ms,
            )
            return

        # PD disaggregation: prefill → decode routing uses original prompt + KV transfer params
        if self._pd_pair is not None and (src_stage_id, next_logical) == self._pd_pair:
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
                    model_config=next_pool.stage_vllm_config.model_config,
                    mm_features=req_state.mm_features,
                    resumable=next_stage_resumable,
                )
                request.external_req_id = request.request_id
                if already_submitted:
                    replica_id = await next_pool.submit_update(req_id, req_state, request)
                else:
                    replica_id = await next_pool.submit_initial(req_id, req_state, request, prompt_text=None)
                self._record_duplex_stage_submission(
                    next_logical,
                    req_id,
                    replica_id,
                    req_state,
                )

            req_state.stage_submit_ts[next_logical] = _time.time()
            _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
            self._emit_tx_edge(
                from_stage=src_stage_id,
                from_replica=src_replica_id if src_replica_id is not None else 0,
                to_stage=next_logical,
                to_pool=next_pool,
                request_id=req_id,
                tx_ms=_tx_ms,
            )
            return

        if req_state.pd_prefill_multimodal_output is not None:
            req_state.streaming.bridge_states.setdefault(
                "pd_prefill_multimodal_output_by_req",
                {},
            )[req_id] = req_state.pd_prefill_multimodal_output

        previous_decoder = req_state.streaming.source_token_decoder
        source_processor = self.stage_pools[src_stage_id].output_processor
        tokenizer = getattr(source_processor, "tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            req_state.streaming.source_token_decoder = decode

        try:
            next_inputs = next_client.process_engine_inputs(
                source_outputs,
                req_state.prompt,
                streaming_context=req_state.streaming,
            )
        except Exception:
            logger.exception(
                "[Orchestrator] req=%s process_engine_inputs FAILED for stage-%s",
                req_id,
                next_logical,
            )
            raise
        finally:
            req_state.streaming.source_token_decoder = previous_decoder

        if not next_inputs:
            if not getattr(output, "finished", False):
                logger.debug(
                    "[Orchestrator] req=%s stage-%s produced no inputs for stage-%s; waiting for more outputs",
                    req_id,
                    src_stage_id,
                    next_logical,
                )
                return

            final_stage_id = req_state.final_stage_id
            final_pool = self.stage_pools[final_stage_id]
            final_output_type = getattr(final_pool.stage_client, "final_output_type", None)
            terminal_output = _build_terminal_empty_output(
                req_id,
                final_output_type=final_output_type,
                audio_sample_rate=final_pool._infer_audio_sample_rate(),
            )
            submit_ts = _time.time()
            req_state.stage_submit_ts[final_stage_id] = submit_ts
            logger.info(
                "[Orchestrator] req=%s stage-%s produced no terminal inputs for stage-%s; "
                "returning empty %s output from final stage-%s",
                req_id,
                src_stage_id,
                next_logical,
                final_output_type or "text",
                final_stage_id,
            )
            await self.output_async_queue.put(
                OutputMessage(
                    request_id=req_id,
                    stage_id=final_stage_id,
                    replica_id=0,
                    engine_outputs=terminal_output,
                    metrics=None,
                    finished=True,
                    stage_submit_ts=submit_ts,
                )
            )
            await self._cleanup_request_ids([req_id, *self._cfg_tracker.cleanup_parent(req_id)])
            return

        # Build and submit requests for each input
        for next_input in next_inputs:
            # Only AR thinker stages consume encoder mm_features; downstream
            # (talker/code2wav/…) must not see them (avoids encoder-cache misses).
            model_stage = getattr(getattr(next_pool.stage_vllm_config, "model_config", None), "model_stage", None)
            mm_features = req_state.mm_features if model_stage == "thinker" else None
            request = self._build_next_stage_request(
                req_id,
                next_logical,
                next_input,
                params=params,
                mm_features=mm_features,
                resumable=next_stage_resumable,
            )

            if already_submitted:
                replica_id = await next_pool.submit_update(req_id, req_state, request)
            else:
                replica_id = await next_pool.submit_initial(req_id, req_state, request, prompt_text=None)
            self._record_duplex_stage_submission(
                next_logical,
                req_id,
                replica_id,
                req_state,
            )

        req_state.stage_submit_ts[next_logical] = _time.time()
        _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
        self._emit_tx_edge(
            from_stage=src_stage_id,
            from_replica=src_replica_id if src_replica_id is not None else 0,
            to_stage=next_logical,
            to_pool=next_pool,
            request_id=req_id,
            tx_ms=_tx_ms,
        )

    async def _prewarm_async_chunk_stages(
        self,
        request_id: str,
        stage0_request: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Pre-submit downstream stages for async-chunk mode."""
        if req_state.final_stage_id <= 0:
            return

        prompt_token_ids = getattr(stage0_request, "prompt_token_ids", None)
        if prompt_token_ids is None:
            logger.warning(
                "[Orchestrator] async_chunk prewarm skipped for req=%s: stage0 prompt_token_ids missing",
                request_id,
            )
            return

        for next_stage_id in range(1, req_state.final_stage_id + 1):
            next_pool = self.stage_pools[next_stage_id]
            params = req_state.sampling_params_list[next_stage_id]
            if not self._stage_receives_async_chunks(next_stage_id):
                # Outgoing-only stages receive their first real input from the
                # orchestrator. Pre-submitting a placeholder lets it race and
                # execute before that conditioning payload arrives.
                continue

            req_state.stage_submit_ts[next_stage_id] = _time.time()
            _t_submit_start = _time.perf_counter()

            if next_pool.stage_type == "diffusion":
                submit_kwargs = {
                    "kv_sender_info": self._build_kv_sender_info(
                        list(getattr(next_pool.stage_client, "engine_input_source", None) or [next_stage_id - 1]),
                        request_id=request_id,
                    )
                }
                submitted = await self._dispatch_or_fail_request(
                    lambda: next_pool.submit_initial(
                        request_id,
                        req_state,
                        req_state.prompt,
                        submit_kwargs=submit_kwargs,
                    ),
                    req_id=request_id,
                    stage_id=next_stage_id,
                    operation="async-chunk prewarm",
                )
            else:
                import copy

                from vllm_omni.distributed.omni_connectors.adapter import compute_talker_prompt_ids_length

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
                downstream_resumable = bool(getattr(stage0_request, "resumable", req_state.streaming.enabled))
                request = build_engine_core_request_from_tokens(
                    request_id=request_id,
                    prompt=base_input,
                    params=params,
                    model_config=next_pool.stage_vllm_config.model_config,
                    resumable=downstream_resumable,
                )
                request.external_req_id = request.request_id
                submitted = await self._dispatch_or_fail_request(
                    lambda: next_pool.submit_initial(
                        request_id,
                        req_state,
                        request,
                        prompt_text=None,
                    ),
                    req_id=request_id,
                    stage_id=next_stage_id,
                    operation="async-chunk prewarm",
                )

            # Attribute the failure to the stage that failed, not stage 0; the
            # request is already cleaned up, so stop prewarming.
            if not submitted:
                return

            # The guard swallows submit_initial's return value; the pool binding
            # it recorded carries the same replica.
            bound_replica_id = next_pool.get_bound_replica_id(request_id)
            self._record_duplex_stage_submission(
                next_stage_id,
                request_id,
                bound_replica_id if bound_replica_id is not None else 0,
                req_state,
            )

            # async_chunk pre-submit fires per stage edge (N-1 -> N). Source
            # replica is stage 0's bound replica (single-replica thinker in
            # all current configs); fall back to 0 if unknown.
            _tx_ms = (_time.perf_counter() - _t_submit_start) * 1000.0
            src_replica = self.stage_pools[next_stage_id - 1].get_bound_replica_id(request_id)
            self._emit_tx_edge(
                from_stage=next_stage_id - 1,
                from_replica=src_replica if src_replica is not None else 0,
                to_stage=next_stage_id,
                to_pool=next_pool,
                request_id=request_id,
                tx_ms=_tx_ms,
            )

    def _build_kv_sender_info(
        self,
        sender_stage_ids: list[int],
        *,
        request_id: str | None = None,
    ) -> dict[int, dict[str, Any]] | None:
        """Build per-request sender info for diffusion KV-transfer receivers."""
        sender_infos: dict[int, dict[str, Any]] = {}
        for sender_stage_id in dict.fromkeys(sender_stage_ids):
            if sender_stage_id < 0 or sender_stage_id >= len(self.stage_pools):
                continue

            sender_pool = self.stage_pools[sender_stage_id]
            sender_stage = sender_pool.get_bound_client(request_id) if request_id is not None else None
            if sender_stage is None:
                sender_stage = sender_pool.stage_client
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

    # ---- Shutdown / lifecycle ----

    def _shutdown_stages(self) -> None:
        """Shutdown all stage pools."""
        if self._stages_shutdown:
            return

        self._stages_shutdown = True
        total = sum(pool.live_num_replicas for pool in self.stage_pools)
        logger.info("[Orchestrator] Shutting down all %d client(s)", total)
        for pool in self.stage_pools:
            for replica_id in pool.live_replica_ids():
                pool.shutdown_replica(replica_id)

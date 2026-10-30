# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""DuplexOrchestrator: the duplex sibling of ``Orchestrator``.

It hosts the ``DuplexSessionManager`` (admission, leases, session runners) on
the orchestrator loop and implements the ``DuplexStagePort`` the runners use
to submit resumable Stage0 requests. Generic stage forwarding, prewarm and
cleanup stay in ``OrchestratorBase``; this class only fills the template seams
and applies the session-owned policy.
"""

from __future__ import annotations

import time as _time
from collections.abc import Coroutine
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.engine import FinishReason
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.core.sched.duplex_stage_work import PERF_CHUNK_KEY, stage_work_rows
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreRequest
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexOutputContext,
    DuplexRequestIdentity,
    DuplexStagePort,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
    duplex_is_auxiliary_request,
)
from vllm_omni.engine.duplex.messages import DuplexSessionError
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, validate_duplex_replica_policy
from vllm_omni.engine.duplex.replica_placement import DuplexReplicaPlacement
from vllm_omni.engine.duplex.session.engine_session import DuplexFenceMismatchError
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.messages import EngineQueueMessage, OutputMessage
from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    build_engine_core_request_from_tokens,
)
from vllm_omni.engine.stage_pool import StageUnavailableError

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.outputs import RequestOutput

    from vllm_omni.metrics.stats import StageRequestStats

logger = init_logger(__name__)


@dataclass
class DuplexOrchestratorRequestState(OrchestratorRequestState):
    """Request bookkeeping for a session-owned (resumable duplex) stage request."""

    session_id: str = ""
    fence: DuplexFence | None = None
    stage_fences: dict[int, DuplexFence] = field(default_factory=dict)
    config_generation: int = -1
    # Session stats (RFC #8325): per stage, the chunks with spans since its last
    # session output, and the chunk of its latest producing step.
    pending_perf_chunks: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    stage_output_chunks: dict[int, tuple[int, int]] = field(default_factory=dict)


class DuplexOrchestrator(Orchestrator, DuplexStagePort):
    """Stage management for a duplex deployment; owns one ``DuplexSessionManager``.

    Extends the turn-based orchestrator rather than sitting beside it, so one
    engine serves both a duplex session and an ordinary request. The
    dependency runs duplex -> turn-based and never the reverse, which is what
    keeps ``Orchestrator`` free of duplex code.
    """

    def __init__(
        self,
        # Any: ``*args`` / ``**kwargs`` are forwarded verbatim to ``OrchestratorBase.__init__``.
        *args: Any,
        plugin: DuplexModelPlugin,
        duplex_session_config: DuplexSessionRuntimeConfig,
        model_config: ModelConfig | None = None,
        frontend_count: int = 1,
        frontend_index: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.plugin = plugin
        self.duplex_session_config = duplex_session_config
        self.replica_placement: DuplexReplicaPlacement | None = None
        admission_limit = None
        if duplex_session_config.replica_policy == "least_sessions":
            if self._membership is not None or any(
                pool.is_distributed or pool._lb is not None for pool in self.stage_pools
            ):
                validate_duplex_replica_policy(plugin, duplex_session_config, distributed=True)
            cap = duplex_session_config.max_sessions_per_replica
            assert cap is not None
            self.replica_placement = DuplexReplicaPlacement(
                stage_replica_counts=[pool.num_replicas for pool in self.stage_pools],
                max_sessions_per_replica=cap,
                replica_groups=duplex_session_config.replica_groups,
                frontend_count=frontend_count,
                frontend_index=frontend_index,
            )
            if frontend_count > 1:
                admission_limit = self.replica_placement.frontend_session_limit(duplex_session_config.max_sessions)
        self.session_manager = DuplexSessionManager(
            plugin=plugin,
            stage_port=self,
            output_sink=self.output_async_queue,
            result_sink=self.rpc_async_queue,
            runtime_config=self.duplex_session_config,
            model_config=model_config,
            admission_limit=admission_limit,
            log_stats=self.log_stats,
        )

    def duplex_stage_names(self) -> list[str]:
        """Each stage's model role (``thinker``, ``talker``, ...) for session stats."""
        names = []
        for stage_id, pool in enumerate(self.stage_pools):
            model_config = getattr(getattr(pool, "stage_vllm_config", None), "model_config", None)
            name = getattr(model_config, "model_stage", None)
            names.append(name if isinstance(name, str) and name else f"stage{stage_id}")
        return names

    # ------------------------------------------------------------------ #
    # Template seams                                                     #
    # ------------------------------------------------------------------ #

    async def _dispatch_message(self, msg: EngineQueueMessage) -> bool:
        if self.session_manager.accepts(msg):
            self.session_manager.dispatch(msg)
            return True
        # Not a session message: it is an ordinary turn-based request.
        return await super()._dispatch_message(msg)

    # Any: ``Coroutine``'s send/yield parameters, as in the ``OrchestratorBase`` seam.
    def _background_tasks(self) -> list[Coroutine[Any, Any, None]]:
        return [self.session_manager.reaper_loop(self._shutdown_event)]

    async def _shutdown_extensions(self) -> None:
        await self.session_manager.shutdown()

    def _session_route(self, stage_id: int, req_state: OrchestratorRequestState) -> int | None:
        if self.replica_placement is None or not isinstance(req_state, DuplexOrchestratorRequestState):
            return None
        try:
            replica_id = self.replica_placement.route(req_state.session_id, stage_id)
        except DuplexSessionError as exc:
            raise StageUnavailableError(str(exc)) from exc
        if not self.stage_pools[stage_id].is_replica_available(replica_id):
            raise StageUnavailableError(f"stage {stage_id}: pinned duplex replica {replica_id} is unavailable")
        return replica_id

    async def _handle_dead_replica(
        self,
        stage_id: int,
        replica_id: int,
        error: EngineDeadError,
        *,
        failed_request_id: str | None = None,
    ) -> None:
        if self.replica_placement is not None:
            affected = self.replica_placement.mark_unavailable(stage_id, replica_id)
            # Prevent dispatch before any cleanup await. Preserve request bindings
            # so the generic failure path can still attribute ordinary requests.
            self.stage_pools[stage_id].evict_replica(replica_id)
            await self.session_manager.fail_sessions(
                affected, reason=f"Stage-{stage_id} replica-{replica_id} is unavailable: {error}"
            )
        await super()._handle_dead_replica(stage_id, replica_id, error, failed_request_id=failed_request_id)

    def _on_stage_submitted(
        self,
        stage_id: int,
        request_id: str,
        replica_id: int,
        req_state: OrchestratorRequestState,
    ) -> None:
        del replica_id
        if not isinstance(req_state, DuplexOrchestratorRequestState) or req_state.fence is None:
            return
        runner = self.session_manager.runner_for_request_id(request_id)
        if runner is None:
            return
        fence = req_state.fence
        req_state.stage_fences[stage_id] = fence
        try:
            runner.session.bind_stage_request(stage_id, request_id, fence=fence)
        except DuplexFenceMismatchError:
            # The session already advanced past this request's epoch (cancel
            # raced the submit); the cancel path aborts the stale request id.
            logger.debug("[DuplexOrchestrator] stale stage binding ignored for %s stage-%s", request_id, stage_id)
            return
        req_state.stage_submit_ts[stage_id] = _time.time()
        if req_state.request_timestamp <= 0.0:
            req_state.request_timestamp = req_state.stage_submit_ts[stage_id]
        self._register_running_request(req_state)
        self.session_manager.register_request(request_id, req_state.session_id)

    async def _intercept_stage_output(
        self,
        stage_id: int,
        replica_id: int,
        output: RequestOutput,
        req_state: OrchestratorRequestState,
        stage_metrics: StageRequestStats | None,
        submit_ts: float | None,
    ) -> bool:
        del replica_id, submit_ts
        if not isinstance(req_state, DuplexOrchestratorRequestState) or req_state.fence is None:
            return False
        request_id = output.request_id
        runner = self.session_manager.runner_for_request_id(request_id)
        if runner is None:
            # Session gone: nothing may forward or reach a client.
            return True
        segment = req_state.streaming.segment(stage_id)
        # A codec segment stop is not the utterance. Leave RequestOutput.finished
        # for the real stop (is_segment_finished is false on that poll).
        if req_state.streaming.enabled and segment.finished and getattr(output, "finished", False):
            output.finished = False
        finished = bool(getattr(output, "finished", False)) or (req_state.streaming.enabled and segment.finished)
        transcript = self.plugin.user_transcript(
            stage_id=stage_id,
            output=output,
            prompt=getattr(req_state, "prompt", None),
            finished=finished,
        )
        prompt = getattr(req_state, "prompt", None)
        info = prompt.get("additional_information") if isinstance(prompt, dict) else None
        item_id = info.get("realtime_item_id") if isinstance(info, dict) else None
        context = DuplexOutputContext(
            identity=DuplexRequestIdentity(
                session_id=req_state.session_id,
                fence=req_state.stage_fences.get(stage_id, req_state.fence),
            ),
            final_stage_id=req_state.final_stage_id,
            segment_finished=finished,
            segment_token_ids=tuple(segment.token_ids),
            segment_output_metadata=segment.output_metadata,
            perf_chunks=self._take_perf_chunks(req_state, stage_id),
        )
        return runner.on_stage_output(
            stage_id,
            output,
            stage_metrics,
            request_id=request_id,
            context=context,
            input_transcript=transcript,
            input_item_id=item_id if isinstance(item_id, str) else None,
        )

    def _collect_duplex_stage_work(
        self,
        stage_id: int,
        eco: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        if not isinstance(req_state, DuplexOrchestratorRequestState):
            return
        rows = stage_work_rows(eco.stage_work)
        if not rows:
            return
        runner = self.session_manager.runners.get(req_state.session_id)
        if runner is not None:
            # Now, not with a later session output: the step may never produce one.
            runner.on_stage_work([(stage_id, *row) for row in rows])
        req_state.pending_perf_chunks.setdefault(stage_id, set()).update((row[0], row[1]) for row in rows)
        # The recorder puts the producing step's chunk last.
        req_state.stage_output_chunks[stage_id] = (rows[-1][0], rows[-1][1])

    @staticmethod
    def _take_perf_chunks(req_state: DuplexOrchestratorRequestState, stage_id: int) -> frozenset[tuple[int, int]]:
        """The chunks a session output of ``stage_id`` finishes handling for.

        Those with spans since the stage's last output, plus its producing chunk:
        outputs and raw outputs need not pair one to one.
        """
        chunks = req_state.pending_perf_chunks.pop(stage_id, set())
        output_chunk = req_state.stage_output_chunks.get(stage_id)
        if output_chunk is not None:
            chunks.add(output_chunk)
        return frozenset(chunks)

    def _annotate_next_stage_inputs(
        self,
        src_stage_id: int,
        req_state: OrchestratorRequestState,
        next_inputs: list[Any],
    ) -> None:
        """Carry the source step's chunk id into the forwarded duplex envelope (RFC #8325)."""
        if not isinstance(req_state, DuplexOrchestratorRequestState):
            return
        chunk = req_state.stage_output_chunks.get(src_stage_id)
        if chunk is None:
            return
        for next_input in next_inputs:
            buffer = next_input.get("model_intermediate_buffer") if isinstance(next_input, dict) else None
            duplex = buffer.get("duplex") if isinstance(buffer, dict) else None
            if isinstance(buffer, dict) and isinstance(duplex, dict):
                # Copy both levels: the stage-input processor may share them with bridge state.
                next_input["model_intermediate_buffer"] = {
                    **buffer,
                    "duplex": {**duplex, PERF_CHUNK_KEY: list(chunk)},
                }

    async def _report_duplex_session_request_error(
        self,
        stage_id: int,
        replica_id: int | None,
        eco: OmniEngineCoreOutput,
        req_state: OrchestratorRequestState,
    ) -> None:
        if not req_state.session_owned or getattr(eco, "finish_reason", None) != FinishReason.ERROR:
            return
        # Segment boundaries do not terminate a resumable request. Context
        # overflow emits a separate terminal ERROR with is_segment_finished=False.
        if getattr(eco, "is_segment_finished", False):
            return
        reason = getattr(eco, "stop_reason", None)
        error = reason if isinstance(reason, str) and reason else "duplex session request failed"
        # Duplex sessions receive session events, not the ordinary request queue's
        # ErrorMessage. Close a resident session before another append can recreate
        # its finished stage request without the model's voice/persona prefill.
        await self._handle_forward_failure(req_state.request_id, stage_id, req_state, RuntimeError(error))

    async def _handle_forward_failure(
        self,
        req_id: str,
        next_stage_id: int,
        req_state: OrchestratorRequestState,
        exc: BaseException,
    ) -> bool:
        if not req_state.session_owned:
            return False
        # Resumable resident Stage0: stage failure closes the session.
        # Ephemeral turn-commit: free this turn's stages without tearing down WS.
        close_session = (
            not duplex_is_auxiliary_request(req_id)
            and self.plugin.capabilities(
                max_sessions=self.duplex_session_config.max_sessions
            ).supports_core_resumable_request
        )
        runner = self.session_manager.runner_for_request_id(req_id)
        if runner is not None:
            runner.on_stage_failure(next_stage_id, exc, request_id=req_id)
        await self._cleanup_request_ids(
            [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
            abort=True,
            release_owners=close_session,
        )
        return True

    async def _cleanup_request_ids(
        self,
        request_ids: list[str],
        *,
        abort: bool = False,
        release_owners: bool = False,
    ) -> list[OutputMessage]:
        if not request_ids:
            return []
        cleanup_ids = list(dict.fromkeys(request_ids))
        closing_session_ids: list[str] = []
        if release_owners:
            closed_sessions = self.session_manager.close_sessions_for_request_ids(
                cleanup_ids,
                abort=abort,
                cleanup_in_progress=True,
            )
            closing_session_ids.extend(closed_sessions)
            for session_id, stale_request_ids in closed_sessions.items():
                logger.info(
                    "[DuplexOrchestrator] closed duplex session %s while cleaning failed request ids %s",
                    session_id,
                    stale_request_ids,
                )
                cleanup_ids.extend(stale_request_ids)
            cleanup_ids = list(dict.fromkeys(cleanup_ids))
        try:
            outputs = await super()._cleanup_request_ids(cleanup_ids, abort=abort)
        except BaseException:
            if closing_session_ids:
                self.session_manager.defer_request_cleanups(closing_session_ids)
            raise
        if closing_session_ids:
            self.session_manager.finalize_closed_sessions(closing_session_ids)
        for request_id in cleanup_ids:
            self.session_manager.unregister_request(request_id)
        return outputs

    # ------------------------------------------------------------------ #
    # DuplexStagePort                                                    #
    # ------------------------------------------------------------------ #

    @property
    def stage_count(self) -> int:
        return len(self.stage_pools)

    def reserve_session(self, session_id: str) -> None:
        if self.replica_placement is None:
            return
        try:
            reservation = self.replica_placement.reserve(session_id)
        except DuplexSessionError as exc:
            if exc.code == "resource_exhausted":
                self.reject_session("replica_capacity_exhausted")
            raise
        try:
            snapshot = self.replica_placement.snapshot()
            occupancy = {
                stage: snapshot["reserved"][(stage, replica)] for stage, replica in reservation.replicas.items()
            }
            logger.info(
                "duplex_session_placement session_id=%s group=%s replicas=%s reserved=%s",
                session_id,
                reservation.group_id,
                dict(reservation.replicas),
                occupancy,
            )
            self._update_placement_metrics()
        except BaseException:
            self.replica_placement.release(session_id)
            raise

    def release_session(self, session_id: str) -> None:
        if self.replica_placement is not None:
            self.replica_placement.release(session_id)
            self._update_placement_metrics()

    def reject_session(self, reason: str) -> None:
        if self.replica_placement is not None and self._prom_metrics is not None:
            self._prom_metrics.inc_duplex_admission_rejected(reason)

    def _update_placement_metrics(self) -> None:
        if self.replica_placement is not None and self._prom_metrics is not None:
            for (stage, replica), reserved in self.replica_placement.snapshot()["reserved"].items():
                self._prom_metrics.set_duplex_reserved_sessions(stage, replica, reserved)

    def sampling_defaults(self) -> tuple[object, ...]:
        defaults = []
        for pool in self.stage_pools:
            client = pool.stage_client
            if client is None:
                # Every replica of this stage was evicted (engine core died); a
                # session cannot be admitted until the deployment is restarted.
                raise RuntimeError(f"stage {pool.stage_id} has no live replica")
            defaults.append(client.default_sampling_params)
        return tuple(defaults)

    def _stage_receives_async_chunks(self, stage_id: int) -> bool:
        """Whether a stage's connector supplies its runtime inputs.

        Stages with a custom orchestrator input processor must be fed via
        process_engine_inputs, not zero-prewarm + connector chunks. Codec edges
        without a custom processor still use async chunk transport.
        """
        pool = self.stage_pools[stage_id]
        client = getattr(pool, "stage_client", None)
        if client is not None and getattr(client, "custom_process_input_func", None) is not None:
            return False
        return super()._stage_receives_async_chunks(stage_id)

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
        current_model_turn_id = duplex_state.get("model_turn_id")
        if (
            not isinstance(current_model_turn_id, int)
            or previous_epoch != context.fence.epoch
            or current_model_turn_id < context.fence.turn_id
        ):
            # A safety boundary can close a response before the model emits
            # its normal turn_eos. Catch up the engine-owned identity when the
            # next fenced append starts.
            duplex_state["model_turn_id"] = context.fence.turn_id
        duplex_state.update(
            {
                "session_id": context.session_id,
                "fence": context.fence,
                "epoch": context.fence.epoch,
                "turn_id": context.fence.turn_id,
                "session_config": dict(context.session_config),
                "runtime_config": dict(context.runtime_config),
            }
        )

    def ensure_request(self, context: DuplexStageRequestContext) -> None:
        if self.replica_placement is not None:
            replica_id = self.replica_placement.route(context.session_id, context.stage_id)
            if not self.stage_pools[context.stage_id].is_replica_available(replica_id):
                raise StageUnavailableError(f"stage {context.stage_id}: pinned duplex replica is unavailable")
        request_state = self.request_states.get(context.request_id)
        if request_state is None:
            request_state = DuplexOrchestratorRequestState(
                request_id=context.request_id,
                prompt=None,
                sampling_params_list=list(context.sampling_params),
                final_stage_id=context.final_stage_id,
                entry_stage_id=context.stage_id,
                request_timestamp=_time.time(),
                session_owned=True,
                session_id=context.session_id,
                fence=context.fence,
                config_generation=context.config_generation,
            )
            self.request_states[context.request_id] = request_state
        elif isinstance(request_state, DuplexOrchestratorRequestState):
            if request_state.config_generation != context.config_generation:
                request_state.sampling_params_list = list(context.sampling_params)
                request_state.config_generation = context.config_generation
            request_state.session_id = context.session_id
            request_state.fence = context.fence
            if request_state.request_timestamp <= 0.0:
                request_state.request_timestamp = _time.time()
        else:
            raise RuntimeError(f"request {context.request_id} is not session-owned")
        self._sync_bridge_state(request_state, context)
        self.session_manager.register_request(context.request_id, context.session_id)

    def _upgrade_processed_stage_request(self, request, raw_prompt):
        request = super()._upgrade_processed_stage_request(request, raw_prompt)
        if not self.plugin.capabilities(
            max_sessions=self.duplex_session_config.max_sessions
        ).supports_core_resumable_request and not isinstance(request, OmniEngineCoreRequest):
            request = OmniEngineCoreRequest.from_request(request)
        return request

    async def _build_next_stage_request(self, req_id, next_stage_id, next_input, params, **kwargs):
        state = self.request_states.get(req_id)
        if (
            self.replica_placement is not None
            and isinstance(state, DuplexOrchestratorRequestState)
            and isinstance(next_input, dict)
            and next_input.get("multi_modal_data")
            and not self._next_stage_input_is_tokens(next_input)
        ):
            # Keep the shared sender cache scoped to the session's pinned receiver.
            # The base builder performs scoping off-loop before input processing.
            replica_id = self.stage_pools[next_stage_id].get_bound_replica_id(req_id)
            if replica_id is None:
                raise StageUnavailableError(f"stage {next_stage_id}: duplex input has no pinned replica")
            kwargs["replica_id"] = replica_id
        return await super()._build_next_stage_request(req_id, next_stage_id, next_input, params, **kwargs)

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        context = submission.context
        input_guard = nullcontext() if submission.resumable else self._stage_input_submission(context.stage_id)
        async with input_guard as mark_submitting:
            request_state = self.request_states.get(context.request_id)
            if not isinstance(request_state, DuplexOrchestratorRequestState):
                raise RuntimeError(f"duplex request was not preregistered: {context.request_id}")
            pinned_replica_id = self._session_route(context.stage_id, request_state)
            if pinned_replica_id is not None:
                self.stage_pools[context.stage_id].bind_request_to_replica(context.request_id, pinned_replica_id)
            route_kwargs: dict[str, Any] = {} if pinned_replica_id is None else {"pinned_replica_id": pinned_replica_id}
            request_state.streaming.enabled = submission.resumable
            if submission.resumable:
                request = build_engine_core_request_from_tokens(
                    request_id=context.request_id,
                    prompt=dict(submission.prompt),
                    params=context.stage_sampling_params,
                    model_config=self.stage_pools[context.stage_id].stage_vllm_config.model_config,
                    resumable=True,
                )
            else:
                # Keep raw Stage0 prompt (additional_information / multi_modal_data) for
                # stage input processors via process_engine_inputs. Resumable requests
                # leave it unset: their prompt is one append with Stage0's duplex
                # buffer, which the async-chunk prewarm would copy downstream.
                request_state.prompt = dict(submission.prompt)
                # Use the ordinary multimodal input processor for turn-model plugins.
                # Its CPU preprocessing runs off the session/orchestrator event loop.
                request = await self._build_next_stage_request(
                    context.request_id,
                    context.stage_id,
                    dict(submission.prompt),
                    context.stage_sampling_params,
                    resumable=False,
                )
                if self.request_states.get(context.request_id) is not request_state:
                    raise RuntimeError("duplex request cancelled during input preprocessing")
            request.external_req_id = request.request_id
            mm_features = getattr(request, "mm_features", None)
            if mm_features is not None:
                request_state.mm_features = mm_features
            pool = self.stage_pools[context.stage_id]
            if mark_submitting is not None:
                mark_submitting()
            try:
                if submission.already_submitted:
                    if not submission.resumable:
                        raise RuntimeError(f"ephemeral duplex request cannot submit_update: {context.request_id}")
                    replica_id = await pool.submit_update(context.request_id, request_state, request, **route_kwargs)
                else:
                    replica_id = await pool.submit_initial(
                        context.request_id,
                        request_state,
                        request,
                        prompt_text=None,
                        **route_kwargs,
                    )
            except EngineDeadError as exc:
                if pinned_replica_id is not None:
                    await self._handle_dead_replica(context.stage_id, pinned_replica_id, exc)
                raise

        # The current-stage send has completed. Downstream prewarm
        # must not hold up another session's use of this stage's sender cache.
        if not submission.already_submitted and self.async_chunk and context.stage_id == 0:
            prewarmed = await self._prewarm_async_chunk_stages(context.request_id, request, request_state)
            if not prewarmed:
                # The prewarm already failed the request, aborted it and
                # popped its state; the runner turns this into an append
                # error instead of writing onto an orphaned object.
                raise RuntimeError(
                    f"async-chunk prewarm failed for duplex request {context.request_id}; the request was aborted"
                )
        request_state.stage_fences[context.stage_id] = context.fence
        now = _time.time()
        request_state.stage_submit_ts[context.stage_id] = now
        request_state.request_timestamp = now
        self._register_running_request(request_state)
        return DuplexStageSubmissionResult(
            request_id=context.request_id,
            stage_id=context.stage_id,
            replica_id=replica_id,
        )

    async def _route_output(
        self,
        stage_id: int,
        replica_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        stage_metrics: Any,
    ) -> None:
        plan = self.plugin.plan_partial_stage_output(self, stage_id, replica_id, output, req_state)
        if plan is not None:
            # A text-bearing final is not itself the end sentinel. Submit the
            # sentence resumable first; the follow-up, if any, closes the stream.
            await self._forward_to_next_stage(
                req_state.request_id,
                stage_id,
                plan.output,
                req_state,
                src_replica_id=replica_id,
                is_streaming_session=True,
                is_final_update=plan.is_final_update and not plan.queue_close_after,
            )
            followup = self.plugin.partial_stage_followup(plan, req_state)
            if followup is not None:
                await self._forward_to_next_stage(
                    req_state.request_id,
                    stage_id,
                    followup.output,
                    req_state,
                    src_replica_id=replica_id,
                    is_streaming_session=True,
                    is_final_update=followup.is_final_update,
                )
            # Sentence TTS already handed this Stage1 result to Talker. The
            # legacy path below would run aura2tts on the original full text
            # again when the next stage is not connector-fed (AURA Talker is
            # a sender: Stage1→2 is orchestrator-fed).
            req_state.skip_legacy_stage_forward = True
        try:
            await super()._route_output(stage_id, replica_id, output, req_state, stage_metrics)
        finally:
            req_state.skip_legacy_stage_forward = False

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
        await self._cleanup_request_ids(request_ids, abort=abort)

    async def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids:
            await self._abort_request_ids(list(dict.fromkeys(request_ids)))


__all__ = ["DuplexOrchestrator", "DuplexOrchestratorRequestState"]

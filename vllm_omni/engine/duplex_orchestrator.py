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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexOutputContext,
    DuplexRequestIdentity,
    DuplexStagePort,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
)
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin
from vllm_omni.engine.duplex.session import DuplexFenceMismatchError
from vllm_omni.engine.duplex.session_manager import DuplexSessionManager
from vllm_omni.engine.messages import EngineQueueMessage, OutputMessage
from vllm_omni.engine.orchestrator import (
    OrchestratorBase,
    OrchestratorRequestState,
    build_engine_core_request_from_tokens,
)

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


class DuplexOrchestrator(OrchestratorBase, DuplexStagePort):
    """Stage management for a duplex deployment; owns one ``DuplexSessionManager``."""

    def __init__(
        self,
        # Any: ``*args`` / ``**kwargs`` are forwarded verbatim to ``OrchestratorBase.__init__``.
        *args: Any,
        plugin: DuplexModelPlugin,
        duplex_session_config: DuplexSessionRuntimeConfig,
        model_config: ModelConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.plugin = plugin
        self.duplex_session_config = duplex_session_config
        self.session_manager = DuplexSessionManager(
            plugin=plugin,
            stage_port=self,
            output_sink=self.output_async_queue,
            result_sink=self.rpc_async_queue,
            runtime_config=self.duplex_session_config,
            model_config=model_config,
        )

    # ------------------------------------------------------------------ #
    # Template seams                                                     #
    # ------------------------------------------------------------------ #

    async def _dispatch_message(self, msg: EngineQueueMessage) -> bool:
        if self.session_manager.accepts(msg):
            self.session_manager.dispatch(msg)
            return True
        return False

    # Any: ``Coroutine``'s send/yield parameters, as in the ``OrchestratorBase`` seam.
    def _background_tasks(self) -> list[Coroutine[Any, Any, None]]:
        return [self.session_manager.reaper_loop(self._shutdown_event)]

    async def _shutdown_extensions(self) -> None:
        await self.session_manager.shutdown()

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
        context = DuplexOutputContext(
            identity=DuplexRequestIdentity(
                session_id=req_state.session_id,
                fence=req_state.stage_fences.get(stage_id, req_state.fence),
            ),
            final_stage_id=req_state.final_stage_id,
            segment_finished=req_state.streaming.enabled and segment.finished,
            segment_token_ids=tuple(segment.token_ids),
            segment_output_metadata=segment.output_metadata,
        )
        return runner.on_stage_output(
            stage_id,
            output,
            stage_metrics,
            request_id=request_id,
            context=context,
        )

    async def _handle_forward_failure(
        self,
        req_id: str,
        next_stage_id: int,
        req_state: OrchestratorRequestState,
        exc: BaseException,
    ) -> bool:
        if not req_state.session_owned:
            return False
        runner = self.session_manager.runner_for_request_id(req_id)
        if runner is not None:
            runner.on_stage_failure(next_stage_id, exc)
        await self._cleanup_request_ids(
            [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
            abort=True,
            release_owners=True,
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
        request_state = self.request_states.get(context.request_id)
        if request_state is None:
            request_state = DuplexOrchestratorRequestState(
                request_id=context.request_id,
                prompt=None,
                sampling_params_list=list(context.sampling_params),
                final_stage_id=context.final_stage_id,
                session_owned=True,
                session_id=context.session_id,
                fence=context.fence,
                config_generation=context.config_generation,
            )
            request_state.streaming.enabled = True
            self.request_states[context.request_id] = request_state
        elif isinstance(request_state, DuplexOrchestratorRequestState):
            if request_state.config_generation != context.config_generation:
                request_state.sampling_params_list = list(context.sampling_params)
                request_state.config_generation = context.config_generation
            request_state.session_id = context.session_id
            request_state.fence = context.fence
        else:
            raise RuntimeError(f"request {context.request_id} is not session-owned")
        self._sync_bridge_state(request_state, context)
        self.session_manager.register_request(context.request_id, context.session_id)

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        context = submission.context
        request_state = self.request_states.get(context.request_id)
        if not isinstance(request_state, DuplexOrchestratorRequestState):
            raise RuntimeError(f"duplex request was not preregistered: {context.request_id}")
        request = build_engine_core_request_from_tokens(
            request_id=context.request_id,
            prompt=dict(submission.prompt),
            params=context.stage_sampling_params,
            model_config=self.stage_pools[context.stage_id].stage_vllm_config.model_config,
            resumable=True,
        )
        request.external_req_id = request.request_id
        pool = self.stage_pools[context.stage_id]
        if submission.already_submitted:
            replica_id = await pool.submit_update(context.request_id, request_state, request)
        else:
            replica_id = await pool.submit_initial(context.request_id, request_state, request, prompt_text=None)
            if self.async_chunk and context.stage_id == 0:
                prewarmed = await self._prewarm_async_chunk_stages(
                    context.request_id,
                    request,
                    request_state,
                )
                if not prewarmed:
                    # The prewarm already failed the request, aborted it and
                    # popped its state; the runner turns this into an append
                    # error instead of writing onto an orphaned object.
                    raise RuntimeError(
                        f"async-chunk prewarm failed for duplex request {context.request_id}; the request was aborted"
                    )
        request_state.stage_fences[context.stage_id] = context.fence
        request_state.stage_submit_ts[context.stage_id] = _time.time()
        self._register_running_request(request_state)
        return DuplexStageSubmissionResult(
            request_id=context.request_id,
            stage_id=context.stage_id,
            replica_id=replica_id,
        )

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
        await self._cleanup_request_ids(request_ids, abort=abort)

    async def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids:
            await self._abort_request_ids(list(dict.fromkeys(request_ids)))


__all__ = ["DuplexOrchestrator", "DuplexOrchestratorRequestState"]

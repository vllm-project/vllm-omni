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

import asyncio
import threading
import time as _time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine import OmniEngineCoreRequest
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
from vllm_omni.engine.duplex.session.engine_session import DuplexFenceMismatchError
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.messages import EngineQueueMessage, OutputMessage
from vllm_omni.engine.orchestrator import (
    Orchestrator,
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
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._prompt_processing_lock = threading.Lock()
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
        # Not a session message: it is an ordinary turn-based request.
        return await super()._dispatch_message(msg)

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
        finished = bool(getattr(output, "finished", False)) or (req_state.streaming.enabled and segment.finished)
        transcript = self.plugin.user_transcript(
            stage_id=stage_id,
            output=output,
            prompt=getattr(req_state, "prompt", None),
            finished=finished,
        )
        if isinstance(transcript, str) and transcript:
            payload: dict[str, object] = {"type": "input.transcribed", "transcript": transcript}
            prompt = getattr(req_state, "prompt", None)
            info = prompt.get("additional_information") if isinstance(prompt, dict) else None
            item_id = info.get("realtime_item_id") if isinstance(info, dict) else None
            if isinstance(item_id, str) and item_id:
                payload["realtime_item_id"] = item_id
            runner.emit(payload)
        context = DuplexOutputContext(
            identity=DuplexRequestIdentity(
                session_id=req_state.session_id,
                fence=req_state.stage_fences.get(stage_id, req_state.fence),
            ),
            final_stage_id=req_state.final_stage_id,
            segment_finished=finished,
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
        # Resumable resident Stage0: stage failure closes the session.
        # Ephemeral turn-commit: free this turn's stages without tearing down WS.
        close_session = self.plugin.capabilities(
            max_sessions=self.duplex_session_config.max_sessions
        ).supports_core_resumable_request
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

    def _upgrade_processed_stage_request(self, request, raw_prompt):
        request = super()._upgrade_processed_stage_request(request, raw_prompt)
        if not self.plugin.capabilities(
            max_sessions=self.duplex_session_config.max_sessions
        ).supports_core_resumable_request and not isinstance(request, OmniEngineCoreRequest):
            request = OmniEngineCoreRequest.from_request(request)
        return request

    def _process_turn_prompt(self, *args, **kwargs):
        # Input processors own mutable caches. Hold a thread lock even if the
        # awaiting session is cancelled while its preprocessing is still running.
        with self._prompt_processing_lock:
            return self._build_next_stage_request(*args, **kwargs)

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        context = submission.context
        request_state = self.request_states.get(context.request_id)
        if not isinstance(request_state, DuplexOrchestratorRequestState):
            raise RuntimeError(f"duplex request was not preregistered: {context.request_id}")
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
            request = await asyncio.to_thread(
                self._process_turn_prompt,
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
        if submission.already_submitted:
            if not submission.resumable:
                raise RuntimeError(f"ephemeral duplex request cannot submit_update: {context.request_id}")
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

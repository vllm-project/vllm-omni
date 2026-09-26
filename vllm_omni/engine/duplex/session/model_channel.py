# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The model side of one duplex session.

Everything between the session and the model runtime: planning an append with
the plugin and submitting it to the stage port, turning the stage output that
comes back into session events, and -- when the model chooses to keep listening
-- offering it another unit of silence so the turn can continue.

Those three read as separate concerns but they call each other in a cycle. An
append submits and immediately projects its own first outputs; a projected
output can schedule a continuation; a continuation is another append. Splitting
them into three modules would mean three mutual imports and no fewer edges, so
they are one component whose seam with the runner is narrow instead.

That seam is three callbacks, for the three things this component triggers but
does not own: closing the session, scheduling the continuation append as a
tracked task, and aborting a stage request in the background.
"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

import pybase64 as base64
from vllm.logger import init_logger

from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputContext,
    DuplexOutputDecision,
    DuplexStageSubmission,
    duplex_data_plane_request_info,
    duplex_same_turn_request_ids,
    duplex_session_id_from_request_id,
)
from vllm_omni.engine.duplex.plugin import (
    DuplexRuntimeConfigError,
    coerce_int,
    payload_turn_id,
)
from vllm_omni.engine.duplex.session import helpers
from vllm_omni.engine.duplex.session.context import DuplexSessionContext, StageOutput
from vllm_omni.engine.duplex.session.emitter import SessionEmitter
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession, DuplexFenceMismatchError
from vllm_omni.engine.duplex.session.lease import DuplexLeaseActivity
from vllm_omni.metrics.duplex_frame_timing import log_audio_emit_event
from vllm_omni.metrics.stats import OrchestratorAggregator, StageRequestStats
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.duplex import attach_duplex_output_decision

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput

logger = init_logger(__name__)


class SilenceContinuationScheduler(Protocol):
    """Runs one continuation append as a tracked task (see ``_schedule_silence_continuation``)."""

    async def __call__(
        self,
        payload: dict[str, object],
        *,
        request_id: str,
        owner_id: str,
        response_id: str | None,
        response_owned: bool,
        expected_epoch: int | None,
        expected_model_turn_id: int | None,
    ) -> bool: ...


class ModelChannel:
    """Appends out to the model, events back from it, for one session."""

    # One MiniCPM model unit (1 s at 16 kHz) is the compatibility default.
    _SILENCE_UNIT_PAYLOAD_AUDIO = base64.b64encode(bytes(16000 * 4)).decode("ascii")
    _RESPONSE_MAX_CONTINUATION_UNITS = 8
    _AUTO_RESPONSE_MAX_CONTINUATION_UNITS = 64

    def __init__(
        self,
        ctx: DuplexSessionContext,
        out: SessionEmitter,
        *,
        close_from_runtime: Callable[[str], Awaitable[None]],
        schedule_silence_continuation: SilenceContinuationScheduler,
        abort_request: Callable[..., Awaitable[None]],
    ) -> None:
        self._ctx = ctx
        self._out = out
        self._close_from_runtime = close_from_runtime
        self._schedule_silence_continuation = schedule_silence_continuation
        self._abort_request = abort_request

    def _draining_stage_ids(self) -> frozenset[int]:
        """Stages the plugin keeps across a concurrent turn. Empty if undeclared."""
        declare = getattr(self._ctx.plugin, "draining_stage_ids", None)
        if not callable(declare):
            return frozenset()
        stage_count = int(getattr(self._ctx.stage_port, "stage_count", 0) or 0)
        return frozenset(int(stage_id) for stage_id in declare(stage_count=stage_count))

    @staticmethod
    def should_commit_response_to_history(session: DuplexEngineSession, response_id: str | None) -> bool:
        if response_id is not None and response_id != session.active_response_id:
            return True
        mode = session.response_config.extra_body.get("realtime_response_conversation")
        return not isinstance(mode, str) or mode.strip().lower() != "none"

    def response_created_payload(
        self,
        response_id: str,
        *,
        epoch: int,
        request_id: str | None = None,
    ) -> dict[str, object]:
        session = self._ctx.session
        response_config = session.response_config
        if not session.capabilities.supports_core_resumable_request and self.should_commit_response_to_history(
            session, response_id
        ):
            # Committed-turn models can receive the next utterance before the
            # speaker drains. Reserve the response's chronological slot now;
            # an ACK fills it later without moving it past newer user input.
            session.reserve_history_item(f"item_{response_id}")
        payload: dict[str, object] = {
            "type": "response.created",
            "session_id": session.session_id,
            "response_id": response_id,
            "epoch": epoch,
            "modalities": list(response_config.modalities),
        }
        if request_id is not None:
            payload["request_id"] = request_id
        metadata = response_config.extra_body.get("realtime_response_metadata")
        if not isinstance(metadata, dict):
            metadata = response_config.extra_body.get("realtime_metadata")
        if isinstance(metadata, dict):
            payload["metadata"] = dict(metadata)
        conversation = response_config.extra_body.get("realtime_response_conversation")
        if isinstance(conversation, str):
            payload["conversation"] = conversation
        prompt = response_config.extra_body.get("realtime_response_prompt")
        if isinstance(prompt, dict):
            payload["prompt"] = dict(prompt)
        return payload

    # ------------------------------------------------------------------ #
    # Submitting an append                                               #
    # ------------------------------------------------------------------ #

    async def append_runtime_input(
        self,
        payload: object,
        *,
        operation_id: str | None = None,
        final: bool,
        expected_epoch: int | None = None,
        on_append_accepted: Callable[[float], None] | None = None,
    ) -> tuple[bool, bool]:
        session = self._ctx.session
        if not session.capabilities.supports_input_append:
            return True, False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return True, False
        # Anchor the submission time before the RPC; the acceptance callback
        # commits timing state only if the append actually submitted.
        submit_time = time.monotonic()
        try:
            result = await self._append_via_data_plane(
                payload,
                final=final,
                operation_id=operation_id,
                expected_epoch=expected_epoch,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Failed to append duplex runtime input: %s", exc)
            self.send_runtime_error("runtime_append_failed", exc)
            return False, False
        if result is None:
            return True, False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return True, False
        self._mark_accepted_append_request_start(payload)
        # Commit timing state before returned output events can clear the
        # silence-continuation chain (e.g. a terminal turn-end).
        if on_append_accepted is not None:
            on_append_accepted(submit_time)
        request_id, _ = duplex_data_plane_request_info(result)
        if request_id is not None:
            self._ctx.plugin.data_plane.begin_request(request_id)
        close_reason, emitted_response = await self._send_model_output_events(result, expected_epoch=expected_epoch)
        emitted_response = emitted_response or request_id is not None
        if close_reason is None and self._start_data_plane_stream(result):
            emitted_response = True
        if close_reason is not None:
            await self._close_from_runtime(close_reason)
            return False, emitted_response
        return True, emitted_response

    def _mark_accepted_append_request_start(self, payload: object) -> None:
        """Stamp TTFT/TTFP origin only after the data plane accepted this append."""
        session = self._ctx.session
        turn_id = payload_turn_id(payload)
        if turn_id is None:
            turn_id = (
                session.active_response_turn_id if session.active_response_turn_id is not None else session.turn_id
            )
        session.mark_model_turn_request_started(turn_id, session._clock())

    async def _append_via_data_plane(
        self,
        payload: object,
        *,
        final: bool,
        operation_id: str | None,
        expected_epoch: int | None,
    ) -> dict[str, object] | None:
        """Submit one planned append to the stage port and bind the resulting stage request."""
        session = self._ctx.session
        if self._ctx.stage_port.stage_count == 0:
            raise RuntimeError("duplex_data_plane_has_no_stage")
        fence = helpers.append_fence(session, payload)
        lease_operation_id = f"append:{operation_id or uuid.uuid4().hex}"
        operation_started = False
        stage_id = 0
        resumable = session.capabilities.supports_core_resumable_request
        request_id = self._ctx.manager.stage_request_id(fence, stage_id=stage_id, resumable=resumable)
        # Ephemeral turn-commit cannot submit_update on a finished stage0 id.
        # Bump turn_id and open a fresh ephemeral request instead.
        if not resumable and session.stage_request_submitted(stage_id, request_id):
            stale_ephemeral_id = request_id
            # Keys are ``(stage_id, request_id)``; values are DuplexRequestResource.
            stale_keys = list(session.request_resources.keys())
            stale_ids = list(dict.fromkeys(rid for _, rid in stale_keys))
            concurrent_turn = session.capabilities.supports_concurrent_turn_requests and (
                self._ctx.run.concurrent_turn_requests_released
            )
            # Prior TTS may still drain under the same response_id; acceptance
            # is gated by supports_concurrent_turn_requests, not a per-turn drain id.
            session.complete_model_turn(fence.turn_id)
            fence = DuplexFence(session.session_id, epoch=session.epoch, turn_id=session.turn_id)
            request_id = self._ctx.manager.stage_request_id(fence, stage_id=stage_id, resumable=False)
            session.request_resources.pop((stage_id, stale_ephemeral_id), None)
            if concurrent_turn:
                # Input gate already released after assistant text/silent final,
                # so non-draining stages are idle. Drop their session bindings
                # only — do not abort engine work. Stages the plugin marks as
                # draining keep running under the prior response_id.
                prior_response_id = session.active_response_id
                draining_stages = self._draining_stage_ids()
                for sid, rid in stale_keys:
                    if sid not in draining_stages:
                        session.request_resources.pop((sid, rid), None)
                    elif prior_response_id is not None and not session.is_draining_request(rid):
                        # Already-draining ids keep their original response_id.
                        session.bind_draining_request(rid, prior_response_id)
                self._ctx.run.concurrent_turn_requests_released = False
                if prior_response_id is not None:
                    session.snapshot_active_response_for_drain()
                    new_response_id = session.begin_response(turn_id=fence.turn_id)
                    self._out.emit(self.response_created_payload(new_response_id, epoch=session.epoch))
                else:
                    session.bind_response_turn(fence.turn_id)
            elif stale_ids:
                # Input gate not released yet (e.g. commit while Stage1 is still
                # running): abort the whole prior ephemeral so a later output
                # stage is not left orphaned.
                try:
                    await self._ctx.stage_port.cleanup(stale_ids, abort=True)
                except Exception:
                    logger.warning(
                        "duplex abort of stale ephemeral request failed session=%s ids=%s",
                        session.session_id,
                        stale_ids,
                        exc_info=True,
                    )
        try:
            session.begin_lease_operation(fence, lease_operation_id)
            operation_started = True
            reservation = session.prepare_append(fence)
            already_submitted = False if not resumable else session.stage_request_submitted(stage_id, request_id)
            request_context = self._ctx.manager.ensure_stage_request(session, stage_id=stage_id, fence=fence)
            if request_context is None:
                raise RuntimeError("duplex_data_plane_has_no_stage")
            if request_context.request_id != request_id:
                request_id = request_context.request_id
            prompt_payload: dict[str, object] = (
                {str(key): value for key, value in payload.items()} if isinstance(payload, Mapping) else {}
            )
            append_plan = await self._ctx.plugin.prepare_append_plan(
                request_id=request_id,
                fence=fence,
                session_config=self._ctx.plugin.prepare_prompt_config(
                    {**request_context.session_config, "conversation": list(session.history)},
                    state=self._ctx.model_state,
                    payload=prompt_payload,
                ),
                runtime_config=dict(request_context.runtime_config),
                seq=reservation.update.seq,
                turn_seq=reservation.update.turn_seq,
                payload=payload,
                final=final,
                sampling_params=request_context.stage_sampling_params,
            )
            # Preparation may yield while cancellation or session teardown runs.
            # Never submit the worker's stale result to the model.
            if session.epoch != fence.epoch or session.state != DuplexSessionState.OPEN:
                raise DuplexFenceMismatchError(session.fence, fence)
            if not isinstance(append_plan, DuplexAppendPlan):
                raise TypeError("duplex plugin plan_append() must return DuplexAppendPlan")
            submission = DuplexStageSubmission(
                context=request_context,
                prompt=append_plan.prompt,
                already_submitted=already_submitted,
                resumable=resumable,
            )
            submission_result = await self._ctx.stage_port.submit(submission)
            try:
                if submission_result.request_id != request_id or submission_result.stage_id != stage_id:
                    raise RuntimeError("duplex stage adapter returned a mismatched submission result")
                if expected_epoch is not None and session.epoch != expected_epoch:
                    raise DuplexFenceMismatchError(session.fence, fence)
                update = session.commit_append(reservation)
                session.bind_stage_request(stage_id, request_id, fence=fence)
            except BaseException:
                try:
                    await self._ctx.stage_port.cleanup([request_id])
                except Exception as cleanup_exc:
                    logger.warning(
                        "duplex append compensation remains pending for session %s: %s",
                        session.session_id,
                        cleanup_exc,
                    )
                raise
            session.touch_lease(DuplexLeaseActivity.APPEND)
            session.bind_request(request_id)
            # Consuming a Stage0 bind closes the concurrent-turn gate even when
            # this turn used a fresh ephemeral id (not the reuse branch above).
            if stage_id == 0:
                self._ctx.run.concurrent_turn_requests_released = False
            return {
                "ok": True,
                "operation": "append",
                "session_id": session.session_id,
                "stage_results": [
                    {
                        "stage_id": stage_id,
                        "replica_id": submission_result.replica_id,
                        "result": {
                            "supported": True,
                            "data_plane_append": True,
                            "request_id": request_id,
                            "response_stage_id": request_context.final_stage_id,
                            "seq": update.seq,
                            "turn_id": update.turn_id,
                            "turn_seq": update.turn_seq,
                            "resumable": resumable,
                        },
                    }
                ],
            }
        finally:
            if operation_started and lease_operation_id in session.lease.active_operations:
                try:
                    session.end_lease_operation(lease_operation_id)
                except Exception:
                    session.lease.active_operations.discard(lease_operation_id)

    def _start_data_plane_stream(self, result: object) -> bool:
        """Bind the resumable stage request as the session's active data-plane stream."""
        request_id, _ = duplex_data_plane_request_info(result) if isinstance(result, dict) else (None, None)
        if request_id is None or self._data_plane_outputs_finished(result):
            return False
        session = self._ctx.session
        session.bind_request(request_id)
        if self._ctx.run.stream_request_id == request_id:
            return False
        self._ctx.run.stream_request_id = request_id
        return True

    def cancel_data_plane_stream(self) -> bool:
        had_stream = self._ctx.run.stream_request_id is not None
        self._ctx.run.stream_request_id = None
        return had_stream

    async def signal_cancel_fence(self, cancelled_fence: DuplexFence) -> bool:
        """In-process equivalent of the old ``barge_in`` control signal."""
        session = self._ctx.session
        try:
            next_fence = session.sync_fence()
            if next_fence.epoch > cancelled_fence.epoch:
                stale_request_ids = session.cancel_fence(cancelled_fence, next_fence)
            else:
                stale_request_ids = []
            if stale_request_ids:
                await self._ctx.stage_port.cleanup(list(stale_request_ids), abort=True)
            session.touch_lease(DuplexLeaseActivity.SIGNAL)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Failed to signal duplex runtime session: %s", exc)
            self.send_runtime_error("runtime_signal_failed", exc)
            return False
        return True

    def send_runtime_error(self, code: str, exc: BaseException) -> None:
        if isinstance(exc, DuplexRuntimeConfigError):
            code = exc.code
        self._out.emit_error(code, str(exc), retryable=False)

    @staticmethod
    def _data_plane_outputs_finished(result: object) -> bool:
        if not isinstance(result, dict):
            return False
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        return bool(getattr(outputs[-1], "finished", False))

    def _runtime_data_plane_context(self) -> object:
        session = self._ctx.session
        response_config = session.response_config
        return self._ctx.plugin.data_plane_context(
            epoch=session.epoch,
            turn_id=session.turn_id,
            active_response_turn_id=session.active_response_turn_id,
            active_response_id=session.active_response_id,
            auto_responds=self._out.auto_responds(),
            response_format=response_config.response_format,
            speed=response_config.speed,
            modalities=tuple(response_config.modalities),
        )

    # ------------------------------------------------------------------ #
    # Stage outputs                                                      #
    # ------------------------------------------------------------------ #

    def decide_output(
        self, stage_id: int, output: RequestOutput, context: DuplexOutputContext
    ) -> DuplexOutputDecision | None:
        """Pure plugin decision (no session mutation: this runs inline on the orchestrator loop)."""
        decision = self._ctx.plugin.decide_output(
            stage_id=stage_id,
            final_stage_id=context.final_stage_id,
            segment_finished=context.segment_finished,
            segment_token_ids=context.segment_token_ids,
            segment_output_metadata=dict(context.segment_output_metadata),
            output=output,
        )
        if decision is not None and not isinstance(decision, DuplexOutputDecision):
            raise TypeError("duplex plugin decide_output() must return DuplexOutputDecision or None")
        return decision

    def project_intermediate_output(self, stage_id: int, output: RequestOutput, context: DuplexOutputContext) -> bool:
        """Project an intermediate stage without short-circuiting the pipeline."""
        return self._ctx.plugin.project_intermediate_output(
            stage_id=stage_id,
            output=output,
            context=context,
        )

    def release_concurrent_turn_requests(
        self, stage_id: int, output: RequestOutput, context: DuplexOutputContext
    ) -> bool:
        """Ask the plugin whether the next commit may start while TTS drains."""
        return self._ctx.plugin.release_concurrent_turn_requests(
            stage_id=stage_id,
            segment_finished=context.segment_finished,
            output=output,
            context=context,
        )

    def stage_metrics_snapshot(
        self, stage_id: int, metrics: object, output: object
    ) -> dict[str, dict[str, object]] | None:
        if not isinstance(metrics, StageRequestStats):
            return None
        event = metrics
        if event.stage_id is None:
            event = replace(event, stage_id=stage_id)
        if event.final_output_type is None:
            final_output_type = getattr(output, "final_output_type", None)
            if isinstance(final_output_type, str):
                event = replace(event, final_output_type=final_output_type)
        self._ctx.session.observe_stage_request_stats(stage_id, event)
        try:
            merged = OrchestratorAggregator._merge_stage_metric_event(None, event)
        except Exception:
            return None
        return {str(stage_id): merged}

    def _build_stage_output(self, item: StageOutput) -> OmniRequestOutput:
        output = item.output
        finished = bool(getattr(output, "finished", False)) or item.context.segment_finished
        if item.decision is not None:
            engine_output = OmniRequestOutput.from_stage_output(
                output,
                request_id=item.request_id,
                stage_id=item.stage_id,
                final_output_type=item.decision.final_output_type,
            )
            # Set after construction, not as a keyword: ``from_stage_output``
            # copies ``finished`` from the source last, and a resumable stage
            # request never finishes -- only its segment does. The decision
            # closes that segment, and the projector recognises a listen only
            # on a finished output. The orchestrator queue used to re-assert
            # ``finished`` on every direct-response message; the mailbox hands
            # the raw stage output over, so it has to happen here.
            engine_output.finished = True
            engine_output = attach_duplex_output_decision(engine_output, item.decision)
        elif isinstance(output, OmniRequestOutput):
            engine_output = copy.copy(output)
            if finished:
                engine_output.finished = True
        else:
            final_output_type = getattr(output, "final_output_type", None)
            engine_output = OmniRequestOutput.from_stage_output(
                output,
                request_id=item.request_id,
                finished=finished,
                stage_id=item.stage_id,
                final_output_type=final_output_type if isinstance(final_output_type, str) else "audio",
            )
        engine_output.stage_id = item.stage_id
        if self._ctx.plugin.projects_intermediate_outputs and item.stage_id < item.context.final_stage_id:
            engine_output.finished = False
        snapshot = self.stage_metrics_snapshot(item.stage_id, item.metrics, output)
        if snapshot is not None:
            existing = engine_output.metrics if isinstance(engine_output.metrics, dict) else {}
            existing_stage_metrics = existing.get("stage_metrics")
            merged_stage_metrics = (
                {**existing_stage_metrics, **snapshot} if isinstance(existing_stage_metrics, dict) else snapshot
            )
            engine_output.metrics = {**existing, "stage_metrics": merged_stage_metrics}
        return engine_output

    async def on_stage_output_item(self, item: StageOutput) -> None:
        session = self._ctx.session
        if session.state == DuplexSessionState.CLOSED or self._ctx.run.closing:
            return
        expected_epoch = item.context.identity.fence.epoch
        if session.epoch != expected_epoch:
            return
        if session.lease.terminal_reason is not None:
            # Lease already closed/expired: the output belongs to a dead session.
            return
        session.touch_lease(DuplexLeaseActivity.MODEL_OUTPUT)
        if self._ctx.plugin.data_plane.is_terminal(item.request_id):
            return
        if self._out.auto_responds():
            active_request_id = session.active_request_id
            if active_request_id is not None and active_request_id != item.request_id:
                if not (
                    session.capabilities.supports_concurrent_turn_requests
                    and session.is_draining_request(item.request_id)
                ):
                    return
        engine_output = self._build_stage_output(item)
        runtime_config = self._ctx.plugin.runtime_config_after_model_output(
            dict(session.runtime_config),
            item.context.segment_output_metadata,
        )
        if runtime_config is not None:
            session.replace_runtime_config(runtime_config)
        drain_result = {"data_plane_outputs": [engine_output]}
        close_reason, emitted_response = await self._send_model_output_events(
            drain_result, expected_epoch=expected_epoch
        )
        if close_reason is not None:
            await self._close_from_runtime(close_reason)
            return
        finished = self._data_plane_outputs_finished(drain_result)
        # Intermediate-stage projection finishing means text is done, not the duplex turn.
        # Closing the stream here drops later Code2Wav chunks / next-turn bind.
        if (
            finished
            and emitted_response
            and not self._out.auto_responds()
            and item.stage_id >= item.context.final_stage_id
        ):
            # A finished, emitted response releases the per-request projector
            # cursor on its way out and offers the model another
            # silence unit.
            self._ctx.plugin.data_plane.close_stream(item.request_id)
            if self._ctx.run.stream_request_id == item.request_id:
                self._ctx.run.stream_request_id = None
            self._ctx.services.spawn(
                self.maybe_continue_response(expected_epoch=expected_epoch),
                name="duplex-continue",
            )

    async def _send_model_output_events(
        self,
        result: object,
        *,
        expected_epoch: int | None = None,
    ) -> tuple[str | None, bool]:
        session = self._ctx.session
        if expected_epoch is not None and session.epoch != expected_epoch:
            return None, False
        close_reason: str | None = None
        emitted_response = False
        request_id, _ = duplex_data_plane_request_info(result) if isinstance(result, dict) else (None, None)
        if self._ctx.plugin.data_plane.is_terminal(request_id):
            return None, False
        if request_id is not None and session.active_request_id is None:
            session.bind_request(request_id)
        context = self._runtime_data_plane_context()
        for model_result in self._ctx.plugin.data_plane.project(result, context=context):
            log_audio_emit_event(
                session.session_id,
                session.epoch,
                request_id,
                model_result,
                session.capabilities.chunk_period_ms,
                # Read before _send_one_model_output_event advances the
                # watermark, so the line carries the pre-emit underrun
                # exposure of this stream.
                session.playback.sent_ms,
            )
            close_reason_for_result, did_emit = await self._send_one_model_output_event(
                model_result,
                expected_epoch=expected_epoch,
            )
            emitted_response = emitted_response or did_emit
            close_reason = close_reason or close_reason_for_result
            if expected_epoch is not None and session.epoch != expected_epoch:
                return None, emitted_response
        return close_reason, emitted_response

    def _fail_response_from_model_error(self, model_result: dict[str, object]) -> None:
        """Report a data-plane error and fail the response it interrupted."""
        session = self._ctx.session
        response_id = session.active_response_id
        self._out.emit_error(
            str(model_result.get("error_code")),
            str(model_result.get("error") or "Duplex native data-plane error"),
        )
        if response_id is None:
            return
        session.end_response(commit_text=False)
        self._out.emit(
            {
                "type": "response.done",
                "session_id": session.session_id,
                "response_id": response_id,
                "epoch": session.epoch,
                "committed": False,
                "status": "failed",
                "status_details": {"type": "failed", "reason": model_result.get("error_code")},
                "playback": session.playback.as_dict(),
            }
        )

    def _complete_model_turn_without_output(
        self,
        model_result: dict[str, object],
        *,
        model_turn_id: int | None,
        data_plane_request_id: object,
    ) -> bool:
        """The model ended its turn with nothing to say.

        Returns whether an event was emitted: an auto-response session reports
        the decision as a listen, a turn-based one simply releases the request.
        """
        session = self._ctx.session
        data_plane = self._ctx.plugin.data_plane
        auto_response = self._out.auto_responds()
        if isinstance(data_plane_request_id, str):
            if not auto_response and data_plane_request_id == session.active_request_id:
                session.clear_request()
            if not auto_response:
                data_plane.mark_terminal(data_plane_request_id)
        if model_turn_id is not None:
            session.complete_model_turn(model_turn_id)
        if not auto_response:
            return False
        self._ctx.model_state.clear_continuation()
        payload = {
            "type": "response.listen",
            "session_id": session.session_id,
            "epoch": session.epoch,
            "reason": "model_turn_completed_without_output",
            "model_listen": True,
        }
        self._attach_runtime_metadata(payload, model_result)
        self._out.emit(payload)
        return True

    async def _release_ephemeral_request(self, request_id: object, *, whole_turn: bool = False) -> None:
        """Drop orchestrator state for a finished non-resumable stage request.

        Resident Stage0 stays bound. Intermediate text must not call this;
        only a completed turn (final stage, silent short-circuit, or a
        draining TTS request) does. ``whole_turn`` also drops the other stage
        ids of this turn. A request still marked draining is left registered.
        """
        if not isinstance(request_id, str) or not request_id:
            return
        session = self._ctx.session
        if session.capabilities.supports_core_resumable_request:
            return
        if session.is_draining_request(request_id):
            await self._ctx.stage_port.cleanup([request_id])
            return
        release_ids = [request_id]
        if whole_turn:
            candidate_ids = [rid for _stage_id, rid in session.request_resources]
            for candidate in duplex_same_turn_request_ids(request_id, candidate_ids):
                if candidate not in release_ids and not session.is_draining_request(candidate):
                    release_ids.append(candidate)
        session.release_resources_for_request_ids(release_ids)
        await self._ctx.stage_port.cleanup(release_ids)

    async def _on_model_listen(
        self,
        model_result: dict[str, object],
        *,
        model_turn_id: int | None,
        data_plane_request_id: object,
        expected_epoch: int | None,
    ) -> tuple[str | None, bool]:
        """The model chose to keep listening rather than speak.

        Either it continues the current response with another unit, or the
        response ends here: an auto-response session that has continuations
        left schedules one, and anything else closes the turn out.
        """
        session = self._ctx.session
        model_state = self._ctx.model_state
        data_plane = self._ctx.plugin.data_plane
        auto_response = self._out.auto_responds()
        close_reason: str | None = None
        emitted_response = False
        self._end_active_response_before_future_model_turn(model_turn_id=model_turn_id)
        if (
            session.active_response_id is not None
            and model_turn_id is not None
            and not session.active_response_accepts_model_turn(model_turn_id)
        ):
            return close_reason, emitted_response
        active_response_id = session.active_response_id
        auto_continuations_remaining = active_response_id is None or self.response_continuations_remaining(
            active_response_id
        )
        non_terminal_auto_listen = (
            auto_response
            and active_response_id is not None
            and session.active_request_id is not None
            and model_result.get("end_of_turn") is not True
            and auto_continuations_remaining
        )
        if non_terminal_auto_listen:
            self._ctx.services.spawn(
                self.maybe_continue_response(expected_epoch=expected_epoch), name="duplex-continue"
            )
            return close_reason, emitted_response
        if not auto_response and data_plane_request_id == session.active_request_id:
            session.clear_request()
        model_listen = model_result.get("model_listen")
        if not isinstance(model_listen, bool):
            model_listen = model_result.get("reason") in {None, "", "model_listen"}
        response_id = session.active_response_id
        if isinstance(data_plane_request_id, str) and not auto_response:
            data_plane.mark_terminal(data_plane_request_id)
        emitted_response = True
        payload = {
            "type": "response.listen",
            "session_id": session.session_id,
            "epoch": session.epoch,
            "reason": model_result.get("reason") or "model_listen",
            "model_listen": model_listen,
        }
        if response_id is not None:
            payload["response_id"] = response_id
        self._attach_runtime_metadata(payload, model_result)
        self._out.emit(payload)
        if model_result.get("abort_data_plane_request") is True and isinstance(data_plane_request_id, str):
            # stage_port.abort_requests expects a list of ids; a bare str is
            # iterated as characters and never matches the prewarmed binding.
            await self._abort_request([data_plane_request_id], notify=False)
        if response_id is not None:
            if not auto_response and self.response_continuations_remaining(response_id):
                self._ctx.services.spawn(
                    self.maybe_continue_response(expected_epoch=expected_epoch), name="duplex-continue"
                )
                # Continuation closes the response; it does not release the
                # finished non-resumable request. Do that here so this early
                # return does not leak orchestrator / manager state.
                await self._release_ephemeral_request(data_plane_request_id)
                return close_reason, emitted_response
            if auto_response:
                model_state.clear_continuation()
                if not auto_continuations_remaining:
                    completed_turn_id = model_turn_id
                    if completed_turn_id is None:
                        completed_turn_id = session.active_response_turn_id
                    if completed_turn_id is not None:
                        session.complete_model_turn(completed_turn_id)
            session.end_response(commit_text=False, preserve_request=auto_response)
            self._out.emit(
                {
                    "type": "response.done",
                    "session_id": session.session_id,
                    "response_id": response_id,
                    "epoch": session.epoch,
                    "committed": False,
                    "playback": session.playback.as_dict(),
                }
            )
        await self._release_ephemeral_request(data_plane_request_id, whole_turn=True)
        return close_reason, emitted_response

    async def _send_one_model_output_event(
        self,
        model_result: dict[str, object],
        *,
        expected_epoch: int | None = None,
    ) -> tuple[str | None, bool]:
        session = self._ctx.session
        data_plane = self._ctx.plugin.data_plane
        close_reason: str | None = None
        emitted_response = False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return close_reason, emitted_response
        data_plane_request_id = model_result.get("data_plane_request_id")
        if isinstance(data_plane_request_id, str) and data_plane.is_terminal(data_plane_request_id):
            return close_reason, emitted_response
        auto_response = self._out.auto_responds()
        draining = session.is_draining_request(
            data_plane_request_id if isinstance(data_plane_request_id, str) else None
        )
        active_request_matches = session.active_request_id == data_plane_request_id or (
            auto_response and session.active_request_id is None
        )
        if isinstance(data_plane_request_id, str) and not active_request_matches and not draining:
            return close_reason, emitted_response
        if isinstance(model_result.get("error_code"), str):
            self._fail_response_from_model_error(model_result)
            return close_reason, True
        if model_result.get("function_call") is True:
            self._out.emit(
                {
                    "type": "function_call.done",
                    "session_id": session.session_id,
                    "epoch": session.epoch,
                    "call_id": model_result.get("call_id"),
                    "name": model_result.get("name"),
                    "arguments": model_result.get("arguments", ""),
                }
            )
            return close_reason, True
        if model_result.get("requires_stage_handoff") is True or model_result.get("requires_tts_stage") is True:
            # Reserve the protocol response only when Stage1 emits text/audio.
            return close_reason, False
        is_listen = model_result.get("is_listen")
        model_turn_id = coerce_int(model_result.get("model_turn_id"))
        if model_result.get("is_buffering") is True or model_result.get("prefill_success") is False:
            if model_result.get("data_plane_request_id") == session.active_request_id:
                session.clear_request()
            payload = {
                "type": "response.listen",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "reason": model_result.get("reason") or "buffering",
                "model_listen": False,
                "buffering": True,
            }
            self._attach_runtime_metadata(payload, model_result)
            self._out.emit(payload)
            return close_reason, emitted_response
        context_text = model_result.get("model_context_text")
        if isinstance(context_text, str) and isinstance(data_plane_request_id, str):
            self._ctx.plugin.commit_model_context(
                session_id=duplex_session_id_from_request_id(data_plane_request_id),
                assistant_text=context_text,
            )
        if is_listen is True:
            return await self._on_model_listen(
                model_result,
                model_turn_id=model_turn_id,
                data_plane_request_id=data_plane_request_id,
                expected_epoch=expected_epoch,
            )

        text = model_result.get("text")
        audio = model_result.get("audio_data", model_result.get("audio"))
        end_of_turn = bool(model_result.get("end_of_turn", False))
        has_text = isinstance(text, str) and bool(text)
        has_audio = isinstance(audio, str) and bool(audio)
        if not has_text and not has_audio and not end_of_turn:
            tts_segment_ended = (
                model_result.get("stage_role") == "tts" and model_result.get("abort_data_plane_request") is True
            )
            if (
                tts_segment_ended
                and auto_response
                and (model_turn_id is not None or session.active_response_id is not None)
            ):
                self._ctx.services.spawn(
                    self.maybe_continue_response(expected_epoch=expected_epoch, expected_model_turn_id=model_turn_id),
                    name="duplex-continue",
                )
            return close_reason, emitted_response
        request_key = data_plane_request_id if isinstance(data_plane_request_id, str) else None
        draining_response_id = (
            session.response_id_for_request(request_key) if session.is_draining_request(request_key) else None
        )
        if (
            end_of_turn
            and not has_text
            and not has_audio
            and session.active_response_id is None
            and draining_response_id is None
        ):
            emitted_response = self._complete_model_turn_without_output(
                model_result,
                model_turn_id=model_turn_id,
                data_plane_request_id=data_plane_request_id,
            )
            await self._release_ephemeral_request(data_plane_request_id, whole_turn=True)
            return close_reason, emitted_response
        if (
            draining_response_id is None
            and session.active_response_id is None
            and model_turn_id is not None
            and model_turn_id < session.turn_id
        ):
            # Late audio of a completed model turn must not reserve a second response.
            # Draining requests are exempt: resolve ownership before this filter.
            return close_reason, emitted_response
        if draining_response_id is None:
            self._end_active_response_before_future_model_turn(model_turn_id=model_turn_id)
        if (
            draining_response_id is None
            and session.active_response_id is not None
            and model_turn_id is not None
            and not session.active_response_accepts_model_turn(model_turn_id)
        ):
            return close_reason, emitted_response
        emitted_response = True
        response_created = False
        response_id = draining_response_id or session.active_response_id
        if response_id is None:
            response_id = session.begin_response(turn_id=model_turn_id)
            response_created = True
        response_request_metrics = session.mark_response_first_outputs(
            observed_at_s=session._clock(),
            has_text=has_text,
            has_audio=has_audio,
        )
        if response_created:
            created_payload = self.response_created_payload(response_id, epoch=session.epoch)
            if response_request_metrics:
                created_payload["response_request_metrics"] = response_request_metrics
            self._out.emit(created_payload)
        stage_metrics = model_result.get("stage_metrics")
        response_stage_metrics = session.accumulate_response_stage_metrics(
            stage_metrics if isinstance(stage_metrics, Mapping) else None
        )
        if response_created:
            speak_payload = {
                "type": "response.speak",
                "session_id": session.session_id,
                "response_id": response_id,
                "epoch": session.epoch,
                "text": text if isinstance(text, str) else "",
                "end_of_turn": end_of_turn,
                "model_speak": True,
            }
            self._attach_runtime_metadata(
                speak_payload,
                model_result,
                stage_metrics=response_stage_metrics,
                response_request_metrics=response_request_metrics,
            )
            self._out.emit(speak_payload)
        target_id = draining_response_id if draining_response_id not in (None, session.active_response_id) else None
        previous_sent_ms = session.playback_for_response(target_id).sent_ms
        text_chars_before_append = len(session.assistant_transcript(target_id))
        if isinstance(text, str) and text:
            if target_id is not None:
                session.append_draining_assistant_text(target_id, text)
            else:
                session.append_assistant_text(text)
        duration_ms = model_result.get("audio_duration_ms")
        text_chars = len(session.assistant_transcript(target_id))
        mark_duration_ms = None
        mark_text_chars: int | None = text_chars
        if model_result.get("audio_text_mark") is False:
            mark_text_chars = None
        if isinstance(duration_ms, int | float):
            mark_duration_ms = int(duration_ms)
            if model_result.get("audio_duration_is_cumulative") is not True:
                # Deltas accumulate on the response that owns this chunk.
                # session.playback is the active cursor and is 0 after overlap
                # opens the next response.
                mark_duration_ms += previous_sent_ms
        audio_text_marks = model_result.get("audio_text_marks")
        audio_text_marks = self._normalize_audio_text_marks(
            audio_text_marks if isinstance(audio_text_marks, list) else None,
            audio_offset_ms=(
                0
                if model_result.get("audio_text_marks_are_cumulative") is True
                or model_result.get("audio_duration_is_cumulative") is True
                else previous_sent_ms
            ),
            text_offset_chars=(
                0 if model_result.get("audio_text_marks_are_cumulative") is True else text_chars_before_append
            ),
        )
        session.mark_audio_sent(
            mark_duration_ms,
            text_chars=mark_text_chars if mark_duration_ms is not None else None,
            audio_text_marks=audio_text_marks,
            text_requires_complete_audio=model_result.get("text_requires_complete_audio") is True,
            audio_complete=model_result.get("audio_complete") is True,
            response_id=target_id,
        )
        payload = {
            "type": "response.output_audio.delta",
            "session_id": session.session_id,
            "response_id": response_id,
            "epoch": session.epoch,
            "text": text if isinstance(text, str) else "",
            "audio": audio if isinstance(audio, str) else "",
            "format": (
                model_result.get("audio_format")
                if isinstance(model_result.get("audio_format"), str)
                else session.response_config.response_format
            ),
            "end_of_turn": end_of_turn,
            "model_speak": True,
        }
        if mark_duration_ms is not None:
            payload["audio_duration_ms"] = mark_duration_ms
        if audio_text_marks:
            payload["audio_text_marks"] = audio_text_marks
        elif mark_duration_ms is not None and mark_text_chars is not None:
            payload["audio_text_marks"] = [
                {"text_chars": max(0, int(mark_text_chars)), "audio_end_ms": max(0, int(mark_duration_ms))}
            ]
        payload["playback"] = session.playback_for_response(target_id).as_dict()
        sample_rate_hz = model_result.get("sample_rate_hz") or model_result.get("audio_sample_rate_hz")
        if isinstance(sample_rate_hz, int | float) and int(sample_rate_hz) > 0:
            payload["sample_rate_hz"] = int(sample_rate_hz)
        self._attach_runtime_metadata(
            payload,
            model_result,
            stage_metrics=response_stage_metrics,
            response_request_metrics=response_request_metrics,
        )
        self._out.emit(payload)
        if (
            not end_of_turn
            and model_result.get("stage_role") == "tts"
            and model_result.get("abort_data_plane_request") is True
            and auto_response
        ):
            self._ctx.services.spawn(
                self.maybe_continue_response(expected_epoch=expected_epoch), name="duplex-continue"
            )
        if end_of_turn:
            data_plane_request_id = model_result.get("data_plane_request_id")
            # Prior TTS finished under its own draining response_id while a
            # newer turn already owns active_response_id — close that response.
            if (
                session.capabilities.supports_concurrent_turn_requests
                and isinstance(data_plane_request_id, str)
                and session.is_draining_request(data_plane_request_id)
            ):
                drained_response_id = session.pop_draining_request(data_plane_request_id)
                data_plane.close_stream(data_plane_request_id)
                data_plane.mark_terminal(data_plane_request_id)
                for stage_id in self._draining_stage_ids():
                    session.request_resources.pop((stage_id, data_plane_request_id), None)
                await self._release_ephemeral_request(data_plane_request_id, whole_turn=True)
                if drained_response_id is not None:
                    playback = session.playback_for_response(drained_response_id).as_dict()
                    session.release_finished_drain_response(drained_response_id)
                    self._out.emit(
                        {
                            "type": "response.done",
                            "session_id": session.session_id,
                            "response_id": drained_response_id,
                            "epoch": session.epoch,
                            "committed": False,
                            "status": "completed",
                            "playback": playback,
                        }
                    )
                return close_reason, emitted_response
            if isinstance(data_plane_request_id, str) and not auto_response:
                data_plane.close_stream(data_plane_request_id)
            if isinstance(data_plane_request_id, str):
                if not auto_response and data_plane_request_id == session.active_request_id:
                    session.clear_request()
                if not auto_response:
                    data_plane.mark_terminal(data_plane_request_id)
            should_commit = self.should_commit_response_to_history(session, response_id)
            committed_message = session.end_response(commit_text=should_commit, preserve_request=auto_response)
            model_turn_id = coerce_int(model_result.get("model_turn_id"))
            if model_turn_id is not None:
                session.complete_model_turn(model_turn_id)
            if should_commit:
                session.register_history_item(f"item_{response_id}", committed_message)
            self._out.emit(
                {
                    "type": "response.done",
                    "session_id": session.session_id,
                    "response_id": response_id,
                    "epoch": session.epoch,
                    "committed": committed_message is not None,
                    "playback": session.playback.as_dict(),
                }
            )
            await self._release_ephemeral_request(data_plane_request_id, whole_turn=True)
        return close_reason, emitted_response

    def _end_active_response_before_future_model_turn(self, *, model_turn_id: int | None) -> None:
        session = self._ctx.session
        if not self._out.auto_responds():
            return
        response_id = session.active_response_id
        active_turn_id = session.active_response_turn_id
        if response_id is None or model_turn_id is None or active_turn_id is None:
            return
        if int(model_turn_id) <= int(active_turn_id):
            return
        session.complete_model_turn(int(model_turn_id) - 1)
        should_commit = self.should_commit_response_to_history(session, response_id)
        committed_message = session.end_response(commit_text=should_commit, preserve_request=True)
        if should_commit:
            session.register_history_item(f"item_{response_id}", committed_message)
        self._out.emit(
            {
                "type": "response.done",
                "session_id": session.session_id,
                "response_id": response_id,
                "epoch": session.epoch,
                "committed": committed_message is not None,
                "playback": session.playback.as_dict(),
            }
        )

    @staticmethod
    def _normalize_audio_text_marks(
        audio_text_marks: list[object] | None,
        *,
        audio_offset_ms: int,
        text_offset_chars: int,
    ) -> list[dict[str, object]] | None:
        if not audio_text_marks:
            return None
        normalized: list[dict[str, object]] = []
        for raw_mark in audio_text_marks:
            if not isinstance(raw_mark, dict):
                continue
            raw_text_chars = raw_mark.get("text_chars")
            raw_audio_end_ms = raw_mark.get("audio_end_ms", raw_mark.get("audio_ms"))
            if not isinstance(raw_text_chars, int | float) or not isinstance(raw_audio_end_ms, int | float):
                continue
            normalized.append(
                {
                    "text_chars": max(0, int(raw_text_chars) + int(text_offset_chars)),
                    "audio_end_ms": max(0, int(raw_audio_end_ms) + int(audio_offset_ms)),
                }
            )
        return normalized or None

    @staticmethod
    def _attach_runtime_metadata(
        payload: dict[str, object],
        model_result: dict[str, object],
        *,
        stage_metrics: Mapping[str, object] | None = None,
        response_request_metrics: Mapping[str, object] | None = None,
    ) -> None:
        metadata: dict[str, object] = {}
        runtime_impl = model_result.get("runtime_impl")
        if isinstance(runtime_impl, str) and runtime_impl:
            metadata["runtime_impl"] = runtime_impl
        owned_runtime = model_result.get("owned_runtime")
        if isinstance(owned_runtime, bool):
            metadata["owned_runtime"] = owned_runtime
        model_turn_id = coerce_int(model_result.get("model_turn_id"))
        if model_turn_id is not None:
            metadata["model_turn_id"] = model_turn_id
        for name in ("uses_model_runner_scheduler", "runner_kv_backed"):
            value = model_result.get(name)
            if isinstance(value, bool):
                metadata[name] = value
        effective_stage_metrics = stage_metrics if stage_metrics is not None else model_result.get("stage_metrics")
        if isinstance(effective_stage_metrics, Mapping):
            metadata["stage_metrics"] = {
                str(stage_id): dict(values)
                for stage_id, values in effective_stage_metrics.items()
                if isinstance(values, Mapping)
            }
        if response_request_metrics:
            metadata["response_request_metrics"] = dict(response_request_metrics)
        if metadata:
            payload["vllm_omni"] = metadata

    # ------------------------------------------------------------------ #
    # Silence continuation                                               #
    # ------------------------------------------------------------------ #

    def silence_unit_payload(self) -> dict[str, object]:
        samples = int(self._ctx.plugin.silence_continuation_samples)
        audio = (
            self._SILENCE_UNIT_PAYLOAD_AUDIO
            if samples == 16000
            else base64.b64encode(bytes(samples * 4)).decode("ascii")
        )
        return {"type": "audio", "audio": audio, "format": "pcm_f32le", "sample_rate_hz": 16000}

    def response_continuations_remaining(self, response_id: str) -> bool:
        model_state = self._ctx.model_state
        owner_id = f"response:{response_id}"
        count = model_state.continuation_units if model_state.continuation_owner_id == owner_id else 0
        limit = (
            self._AUTO_RESPONSE_MAX_CONTINUATION_UNITS
            if self._out.auto_responds()
            else self._RESPONSE_MAX_CONTINUATION_UNITS
        )
        return count < limit

    async def _finish_bounded_auto_response(
        self,
        *,
        expected_epoch: int | None,
        model_turn_id: int | None = None,
    ) -> None:
        session = self._ctx.session
        model_state = self._ctx.model_state
        response_id = session.active_response_id
        response_epoch = session.epoch
        response_turn_id = model_turn_id if model_turn_id is not None else session.active_response_turn_id
        if expected_epoch is not None and response_epoch != expected_epoch:
            return
        model_state.clear_continuation()
        payload: dict[str, object] = {
            "type": "response.listen",
            "session_id": session.session_id,
            "epoch": response_epoch,
            "reason": "continuation_limit",
            "model_listen": False,
        }
        if response_id is not None:
            payload["response_id"] = response_id
        self._out.emit(payload)
        if response_id is None:
            if session.epoch == response_epoch and response_turn_id is not None:
                session.complete_model_turn(response_turn_id)
            return
        if session.epoch != response_epoch or session.active_response_id != response_id:
            return
        if response_turn_id is not None:
            session.complete_model_turn(response_turn_id)
        session.end_response(commit_text=False, preserve_request=True)
        self._out.emit(
            {
                "type": "response.done",
                "session_id": session.session_id,
                "response_id": response_id,
                "epoch": response_epoch,
                "committed": False,
                "playback": session.playback.as_dict(),
            }
        )

    def silence_continuation_is_stale(
        self,
        *,
        request_id: str,
        response_id: str | None,
        response_owned: bool,
        expected_epoch: int | None,
        expected_model_turn_id: int | None,
    ) -> bool:
        session = self._ctx.session
        stale_common_owner = (
            session.state == DuplexSessionState.CLOSED
            or session.active_request_id != request_id
            or (expected_epoch is not None and session.epoch != expected_epoch)
        )
        stale_response_owner = response_owned and session.active_response_id != response_id
        stale_model_turn_owner = not response_owned and (
            session.active_response_id is not None or session.turn_id != expected_model_turn_id
        )
        return stale_common_owner or stale_response_owner or stale_model_turn_owner

    async def maybe_continue_response(
        self,
        *,
        expected_epoch: int | None,
        expected_model_turn_id: int | None = None,
    ) -> None:
        session = self._ctx.session
        model_state = self._ctx.model_state
        response_id = session.active_response_id
        auto_response = self._out.auto_responds()
        if session.state == DuplexSessionState.CLOSED or self._ctx.run.closing:
            model_state.clear_continuation()
            return
        if expected_epoch is not None and session.epoch != expected_epoch:
            return
        # Non-resumable stage0 cannot submit_update after the request finishes;
        # clear continuation and close the response (silent / listen final).
        # A continue for a response that still has draining TTS must not emit
        # response.done here; that finish owns the done event.
        if not session.capabilities.supports_core_resumable_request:
            model_state.clear_continuation()
            if response_id is not None and not session.response_has_draining_request(response_id):
                should_commit = self.should_commit_response_to_history(session, response_id)
                committed_message = session.end_response(commit_text=should_commit, preserve_request=auto_response)
                if should_commit and committed_message is not None:
                    session.register_history_item(f"item_{response_id}", committed_message)
                self._out.emit(
                    {
                        "type": "response.done",
                        "session_id": session.session_id,
                        "response_id": response_id,
                        "epoch": session.epoch,
                        "committed": committed_message is not None if should_commit else False,
                        "playback": session.playback.as_dict(),
                    }
                )
            return
        request_id = session.active_request_id
        if request_id is None:
            model_state.clear_continuation()
            return
        if expected_epoch is not None and session.epoch != expected_epoch:
            return
        response_owned = response_id is not None
        if response_owned:
            owner_id = f"response:{response_id}"
            payload_turn_id_value = (
                session.active_response_turn_id if session.active_response_turn_id is not None else session.turn_id
            )
        else:
            if not auto_response or expected_model_turn_id is None or session.turn_id != expected_model_turn_id:
                model_state.clear_continuation()
                return
            owner_id = f"model-turn:{expected_model_turn_id}"
            payload_turn_id_value = expected_model_turn_id
        count = model_state.continuation_units if model_state.continuation_owner_id == owner_id else 0
        continuation_limit = (
            self._AUTO_RESPONSE_MAX_CONTINUATION_UNITS if auto_response else self._RESPONSE_MAX_CONTINUATION_UNITS
        )
        if count >= continuation_limit:
            if auto_response:
                await self._finish_bounded_auto_response(
                    expected_epoch=expected_epoch,
                    model_turn_id=payload_turn_id_value,
                )
            return
        payload = self.silence_unit_payload()
        if auto_response and count + 1 == continuation_limit:
            payload["force_listen"] = True
        payload["duplex_turn_id"] = payload_turn_id_value
        try:
            scheduled = await self._schedule_silence_continuation(
                payload,
                request_id=request_id,
                owner_id=owner_id,
                response_id=response_id,
                response_owned=response_owned,
                expected_epoch=expected_epoch,
                expected_model_turn_id=expected_model_turn_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Failed to schedule duplex native response continuation: %s", exc)
            scheduled = False
        if scheduled:
            model_state.continuation_owner_id = owner_id
            model_state.continuation_units = count + 1

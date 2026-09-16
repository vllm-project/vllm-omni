# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""One append in flight, and what its failure owes the session.

An append is not a call: it is a task queued behind the previous append, which
may be cancelled, may find the session closed before it starts, and on any of
those paths has to give back what it reserved --- the PCM bytes it took from the
input budget, the committed audio it was going to consume, and the response it
precreated.

That compensation is the reason this is an object. The five steps of an append
all need the same eleven values, so as closures over ``_start_append`` they read
as one function with five entry points and no way to test any of them. Named
fields make the captured state explicit and the compensation a method.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from vllm.logger import init_logger

from vllm_omni.engine.duplex.config import DuplexSessionState, DuplexTurnEventType
from vllm_omni.engine.duplex.plugin import PcmAppendReservation
from vllm_omni.engine.duplex.session import helpers
from vllm_omni.engine.duplex.session.context import DuplexSessionContext
from vllm_omni.engine.duplex.session.emitter import SessionEmitter
from vllm_omni.engine.duplex.session.model_channel import ModelChannel

logger = init_logger(__name__)


@dataclass(slots=True)
class AppendAttempt:
    """One queued append to the model, with the rollback each failure path owes."""

    ctx: DuplexSessionContext
    out: SessionEmitter
    model: ModelChannel
    #: Marks the session closing before the close coroutine gets to run.
    fail_session: Callable[[str], None]

    payload: dict[str, object]
    epoch: int
    request_id: str
    final: bool
    pcm_reservation: PcmAppendReservation | None
    operation_id: str | None
    #: Committed audio this append consumes; released once it is safely submitted.
    retained_committed_payload: dict[str, object] | None
    #: Response reserved before submission, to be failed if submission does not happen.
    precreated_response_id: str | None
    #: Last chance to call the append off, checked once the predecessor is done.
    before_append: Callable[[], bool] | None

    # ------------------------------------------------------------------ #
    # Compensation                                                       #
    # ------------------------------------------------------------------ #

    def discard_retained_audio(self) -> None:
        """Release the committed audio this append was going to consume.

        Only if it is still the session's: a later commit may have replaced it,
        and that one belongs to the append that will carry it.
        """
        model_state = self.ctx.model_state
        if self.retained_committed_payload is not None and (
            model_state.committed_audio_payload is self.retained_committed_payload
        ):
            self.ctx.session.release_input_bytes(model_state.clear_committed_audio())

    def abandon(self) -> None:
        """Give back everything this append reserved but never used."""
        if self.pcm_reservation is not None:
            self.pcm_reservation.rollback()
        self.discard_retained_audio()

    def release_on_failure(self, done: asyncio.Task[bool]) -> None:
        """Done-callback: a cancelled or failed append must not hold the audio."""
        if done.cancelled():
            self.discard_retained_audio()
            return
        try:
            append_ok = done.result()
        except Exception:
            append_ok = False
        if not append_ok:
            self.discard_retained_audio()

    def clear_pending_silence(self, done: asyncio.Task[bool]) -> None:
        """Done-callback of a silence continuation: it is no longer pending."""
        model_state = self.ctx.model_state
        if model_state.pending_silence_task is done:
            model_state.pending_silence_task = None
            model_state.pending_silence_owner_id = None

    # ------------------------------------------------------------------ #
    # Running                                                            #
    # ------------------------------------------------------------------ #

    async def run_in_wire_order(self, predecessor: asyncio.Task[bool] | None) -> bool:
        """Wait for the previous append, then submit this one if it still applies."""
        if predecessor is not None:
            try:
                predecessor_ok = await predecessor
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                predecessor_ok = False
            except Exception:
                predecessor_ok = False
            if not predecessor_ok:
                self.abandon()
                return False
        run = self.ctx.run
        if run.closing or run.runtime_closed or self.ctx.session.state != DuplexSessionState.OPEN:
            self.abandon()
            return False
        if self.before_append is not None and not self.before_append():
            self.abandon()
            # Called off, not failed: the chain behind it still runs.
            return True
        if self.pcm_reservation is not None and not self.pcm_reservation.active:
            self.discard_retained_audio()
            return False
        return await self._submit()

    async def _submit(self) -> bool:
        session = self.ctx.session
        model_state = self.ctx.model_state
        try:
            append_ok, emitted_response = await self.model.append_runtime_input(
                self.payload,
                operation_id=(
                    self.pcm_reservation.operation_id if self.pcm_reservation is not None else self.operation_id
                ),
                final=self.final,
                expected_epoch=self.epoch,
            )
            if append_ok:
                model_state.context_locked = True
                if self.pcm_reservation is not None:
                    self.pcm_reservation.commit()
                    session.release_input_bytes(self.pcm_reservation.byte_count)
                self.discard_retained_audio()
            else:
                self.abandon()
            if not append_ok and self._precreated_response_still_active():
                self._fail_precreated_response()
            if not append_ok and session.state == DuplexSessionState.CLOSED:
                self.ctx.run.runtime_closed = True
                return False
            if not emitted_response and session.epoch == self.epoch:
                if session.active_request_id == helpers.stage0_request_id(session, self.epoch):
                    session.clear_request(self.request_id)
                if self.final:
                    self.out.emit_events([session.signal_turn(DuplexTurnEventType.USER_STARTED.value)])
            return append_ok
        except asyncio.CancelledError:
            if self.pcm_reservation is not None:
                self.pcm_reservation.rollback()
            raise
        except Exception as exc:
            self.abandon()
            logger.exception("Native duplex append task failed: %s", exc)
            self.model.send_runtime_error("runtime_append_task_failed", exc)
            if session.state != DuplexSessionState.CLOSED:
                self.fail_session("runtime_append_task_failed")
            return False

    def _precreated_response_still_active(self) -> bool:
        return (
            self.precreated_response_id is not None
            and self.ctx.session.active_response_id == self.precreated_response_id
        )

    def _fail_precreated_response(self) -> None:
        session = self.ctx.session
        session.end_response(commit_text=False)
        self.out.emit(
            {
                "type": "response.done",
                "session_id": session.session_id,
                "response_id": self.precreated_response_id,
                "epoch": session.epoch,
                "committed": False,
                "status": "failed",
                "status_details": {"type": "failed", "reason": "runtime_append_failed"},
                "playback": session.playback.as_dict(),
            }
        )

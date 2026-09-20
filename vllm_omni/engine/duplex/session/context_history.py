# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded model history and replacement through ordinary resumable requests.

The owning engine session serializes input. Stage0 completion, delivered on
its orchestrator loop, acknowledges application; enqueueing a StreamingUpdate
alone is never an applied receipt. No scheduler utility or second session is
introduced. Models own prompt selection and reconstruction through a policy.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from copy import deepcopy
from typing import TYPE_CHECKING

from vllm_omni.engine.duplex.contracts import DuplexFence, DuplexOutputContext, DuplexStageSubmission
from vllm_omni.engine.duplex.plugin import DuplexContextPolicy, DuplexRuntimeConfigError
from vllm_omni.engine.duplex.session.helpers import assistant_playback_active

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.session.context import DuplexSessionContext
    from vllm_omni.engine.duplex.session.emitter import SessionEmitter
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
    from vllm_omni.engine.duplex.session.model_channel import ModelChannel


class DuplexContextHistory:
    """One session's journal, completion fence and validated replacement."""

    def __init__(
        self,
        ctx: DuplexSessionContext,
        policy: DuplexContextPolicy,
        *,
        out: SessionEmitter,
        model: ModelChannel,
        max_tokens: int,
        wait_for_append_tail: Callable[[], Awaitable[bool]],
        close_from_runtime: Callable[[str], Awaitable[None]],
    ) -> None:
        self._ctx = ctx
        self._out = out
        self._model = model
        self._wait_for_append_tail = wait_for_append_tail
        self._close_from_runtime = close_from_runtime
        self.policy = policy
        self.prompts: list[dict] = []
        self.pending: asyncio.Future[None] | None = None
        self.pending_request: str | None = None
        self.replaying = False
        self.changing = False
        self.resource_generation = 0
        self.max_tokens = max_tokens
        self.max_bytes = policy.max_bytes
        self._epoch = self.session.epoch
        self._input_epoch_floor = self._epoch

    @property
    def session(self) -> DuplexEngineSession:
        return self._ctx.session

    def _size(self, prompts: list[dict]) -> int:
        # Prompt metadata contains a frozen fence and base64 media, not tensors.
        # JSON measures encoded media plus tokens/config, including replay data.
        return len(json.dumps(prompts, default=lambda o: vars(o) if hasattr(o, "__dict__") else str(o)).encode())

    def check_budget(self, prompts: list[dict]) -> None:
        tokens = sum(self.policy.token_count(p) for p in prompts)
        if tokens >= self.max_tokens or self._size(prompts) > self.max_bytes:
            raise ValueError("context replacement exceeds token/byte budget")

    def record(self, request_id: str, prompt: dict) -> None:
        """Reserve journal space before submitting a model input."""
        # The submission and model hooks mutate metadata; the journal owns a stable snapshot.
        candidate = deepcopy(prompt)
        self.check_budget([*self.prompts, candidate])
        self.prompts.append(candidate)
        self.pending_request = request_id
        self.pending = asyncio.get_running_loop().create_future()

    def observe(self, stage_id: int, request_id: str, output: object, context: DuplexOutputContext) -> bool:
        """Complete one input at its actual model boundary; suppress replay output."""
        if context.identity.fence.epoch != self.session.epoch:
            return True
        if stage_id == 0 and context.segment_finished and request_id == self.pending_request:
            if self.pending is not None and not self.pending.done() and self.prompts:
                try:
                    updated = self.policy.complete(self.prompts[-1], output, context)
                    if updated is None:
                        raise RuntimeError("context completion has no matching model unit")
                    candidate = [*self.prompts[:-1], updated]
                    self.check_budget(candidate)
                    if not self.replaying:
                        self.prompts[-1] = updated
                    self.pending.set_result(None)
                except Exception as exc:
                    self.pending.set_exception(exc)
                    # No waiter may exist when the completion arrives.
                    self.pending.exception()
                    # A missing output journal must never silently become the
                    # source of the next history reconstruction.
                    self._ctx.services.spawn(self._close_from_runtime("context_output_failed"), name="context-close")
        return self.replaying

    def fail_pending(self, exc: BaseException) -> None:
        """Wake waiters when a journalised model unit can no longer complete.

        Session-owned requests never surface as processed terminal outputs, so
        a scheduler-side failure must be handed to the journal explicitly;
        otherwise the next ``wait_applied`` would block for its full timeout.
        """
        if self.pending is not None and not self.pending.done():
            self.pending.set_exception(exc)
            self.pending.exception()

    def synchronize_epoch(self) -> None:
        """Invalidate abandoned history and wake waiters at a cancellation boundary."""
        if self._epoch == self.session.epoch:
            return
        self._epoch = self.session.epoch
        self._input_epoch_floor = self._epoch
        self.prompts.clear()
        if self.pending is not None and not self.pending.done():
            self.pending.set_exception(
                DuplexRuntimeConfigError("Context epoch was cancelled", code="context_not_initialized")
            )
            # There need not be a waiter when cancellation arrives.
            self.pending.exception()
        self.pending = None
        self.pending_request = None

    def resolve_append_epoch(self, epoch: int) -> int:
        """Carry accepted inputs across automatic rollovers, never cancellations."""
        self.synchronize_epoch()
        if self._input_epoch_floor <= epoch <= self._epoch:
            return self._epoch
        return epoch

    async def wait_applied(self) -> None:
        self.synchronize_epoch()
        if self.pending is not None:
            await asyncio.wait_for(asyncio.shield(self.pending), timeout=60)

    async def before_append(self) -> None:
        """Keep input/output identity aligned without a scheduler append RPC."""
        await self.wait_applied()
        if (
            self.prompts
            and not self.changing
            and self.session.active_response_id is None
            and not assistant_playback_active(self.session)
            and self.policy.should_rollover(self.prompts, self.session.runtime_config)
        ):
            await self.replace(None, automatic=True)

    def snapshot(self) -> dict:
        return {
            "epoch": self.session.epoch,
            "context_version": self.session.runtime_config.get("duplex_context_version", 0),
            "resource_generation": self.resource_generation,
            "units": [self.policy.describe(p) for p in self.prompts],
        }

    async def handle(self, event: str, item: dict) -> bool:
        """Validate a context command, retaining old KV on validation errors."""
        self.changing = True
        try:
            if not await self._wait_for_append_tail():
                raise RuntimeError("preceding append failed")
            await self.wait_applied()
            if not self.prompts:
                raise DuplexRuntimeConfigError("Initialize the audio session first", code="context_not_initialized")
            if event == "input.context.get":
                self._out.emit({"type": "input.context.snapshot", **self.snapshot()})
            elif event == "input.context.replace" or self.policy.requires_replacement(item):
                await self.replace(item)
            else:
                await self.append(item)
            return True
        except (ValueError, DuplexRuntimeConfigError) as exc:
            self._out.emit_error(getattr(exc, "code", "invalid_context_input"), str(exc), event_id=item.get("event_id"))
        except Exception as exc:
            self._out.emit_error("context_operation_failed", str(exc), event_id=item.get("event_id"))
            await self._close_from_runtime("context_operation_failed")
        finally:
            self.changing = False
        return False

    async def append(self, item: dict) -> None:
        candidate, payload = self.policy.prepare_input(
            item, dict(self.session.runtime_config), epoch=self.session.epoch
        )
        version = candidate.get("duplex_context_version", 0)
        if payload is not None:
            # Validate the exact append before committing its receipt/config.
            session = self.session
            context = self._ctx.manager.ensure_stage_request(session, stage_id=0, fence=session.fence)
            if context is None:
                raise RuntimeError("duplex_data_plane_has_no_stage")
            reservation = session.prepare_append(session.fence)
            plan = self._ctx.plugin.plan_append(
                request_id=context.request_id,
                fence=session.fence,
                session_config=session.config.as_dict(),
                runtime_config=candidate,
                seq=reservation.update.seq,
                turn_seq=reservation.update.turn_seq,
                payload=payload,
                final=False,
                sampling_params=context.stage_sampling_params,
            )
            self.check_budget([*self.prompts, plan.prompt])
            self.session.replace_runtime_config(candidate)
            ok, _ = await self._model.append_runtime_input(payload, final=False, expected_epoch=self.session.epoch)
            if not ok:
                raise RuntimeError("context append failed")
        self._out.emit(
            {
                "type": "input.context.appended",
                "epoch": self.session.epoch,
                "event_id": item["event_id"],
                "context_version": version,
                "duplicate": payload is None,
            }
        )
        await self.wait_applied()
        self._out.emit(
            {
                "type": "input.context.applied",
                "epoch": self.session.epoch,
                "event_id": item["event_id"],
                "context_version": version,
            }
        )

    def _retire_replay_requests(self, request_ids: list[str]) -> None:
        """Best-effort cleanup of replay stage requests orphaned by a supersede."""
        if not request_ids:
            return
        with suppress(Exception):
            self._ctx.services.spawn(
                self._ctx.stage_port.cleanup(request_ids, abort=True), name="context-replay-cleanup"
            )

    async def replace(self, item: dict | None, *, automatic: bool = False) -> None:
        session = self.session
        request: dict | None
        if automatic:
            # Model policy owns the snapshot and version increment.
            candidate, request = self.policy.rollover(dict(session.runtime_config), epoch=session.epoch)
            version = int(candidate["duplex_context_version"])
            event_id = f"rollover:{session.epoch}:{version}"
        else:
            if item is None:
                raise ValueError("context replacement requires an item")
            candidate, request = self.policy.prepare_replacement(
                item, dict(session.runtime_config), epoch=session.epoch
            )
            event_id = item["event_id"]
        if request is None:
            self._out.emit(
                {"type": "input.context.replaced", "event_id": event_id, "duplicate": True, **self.snapshot()}
            )
            return
        new_fence = DuplexFence(session.session_id, epoch=session.epoch + 1, turn_id=session.turn_id)
        request_id = self._ctx.manager.stage_request_id(new_fence, stage_id=0)
        if not automatic:
            request["discard_turn_id"] = session.active_response_turn_id
        plan = self.policy.plan(
            prompts=self.prompts,
            runtime_config=candidate,
            session_config=session.config.as_dict(),
            request_id=request_id,
            fence=new_fence,
            context=request,
        )
        self.check_budget([dict(unit.prompt) for unit in plan.units])
        # All validation precedes destructive cleanup. Only this session owns
        # the transition, and old epoch output is rejected from this point.
        old_epoch = session.epoch
        old_response_id = session.active_response_id
        if old_response_id is None and assistant_playback_active(session):
            old_response_id = session.last_response_id
        old_requests = session.resource_request_ids()
        session.end_response(commit_text=False)
        session.epoch = new_fence.epoch
        self._epoch = session.epoch
        if not automatic:
            self._input_epoch_floor = self._epoch
        session.sync_fence()
        session.clear_playback_cursor()
        self._ctx.model_state.clear_continuation()
        self._model.cancel_data_plane_stream()
        session.replace_runtime_config(candidate)
        self.replaying = True
        self.resource_generation += 1
        self._out.emit(
            {
                "type": "audio.cancelled",
                "response_id": old_response_id,
                "cancelled_epoch": old_epoch,
                "epoch": session.epoch,
                "reason": "context_replaced",
            }
        )
        replayed_request_ids: list[str] = []
        try:
            await self._ctx.stage_port.cleanup(old_requests, abort=True)
            for old_request in old_requests:
                self._ctx.plugin.data_plane.close_stream(old_request)
            session.release_all_requests()
            self.prompts = []
            self.pending = None
            for index, unit in enumerate(plan.units):
                if session.epoch != new_fence.epoch:
                    # A cancellation superseded this replacement mid-replay:
                    # its epoch transition already owns the journal cleanup.
                    self._retire_replay_requests(replayed_request_ids)
                    return
                context = self._ctx.manager.ensure_stage_request(session, stage_id=0, fence=session.fence)
                if context is None:
                    raise RuntimeError("duplex_data_plane_has_no_stage")
                reservation = session.prepare_append(session.fence)
                self.record(context.request_id, dict(unit.prompt))
                replayed_request_ids.append(context.request_id)
                await self._ctx.stage_port.submit(
                    DuplexStageSubmission(context=context, prompt=unit.prompt, already_submitted=index > 0)
                )
                session.commit_append(reservation)
                session.bind_stage_request(0, context.request_id, fence=session.fence)
                await self.wait_applied()
            session.bind_request(request_id)
            self._ctx.run.stream_request_id = request_id
            self._ctx.plugin.data_plane.begin_request(request_id)
            self._out.emit(
                {
                    "type": "input.context.replaced",
                    "event_id": event_id,
                    "retained_unit_ids": list(plan.retained_unit_ids),
                    "dropped_unit_ids": list(plan.dropped_unit_ids),
                    "duplicate": False,
                    **self.snapshot(),
                }
            )
        except asyncio.CancelledError:
            # Cancellation is the normal way an append-owned automatic
            # rollover is superseded by input.cancel/barge_in. The runner's
            # epoch transition performs the shared journal cleanup; closing
            # the session here would turn a user cancellation into a runtime
            # failure. Retire only the replay requests this replacement already
            # created, so they cannot linger until session close.
            self._retire_replay_requests(replayed_request_ids)
            raise
        except BaseException:
            await self._close_from_runtime("context_replacement_failed")
            raise
        finally:
            self.replaying = False
        if request.get("generate"):
            ok, _ = await self._model.append_runtime_input(
                self.policy.wake_payload(candidate), final=False, expected_epoch=session.epoch
            )
            if not ok:
                raise RuntimeError("context wakeup failed")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session-owned execution of model-planned memory work.

The model plugin owns memory content, prompts and source validity. This helper
owns one private request, its retry budget and interaction priority. Tasks remain
tracked by the runner, and results return through its mailbox before writeback.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Callable, Hashable, Mapping

from vllm.logger import init_logger

from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.contracts import (
    DuplexStageRequestContext,
    DuplexStageSubmission,
    duplex_is_auxiliary_request,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.plugin import DuplexAuxiliaryRequest, duplex_text_history
from vllm_omni.engine.duplex.session import helpers
from vllm_omni.engine.duplex.session.context import DuplexSessionContext

logger = init_logger(__name__)

_CLEANUP_TIMEOUT_S = 5.0


class SessionMemoryRuntime:
    """Execute memory work through the existing plugin and Session contracts."""

    def __init__(
        self,
        ctx: DuplexSessionContext,
        *,
        post: Callable[[str, dict[str, object]], None],
        wake: Callable[[object], None],
    ) -> None:
        self._ctx = ctx
        self._post = post
        self._wake = wake
        self._task: asyncio.Task[None] | None = None
        self._output: asyncio.Future[object] | None = None
        self._request_id: str | None = None
        self._source: tuple[int, int, Hashable] | None = None
        self._completion_pending = False
        self._epoch = ctx.session.epoch
        self._generation = 0
        self._history_revision = -1
        self._history: tuple[dict[str, object], ...] = ()
        self._retry_source: tuple[int, int, Hashable] | None = None
        self._failures = 0
        self._retry_timer: asyncio.TimerHandle | None = None
        self._input_pending = False
        self._pending_user_items: set[str] = set()
        self._interaction_active = False
        self._interaction_epoch = ctx.session.epoch
        self._interaction_turn: int | None = None

    @property
    def _closing(self) -> bool:
        return self._ctx.run.closing or self._ctx.session.state != DuplexSessionState.OPEN

    def on_output(self, request_id: str, output: object, *, finished: bool) -> bool:
        """Intercept private output before response policy or speech routing."""
        if not duplex_is_auxiliary_request(request_id):
            return False
        if request_id == self._request_id and finished:
            self._post("auxiliary_output", {"request_id": request_id, "output": output})
        return True

    def on_error(self, request_id: str | None, error: BaseException) -> bool:
        if not duplex_is_auxiliary_request(request_id):
            return False
        self._post("auxiliary_error", {"request_id": request_id, "error": error})
        return True

    def handle_event(self, kind: str, payload: dict[str, object]) -> bool:
        """Handle private completions and timers on the runner's mailbox."""
        if kind == "auxiliary_ready":
            self.maybe_start()
            return True
        if kind == "auxiliary_retry":
            if payload["operation"] == self._generation and payload["source"] == self._source:
                self._retry_timer = None
                self._source = None
                self.maybe_start()
            return True
        if kind in {"auxiliary_output", "auxiliary_error"}:
            future = self._output
            if payload["request_id"] == self._request_id and future is not None and not future.done():
                if kind == "auxiliary_error":
                    error = payload["error"]
                    assert isinstance(error, BaseException)
                    future.set_exception(error)
                else:
                    future.set_result(payload["output"])
            return True
        if kind == "auxiliary_result":
            if payload["operation"] != self._generation:
                return True
            self._completion_pending = False
            request = payload["request"]
            epoch, config_revision = payload["epoch"], payload["config_revision"]
            assert isinstance(request, DuplexAuxiliaryRequest)
            assert isinstance(epoch, int) and isinstance(config_revision, int)
            if self._context_matches(request, epoch, config_revision):
                output = payload["output"]
                try:
                    applied = output is not None and self._ctx.plugin.apply_auxiliary_result(
                        request=request, output=output, state=self._ctx.model_state
                    )
                except Exception as exc:
                    logger.warning("Duplex auxiliary result failed: %s", type(exc).__name__)
                    applied = False
                if applied:
                    self._source = None
                    self.cancel_retry()
                else:
                    self._schedule_retry(request)
            self.maybe_start()
            return True
        return False

    def close(self) -> None:
        self.cancel_retry()
        if self._task is not None and not helpers.task_is_cancelling(self._task):
            self._task.cancel()

    def note_user_item(self, payload: Mapping[str, object], item_id: str) -> None:
        if self._interactive_input(payload, has_new_user_input=True):
            self._pending_user_items.add(item_id)

    def forget_user_item(self, item_id: str) -> None:
        self._pending_user_items.discard(item_id)

    def note_committed_input(self, payload: Mapping[str, object], *, has_new_user_input: bool | None = None) -> None:
        if self._interactive_input(
            payload,
            has_new_user_input=self._interaction_pending if has_new_user_input is None else has_new_user_input,
        ):
            self._input_pending = True

    def forget_committed_input(self) -> None:
        """Discard cleared audio intent, preserving text items and active work."""
        self._input_pending = False

    def begin_append(self, payload: Mapping[str, object], *, silence_continuation: bool) -> bool:
        interactive = not silence_continuation and self._interactive_input(
            payload, has_new_user_input=self._interaction_pending
        )
        if interactive:
            self._clear_interaction_pending()
            self._interaction_active = True
            self._interaction_epoch = self._ctx.session.epoch
        return interactive

    def bind_interaction_turn(self, turn_id: int) -> None:
        self._interaction_turn = turn_id

    def cancel_interaction(self) -> None:
        self._clear_interaction_pending()
        self._interaction_active = False

    def _context_matches(self, request: DuplexAuxiliaryRequest, epoch: int, config_revision: int) -> bool:
        if self._closing or self._ctx.session.epoch != epoch or self._ctx.session.config_revision != config_revision:
            return False
        return self._ctx.plugin.auxiliary_source_matches(
            request=request, state=self._ctx.model_state, history=self._current_history()
        )

    def _current_history(self) -> tuple[dict[str, object], ...]:
        """Reuse the conversation view until a committed history revision changes."""
        if self._history_revision != self._ctx.session.history_revision:
            self._history = self._ctx.session.history
            self._history_revision = self._ctx.session.history_revision
        return self._history

    @property
    def _interaction_pending(self) -> bool:
        """Unconsumed input intent, including only user items still present."""
        return self._input_pending or bool(self._pending_user_items)

    def _clear_interaction_pending(self) -> None:
        """Consume or cancel both item-backed and committed-input intent."""
        self._input_pending = False
        self._pending_user_items.clear()

    def cancel_retry(self) -> None:
        """Forget a retired source's retry budget and its single timer."""
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None
        self._retry_source = None
        self._failures = 0

    def _interactive_input(self, payload: Mapping[str, object], *, has_new_user_input: bool) -> bool:
        """Record input intent without cancelling work needed to admit it."""
        return self._ctx.plugin.supports_auxiliary_requests and self._ctx.plugin.auxiliary_preempted_by_input(
            payload=payload, state=self._ctx.model_state, has_new_user_input=has_new_user_input
        )

    def preempt_for_submission(self) -> None:
        """Give a successfully prepared interactive append the model slot.

        Capacity rejection happens before this point: an inadmissible frame
        must leave the summary that frees visual memory running.
        """
        self.cancel_retry()
        self._generation += 1
        self._completion_pending = False
        self._source = None
        if self._task is not None and not self._task.done():
            if not helpers.task_is_cancelling(self._task):
                self._task.cancel()

    def _schedule_retry(self, request: DuplexAuxiliaryRequest) -> None:
        """Back off submitted failures without letting traffic reset the budget."""
        source = self._source
        if source is None or (not request.retry_delays_s and request.retry_cooldown_s is None):
            return
        if self._retry_source != source:
            self.cancel_retry()
            self._retry_source = source
        self._failures = min(self._failures + 1, len(request.retry_delays_s) + 1)
        if self._failures > len(request.retry_delays_s):
            logger.warning("Duplex background memory exhausted its short retry budget")
            delay = request.retry_cooldown_s
            if delay is None:
                return
        else:
            delay = request.retry_delays_s[self._failures - 1]
        self._retry_timer = asyncio.get_running_loop().call_later(
            delay,
            self._post,
            "auxiliary_retry",
            {"source": source, "operation": self._generation},
        )

    def maybe_start(self) -> None:
        if not self._ctx.plugin.supports_auxiliary_requests or self._closing:
            return
        history = self._current_history()
        # Policy also advances model-owned camera chunks while summaries run;
        # unchanged text history alone cannot establish that it is a no-op.
        policy = self._ctx.plugin.auxiliary_policy(
            state=self._ctx.model_state, history_revision=self._ctx.session.history_revision, history=history
        )
        if self._interaction_epoch != self._ctx.session.epoch:
            self._interaction_epoch = self._ctx.session.epoch
            self._interaction_active = False
            self._clear_interaction_pending()
            self._interaction_turn = None
        if self._interaction_active and (
            (
                self._interaction_turn is not None
                and self._ctx.session.turn_id > self._interaction_turn
                and self._ctx.session.active_response_turn_id != self._interaction_turn
            )
            or (
                self._ctx.session.active_response_id is None
                and not self._ctx.tasks.append_tasks
                and self._ctx.run.stream_request_id is None
            )
        ):
            self._interaction_active = False
            self._interaction_turn = None
        if self._completion_pending:
            return
        if self._task is not None and not self._task.done():
            if self._ctx.session.epoch != self._epoch:
                self._task.cancel()
            return
        # A failed abort retains its resource for teardown retry. Do not build
        # an unbounded collection of abandoned background requests behind it.
        if any(duplex_is_auxiliary_request(request_id) for _, request_id in self._ctx.session.request_resources):
            return
        if policy is None:
            self._source = None
            self.cancel_retry()
            return
        source = (self._ctx.session.epoch, self._ctx.session.config_revision, policy.source_key)
        if self._retry_source is not None and self._retry_source != source:
            self.cancel_retry()
        if self._interaction_pending or self._interaction_active:
            return
        if not policy.allow_during_response and (
            self._ctx.session.active_response_id is not None or self._ctx.tasks.append_tasks
        ):
            return
        # An attempted source remains parked until success, a bounded retry,
        # changed evidence, or explicit user interaction releases it.
        if source == self._source:
            return
        self._source = source
        self._epoch = self._ctx.session.epoch
        self._generation += 1
        self._task = asyncio.create_task(self._run(duplex_text_history(history)), name="duplex-auxiliary")
        self._ctx.services.spawn(self._task, name="duplex-auxiliary")
        self._task.add_done_callback(self._wake)

    async def _run(self, history: tuple[tuple[str, str], ...]) -> None:
        session = self._ctx.session
        epoch, config_revision = session.epoch, session.config_revision
        generation = session.config_generation
        operation = self._generation
        request_id = None
        request = None
        output = None
        submission_started = False
        try:
            request = await asyncio.wait_for(
                self._ctx.plugin.prepare_auxiliary_request(
                    history=history,
                    runtime_config=session.runtime_config,
                    state=self._ctx.model_state,
                ),
                timeout=30.0,
            )
            if request is None or not self._context_matches(request, epoch, config_revision):
                return
            if (
                not 0 <= request.stage_id < self._ctx.stage_port.stage_count
                or not math.isfinite(request.timeout_s)
                or request.timeout_s <= 0
                or any(not math.isfinite(delay) or delay < 0 for delay in request.retry_delays_s)
                or (
                    request.retry_cooldown_s is not None
                    and (not math.isfinite(request.retry_cooldown_s) or request.retry_cooldown_s <= 0)
                )
            ):
                raise ValueError("Invalid auxiliary stage or deadline")
            request_id = duplex_resource_request_id(session.fence, f"auxiliary-{uuid.uuid4().hex}")
            self._request_id = request_id
            output_future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
            self._output = output_future
            sampling = list(self._ctx.manager.sampling_params_for(session))
            sampling[request.stage_id] = request.sampling_params
            context = DuplexStageRequestContext(
                request_id=request_id,
                session_id=session.session_id,
                fence=session.fence,
                stage_id=request.stage_id,
                final_stage_id=request.stage_id,
                config_generation=generation,
                sampling_params=tuple(sampling),
            )
            session.reserve_stage_request(request.stage_id, request_id, fence=context.fence)
            # Mark submitted before yielding so concurrent close always aborts
            # an in-flight submit rather than merely dropping its reservation.
            session.request_resources[(request.stage_id, request_id)].submitted = True
            self._ctx.stage_port.ensure_request(context)
            self._ctx.manager.register_request(request_id, session.session_id)

            async def submit_and_wait() -> object:
                nonlocal submission_started
                submission_started = True
                result = await self._ctx.stage_port.submit(
                    DuplexStageSubmission(context, request.prompt, already_submitted=False, resumable=False)
                )
                if result.request_id != request_id or result.stage_id != request.stage_id:
                    raise RuntimeError("Auxiliary stage submission returned a mismatched identity")
                if not self._context_matches(request, epoch, config_revision):
                    return None
                return await output_future

            output = await asyncio.wait_for(submit_and_wait(), timeout=request.timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Duplex auxiliary request failed: %s", type(exc).__name__)
        finally:
            if request_id is not None:
                assert request is not None
                cancelled_during_cleanup = False
                try:
                    # Cancellation may already have cleaned and released this
                    # request. A failed cancellation leaves it owned for retry.
                    if (request.stage_id, request_id) in session.request_resources:
                        cleanup = asyncio.create_task(
                            asyncio.wait_for(
                                self._ctx.stage_port.cleanup([request_id], abort=True),
                                timeout=min(request.timeout_s, _CLEANUP_TIMEOUT_S),
                            )
                        )
                        # Close/preemption and runner shutdown can all cancel
                        # this task. Join the same cleanup RPC before releasing
                        # ownership, even when cancellation arrives repeatedly.
                        # Cleanup has its own short cap even when model work
                        # permits a much longer generation deadline.
                        while not cleanup.done():
                            try:
                                await asyncio.shield(cleanup)
                            except asyncio.CancelledError:
                                cancelled_during_cleanup = True
                        cleanup.result()
                except Exception as exc:
                    # Keep the reservation for the manager's teardown retry.
                    logger.warning("Duplex auxiliary cleanup failed: %s", type(exc).__name__)
                else:
                    session.request_resources.pop((request.stage_id, request_id), None)
                    self._ctx.manager.unregister_request(request_id)
                finally:
                    if self._request_id == request_id:
                        if self._output is not None and self._output.done():
                            if not self._output.cancelled():
                                self._output.exception()
                        self._request_id = None
                        self._output = None
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError
        # Queue completion after cleanup, so the mailbox may schedule the next
        # bounded chunk without racing the previous task's owned resources.
        if request is not None and submission_started and operation == self._generation:
            self._completion_pending = True
            self._post(
                "auxiliary_result",
                {
                    "request": request,
                    "output": output,
                    "epoch": epoch,
                    "config_revision": config_revision,
                    "operation": operation,
                },
            )

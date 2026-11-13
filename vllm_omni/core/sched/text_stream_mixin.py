# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The AR scheduler's side of streamed text (``append_text``; see ``text_stream.py``).

Admission of a request that declares a text stream, the waiting and running
holds, the deadline, and the per-step hand-over of new ids to the worker. All
of it runs only while some request has a text stream, apart from one key
lookup when a request is added.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request

from vllm_omni.core.sched.omni_scheduler_mixin import finish_requests_with_error
from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.core.sched.text_stream import (
    TEXT_STREAM_KEY,
    TextStreams,
    declares_text_stream,
    load_text_stream_policy,
    text_stream_timeout_s,
)
from vllm_omni.engine.serialization import deserialize_additional_information

if TYPE_CHECKING:
    # Mixed into the AR scheduler beside OmniSchedulerMixin.
    from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin as _SchedulerBase
else:
    _SchedulerBase = object

logger = init_logger(__name__)


def _payload_has_key(payload: object, key: str) -> bool:
    """Whether an additional_information payload carries ``key``, without decoding its values."""
    if isinstance(payload, dict):
        return key in payload
    entries = getattr(payload, "entries", None)
    return isinstance(entries, dict) and key in entries


def restore_running_order(running: list[Request], held: list[Request], order: list[Request]) -> list[Request]:
    """``running`` after a step, with the held requests back where they stood before it.

    ``order`` is ``running`` before the held requests were taken out. A held
    request goes back before the first request that stood after it, and before
    any request admitted in this step.
    """
    rank = {id(request): index for index, request in enumerate(order)}
    pending = sorted(held, key=lambda request: rank[id(request)])
    merged: list[Request] = []
    next_held = 0
    for request in running:
        position = rank.get(id(request))
        while next_held < len(pending) and (position is None or rank[id(pending[next_held])] < position):
            merged.append(pending[next_held])
            next_held += 1
        merged.append(request)
    merged.extend(pending[next_held:])
    return merged


def restore_queue_order(queue: RequestQueue, parked: list[Request], order: list[Request]) -> None:
    """Put parked requests back into a waiting queue where they stood before the step.

    ``order`` is the queue before they were taken out. A first-come-first-served
    queue gets each one back before the first request that stood after it; a
    priority queue places them by its own order.
    """
    if not isinstance(queue, deque):
        queue.prepend_requests(parked)
        return
    rank = {id(request): index for index, request in enumerate(order)}
    pending = sorted(parked, key=lambda request: rank[id(request)])
    merged: list[Request] = []
    next_parked = 0
    for request in queue:
        position = rank.get(id(request))
        while position is not None and next_parked < len(pending) and rank[id(pending[next_parked])] < position:
            merged.append(pending[next_parked])
            next_parked += 1
        merged.append(request)
    merged.extend(pending[next_parked:])
    queue.clear()
    queue.extend(merged)


@dataclass
class TextHold:
    """Requests kept out of one scheduler step because their text has not arrived."""

    # (waiting queue, the requests taken from it, the queue before): no KV cache, no seat yet.
    parked: list[tuple[RequestQueue, list[Request], list[Request]]] = field(default_factory=list)
    # Requests taken out of ``running``: they keep their KV cache and seat.
    held: list[Request] = field(default_factory=list)
    # ``running`` before the held requests were taken out.
    order: list[Request] = field(default_factory=list)


class TextStreamMixin(_SchedulerBase):
    """Streamed text for the AR scheduler."""

    _text_streams: TextStreams

    def _init_text_streams(self) -> None:
        """Load the stage's text-stream policy (none: requests that declare a stream are refused)."""
        model_config = self.vllm_config.model_config
        policy = load_text_stream_policy(getattr(model_config, "text_stream_policy", None))
        self._text_streams = TextStreams(
            policy,
            # Only a stage that streams text reads the deadline setting.
            text_stream_timeout_s() if policy is not None else 0.0,
            model_config=model_config,
        )

    def _open_text_stream(self, request: Request) -> None:
        """Register the text stream a new request declares, or fail a request that cannot stream."""
        streams = getattr(self, "_text_streams", None)
        payload = getattr(request, "additional_information", None)
        if streams is None or request.request_id in streams or not _payload_has_key(payload, TEXT_STREAM_KEY):
            return
        info = deserialize_additional_information(payload)
        if not declares_text_stream(info):
            return
        if self._talker_kstep_armed():
            reason = "Streamed text is not supported with multi-frame Talker decode."
        else:
            reason = streams.register(request.request_id, info)
        if reason is not None:
            self._fail_with_error({request.request_id: reason})

    if TYPE_CHECKING:
        # Provided by the AR scheduler.
        def _talker_kstep_armed(self) -> bool: ...

    def append_text(self, request_id: str, token_ids: list[int], final: bool) -> None:
        """Add text ids that arrived for a request that declared a text stream.

        Raises ``ValueError`` when the append cannot be applied. An append that
        breaks a live request's stream (no stream declared, an ended stream, a
        bad id) also ends that request with the reason as an ERROR output; an
        append for a request that already finished changes nothing.
        """
        request = self.requests.get(request_id)
        if request is None or request.is_finished():
            raise ValueError(f"Request {request_id} is not running on this stage.")
        try:
            self._text_streams.append(request_id, token_ids, final)
        except ValueError as exc:
            self._fail_with_error({request_id: str(exc)})
            raise

    def _fail_with_error(self, reasons: dict[str, str]) -> None:
        """Finish the requests the scheduler still holds, each with its reason as an ERROR output."""
        present = {request_id: reason for request_id, reason in reasons.items() if request_id in self.requests}
        if present:
            logger.warning("Failing %d request(s) on their text stream: %s", len(present), present)
            finish_requests_with_error(self, present)

    def _preempt_request(self, request: Request, timestamp: float, drop_stale_output: bool = False) -> None:
        """End a streamed request instead of replaying it, whatever the stage's preemption mode.

        A replay (recompute preemption, or a prefix-cache reset of running
        requests) rebuilds the request from its token stream, which holds none
        of the streamed text rows the Talker has already read. Requests without
        a stream preempt as before.
        """
        streams = getattr(self, "_text_streams", None)
        if streams is not None and request.request_id in streams:
            self._fail_preempted_request(
                request,
                "Request was preempted and its streamed text cannot be replayed; "
                "the output generated so far is truncated.",
            )
            return
        super()._preempt_request(request, timestamp, drop_stale_output=drop_stale_output)

    def _text_stream_frame(self, request: Request) -> int:
        """The codec frame the request's next step samples (0 before and during its prefill).

        ``num_computed_tokens`` counts every scheduled token, so under async
        scheduling it includes the steps still in flight (the confirmed tokens
        alone would lag one frame per step in flight). A streamed request is
        never replayed (see ``_preempt_request``), so ``num_computed_tokens``
        below the prompt means its first prefill.
        """
        prompt_len = request.num_prompt_tokens
        if request.num_computed_tokens < prompt_len:
            return 0
        return request.num_computed_tokens - prompt_len + 1

    def _fail_text_stream_requests(self) -> None:
        """Fail requests past their text deadline, and requests whose text ended shorter than
        their prompt needs. Runs before any request leaves a queue for the step."""
        streams = getattr(self, "_text_streams", None)
        if not streams:
            return
        reasons: dict[str, str] = {}
        expired = streams.expired()
        if expired:
            reason = f"No text arrived for {streams.timeout_s:g}s while the request waited for it."
            reasons.update(dict.fromkeys(expired, reason))
        for queue in (self.waiting, self.kv_holding_waiting):
            for request in queue:
                if request.request_id in streams and request.num_computed_tokens == 0:
                    short = streams.ended_short(request.request_id)
                    if short is not None:
                        reasons[request.request_id] = short
        if reasons:
            self._fail_with_error(reasons)

    def _hold_for_text(self) -> TextHold | None:
        """Keep out of this step every request whose text for the step has not arrived.

        ``None`` while no request streams text. Release the hold with
        :meth:`_release_text_hold` once the step is built.
        """
        streams = getattr(self, "_text_streams", None)
        if not streams:
            return None
        hold = TextHold(parked=self._park_text_starved_waiting(), order=list(self.running))
        hold.held = self._hold_text_starved_running()
        return hold

    def _park_text_starved_waiting(self) -> list[tuple[RequestQueue, list[Request], list[Request]]]:
        """Take out of the waiting queues the requests whose prompt text has not arrived."""
        streams = self._text_streams
        parked: list[tuple[RequestQueue, list[Request], list[Request]]] = []
        for queue in (self.waiting, self.kv_holding_waiting):
            starved = [
                request
                for request in queue
                if request.request_id in streams
                and request.num_computed_tokens == 0
                and not streams.ready(request.request_id, 0)
            ]
            if starved:
                order = list(queue)
                queue.remove_requests(starved)
                parked.append((queue, starved, order))
        return parked

    def _hold_text_starved_running(self) -> list[Request]:
        """Take out of ``running`` the requests whose next step reads text that has not arrived."""
        streams = self._text_streams
        held = [
            request
            for request in self.running
            if request.request_id in streams and not streams.ready(request.request_id, self._text_stream_frame(request))
        ]
        if held:
            held_ids = {id(request) for request in held}
            self.running[:] = [request for request in self.running if id(request) not in held_ids]
        return held

    def _release_text_hold(self, hold: TextHold) -> None:
        """Put the held and parked requests back where they stood in ``running`` and the waiting queues."""
        if hold.held:
            self.running[:] = restore_running_order(self.running, hold.held, hold.order)
        for queue, parked, order in hold.parked:
            restore_queue_order(queue, parked, order)

    def _attach_text_stream_deliveries(self, scheduler_output: OmniSchedulerOutput) -> None:
        """Hand each scheduled streamed request the text that arrived since its last step."""
        streams = self._text_streams
        deliveries = scheduler_output.text_stream_deliveries
        for request_id in scheduler_output.num_scheduled_tokens:
            if request_id in streams:
                delivery = streams.take_delivery(request_id)
                if delivery is not None:
                    deliveries[request_id] = delivery

    def _release_text_stream(self, request_id: str) -> None:
        """Drop the stream of a finished or aborted request."""
        streams = getattr(self, "_text_streams", None)
        if streams:
            streams.release((request_id,))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Text that reaches a running request after it starts.

A request declares ``additional_information["text_stream"]`` when it is
added, and its text token ids then arrive through ``append_text`` until a
final append ends the stream. The stage opts in by naming, in its model config
(``text_stream_policy``), a function of the request's additional information
and the stage's model config that returns a :class:`TextStreamSpec`: how many
streamed ids the request's prompt reads, and the ids the model accepts. After
its prompt the model reads one more id for every codec frame it samples
(:func:`text_row`).

The scheduler owns the stream state (:class:`TextStreams`). It keeps a request
out of any step whose text has not arrived, hands newly arrived ids to the
worker with the request's next scheduled step, and fails a request that has
waited for text longer than the deadline. Requests without a stream never
reach this module.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.core.sched.output import TextStreamDelivery

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

TEXT_STREAM_KEY = "text_stream"

# How long a request may wait for text before the scheduler fails it. The
# clock starts when the request first cannot run for lack of text and restarts
# on every append. Set exactly 0 to disable the deadline. It is separate from
# VLLM_OMNI_INPUT_WAIT_TIMEOUT_S (stage input, 600 s by default): a running
# request that waits for text keeps its seat and its KV cache, and a client
# that streams model output is gone after a few seconds without text.
_TIMEOUT_ENV = "VLLM_OMNI_TEXT_STREAM_TIMEOUT_S"
_TIMEOUT_DEFAULT_S = 5.0


@dataclass(frozen=True)
class TextStreamSpec:
    """How a model reads one request's streamed text.

    Attributes:
        prompt_ids: Streamed ids the request's prompt reads. The request is not
            prefilled before they have arrived (or the stream has ended).
        min_ids: Fewest ids a stream may end with before the request's prefill;
            a shorter text fails the request.
        id_limit: Appended ids must be below this (the model's text
            vocabulary); ``None`` sets no limit.
    """

    prompt_ids: int
    min_ids: int = 0
    id_limit: int | None = None

    def __post_init__(self) -> None:
        if self.prompt_ids < 0 or self.min_ids < 0 or (self.id_limit is not None and self.id_limit <= 0):
            raise ValueError(f"text stream counts must be >= 0 and the id limit > 0, got {self}")


# (the request's additional information, the stage's model config) -> spec
TextStreamPolicy = Callable[[Mapping[str, Any], "ModelConfig | None"], TextStreamSpec]


def first_value(value: object) -> object:
    """Unwrap the list-wrapped ``additional_information`` value convention."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def declares_text_stream(info: Mapping[str, Any] | None) -> bool:
    """Whether a request's additional information declares a text stream."""
    if not info:
        return False
    return bool(first_value(info.get(TEXT_STREAM_KEY)))


def text_row(prompt_ids: int, frame: int) -> int:
    """Stream index of the text row read when the model samples codec ``frame``.

    Frame 0 is sampled at the end of the prefill, whose prompt holds ids
    ``[0, prompt_ids)``; each later frame reads the next id. Once the stream has
    ended, index ``num_ids`` is its END row, and PAD rows follow.
    """
    return prompt_ids - 1 + frame


def text_ready(prompt_ids: int, frame: int, num_ids: int, ended: bool) -> bool:
    """Whether the step that samples ``frame`` has every text row it reads."""
    return ended or text_row(prompt_ids, frame) < num_ids


def end_read(prompt_ids: int, frame: int, num_ids: int, ended: bool) -> bool:
    """Whether the model has read the text END by the time it samples ``frame``.

    Training never ends audio before the text, so codec EOS waits for this.
    """
    return ended and text_row(prompt_ids, frame) >= num_ids


def text_stream_timeout_s() -> float:
    """The deadline from ``VLLM_OMNI_TEXT_STREAM_TIMEOUT_S`` (seconds, 0 disables it)."""
    raw = os.environ.get(_TIMEOUT_ENV, str(_TIMEOUT_DEFAULT_S))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{_TIMEOUT_ENV}={raw!r} is not a number of seconds") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{_TIMEOUT_ENV}={raw!r} must be a finite number of seconds >= 0 (0 disables it)")
    return value


def deliver_streamed_text(model: object, scheduler_output: object) -> None:
    """Hand a model the streamed text that reaches a step's requests; runners call it once per step."""
    deliveries = getattr(scheduler_output, "text_stream_deliveries", None)
    if not deliveries:
        return
    append = getattr(model, "append_streamed_text", None)
    if append is None:
        raise RuntimeError(
            f"{type(model).__name__} does not accept streamed text; remove text_stream_policy from its stage config."
        )
    append(deliveries)


def load_text_stream_policy(qualname: str | None) -> TextStreamPolicy | None:
    """Import the stage's ``text_stream_policy`` function, or ``None`` when streaming is off."""
    if not qualname:
        return None
    policy = resolve_obj_by_qualname(str(qualname))
    if not callable(policy):
        raise TypeError(f"text_stream_policy {qualname!r} is not callable")
    return policy


@dataclass
class _Stream:
    spec: TextStreamSpec
    token_ids: list[int] = field(default_factory=list)
    ended: bool = False
    # Whether the worker has heard of the stream, and the ids (and whether the
    # END) already handed to it.
    announced: bool = False
    delivered: int = 0
    end_delivered: bool = False
    # Monotonic time since which the request has been unable to run for lack
    # of text, or None while it is not waiting for text.
    waiting_since: float | None = None


class TextStreams:
    """The text streams of the requests that declared one, owned by the scheduler."""

    def __init__(
        self,
        policy: TextStreamPolicy | None,
        timeout_s: float,
        *,
        model_config: ModelConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy = policy
        self._timeout_s = timeout_s
        self._model_config = model_config
        self._clock = clock
        self._streams: dict[str, _Stream] = {}

    def __bool__(self) -> bool:
        return bool(self._streams)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._streams

    @property
    def timeout_s(self) -> float:
        """Seconds a request may wait for text before it fails (0: no deadline)."""
        return self._timeout_s

    def register(self, request_id: str, info: Mapping[str, Any]) -> str | None:
        """Start the stream of a request that declared one; returns why it cannot stream, if it cannot."""
        if self._policy is None:
            return "This stage does not accept streamed text: its model config sets no text_stream_policy."
        try:
            spec = self._policy(info, self._model_config)
        except ValueError as exc:
            return str(exc)
        self._streams[request_id] = _Stream(spec=spec)
        return None

    def append(self, request_id: str, token_ids: Sequence[int], final: bool) -> None:
        """Add arrived text ids, and the END when ``final``; raises ``ValueError`` for a bad append."""
        stream = self._streams.get(request_id)
        if stream is None:
            raise ValueError(f"Request {request_id} has no open text stream.")
        if stream.ended:
            raise ValueError(f"The text stream of request {request_id} has already ended.")
        ids = [int(token_id) for token_id in token_ids]
        limit = stream.spec.id_limit
        if any(token_id < 0 or (limit is not None and token_id >= limit) for token_id in ids):
            bound = "" if limit is None else f" and below {limit}"
            raise ValueError(f"Text token ids must be >= 0{bound}.")
        stream.token_ids.extend(ids)
        stream.ended = bool(final)
        # New text restarts the deadline; the next schedule() decides whether
        # the request still waits.
        stream.waiting_since = None

    def ready(self, request_id: str, frame: int) -> bool:
        """Whether the request's step that samples ``frame`` may run, and the wait clock."""
        stream = self._streams.get(request_id)
        if stream is None:
            return True
        if text_ready(stream.spec.prompt_ids, frame, len(stream.token_ids), stream.ended):
            stream.waiting_since = None
            return True
        if stream.waiting_since is None:
            stream.waiting_since = self._clock()
        return False

    def ended_short(self, request_id: str) -> str | None:
        """Why a not-yet-prefilled request cannot start: its text ended below the model's minimum."""
        stream = self._streams.get(request_id)
        if stream is None or not stream.ended or len(stream.token_ids) >= stream.spec.min_ids:
            return None
        return (
            f"The text stream of request {request_id} ended after {len(stream.token_ids)} ids; "
            f"this request needs at least {stream.spec.min_ids}."
        )

    def expired(self) -> list[str]:
        """Requests that have waited for text longer than the deadline."""
        if self._timeout_s <= 0:
            return []
        now = self._clock()
        return [
            request_id
            for request_id, stream in self._streams.items()
            if stream.waiting_since is not None and now - stream.waiting_since > self._timeout_s
        ]

    def take_delivery(self, request_id: str) -> TextStreamDelivery | None:
        """The ids (and END) the worker has not received yet, marked as handed over."""
        stream = self._streams.get(request_id)
        if stream is None:
            return None
        start = stream.delivered
        new_ids = stream.token_ids[start:]
        if stream.announced and not new_ids and stream.end_delivered == stream.ended:
            return None
        stream.announced = True
        stream.delivered = len(stream.token_ids)
        stream.end_delivered = stream.ended
        return TextStreamDelivery(
            start=start,
            token_ids=list(new_ids),
            ended=stream.ended,
            prompt_ids=stream.spec.prompt_ids,
        )

    def release(self, request_ids: Iterable[str]) -> None:
        """Drop the streams of finished or aborted requests."""
        for request_id in request_ids:
            self._streams.pop(request_id, None)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The one duplex session state, owned by the engine-side ``DuplexSessionRunner``.

``DuplexEngineSession`` is the single session object: the input / response /
playback / conversation ledgers, the lease, the identity fence, the stage
request resources, the append sequencing, the model plugin's per-session state
(``model_state``) and the Realtime projection state (``projector``). There is
no cross-boundary fence protocol; ``session.fence`` is derived from the
session's own epoch/turn and ``accepted_fence`` is the monotonic high-water
mark of fences accepted for stage requests (``sync_fence()`` publishes the
current identity there).
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING
from uuid import uuid4

from vllm_omni.engine.duplex.config import (
    DuplexAssistantAudioTextMark,
    DuplexCapabilities,
    DuplexCommittedInput,
    DuplexPlaybackCommitPolicy,
    DuplexPlaybackCursor,
    DuplexPlaybackView,
    DuplexSessionConfig,
    DuplexSessionState,
    DuplexTurnEventType,
    DuplexTurnState,
    ResponseCreateOptions,
)
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.events import TurnEvent
from vllm_omni.engine.duplex.lease import (
    DuplexLeaseActivity,
    DuplexLeaseConfig,
    DuplexLeaseState,
)

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.plugin import DuplexModelSessionState
    from vllm_omni.engine.duplex.realtime_events import RealtimeProjectionState


def _default_lease() -> DuplexLeaseState:
    return DuplexLeaseState(config=DuplexLeaseConfig(), generation=0, last_activity=time.monotonic())


@dataclass
class InputBufferState:
    commit_seq: int = 0
    overlap_speech_ms: int = 0
    reserved_input_bytes: int = 0
    pending_turns: int = 0


@dataclass
class ResponseState:
    active_request_id: str | None = None
    active_response_id: str | None = None
    active_response_turn_id: int | None = None
    active_response_input_commit_seq: int | None = None
    active_response_awaits_input_commit: bool = False
    last_response_id: str | None = None
    assistant_text_buffer: list[str] = field(default_factory=list)
    assistant_audio_text_marks: list[DuplexAssistantAudioTextMark] = field(default_factory=list)
    pending_options: ResponseCreateOptions | None = None
    active_options: ResponseCreateOptions | None = None
    active_config: DuplexSessionConfig | None = None
    stage_metrics: dict[str, dict[str, object]] = field(default_factory=dict)
    stage_metric_tpot_weighted_ms: dict[str, float] = field(default_factory=dict)
    stage_metric_tpot_weight: dict[str, int] = field(default_factory=dict)


@dataclass
class PlaybackLedger:
    current: DuplexPlaybackCursor = field(default_factory=DuplexPlaybackCursor)
    by_response: dict[str, DuplexPlaybackCursor] = field(default_factory=dict)


@dataclass
class ConversationHistory:
    messages: list[dict[str, object]] = field(default_factory=list)
    item_ids: dict[str, dict[str, object]] = field(default_factory=dict)
    history_item_placeholders: dict[str, dict[str, object]] = field(default_factory=dict)
    item_audio_text_marks: dict[str, list[DuplexAssistantAudioTextMark]] = field(default_factory=dict)
    pending_item_ids: dict[str, dict[str, object]] = field(default_factory=dict)
    pending_item_audio_text_marks: dict[str, list[DuplexAssistantAudioTextMark]] = field(default_factory=dict)
    pending_item_input_commit_seqs: dict[str, int] = field(default_factory=dict)
    pending_truncations_ms: dict[str, int] = field(default_factory=dict)
    hard_truncations_ms: dict[str, int] = field(default_factory=dict)
    last_assistant_full_message: dict[str, object] | None = None
    last_assistant_audio_text_marks: list[DuplexAssistantAudioTextMark] = field(default_factory=list)
    assistant_response_snapshots: dict[str, tuple[dict[str, object], tuple[DuplexAssistantAudioTextMark, ...], int]] = (
        field(default_factory=dict)
    )


class DuplexFenceMismatchError(RuntimeError):
    def __init__(self, expected: DuplexFence, actual: DuplexFence) -> None:
        super().__init__(f"duplex fence mismatch: expected {expected!r}, got {actual!r}")
        self.expected = expected
        self.actual = actual


@dataclass
class DuplexRequestResource:
    stage_id: int
    request_id: str
    fence: DuplexFence
    submitted: bool = False


@dataclass
class DuplexInputAppend:
    seq: int
    turn_seq: int
    turn_id: int


@dataclass(frozen=True)
class DuplexAppendReservation:
    fence: DuplexFence
    base_fence: DuplexFence
    base_input_seq: int
    base_input_turn_seq: int
    base_append_turn_key: tuple[int, int, int] | None
    update: DuplexInputAppend


@dataclass
class DuplexEngineSession:
    """The one session state (see module docstring).

    Owned and mutated only by its ``DuplexSessionRunner`` on the orchestrator loop.
    """

    session_id: str
    config: DuplexSessionConfig
    capabilities: DuplexCapabilities = field(default_factory=DuplexCapabilities)
    state: DuplexSessionState = DuplexSessionState.OPEN
    turn_state: DuplexTurnState = DuplexTurnState.IDLE
    epoch: int = 0
    turn_id: int = 0
    lease: DuplexLeaseState = field(default_factory=_default_lease, repr=False)
    _clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _runtime_config: dict[str, object] = field(default_factory=dict, repr=False)
    #: Bumped on every published session / runtime config change.
    config_generation: int = 0
    #: Stage request ids reserved or submitted for this session, keyed by ``(stage_id, request_id)``.
    request_resources: dict[tuple[int, str], DuplexRequestResource] = field(default_factory=dict, repr=False)
    #: Highest fence accepted for a stage request (monotonic; reset of the append
    #: sequence happens when its epoch advances).
    accepted_fence: DuplexFence = field(default=None, repr=False)  # type: ignore[assignment]
    input_seq: int = 0
    input_turn_seq: int = 0
    _append_turn_key: tuple[int, int, int] | None = field(default=None, repr=False)
    _input: InputBufferState = field(default_factory=InputBufferState, repr=False)
    _response: ResponseState = field(default_factory=ResponseState, repr=False)
    #: Stage snapshots that arrived while no response was open. A stage whose
    #: output feeds the next stage rather than the client reports its metrics
    #: before the response those tokens end up in exists, so they are held here
    #: and folded into the first response that opens after them.
    _pending_stage_metrics: list[dict[str, dict[str, object]]] = field(default_factory=list, repr=False)
    _playback: PlaybackLedger = field(default_factory=PlaybackLedger, repr=False)
    _conversation: ConversationHistory = field(default_factory=ConversationHistory, repr=False)
    model_state: DuplexModelSessionState | None = field(default=None, repr=False)
    projector: RealtimeProjectionState | None = field(default=None, repr=False)
    created_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.accepted_fence is None:
            self.accepted_fence = self.fence
        else:
            self.accept_fence(self.fence)

    # ---- identity / fence ----

    @property
    def fence(self) -> DuplexFence:
        return DuplexFence(self.session_id, epoch=self.epoch, turn_id=self.turn_id)

    def sync_fence(self) -> DuplexFence:
        """Publish the current (epoch, turn_id) identity as the accepted fence."""
        fence = self.fence
        self.accept_fence(fence)
        return fence

    def _validate_fence(self, fence: DuplexFence) -> None:
        current = self.accepted_fence
        if fence.session_id != self.session_id:
            raise DuplexFenceMismatchError(current, fence)
        if fence.epoch < current.epoch or (fence.epoch == current.epoch and fence.turn_id < current.turn_id):
            raise DuplexFenceMismatchError(current, fence)

    def accept_fence(self, fence: DuplexFence) -> None:
        self._validate_fence(fence)
        if fence.epoch != self.accepted_fence.epoch:
            self.input_seq = 0
            self.input_turn_seq = 0
            self._append_turn_key = None
        self.accepted_fence = fence

    # ---- lease ----

    @property
    def lease_generation(self) -> int:
        return self.lease.generation

    def touch_lease(self, activity: DuplexLeaseActivity) -> None:
        self._validate_fence(self.fence)
        self.lease.touch(self._clock(), activity)

    def detach_lease(self) -> None:
        self._validate_fence(self.fence)
        self.lease.detach(self._clock())

    def resume_lease(self, *, expected_lease_generation: int) -> int:
        self._validate_fence(self.fence)
        return self.lease.resume(self._clock(), expected_generation=expected_lease_generation)

    def begin_lease_operation(self, fence: DuplexFence, operation_id: str) -> None:
        self._validate_fence(fence)
        self.lease.begin_operation(self._clock(), operation_id)

    def end_lease_operation(self, operation_id: str) -> None:
        self._validate_fence(self.fence)
        self.lease.end_operation(self._clock(), operation_id)

    # ---- stage request resources ----

    def reserve_stage_request(self, stage_id: int, request_id: str, *, fence: DuplexFence) -> None:
        self._validate_fence(fence)
        resource_key = (stage_id, request_id)
        existing = self.request_resources.get(resource_key)
        if existing is not None:
            if existing.fence.session_id != fence.session_id or existing.fence.epoch != fence.epoch:
                raise ValueError(f"Duplex request resource already reserved with different identity: {request_id}")
            return
        self.request_resources[resource_key] = DuplexRequestResource(
            stage_id=stage_id,
            request_id=request_id,
            fence=fence,
        )

    def bind_stage_request(self, stage_id: int, request_id: str, *, fence: DuplexFence) -> None:
        self.reserve_stage_request(stage_id, request_id, fence=fence)
        self.accept_fence(fence)
        resource = self.request_resources[(stage_id, request_id)]
        resource.fence = fence
        resource.submitted = True

    def stage_request_submitted(self, stage_id: int, request_id: str) -> bool:
        resource = self.request_resources.get((stage_id, request_id))
        return resource is not None and resource.submitted

    def resource_request_ids(
        self,
        fence: DuplexFence | None = None,
        *,
        submitted: bool | None = None,
    ) -> list[str]:
        return list(
            dict.fromkeys(
                resource.request_id
                for resource in self.request_resources.values()
                if (fence is None or resource.fence == fence) and (submitted is None or resource.submitted is submitted)
            )
        )

    def release_all_requests(self) -> list[str]:
        """Drop every reserved/submitted stage request; returns their ids for stage cleanup."""
        stale = self.resource_request_ids()
        self.request_resources.clear()
        return stale

    # ---- append sequencing ----

    def prepare_append(self, fence: DuplexFence) -> DuplexAppendReservation:
        self._validate_fence(fence)
        same_epoch = fence.epoch == self.accepted_fence.epoch
        input_seq = self.input_seq if same_epoch else 0
        input_turn_seq = self.input_turn_seq if same_epoch else 0
        append_turn_key = self._append_turn_key if same_epoch else None
        turn_key = (fence.epoch, fence.turn_id, 0)
        turn_seq = input_turn_seq + 1 if turn_key == append_turn_key else 1
        return DuplexAppendReservation(
            fence=fence,
            base_fence=self.accepted_fence,
            base_input_seq=self.input_seq,
            base_input_turn_seq=self.input_turn_seq,
            base_append_turn_key=self._append_turn_key,
            update=DuplexInputAppend(
                seq=input_seq + 1,
                turn_seq=turn_seq,
                turn_id=fence.turn_id,
            ),
        )

    def commit_append(self, reservation: DuplexAppendReservation) -> DuplexInputAppend:
        if (
            self.accepted_fence != reservation.base_fence
            or self.input_seq != reservation.base_input_seq
            or self.input_turn_seq != reservation.base_input_turn_seq
            or self._append_turn_key != reservation.base_append_turn_key
        ):
            raise RuntimeError("duplex append reservation is stale")
        self.accept_fence(reservation.fence)
        self.input_seq = reservation.update.seq
        self.input_turn_seq = reservation.update.turn_seq
        self._append_turn_key = (reservation.fence.epoch, reservation.fence.turn_id, 0)
        return reservation.update

    # ---- cancel / close of stage resources ----

    def release_fence(self, fence: DuplexFence) -> list[str]:
        stale = self.resource_request_ids(fence)
        self.request_resources = {
            resource_key: resource
            for resource_key, resource in self.request_resources.items()
            if resource.fence != fence
        }
        return stale

    def cancel_fence(self, cancelled_fence: DuplexFence, next_fence: DuplexFence) -> list[str]:
        stale = self.prepare_cancel_fence(cancelled_fence, next_fence)
        self.release_fence(cancelled_fence)
        return stale

    def prepare_cancel_fence(self, cancelled_fence: DuplexFence, next_fence: DuplexFence) -> list[str]:
        """Advance the cancellation fence without dropping cleanup records."""
        current = self.accepted_fence
        if cancelled_fence.session_id != self.session_id:
            raise DuplexFenceMismatchError(current, cancelled_fence)
        if next_fence.session_id != self.session_id or next_fence.epoch <= cancelled_fence.epoch:
            raise DuplexFenceMismatchError(cancelled_fence, next_fence)
        current_key = (current.epoch, current.turn_id)
        cancelled_key = (cancelled_fence.epoch, cancelled_fence.turn_id)
        next_key = (next_fence.epoch, next_fence.turn_id)
        if cancelled_key > current_key:
            raise DuplexFenceMismatchError(current, cancelled_fence)
        if next_key > current_key:
            self.accept_fence(next_fence)
        return self.resource_request_ids(cancelled_fence)

    def begin_close(self, *, reason: str) -> bool:
        """Make close irreversible while retaining stage resources for cleanup retry."""
        self.accept_fence(self.fence)
        if self.lease.terminal_reason is not None:
            return True
        return self.lease.mark_terminal(reason)

    # ---- configuration ----

    @property
    def response_config(self) -> DuplexSessionConfig:
        """Return the immutable-for-this-lifecycle response configuration.

        ``config`` remains the session defaults.  A response takes a deep
        snapshot when it begins so response.create overrides and concurrent
        session updates cannot mutate each other's ownership domains.
        """
        return self._response.active_config or self.config

    @property
    def runtime_config(self) -> Mapping[str, object]:
        return MappingProxyType(dict(self._runtime_config))

    def replace_runtime_config(self, runtime_config: Mapping[str, object]) -> None:
        self._runtime_config = dict(runtime_config)
        self.config_generation += 1

    @property
    def input_commit_seq(self) -> int:
        return self._input.commit_seq

    @property
    def history(self) -> tuple[dict[str, object], ...]:
        placeholders = {id(message) for message in self._conversation.history_item_placeholders.values()}
        return tuple(dict(message) for message in self._conversation.messages if id(message) not in placeholders)

    @property
    def active_request_id(self) -> str | None:
        return self._response.active_request_id

    @property
    def active_response_id(self) -> str | None:
        return self._response.active_response_id

    @property
    def active_response_turn_id(self) -> int | None:
        return self._response.active_response_turn_id

    @property
    def last_response_id(self) -> str | None:
        return self._response.last_response_id

    @property
    def overlap_speech_ms(self) -> int:
        return self._input.overlap_speech_ms

    @property
    def assistant_text_buffer(self) -> tuple[str, ...]:
        return tuple(self._response.assistant_text_buffer)

    @property
    def assistant_audio_text_marks(self) -> tuple[DuplexAssistantAudioTextMark, ...]:
        return tuple(self._response.assistant_audio_text_marks)

    @property
    def last_assistant_full_message(self) -> dict[str, object] | None:
        message = self._conversation.last_assistant_full_message
        return dict(message) if message is not None else None

    @property
    def last_assistant_audio_text_marks(self) -> tuple[DuplexAssistantAudioTextMark, ...]:
        return tuple(self._conversation.last_assistant_audio_text_marks)

    def has_assistant_response_item(self, response_id: str, item_id: str) -> bool:
        if item_id != f"item_{response_id}":
            return False
        if self.active_response_id == response_id:
            return True
        message = self._conversation.item_ids.get(item_id) or self._conversation.pending_item_ids.get(item_id)
        if isinstance(message, dict) and message.get("role") == "assistant":
            return True
        return response_id in self._conversation.assistant_response_snapshots

    def playback_ack_is_too_late(self, response_id: str, item_id: str) -> bool:
        # An earlier ACK reserved this item's exact history position.  Later
        # ACKs update that slot in place, so they cannot append an old
        # assistant turn after a newer user input.
        if item_id in self._conversation.item_ids or item_id in self._conversation.history_item_placeholders:
            return False
        input_commit_seq = self._conversation.pending_item_input_commit_seqs.get(item_id)
        if input_commit_seq is None:
            snapshot = self._conversation.assistant_response_snapshots.get(response_id)
            if snapshot is not None:
                input_commit_seq = snapshot[2]
        if input_commit_seq is None and self.active_response_id == response_id:
            input_commit_seq = self._response.active_response_input_commit_seq
        return input_commit_seq is not None and self.input_commit_seq > input_commit_seq

    def reserve_history_item(self, item_id: str) -> None:
        """Reserve the current history position for a response-owned item."""
        if item_id in self._conversation.item_ids or item_id in self._conversation.history_item_placeholders:
            return
        placeholder: dict[str, object] = {}
        self._conversation.history_item_placeholders[item_id] = placeholder
        self._conversation.messages.append(placeholder)

    def _store_history_item_message(
        self,
        item_id: str,
        message: dict[str, object],
    ) -> dict[str, object]:
        """Materialize or update a response item without changing its order."""
        existing = self._conversation.item_ids.get(item_id)
        if existing is not None:
            if existing is not message:
                existing.clear()
                existing.update(message)
            return existing

        placeholder = self._conversation.history_item_placeholders.pop(item_id, None)
        if placeholder is not None:
            for index, candidate in enumerate(self._conversation.messages):
                if candidate is placeholder:
                    self._conversation.messages[index] = message
                    break
            else:
                self._conversation.messages.append(message)
        elif not any(candidate is message for candidate in self._conversation.messages):
            self._conversation.messages.append(message)
        self._conversation.item_ids[item_id] = message
        return message

    def _discard_history_item_placeholder(self, item_id: str) -> bool:
        placeholder = self._conversation.history_item_placeholders.pop(item_id, None)
        if placeholder is None:
            return False
        self._conversation.messages = [
            candidate for candidate in self._conversation.messages if candidate is not placeholder
        ]
        return True

    def release_response_history_snapshot(self, response_id: str | None) -> None:
        """Release a final response snapshot after its playback fully commits."""
        if response_id is None or response_id == self.active_response_id:
            return
        self._conversation.assistant_response_snapshots.pop(response_id, None)
        self._conversation.hard_truncations_ms.pop(f"item_{response_id}", None)

    @property
    def playback(self) -> DuplexPlaybackView:
        return self._playback.current.snapshot()

    @property
    def history_item_ids(self) -> Mapping[str, dict[str, object]]:
        return MappingProxyType({key: dict(value) for key, value in self._conversation.item_ids.items()})

    @property
    def pending_history_item_ids(self) -> Mapping[str, dict[str, object]]:
        return MappingProxyType({key: dict(value) for key, value in self._conversation.pending_item_ids.items()})

    @property
    def pending_history_truncations_ms(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._conversation.pending_truncations_ms))

    def replace_config(self, config: DuplexSessionConfig) -> None:
        self.config = config
        self.config_generation += 1

    def transition_turn(self, state: DuplexTurnState) -> None:
        self.turn_state = state

    def transition_session(self, state: DuplexSessionState) -> None:
        self.state = state

    def bind_request(self, request_id: str | None) -> None:
        self._response.active_request_id = request_id

    def clear_request(self, expected_request_id: str | None = None) -> bool:
        if expected_request_id is not None and self._response.active_request_id != expected_request_id:
            return False
        self._response.active_request_id = None
        return True

    def bind_response_turn(self, turn_id: int | None) -> None:
        self._response.active_response_turn_id = turn_id

    def active_response_accepts_model_turn(self, turn_id: int | None) -> bool:
        if self._response.active_response_id is None:
            return False
        if turn_id is None:
            return True
        active_turn_id = self._response.active_response_turn_id
        return active_turn_id is None or int(turn_id) == active_turn_id

    def append_history_message(self, message: dict[str, object]) -> None:
        self._conversation.messages.append(message)

    @property
    def pending_input_bytes(self) -> int:
        return self._input.reserved_input_bytes

    @property
    def pending_input_turns(self) -> int:
        return self._input.pending_turns

    def reserve_input_bytes(self, size: int, *, limit: int) -> bool:
        size = max(0, int(size))
        if self._input.reserved_input_bytes + size > int(limit):
            return False
        self._input.reserved_input_bytes += size
        return True

    def release_input_bytes(self, size: int) -> None:
        self._input.reserved_input_bytes = max(0, self._input.reserved_input_bytes - max(0, int(size)))

    def release_all_input_bytes(self) -> None:
        self._input.reserved_input_bytes = 0

    def reserve_pending_turn(self, *, limit: int) -> bool:
        if self._input.pending_turns >= int(limit):
            return False
        self._input.pending_turns += 1
        return True

    def release_pending_turn(self) -> None:
        self._input.pending_turns = max(0, self._input.pending_turns - 1)

    def mark_user_input_activity(self) -> None:
        self.turn_state = DuplexTurnState.USER_SPEAKING

    def cancel_pending_input(self) -> dict[str, int]:
        """Drop every pending reservation; the PCM buffer itself is model state cleared by the runner."""
        cancelled = {"text_chunks": 0, "audio_chunks": 0}
        self._input.reserved_input_bytes = 0
        self._input.pending_turns = 0
        self.turn_state = DuplexTurnState.IDLE
        return cancelled

    def commit_audio_input(
        self,
        *,
        transcript: str | None = None,
        turn_id: int | None = None,
    ) -> DuplexCommittedInput:
        input_audio_part: dict[str, object] = {
            "type": "audio_url",
            "audio_url": {"url": "native-duplex:input-audio"},
        }
        if transcript:
            input_audio_part["transcript"] = transcript
        self._bind_active_response_to_input_commit(self._input.commit_seq + 1)
        self._input.commit_seq += 1
        message = {"role": "user", "content": [input_audio_part]}
        if transcript:
            message["transcript"] = transcript
        self._conversation.messages.append(message)
        self.turn_state = DuplexTurnState.USER_COMMITTED
        return DuplexCommittedInput(
            message=message,
            turn_id=self.turn_id if turn_id is None else int(turn_id),
            epoch=self.epoch,
            input_commit_seq=self._input.commit_seq,
        )

    def complete_model_turn(self, turn_id: int) -> None:
        """Advance the model-owned output identity after its terminal signal."""
        completed_turn_id = int(turn_id)
        if completed_turn_id >= self.turn_id:
            self.turn_id = completed_turn_id + 1
            self.sync_fence()

    def _bind_active_response_to_input_commit(self, input_commit_seq: int) -> None:
        if self.active_response_id is not None and self._response.active_response_awaits_input_commit:
            self._response.active_response_input_commit_seq = int(input_commit_seq)
            self._response.active_response_awaits_input_commit = False

    def reserve_response_options(self, options: ResponseCreateOptions) -> None:
        if self._response.active_response_id is not None:
            raise RuntimeError("response options cannot be reserved while a response is active")
        if self._response.pending_options is not None:
            raise RuntimeError("response options are already reserved")
        self._response.pending_options = options

    def discard_response_options(self) -> None:
        self._response.pending_options = None

    def _activate_response_options(self) -> None:
        options = self._response.pending_options
        self._response.active_config = copy.deepcopy(self.config)
        self._response.active_options = options
        self._response.pending_options = None
        if options is not None:
            options.apply_to(self._response.active_config)

    def _restore_response_config(self) -> None:
        self._response.active_config = None
        self._response.active_options = None
        self._response.pending_options = None

    def begin_response(self, *, turn_id: int | None = None) -> str:
        self._activate_response_options()
        response_id = f"resp-{self.session_id}-{self.epoch}-{uuid4().hex[:8]}"
        self._response.active_response_id = response_id
        self._response.active_response_turn_id = self.turn_id if turn_id is None else int(turn_id)
        self._response.active_response_input_commit_seq = self.input_commit_seq
        self._response.active_response_awaits_input_commit = self.turn_state == DuplexTurnState.USER_SPEAKING
        self._response.last_response_id = response_id
        self._response.assistant_text_buffer.clear()
        self._response.assistant_audio_text_marks.clear()
        self._clear_response_metrics()
        self._conversation.last_assistant_full_message = None
        self._conversation.last_assistant_audio_text_marks.clear()
        self._playback.current = DuplexPlaybackCursor()
        self._playback.by_response[response_id] = self._playback.current
        self.turn_state = DuplexTurnState.ASSISTANT_GENERATING
        return response_id

    def _clear_response_metrics(self) -> None:
        self._response.stage_metrics.clear()
        self._response.stage_metric_tpot_weighted_ms.clear()
        self._response.stage_metric_tpot_weight.clear()

    def stash_stage_metrics(self, stage_metrics: Mapping[object, object] | None) -> None:
        """Hold a stage snapshot until a response exists to attribute it to.

        A stage that hands its output to the next stage instead of the client
        (stage 0 feeding the TTS stage) reports its token metrics before the
        response carrying those tokens is created. Dropping them would leave
        every response without an engine-side token count; holding them keeps
        the count whole, at the cost of attributing a turn the model never
        spoke to the next response it does speak.
        """
        if not isinstance(stage_metrics, Mapping):
            return
        snapshot = {
            str(stage_id): dict(values) for stage_id, values in stage_metrics.items() if isinstance(values, Mapping)
        }
        if not snapshot:
            return
        if self.active_response_id is not None:
            self.accumulate_response_stage_metrics(snapshot)
            return
        self._pending_stage_metrics.append(snapshot)

    def accumulate_response_stage_metrics(
        self,
        stage_metrics: Mapping[object, object] | None,
    ) -> dict[str, dict[str, object]]:
        if self.active_response_id is None:
            return copy.deepcopy(self._response.stage_metrics)
        if self._pending_stage_metrics:
            pending, self._pending_stage_metrics = self._pending_stage_metrics, []
            for held in pending:
                self.accumulate_response_stage_metrics(held)
        if not isinstance(stage_metrics, Mapping):
            return copy.deepcopy(self._response.stage_metrics)

        additive_fields = (
            "num_tokens_in",
            "num_tokens_out",
            "stage_gen_time_ms",
            "postprocess_time_ms",
            "audio_generated_frames",
            "audio_duration_s",
            "image_pixels",
            "output_unit_count",
        )
        first_positive_fields = (
            "serving_time_to_first_output_ms",
            "vllm_ttft_ms",
        )
        interval_fields = (
            ("inter_output_latencies_ms", "inter_output_latency_ms"),
            ("vllm_itls_ms", "vllm_itl_ms"),
        )
        handled_fields = {
            *additive_fields,
            *first_positive_fields,
            "vllm_tpot_ms",
            *(name for pair in interval_fields for name in pair),
        }

        for raw_stage_id, raw_values in stage_metrics.items():
            if not isinstance(raw_values, Mapping):
                continue
            stage_id = str(raw_stage_id)
            current = self._response.stage_metrics.setdefault(stage_id, {})
            for name in additive_fields:
                value = raw_values.get(name)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    current[name] = current.get(name, 0) + value
            for name in first_positive_fields:
                value = raw_values.get(name)
                current_value = current.get(name)
                if (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and value > 0
                    and not (isinstance(current_value, int | float) and current_value > 0)
                ):
                    current[name] = value
            for list_name, mean_name in interval_fields:
                values = raw_values.get(list_name)
                if isinstance(values, list):
                    combined = list(current.get(list_name, []))
                    combined.extend(
                        value for value in values if isinstance(value, int | float) and not isinstance(value, bool)
                    )
                    current[list_name] = combined
                    current[mean_name] = sum(combined) / len(combined) if combined else 0.0

            tpot_ms = raw_values.get("vllm_tpot_ms")
            token_count = raw_values.get("num_tokens_out")
            if isinstance(tpot_ms, int | float) and tpot_ms > 0:
                weight = max(int(token_count) - 1, 1) if isinstance(token_count, int | float) else 1
                self._response.stage_metric_tpot_weighted_ms[stage_id] = (
                    self._response.stage_metric_tpot_weighted_ms.get(stage_id, 0.0) + float(tpot_ms) * weight
                )
                self._response.stage_metric_tpot_weight[stage_id] = (
                    self._response.stage_metric_tpot_weight.get(stage_id, 0) + weight
                )
                current["vllm_tpot_ms"] = self._response.stage_metric_tpot_weighted_ms[stage_id] / float(
                    self._response.stage_metric_tpot_weight[stage_id]
                )

            for name, value in raw_values.items():
                if name not in handled_fields:
                    current[str(name)] = copy.deepcopy(value)

        return copy.deepcopy(self._response.stage_metrics)

    def replace_response_stage_metric_snapshots(
        self,
        stage_metrics: Mapping[object, object] | None,
    ) -> dict[str, dict[str, object]]:
        """Merge cumulative chat snapshots by replacing each stage's latest value."""
        if self.active_response_id is None or not isinstance(stage_metrics, Mapping):
            return copy.deepcopy(self._response.stage_metrics)

        for raw_stage_id, raw_values in stage_metrics.items():
            if not isinstance(raw_values, Mapping):
                continue
            stage_id = str(raw_stage_id)
            self._response.stage_metrics[stage_id] = copy.deepcopy(dict(raw_values))
            self._response.stage_metric_tpot_weighted_ms.pop(stage_id, None)
            self._response.stage_metric_tpot_weight.pop(stage_id, None)
        return copy.deepcopy(self._response.stage_metrics)

    def accumulate_overlap_speech(self, duration_ms: int) -> int:
        self._input.overlap_speech_ms += max(0, int(duration_ms))
        return self._input.overlap_speech_ms

    def reset_overlap_speech(self) -> int:
        previous = self._input.overlap_speech_ms
        self._input.overlap_speech_ms = 0
        return previous

    def append_assistant_text(self, text: str) -> None:
        if text:
            self._response.assistant_text_buffer.append(text)

    def mark_audio_sent(
        self,
        duration_ms: int | None = None,
        *,
        text_chars: int | None = None,
        audio_text_marks: list[dict[str, object]] | None = None,
    ) -> None:
        playback = self._playback.current
        if duration_ms is not None:
            playback.generated_ms = max(playback.generated_ms, duration_ms)
            playback.sent_ms = max(playback.sent_ms, duration_ms)
            if text_chars is not None and text_chars >= 0:
                self._response.assistant_audio_text_marks.append(
                    DuplexAssistantAudioTextMark(
                        text_chars=int(text_chars),
                        audio_end_ms=max(0, int(duration_ms)),
                    )
                )
        if audio_text_marks:
            for raw_mark in audio_text_marks:
                if not isinstance(raw_mark, dict):
                    continue
                raw_text_chars = raw_mark.get("text_chars")
                raw_audio_end_ms = raw_mark.get("audio_end_ms", raw_mark.get("audio_ms"))
                if not isinstance(raw_text_chars, int | float) or not isinstance(raw_audio_end_ms, int | float):
                    continue
                self._response.assistant_audio_text_marks.append(
                    DuplexAssistantAudioTextMark(
                        text_chars=max(0, int(raw_text_chars)),
                        audio_end_ms=max(0, int(raw_audio_end_ms)),
                    )
                )
        self.turn_state = DuplexTurnState.ASSISTANT_PLAYING

    def _playback_cursor_for_response(self, response_id: str | None = None) -> DuplexPlaybackCursor:
        if response_id is None:
            return self._playback.current
        playback = self._playback.by_response.get(response_id)
        if playback is None:
            # A restored or legacy session may not have response-scoped state.
            # Keep its acknowledgement isolated from the active response.
            playback = DuplexPlaybackCursor()
            self._playback.by_response[response_id] = playback
        return playback

    def playback_for_response(self, response_id: str | None = None) -> DuplexPlaybackView:
        return self._playback_cursor_for_response(response_id).snapshot()

    def acknowledge_playback(
        self,
        played_ms: int,
        committed_ms: int | None = None,
        *,
        response_id: str | None = None,
    ) -> DuplexPlaybackView:
        playback = self._playback_cursor_for_response(response_id)
        playback.acknowledge(played_ms, committed_ms)
        return playback.snapshot()

    def truncate_playback_commit(
        self,
        committed_ms: int,
        *,
        response_id: str | None = None,
    ) -> DuplexPlaybackView:
        playback = self._playback_cursor_for_response(response_id)
        playback.truncate_committed(committed_ms)
        return playback.snapshot()

    def release_response_playback(self, response_id: str | None) -> None:
        if response_id is None or response_id == self.active_response_id:
            return
        self._playback.by_response.pop(response_id, None)

    def clear_playback_cursor(self) -> None:
        self._playback.current = DuplexPlaybackCursor()
        if self.active_response_id is not None:
            self._playback.by_response[self.active_response_id] = self._playback.current

    def end_response(
        self,
        *,
        commit_text: bool = True,
        playback_commit_policy: str | None = None,
        preserve_request: bool = False,
    ) -> dict[str, object] | None:
        response_id = self._response.active_response_id
        response_input_commit_seq = self._response.active_response_input_commit_seq
        if response_input_commit_seq is None:
            response_input_commit_seq = self.input_commit_seq
        response_history_is_late = self.input_commit_seq > response_input_commit_seq
        assistant_text = "".join(self._response.assistant_text_buffer).strip()
        message = None
        if assistant_text:
            self._conversation.last_assistant_full_message = {"role": "assistant", "content": assistant_text}
            self._conversation.last_assistant_audio_text_marks = list(self._response.assistant_audio_text_marks)
            if response_id is not None:
                self._conversation.assistant_response_snapshots[response_id] = (
                    copy.deepcopy(self._conversation.last_assistant_full_message),
                    tuple(copy.deepcopy(self._response.assistant_audio_text_marks)),
                    response_input_commit_seq,
                )
        if commit_text and assistant_text and not response_history_is_late:
            committed_text = self._playback_committed_text(
                assistant_text,
                playback_commit_policy=playback_commit_policy,
            )
        else:
            committed_text = ""
        if commit_text and committed_text:
            message = {"role": "assistant", "content": committed_text}
            item_id = f"item_{response_id}" if response_id is not None else None
            if item_id is not None and item_id in self._conversation.history_item_placeholders:
                self._store_history_item_message(item_id, message)
            else:
                self._conversation.messages.append(message)
        effective_playback_policy = playback_commit_policy or self.config.playback_commit_policy
        if (
            response_id is not None
            and assistant_text
            and message is None
            and effective_playback_policy == DuplexPlaybackCommitPolicy.ACK_ONLY.value
        ):
            self.register_history_item(f"item_{response_id}", None)
        elif response_id is not None and not assistant_text:
            self._discard_history_item_placeholder(f"item_{response_id}")
        self._response.assistant_text_buffer.clear()
        if not preserve_request:
            self._response.active_request_id = None
        self._response.active_response_id = None
        self._response.active_response_turn_id = None
        self._response.active_response_input_commit_seq = None
        self._response.active_response_awaits_input_commit = False
        self._clear_response_metrics()
        self.turn_state = DuplexTurnState.IDLE
        self._restore_response_config()
        return message

    def register_history_item(self, item_id: str | None, message: dict[str, object] | None) -> None:
        if not item_id:
            return
        response_id = item_id.removeprefix("item_") if item_id.startswith("item_") else None
        response_snapshot = (
            self._conversation.assistant_response_snapshots.get(response_id) if response_id is not None else None
        )
        response_audio_text_marks = list(response_snapshot[1]) if response_snapshot is not None else None
        if message is None:
            if response_snapshot is None:
                return
            last_message, audio_text_marks, _ = response_snapshot
            self._conversation.pending_item_ids[item_id] = copy.deepcopy(last_message)
            self._conversation.pending_item_input_commit_seqs[item_id] = response_snapshot[2]
            if audio_text_marks:
                self._conversation.pending_item_audio_text_marks[item_id] = list(copy.deepcopy(audio_text_marks))
            pending_audio_ms = self._conversation.pending_truncations_ms.get(item_id)
            if pending_audio_ms is not None:
                self.truncate_history_item(
                    item_id,
                    audio_end_ms=pending_audio_ms,
                    playback=self._playback_cursor_for_item_id(item_id),
                )
            return
        message = self._store_history_item_message(item_id, message)
        pending_audio_ms = self._conversation.pending_truncations_ms.get(item_id)
        self._conversation.pending_item_ids.pop(item_id, None)
        self._conversation.pending_item_audio_text_marks.pop(item_id, None)
        self._conversation.pending_item_input_commit_seqs.pop(item_id, None)
        if message.get("role") == "assistant":
            marks = (
                response_audio_text_marks
                if response_audio_text_marks is not None
                else self._response.assistant_audio_text_marks or self._conversation.last_assistant_audio_text_marks
            )
            if marks:
                self._conversation.item_audio_text_marks[item_id] = list(marks)
        if pending_audio_ms is not None:
            self.truncate_history_item(
                item_id,
                audio_end_ms=pending_audio_ms,
                playback=self._playback_cursor_for_item_id(item_id),
            )

    def delete_history_item(self, item_id: str) -> bool:
        response_id = item_id.removeprefix("item_") if item_id.startswith("item_") else None
        message = self._conversation.item_ids.pop(item_id, None)
        self._conversation.item_audio_text_marks.pop(item_id, None)
        pending = self._conversation.pending_item_ids.pop(item_id, None)
        self._conversation.pending_item_audio_text_marks.pop(item_id, None)
        self._conversation.pending_item_input_commit_seqs.pop(item_id, None)
        self._conversation.pending_truncations_ms.pop(item_id, None)
        self._conversation.hard_truncations_ms.pop(item_id, None)
        removed_placeholder = self._discard_history_item_placeholder(item_id)
        if response_id is not None:
            self._conversation.assistant_response_snapshots.pop(response_id, None)
        if message is None:
            return pending is not None or removed_placeholder
        try:
            self._conversation.messages.remove(message)
        except ValueError:
            pass
        return True

    def truncate_history_item(
        self,
        item_id: str,
        *,
        audio_end_ms: int,
        playback: DuplexPlaybackCursor | DuplexPlaybackView | None = None,
        hard: bool = False,
    ) -> bool:
        playback = playback or self._playback_cursor_for_item_id(item_id)
        audio_end_ms = max(0, int(audio_end_ms))
        if hard:
            previous_cap_ms = self._conversation.hard_truncations_ms.get(item_id)
            self._conversation.hard_truncations_ms[item_id] = (
                audio_end_ms if previous_cap_ms is None else min(previous_cap_ms, audio_end_ms)
            )
        hard_cap_ms = self._conversation.hard_truncations_ms.get(item_id)
        if hard_cap_ms is not None:
            audio_end_ms = min(audio_end_ms, hard_cap_ms)
        response_id = item_id.removeprefix("item_") if item_id.startswith("item_") else None
        response_snapshot = (
            self._conversation.assistant_response_snapshots.get(response_id) if response_id is not None else None
        )
        if response_snapshot is not None:
            full_message, full_marks, _ = response_snapshot
            message = copy.deepcopy(full_message)
            changed = self._truncate_message_to_audio_ms(
                message,
                audio_end_ms=audio_end_ms,
                marks=list(full_marks),
                playback=playback,
            )
            if not changed or self._message_text_len(message) <= 0:
                self._conversation.pending_truncations_ms[item_id] = max(0, int(audio_end_ms))
                return False
            self._store_history_item_message(item_id, message)
            if full_marks:
                self._conversation.item_audio_text_marks[item_id] = list(copy.deepcopy(full_marks))
            self._conversation.pending_item_ids.pop(item_id, None)
            self._conversation.pending_item_audio_text_marks.pop(item_id, None)
            self._conversation.pending_item_input_commit_seqs.pop(item_id, None)
            self._conversation.pending_truncations_ms.pop(item_id, None)
            return True

        message = self._conversation.item_ids.get(item_id)
        if message is None:
            pending = self._conversation.pending_item_ids.get(item_id)
            if pending is None:
                self._conversation.pending_truncations_ms[item_id] = max(0, int(audio_end_ms))
                return False
            message = dict(pending)
            changed = self._truncate_message_to_audio_ms(
                message,
                audio_end_ms=audio_end_ms,
                marks=self._conversation.pending_item_audio_text_marks.get(item_id),
                playback=playback,
            )
            if not changed or self._message_text_len(message) <= 0:
                if changed:
                    self._conversation.pending_item_ids.pop(item_id, None)
                    self._conversation.pending_item_audio_text_marks.pop(item_id, None)
                    self._conversation.pending_item_input_commit_seqs.pop(item_id, None)
                    self._conversation.pending_truncations_ms.pop(item_id, None)
                    if item_id.startswith("item_"):
                        self._conversation.assistant_response_snapshots.pop(item_id.removeprefix("item_"), None)
                return changed
            self._conversation.messages.append(message)
            self._conversation.item_ids[item_id] = message
            if item_id in self._conversation.pending_item_audio_text_marks:
                self._conversation.item_audio_text_marks[item_id] = list(
                    self._conversation.pending_item_audio_text_marks[item_id]
                )
            self._conversation.pending_item_ids.pop(item_id, None)
            self._conversation.pending_item_audio_text_marks.pop(item_id, None)
            self._conversation.pending_item_input_commit_seqs.pop(item_id, None)
            self._conversation.pending_truncations_ms.pop(item_id, None)
            if item_id.startswith("item_"):
                self._conversation.assistant_response_snapshots.pop(item_id.removeprefix("item_"), None)
            return True
        changed = self._truncate_message_to_audio_ms(
            message,
            audio_end_ms=audio_end_ms,
            marks=self._conversation.item_audio_text_marks.get(item_id),
            playback=playback,
        )
        if changed and self._message_text_len(message) <= 0:
            self._conversation.item_ids.pop(item_id, None)
            self._conversation.item_audio_text_marks.pop(item_id, None)
            try:
                self._conversation.messages.remove(message)
            except ValueError:
                pass
            if item_id.startswith("item_"):
                self._conversation.assistant_response_snapshots.pop(item_id.removeprefix("item_"), None)
        elif changed and item_id.startswith("item_"):
            self._conversation.assistant_response_snapshots.pop(item_id.removeprefix("item_"), None)
        return changed

    def _truncate_message_to_audio_ms(
        self,
        message: dict[str, object],
        *,
        audio_end_ms: int,
        marks: list[DuplexAssistantAudioTextMark] | None = None,
        playback: DuplexPlaybackCursor | DuplexPlaybackView | None = None,
    ) -> bool:
        content = message.get("content")
        if isinstance(content, str):
            keep_chars = self._text_chars_for_audio_ms(
                audio_end_ms,
                len(content),
                marks=marks,
                playback=playback,
            )
            message["content"] = content[:keep_chars].rstrip()
            return True
        if not isinstance(content, list):
            return False
        changed = False
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"output_audio", "audio", "audio_transcript"}:
                transcript = part.get("transcript")
                if isinstance(transcript, str):
                    keep_chars = self._text_chars_for_audio_ms(
                        audio_end_ms,
                        len(transcript),
                        marks=marks,
                        playback=playback,
                    )
                    part["transcript"] = transcript[:keep_chars].rstrip()
                    changed = True
            if part_type in {"output_text", "text"}:
                text = part.get("text")
                if isinstance(text, str):
                    keep_chars = self._text_chars_for_audio_ms(
                        audio_end_ms,
                        len(text),
                        marks=marks,
                        playback=playback,
                    )
                    part["text"] = text[:keep_chars].rstrip()
                    changed = True
        if not changed:
            return False
        return True

    @staticmethod
    def _message_text_len(message: dict[str, object]) -> int:
        content = message.get("content")
        if isinstance(content, str):
            return len(content)
        if not isinstance(content, list):
            return 0
        total = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("text", "transcript"):
                value = part.get(key)
                if isinstance(value, str):
                    total += len(value)
        return total

    def _playback_committed_text(
        self,
        assistant_text: str,
        *,
        playback_commit_policy: str | None = None,
    ) -> str:
        sent_ms = max(self.playback.sent_ms, self.playback.generated_ms)
        committed_ms = self.playback.committed_ms
        policy = playback_commit_policy or self.config.playback_commit_policy
        if sent_ms <= 0 or committed_ms >= sent_ms:
            return assistant_text
        if committed_ms <= 0:
            if policy == DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value:
                return assistant_text
            return ""
        keep_chars = self._text_chars_for_audio_ms(committed_ms, len(assistant_text))
        if keep_chars <= 0:
            return ""
        return assistant_text[:keep_chars].rstrip()

    def _text_chars_for_audio_ms(
        self,
        audio_end_ms: int,
        text_len: int,
        *,
        marks: list[DuplexAssistantAudioTextMark] | None = None,
        playback: DuplexPlaybackCursor | DuplexPlaybackView | None = None,
    ) -> int:
        if text_len <= 0:
            return 0
        audio_end_ms = max(0, int(audio_end_ms))
        marks = marks if marks is not None else self._response.assistant_audio_text_marks
        playback = playback or self._playback.current
        if not marks:
            sent_ms = max(1, playback.sent_ms, playback.generated_ms)
            return int(text_len * max(0.0, min(1.0, audio_end_ms / sent_ms)))
        marks = sorted(
            (mark for mark in marks if mark.audio_end_ms >= 0 and mark.text_chars >= 0),
            key=lambda mark: mark.audio_end_ms,
        )
        if not marks:
            return 0
        if audio_end_ms <= 0:
            return 0
        previous_ms = 0
        previous_chars = 0
        for mark in marks:
            mark_ms = max(previous_ms, mark.audio_end_ms)
            mark_chars = min(text_len, max(previous_chars, mark.text_chars))
            if audio_end_ms <= mark_ms:
                if mark_ms <= previous_ms:
                    return mark_chars
                ratio = (audio_end_ms - previous_ms) / max(1, mark_ms - previous_ms)
                return int(previous_chars + (mark_chars - previous_chars) * max(0.0, min(1.0, ratio)))
            previous_ms = mark_ms
            previous_chars = mark_chars
        final_ms = max(playback.sent_ms, playback.generated_ms, previous_ms)
        if audio_end_ms >= final_ms:
            return text_len
        ratio = (audio_end_ms - previous_ms) / max(1, final_ms - previous_ms)
        return int(previous_chars + (text_len - previous_chars) * max(0.0, min(1.0, ratio)))

    def _playback_cursor_for_item_id(self, item_id: str) -> DuplexPlaybackCursor | None:
        if not item_id.startswith("item_"):
            return None
        return self._playback.by_response.get(item_id.removeprefix("item_"))

    def barge_in(self) -> int:
        self.epoch += 1
        self.sync_fence()
        self._response.assistant_text_buffer.clear()
        self._response.assistant_audio_text_marks.clear()
        self._response.active_request_id = None
        self._response.active_response_id = None
        self._response.active_response_turn_id = None
        self._response.active_response_input_commit_seq = None
        self._response.active_response_awaits_input_commit = False
        self._clear_response_metrics()
        self._restore_response_config()
        self.turn_state = DuplexTurnState.BARGE_IN
        return self.epoch

    def mark_closing(self) -> None:
        if self.state != DuplexSessionState.CLOSED:
            self.state = DuplexSessionState.CLOSING

    def close(self) -> None:
        self.state = DuplexSessionState.CLOSED
        self.turn_state = DuplexTurnState.IDLE
        self._response.active_response_turn_id = None
        self._response.active_response_input_commit_seq = None
        self._response.active_response_awaits_input_commit = False
        self._clear_response_metrics()
        self._restore_response_config()

    def signal_turn(self, event_type: str, payload: Mapping[str, object] | None = None) -> TurnEvent:
        """Apply one external turn signal and return the typed ``turn.event``."""
        payload = payload or {}
        if event_type == DuplexTurnEventType.USER_STARTED.value:
            self.transition_turn(DuplexTurnState.USER_SPEAKING)
        elif event_type == DuplexTurnEventType.USER_COMMITTED.value:
            self.transition_turn(DuplexTurnState.USER_COMMITTED)
        elif event_type == DuplexTurnEventType.ASSISTANT_STARTED.value:
            self.transition_turn(DuplexTurnState.ASSISTANT_GENERATING)
        elif event_type == DuplexTurnEventType.ASSISTANT_DONE.value:
            self.transition_turn(DuplexTurnState.IDLE)
        elif event_type == DuplexTurnEventType.PLAYBACK_ACK.value:
            played_ms = int(payload.get("played_ms", 0) or 0)
            committed_ms = payload.get("committed_ms")
            self.acknowledge_playback(
                played_ms,
                int(committed_ms) if isinstance(committed_ms, int | float) else None,
            )
        elif event_type == DuplexTurnEventType.BARGE_IN.value:
            self.transition_turn(DuplexTurnState.BARGE_IN)
        elif event_type in {DuplexTurnEventType.CLOSE.value, DuplexTurnEventType.TIMEOUT.value}:
            self.transition_session(DuplexSessionState.CLOSING)
        return TurnEvent(event=event_type, turn_state=self.turn_state.value)

    def as_public_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.session_id,
            "state": self.state.value,
            "turn_state": self.turn_state.value,
            "epoch": self.epoch,
            "turn_id": self.turn_id,
            "active_request_id": self.active_request_id,
            "active_response_id": self.active_response_id,
            "active_response_turn_id": self.active_response_turn_id,
            "model": self.config.model,
            "modalities": list(self.config.modalities),
            "instructions": self.config.instructions,
            "voice": self.config.voice,
            "response_format": self.config.response_format,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "speed": self.config.speed,
            "idle_timeout_s": self.config.idle_timeout_s,
            "overlap_policy": self.config.overlap_policy,
            "overlap_short_ack_ms": self.config.overlap_short_ack_ms,
            "overlap_barge_in_ms": self.config.overlap_barge_in_ms,
            "overlap_silence_rms": self.config.overlap_silence_rms,
            "playback_commit_policy": self.config.playback_commit_policy,
            "playback": self.playback.as_dict(),
            "capabilities": self.capabilities.as_dict(),
        }
        if isinstance(self.config.extra_body.get("realtime_tools"), list):
            payload["tools"] = self.config.extra_body["realtime_tools"]
        if isinstance(self.config.extra_body.get("realtime_tool_choice"), str | dict):
            payload["tool_choice"] = self.config.extra_body["realtime_tool_choice"]
        if isinstance(self.config.extra_body.get("realtime_metadata"), dict):
            payload["metadata"] = dict(self.config.extra_body["realtime_metadata"])
        if isinstance(self.config.extra_body.get("realtime_include"), list):
            payload["include"] = list(self.config.extra_body["realtime_include"])
        if isinstance(self.config.extra_body.get("realtime_prompt"), dict):
            payload["prompt"] = dict(self.config.extra_body["realtime_prompt"])
        if isinstance(self.config.extra_body.get("realtime_input_audio_transcription"), dict):
            payload["input_audio_transcription"] = dict(self.config.extra_body["realtime_input_audio_transcription"])
        if isinstance(self.config.extra_body.get("realtime_input_audio_noise_reduction"), dict):
            payload["input_audio_noise_reduction"] = dict(
                self.config.extra_body["realtime_input_audio_noise_reduction"]
            )
        if isinstance(self.config.extra_body.get("realtime_audio"), dict):
            payload["audio"] = dict(self.config.extra_body["realtime_audio"])
        if isinstance(self.config.extra_body.get("realtime_tracing"), str | dict):
            payload["tracing"] = self.config.extra_body["realtime_tracing"]
        raw_realtime_session = self.config.extra_body.get("realtime_session_payload")
        if isinstance(raw_realtime_session, dict):
            for key, value in raw_realtime_session.items():
                if key not in payload and key != "extra_body":
                    payload[key] = value
        return payload


__all__ = [
    "ConversationHistory",
    "DuplexAppendReservation",
    "DuplexEngineSession",
    "DuplexFenceMismatchError",
    "DuplexInputAppend",
    "DuplexRequestResource",
    "InputBufferState",
    "PlaybackLedger",
    "ResponseState",
]

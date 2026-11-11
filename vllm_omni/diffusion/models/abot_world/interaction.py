# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Mid-generation keyboard camera interaction for ABot-World."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler
from vllm_omni.diffusion.interaction.types import (
    MAX_PENDING_INTERACTION_EVENTS,
    InteractionChunkMetadata,
    InteractionPayload,
    InteractionSession,
)
from vllm_omni.diffusion.models.abot_world.actions import (
    CAMERA_ACTION_SCRIPT_KEY,
    PREPROCESSED_CAMERA_ACTION_SCRIPT_KEY,
    ABotCameraUpdate,
    parse_abot_camera_update,
)
from vllm_omni.diffusion.worker.utils import StepRequestState

ActionFrames = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _QueuedCameraUpdate:
    event_id: str
    received_at: float
    update: ABotCameraUpdate


@dataclass
class ABotCameraSession(InteractionSession):
    """Request-local ABot key state.

    ``chunk_actions`` holds one action list per latent frame for the chunk
    about to be generated, or ``None`` until an update has been applied.
    """

    pending: list[_QueuedCameraUpdate] = field(default_factory=list)
    mode: Literal["state", "script"] | None = None
    held_actions: tuple[str, ...] = ()
    script_frames: ActionFrames = ()
    script_event_id: str | None = None
    chunk_actions: ActionFrames | None = None


class ABotCameraInteractionHandler(InteractionHandler):
    """Apply ABot key actions at chunk boundaries.

    ``state`` updates hold keys across chunks; a press and release that both
    arrive before one boundary still produce a one-frame pulse. A ``script``
    update is a finite FIFO padded with neutral frames once exhausted. An
    update in one mode replaces the other mode's state.
    """

    modality: ClassVar[str] = "camera"
    needs_chunk_media: ClassVar[bool] = True
    lazy_initialize_session: ClassVar[bool] = False

    def validate_payload(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        self._parse(state, event_id=event_id, payload=payload, transition_chunks=transition_chunks)

    def enqueue(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        received_at: float,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        update = self._parse(state, event_id=event_id, payload=payload, transition_chunks=transition_chunks)
        session = self._session(state)
        with session.lock:
            session.pending.append(_QueuedCameraUpdate(event_id, received_at, update))

    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        boundary_at: float,
        chunk_index: int | None = None,
        num_media_frames: int | None = None,
        fps: float | None = None,
        num_latent_frames: int | None = None,
    ) -> InteractionChunkMetadata | None:
        del chunk_index, num_media_frames, fps
        if num_latent_frames is None:
            raise RuntimeError("ABot camera interaction requires the chunk's latent frame count.")
        session = self._session(state)
        with session.lock:
            session.last_boundary_at = boundary_at
            due = sorted(
                (queued for queued in session.pending if queued.received_at <= boundary_at),
                key=lambda queued: queued.received_at,
            )
            session.pending = [queued for queued in session.pending if queued.received_at > boundary_at]
            return _advance(session, due, num_latent_frames)

    def _parse(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> ABotCameraUpdate:
        if not event_id:
            raise ValueError("event_id must be nonempty.")
        if transition_chunks not in (None, 0):
            raise ValueError("ABot applies camera updates at one chunk boundary; transition_chunks must be 0.")
        if _request_is_scripted(state):
            raise ValueError("Cannot use mid-generation camera interaction together with camera_action_script.")
        update = parse_abot_camera_update(payload)
        session = state.interaction_sessions.get(self.modality)
        if isinstance(session, ABotCameraSession):
            with session.lock:
                if len(session.pending) >= MAX_PENDING_INTERACTION_EVENTS:
                    raise ValueError("Too many pending ABot camera interactions; wait for a chunk boundary.")
        return update

    def _session(self, state: StepRequestState) -> ABotCameraSession:
        session = state.interaction_sessions.setdefault(self.modality, ABotCameraSession())
        if not isinstance(session, ABotCameraSession):
            raise TypeError("ABot camera interaction session has an unexpected type.")
        return session


def _request_is_scripted(state: StepRequestState) -> bool:
    """Whether the request carries ``camera_action_script``.

    The sampling arguments exist from request creation; the parsed inputs only
    after ``prepare_encode``.
    """
    if getattr(state.extra.get("inputs"), "camera_action_script", None) is not None:
        return True
    extra_args = getattr(state.sampling, "extra_args", None) or {}
    return any(
        extra_args.get(key) is not None for key in (CAMERA_ACTION_SCRIPT_KEY, PREPROCESSED_CAMERA_ACTION_SCRIPT_KEY)
    )


def _advance(
    session: ABotCameraSession,
    due: list[_QueuedCameraUpdate],
    num_frames: int,
) -> InteractionChunkMetadata | None:
    started: list[str] = []
    completed: list[str] = []
    coalesced: list[str] = []
    window: list[str] = []
    transitions: ActionFrames = ()
    new_script: str | None = None

    def retire_script() -> None:
        nonlocal new_script
        if new_script is not None:
            coalesced.append(new_script)
            new_script = None
        elif session.script_event_id is not None:
            completed.append(session.script_event_id)
            session.script_event_id = None

    for queued in due:
        update = queued.update
        if update.mode == "script":
            coalesced.extend(window)
            window = []
            transitions = ()
            retire_script()
            new_script = queued.event_id
            session.mode = "script"
            session.script_frames = update.actions
        else:
            if session.mode != "state":
                retire_script()
                session.mode = "state"
                session.held_actions = ()
            window.append(queued.event_id)
            transitions += update.actions

    active: list[str] = []
    if session.mode == "script":
        frames = session.script_frames[:num_frames]
        session.script_frames = session.script_frames[num_frames:]
        script_id = new_script if new_script is not None else session.script_event_id
        if new_script is not None:
            started.append(new_script)
        if script_id is not None:
            active.append(script_id)
            if session.script_frames:
                session.script_event_id = script_id
            else:
                completed.append(script_id)
                session.script_event_id = None
        session.chunk_actions = frames + ((),) * (num_frames - len(frames))
    elif session.mode == "state":
        final = transitions[-1] if transitions else session.held_actions
        pulse = next((actions for actions in reversed(transitions) if actions), None)
        if pulse is not None and pulse != final:
            session.chunk_actions = (pulse,) + (final,) * (num_frames - 1)
        else:
            session.chunk_actions = (final,) * num_frames
        session.held_actions = final
        started.extend(window)
        active.extend(window)
        completed.extend(window)
    else:
        session.chunk_actions = None

    if not (started or active or completed or coalesced):
        return None
    return InteractionChunkMetadata(
        started_event_ids=started,
        active_event_ids=active,
        completed_event_ids=completed,
        coalesced_event_ids=coalesced,
    )

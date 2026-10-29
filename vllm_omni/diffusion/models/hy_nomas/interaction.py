# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler
from vllm_omni.diffusion.interaction.modality_handlers.prompt import PromptSession
from vllm_omni.diffusion.interaction.types import (
    MAX_PENDING_INTERACTION_EVENTS,
    InteractionChunkMetadata,
    InteractionPayload,
)
from vllm_omni.diffusion.worker.utils import StepRequestState


def validate_camera_payload(payload: Mapping[str, object]) -> None:
    if set(payload) != {"camera_trajectory"}:
        raise ValueError("Hy-Nomas camera updates require exactly camera_trajectory.")
    camera = payload["camera_trajectory"]
    if camera is not None and (not isinstance(camera, str) or not camera.strip()):
        raise ValueError("camera_trajectory must be a nonempty reference trajectory or null to disable camera control.")


@dataclass
class HyNomasInteractionSession(PromptSession):
    pending: tuple[str, float, object] | None = None
    coalesced: list[str] = field(default_factory=list)


class HyNomasInteractionHandler(InteractionHandler):
    """Apply a discrete update atomically at the next chunk boundary."""

    modality: ClassVar[str]
    field_name: ClassVar[str]

    def validate_payload(
        self, state: StepRequestState, *, event_id: str, payload: InteractionPayload, transition_chunks: int | None
    ) -> None:
        if not event_id:
            raise ValueError("event_id must be nonempty.")
        if transition_chunks not in (None, 0):
            raise ValueError("Hy-Nomas applies controls at one chunk boundary; transition_chunks must be 0.")
        if self.modality == "camera":
            validate_camera_payload(payload)
        elif set(payload) != {"prompt"} or not isinstance(payload["prompt"], str) or not payload["prompt"].strip():
            raise ValueError("Hy-Nomas prompt updates require one nonempty prompt string.")
        elif "|||" in payload["prompt"]:
            raise ValueError("A live prompt update must describe one segment.")
        session = state.interaction_sessions.get(self.modality)
        if isinstance(session, HyNomasInteractionSession):
            with session.lock:
                if len(session.coalesced) + int(session.pending is not None) >= MAX_PENDING_INTERACTION_EVENTS:
                    raise ValueError("Too many pending Hy-Nomas interactions; wait for a chunk boundary.")

    def enqueue(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        received_at: float,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        self.validate_payload(state, event_id=event_id, payload=payload, transition_chunks=transition_chunks)
        session = state.interaction_sessions.setdefault(self.modality, HyNomasInteractionSession())
        assert isinstance(session, HyNomasInteractionSession)
        with session.lock:
            if session.pending is not None:
                if session.pending[1] > received_at:
                    session.coalesced.append(event_id)
                    return
                session.coalesced.append(session.pending[0])
            session.pending = (event_id, received_at, payload[self.field_name])

    def apply_at_chunk_boundary(
        self, state: StepRequestState, *, boundary_at: float, **kwargs: Any
    ) -> InteractionChunkMetadata | None:
        del kwargs
        session = state.interaction_sessions[self.modality]
        assert isinstance(session, HyNomasInteractionSession)
        with session.lock:
            session.last_boundary_at = boundary_at
            if session.pending is None or session.pending[1] > boundary_at:
                return None
            event_id, _, value = session.pending
            session.pending = None
            coalesced, session.coalesced = session.coalesced, []
        state.extra[f"hy_nomas_{self.modality}_override"] = value
        if self.modality == "prompt":
            state.extra["hy_nomas_prompt_version"] = state.extra.get("hy_nomas_prompt_version", 0) + 1
        return InteractionChunkMetadata(
            started_event_ids=[event_id],
            active_event_ids=[event_id],
            completed_event_ids=[event_id],
            coalesced_event_ids=coalesced,
        )


class HyNomasPromptInteractionHandler(HyNomasInteractionHandler):
    modality = "prompt"
    field_name = "prompt"


class HyNomasCameraInteractionHandler(HyNomasInteractionHandler):
    modality = "camera"
    field_name = "camera_trajectory"

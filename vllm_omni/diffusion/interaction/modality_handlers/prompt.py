# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prompt-track interaction handler (midway prompt updates)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from typing_extensions import override

from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler
from vllm_omni.diffusion.interaction.types import (
    InteractionChunkMetadata,
    InteractionEvent,
    InteractionMode,
    InteractionPayload,
    InteractionSession,
)
from vllm_omni.diffusion.worker.utils import StepRequestState

DEFAULT_TRANSITION_CHUNKS = 3

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class QueuedPromptEvent(InteractionEvent):
    """Last-write-wins prompt update waiting for / active at a chunk boundary.

    ``source_prompt_embeds`` is the lerp start endpoint, staged when this event
    is activated (not at enqueue). Live blended embeds live on
    ``StepRequestState.prompt_embeds``.
    """

    mode: InteractionMode = "target"
    prompt: str
    target_prompt_embeds: torch.Tensor
    source_prompt_embeds: torch.Tensor | None = None

    def blended_prompt_embeds(self) -> torch.Tensor:
        assert self.source_prompt_embeds is not None
        alpha = self._current_alpha()
        if alpha >= 1.0:
            return self.target_prompt_embeds
        return (1.0 - alpha) * self.source_prompt_embeds + alpha * self.target_prompt_embeds

    def advance_transition(self) -> None:
        assert self.source_prompt_embeds is not None
        if self.transition_chunks <= 0:
            self.source_prompt_embeds = self.target_prompt_embeds
            return
        self.elapsed_transition_chunks += 1
        if self.elapsed_transition_chunks >= self.transition_chunks:
            self.source_prompt_embeds = self.target_prompt_embeds
            self.elapsed_transition_chunks = float(self.transition_chunks)

    def _current_alpha(self) -> float:
        if self.transition_chunks <= 0:
            return 1.0
        return min(1.0, self.elapsed_transition_chunks / self.transition_chunks)


@dataclass
class PromptSession(InteractionSession):
    """Request-local prompt-update session under ``state.interaction_sessions['prompt']``."""

    pending_event: QueuedPromptEvent | None = None
    active_event: QueuedPromptEvent | None = None
    version: int = 0


def prompt_update_versions(states: Sequence[StepRequestState]) -> tuple[int, ...]:
    """Return per-request prompt-update versions for batch cache comparison."""
    versions: list[int] = []
    for state in states:
        session = state.interaction_sessions.get("prompt")
        versions.append(session.version if isinstance(session, PromptSession) else 0)
    return tuple(versions)


class PromptInteractionHandler(InteractionHandler):
    """Encode-and-queue prompt updates; lerp embeds at chunk boundaries.

    Mirrors the former ``PromptUpdateMixin`` behavior. Encoding happens at
    enqueue time via an injected ``encode_prompt`` callable.
    """

    modality: ClassVar[str] = "prompt"

    def __init__(
        self,
        *,
        encode_prompt: Callable[..., tuple[torch.Tensor, ...]],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self._encode_prompt = encode_prompt
        self._device = device
        self._dtype = dtype

    @classmethod
    @override
    def from_pipeline(cls, pipeline: Any) -> PromptInteractionHandler:
        """Build a handler from a diffusion pipeline that exposes ``encode_prompt``."""
        return cls(
            encode_prompt=pipeline.encode_prompt,
            device=pipeline.device,
            dtype=pipeline.transformer.dtype,
        )

    @override
    def enqueue(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        received_at: float,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        """Prompt updates are last-write-win and unbuffered at chunk boundary."""
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty")
        if not event_id:
            raise ValueError("event_id must be non-empty")
        if state.prompt_embeds is None:
            raise ValueError(
                f"prompt_update is not allowed before initial generation has started (request_id={state.request_id!r})"
            )
        duration = DEFAULT_TRANSITION_CHUNKS if transition_chunks is None else transition_chunks
        if duration < 0:
            raise ValueError("transition_chunks must be >= 0")

        target_prompt_embeds, _ = self._encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=state.sampling.num_outputs_per_prompt,
            max_sequence_length=state.sampling.max_sequence_length,
            device=self._device,
            dtype=self._dtype,
        )
        session = state.interaction_sessions.setdefault("prompt", PromptSession())
        assert isinstance(session, PromptSession)
        with session.lock:
            # Chunk-level LWW: replace any prior pending event.
            session.pending_event = QueuedPromptEvent(
                event_id=event_id,
                received_at=received_at,
                transition_chunks=duration,
                prompt=prompt,
                target_prompt_embeds=target_prompt_embeds,
            )

    @override
    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        boundary_at: float,
        chunk_index: int | None = None,
        num_frames: int | None = None,
        fps: float | None = None,
    ) -> InteractionChunkMetadata | None:
        """Advance or start prompt interpolation before the next chunk."""
        # Prompt lerp is chunk-LWW; media timeline and caller chunk_index unused.
        del chunk_index, num_frames, fps
        session = state.interaction_sessions.get("prompt")
        assert isinstance(session, PromptSession)

        embeds_changed = False
        next_chunk_index = state.chunk_index
        started_event_ids: list[str] = []
        active_event_ids: list[str] = []
        completed_event_ids: list[str] = []

        with session.lock:
            # Prompt ignores frame scheduling but still records the boundary clock.
            session.last_boundary_at = boundary_at
            pending_event = session.pending_event
            session.pending_event = None
            active_event = session.active_event

            # If current transition is not complete, advance it.
            # After completion, leave active_event in place (so a later pending
            # update can abort/overwrite), but do not keep bumping the version.
            if active_event is not None:
                in_transition = (
                    active_event.transition_chunks > 0
                    and active_event.elapsed_transition_chunks < active_event.transition_chunks
                )
                if in_transition:
                    active_event.advance_transition()
                    state.prompt_embeds = active_event.blended_prompt_embeds()
                    embeds_changed = True
                    active_event_ids.append(active_event.event_id)
                    if active_event.elapsed_transition_chunks >= active_event.transition_chunks:
                        state.prompt_embeds = active_event.target_prompt_embeds
                        active_event.source_prompt_embeds = active_event.target_prompt_embeds
                        completed_event_ids.append(active_event.event_id)
                        logger.debug(
                            "prompt_update transition complete request_id=%s next_chunk_index=%d prompt=%.20s...",
                            state.request_id,
                            next_chunk_index,
                            active_event.prompt,
                        )

            # If a new prompt update is pending, start a new transition.
            if pending_event is not None:
                if state.prompt_embeds is None:
                    raise RuntimeError(
                        "internal error: trying to apply a pending prompt update but "
                        f"current prompt_embeds is None (request_id={state.request_id!r})"
                    )
                # Stage lerp-start embeds at activation (after any same-boundary advance).
                pending_event.source_prompt_embeds = state.prompt_embeds.detach().clone()
                pending_event.elapsed_transition_chunks = 0.0
                session.active_event = pending_event
                active_event = pending_event
                event_id = pending_event.event_id
                duration = pending_event.transition_chunks
                prompt = pending_event.prompt
                target = pending_event.target_prompt_embeds
                started_event_ids.append(event_id)
                active_event_ids.append(event_id)
                if duration <= 0:
                    # A hard/immediate transition
                    state.prompt_embeds = target
                    pending_event.source_prompt_embeds = target
                    completed_event_ids.append(event_id)
                    logger.debug(
                        "prompt_update sharp transition request_id=%s next_chunk_index=%d prompt=%.20s...",
                        state.request_id,
                        next_chunk_index,
                        prompt,
                    )
                else:
                    # A transition that really takes some time
                    active_event.advance_transition()
                    state.prompt_embeds = active_event.blended_prompt_embeds()
                    if active_event.elapsed_transition_chunks >= active_event.transition_chunks:
                        state.prompt_embeds = active_event.target_prompt_embeds
                        active_event.source_prompt_embeds = active_event.target_prompt_embeds
                        completed_event_ids.append(event_id)
                    logger.debug(
                        "prompt_update transition start request_id=%s next_chunk_index=%d prompt=%.20s...",
                        state.request_id,
                        next_chunk_index,
                        prompt,
                    )
                embeds_changed = True

            # Indicate that prompt embeddings have changed---clear input batch cache.
            if embeds_changed:
                session.version += 1

        return InteractionChunkMetadata(
            started_event_ids=started_event_ids,
            active_event_ids=active_event_ids,
            completed_event_ids=completed_event_ids,
        )

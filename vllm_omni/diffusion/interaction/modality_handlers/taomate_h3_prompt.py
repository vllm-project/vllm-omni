# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prompt interaction handler for TaoMate-H3 streaming sessions.

TaoMate-H3 encodes one prompt per five-second request and applies it to the
whole request (the audio teacher and all four phases), so a prompt update is a
hard switch that takes effect at the next request boundary. Prompt lengths
differ between updates, which rules out the generic handler's embedding
interpolation; this subclass forces an immediate transition.

The step runner pads every request's prompt embeddings to the length it
recorded for the request (``txt_seq_lens``) and writes the padded tensor back
into the request state. A new prompt with another token count would therefore
reach the next request zero-padded to the old length (or truncated to it), and
with the old prompt's token tags. The handler keeps the recorded length and
the tags in step with the prompt it applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from typing_extensions import override

from vllm_omni.diffusion.interaction.modality_handlers.prompt import (
    PromptInteractionHandler,
    PromptSession,
    QueuedPromptEvent,
)
from vllm_omni.diffusion.interaction.types import InteractionChunkMetadata, InteractionPayload
from vllm_omni.diffusion.worker.utils import StepRequestState


@dataclass(kw_only=True)
class TaoMateQueuedPromptEvent(QueuedPromptEvent):
    """A queued prompt with the text encoder's per-token tags (0/1) of the new prompt."""

    target_text_tags: torch.Tensor | None = None


def sync_prompt_length(state: StepRequestState, *, text_tags: torch.Tensor | None) -> int:
    """Record the current prompt embeddings' row count and tags on the request state.

    Returns the number of text rows. The runner then pads to this length (a
    no-op) instead of the previous prompt's length.
    """
    embeds = state.prompt_embeds
    if embeds is None:
        raise RuntimeError("sync_prompt_length needs prompt embeddings on the request state")
    rows = int(embeds.shape[-2])
    state.prompt_embeds_mask = None
    state.txt_seq_lens = [rows]
    if text_tags is not None and text_tags.ndim == 1 and int(text_tags.shape[0]) == rows:
        state.extra["text_tags"] = text_tags.detach()
    else:
        # The pipeline treats every row as a plain text token when no tags fit.
        state.extra.pop("text_tags", None)
    return rows


class TaoMateH3PromptInteractionHandler(PromptInteractionHandler):
    modality: ClassVar[str] = "prompt"

    @classmethod
    @override
    def from_pipeline(cls, pipeline: Any) -> TaoMateH3PromptInteractionHandler:
        # The H3 DiT keeps mixed parameter dtypes and exposes no ``dtype``; the
        # text encoder returns BF16 hidden states regardless.
        return cls(encode_prompt=pipeline.encode_prompt, device=pipeline.device, dtype=torch.bfloat16)

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
        """Encode the new prompt now (with its tags); it applies at the next request boundary."""
        del transition_chunks
        self.validate_payload(state, event_id=event_id, payload=payload, transition_chunks=0)
        prompt = payload.get("prompt")
        assert isinstance(prompt, str)
        encoded = self._encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=state.sampling.num_outputs_per_prompt,
            max_sequence_length=state.sampling.max_sequence_length,
            device=self._device,
            dtype=self._dtype,
        )
        target = encoded[0]
        tags = encoded[1] if len(encoded) > 1 and isinstance(encoded[1], torch.Tensor) else None
        session = state.interaction_sessions.setdefault("prompt", PromptSession())
        assert isinstance(session, PromptSession)
        with session.lock:
            # Chunk-level last-write-wins: replace any prior pending event.
            session.pending_event = TaoMateQueuedPromptEvent(
                event_id=event_id,
                received_at=received_at,
                transition_chunks=0,
                prompt=prompt,
                target_prompt_embeds=target,
                target_text_tags=tags,
            )

    @override
    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        boundary_at: float,
        **kwargs: Any,
    ) -> InteractionChunkMetadata | None:
        metadata = super().apply_at_chunk_boundary(state, boundary_at=boundary_at, **kwargs)
        session = state.interaction_sessions.get("prompt")
        active = getattr(session, "active_event", None)
        embeds = state.prompt_embeds
        if (
            isinstance(active, TaoMateQueuedPromptEvent)
            and embeds is not None
            and embeds is active.target_prompt_embeds
        ):
            # The sharp transition just installed this event's embeddings.
            sync_prompt_length(state, text_tags=active.target_text_tags)
        return metadata


__all__ = ["TaoMateH3PromptInteractionHandler", "TaoMateQueuedPromptEvent", "sync_prompt_length"]

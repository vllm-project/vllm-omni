# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Base interaction handler interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Self

from vllm_omni.diffusion.interaction.types import (
    InteractionChunkMetadata,
    InteractionPayload,
)
from vllm_omni.diffusion.worker.utils import StepRequestState


class InteractionHandler(ABC):
    """Strategy object for one interaction modality.

    Handler instances are pipeline/runner-owned and request-agnostic. Per-request
    session state lives on ``StepRequestState.interaction_sessions``.
    Each concrete modality handler decides how to handle timing from ``received_at``.
    """

    modality: ClassVar[str]
    # When True, chunk-boundary apply needs ``ChunkMediaSpec`` (num_frames/fps)
    # Those information are useful when interaction handler needs interpolation/integration on a frame-by-frame basis
    needs_chunk_media: ClassVar[bool] = False

    @classmethod
    def from_pipeline(cls, pipeline: object) -> Self:
        """Bind a request-agnostic handler to ``pipeline`` when construction needs it.

        Default construction ignores the pipeline. Prompt handlers override this to
        capture ``encode_prompt`` / device / dtype.
        """
        del pipeline
        return cls()

    @abstractmethod
    def enqueue(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        received_at: float,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        """Validate and queue this track on request-local state."""

    @abstractmethod
    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        boundary_at: float,
        chunk_index: int | None = None,  # defaults to state.chunk_index when omitted
        num_frames: int | None = None,  # only set when at least one interaction handler needs it
        fps: float | None = None,  # only set when at least one interaction handler needs it
    ) -> InteractionChunkMetadata | None:
        """Advance request-local state and materialize this chunk's effects."""

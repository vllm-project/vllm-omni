# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Interaction-handling mixin for diffusion pipelines. Process interactions at chunk boundaries."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from vllm_omni.diffusion.models.interface import SupportsInteractionApply
from vllm_omni.diffusion.worker.utils import StepRequestState

if TYPE_CHECKING:
    from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator


class InteractionMixin:
    """Unified chunk-boundary interaction hook.

    Pipelines inherit this mixin and implement ``peek_chunk_media`` to offer playback information about the next chunk.
    Diffusion runner calls ``apply_interaction_at_chunk_boundary`` without knowing which modality is being applied.

    Per-modality session state lives on ``StepRequestState.interaction_sessions``;
    handler strategy objects live on the pipeline/runner ``InteractionCoordinator``.
    """

    _interaction_coordinator: InteractionCoordinator | None = None

    def apply_interaction_at_chunk_boundary(self, state: StepRequestState) -> None:
        """Advance all active interaction tracks before the next chunk."""
        if self._interaction_coordinator is None:
            raise RuntimeError(
                "interaction coordinator is not initialized; "
                "DiffusionModelRunner.load_model must wire InteractionCoordinator "
                "onto the pipeline before chunked generation"
            )

        num_frames: int | None = None
        fps: float | None = None
        if self._interaction_coordinator.needs_chunk_media:
            media = cast(SupportsInteractionApply, self).peek_chunk_media(state)
            num_frames = media.num_frames
            fps = media.fps

        merged = self._interaction_coordinator.apply_at_chunk_boundary(
            state,
            boundary_at=time.monotonic(),
            num_frames=num_frames,
            fps=fps,
        )
        state.interaction_chunk_metadata = merged

    def prepare_next_chunk(self, state: StepRequestState) -> None:
        """Set up pipeline state for the next chunk after interaction apply.

        Default no-op. Pipelines may override this if needed.
        """
        pass

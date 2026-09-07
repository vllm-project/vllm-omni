# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Interaction coordinator to be co-owned by a pipeline and a runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_omni.diffusion.interaction.registry import STRUCTURED_HANDLER_REGISTRY
from vllm_omni.diffusion.interaction.types import (
    InteractionChunkMetadata,
    InteractionPayload,
    merge_interaction_metadata,
)
from vllm_omni.diffusion.models.interface import supports_interaction_apply
from vllm_omni.diffusion.worker.utils import StepRequestState

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler


class InteractionCoordinator:
    """Handles mid-way interaction inputs for one loaded pipeline.

    Request-local session state lives on ``StepRequestState.interaction_sessions``;
    this object is request-agnostic.
    """

    def __init__(
        self,
        handlers: dict[str, InteractionHandler],
        *,
        model_class_name: str | None = None,
    ) -> None:
        self._handlers = handlers
        self.model_class_name = model_class_name

    @classmethod
    def build(cls, pipeline: object, od_config: OmniDiffusionConfig) -> InteractionCoordinator:
        model_class_name = od_config.model_class_name
        handlers: dict[str, InteractionHandler] = {}

        if supports_interaction_apply(pipeline):
            for modality, handler_cls in STRUCTURED_HANDLER_REGISTRY.get(model_class_name or "", {}).items():
                handlers[modality] = handler_cls.from_pipeline(pipeline)

        return cls(handlers, model_class_name=model_class_name)

    def has_modality(self, modality: str) -> bool:
        return modality in self._handlers

    @property
    def needs_chunk_media(self) -> bool:
        """Whether any registered handler needs chunk ``num_frames``/``fps``."""
        return any(handler.needs_chunk_media for handler in self._handlers.values())

    def get_handler(self, modality: str) -> InteractionHandler:
        handler = self._handlers.get(modality)
        if handler is None:
            raise ValueError(
                f"interaction modality {modality!r} is not supported by pipeline {self.model_class_name!r}"
            )
        return handler

    def enqueue(
        self,
        state: StepRequestState,
        *,
        modality: str,
        event_id: str,
        received_at: float,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        """Fan out enqueue to handlers in stable order (prompt first)."""
        handler = self.get_handler(modality)
        handler.enqueue(
            state,
            event_id=event_id,
            received_at=received_at,
            payload=payload,
            transition_chunks=transition_chunks,
        )

    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        boundary_at: float,
        chunk_index: int | None = None,
        num_frames: int | None = None,
        fps: float | None = None,
    ) -> InteractionChunkMetadata:
        """Fan out chunk-boundary apply to handlers in stable order (prompt first)."""
        if chunk_index is None:
            chunk_index = state.chunk_index
        metas: list[InteractionChunkMetadata] = []
        for handler in self._handlers_in_apply_order():
            if handler.modality not in state.interaction_sessions:
                # This function is always called at chunk boundary without knowing if an interaction is really enqueued.
                # `state.interaction_sessions[this modality]` is lazy-populated after the first interaction is enqueued.
                # So, check here and skip a modality handler if this modality's session is not yet created.
                continue
            meta = handler.apply_at_chunk_boundary(
                state,
                boundary_at=boundary_at,
                chunk_index=chunk_index,
                num_frames=num_frames,
                fps=fps,
            )
            if meta is not None:
                metas.append(meta)
        return merge_interaction_metadata(metas)

    def _handlers_in_apply_order(self) -> list[InteractionHandler]:
        ordered: list[InteractionHandler] = []
        if "prompt" in self._handlers:
            ordered.append(self._handlers["prompt"])
        for modality, handler in self._handlers.items():
            if modality == "prompt":
                continue
            ordered.append(handler)
        return ordered

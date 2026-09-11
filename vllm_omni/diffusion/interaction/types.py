# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared types for midway interaction handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal

# Wire payload for one modality track. Handlers validate keys they need.
InteractionPayload = Mapping[str, object]

# Shared command semantics across modalities that support them.
# Prompt specialization narrows this to ``"target"`` only.
InteractionMode = Literal["target", "velocity"]


@dataclass(frozen=True)
class ChunkMediaSpec:
    """Media extent of one generation chunk, for interaction timeline mapping."""

    num_frames: int
    fps: float

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError(f"ChunkMediaSpec.num_frames must be > 0, got {self.num_frames}")
        if self.fps <= 0:
            raise ValueError(f"ChunkMediaSpec.fps must be > 0, got {self.fps}")

    @property
    def duration_s(self) -> float:
        return float(self.num_frames) / float(self.fps)


@dataclass(kw_only=True)
class InteractionEvent:
    """Shared envelope for one queued or in-flight interaction command. **Intended to be subclassed**.

    Example:
    - ``QueuedPromptEvent`` in ``vllm_omni.diffusion.interaction.modality_handlers.prompt``
    - ``QueuedCameraEvent`` in ``vllm_omni.diffusion.interaction.modality_handlers.camera``
    """

    event_id: str
    received_at: float
    mode: InteractionMode
    transition_chunks: int
    elapsed_transition_chunks: float = 0.0


@dataclass
class InteractionSession:
    """Request-local session base for one interaction modality. **Intended to be subclassed**.

    Concrete modalities subclass this and store their own pending/active state.
    ``last_boundary_at`` is available for frame-scheduled modalities; chunk-LWW
    modalities may ignore it.

    Example:
    - ``PromptSession`` in ``vllm_omni.diffusion.interaction.modality_handlers.prompt``
    - ``CameraSession`` in ``vllm_omni.diffusion.interaction.modality_handlers.camera``
    """

    lock: Lock = field(default_factory=Lock, repr=False)
    last_boundary_at: float | None = None


def resolve_event_frame_offset(
    *,
    received_at: float,
    previous_boundary_at: float | None,
    num_frames: int,
    fps: float,
) -> int:
    """Map an event arrival time onto ``[0, num_frames - 1]`` for this chunk.

    * Uses ``floor((received_at - previous_boundary_at) * fps)``.
    * Events without a prior boundary (or with non-positive fps) map to frame 0.
    * Offsets past the represented media window are clamped to the final frame.
    """
    import math

    num_frames = max(int(num_frames), 1)
    if previous_boundary_at is None or fps <= 0:
        return 0
    offset = math.floor((received_at - previous_boundary_at) * float(fps))
    if offset < 0:
        return 0
    if offset >= num_frames:
        return num_frames - 1
    return int(offset)


@dataclass
class InteractionChunkMetadata:
    """Per-modality (or merged) event-id bookkeeping for one chunk boundary."""

    started_event_ids: list[str] = field(default_factory=list)
    active_event_ids: list[str] = field(default_factory=list)
    completed_event_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "started_event_ids": list(self.started_event_ids),
            "active_event_ids": list(self.active_event_ids),
            "completed_event_ids": list(self.completed_event_ids),
        }


def merge_interaction_metadata(
    metas: list[InteractionChunkMetadata],
) -> InteractionChunkMetadata:
    """Merge per-handler metadata while preserving order and de-duplicating ids."""
    started: list[str] = []
    active: list[str] = []
    completed: list[str] = []
    seen_started: set[str] = set()
    seen_active: set[str] = set()
    seen_completed: set[str] = set()
    for meta in metas:
        for event_id in meta.started_event_ids:
            if event_id not in seen_started:
                seen_started.add(event_id)
                started.append(event_id)
        for event_id in meta.active_event_ids:
            if event_id not in seen_active:
                seen_active.add(event_id)
                active.append(event_id)
        for event_id in meta.completed_event_ids:
            if event_id not in seen_completed:
                seen_completed.add(event_id)
                completed.append(event_id)
    return InteractionChunkMetadata(
        started_event_ids=started,
        active_event_ids=active,
        completed_event_ids=completed,
    )

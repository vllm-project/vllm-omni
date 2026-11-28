# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed per-chunk GEN layout, token metadata and caption documents.

Each chunk packs ``[RGB control | RGB target | LiDAR control | LiDAR target]``;
camera items are camera-major (all frames of view 0, then view 1, ...) and
every frame is a row-major patch grid. The metadata built here is what the
causal visibility predicate consumes, both for the current chunk (queries and
keys) and for chunks retained in the KV ring (keys only). Roles follow the
reference replay semantics exactly: controls, observed target conditions, the
noisy targets of the current chunk, and clean (committed) targets.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import torch

from .geometry import Chunk, SensorGrid

ROLE_PADDING = -1
ROLE_UND = 0
ROLE_CONTROL = 1
ROLE_TARGET_CONDITION = 2
ROLE_CURRENT_TARGET = 3
ROLE_CLEAN_TARGET = 4

#: Camera tokens read their own camera's captions; LiDAR tokens read every caption.
CAPTION_SCOPE_SAME_VIEW = 1
CAPTION_SCOPE_ALL = 2

PassKind = Literal["noisy", "clean"]
ItemKind = Literal["rgb_control", "rgb_target", "lidar_control", "lidar_target"]
ITEM_ORDER: tuple[ItemKind, ...] = ("rgb_control", "rgb_target", "lidar_control", "lidar_target")


@dataclass(frozen=True)
class StreamItem:
    """One packed sensor item of a chunk.

    ``view_ids`` are the request's view ids in packing order (LiDAR uses the
    single id one past the cameras); ``frames`` are absolute latent/sweep
    indexes shared by every view of the item. ``condition_frames`` marks the
    frames of a *target* item whose latents are observed (role
    ``TARGET_CONDITION``) rather than generated.
    """

    kind: ItemKind
    view_ids: tuple[int, ...]
    frames: tuple[int, ...]
    grid: SensorGrid
    seconds_per_frame: float
    condition_frames: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if self.kind not in ITEM_ORDER:
            raise ValueError(f"Unknown stream item kind {self.kind!r}.")
        if not self.view_ids or len(set(self.view_ids)) != len(self.view_ids):
            raise ValueError(f"Stream item view ids must be non-empty and unique, got {self.view_ids}.")
        if not self.frames or any(frame < 0 for frame in self.frames) or list(self.frames) != sorted(self.frames):
            raise ValueError(f"Stream item frames must be non-empty, non-negative and increasing, got {self.frames}.")
        if self.seconds_per_frame <= 0:
            raise ValueError(f"seconds_per_frame must be positive, got {self.seconds_per_frame}.")
        if self.is_control and self.condition_frames:
            raise ValueError("Control items are fully observed; condition_frames applies to targets only.")
        if not self.condition_frames <= set(self.frames):
            raise ValueError(
                f"condition_frames {sorted(self.condition_frames)} must be frames of the item {self.frames}."
            )

    @property
    def is_control(self) -> bool:
        return self.kind.endswith("_control")

    @property
    def is_lidar(self) -> bool:
        return self.kind.startswith("lidar_")

    @property
    def tokens_per_frame(self) -> int:
        return self.grid.tokens_per_frame

    @property
    def num_tokens(self) -> int:
        return len(self.view_ids) * len(self.frames) * self.tokens_per_frame

    @property
    def caption_scope(self) -> int:
        return CAPTION_SCOPE_ALL if self.is_lidar else CAPTION_SCOPE_SAME_VIEW


def build_chunk_items(
    chunk: Chunk,
    *,
    num_views: int,
    rgb_grid: SensorGrid,
    rgb_seconds_per_frame: float,
    lidar_grid: SensorGrid | None = None,
    lidar_seconds_per_frame: float | None = None,
    with_controls: bool = True,
    rgb_condition_frames: Iterable[int] = (),
    lidar_condition_sweeps: Iterable[int] = (),
) -> tuple[StreamItem, ...]:
    """Reference item order for one chunk: RGB control, RGB target, LiDAR control, LiDAR target."""
    if num_views < 1:
        raise ValueError(f"num_views must be positive, got {num_views}.")
    cameras = tuple(range(num_views))
    rgb_condition = frozenset(rgb_condition_frames) & set(chunk.rgb_frames)
    items: list[StreamItem] = []
    if with_controls:
        items.append(StreamItem("rgb_control", cameras, chunk.rgb_frames, rgb_grid, rgb_seconds_per_frame))
    items.append(StreamItem("rgb_target", cameras, chunk.rgb_frames, rgb_grid, rgb_seconds_per_frame, rgb_condition))
    if chunk.lidar_sweeps:
        if lidar_grid is None or lidar_seconds_per_frame is None:
            raise ValueError("Joint chunks require lidar_grid and lidar_seconds_per_frame.")
        lidar_view = (num_views,)
        lidar_condition = frozenset(lidar_condition_sweeps) & set(chunk.lidar_sweeps)
        if with_controls:
            items.append(
                StreamItem("lidar_control", lidar_view, chunk.lidar_sweeps, lidar_grid, lidar_seconds_per_frame)
            )
        items.append(
            StreamItem(
                "lidar_target", lidar_view, chunk.lidar_sweeps, lidar_grid, lidar_seconds_per_frame, lidar_condition
            )
        )
    return tuple(items)


@dataclass
class TokenMetadata:
    """Per-token predicate fields of one key or query stream (CPU tensors, ``[N]``)."""

    view_id: torch.Tensor
    frame_id: torch.Tensor
    timestamp: torch.Tensor
    role: torch.Tensor
    step: torch.Tensor
    caption_scope: torch.Tensor

    def __post_init__(self) -> None:
        length = self.view_id.numel()
        for name in ("frame_id", "timestamp", "role", "step", "caption_scope"):
            if getattr(self, name).numel() != length:
                raise ValueError(f"TokenMetadata.{name} has {getattr(self, name).numel()} entries, expected {length}.")

    @property
    def num_tokens(self) -> int:
        return int(self.view_id.numel())

    @property
    def is_control(self) -> torch.Tensor:
        return self.role == ROLE_CONTROL

    @property
    def is_noisy(self) -> torch.Tensor:
        return self.role == ROLE_CURRENT_TARGET

    def index_select(self, index: torch.Tensor) -> TokenMetadata:
        return TokenMetadata(
            view_id=self.view_id[index],
            frame_id=self.frame_id[index],
            timestamp=self.timestamp[index],
            role=self.role[index],
            step=self.step[index],
            caption_scope=self.caption_scope[index],
        )

    @staticmethod
    def concat(parts: Sequence[TokenMetadata]) -> TokenMetadata:
        if not parts:
            raise ValueError("TokenMetadata.concat requires at least one part.")
        return TokenMetadata(
            view_id=torch.cat([part.view_id for part in parts]),
            frame_id=torch.cat([part.frame_id for part in parts]),
            timestamp=torch.cat([part.timestamp for part in parts]),
            role=torch.cat([part.role for part in parts]),
            step=torch.cat([part.step for part in parts]),
            caption_scope=torch.cat([part.caption_scope for part in parts]),
        )

    @staticmethod
    def empty() -> TokenMetadata:
        return TokenMetadata(
            view_id=torch.empty(0, dtype=torch.int64),
            frame_id=torch.empty(0, dtype=torch.int64),
            timestamp=torch.empty(0, dtype=torch.float32),
            role=torch.empty(0, dtype=torch.int64),
            step=torch.empty(0, dtype=torch.int64),
            caption_scope=torch.empty(0, dtype=torch.int64),
        )


def item_token_metadata(item: StreamItem, *, step: int, pass_kind: PassKind) -> TokenMetadata:
    """Camera-major, frame-inner, spatial-innermost token metadata of one item.

    Timestamps are absolute (``frame * seconds_per_frame``) so the cross-view
    window applies identically to current and retained keys, and every token of
    a chunk carries the chunk step: the shared clock, not a per-sensor division,
    decides causality (this matters for LiDAR, whose sweeps per chunk differ
    from the RGB latents per chunk).
    """
    if pass_kind not in ("noisy", "clean"):
        raise ValueError(f"pass_kind must be 'noisy' or 'clean', got {pass_kind!r}.")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}.")
    spatial = item.tokens_per_frame
    frames = torch.tensor(item.frames, dtype=torch.int64)
    if item.is_control:
        frame_roles = torch.full_like(frames, ROLE_CONTROL)
    else:
        generated_role = ROLE_CURRENT_TARGET if pass_kind == "noisy" else ROLE_CLEAN_TARGET
        is_condition = torch.tensor([frame in item.condition_frames for frame in item.frames], dtype=torch.bool)
        frame_roles = torch.where(is_condition, torch.full_like(frames, ROLE_TARGET_CONDITION), generated_role)
    per_view_frames = frames.repeat_interleave(spatial)
    per_view_roles = frame_roles.repeat_interleave(spatial)
    num_views = len(item.view_ids)
    frame_id = per_view_frames.repeat(num_views)
    role = per_view_roles.repeat(num_views)
    view_id = torch.tensor(item.view_ids, dtype=torch.int64).repeat_interleave(len(item.frames) * spatial)
    return TokenMetadata(
        view_id=view_id,
        frame_id=frame_id,
        timestamp=frame_id.to(torch.float32) * item.seconds_per_frame,
        role=role,
        step=torch.full_like(frame_id, step),
        caption_scope=torch.full_like(frame_id, item.caption_scope),
    )


def chunk_token_metadata(
    items: Sequence[StreamItem], *, step: int, pass_kind: PassKind
) -> tuple[TokenMetadata, tuple[int, ...]]:
    """Concatenate item metadata in packing order; also return each item's token offset."""
    if not items:
        raise ValueError("A chunk layout needs at least one stream item.")
    offsets: list[int] = []
    parts: list[TokenMetadata] = []
    cursor = 0
    for item in items:
        offsets.append(cursor)
        parts.append(item_token_metadata(item, step=step, pass_kind=pass_kind))
        cursor += item.num_tokens
    return TokenMetadata.concat(parts), tuple(offsets)


def committed_metadata(metadata: TokenMetadata) -> TokenMetadata:
    """What a KV-ring slot stores after a chunk's clean commit.

    The clean refresh relabels generated targets as ``CLEAN_TARGET``; observed
    conditions and controls keep their roles. Retained slots are never queries,
    so the caption scope is irrelevant but kept for uniform concatenation.
    """
    role = torch.where(
        metadata.role == ROLE_CURRENT_TARGET, torch.full_like(metadata.role, ROLE_CLEAN_TARGET), metadata.role
    )
    return replace(metadata, role=role)


@dataclass(frozen=True)
class CaptionDocument:
    """One causal UND text document: a view's sink prefix or one caption segment.

    ``start_seconds``/``end_seconds`` bound the query timestamps that may read
    the document (half-open); sink documents are visible at every time.
    """

    view_id: int
    num_tokens: int
    is_sink: bool = False
    start_seconds: float = float("-inf")
    end_seconds: float = float("inf")

    def __post_init__(self) -> None:
        if self.num_tokens < 1:
            raise ValueError(f"Caption documents need at least one token, got {self.num_tokens}.")
        if self.is_sink and (self.start_seconds != float("-inf") or self.end_seconds != float("inf")):
            raise ValueError("Sink documents are visible at every time; do not bound them.")
        if self.start_seconds >= self.end_seconds:
            raise ValueError(f"Caption bounds must be increasing, got [{self.start_seconds}, {self.end_seconds}).")


@dataclass
class CaptionMetadata:
    """Per-token UND fields: owning view, sink flag, time bounds and document id (CPU ``[U]``)."""

    view_id: torch.Tensor
    is_sink: torch.Tensor
    start_seconds: torch.Tensor
    end_seconds: torch.Tensor
    doc_id: torch.Tensor
    doc_offsets: tuple[int, ...]

    @property
    def num_tokens(self) -> int:
        return int(self.view_id.numel())


def caption_metadata(documents: Sequence[CaptionDocument]) -> CaptionMetadata:
    """Expand documents in packing order ``[sink_v0 | caption_v0 | sink_v1 | ...]``."""
    if not documents:
        raise ValueError("Caption metadata requires at least one document.")
    lengths = torch.tensor([doc.num_tokens for doc in documents], dtype=torch.int64)
    offsets = [0]
    for length in lengths.tolist():
        offsets.append(offsets[-1] + length)
    return CaptionMetadata(
        view_id=torch.tensor([doc.view_id for doc in documents], dtype=torch.int64).repeat_interleave(lengths),
        is_sink=torch.tensor([doc.is_sink for doc in documents], dtype=torch.bool).repeat_interleave(lengths),
        start_seconds=torch.tensor([doc.start_seconds for doc in documents], dtype=torch.float32).repeat_interleave(
            lengths
        ),
        end_seconds=torch.tensor([doc.end_seconds for doc in documents], dtype=torch.float32).repeat_interleave(
            lengths
        ),
        doc_id=torch.arange(len(documents), dtype=torch.int64).repeat_interleave(lengths),
        doc_offsets=tuple(offsets[:-1]),
    )


@dataclass(frozen=True)
class RingSlot:
    """A committed chunk retained in the KV ring, addressed in virtual slot coordinates.

    ``base`` is the first virtual token index of the slot (``slot * frame_unit``);
    the slot holds ``metadata.num_tokens`` real tokens followed by padding that
    never enters attention.
    """

    slot: int
    base: int
    metadata: TokenMetadata

    def __post_init__(self) -> None:
        if self.slot < 0 or self.base < 0:
            raise ValueError("Ring slots need non-negative slot and base offsets.")

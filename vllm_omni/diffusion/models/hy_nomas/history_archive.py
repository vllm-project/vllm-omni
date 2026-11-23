# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, session-local CPU copies of committed Hy-Nomas history.

Records contain post-RoPE KV and original visibility masks, not a resumable
generation checkpoint. GPU allocation handles must never escape into an archive.
"""

from dataclasses import dataclass

import torch

from vllm_omni.errors import OmniClientError


@dataclass(frozen=True)
class HyNomasArchivedChunk:
    frame_start: int
    frame_count: int
    tokens_per_frame: int
    segment: int
    text_tokens: int
    action_start: int
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    mask: torch.Tensor

    def frame_rows(self, position: int) -> slice:
        """Map an absolute latent position to video rows, excluding chunk text."""
        if not self.frame_start <= position < self.frame_start + self.frame_count:
            raise IndexError("Latent frame is outside this archived chunk.")
        start = (position - self.frame_start) * self.tokens_per_frame
        return slice(start, start + self.tokens_per_frame)


class HyNomasHistoryArchive:
    """Append-only archive with explicit payload and metadata bounds per rank.

    Stage CPU copies before the rank agreement and KV publication. ``publish`` only
    swaps prepared references after KV commit; rollback discards speculative data.
    The payload quota includes both committed and pending CPU tensors. The frame
    limit bounds Python metadata separately; it is not a host RSS measurement.
    """

    def __init__(self, *, max_bytes: int, max_frames: int) -> None:
        if any(type(value) is not int or value <= 0 for value in (max_bytes, max_frames)):
            raise ValueError("Hy-Nomas archive limits must be positive integers.")
        self.max_bytes = max_bytes
        self.max_frames = max_frames
        self.records: tuple[HyNomasArchivedChunk, ...] = ()
        self.payload_bytes = 0
        self.frames = 0
        self._pending: tuple[HyNomasArchivedChunk, ...] | None = None
        self._pending_bytes = 0

    def stage(
        self,
        layers: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        mask: torch.Tensor,
        *,
        frame_start: int,
        frame_count: int,
        tokens_per_frame: int,
        segment: int,
        text_tokens: int,
        action_start: int,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("Hy-Nomas archive already has a pending chunk.")
        if frame_start != self.frames or frame_count <= 0 or tokens_per_frame <= 0 or text_tokens < 0:
            raise ValueError("Hy-Nomas archive requires contiguous absolute frames and valid token geometry.")
        tokens = frame_count * tokens_per_frame + text_tokens
        if not layers or mask.shape != (1, tokens) or mask.dtype != torch.bool:
            raise ValueError("Hy-Nomas archive mask must cover exactly the new video/text rows.")
        tensors = [tensor for pair in layers for tensor in pair]
        shape = tensors[0].shape
        if (
            len(shape) != 3
            or shape[0] != tokens
            or any(
                tensor.shape != shape or tensor.dtype != tensors[0].dtype or not tensor.is_floating_point()
                for tensor in tensors
            )
        ):
            raise ValueError("Hy-Nomas archive requires matching floating-point KV rows for every layer.")
        size = sum(tensor.numel() * tensor.element_size() for tensor in (*tensors, mask))
        if self.frames + frame_count > self.max_frames or self.payload_bytes + size > self.max_bytes:
            raise OmniClientError(
                "Hy-Nomas CPU history archive quota exceeded; close/reset or increase archive limits."
            )

        # Copy once per layer/chunk. Views into shared staging would be overwritten
        # by the next forward; blocking D2H makes rank agreement cover copy errors.
        copied = tuple(
            (
                key.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format),
                value.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format),
            )
            for key, value in layers
        )
        record = HyNomasArchivedChunk(
            frame_start,
            frame_count,
            tokens_per_frame,
            segment,
            text_tokens,
            action_start,
            copied,
            mask.detach().to(device="cpu", copy=True),
        )
        self._pending = (*self.records, record)
        self._pending_bytes = size

    def publish(self) -> None:
        if self._pending is None:
            raise RuntimeError("Hy-Nomas archive has no pending chunk.")
        self.records = self._pending
        self.payload_bytes += self._pending_bytes
        self.frames += self.records[-1].frame_count
        self.rollback()

    def rollback(self) -> None:
        self._pending = None
        self._pending_bytes = 0

    def clear(self) -> None:
        self.rollback()
        self.records = ()
        self.payload_bytes = self.frames = 0

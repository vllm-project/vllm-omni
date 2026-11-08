# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Window softmax of VDN-H3 (Video DeltaNet MiniMax-H3) hybrid attention.

A VDN checkpoint replaces each DiT block's dense self-attention with the sum of
two branches: an exact softmax over a chunk-aligned frame window, computed
here, and a linear-attention branch over everything outside that window, which
the model computes itself (``models/minimax_h3/vdnh3.py``).

The window in a packed ``[text | conditions | audio | video | pad]`` document:

* video frame ``t`` belongs to chunk ``t // chunk`` and sees whole chunks
  ``[c - radius, c + radius]`` (``chunk == 0``: frames ``t +- radius``);
* every non-video row is global: it sees, and is seen by, every row;
* ``anchor_frames`` makes the first and last frame dense as query rows
  (``rows``), as key columns (``columns``) or both;
* padding rows sit outside every window and come back as zeros.

The mask is executed as a union of dense attention calls, one per group of
query rows that share a key set, so each call is an ordinary FlashAttention
forward over gathered keys. Without window metadata (the token refiner, a dense
checkpoint) the backend is exactly ``FLASH_ATTN``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm_omni.diffusion.attention.capabilities import ExecutionContext, ExecutionPathResult
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

VDN_ANCHOR_FRAME_MODES = ("none", "columns", "rows", "both")


@dataclass(frozen=True)
class VDNLayout:
    """One packed document's layout and the frame window over it.

    Both VDN branches read the same instance, so they split the keys exactly:
    the softmax branch covers the window, the linear branch everything outside
    it. Plain ints, so it hashes (the attention plan is cached on it) and never
    forces a device sync. Text rows are ``[0, text_len)``; the target video is
    ``num_frames`` t-major frames of ``frame_height x frame_width`` tokens
    starting at ``video_start``; rows at and past ``used`` are padding.
    ``chunk``, ``radius`` and ``anchor_frames`` define the window.
    """

    used: int
    text_len: int
    video_start: int
    num_frames: int
    frame_height: int
    frame_width: int
    chunk: int
    radius: int
    anchor_frames: str = "both"

    def __post_init__(self) -> None:
        if self.anchor_frames not in VDN_ANCHOR_FRAME_MODES:
            raise ValueError(f"anchor_frames must be one of {VDN_ANCHOR_FRAME_MODES}, got {self.anchor_frames!r}")
        if self.chunk < 0 or self.radius < 0:
            raise ValueError("VDN window chunk and radius must be >= 0")
        if not 0 <= self.text_len <= self.video_start <= self.video_end <= self.used:
            raise ValueError(f"invalid VDN layout: {self}")

    @property
    def tokens_per_frame(self) -> int:
        return self.frame_height * self.frame_width

    @property
    def video_end(self) -> int:
        return self.video_start + self.num_frames * self.tokens_per_frame

    def frame_rows(self, frame: int) -> tuple[int, int]:
        start = self.video_start + frame * self.tokens_per_frame
        return start, start + self.tokens_per_frame

    def window_bounds(self) -> list[tuple[int, int]]:
        """Inclusive, unclamped frame range ``[lo, hi]`` each video frame sees."""
        if self.chunk <= 0:
            return [(t - self.radius, t + self.radius) for t in range(self.num_frames)]
        return [
            ((t // self.chunk - self.radius) * self.chunk, (t // self.chunk + self.radius + 1) * self.chunk - 1)
            for t in range(self.num_frames)
        ]

    @property
    def dense_row_frames(self) -> tuple[int, ...]:
        if self.anchor_frames not in ("rows", "both"):
            return ()
        return tuple(sorted({0, self.num_frames - 1}))

    @property
    def dense_column_frames(self) -> tuple[int, ...]:
        if self.anchor_frames not in ("columns", "both"):
            return ()
        return tuple(sorted({0, self.num_frames - 1}))

    @property
    def full_cover(self) -> bool:
        """Every window spans the whole clip: the softmax branch is dense attention."""
        return all(lo <= 0 and hi >= self.num_frames - 1 for lo, hi in self.window_bounds())


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(r for r in ranges if r[0] < r[1]):
        if merged and merged[-1][1] >= start:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


@functools.lru_cache(maxsize=16)
def window_plan(window: VDNLayout, device: torch.device) -> tuple[tuple[int, int, torch.Tensor | None], ...]:
    """The window mask as ``(query_start, query_stop, key_rows)`` dense calls.

    ``key_rows`` indexes the valid prefix; ``None`` means every valid row. The
    query ranges partition ``[0, used)`` exactly.
    """
    if window.full_cover:
        return ((0, window.used, None),)
    globals_ = [(0, window.video_start), (window.video_end, window.used)]
    dense_rows = _merge(globals_ + [window.frame_rows(f) for f in window.dense_row_frames])
    plan: list[tuple[int, int, torch.Tensor | None]] = [(start, stop, None) for start, stop in dense_rows]

    # Consecutive frames with the same window share one call (a whole chunk).
    bounds = window.window_bounds()
    groups: list[list[int]] = []
    for frame in range(window.num_frames):
        if frame in window.dense_row_frames:
            continue
        if groups and groups[-1][-1] == frame - 1 and bounds[groups[-1][-1]] == bounds[frame]:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    for frames in groups:
        lo, hi = bounds[frames[0]]
        key_frames = set(range(max(lo, 0), min(hi, window.num_frames - 1) + 1))
        key_frames.update(window.dense_column_frames)
        key_ranges = _merge(globals_ + [window.frame_rows(f) for f in key_frames])
        key_rows = torch.cat([torch.arange(a, b, dtype=torch.long) for a, b in key_ranges]).to(device)
        plan.append((window.frame_rows(frames[0])[0], window.frame_rows(frames[-1])[1], key_rows))

    covered = sum(stop - start for start, stop, _ in plan)
    if covered != window.used:
        raise AssertionError(f"VDN window plan covers {covered} of {window.used} rows")
    return tuple(plan)


def windowed_attention(query, key, value, window: VDNLayout, attend) -> torch.Tensor:
    """Run ``window_plan`` with ``attend(q, k, v) -> out`` on [B, S, H, D] tensors."""
    out = torch.zeros_like(query)
    key, value = key[:, : window.used], value[:, : window.used]
    for start, stop, key_rows in window_plan(window, query.device):
        if key_rows is None:
            out[:, start:stop] = attend(query[:, start:stop], key, value)
        else:
            out[:, start:stop] = attend(
                query[:, start:stop], key.index_select(1, key_rows), value.index_select(1, key_rows)
            )
    return out


class VDNAttentionBackend(FlashAttentionBackend):
    """``FLASH_ATTN`` plus the VDN-H3 window for layers that publish one."""

    supports_paged_kv: bool = False
    supports_piecewise_spans: bool = False
    supported_platforms = ("cuda",)

    @classmethod
    def supports_multi_doc_packed_varlen(cls) -> bool:
        # The window describes one document; co-batched requests run apart.
        return False

    @classmethod
    def validate_available(cls) -> None:
        if not current_omni_platform.supports_diffusion_dense_flash_attention():
            raise ImportError(
                "VDNH3_ATTN runs its window as dense FlashAttention calls; FlashAttention is not available"
            )

    @staticmethod
    def get_name() -> str:
        return "VDNH3_ATTN"

    @staticmethod
    def get_impl_cls() -> type[VDNAttentionImpl]:
        return VDNAttentionImpl

    @classmethod
    def resolve_capabilities(cls, context: ExecutionContext) -> ExecutionPathResult:
        # FLASH_ATTN's verified paths do not describe the window.
        return ExecutionPathResult.unmigrated(cls.get_name(), context)


class VDNAttentionImpl(FlashAttentionImpl):
    def _dense(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # With no metadata the parent runs one plain flash_attn_func call.
        return super().forward_cuda(query, key, value)

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
        if attn_metadata is None or attn_metadata.extra.get("vdn_window") is None:
            return super().resolve_execution_path(context, query, key, value, attn_metadata)
        return ExecutionPathResult.unmigrated("VDNH3_ATTN", context, path="window")

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        window = attn_metadata.extra.get("vdn_window") if attn_metadata is not None else None
        if window is None:
            return super().forward_cuda(query, key, value, attn_metadata)
        if query.shape[0] != 1 or query.shape[1] < window.used or key.shape[1] != query.shape[1]:
            raise ValueError(
                f"VDNH3_ATTN expects one packed document of >= {window.used} rows, got {tuple(query.shape)}"
            )
        logger.info_once(
            "VDNH3_ATTN window: frames=%d tokens/frame=%d video_start=%d used=%d chunk=%d radius=%d anchors=%s",
            window.num_frames,
            window.tokens_per_frame,
            window.video_start,
            window.used,
            window.chunk,
            window.radius,
            window.anchor_frames,
        )
        return windowed_attention(query, key, value, window, self._dense)

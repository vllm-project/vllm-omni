# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-session non-KV state for Cosmos3-Nano-Sim-Auto and the dense ring oracle.

The runner owns the paged KV pool; the pipeline owns everything else a
rollout needs between chunks: the prepared per-camera control latents, the
observed condition latents, the generated target latents, the compact caption
K/V, the ring-slot metadata the visibility planner reads, the chunk cursor and
per-camera decoder caches. ``DenseReplayRing`` is the model-owned twin of the
paged ring used off the AR engine and as the CPU-testable numerical oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .geometry import Chunk
from .layout import RingSlot, TokenMetadata


def ring_slot_for_step(step: int, *, cache_chunks: int, sink_chunks: int) -> int:
    """Reference slot assignment: sinks are pinned, later steps rotate through the rest."""
    if step < 0 or cache_chunks <= sink_chunks or sink_chunks < 0:
        raise ValueError("ring_slot_for_step needs step >= 0 and cache_chunks > sink_chunks >= 0.")
    if step < sink_chunks:
        return step
    return sink_chunks + (step - sink_chunks) % (cache_chunks - sink_chunks)


class DenseReplayRing:
    """Model-owned per-layer K/V ring ``[cache_chunks * frame_unit, H_kv, D]``.

    Slot ``s`` occupies virtual tokens ``[s * frame_unit, (s + 1) * frame_unit)``;
    the first ``num_tokens`` of a populated slot hold committed K/V and the rest
    is padding that no plan ever addresses. Virtual and physical coordinates
    coincide, so plans need no remap on this path.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        cache_chunks: int,
        frame_unit: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if min(num_layers, cache_chunks, frame_unit, num_kv_heads, head_dim) < 1:
            raise ValueError("DenseReplayRing dimensions must be positive.")
        self.cache_chunks = cache_chunks
        self.frame_unit = frame_unit
        capacity = cache_chunks * frame_unit
        self.keys = [
            torch.zeros(capacity, num_kv_heads, head_dim, device=device, dtype=dtype) for _ in range(num_layers)
        ]
        self.values = [
            torch.zeros(capacity, num_kv_heads, head_dim, device=device, dtype=dtype) for _ in range(num_layers)
        ]

    def slot_base(self, slot: int) -> int:
        if not 0 <= slot < self.cache_chunks:
            raise ValueError(f"Ring slot {slot} outside [0, {self.cache_chunks}).")
        return slot * self.frame_unit

    def write_slots(self, slot: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        if not 0 < num_tokens <= self.frame_unit:
            raise ValueError(f"Cannot write {num_tokens} tokens into a frame unit of {self.frame_unit}.")
        base = self.slot_base(slot)
        return torch.arange(base, base + num_tokens, device=device, dtype=torch.long)

    def clear_slot(self, slot: int) -> None:
        base = self.slot_base(slot)
        for key, value in zip(self.keys, self.values, strict=True):
            key[base : base + self.frame_unit].zero_()
            value[base : base + self.frame_unit].zero_()

    def layer_kv(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return list(zip(self.keys, self.values, strict=True))

    def publish(self, slot: int, current_kv: list[tuple[torch.Tensor, torch.Tensor]], num_tokens: int) -> None:
        """Publish staged clean-refresh K/V only after all layers have read history."""
        slots = self.write_slots(slot, num_tokens, self.keys[0].device)
        expected = (1, num_tokens, *self.keys[0].shape[1:])
        if len(current_kv) != len(self.keys) or any(
            tuple(key.shape) != expected or tuple(value.shape) != expected for key, value in current_kv
        ):
            raise ValueError(f"Expected {len(self.keys)} refreshed K/V layers with shape {expected}.")
        self.clear_slot(slot)
        for (key, value), key_pool, value_pool in zip(current_kv, self.keys, self.values, strict=True):
            key_pool.index_copy_(0, slots, key[0].to(key_pool))
            value_pool.index_copy_(0, slots, value[0].to(value_pool))


@dataclass
class Cosmos3NanoSimAutoSessionState:
    """Everything a session keeps between chunks besides the runner-owned paged KV."""

    session_id: str
    fingerprint: str | None = None
    terminal: bool = False
    next_chunk: int = 0
    # Request geometry (fixed for the session).
    cameras: tuple[str, ...] = ()
    joint: bool = False
    height: int = 0
    width: int = 0
    fps: float = 0.0
    num_frames: int = 0
    schedule: tuple[Chunk, ...] = ()
    frame_unit: int = 0
    text_extent: int = 0
    seed: int = 0
    # Prepared inputs (camera-major ``[1, C, V*T, h, w]``; LiDAR ``[1, C_l, S, h_l, w_l]``).
    rgb_control_latents: torch.Tensor | None = None
    rgb_target_latents: torch.Tensor | None = None
    rgb_condition_frames: frozenset[int] = frozenset()
    lidar_control_latents: torch.Tensor | None = None
    lidar_target_latents: torch.Tensor | None = None
    lidar_condition_sweeps: frozenset[int] = frozenset()
    # Captions: compact per-view ``[sink | caption]`` documents and their K/V per layer.
    caption_documents: tuple[Any, ...] = ()
    und_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    # Ring bookkeeping: slot -> committed metadata (what the planner sees as HISTORY).
    ring_metadata: dict[int, TokenMetadata] = field(default_factory=dict)
    ring_steps: dict[int, int] = field(default_factory=dict)
    dense_ring: DenseReplayRing | None = None
    # Per-camera Wan decoder caches for incremental decode (phase 2); unused by one-shot decode.
    decoder_caches: list[list[Any] | None] = field(default_factory=list)
    decoder_initialized: list[bool] = field(default_factory=list)

    @property
    def num_views(self) -> int:
        return len(self.cameras)

    def history_slots(self, *, frame_unit: int) -> list[RingSlot]:
        """Populated ring slots in slot order, addressed in virtual ring coordinates."""
        return [
            RingSlot(slot=slot, base=slot * frame_unit, metadata=self.ring_metadata[slot])
            for slot in sorted(self.ring_metadata)
        ]

    def record_commit(self, *, slot: int, step: int, metadata: TokenMetadata) -> None:
        self.ring_metadata[slot] = metadata
        self.ring_steps[slot] = step

    def reset(self) -> None:
        self.fingerprint = None
        self.terminal = False
        self.next_chunk = 0
        self.rgb_control_latents = None
        self.rgb_target_latents = None
        self.lidar_control_latents = None
        self.lidar_target_latents = None
        self.caption_documents = ()
        self.und_kv = None
        self.ring_metadata.clear()
        self.ring_steps.clear()
        self.dense_ring = None
        self.decoder_caches = []
        self.decoder_initialized = []


__all__ = ["Cosmos3NanoSimAutoSessionState", "DenseReplayRing", "ring_slot_for_step"]

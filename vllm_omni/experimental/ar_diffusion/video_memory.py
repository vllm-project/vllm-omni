# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, transactional video KV memory for eager autoregressive inference.

WorldKV pose retrieval and bank compression are independent opt-in policies.
Stored and retrieved keys retain their original RoPE positions; native sink
retention is preserved without Deep Forcing rebasing or query-based selection.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VideoMemoryConfig:
    window_frames: int = 18
    sink_frames: int = 0
    retrieval_frames: int = 0
    bank_bytes: int = 0
    world_keep_ratio: float = 1.0
    world_score: str = "pairwise_mean"

    def __post_init__(self) -> None:
        for name in (
            "window_frames",
            "sink_frames",
            "retrieval_frames",
            "bank_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"video_memory.{name} must be a non-negative integer")
        if self.window_frames <= self.sink_frames:
            raise ValueError("video_memory.window_frames must exceed sink_frames")
        if (
            isinstance(self.world_keep_ratio, bool)
            or not isinstance(self.world_keep_ratio, (int, float))
            or not 0 <= self.world_keep_ratio <= 1
        ):
            raise ValueError("video_memory.world_keep_ratio must be in [0,1]")
        if self.world_score not in ("pairwise_mean", "centroid"):
            raise ValueError("video_memory.world_score must be pairwise_mean or centroid")
        if self.world_keep_ratio != 1 and not self.retrieval_frames:
            raise ValueError("World compression requires retrieval")
        if self.retrieval_frames and not self.bank_bytes:
            raise ValueError("video_memory retrieval requires a positive bank_bytes limit")
        if self.sink_frames + self.retrieval_frames >= self.window_frames:
            raise ValueError("video_memory window must leave room for current frames")

    @classmethod
    def parse(cls, value: Mapping[str, object] | None) -> VideoMemoryConfig | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("model_config.video_memory must be a mapping")
        # Dataclass construction rejects unknown keys; __post_init__ validates
        # each untyped configuration value before it reaches model execution.
        return cls(**value)  # type: ignore[arg-type]

    def validate_bank_capacity(
        self, layers: int, tokens_per_frame: int, heads: int, dim: int, itemsize: int, chunk_frames: int
    ) -> None:
        if not self.retrieval_frames:
            return
        tokens = tokens_per_frame + math.ceil((chunk_frames - 1) * tokens_per_frame * self.world_keep_ratio)
        required = 128 + layers * tokens * (2 * heads * dim * itemsize + 16)
        if self.bank_bytes < required:
            raise ValueError(
                f"video_memory.bank_bytes={self.bank_bytes} cannot fit one compressed chunk ({required} bytes)"
            )

    def reservation_bytes(
        self,
        layers: int,
        tokens_per_frame: int,
        heads: int,
        dim: int,
        itemsize: int,
        *,
        chunk_frames: int | None = None,
    ) -> int:
        # All-layer owners: canonical + transactional replacement + pending,
        # retrieved device rows. Attention assembly, selection and scoring
        # scratch exist one layer at
        # a time; do not multiply those views by the transformer layer count.
        chunk = (
            chunk_frames if chunk_frames is not None else self.window_frames - self.sink_frames - self.retrieval_frames
        )
        self.validate_bank_capacity(layers, tokens_per_frame, heads, dim, itemsize, chunk)
        capacity = self.window_frames - self.retrieval_frames
        history = capacity
        publication = 2 * capacity
        kv = 2 * heads * dim * itemsize
        row = kv + 16
        noisy_or_clean = layers * tokens_per_frame * (history + chunk + self.retrieval_frames) * row
        publishing = layers * tokens_per_frame * publication * row
        resident = max(noisy_or_clean, publishing)
        # Four full-width fp32/fp64 K/V windows bound one layer's gathered
        # rows, scoring intermediates and concatenated attention view.
        # Model activations/backend workspace are charged by the runner.
        scratch = 4 * self.window_frames * tokens_per_frame * (2 * heads * dim * max(4, itemsize) + 16)
        return resident + scratch


@dataclass(frozen=True)
class MemoryRows:
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor
    token_ids: torch.Tensor

    def select(self, indices: torch.Tensor) -> MemoryRows:
        return MemoryRows(
            self.key[:, indices], self.value[:, indices], self.positions[indices], self.token_ids[indices]
        )

    def to(self, device: torch.device) -> MemoryRows:
        return MemoryRows(
            self.key.to(device), self.value.to(device), self.positions.to(device), self.token_ids.to(device)
        )


def world_compress(rows: MemoryRows, tokens_per_frame: int, keep_ratio: float, score: str) -> MemoryRows:
    """Anchor + pooled non-anchor selection, independently for each layer.

    pairwise_mean exactly factors mean cosine similarity as normalized keys
    dotted with the mean *normalized* anchor. centroid reproduces the released
    code's normalized raw-anchor centroid. Neither materializes T x T scores.
    """
    count = rows.positions.numel()
    if count <= tokens_per_frame or keep_ratio == 1:
        return rows
    dtype = torch.float64 if rows.key.dtype == torch.float64 else torch.float32
    flat = rows.key[0].to(dtype).flatten(1)

    def normalize(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, dim=-1)

    anchor, candidates = flat[:tokens_per_frame], flat[tokens_per_frame:]
    center = normalize(anchor).mean(0) if score == "pairwise_mean" else normalize(anchor.mean(0))
    scores = normalize(candidates) @ center
    keep = math.ceil((count - tokens_per_frame) * keep_ratio)
    selected = torch.argsort(scores, stable=True)[:keep].sort().values + tokens_per_frame
    indices = torch.cat((torch.arange(tokens_per_frame, device=flat.device), selected))
    return rows.select(indices)


@dataclass(frozen=True)
class BankChunk:
    start: int
    frames: int
    pose: torch.Tensor
    layers: tuple[MemoryRows, ...]

    @property
    def nbytes(self) -> int:
        return self.pose.numel() * self.pose.element_size() + sum(
            tensor.numel() * tensor.element_size()
            for rows in self.layers
            for tensor in (rows.key, rows.value, rows.positions, rows.token_ids)
        )


def camera_distances(current: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """WorldKV translation/rotation ranking, with finite zero-distance handling."""
    translation = (candidates[:, :3, 3] - current[:3, 3]).square().sum(-1)
    relative = candidates[:, :3, :3].transpose(-1, -2) @ current[:3, :3]
    rotation = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    return translation / translation.max().clamp_min(1e-12) + rotation / rotation.max().clamp_min(1e-12)


class VideoMemory:
    """One request/branch's clean history, published only after all layers succeed.

    Single-sequence eager execution is intentional. Per-layer selections are
    shared across heads; head-sharded execution requires a distributed scoring
    reduction before enabling compression.
    """

    def __init__(self, config: VideoMemoryConfig, *, num_layers: int, tokens_per_frame: int):
        if num_layers <= 0 or tokens_per_frame <= 0:
            raise ValueError("Invalid video memory geometry")
        self.config = config
        self.num_layers = num_layers
        self.tokens_per_frame = tokens_per_frame
        self.history: list[MemoryRows | None] = [None] * num_layers
        self.next_frame = 0
        self.bank: dict[int, BankChunk] = {}
        self.pose: torch.Tensor | None = None
        self._retrieved: tuple[BankChunk, ...] = ()
        self._retrieval_frames = 0
        self._device_retrieved: dict[int, tuple[MemoryRows, ...]] = {}
        self._active = False
        self._closed = False
        self._pending: dict[int, MemoryRows] = {}
        self._seen: set[int] = set()

    def set_pose(self, pose: torch.Tensor) -> None:
        """Supply a representative absolute C2W pose in the session coordinate frame."""
        if self._active or self._closed:
            raise RuntimeError("Set camera pose only between video memory forwards")
        pose = pose.detach().to(device="cpu", dtype=torch.float64, copy=True)
        if pose.shape != (4, 4) or not torch.isfinite(pose).all():
            raise ValueError("Video memory requires a finite 4x4 camera pose")
        rotation = pose[:3, :3]
        if (
            not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=pose.dtype), atol=1e-4)
            or not torch.allclose(torch.linalg.det(rotation), torch.tensor(1.0, dtype=pose.dtype), atol=1e-4)
            or not torch.allclose(pose[3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=pose.dtype))
        ):
            raise ValueError("Video memory camera pose must be a rigid C2W transform")
        if self.pose is None or not torch.equal(self.pose, pose):
            self._clear_retrieval()
        self.pose = pose

    def _clear_retrieval(self) -> None:
        self._retrieved = ()
        self._retrieval_frames = 0
        self._device_retrieved.clear()

    def _retrieved_rows(self, layer: int, device: torch.device) -> tuple[MemoryRows, ...]:
        if layer not in self._device_retrieved:
            self._device_retrieved[layer] = tuple(chunk.layers[layer].to(device) for chunk in self._retrieved)
        return self._device_retrieved[layer]

    @property
    def bank_nbytes(self) -> int:
        return sum(chunk.nbytes for chunk in self.bank.values())

    def _retrieve(self, frames: int) -> tuple[BankChunk, ...]:
        if not self.config.retrieval_frames:
            return ()
        if self.pose is None:
            raise ValueError("World retrieval requires an absolute camera pose for every chunk")
        recent_start = max(
            self.config.sink_frames,
            self.next_frame
            + frames
            - self.config.window_frames
            + self.config.sink_frames
            + self.config.retrieval_frames,
        )
        candidates = [
            chunk
            for chunk in self.bank.values()
            if chunk.start >= self.config.sink_frames and chunk.start + chunk.frames <= recent_start
        ]
        if not candidates:
            return ()
        distances = camera_distances(self.pose, torch.stack([chunk.pose for chunk in candidates]))
        ranked = torch.argsort(distances, stable=True).tolist()
        budget = self.config.retrieval_frames * self.tokens_per_frame
        selected = []
        for index in ranked:
            chunk = candidates[index]
            cost = max(rows.positions.numel() for rows in chunk.layers)
            if cost <= budget:
                selected.append(chunk)
                budget -= cost
        return tuple(sorted(selected, key=lambda chunk: chunk.start))

    def _publish_bank(self) -> dict[int, BankChunk]:
        bank = dict(self.bank)
        if not self.config.retrieval_frames or self.next_frame < self.config.sink_frames:
            return bank
        assert self.pose is not None
        layers = tuple(
            MemoryRows(
                rows.key.to(device="cpu", copy=True),
                rows.value.to(device="cpu", copy=True),
                rows.positions.to(device="cpu", copy=True),
                rows.token_ids.to(device="cpu", copy=True),
            )
            for rows in (
                world_compress(
                    self._pending[i], self.tokens_per_frame, self.config.world_keep_ratio, self.config.world_score
                )
                for i in range(self.num_layers)
            )
        )
        chunk = BankChunk(self.next_frame, self._frames, self.pose.clone(), layers)
        bank[chunk.start] = chunk
        size = sum(entry.nbytes for entry in bank.values())
        while size > self.config.bank_bytes:
            oldest = next(iter(bank))
            size -= bank.pop(oldest).nbytes
        return bank

    @contextmanager
    def forward(self, start_frame: int, frames: int, *, commit: bool) -> Iterator[None]:
        if self._closed or self._active:
            raise RuntimeError("Video memory is closed or already in a forward")
        if start_frame != self.next_frame or frames <= 0:
            raise ValueError("Video memory chunks must be contiguous and nonempty")
        if frames + self.config.sink_frames + self.config.retrieval_frames > self.config.window_frames:
            raise ValueError("Video memory must fit sink and the complete current chunk")
        success = False
        if self._retrieval_frames != frames:
            self._clear_retrieval()
            self._retrieved = self._retrieve(frames)
            self._retrieval_frames = frames
        self._active = True
        self._frames = frames
        self._commit = commit
        try:
            yield
            if len(self._seen) != self.num_layers:
                raise RuntimeError("Video memory forward did not execute all layers")
            if commit:
                # No attention consumer remains after the output head succeeds.
                # Publish CPU rows before consuming pending GPU rows, preserving
                # rollback: canonical history/bank are assigned only at the end.
                self._clear_retrieval()
                bank = self._publish_bank()
                replacement: list[MemoryRows | None] = [
                    self._retained(i, self._pending.pop(i)) for i in range(self.num_layers)
                ]
                self.history, self.bank = replacement, bank
                self.next_frame += frames
                self.pose = None
            success = True
        finally:
            self._pending.clear()
            self._seen.clear()
            self._active = False
            if not success or commit:
                self._clear_retrieval()

    def _retained(self, layer: int, current: MemoryRows) -> MemoryRows:
        previous = self.history[layer]
        if previous is None:
            return current
        capacity = (self.config.window_frames - self.config.retrieval_frames) * self.tokens_per_frame
        previous_capacity = capacity - current.positions.numel()
        count = previous.positions.numel()
        if count <= previous_capacity:
            slices = [slice(None)]
        else:
            sink = min(self.config.sink_frames * self.tokens_per_frame, count)
            tail = previous_capacity - sink
            slices = [slice(0, sink)]
            if tail:
                slices.append(slice(count - tail, count))
        # Slice before concatenation: never allocate a full old+current window
        # only to copy it again during truncation. All slices are temporary
        # views, and the one concatenation per tensor owns exactly retained KV.
        return MemoryRows(
            torch.cat([*(previous.key[:, part] for part in slices), current.key], 1),
            torch.cat([*(previous.value[:, part] for part in slices), current.value], 1),
            torch.cat([*(previous.positions[part] for part in slices), current.positions]),
            torch.cat([*(previous.token_ids[part] for part in slices), current.token_ids]),
        )

    def assemble(
        self, layer: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._active or layer in self._seen or not 0 <= layer < self.num_layers:
            raise RuntimeError("Each video memory layer must execute exactly once per forward")
        if key.ndim != 4 or key.shape != value.shape or query.shape != key.shape or key.shape[0] != 1:
            raise ValueError("Video memory requires matching single-sequence Q/K/V [1,T,H,D]")
        if key.shape[1] != self._frames * self.tokens_per_frame:
            raise ValueError("Current token count does not match video memory geometry")
        self.config.validate_bank_capacity(
            self.num_layers, self.tokens_per_frame, key.shape[2], key.shape[3], key.element_size(), self._frames
        )
        self._seen.add(layer)
        positions = torch.arange(self.next_frame, self.next_frame + self._frames, device=key.device)
        positions = positions.repeat_interleave(self.tokens_per_frame)
        if self._commit:
            # These slices must own storage: QKV projections otherwise keep Q
            # and the unselected tokens alive after a compression/commit.
            token_ids = torch.arange(
                self.next_frame * self.tokens_per_frame,
                (self.next_frame + self._frames) * self.tokens_per_frame,
                device=key.device,
            )
            self._pending[layer] = MemoryRows(key.detach().clone(), value.detach().clone(), positions, token_ids)
        previous = self.history[layer]
        if previous is None:
            return key, value
        sink = previous.positions < self.config.sink_frames
        tail_start = max(
            self.config.sink_frames,
            self.next_frame
            + self._frames
            - self.config.window_frames
            + self.config.sink_frames
            + self.config.retrieval_frames,
        )
        keep = sink | (previous.positions >= tail_start)
        rows = previous.select(torch.where(keep)[0])
        sink_count = int((rows.positions < self.config.sink_frames).sum())
        keys, values = [rows.key[:, :sink_count]], [rows.value[:, :sink_count]]
        for retrieved in self._retrieved_rows(layer, key.device):
            keys.append(retrieved.key)
            values.append(retrieved.value)
        keys.extend((rows.key[:, sink_count:], key))
        values.extend((rows.value[:, sink_count:], value))
        return torch.cat(keys, 1), torch.cat(values, 1)

    def close(self) -> None:
        self.history = [None] * self.num_layers
        self._pending.clear()
        self._seen.clear()
        self.bank.clear()
        self.pose = None
        self._clear_retrieval()
        self._closed = True

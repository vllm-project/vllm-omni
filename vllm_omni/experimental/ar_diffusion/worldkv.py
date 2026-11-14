# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WorldKV pose retrieval and compressed KV history for eager LingBot inference.

WorldKV pose retrieval and bank compression are independent opt-in policies.
Stored and retrieved keys retain their original RoPE positions; native sink
retention is preserved without Deep Forcing rebasing or query-based selection.
"""

from __future__ import annotations

import math
import threading
import weakref
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import psutil
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


# type() instead of isinstance() below: bool is a subclass of int, and YAML
# values such as `true` must not pass as 1.
def _require_int(name: str, value: object, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"worldkv.{name} must be an integer >= {minimum}, got {value!r}")


def _require_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise ValueError(f"worldkv.{name} must be a boolean, got {value!r}")


def _require_ratio(name: str, value: object) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
        raise ValueError(f"worldkv.{name} must be a number in [0, 1], got {value!r}")


@dataclass(frozen=True)
class WorldKVConfig:
    window_frames: int = 18
    sink_frames: int = 0
    retrieval_frames: int = 0
    bank_bytes: int = 0
    world_keep_ratio: float = 1.0
    world_score: str = "pairwise_mean"
    reuse_attention_kv: bool = False
    prefetch_kv: bool = False
    prefetch_bank_allocation: bool = False
    layerwise_bank_copy: bool = False
    # Process-wide cap on the bank_bytes reserved by live memories; None uses
    # the host's physical memory.
    host_bank_limit_bytes: int | None = None

    def __post_init__(self) -> None:
        # Values come from untyped YAML, so check each field's type first and
        # the rules that relate fields afterwards.
        _require_int("window_frames", self.window_frames, minimum=0)
        _require_int("sink_frames", self.sink_frames, minimum=0)
        _require_int("retrieval_frames", self.retrieval_frames, minimum=0)
        _require_int("bank_bytes", self.bank_bytes, minimum=0)
        _require_bool("reuse_attention_kv", self.reuse_attention_kv)
        _require_bool("prefetch_kv", self.prefetch_kv)
        _require_bool("prefetch_bank_allocation", self.prefetch_bank_allocation)
        _require_bool("layerwise_bank_copy", self.layerwise_bank_copy)
        _require_ratio("world_keep_ratio", self.world_keep_ratio)
        if self.host_bank_limit_bytes is not None:
            _require_int("host_bank_limit_bytes", self.host_bank_limit_bytes, minimum=1)

        if self.window_frames <= self.sink_frames:
            raise ValueError("worldkv.window_frames must exceed sink_frames")
        if self.world_score not in ("pairwise_mean", "centroid"):
            raise ValueError("worldkv.world_score must be pairwise_mean or centroid")
        if self.world_keep_ratio != 1 and not self.retrieval_frames:
            raise ValueError("World compression requires retrieval")
        if self.retrieval_frames and not self.bank_bytes:
            raise ValueError("worldkv retrieval requires a positive bank_bytes limit")
        if self.sink_frames + self.retrieval_frames >= self.window_frames:
            raise ValueError("worldkv window must leave room for current frames")
        if self.host_bank_limit_bytes is not None:
            if not self.retrieval_frames:
                raise ValueError("worldkv.host_bank_limit_bytes requires retrieval")
            if self.host_bank_limit_bytes < self.bank_bytes:
                raise ValueError("worldkv.host_bank_limit_bytes must be at least bank_bytes")

    @classmethod
    def parse(
        cls,
        value: Mapping[str, object] | None,
        *,
        window_frames: int,
        sink_frames: int,
        chunk_frames: int,
    ) -> WorldKVConfig | None:
        """Parse the model_config mapping against the model's native geometry.

        Omitted window and sink sizes inherit the model's native values, so
        enabling WorldKV never silently drops the attention sink.
        """
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("model_config.worldkv must be a mapping")
        # Dataclass construction rejects unknown keys; __post_init__ validates
        # each untyped configuration value before it reaches model execution.
        config = cls(**{"window_frames": window_frames, "sink_frames": sink_frames, **value})  # type: ignore[arg-type]
        config.validate_chunk_frames(chunk_frames)
        return config

    def validate_chunk_frames(self, chunk_frames: int) -> None:
        """Reject chunk geometry that would fail or silently lose history at run time."""
        if self.sink_frames + self.retrieval_frames + chunk_frames > self.window_frames:
            raise ValueError("WorldKV must fit sink, retrieval and the complete current chunk")
        if not self.retrieval_frames:
            return
        if self.sink_frames % chunk_frames:
            # Only chunks starting after the sink are banked, so the non-sink
            # frames of a chunk straddling the sink could never be retrieved.
            raise ValueError("worldkv.sink_frames must be a multiple of the chunk size when retrieval is enabled")
        # Token rounding cannot exceed this frame bound: the budget is a whole
        # number of frames and the compressed count is rounded up only once.
        if 1 + (chunk_frames - 1) * self.world_keep_ratio > self.retrieval_frames:
            raise ValueError("worldkv.retrieval_frames cannot fit one compressed chunk")

    def validate_bank_capacity(
        self, layers: int, tokens_per_frame: int, heads: int, dim: int, itemsize: int, chunk_frames: int
    ) -> None:
        if not self.retrieval_frames:
            return
        tokens = tokens_per_frame + math.ceil((chunk_frames - 1) * tokens_per_frame * self.world_keep_ratio)
        required = 128 + layers * tokens * (2 * heads * dim * itemsize + 16)
        if self.bank_bytes < required:
            raise ValueError(f"worldkv.bank_bytes={self.bank_bytes} cannot fit one compressed chunk ({required} bytes)")

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
        if self.layerwise_bank_copy and self.retrieval_frames and self.world_keep_ratio == 1:
            # Early D2H borrows pending clones rather than the replacement tail.
            # Their storage may remain in flight while both histories are live.
            publishing += layers * tokens_per_frame * chunk * kv
        if self.retrieval_frames and self.world_keep_ratio != 1:
            # Uncompressed D2H borrows canonical history. Compressed rows own
            # separate storage until the single in-flight copy completes.
            compressed = tokens_per_frame + math.ceil((chunk - 1) * tokens_per_frame * self.world_keep_ratio)
            noisy_or_clean += layers * compressed * row
            publishing += layers * compressed * row
        resident = max(noisy_or_clean, publishing)
        # Four full-width fp32/fp64 K/V windows bound one layer's gathered
        # rows, scoring intermediates and concatenated attention view.
        # Model activations/backend workspace are charged by the runner.
        scratch = 4 * self.window_frames * tokens_per_frame * (2 * heads * dim * max(4, itemsize) + 16)
        if self.reuse_attention_kv:
            resident += layers * self.window_frames * tokens_per_frame * kv
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

    def to_kv_device(self, device: torch.device) -> MemoryRows:
        """Move attention payload only; retrieval metadata stays on its source device."""
        return MemoryRows(
            self.key.to(device, non_blocking=True),
            self.value.to(device, non_blocking=True),
            self.positions,
            self.token_ids,
        )

    def slice(self, start: int, end: int | None = None) -> MemoryRows:
        return MemoryRows(
            self.key[:, start:end], self.value[:, start:end], self.positions[start:end], self.token_ids[start:end]
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
    ready: torch.Event | None = None

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


class _HostBankLedger:
    """Worst-case CPU bank bytes reserved by all live memories in this process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reserved = 0

    def reserve(self, nbytes: int, limit: int) -> None:
        with self._lock:
            if self.reserved + nbytes > limit:
                raise RuntimeError(
                    f"WorldKV bank reservation of {nbytes} bytes exceeds the host limit of {limit} bytes "
                    f"({self.reserved} bytes already reserved); close idle sessions or lower worldkv.bank_bytes"
                )
            self.reserved += nbytes

    def release(self, nbytes: int) -> None:
        with self._lock:
            self.reserved -= nbytes


_HOST_BANKS = _HostBankLedger()


def _allocate_pinned_bank(sizes: dict[torch.dtype, int], device: torch.device) -> dict[torch.dtype, torch.Tensor]:
    # The worker owns no request or GPU tensor references. Select the producer's
    # device explicitly because accelerator current-device state is thread-local.
    with torch.accelerator.device_index(device.index):
        return {dtype: torch.empty(size, dtype=dtype, device="cpu", pin_memory=True) for dtype, size in sizes.items()}


class WorldKV:
    """One request/branch's clean history, published only after all layers succeed.

    Single-sequence eager execution is intentional. Per-layer selections are
    shared across heads; head-sharded execution requires a distributed scoring
    reduction before enabling compression.
    """

    def __init__(self, config: WorldKVConfig, *, num_layers: int, tokens_per_frame: int):
        if num_layers <= 0 or tokens_per_frame <= 0:
            raise ValueError("Invalid WorldKV geometry")
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
        self._assembly: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._prefetch_stream: torch.Stream | None = None
        self._prefetch_ready: dict[int, torch.Event] = {}
        self._active = False
        self._closed = False
        self._pending: dict[int, MemoryRows] = {}
        self._seen: set[int] = set()
        self._copy_stream: torch.Stream | None = None
        self._publication: torch.Event | None = None
        self._bank_allocator: ThreadPoolExecutor | None = None
        self._stop_allocator: weakref.finalize | None = None
        self._bank_allocation: Future[dict[torch.dtype, torch.Tensor]] | None = None
        self._bank_allocation_sizes: dict[torch.dtype, int] = {}
        self._staged_bank: tuple[MemoryRows, ...] | None = None
        self.evicted_chunks = 0
        self._release_bank: weakref.finalize | None = None
        if config.retrieval_frames:
            limit = config.host_bank_limit_bytes or psutil.virtual_memory().total
            _HOST_BANKS.reserve(config.bank_bytes, limit)
            # Request-local caches are dropped rather than closed.
            self._release_bank = weakref.finalize(self, _HOST_BANKS.release, config.bank_bytes)

    def set_pose(self, pose: torch.Tensor) -> None:
        """Supply a representative absolute C2W pose in the session coordinate frame."""
        if self._active or self._closed:
            raise RuntimeError("Set camera pose only between WorldKV forwards")
        pose = pose.detach().to(device="cpu", dtype=torch.float64, copy=True)
        if pose.shape != (4, 4) or not torch.isfinite(pose).all():
            raise ValueError("WorldKV requires a finite 4x4 camera pose")
        rotation = pose[:3, :3]
        if (
            not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=pose.dtype), atol=1e-4)
            or not torch.allclose(torch.linalg.det(rotation), torch.tensor(1.0, dtype=pose.dtype), atol=1e-4)
            or not torch.allclose(pose[3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=pose.dtype))
        ):
            raise ValueError("WorldKV camera pose must be a rigid C2W transform")
        if self.pose is None or not torch.equal(self.pose, pose):
            self._clear_retrieval()
        self.pose = pose

    def _clear_retrieval(self) -> None:
        self._retrieved = ()
        self._retrieval_frames = 0
        if self._prefetch_stream is not None and self._prefetch_ready:
            self._prefetch_stream.synchronize()
        self._prefetch_ready.clear()
        self._device_retrieved.clear()
        self._assembly.clear()

    def _retrieved_rows(self, layer: int, device: torch.device) -> tuple[MemoryRows, ...]:
        if layer not in self._device_retrieved:
            rows = []
            for chunk in self._retrieved:
                resident = self._resident_rows(layer, chunk)
                if resident is not None and resident.key.device == device:
                    rows.append(resident)
                    continue
                if chunk.ready is not None:
                    if device.type == "cuda":
                        torch.accelerator.current_stream(device).wait_event(chunk.ready)
                    else:
                        chunk.ready.synchronize()
                rows.append(chunk.layers[layer].to_kv_device(device))
            self._device_retrieved[layer] = tuple(rows)
        if layer in self._prefetch_ready:
            stream = torch.accelerator.current_stream(device)
            stream.wait_event(self._prefetch_ready.pop(layer))
            for retrieved in self._device_retrieved[layer]:
                retrieved.key.record_stream(stream)
                retrieved.value.record_stream(stream)
        return self._device_retrieved[layer]

    def _prefetch(self, device: torch.device) -> None:
        if not self.config.prefetch_kv or device.type != "cuda" or not self._retrieved or self._device_retrieved:
            return
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.Stream(device=device)
        stream = self._prefetch_stream
        stream.wait_stream(torch.accelerator.current_stream(device))
        try:
            with stream:
                for layer in range(self.num_layers):
                    self._retrieved_rows(layer, device)
                    self._prefetch_ready[layer] = stream.record_event()
        except BaseException:
            stream.synchronize()
            raise

    def _resident_rows(self, layer: int, chunk: BankChunk) -> MemoryRows | None:
        # History keeps a full window until commit; attention excludes the
        # oldest current-chunk-sized prefix *before* that. Reuse this prefix
        # when pose retrieval selects it, without another GPU cache or copy.
        previous = self.history[layer]
        if previous is None or self.config.world_keep_ratio != 1:
            return None
        sink = min(self.config.sink_frames, self.next_frame) * self.tokens_per_frame
        tail_start = self.next_frame - (previous.positions.numel() - sink) // self.tokens_per_frame
        if chunk.start < tail_start or chunk.start + chunk.frames > self.next_frame:
            return None
        start = sink + (chunk.start - tail_start) * self.tokens_per_frame
        end = start + chunk.frames * self.tokens_per_frame
        return previous.slice(start, end)

    def _wait_publication(self) -> None:
        if self._publication is not None:
            self._publication.synchronize()
            self._publication = None

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

    def _discard_bank_allocation(self) -> None:
        allocation, self._bank_allocation = self._bank_allocation, None
        self._bank_allocation_sizes = {}
        if allocation is not None:
            # Wait for completion without replacing an existing forward error
            # with a background allocation failure. Publication surfaces failures.
            allocation.exception()

    def _prefetch_bank_allocation(self, frames: int) -> None:
        if not self.config.prefetch_bank_allocation:
            return
        if not self.config.retrieval_frames or self.next_frame < self.config.sink_frames:
            return
        first = self.history[0]
        if first is None or first.key.device.type != "cuda":
            return
        count = self.tokens_per_frame + math.ceil((frames - 1) * self.tokens_per_frame * self.config.world_keep_ratio)
        sizes: dict[torch.dtype, int] = {}
        for rows in self.history:
            if rows is None:
                return
            self.config.validate_bank_capacity(
                self.num_layers,
                self.tokens_per_frame,
                rows.key.shape[2],
                rows.key.shape[3],
                rows.key.element_size(),
                frames,
            )
            for tensor in (rows.key, rows.value):
                sizes[tensor.dtype] = sizes.get(tensor.dtype, 0) + count * tensor.shape[2] * tensor.shape[3]
        if self._bank_allocation is not None:
            if sizes == self._bank_allocation_sizes:
                return
            self._discard_bank_allocation()
        if self._bank_allocator is None:
            self._bank_allocator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="worldkv-bank")
            self._stop_allocator = weakref.finalize(self, self._bank_allocator.shutdown, wait=True)
        self._bank_allocation_sizes = sizes
        self._bank_allocation = self._bank_allocator.submit(_allocate_pinned_bank, sizes, first.key.device)

    def _bank_storage(self, sizes: dict[torch.dtype, int], asynchronous: bool) -> dict[torch.dtype, torch.Tensor]:
        if asynchronous and self._bank_allocation is not None and sizes == self._bank_allocation_sizes:
            allocation, self._bank_allocation = self._bank_allocation, None
            self._bank_allocation_sizes = {}
            storage = allocation.result()
        else:
            self._discard_bank_allocation()
            storage = {
                dtype: torch.empty(size, dtype=dtype, device="cpu", pin_memory=asynchronous)
                for dtype, size in sizes.items()
            }
        return storage

    def _prepare_staged_bank(self) -> None:
        if not self._commit or not self.config.layerwise_bank_copy or self.config.world_keep_ratio != 1:
            return
        if not self.config.retrieval_frames or self.next_frame < self.config.sink_frames:
            return
        if any(rows is None or rows.key.device.type != "cuda" for rows in self.history):
            return
        self._wait_publication()
        shapes = []
        sizes: dict[torch.dtype, int] = {}
        for rows in self.history:
            assert rows is not None
            self.config.validate_bank_capacity(
                self.num_layers,
                self.tokens_per_frame,
                rows.key.shape[2],
                rows.key.shape[3],
                rows.key.element_size(),
                self._frames,
            )
            for tensor in (rows.key, rows.value):
                shape = (1, self._frames * self.tokens_per_frame, *tensor.shape[2:])
                shapes.append((tensor.dtype, shape))
                sizes[tensor.dtype] = sizes.get(tensor.dtype, 0) + math.prod(shape)
        storage = self._bank_storage(sizes, True)
        offsets = dict.fromkeys(sizes, 0)
        targets = []
        for dtype, shape in shapes:
            count = math.prod(shape)
            targets.append(storage[dtype].narrow(0, offsets[dtype], count).view(shape))
            offsets[dtype] += count
        positions = torch.arange(self.next_frame, self.next_frame + self._frames, device="cpu").repeat_interleave(
            self.tokens_per_frame
        )
        token_ids = torch.arange(
            self.next_frame * self.tokens_per_frame,
            (self.next_frame + self._frames) * self.tokens_per_frame,
            device="cpu",
        )
        self._staged_bank = tuple(
            MemoryRows(targets[2 * layer], targets[2 * layer + 1], positions, token_ids)
            for layer in range(self.num_layers)
        )

    def _stage_bank_layer(self, layer: int, source: MemoryRows) -> None:
        if self._staged_bank is None:
            return
        target = self._staged_bank[layer]
        if any(
            a.shape != b.shape or a.dtype != b.dtype
            for a, b in ((source.key, target.key), (source.value, target.value))
        ):
            # Geometry can change between chunks. Drain partial copies before
            # releasing their storage, then use the ordinary publication path.
            self._wait_publication()
            self._staged_bank = None
            return
        if self._copy_stream is None:
            self._copy_stream = torch.Stream(device=source.key.device)
        stream = self._copy_stream
        stream.wait_stream(torch.accelerator.current_stream(source.key.device))
        try:
            with stream:
                source.key.record_stream(stream)
                source.value.record_stream(stream)
                target.key.copy_(source.key, non_blocking=True)
                target.value.copy_(source.value, non_blocking=True)
                self._publication = stream.record_event()
        except BaseException:
            stream.synchronize()
            raise

    def _bank_with_chunk(self, chunk: BankChunk) -> tuple[dict[int, BankChunk], int]:
        bank = dict(self.bank)
        bank[chunk.start] = chunk
        size = sum(entry.nbytes for entry in bank.values())
        evicted = 0
        while size > self.config.bank_bytes:
            oldest = next(iter(bank))
            size -= bank.pop(oldest).nbytes
            evicted += 1
        return bank, evicted

    def _publish_bank(self, replacement: list[MemoryRows | None]) -> tuple[dict[int, BankChunk], int]:
        if not self.config.retrieval_frames or self.next_frame < self.config.sink_frames:
            return dict(self.bank), 0
        assert self.pose is not None
        # forward checks that all layers succeeded before publication. A shape
        # fallback discards staged storage, and copy failures abort the forward.
        if self._staged_bank is not None:
            return self._bank_with_chunk(
                BankChunk(self.next_frame, self._frames, self.pose.clone(), self._staged_bank, self._publication)
            )
        self._wait_publication()
        sources = []
        for rows in replacement:
            assert rows is not None
            sources.append(
                world_compress(
                    rows.slice(-self._frames * self.tokens_per_frame),
                    self.tokens_per_frame,
                    self.config.world_keep_ratio,
                    self.config.world_score,
                )
            )
        device = sources[0].key.device
        asynchronous = device.type == "cuda"
        # PyTorch's pinned allocator caches freed blocks after FIFO eviction.
        # Never overwrite a live bank entry: publication can still fail below.
        # One allocation per dtype and publication, rather than two per layer.
        # Each bank chunk owns its backing storage; views never overlap or reuse
        # a live entry. Grouping by dtype also permits different layer geometry.
        tensors = [tensor for rows in sources for tensor in (rows.key, rows.value)]
        sizes: dict[torch.dtype, int] = {}
        for tensor in tensors:
            sizes[tensor.dtype] = sizes.get(tensor.dtype, 0) + tensor.numel()
        storage = self._bank_storage(sizes, asynchronous)
        offsets = dict.fromkeys(sizes, 0)
        targets = []
        for tensor in tensors:
            offset = offsets[tensor.dtype]
            targets.append(storage[tensor.dtype].narrow(0, offset, tensor.numel()).view(tensor.shape))
            offsets[tensor.dtype] += tensor.numel()
        # Uncompressed publication is a contiguous current chunk. Build its
        # immutable metadata once on CPU instead of synchronously reading it
        # back from every layer. Compressed selections remain layer-specific.
        positions = token_ids = None
        if self.config.world_keep_ratio == 1:
            positions = torch.arange(self.next_frame, self.next_frame + self._frames, device="cpu").repeat_interleave(
                self.tokens_per_frame
            )
            token_ids = torch.arange(
                self.next_frame * self.tokens_per_frame,
                (self.next_frame + self._frames) * self.tokens_per_frame,
                device="cpu",
            )
        layers = tuple(
            MemoryRows(
                targets[2 * layer],
                targets[2 * layer + 1],
                positions if positions is not None else rows.positions.to(device="cpu", copy=True),
                token_ids if token_ids is not None else rows.token_ids.to(device="cpu", copy=True),
            )
            for layer, rows in enumerate(sources)
        )
        ready = None
        if asynchronous:
            if self._copy_stream is None:
                self._copy_stream = torch.Stream(device=device)
            stream = self._copy_stream
            stream.wait_stream(torch.accelerator.current_stream(device))
            try:
                with stream:
                    for source, target in zip(sources, layers):
                        # Mark before enqueueing, including the partial-copy
                        # failure path and compressed temporary rows.
                        source.key.record_stream(stream)
                        source.value.record_stream(stream)
                        target.key.copy_(source.key, non_blocking=True)
                        target.value.copy_(source.value, non_blocking=True)
                    ready = stream.record_event()
                self._publication = ready
            except BaseException:
                stream.synchronize()
                raise
        else:
            for source, target in zip(sources, layers):
                target.key.copy_(source.key)
                target.value.copy_(source.value)
        chunk = BankChunk(self.next_frame, self._frames, self.pose.clone(), layers, ready)
        return self._bank_with_chunk(chunk)

    @contextmanager
    def forward(self, start_frame: int, frames: int, *, commit: bool) -> Iterator[None]:
        if self._closed or self._active:
            raise RuntimeError("WorldKV is closed or already in a forward")
        if start_frame != self.next_frame or frames <= 0:
            raise ValueError("WorldKV chunks must be contiguous and nonempty")
        self.config.validate_chunk_frames(frames)
        success = False
        if self._retrieval_frames != frames:
            self._clear_retrieval()
            self._retrieved = self._retrieve(frames)
            self._retrieval_frames = frames
        self._active = True
        self._frames = frames
        self._commit = commit
        try:
            self._prefetch_bank_allocation(frames)
            self._prepare_staged_bank()
            yield
            if len(self._seen) != self.num_layers:
                raise RuntimeError("WorldKV forward did not execute all layers")
            if commit:
                # No attention consumer remains after the output head succeeds.
                # Ordinary uncompressed D2H borrows the replacement's tail;
                # layerwise D2H has already queued copies from pending KV.
                # Canonical history/bank are assigned only after both succeed.
                self._clear_retrieval()
                replacement: list[MemoryRows | None] = [
                    self._retained(i, self._pending.pop(i)) for i in range(self.num_layers)
                ]
                bank, evicted = self._publish_bank(replacement)
                self.history, self.bank = replacement, bank
                self.next_frame += frames
                self.pose = None
                if evicted and not self.evicted_chunks:
                    logger.warning(
                        "WorldKV bank reached bank_bytes=%d at latent frame %d; the oldest chunks are now "
                        "evicted and can no longer be retrieved on revisits. Raise worldkv.bank_bytes to "
                        "cover the full trajectory.",
                        self.config.bank_bytes,
                        self.next_frame,
                    )
                self.evicted_chunks += evicted
            success = True
        finally:
            if not success:
                self._discard_bank_allocation()
                self._wait_publication()
            self._staged_bank = None
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
            raise RuntimeError("Each WorldKV layer must execute exactly once per forward")
        if key.ndim != 4 or key.shape != value.shape or query.shape != key.shape or key.shape[0] != 1:
            raise ValueError("WorldKV requires matching single-sequence Q/K/V [1,T,H,D]")
        if key.shape[1] != self._frames * self.tokens_per_frame:
            raise ValueError("Current token count does not match WorldKV geometry")
        self.config.validate_bank_capacity(
            self.num_layers, self.tokens_per_frame, key.shape[2], key.shape[3], key.element_size(), self._frames
        )
        self._seen.add(layer)
        if self._commit:
            # These slices must own storage: QKV projections otherwise keep Q
            # and the unselected tokens alive after a compression/commit.
            positions = torch.arange(self.next_frame, self.next_frame + self._frames, device=key.device)
            positions = positions.repeat_interleave(self.tokens_per_frame)
            token_ids = torch.arange(
                self.next_frame * self.tokens_per_frame,
                (self.next_frame + self._frames) * self.tokens_per_frame,
                device=key.device,
            )
            self._pending[layer] = MemoryRows(key.detach().clone(), value.detach().clone(), positions, token_ids)
            self._stage_bank_layer(layer, self._pending[layer])
        previous = self.history[layer]
        if previous is None:
            return key, value
        self._prefetch(key.device)
        if layer in self._assembly:
            assembled_key, assembled_value = self._assembly[layer]
            assembled_key[:, -key.shape[1] :].copy_(key)
            assembled_value[:, -value.shape[1] :].copy_(value)
            return assembled_key, assembled_value
        # Canonical history is an uncompressed sink followed by a contiguous
        # recent suffix. Its boundaries are known on the host, even for short
        # chunks. Avoid a GPU nonzero/scalar read and a gathered copy per layer.
        sink_count = min(self.config.sink_frames, self.next_frame) * self.tokens_per_frame
        tail_start = max(
            self.config.sink_frames,
            self.next_frame
            + self._frames
            - self.config.window_frames
            + self.config.sink_frames
            + self.config.retrieval_frames,
        )
        tail_count = min(
            previous.positions.numel() - sink_count,
            max(0, self.next_frame - tail_start) * self.tokens_per_frame,
        )
        keys, values = [previous.key[:, :sink_count]], [previous.value[:, :sink_count]]
        for retrieved in self._retrieved_rows(layer, key.device):
            keys.append(retrieved.key)
            values.append(retrieved.value)
        if tail_count:
            keys.append(previous.key[:, -tail_count:])
            values.append(previous.value[:, -tail_count:])
        keys.append(key)
        values.append(value)
        assembled = torch.cat(keys, 1), torch.cat(values, 1)
        if self.config.reuse_attention_kv:
            self._assembly[layer] = assembled
        return assembled

    def close(self) -> None:
        self._discard_bank_allocation()
        if self._stop_allocator is not None:
            self._stop_allocator()
        self._bank_allocator = None
        self._wait_publication()
        self._staged_bank = None
        self.history = [None] * self.num_layers
        self._pending.clear()
        self._seen.clear()
        self.bank.clear()
        self.pose = None
        self._clear_retrieval()
        self._closed = True
        self._copy_stream = None
        self._prefetch_stream = None
        if self._release_bank is not None:
            self._release_bank()

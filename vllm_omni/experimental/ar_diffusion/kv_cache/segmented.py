# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact, transactional token segments in the runner-owned paged KV pool.

The model chooses which immutable segments to retain and their order. Storage,
publication, rollback, and contiguous attention staging belong to the runtime.
This path does not change the frame-window attention contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.paged import compute_slot_mapping

if TYPE_CHECKING:
    from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache, ARDiffusionRequestAdapter


@dataclass(frozen=True, eq=False)
class SegmentHandle:
    """An opaque allocation identity with an exact count of initialized rows."""

    num_tokens: int


@dataclass
class SegmentedStaging:
    """Worker-local, fixed-address K/V buffers shared by successive sessions."""

    layers: list[tuple[torch.Tensor, torch.Tensor]]
    capacity: int
    owner: object | None = None
    revision: int = -1


@dataclass
class _Allocation:
    adapter: ARDiffusionRequestAdapter
    slots: torch.Tensor


@dataclass
class _Pending:
    handles: tuple[SegmentHandle, ...]
    slots: torch.Tensor
    written_layers: set[int] = field(default_factory=set)


class ARDiffusionSegmentedKVState:
    """Session/branch-local immutable segments backed by a shared KV manager.

    Call ``prepare_layer_views`` outside compilation/capture before a forward.
    Each layer then assembles current rows into stable contiguous buffers. A
    clean forward may stage those rows into unpublished paged allocations;
    ``commit`` publishes only after all layers succeed. Noisy forwards write
    only staging, so neither failures nor denoise steps mutate committed KV.
    """

    def __init__(self, cache: ARDiffusionKVCache, request_id: str, kv_branch: str) -> None:
        if cache.segmented_spec is None:
            raise RuntimeError("Segmented KV requires a segmented cache capability")
        self.cache = cache
        self.spec = cache.segmented_spec
        self.request_id = request_id
        self.kv_branch = kv_branch
        self._allocations: dict[SegmentHandle, _Allocation] = {}
        self._retained: tuple[SegmentHandle, ...] = ()
        self._pending: _Pending | None = None
        self._next_id = 0
        self._revision = 0
        self._closed = False
        self._prepared: tuple[int, int] | None = None
        self._assembled: set[int] = set()
        self.history_gathers = 0

    @property
    def history_tokens(self) -> int:
        return sum(handle.num_tokens for handle in self._retained)

    @property
    def handles(self) -> tuple[SegmentHandle, ...]:
        return self._retained

    @property
    def commit_active(self) -> bool:
        return self._pending is not None

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Segmented KV session is closed")

    def _staging(self) -> SegmentedStaging:
        return self.cache.segmented_staging[self.cache._kv_branch_index(self.kv_branch)]

    def prepare_layer_views(self, media_tokens: int, condition_tokens: int) -> None:
        """Admit the shape and stage history once, outside a captured forward.

        Long, unpinned conditioning may grow the buffers after a memory check.
        Ordinary forwards keep the original addresses, and unchanged history
        is never gathered again during a chunk's denoise/clean forwards.
        """
        self._check_open()
        if self.cache.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare segmented KV views outside CUDA graph capture")
        if media_tokens <= 0 or media_tokens > self.spec.max_pending_tokens:
            raise ValueError(f"Segmented current rows must be in [1, {self.spec.max_pending_tokens}]")
        if condition_tokens < 0:
            raise ValueError("Segmented conditioning rows must be non-negative")
        if self._pending is not None:
            raise RuntimeError("Prepare segmented views before beginning a commit")
        needed = self.history_tokens + media_tokens + condition_tokens
        self.cache.ensure_segmented_staging(self.kv_branch, needed)
        staging = self._staging()
        if staging.owner is not self or staging.revision != self._revision:
            if self._retained:
                slots = torch.cat([self._allocations[handle].slots for handle in self._retained])
                for index, (key, value) in enumerate(staging.layers):
                    torch.index_select(self.cache._k_pools[index], 0, slots, out=key[: self.history_tokens])
                    torch.index_select(self.cache._v_pools[index], 0, slots, out=value[: self.history_tokens])
                self.history_gathers += 1
            staging.owner, staging.revision = self, self._revision
        self._prepared = (int(media_tokens), int(condition_tokens))
        self._assembled.clear()

    def assemble(
        self,
        layer_idx: int,
        k_condition: torch.Tensor,
        v_condition: torch.Tensor,
        k_media: torch.Tensor,
        v_media: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact initialized ``[history | media | condition]`` rows."""
        self._check_open()
        self._validate_layer(layer_idx)
        staging = self._staging()
        if self._prepared is None or staging.owner is not self or staging.revision != self._revision:
            raise RuntimeError("prepare_layer_views() must precede segmented attention")
        media_tokens, condition_tokens = self._prepared
        for key, value, count in (
            (k_media, v_media, media_tokens),
            (k_condition, v_condition, condition_tokens),
        ):
            expected = (count, self.cache.num_kv_heads, self.cache.head_size)
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError(f"Segmented KV expected key/value shape {expected}")
            if key.dtype != self.cache.dtype or value.dtype != self.cache.dtype:
                raise ValueError(f"Segmented KV must use {self.cache.dtype}")
            device = staging.layers[layer_idx][0].device
            if key.device != device or value.device != device:
                raise ValueError(f"Segmented KV must be on {device}")
        key, value = staging.layers[layer_idx]
        start = self.history_tokens
        media_end = start + media_tokens
        end = media_end + condition_tokens
        key[start:media_end].copy_(k_media)
        value[start:media_end].copy_(v_media)
        key[media_end:end].copy_(k_condition)
        value[media_end:end].copy_(v_condition)
        self._assembled.add(layer_idx)
        return key[:end], value[:end]

    def begin_commit(self, segment_lengths: Sequence[int]) -> tuple[SegmentHandle, ...]:
        """Reserve one unpublished segment per contiguous current-media span."""
        self._check_open()
        lengths = tuple(int(length) for length in segment_lengths)
        if self._pending is not None:
            raise RuntimeError("A segmented KV commit is already active")
        staging = self._staging()
        if self._prepared is None or staging.owner is not self or staging.revision != self._revision:
            raise RuntimeError("prepare_layer_views() must precede begin_commit()")
        if not lengths or len(lengths) > self.spec.max_pending_segments or any(length <= 0 for length in lengths):
            raise ValueError("Invalid segmented KV pending segment lengths")
        if sum(lengths) != self._prepared[0]:
            raise ValueError("Pending segment lengths must exactly cover current media rows")
        handles: list[SegmentHandle] = []
        try:
            for length in lengths:
                handle = SegmentHandle(length)
                request_id = f"{self.request_id}__segment_{self._next_id}"
                self._next_id += 1
                adapter = self.cache.begin_segment(request_id, length)
                handles.append(handle)
                self._allocations[handle] = _Allocation(adapter, torch.empty(0, device=self.cache.device))
                slots = compute_slot_mapping(self.cache.block_table(adapter), range(length), self.cache.block_size)
                self._allocations[handle].slots = slots.to(device=self.cache.device)
            pending_slots = torch.cat([self._allocations[handle].slots for handle in handles])
        except BaseException:
            for handle in handles:
                self._release(handle)
            raise
        self._pending = _Pending(tuple(handles), pending_slots)
        self._assembled.clear()
        return tuple(handles)

    def stage(self, layer_idx: int) -> None:
        """Write this clean layer's assembled media into unpublished pages."""
        self._check_open()
        self._validate_layer(layer_idx)
        pending = self._pending
        if pending is None:
            raise RuntimeError("begin_commit() must precede stage()")
        if layer_idx in pending.written_layers:
            raise RuntimeError("Segmented KV layer staged more than once")
        if layer_idx not in self._assembled:
            raise RuntimeError("Assemble this clean layer before staging it")
        staging = self._staging()
        if staging.owner is not self or staging.revision != self._revision:
            raise RuntimeError("Segmented staging changed during a clean commit")
        key, value = staging.layers[layer_idx]
        start = self.history_tokens
        end = start + pending.slots.numel()
        self.cache._k_pools[layer_idx].index_copy_(0, pending.slots, key[start:end])
        self.cache._v_pools[layer_idx].index_copy_(0, pending.slots, value[start:end])
        pending.written_layers.add(layer_idx)

    def _validate_retention(self, handles: Sequence[SegmentHandle]) -> tuple[SegmentHandle, ...]:
        retained = tuple(handles)
        if len(retained) != len(set(retained)) or any(handle not in self._allocations for handle in retained):
            raise ValueError("Retention requires unique segment handles owned by this session")
        if len(retained) > self.spec.max_retained_segments:
            raise ValueError("Segmented KV retained segment count exceeds the declared bound")
        if sum(handle.num_tokens for handle in retained) > self.spec.max_retained_tokens:
            raise ValueError("Segmented KV retained token count exceeds the declared bound")
        return retained

    def commit(self, *, retain: Sequence[SegmentHandle]) -> None:
        """Publish all layers together, then release excluded allocations."""
        self._check_open()
        pending = self._pending
        if pending is None:
            raise RuntimeError("No segmented KV commit is active")
        if len(pending.written_layers) != self.cache.num_layers:
            raise RuntimeError("Segmented KV commit is missing transformer layers")
        retained = self._validate_retention(retain)
        self._pending = None
        self._publish_retention(retained)

    def retain(self, handles: Sequence[SegmentHandle]) -> None:
        """Drop/reorder committed segments without changing their valid rows."""
        self._check_open()
        if self._pending is not None:
            raise RuntimeError("Cannot change retention during a segmented KV commit")
        self._publish_retention(self._validate_retention(handles))

    def _publish_retention(self, retained: tuple[SegmentHandle, ...]) -> None:
        keep = set(retained)
        for handle in tuple(self._allocations):
            if handle not in keep:
                self._release(handle)
        if retained != self._retained:
            self._retained = retained
            self._revision += 1
            self._prepared = None
        self._assembled.clear()

    def rollback(self) -> None:
        if self._pending is not None:
            for handle in self._pending.handles:
                self._release(handle)
            self._pending = None
        self._assembled.clear()

    def _release(self, handle: SegmentHandle) -> None:
        self.cache.end_request(self._allocations.pop(handle).adapter)

    def close(self) -> None:
        """Release committed and pending allocations; safe to call twice."""
        if self._closed:
            return
        self.rollback()
        for handle in tuple(self._allocations):
            self._release(handle)
        staging = self._staging()
        if staging.owner is self:
            staging.owner, staging.revision = None, -1
        self._retained = ()
        self._prepared = None
        self._closed = True

    def _validate_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.cache.num_layers:
            raise IndexError(f"Invalid segmented KV layer {layer_idx}")

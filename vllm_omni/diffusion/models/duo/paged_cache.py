# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realization domain policy over vLLM's native BlockPool.

Each semantic group owns independent pages: a short reference/shared/clean tail
is never interpreted as a full page in a concatenated block table. Published
pages are immutable. Appends allocate only new groups, eviction drops references
without copying retained K/V. Cross-session prefix entries pin bounded pages.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import torch
from vllm.v1.core.block_pool import BlockPool

from .contracts import PERSISTENT, RealizationGroup, validate_groups


@dataclass(frozen=True)
class RealizationPageSpan:
    group: RealizationGroup
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class RealizationPagedCache:
    owner: object
    epoch: int
    spans: tuple[RealizationPageSpan, ...]
    lease: object = field(default_factory=object)

    @property
    def groups(self):
        return tuple(s.group for s in self.spans)

    @property
    def length(self):
        return self.groups[-1].end if self.spans else 0


@dataclass
class RealizationPageTransaction:
    prior: RealizationPagedCache
    candidate: RealizationPagedCache
    new_blocks: tuple
    slots: torch.Tensor
    retain: torch.Tensor
    pending: bool = True
    written_layers: set | None = None
    workspace_offset: torch.Tensor | None = None

    def __post_init__(self):
        self.written_layers = set()


@dataclass
class RealizationGraphWriteBinding:
    """Runner-private stable addresses; never an allocator publication token."""

    slots: torch.Tensor
    retain: torch.Tensor
    candidate: object = None
    workspace_offset: torch.Tensor | None = None


class RealizationPagedCachePool:
    """One serialized worker's native block pool and layer-outermost storage.

    The pool owns physical tensors; snapshots describe resource ownership, not
    allocator clones. release/evict consume an owned snapshot. A fork explicitly
    increments native refcounts. All publication requires a completed GPU ack.
    """

    def __init__(
        self,
        *,
        layers,
        heads,
        head_dim,
        num_blocks,
        page_size=32,
        dtype=torch.bfloat16,
        device="cuda",
        prefix_capacity=0,
    ):
        if min(layers, heads, head_dim, num_blocks, page_size) < 1 or num_blocks < 2:
            raise ValueError("Invalid Realization page geometry")
        if prefix_capacity < 0:
            raise ValueError("Invalid prefix capacity")
        self.native = BlockPool(num_blocks, enable_caching=False, hash_block_size=page_size)
        self.page_size = page_size
        self.device = torch.device(device)
        self.storage = torch.empty((layers, 2, num_blocks, page_size, heads, head_dim), dtype=dtype, device=device)
        self.owner = object()
        self.epoch = 1
        self.prefix_capacity = prefix_capacity
        self.prefixes = OrderedDict()
        self._pending = None
        self._closed = False
        self._live = {}

    def empty(self):
        if self._closed:
            raise ValueError("Closed Realization page pool")
        cache = RealizationPagedCache(self.owner, self.epoch, ())
        self._live[cache.lease] = cache
        return cache

    def validate(self, cache):
        if (
            self._closed
            or not isinstance(cache, RealizationPagedCache)
            or cache.owner is not self.owner
            or cache.epoch != self.epoch
        ):
            raise ValueError("Foreign, closed, or stale Realization paged cache")
        if cache.lease not in self._live:
            raise ValueError("Consumed Realization cache lease")
        validate_groups(cache.groups, cache.length)
        for span in cache.spans:
            n = span.group.end - span.group.start
            if span.group.kind not in PERSISTENT or len(span.block_ids) != (n + self.page_size - 1) // self.page_size:
                raise ValueError("Invalid persistent Realization page span")
            if any(b <= 0 or b >= self.storage.shape[2] or self.native.blocks[b].ref_cnt < 1 for b in span.block_ids):
                raise ValueError("Released Realization page span")

    def fork(self, cache):
        self.validate(cache)
        self.native.touch([self.native.blocks[b] for s in cache.spans for b in s.block_ids])
        fork = replace(cache, lease=object())
        self._live[fork.lease] = fork
        return fork

    def release(self, cache):
        self.validate(cache)
        if self._pending is not None and self._pending.prior.lease is cache.lease:
            raise RuntimeError("Cannot release a cache with a pending transaction")
        self._live.pop(cache.lease)
        self.native.free_blocks([self.native.blocks[b] for s in cache.spans for b in s.block_ids])

    def reserve(self, prior, groups):
        self.validate(prior)
        groups = tuple(groups)
        if any(not isinstance(g, RealizationGroup) for g in groups):
            raise ValueError("Realization reservation requires semantic groups")
        validate_groups(groups, groups[-1].end if groups else 0)
        if self._pending is not None:
            raise RuntimeError("Realization worker already has an unacknowledged page transaction")
        new_groups, retain, cursor = [], [], prior.length
        for g in groups:
            if g.kind in PERSISTENT:
                retain.extend(range(g.start, g.end))
                new_groups.append(replace(g, start=cursor, end=cursor + g.end - g.start))
                cursor = new_groups[-1].end
        count = sum((g.end - g.start + self.page_size - 1) // self.page_size for g in new_groups)
        if count > self.native.get_num_free_blocks():
            raise MemoryError("Realization page pool exhausted; no partial reservation")
        blocks = tuple(self.native.get_new_blocks(count))
        if hasattr(self, "_realization_fp8_generations"):
            self._realization_fp8_allocation += 1
            for block in blocks:
                self._realization_fp8_generations[block.block_id] = self._realization_fp8_allocation
        spans, slots, offset = [], [], 0
        for g in new_groups:
            n = g.end - g.start
            nb = (n + self.page_size - 1) // self.page_size
            ids = tuple(b.block_id for b in blocks[offset : offset + nb])
            offset += nb
            spans.append(RealizationPageSpan(g, ids))
            slots.extend(ids[i // self.page_size] * self.page_size + i % self.page_size for i in range(n))
        tx = RealizationPageTransaction(
            prior,
            RealizationPagedCache(self.owner, self.epoch, prior.spans + tuple(spans)),
            blocks,
            torch.tensor(slots, device=self.device, dtype=torch.long),
            torch.tensor(retain, device=self.device, dtype=torch.long),
        )
        self._pending = tx
        return tx

    def write_layer(self, tx, layer, k, v):
        if not isinstance(tx, RealizationGraphWriteBinding) and (tx is not self._pending or not tx.pending):
            raise RuntimeError("Stale Realization page write transaction")
        # Only new rows. Never gather/copy historical K/V. index_copy is graph safe.
        if getattr(self, "_realization_fp8_active", False):
            from .fp8_incremental import write_layer

            write_layer(self, tx, layer, k, v)
        else:
            for side, value in enumerate((k, v)):
                flat = self.storage[layer, side].flatten(0, 1)
                flat.index_copy_(0, tx.slots, value.index_select(0, tx.retain).to(flat.dtype))
        if not isinstance(tx, RealizationGraphWriteBinding):
            tx.written_layers.add(layer)

    def publish(self, tx, *, acknowledged, finite):
        if tx is not self._pending or not tx.pending:
            raise RuntimeError("Stale Realization page publication")
        if not acknowledged or not finite or tx.written_layers != set(range(self.storage.shape[0])):
            self.abort(tx)
            raise RuntimeError("Realization publication requires all-layer finite GPU acknowledgement")
        tx.pending = False
        self._pending = None
        self._live.pop(tx.prior.lease)
        self._live[tx.candidate.lease] = tx.candidate
        return tx.candidate

    def abort(self, tx):
        if tx is not self._pending or not tx.pending:
            raise RuntimeError("Stale Realization transaction abort")
        self.native.free_blocks(tx.new_blocks)
        tx.pending = False
        self._pending = None

    def evict(self, cache, first_block):
        self.validate(cache)
        if self._pending is not None and self._pending.prior.lease is cache.lease:
            raise RuntimeError("Cannot evict a cache with a pending transaction")
        kept, cursor = [], 0
        for span in cache.spans:
            if span.group.kind == "clean" and span.group.block < first_block:
                self.native.free_blocks([self.native.blocks[b] for b in span.block_ids])
            else:
                n = span.group.end - span.group.start
                kept.append(replace(span, group=replace(span.group, start=cursor, end=cursor + n)))
                cursor += n
        self._live.pop(cache.lease)
        result = RealizationPagedCache(self.owner, self.epoch, tuple(kept))
        self._live[result.lease] = result
        return result

    def remember_prefix(self, identity, cache):
        self.validate(cache)
        if any(g.kind not in ("reference", "shared") for g in cache.groups):
            raise ValueError("Only immutable reference/shared prefix may be shared")
        if not self.prefix_capacity:
            return
        if identity in self.prefixes:
            self.release(self.prefixes.pop(identity))
        self.prefixes[identity] = self.fork(cache)
        while len(self.prefixes) > self.prefix_capacity:
            self.release(self.prefixes.popitem(last=False)[1])

    def lookup_prefix(self, identity):
        cache = self.prefixes.get(identity)
        if cache is None:
            return None
        self.prefixes.move_to_end(identity)
        return self.fork(cache)

    def close(self):
        if self._pending is not None:
            self.abort(self._pending)
        for cache in tuple(self._live.values()):
            self.release(cache)
        self.prefixes.clear()
        self._closed = True
        self.epoch += 1


def prefix_identity(sequence, *, namespace):
    """Entire domain identity, never a causal token-page hash.

    Namespace must include tenant, effective checkpoint, dtype/TP layout and
    mask version. This cold admission operation may read tensors on CPU.
    """
    if not namespace or any(g.kind not in ("reference", "shared", "padding") for g in sequence.groups):
        raise ValueError("Prefix identity requires namespace and prefix-only sequence")
    digest = hashlib.sha256(namespace.encode())
    for g in (*sequence.groups, *sequence.refiner_groups):
        digest.update(repr(g).encode())
    for name in sequence.__dataclass_fields__:
        value = getattr(sequence, name)
        if isinstance(value, torch.Tensor):
            value = value.detach().contiguous().cpu()
            digest.update(str((name, tuple(value.shape), value.dtype)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

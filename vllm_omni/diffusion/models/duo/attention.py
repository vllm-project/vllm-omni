# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact paged frame-v5 attention, one joint online softmax per query domain.

Only page/slot metadata is compiled on the host. The GPU reads physical pages
in place; current k/v stay in scratch. No historical tensor gather, dense mask,
or independent-softmax addition. Partial pages are represented by exact slots.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import triton

from .attention_normalize import merge_kv_partial, split_kv_partial
from .contracts import visible


class RealizationPagedAttentionPlan:
    """Persistent metadata addresses. update() never replaces captured tensors."""

    DEFAULT_QUERY_TILE = 32
    MERGE_QUERY_DOMAINS = True
    NATIVE_TP4_DEFAULT = True

    @staticmethod
    def _launches(groups, merge):
        launches, previous = [], None
        for i, g in enumerate(groups):
            identity = (g.sample, g.block, g.kind)
            if merge and previous == identity and launches[-1][2] == g.start:
                launches[-1] = (launches[-1][0], launches[-1][1], g.end)
            else:
                launches.append((i, g.start, g.end))
            previous = identity
        return tuple(launches)

    def __init__(self, query_groups, *, capacity, device, clean_history_chunks=None):
        if capacity < 1:
            raise ValueError("Positive KV capacity required")
        self.query_tile = self.DEFAULT_QUERY_TILE
        if self.query_tile not in (16, 32):
            raise ValueError("Realization attention query tile must be 16 or 32")
        self.ranges = tuple((g.start, g.end) for g in query_groups)
        self.merge_query_domains = self.MERGE_QUERY_DOMAINS
        self.launches = self._launches(query_groups, self.merge_query_domains)
        self.capacity = capacity
        self.clean_history_chunks = clean_history_chunks
        self.slots = torch.zeros((len(query_groups), capacity), device=device, dtype=torch.long)
        self.lengths = torch.zeros(len(query_groups), device=device, dtype=torch.int32)
        self._ready = False
        self._metadata_key = None
        self.metadata_updates = 0
        self._normalize_scratch = {}
        self.native_runtime = None
        self.text_valid_rows = None
        self._native_decided = False

    def sees_past(self, query, key):
        # Limit history only while building the next clean KV. Apply the same
        # visibility to text and media so text cannot relay excluded clean KV.
        if self.clean_history_chunks is not None and key.kind == "clean":
            if key.block < query.block - self.clean_history_chunks:
                return False
        return visible(query, key)

    def update(self, groups, cache, pool, *, text_valid_rows=None):
        pool.validate(cache)
        if text_valid_rows is not None:
            if self.text_valid_rows is None:
                self.text_valid_rows = torch.empty_like(text_valid_rows)
            self.text_valid_rows.copy_(text_valid_rows)
        if tuple((g.start, g.end) for g in groups) != self.ranges:
            raise ValueError("Realization query geometry differs from captured plan")
        if self._launches(groups, self.merge_query_domains) != self.launches:
            raise ValueError("Realization captured query-domain coalescing changed")
        # Published snapshots are frozen; still validate the live lease on
        # EVERY call. Identity alone must never authorize released/reused pages.
        key = (pool, cache, tuple(groups))
        if (
            self._metadata_key is not None
            and self._metadata_key[0] is pool
            and self._metadata_key[1] is cache
            and self._metadata_key[2] == key[2]
        ):
            self._prepare_native(groups, cache, pool)
            return
        if not self._native_decided:
            self._native_decided = True
            if self.NATIVE_TP4_DEFAULT:
                from .native_attention import maybe_create

                self.native_runtime = maybe_create(self, groups, cache, pool)
        if self.native_runtime is not None:
            self.native_runtime.update_slot_metadata(groups, cache, pool)
            self._ready = True
            self._metadata_key = key
            self.metadata_updates += 1
            self._prepare_native(groups, cache, pool)
            return
        maps = []
        for q in groups:
            row = []
            for span in cache.spans:
                if self.sees_past(q, span.group):
                    n = span.group.end - span.group.start
                    row.extend(
                        span.block_ids[i // pool.page_size] * pool.page_size + i % pool.page_size for i in range(n)
                    )
            for k in groups:
                if visible(q, k):
                    row.extend(-(i + 1) for i in range(k.start, k.end))
            if not row or len(row) > self.capacity:
                raise ValueError("All-masked Realization query or KV capacity exceeded")
            maps.append(row)
        # Build on CPU then one bounded metadata copy. No per-layer history copy.
        slots = torch.zeros(self.slots.shape, dtype=torch.long, device="cpu")
        for i, row in enumerate(maps):
            slots[i, : len(row)] = torch.tensor(row, device="cpu")
        self.slots.copy_(slots)
        self.lengths.copy_(torch.tensor([len(r) for r in maps], dtype=torch.int32, device="cpu"))
        self._ready = True
        self._metadata_key = key
        self.metadata_updates += 1
        self._prepare_native(groups, cache, pool)

    def _prepare_native(self, groups, cache, pool):
        if not self._native_decided:
            self._native_decided = True
            if self.NATIVE_TP4_DEFAULT:
                from .native_attention import maybe_create

                self.native_runtime = maybe_create(self, groups, cache, pool)
        if self.text_valid_rows is not None and self.native_runtime is None:
            raise ValueError("Text graph buckets require native TP4/TP8 FA3 attention")
        if self.native_runtime is not None:
            self.native_runtime.prepare(groups, cache, pool)

    def forward(self, q, k, v, pool, layer, scale):
        if not self._ready or q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Paged Realization kernel requires prepared CUDA BF16/FP16 input")
        if q.shape[2] != k.shape[2] or k.shape != v.shape or q.shape[1] % k.shape[1]:
            raise ValueError("Invalid Realization q/k/v head geometry")
        if self.native_runtime is not None:
            return self.native_runtime.forward(q, k, v, layer, scale)
        output = torch.empty_like(q)
        # Keep one partition: the original online-softmax reduction order is
        # unchanged. All layers reuse this scratch in stream order.
        scratch_key = (tuple(q.shape), q.device)
        if scratch_key not in self._normalize_scratch:
            partial = torch.empty((1, *q.shape), device=q.device, dtype=torch.float32)
            maximum = torch.empty((1, q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
            self._normalize_scratch[scratch_key] = (partial, maximum, torch.empty_like(maximum))
        partial, maximum, denominator = self._normalize_scratch[scratch_key]
        # Equal (sample, block, kind) domains have identical visible key slots.
        # Their adjacent rows may share a launch without changing softmax.
        for group, start, end in self.launches:
            # Tiny query domains otherwise spend tensor-core work on masked rows.
            # Keep the wider tile for the video/audio domain.
            query_tile = 16 if end - start <= 16 else self.query_tile
            split_kv_partial[(triton.cdiv(end - start, query_tile), q.shape[1], 1)](
                q,
                k,
                v,
                pool.storage[layer, 0],
                pool.storage[layer, 1],
                self.slots,
                self.lengths,
                partial,
                q.stride(0),
                q.stride(1),
                k.stride(0),
                k.stride(1),
                v.stride(0),
                v.stride(1),
                q.shape[1],
                k.shape[1],
                q.shape[2],
                start,
                end,
                group,
                self.capacity,
                scale,
                query_tile,
                64,
                triton.next_power_of_2(q.shape[2]),
                maximum,
                denominator,
                1,
                q.shape[0],
                num_warps=4,
            )
            merge_kv_partial[(end - start, q.shape[1])](
                partial,
                maximum,
                denominator,
                output,
                q.shape[0],
                q.shape[1],
                q.shape[2],
                start,
                end,
                1,
                1,
                triton.next_power_of_2(q.shape[2]),
                num_warps=4,
            )
        return output

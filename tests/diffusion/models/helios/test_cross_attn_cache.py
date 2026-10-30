# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the cross-attention KV cache liveness guard.

These tests pin the fix for cross-prompt contamination.  The cache is keyed
by a *storage fingerprint* of the source tensor (``data_ptr`` / shape /
stride / dtype / device / ``_version``).  That fingerprint is **not** unique
across requests: once the source tensor is freed, PyTorch's caching
allocator may hand its address to an unrelated tensor of the same shape,
producing a false cache hit -- request B would silently reuse request A's
projected cross-attention K/V.

``SourceTensorLRUCache`` stores a :mod:`weakref` to the source tensor next
to the cached value and verifies the source is still alive on every lookup.
An address reused by a *different* tensor still matches the fingerprint,
but the original source's weakref is dead by then, so the lookup becomes a
miss instead of a silent wrong hit.

The address-reuse scenario is reproduced deterministically with two views
over the same storage: the second view's fingerprint is identical to the
first's (same ``data_ptr`` / shape / dtype), but the first view's weakref
is dead once the view object is freed, so the second lookup must miss.
"""

import gc

import pytest
import torch

from vllm_omni.diffusion.models.helios.cross_attn_cache import SourceTensorLRUCache

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Intra-request reuse (the cache's intended purpose)
# ---------------------------------------------------------------------------


class TestIntraRequestReuse:
    """Within one request the source tensor is alive across denoise steps."""

    def test_same_source_hits(self):
        cache = SourceTensorLRUCache(max_size=2)
        a = torch.randn(4, 8)
        cache.put(a, "A")
        assert cache.get(a) == "A"

    def test_repeated_get_keeps_hitting(self):
        cache = SourceTensorLRUCache(max_size=2)
        a = torch.randn(4, 8)
        cache.put(a, "A")
        for _ in range(50):
            assert cache.get(a) == "A"

    def test_distinct_live_sources_do_not_collide(self):
        cache = SourceTensorLRUCache(max_size=2)
        a = torch.randn(4, 8)
        b = torch.randn(4, 8)
        assert a.data_ptr() != b.data_ptr()
        cache.put(a, "A")
        # b has a different data_ptr -> different key -> miss
        assert cache.get(b) is None
        assert cache.get(a) == "A"


# ---------------------------------------------------------------------------
# The bug: an address reused after the source is freed must NOT false-hit
# ---------------------------------------------------------------------------


class TestAddressReuseAfterFree:
    """The contamination bug: freed source's address reused by a new tensor."""

    def test_freed_source_reused_address_does_not_false_hit(self):
        cache = SourceTensorLRUCache(max_size=2)
        storage = torch.empty(4, 8)
        a = storage[:]  # view; a.data_ptr() == storage.data_ptr()
        cache.put(a, "A")
        assert cache.get(a) == "A"
        ptr_a = a.data_ptr()
        del a
        gc.collect()
        # A new view over the SAME storage -> identical fingerprint
        # (data_ptr / shape / stride / dtype / device / version=0).
        b = storage[:]
        assert b.data_ptr() == ptr_a
        # Without the weakref guard this would falsely return "A"
        # (cross-prompt contamination).  With the guard the dead weakref
        # turns the fingerprint match into a miss.
        assert cache.get(b) is None
        assert len(cache) == 0  # stale entry evicted on miss

    def test_recompute_after_miss_stores_new_source(self):
        cache = SourceTensorLRUCache(max_size=2)
        storage = torch.empty(4, 8)
        a = storage[:]
        cache.put(a, "A")
        del a
        gc.collect()
        b = storage[:]  # reuses address
        assert cache.get(b) is None
        cache.put(b, "B")  # the recompute path in the transformer
        assert cache.get(b) == "B"
        assert len(cache) == 1

    def test_two_consecutive_requests_no_contamination(self):
        # End-to-end simulation of the serving sequence that triggered the
        # bug: request A populates the cache, A is freed, request B reuses
        # A's address.  B must never observe A's cached value.
        cache = SourceTensorLRUCache(max_size=2)
        storage = torch.empty(2, 16)

        a = storage[:]
        cache.put(a, "A_projected_kv")
        del a
        gc.collect()

        b = storage[:]
        assert cache.get(b) is None  # B does not inherit A's projection
        cache.put(b, "B_projected_kv")
        assert cache.get(b) == "B_projected_kv"

    def test_freed_then_distinct_address_misses(self):
        # When the new tensor gets a DIFFERENT address, the key differs and
        # it trivially misses.  The weakref guard must not break this path.
        cache = SourceTensorLRUCache(max_size=2)
        a = torch.randn(4, 8)
        cache.put(a, "A")
        del a
        gc.collect()
        b = torch.randn(4, 8)
        assert cache.get(b) is None


# ---------------------------------------------------------------------------
# LRU semantics (unchanged by the fix, but must still hold)
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_lru_evicts_oldest(self):
        cache = SourceTensorLRUCache(max_size=2)
        a, b, c = (torch.randn(1) for _ in range(3))
        cache.put(a, "A")
        cache.put(b, "B")
        cache.put(c, "C")
        assert cache.get(a) is None  # evicted
        assert cache.get(b) == "B"
        assert cache.get(c) == "C"

    def test_get_renews_recency(self):
        cache = SourceTensorLRUCache(max_size=2)
        a, b, c = (torch.randn(1) for _ in range(3))
        cache.put(a, "A")
        cache.put(b, "B")
        assert cache.get(a) == "A"  # a is now most-recent
        cache.put(c, "C")  # evicts b (oldest)
        assert cache.get(b) is None
        assert cache.get(a) == "A"
        assert cache.get(c) == "C"


class TestPutOverwrite:
    def test_same_source_overwrites_value(self):
        cache = SourceTensorLRUCache(max_size=2)
        a = torch.randn(4, 8)
        cache.put(a, "A1")
        cache.put(a, "A2")
        assert cache.get(a) == "A2"
        assert len(cache) == 1


class TestClear:
    def test_clear_empties_cache(self):
        cache = SourceTensorLRUCache(max_size=2)
        a, b = torch.randn(1), torch.randn(1)
        cache.put(a, "A")
        cache.put(b, "B")
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0
        assert cache.get(a) is None
        assert cache.get(b) is None


class TestProactiveStaleEviction:
    def test_put_evicts_dead_entries(self):
        cache = SourceTensorLRUCache(max_size=4)
        a = torch.randn(4, 8)
        cache.put(a, "A")
        del a
        gc.collect()
        b = torch.randn(4, 8)
        cache.put(b, "B")  # proactively drops the dead "a" entry
        assert len(cache) == 1
        assert cache.get(b) == "B"

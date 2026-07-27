# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NaiveCache merge/split logic used in batched CFG."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_LAYERS = 2
NUM_KV_HEADS = 4
HEAD_DIM = 8


def _make_cache(num_layers, seq_len, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, seed=0):
    """Create a NaiveCache with deterministic random data. seq_len=0 returns an empty cache."""
    gen = torch.Generator().manual_seed(seed)
    cache = NaiveCache(num_layers)
    if seq_len == 0:
        return cache
    for layer in range(num_layers):
        cache.key_cache[layer] = torch.randn(seq_len, num_kv_heads, head_dim, generator=gen)
        cache.value_cache[layer] = torch.randn(seq_len, num_kv_heads, head_dim, generator=gen)
    return cache


def test_init_creates_none_entries():
    """Ensure the NaiveCache is initialized with None values per layer."""
    cache = NaiveCache(NUM_LAYERS)
    assert cache.num_layers == NUM_LAYERS
    for layer in range(NUM_LAYERS):
        assert cache.key_cache[layer] is None
        assert cache.value_cache[layer] is None


@pytest.mark.parametrize("seq_len", [0, 10])
def test_seq_lens_empty(seq_len):
    """Ensure that by default, we have 0 seq lens."""
    cache = _make_cache(NUM_LAYERS, seq_len=seq_len)
    assert cache.seq_lens == seq_len
    assert cache.num_layers == NUM_LAYERS


### Merge tests
def test_merge_two_equal_length():
    """Ensure that we can merge two NaiveCaches that are identically shaped."""
    c0 = _make_cache(NUM_LAYERS, seq_len=5, seed=0)
    c1 = _make_cache(NUM_LAYERS, seq_len=5, seed=1)
    merged = NaiveCache.merge([c0, c1])

    assert merged.key_values_lens == [5, 5]
    for layer in range(NUM_LAYERS):
        assert merged.key_cache[layer].shape[0] == 10
        # the merged cache will just have keys and values per layer concatenated
        assert torch.equal(merged.key_cache[layer][:5], c0.key_cache[layer])
        assert torch.equal(merged.key_cache[layer][5:], c1.key_cache[layer])
        assert torch.equal(merged.value_cache[layer][:5], c0.value_cache[layer])
        assert torch.equal(merged.value_cache[layer][5:], c1.value_cache[layer])


def test_merge_three_zero_len_cache():
    """Ensure we handle zero length cache correctly in merge."""
    # NOTE: This is relevant for text_cfg in Bagel, which has a len of 0 by default
    gen_cache = _make_cache(NUM_LAYERS, seq_len=10, seed=0)
    text_cfg_cache = _make_cache(NUM_LAYERS, seq_len=0)
    img_cfg_cache = _make_cache(NUM_LAYERS, seq_len=7, seed=2)
    merged = NaiveCache.merge([gen_cache, text_cfg_cache, img_cfg_cache])

    assert merged.key_values_lens == [10, 0, 7]
    for layer in range(NUM_LAYERS):
        assert merged.key_cache[layer].shape[0] == 17
        assert torch.equal(merged.key_cache[layer][:10], gen_cache.key_cache[layer])
        assert torch.equal(merged.key_cache[layer][10:], img_cfg_cache.key_cache[layer])


def test_merge_all_empty():
    """Ensure that merging empty caches is well defined."""
    caches = [_make_cache(NUM_LAYERS, seq_len=0) for _ in range(3)]
    merged = NaiveCache.merge(caches)

    assert merged.key_values_lens == [0, 0, 0]
    for layer in range(NUM_LAYERS):
        assert merged.key_cache[layer] is None
        assert merged.value_cache[layer] is None


def test_merge_single_cache():
    """Ensure merging one cache returns an identical cache."""
    c = _make_cache(NUM_LAYERS, seq_len=8, seed=42)
    merged = NaiveCache.merge([c])

    assert merged.key_values_lens == [8]
    for layer in range(NUM_LAYERS):
        assert torch.equal(merged.key_cache[layer], c.key_cache[layer])


def test_merge_preserves_dtype():
    """Ensure merging doesn't modify dtypes."""
    cache = NaiveCache(NUM_LAYERS)
    for layer in range(NUM_LAYERS):
        cache.key_cache[layer] = torch.randn(5, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        cache.value_cache[layer] = torch.randn(5, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)

    merged = NaiveCache.merge([cache])
    assert merged.key_cache[0].dtype == torch.bfloat16
    assert merged.value_cache[0].dtype == torch.bfloat16


### Split tests
def test_split_all_nonzero():
    """Ensure NaiveCache splits in the simple case (all lens nonzero)."""
    t = torch.randn(15, NUM_KV_HEADS, HEAD_DIM)
    parts = NaiveCache.split_with_zeros(t, [5, 4, 6])

    assert len(parts) == 3
    assert all(p is not None for p in parts)
    assert parts[0].shape[0] == 5
    assert parts[1].shape[0] == 4
    assert parts[2].shape[0] == 6
    assert torch.equal(torch.cat(parts), t)


def test_split_with_zero():
    """Ensure NaiveCache split handles zero length correctly (used in Bagel)."""
    t = torch.randn(17, NUM_KV_HEADS, HEAD_DIM)
    parts = NaiveCache.split_with_zeros(t, [10, 0, 7])

    assert parts[0].shape[0] == 10
    assert parts[1] is None
    assert parts[2].shape[0] == 7
    assert torch.equal(torch.cat([parts[0], parts[2]]), t)


def test_split_wrong_sum_raises():
    """Ensure NaiveCache raises if splits don't match the sum of dims on axis 0."""
    t = torch.randn(10, NUM_KV_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="dim 0"):
        NaiveCache.split_with_zeros(t, [5, 3])


def test_split_negative_length_raises():
    """Ensure NaiveCache raises if splits have any negative values."""
    t = torch.randn(10, NUM_KV_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        NaiveCache.split_with_zeros(t, [5, -1, 6])


def test_split_preserves_dtype():
    """Ensure NaiveCache split preserves dtype."""
    t = torch.randn(10, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    parts = NaiveCache.split_with_zeros(t, [4, 6])
    assert parts[0].dtype == torch.bfloat16
    assert parts[1].dtype == torch.bfloat16


### from_object tests (for kv cache transfer)
def test_from_object_passthrough():
    """Ensure a NaiveCache input is returned as is."""
    cache = _make_cache(NUM_LAYERS, seq_len=5)
    assert NaiveCache.from_object(cache) is cache


def test_from_object_converts_simple_namespace():
    """Ensure SimpleNamespace with list-based caches converts to NaiveCache."""
    keys = [torch.randn(5, NUM_KV_HEADS, HEAD_DIM) for _ in range(NUM_LAYERS)]
    values = [torch.randn(5, NUM_KV_HEADS, HEAD_DIM) for _ in range(NUM_LAYERS)]
    ns = SimpleNamespace(key_cache=keys, value_cache=values)

    cache = NaiveCache.from_object(ns)

    assert isinstance(cache, NaiveCache)
    assert cache.num_layers == NUM_LAYERS
    for i in range(NUM_LAYERS):
        assert torch.equal(cache.key_cache[i], keys[i])
        assert torch.equal(cache.value_cache[i], values[i])


def test_from_object_mismatched_lengths_raises():
    """Ensure mismatched key/value cache lengths raise due to strict=True in zip."""
    keys = [torch.randn(5, NUM_KV_HEADS, HEAD_DIM) for _ in range(2)]
    values = [torch.randn(5, NUM_KV_HEADS, HEAD_DIM) for _ in range(3)]
    ns = SimpleNamespace(key_cache=keys, value_cache=values)

    with pytest.raises(ValueError):
        NaiveCache.from_object(ns)


### End to end test for split / merge
def test_round_trip_two_populated():
    """Roundtrip test for merging and resplitting two simple caches."""
    c0 = _make_cache(NUM_LAYERS, seq_len=5, seed=0)
    c1 = _make_cache(NUM_LAYERS, seq_len=8, seed=1)
    merged = NaiveCache.merge([c0, c1])

    for layer in range(NUM_LAYERS):
        k_parts = NaiveCache.split_with_zeros(merged.key_cache[layer], merged.key_values_lens)
        v_parts = NaiveCache.split_with_zeros(merged.value_cache[layer], merged.key_values_lens)
        assert torch.equal(k_parts[0], c0.key_cache[layer])
        assert torch.equal(k_parts[1], c1.key_cache[layer])
        assert torch.equal(v_parts[0], c0.value_cache[layer])
        assert torch.equal(v_parts[1], c1.value_cache[layer])


def test_round_trip_three_branches_with_zero_cfg():
    """Roundtrip test with a zero entry (i.e., same as Bagel's gen/text_cfg/img_cfg case)."""
    gen_cache = _make_cache(NUM_LAYERS, seq_len=10, seed=0)
    text_cfg_cache = _make_cache(NUM_LAYERS, seq_len=0)
    img_cfg_cache = _make_cache(NUM_LAYERS, seq_len=7, seed=2)
    merged = NaiveCache.merge([gen_cache, text_cfg_cache, img_cfg_cache])

    for layer in range(NUM_LAYERS):
        k_parts = NaiveCache.split_with_zeros(merged.key_cache[layer], merged.key_values_lens)
        v_parts = NaiveCache.split_with_zeros(merged.value_cache[layer], merged.key_values_lens)
        assert torch.equal(k_parts[0], gen_cache.key_cache[layer])
        assert k_parts[1] is None
        assert torch.equal(k_parts[2], img_cfg_cache.key_cache[layer])
        assert torch.equal(v_parts[0], gen_cache.value_cache[layer])
        assert v_parts[1] is None
        assert torch.equal(v_parts[2], img_cfg_cache.value_cache[layer])

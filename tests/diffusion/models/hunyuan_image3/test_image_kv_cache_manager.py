# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ImageKVCacheManager.

Covers: cache → reuse flow, AR KV injection, CFG (sequential & parallel), SP, cross-request isolation.
"""

from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TRANSFORMER_MODULE = "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer"

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
IMAGE_TOKEN_LEN = 8
SCALING = 1.0 / math.sqrt(HEAD_DIM)


# ============================================================
# Mocks + helpers
# ============================================================


class MockAttention(nn.Module):
    def __init__(self, num_heads, head_size, causal=False, softmax_scale=None, num_kv_heads=None, **kwargs):
        super().__init__()

    def forward(self, query, key, value, attn_metadata=None, **kwargs):
        return query


@contextmanager
def patched_mgr_env(sp_size=1):
    target = _TRANSFORMER_MODULE
    patches = [
        patch(f"{target}.get_sequence_parallel_world_size", return_value=sp_size),
        patch(f"{target}.get_sequence_parallel_rank", return_value=0),
        patch(f"{target}.Attention", MockAttention),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


def _make_cache_mgr(image_token_len=IMAGE_TOKEN_LEN, sp_size=1):
    with patched_mgr_env(sp_size=sp_size):
        from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
            ImageKVCacheManager,
        )

        mgr = ImageKVCacheManager(
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            scaling=SCALING,
            image_token_len=image_token_len,
        )
    return mgr


def _make_known_kv(num_tokens, base=0.0):
    """Create key/value with known values. Token i has all elements = base+i / base+i+0.5."""
    k = torch.full((num_tokens, NUM_KV_HEADS, HEAD_DIM), 0.0)
    v = torch.full((num_tokens, NUM_KV_HEADS, HEAD_DIM), 0.0)
    for i in range(num_tokens):
        k[i] = base + i
        v[i] = base + i + 0.5
    return k, v


def _call_mgr(
    mgr,
    bs,
    q_len,
    seq_len,
    key_flat,
    value_flat,
    first_step=False,
    uncond_cfg_prefill=False,
    num_image_tokens=IMAGE_TOKEN_LEN,
    shard_image_size=None,
):
    query = torch.randn(bs * q_len, NUM_HEADS, HEAD_DIM)
    attn_mask = torch.zeros(bs, 1, seq_len, seq_len)
    mgr(
        query,
        key_flat,
        value_flat,
        attn_mask,
        query_lens=[q_len] * bs,
        seq_lens=[seq_len] * bs,
        first_step=first_step,
        uncond_cfg_prefill=uncond_cfg_prefill,
        num_image_tokens=num_image_tokens,
        shard_image_size=shard_image_size,
    )


# ============================================================
# Test 1: No AR KV — basic cache → reuse
# ============================================================


@pytest.mark.parametrize("bs", [1, 2])
def test_no_ar_kv(bs):
    """
    No AR KV injected. Tests the basic first_step cache → update reuse path.

    Sequence layout per batch on first_step (q_len=12, IMAGE_TOKEN_LEN=8):
        [prompt(3) | image(8) | eoi(1)]
        image_size = IMAGE_TOKEN_LEN + 1 = 9
        cached_prompt_len = seq_len - image_size = 12 - 9 = 3

    After first_step:
        image_kv_cache_map stores prompt tokens [0:3] for each batch (total 3*bs tokens flat).

    Update step (q_len=IMAGE_TOKEN_LEN=8, seq_len = cached_prompt(3) + eoi(1) + image(8) = 12):
        _reuse_prompt_kv produces [cached_prompt(3) | new_image(8) | zero_eoi(1)] per batch.
    """
    mgr = _make_cache_mgr()
    assert mgr.image_kv_cache_map is None

    # --- first_step ---
    q_len = 12
    k_flat, v_flat = _make_known_kv(bs * q_len, base=1.0)

    _call_mgr(mgr, bs, q_len=q_len, seq_len=q_len, key_flat=k_flat, value_flat=v_flat, first_step=True)

    cached_key, cached_value = mgr.image_kv_cache_map
    # 3 prompt tokens cached per batch
    assert cached_key.shape[0] == 3 * bs
    for b in range(bs):
        flat_offset = b * q_len
        cache_offset = b * 3
        assert torch.allclose(cached_key[cache_offset : cache_offset + 3], k_flat[flat_offset : flat_offset + 3])
        assert torch.allclose(cached_value[cache_offset : cache_offset + 3], v_flat[flat_offset : flat_offset + 3])

    # --- update step ---
    img_q_len = IMAGE_TOKEN_LEN
    update_seq_len = 3 + 1 + img_q_len  # cached_prompt(3) + eoi(1) + image(8) = 12
    new_img_k, new_img_v = _make_known_kv(bs * img_q_len, base=50.0)

    key_input = new_img_k.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)
    val_input = new_img_v.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)
    result_k, result_v = mgr._reuse_prompt_kv(key_input, val_input, update_seq_len, bs=bs)

    assert result_k.shape == (bs, update_seq_len, NUM_KV_HEADS, HEAD_DIM)
    for b in range(bs):
        cache_offset = b * 3
        img_offset = b * img_q_len
        # Cached prompt preserved
        assert torch.allclose(result_k[b, :3], cached_key[cache_offset : cache_offset + 3])
        # New image tokens
        assert torch.allclose(result_k[b, 3 : 3 + img_q_len], new_img_k[img_offset : img_offset + img_q_len])
        # Zero eoi at the end
        assert torch.allclose(result_k[b, -1], torch.zeros(NUM_KV_HEADS, HEAD_DIM))


# ============================================================
# Test 2: AR KV, no CFG
# ============================================================


@pytest.mark.parametrize("sp_size", [1, 2])
def test_ar_kv_no_cfg(sp_size):
    """
    AR KV injected, bs=1, no CFG. Tests AR prefix prepend + cache + reuse.

    sp_size=1:
        first_step input: q_len=12, ar_len=5, seq_len=17
        Sequence layout: [ar(5) | prompt(3) | image(8) | eoi(1)]
        image_size = IMAGE_TOKEN_LEN + 1 = 9
        cached_prompt_len = seq_len - image_size = 17 - 9 = 8 (= ar(5) + prompt(3))

        Update step: q_len=8, seq_len = cached_prompt(8) + eoi(1) + image(8) = 17
        Result: [cached(8) | new_image(8) | zero_eoi(1)]

    sp_size=2:
        shard_image_size = 4 (passed externally, simulating SP sharding)
        cached_prompt_len = seq_len - shard_image_size = 17 - 4 = 13 (= ar(5) + prompt(8))
        Note: with SP, more of the sequence is treated as "prompt" because
        image tokens are sharded across ranks.

        Update step: q_len=4 (shard_image_size), seq_len = cached_prompt(13) + shard_image(4) = 17
        SP path returns only cached prompt [bs, cached_prompt_len, ...] (no eoi concat).

    Both sp_size cases test _cache_prompt_kv and _reuse_prompt_kv directly to avoid
    needing full SP infrastructure mocks in __call__.
    """
    mgr = _make_cache_mgr(sp_size=sp_size)
    ar_len = 5
    ar_k, ar_v = _make_known_kv(ar_len, base=100.0)
    mgr._injected_ar_kv = [(ar_k.clone(), ar_v.clone())]

    # --- first_step: call _cache_prompt_kv directly ---
    bs, q_len = 1, 12
    seq_len = q_len + ar_len  # 17
    k_raw, v_raw = _make_known_kv(q_len, base=1.0)
    key_4d = k_raw.reshape(1, q_len, NUM_KV_HEADS, HEAD_DIM)
    val_4d = v_raw.reshape(1, q_len, NUM_KV_HEADS, HEAD_DIM)

    shard_image_size = 4 if sp_size > 1 else None
    mgr.image_kv_cache_map = None
    mgr._cache_prompt_kv(key_4d, val_4d, seq_len, shard_image_size)

    cached_key, cached_value = mgr.image_kv_cache_map
    if sp_size == 1:
        # cached = ar(5) + prompt(3) = 8
        expected_cached_len = 8
    else:
        # cached = seq_len - shard_image_size = 17 - 4 = 13
        expected_cached_len = seq_len - shard_image_size

    assert cached_key.shape[0] == expected_cached_len
    # AR KV always at the front
    assert torch.allclose(cached_key[:ar_len], ar_k)
    assert torch.allclose(cached_value[:ar_len], ar_v)
    # Prompt tokens follow AR
    prompt_cached = expected_cached_len - ar_len
    assert torch.allclose(cached_key[ar_len : ar_len + prompt_cached], k_raw[:prompt_cached])
    # AR KV consumed
    assert mgr._injected_ar_kv is None

    # --- update step: call _reuse_prompt_kv directly ---
    if sp_size == 1:
        img_q_len = IMAGE_TOKEN_LEN
        update_seq_len = expected_cached_len + 1 + img_q_len  # 8 + 1 + 8 = 17
        new_img_k, new_img_v = _make_known_kv(img_q_len, base=50.0)

        key_input = new_img_k.reshape(1, img_q_len, NUM_KV_HEADS, HEAD_DIM)
        val_input = new_img_v.reshape(1, img_q_len, NUM_KV_HEADS, HEAD_DIM)
        result_k, result_v = mgr._reuse_prompt_kv(key_input, val_input, update_seq_len, bs=1)

        assert result_k.shape == (1, update_seq_len, NUM_KV_HEADS, HEAD_DIM)
        # AR + prompt preserved
        assert torch.allclose(result_k[0, :ar_len], ar_k)
        assert torch.allclose(result_k[0, ar_len : ar_len + prompt_cached], k_raw[:prompt_cached])
        # New image tokens
        assert torch.allclose(result_k[0, expected_cached_len : expected_cached_len + img_q_len], new_img_k)
        # Zero eoi
        assert torch.allclose(result_k[0, -1], torch.zeros(NUM_KV_HEADS, HEAD_DIM))
    else:
        # SP path: _reuse_prompt_kv returns only cached prompt (no image concat)
        img_q_len = shard_image_size  # 4
        update_seq_len = expected_cached_len + img_q_len  # 13 + 4 = 17
        new_img_k, new_img_v = _make_known_kv(img_q_len, base=50.0)

        key_input = new_img_k.reshape(1, img_q_len, NUM_KV_HEADS, HEAD_DIM)
        val_input = new_img_v.reshape(1, img_q_len, NUM_KV_HEADS, HEAD_DIM)
        result_k, result_v = mgr._reuse_prompt_kv(
            key_input,
            val_input,
            update_seq_len,
            bs=bs,
            shard_image_size=img_q_len,
        )

        # SP returns only the cached prompt portion
        assert result_k.shape == (1, expected_cached_len, NUM_KV_HEADS, HEAD_DIM)
        assert torch.allclose(result_k[0, :ar_len], ar_k)
        assert torch.allclose(result_k[0, ar_len : ar_len + prompt_cached], k_raw[:prompt_cached])


# ============================================================
# Test 3: AR KV + CFG (sequential & parallel)
# ============================================================


@pytest.mark.parametrize("cfg_parallel,bs", [(False, 2), (True, 1)])
def test_ar_kv_with_cfg(cfg_parallel, bs):
    """
    AR KV + CFG. Tests uncond_cfg_prefill → first_step → update.

    Common setup:
        positive_reuse_len = 10, negative_reuse_len = 6, neg_uncond_cfg_q_len = 4
        AR KV: 10 tokens (base=100)

    Sequential CFG (cfg_parallel=False, bs=2):
        uncond_cfg_prefill (bs=1):
            Builds neg AR KV = [shared_prefix(6) from pos_ar | neg_prefill_tokens(4)]
            → _injected_ar_kv becomes [(pos_ar(10), pos_av(10)), (neg_k(10), neg_v(10))]

        first_step (bs=2, q_len=12, seq_len=22):
            Batch 0 (pos): [pos_ar(10) | prompt(3) | image(8) | eoi(1)]
            Batch 1 (neg): [neg_ar(10) | prompt(3) | image(8) | eoi(1)]
            cached_prompt_len per batch = 22 - 9 = 13
            Total cached = 13 * 2 = 26

        Update (bs=2, seq_len = 13 + 1 + 8 = 22):
            Result per batch: [cached(13) | new_image(8) | zero_eoi(1)]

    CFG Parallel (cfg_parallel=True, bs=1):
        This rank handles only the negative branch.
        uncond_cfg_prefill (bs=1):
            Same as above: builds neg AR KV.
            Then we keep only the negative entry (simulating _keep_negative_kv_only).
            → _injected_ar_kv = [(neg_k(10), neg_v(10))]

        first_step (bs=1, q_len=12, seq_len=22):
            [neg_ar(10) | prompt(3) | image(8) | eoi(1)]
            cached_prompt_len = 22 - 9 = 13

        Update (bs=1, seq_len = 13 + 1 + 8 = 22):
            Result: [cached(13) | new_image(8) | zero_eoi(1)]
    """
    positive_reuse_len = 10
    negative_reuse_len = 6
    neg_uncond_cfg_q_len = positive_reuse_len - negative_reuse_len  # 4

    mgr = _make_cache_mgr()
    pos_ar_k, pos_ar_v = _make_known_kv(positive_reuse_len, base=100.0)
    mgr._injected_ar_kv = [(pos_ar_k.clone(), pos_ar_v.clone())]

    # --- uncond_cfg_prefill ---
    neg_k, neg_v = _make_known_kv(neg_uncond_cfg_q_len, base=200.0)
    prefill_seq_len = negative_reuse_len + neg_uncond_cfg_q_len  # 6 + 4 = 10

    _call_mgr(
        mgr,
        bs=1,
        q_len=neg_uncond_cfg_q_len,
        seq_len=prefill_seq_len,
        key_flat=neg_k,
        value_flat=neg_v,
        first_step=True,
        uncond_cfg_prefill=True,
        num_image_tokens=0,
    )

    # After prefill: _injected_ar_kv = [(pos), (neg)]
    assert len(mgr._injected_ar_kv) == 2
    neg_ar_k, neg_ar_v = mgr._injected_ar_kv[1]
    # neg AR KV = [shared_prefix(6) from pos | neg_prefill(4)], total 10
    assert neg_ar_k.shape[0] == positive_reuse_len
    assert torch.allclose(neg_ar_k[:negative_reuse_len], pos_ar_k[:negative_reuse_len])
    assert torch.allclose(neg_ar_k[negative_reuse_len:], neg_k)

    # --- simulate cfg_parallel: keep only negative ---
    if cfg_parallel:
        mgr._injected_ar_kv = [mgr._injected_ar_kv[1]]

    # --- first_step ---
    q_len = 12
    seq_len = q_len + positive_reuse_len  # 22
    k_flat, v_flat = _make_known_kv(bs * q_len, base=1.0)

    _call_mgr(mgr, bs, q_len=q_len, seq_len=seq_len, key_flat=k_flat, value_flat=v_flat, first_step=True)

    cached_key, cached_value = mgr.image_kv_cache_map
    cached_prompt_len_per_batch = seq_len - (IMAGE_TOKEN_LEN + 1)  # 22 - 9 = 13
    assert cached_key.shape[0] == cached_prompt_len_per_batch * bs
    assert mgr._injected_ar_kv is None

    if not cfg_parallel:
        # bs=2: batch 0 = pos, batch 1 = neg
        # Batch 0: pos_ar(10) + prompt(3)
        assert torch.allclose(cached_key[:positive_reuse_len], pos_ar_k)
        assert torch.allclose(cached_key[positive_reuse_len:13], k_flat[:3])
        # Batch 1: neg_ar(10) + prompt(3)
        assert torch.allclose(cached_key[13:23], neg_ar_k)
        assert torch.allclose(cached_key[23:26], k_flat[q_len : q_len + 3])
    else:
        # bs=1: only neg branch
        assert torch.allclose(cached_key[:positive_reuse_len], neg_ar_k)
        assert torch.allclose(cached_key[positive_reuse_len:13], k_flat[:3])

    # --- update step ---
    img_q_len = IMAGE_TOKEN_LEN
    update_seq_len = cached_prompt_len_per_batch + 1 + img_q_len  # 13 + 1 + 8 = 22
    new_img_k, new_img_v = _make_known_kv(bs * img_q_len, base=50.0)

    key_input = new_img_k.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)
    val_input = new_img_v.reshape(bs, img_q_len, NUM_KV_HEADS, HEAD_DIM)
    result_k, result_v = mgr._reuse_prompt_kv(key_input, val_input, update_seq_len, bs=bs)

    assert result_k.shape == (bs, update_seq_len, NUM_KV_HEADS, HEAD_DIM)
    for b in range(bs):
        cache_offset = b * cached_prompt_len_per_batch
        # Cached prompt preserved
        assert torch.allclose(
            result_k[b, :cached_prompt_len_per_batch],
            cached_key[cache_offset : cache_offset + cached_prompt_len_per_batch],
        )
        # New image tokens
        img_offset = b * img_q_len
        assert torch.allclose(
            result_k[b, cached_prompt_len_per_batch : cached_prompt_len_per_batch + img_q_len],
            new_img_k[img_offset : img_offset + img_q_len],
        )
        # Zero eoi
        assert torch.allclose(result_k[b, -1], torch.zeros(NUM_KV_HEADS, HEAD_DIM))


# ============================================================
# Test 4: Cross-request isolation
# ============================================================


def test_cross_request_isolation():
    """
    Verify leftover image_kv_cache_map from a previous request is NOT treated as AR KV.

    Setup: mgr has stale cache from a prior request (9 tokens, base=999).
    New request: first_step with q_len=12, no AR KV.

    Expected: stale cache is overwritten. New cache = prompt tokens from current request.
    The stale values (999.x) must NOT appear in the new cache.
    """
    mgr = _make_cache_mgr()

    # Simulate leftover from previous request
    leftover_k, leftover_v = _make_known_kv(9, base=999.0)
    mgr.image_kv_cache_map = (leftover_k, leftover_v)
    assert mgr._injected_ar_kv is None

    # New request first_step
    bs, seq_len = 1, 12
    k_flat, v_flat = _make_known_kv(bs * seq_len, base=1.0)

    _call_mgr(mgr, bs, q_len=seq_len, seq_len=seq_len, key_flat=k_flat, value_flat=v_flat, first_step=True)

    cached_key, cached_value = mgr.image_kv_cache_map
    # cached_prompt_len = 12 - 9 = 3
    assert cached_key.shape[0] == 3
    # Must be from current request, not stale
    assert torch.allclose(cached_key[:3], k_flat[:3])
    assert torch.allclose(cached_value[:3], v_flat[:3])
    # Stale values must not be present
    assert not torch.any(cached_key >= 999.0)

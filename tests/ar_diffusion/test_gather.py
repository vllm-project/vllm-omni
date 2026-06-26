# SPDX-License-Identifier: Apache-2.0
"""Tests for pool write + window gather (Step 4)."""

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionKVCache,
    ARDiffusionKVConfig,
    allocate_kv_pool,
    pool_gather_window,
    pool_write_chunk,
)

BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64


def test_pool_write_gather_roundtrip():
    """Write a single chunk to the pool then gather — K/V comes back identical."""
    num_blocks = 4
    kp, vp = allocate_kv_pool(num_blocks, BLOCK, 1, N_HEADS, HEAD_DIM, torch.float32, "cpu")
    new_k = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM)
    new_v = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM)
    slot_mapping = torch.arange(num_blocks * BLOCK)  # identity mapping for the test
    slot_mapping = slot_mapping[:BLOCK]

    pool_write_chunk(kp[0], vp[0], new_k, new_v, slot_mapping)
    window = pool_gather_window(kp[0], vp[0], window_block_ids=[0], block_size=BLOCK, max_attention_size=BLOCK)

    assert window.shape == (2, 1, BLOCK, N_HEADS, HEAD_DIM)
    assert torch.allclose(window[0, 0], new_k[0])
    assert torch.allclose(window[1, 0], new_v[0])


def test_pool_write_gather_multiple_blocks():
    """Write 2 blocks (distinct positions) and gather both from the window."""
    num_blocks = 8
    kp, vp = allocate_kv_pool(num_blocks, BLOCK, 1, N_HEADS, HEAD_DIM, torch.float32, "cpu")

    # Write block 0
    k0 = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM)
    v0 = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM)
    s0 = torch.arange(BLOCK)  # block 0 starts at slot 0
    pool_write_chunk(kp[0], vp[0], k0, v0, s0)

    # Write block 2 (holes OK, block id doesn't need to be contiguous in slot space)
    k2 = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM)
    v2 = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM)
    s2 = 2 * BLOCK + torch.arange(BLOCK)  # block 2
    pool_write_chunk(kp[0], vp[0], k2, v2, s2)

    # Gather blocks 0 and 2 — they appear in table order.
    window = pool_gather_window(kp[0], vp[0], window_block_ids=[0, 2], block_size=BLOCK, max_attention_size=2 * BLOCK)
    assert window.shape == (2, 1, 2 * BLOCK, N_HEADS, HEAD_DIM)
    assert torch.allclose(window[0, 0, :BLOCK], k0[0])
    assert torch.allclose(window[1, 0, :BLOCK], v0[0])
    assert torch.allclose(window[0, 0, BLOCK:], k2[0])
    assert torch.allclose(window[1, 0, BLOCK:], v2[0])


def test_pool_write_gather_window_trim():
    """Gather with max_attention_size smaller than the resident window trims to tail."""
    num_blocks = 4
    kp, vp = allocate_kv_pool(num_blocks, BLOCK, 1, N_HEADS, HEAD_DIM, torch.float32, "cpu")
    k = torch.randn(1, 2 * BLOCK, N_HEADS, HEAD_DIM)
    v = torch.randn(1, 2 * BLOCK, N_HEADS, HEAD_DIM)
    slots = torch.cat([torch.arange(0 * BLOCK, 1 * BLOCK), torch.arange(1 * BLOCK, 2 * BLOCK)])
    pool_write_chunk(kp[0], vp[0], k[:, :BLOCK], v[:, :BLOCK], slots[:BLOCK])  # block 0
    pool_write_chunk(kp[0], vp[0], k[:, BLOCK:], v[:, BLOCK:], slots[BLOCK:])  # block 1
    # max_attention = BLOCK — only last block.
    window = pool_gather_window(kp[0], vp[0], window_block_ids=[0, 1], block_size=BLOCK, max_attention_size=BLOCK)
    assert window.shape == (2, 1, BLOCK, N_HEADS, HEAD_DIM)
    assert torch.allclose(window[0, 0], k[0, BLOCK:])
    assert torch.allclose(window[1, 0], v[0, BLOCK:])


def test_bde_kv_cache_write_gather_integration():
    """ARDiffusionKVCache.write_chunk_kv → gather_window roundtrip through a real manager."""
    cfg = ARDiffusionKVConfig(enable=True, chunk_size=BLOCK, window_chunks=2)
    kv = ARDiffusionKVCache(
        cfg,
        num_layers=1,
        num_kv_heads=N_HEADS,
        head_size=HEAD_DIM,
        dtype=torch.float32,
        block_size=BLOCK,
        max_model_len=512,
        available_bytes=1 << 24,
        device=torch.device("cpu"),
    )
    adapter = kv.begin_request("r")
    kv.allocate_chunk(adapter)
    # Use the orchestrator's own slot mapping (which resolves the actual physical
    # block id from the request's block table).
    write_slots = kv.chunk_write_slots(adapter)

    new_k = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM)
    new_v = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM)
    kv.write_chunk_kv(0, new_k, new_v, adapter)
    kv.commit_chunk(adapter)

    # Read back the chunk's K/V from the pool at exactly the slots it was written to.
    assert torch.allclose(kv._k_pools[0][write_slots], new_k[0])
    assert torch.allclose(kv._v_pools[0][write_slots], new_v[0])

    # Allocate a second chunk so there's a past-window + current chunk.
    kv.allocate_chunk(adapter)
    window = kv.gather_window(0, adapter)
    assert window.shape == (2, 1, kv.spec.sliding_window, N_HEADS, HEAD_DIM)

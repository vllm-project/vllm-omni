# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for W8 entry-scoped BDE cache views."""

import torch

from vllm_omni.experimental.bde.kv_cache import (
    BDECacheEntryKey,
    BDECacheEntryStore,
    BDEKVCache,
    BDEKVConfig,
    bde_cache_entry_key,
    bde_cache_entry_key_dict,
)
from vllm_omni.experimental.bde.kv_cache.gather import BDEKVState

BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64


def make_cache(*, num_layers=1, window_chunks=8):
    cfg = BDEKVConfig(enable=True, chunk_size=BLOCK, window_chunks=window_chunks)
    return BDEKVCache(
        cfg,
        num_layers=num_layers,
        num_kv_heads=N_HEADS,
        head_size=HEAD_DIM,
        dtype=torch.float32,
        block_size=BLOCK,
        max_model_len=4096,
        available_bytes=1 << 24,
        device=torch.device("cpu"),
    )


def entry_key(index: int, *, sim_depth: int = 0) -> BDECacheEntryKey:
    return BDECacheEntryKey(
        session_id="session-a",
        session_epoch=1,
        observation_index=index,
        sim_depth=sim_depth,
    )


def test_cache_entry_key_round_trips_plain_dict():
    key = entry_key(3, sim_depth=1)
    assert bde_cache_entry_key(bde_cache_entry_key_dict(key)) == key


def window(n_chunks: int, value: float) -> torch.Tensor:
    return torch.full(
        (2, 1, n_chunks * BLOCK, N_HEADS, HEAD_DIM),
        value,
        dtype=torch.float32,
    )


def test_entry_store_gathers_exact_real_and_simulated_prefix():
    kv = make_cache(num_layers=1)
    store = BDECacheEntryStore(kv)
    o1_real = entry_key(1, sim_depth=0)
    o2_sim = entry_key(2, sim_depth=1)

    store.update_entry_kv(
        o1_real,
        layer_idx=0,
        updated_kv=window(1, 1.0),
        is_negative=False,
        seq_len=BLOCK,
    )
    store.update_entry_kv(
        o2_sim,
        layer_idx=0,
        updated_kv=window(1, 2.0),
        is_negative=False,
        seq_len=BLOCK,
    )

    gathered = store.get_prefix_kv_caches((o1_real, o2_sim), is_negative=False)[0]

    assert gathered.shape == (2, 1, 2 * BLOCK, N_HEADS, HEAD_DIM)
    assert torch.allclose(gathered[:, :, :BLOCK], window(1, 1.0))
    assert torch.allclose(gathered[:, :, BLOCK:], window(1, 2.0))


def test_entry_owner_release_waits_for_forward_lease():
    kv = make_cache()
    store = BDECacheEntryStore(kv)
    key = entry_key(2, sim_depth=1)
    store.update_entry_kv(
        key,
        layer_idx=0,
        updated_kv=window(1, 2.0),
        is_negative=False,
        seq_len=BLOCK,
    )
    free_with_entry = kv.manager.block_pool.get_num_free_blocks()

    lease = store.lease_entries((key,))
    store.release_owner(key)

    assert store.has_entry(key) is False
    assert kv.manager.block_pool.get_num_free_blocks() == free_with_entry
    assert torch.allclose(lease.get_kv_caches(is_negative=False)[0], window(1, 2.0))

    lease.close()

    assert kv.manager.block_pool.get_num_free_blocks() > free_with_entry


def test_released_entry_cannot_be_used_in_new_prefix_view():
    kv = make_cache()
    store = BDECacheEntryStore(kv)
    key = entry_key(2, sim_depth=1)
    store.update_entry_kv(
        key,
        layer_idx=0,
        updated_kv=window(1, 2.0),
        is_negative=False,
        seq_len=BLOCK,
    )

    store.release_owner(key)

    try:
        store.lease_entries((key,))
    except KeyError:
        pass
    else:
        raise AssertionError("released entry should not be leaseable")


def test_entry_store_keeps_positive_and_negative_branches_independent():
    kv = make_cache()
    store = BDECacheEntryStore(kv)
    key = entry_key(1)

    store.update_entry_kv(
        key,
        layer_idx=0,
        updated_kv=window(1, 1.0),
        is_negative=False,
        seq_len=BLOCK,
    )
    store.update_entry_kv(
        key,
        layer_idx=0,
        updated_kv=window(1, -1.0),
        is_negative=True,
        seq_len=BLOCK,
    )

    positive = store.get_prefix_kv_caches((key,), is_negative=False)[0]
    negative = store.get_prefix_kv_caches((key,), is_negative=True)[0]

    assert torch.allclose(positive, window(1, 1.0))
    assert torch.allclose(negative, window(1, -1.0))


def test_bde_kv_state_exposes_entry_leases_and_owner_release():
    kv = make_cache()
    state = BDEKVState(
        kv,
        kv.begin_request("linear-pos"),
        kv.begin_request("linear-neg"),
        num_layers=kv.num_layers,
    )
    key = entry_key(2, sim_depth=1)
    state.entries.update_entry_kv(
        key,
        layer_idx=0,
        updated_kv=window(1, 2.0),
        is_negative=False,
        seq_len=BLOCK,
    )

    lease = state.lease_prefix((key,))
    state.drop_cache_entry_owner(key)

    assert torch.allclose(lease.get_kv_caches(is_negative=False)[0], window(1, 2.0))
    lease.close()


def test_bde_kv_state_use_prefix_routes_reads_temporarily():
    kv = make_cache()
    state = BDEKVState(
        kv,
        kv.begin_request("linear-pos"),
        kv.begin_request("linear-neg"),
        num_layers=kv.num_layers,
    )
    state.update_kv_cache(0, window(1, 9.0), False, seq_len=BLOCK)
    key = entry_key(2, sim_depth=1)
    state.entries.update_entry_kv(
        key,
        layer_idx=0,
        updated_kv=window(1, 2.0),
        is_negative=False,
        seq_len=BLOCK,
    )

    assert torch.allclose(state.get_kv_caches(False, lambda: ["EMPTY"])[0], window(1, 9.0))
    with state.use_prefix((key,)):
        assert torch.allclose(state.get_kv_caches(False, lambda: ["EMPTY"])[0], window(1, 2.0))
    assert torch.allclose(state.get_kv_caches(False, lambda: ["EMPTY"])[0], window(1, 9.0))


def test_bde_kv_state_use_prefix_falls_back_when_branch_has_no_resident_blocks():
    kv = make_cache()
    state = BDEKVState(
        kv,
        kv.begin_request("linear-pos"),
        kv.begin_request("linear-neg"),
        num_layers=kv.num_layers,
    )
    key = entry_key(1)
    state.entries.update_entry_kv(
        key,
        layer_idx=0,
        updated_kv=window(1, 1.0),
        is_negative=False,
        seq_len=BLOCK,
    )

    with state.use_prefix((key,)):
        assert torch.allclose(state.get_kv_caches(False, lambda: ["EMPTY"])[0], window(1, 1.0))
        assert state.get_kv_caches(True, lambda: ["NEG_EMPTY"]) == ["NEG_EMPTY"]


def test_bde_kv_state_write_entry_routes_writes_temporarily():
    kv = make_cache()
    state = BDEKVState(
        kv,
        kv.begin_request("linear-pos"),
        kv.begin_request("linear-neg"),
        num_layers=kv.num_layers,
    )
    key = entry_key(1)

    with state.write_entry(key):
        state.update_kv_cache(0, window(1, 3.0), False, seq_len=BLOCK)

    assert state.get_kv_caches(False, lambda: ["EMPTY"]) == ["EMPTY"]
    with state.use_prefix((key,)):
        assert torch.allclose(state.get_kv_caches(False, lambda: ["EMPTY"])[0], window(1, 3.0))

    state.update_kv_cache(0, window(1, 4.0), False, seq_len=BLOCK)
    assert torch.allclose(state.get_kv_caches(False, lambda: ["EMPTY"])[0], window(1, 4.0))


def test_bde_kv_state_reads_current_entry_without_prefix_after_write():
    kv = make_cache()
    state = BDEKVState(
        kv,
        kv.begin_request("linear-pos"),
        kv.begin_request("linear-neg"),
        num_layers=kv.num_layers,
    )
    o1_real = entry_key(1)

    with state.write_entry(o1_real):
        assert state.get_kv_caches(False, lambda: ["EMPTY"]) == ["EMPTY"]
        state.update_kv_cache(0, window(1, 1.0), False, seq_len=BLOCK)
        assert torch.allclose(state.get_kv_caches(False, lambda: ["EMPTY"])[0], window(1, 1.0))


def test_bde_kv_state_write_entry_created_in_other_branch_uses_fallback_until_resident():
    kv = make_cache()
    state = BDEKVState(
        kv,
        kv.begin_request("linear-pos"),
        kv.begin_request("linear-neg"),
        num_layers=kv.num_layers,
    )
    o1_real = entry_key(1)

    with state.write_entry(o1_real):
        state.update_kv_cache(0, window(1, 1.0), False, seq_len=BLOCK)

        assert state.get_kv_caches(True, lambda: ["NEG_EMPTY"]) == ["NEG_EMPTY"]


def test_bde_kv_state_reads_prefix_plus_current_entry_after_write():
    kv = make_cache()
    state = BDEKVState(
        kv,
        kv.begin_request("linear-pos"),
        kv.begin_request("linear-neg"),
        num_layers=kv.num_layers,
    )
    o1_real = entry_key(1)
    o2_sim = entry_key(2, sim_depth=1)
    state.entries.update_entry_kv(
        o1_real,
        layer_idx=0,
        updated_kv=window(1, 1.0),
        is_negative=False,
        seq_len=BLOCK,
    )

    with state.use_prefix((o1_real,)):
        assert torch.allclose(state.get_kv_caches(False, lambda: ["EMPTY"])[0], window(1, 1.0))
        with state.write_entry(o2_sim):
            state.update_kv_cache(0, window(1, 2.0), False, seq_len=BLOCK)
            gathered = state.get_kv_caches(False, lambda: ["EMPTY"])[0]

    assert gathered.shape == (2, 1, 2 * BLOCK, N_HEADS, HEAD_DIM)
    assert torch.allclose(gathered[:, :, :BLOCK], window(1, 1.0))
    assert torch.allclose(gathered[:, :, BLOCK:], window(1, 2.0))

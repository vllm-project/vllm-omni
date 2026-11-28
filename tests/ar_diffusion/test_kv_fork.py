# SPDX-License-Identifier: Apache-2.0
"""Tests for copy-on-write session fork over the AR-Diffusion KV cache.

Exercised against the installed vLLM KV stack on CPU (block bookkeeping
only, no GPU tensors), like the rest of the AR-Diffusion KV tests.
"""

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

DIMS = dict(num_layers=2, num_kv_heads=4, head_size=64, dtype=torch.float16, block_size=16)


def make_cache(*, chunk_size=16, window_chunks=4, available_bytes=1 << 24):
    cfg = ARDiffusionKVConfig(enable=True, chunk_size=chunk_size, window_chunks=window_chunks)
    return ARDiffusionKVCache(cfg, max_model_len=4096, available_bytes=available_bytes, **DIMS)


def run_chunks(kv, adapter, n):
    for _ in range(n):
        kv.allocate_chunk(adapter)
        kv.commit_chunk(adapter)


def resident(kv, adapter):
    return kv.window_block_ids(adapter)


class TestForkAtLastCommit:
    def test_fork_shares_blocks_without_allocation(self):
        kv = make_cache()
        parent = kv.begin_request("parent")
        run_chunks(kv, parent, 3)
        free_before = kv.manager.block_pool.get_num_free_blocks()

        child = kv.fork_at_last_commit(parent, "child")

        # Zero new blocks consumed: the fork is a table copy plus refcounts.
        assert kv.manager.block_pool.get_num_free_blocks() == free_before
        assert resident(kv, child) == resident(kv, parent)
        assert child.num_computed_tokens == parent.num_computed_tokens
        assert child.completed_chunks == parent.completed_chunks

    def test_post_fork_divergence_is_isolated(self):
        kv = make_cache()
        parent = kv.begin_request("parent")
        run_chunks(kv, parent, 2)
        child = kv.fork_at_last_commit(parent, "child")

        shared = set(resident(kv, parent))
        run_chunks(kv, child, 1)
        run_chunks(kv, parent, 1)

        parent_new = set(resident(kv, parent)) - shared
        child_new = set(resident(kv, child)) - shared
        # Each branch appended fresh physical blocks, and they do not collide.
        assert parent_new and child_new
        assert parent_new.isdisjoint(child_new)

    def test_shared_blocks_survive_parent_free(self):
        kv = make_cache()
        parent = kv.begin_request("parent")
        run_chunks(kv, parent, 2)
        child = kv.fork_at_last_commit(parent, "child")
        shared = resident(kv, child)

        kv.end_request(parent)
        # The child still resolves the shared history blocks.
        assert resident(kv, child) == shared

        # Allocating a fresh session must not hand out the child's blocks.
        other = kv.begin_request("other")
        run_chunks(kv, other, 2)
        assert set(resident(kv, other)).isdisjoint(set(shared))

    def test_refcounts_balance_after_both_free(self):
        kv = make_cache()
        free_at_start = kv.manager.block_pool.get_num_free_blocks()
        parent = kv.begin_request("parent")
        run_chunks(kv, parent, 3)
        child = kv.fork_at_last_commit(parent, "child")
        run_chunks(kv, child, 1)

        kv.end_request(parent)
        kv.end_request(child)
        assert kv.manager.block_pool.get_num_free_blocks() == free_at_start

    def test_fork_mid_chunk_is_rejected(self):
        kv = make_cache()
        parent = kv.begin_request("parent")
        run_chunks(kv, parent, 1)
        kv.allocate_chunk(parent)  # in-flight, not committed
        with pytest.raises(RuntimeError, match="chunk-commit boundary"):
            kv.fork_at_last_commit(parent, "child")

    def test_fork_rejects_duplicate_and_unknown_ids(self):
        kv = make_cache()
        parent = kv.begin_request("parent")
        run_chunks(kv, parent, 1)
        kv.fork_at_last_commit(parent, "child")
        with pytest.raises(ValueError, match="already exists"):
            kv.fork_at_last_commit(parent, "child")
        ghost = kv.begin_request("ghost")
        kv._adapters.pop("ghost")
        kv.manager.coordinator.single_type_managers[0].req_to_blocks.pop("ghost", None)
        with pytest.raises(ValueError, match="unknown parent"):
            kv.fork_at_last_commit(ghost, "child2")

    def test_fork_after_window_eviction_shares_only_resident_blocks(self):
        # Roll past the window so old blocks were evicted (null placeholders);
        # the fork must share exactly the resident window, and rolling the
        # child forward must not corrupt the parent's view.
        kv = make_cache(window_chunks=2)
        parent = kv.begin_request("parent")
        run_chunks(kv, parent, 5)
        child = kv.fork_at_last_commit(parent, "child")
        assert resident(kv, child) == resident(kv, parent)

        parent_view = list(resident(kv, parent))
        run_chunks(kv, child, 2)  # child evicts shared blocks from ITS table
        assert resident(kv, parent) == parent_view

    def test_multi_branch_fanout(self):
        # RL-style: N branches from one state, each rolled independently.
        kv = make_cache(available_bytes=1 << 26)
        parent = kv.begin_request("parent")
        run_chunks(kv, parent, 2)
        free_before = kv.manager.block_pool.get_num_free_blocks()
        branches = [kv.fork_at_last_commit(parent, f"b{i}") for i in range(4)]
        assert kv.manager.block_pool.get_num_free_blocks() == free_before

        for branch in branches:
            run_chunks(kv, branch, 1)
        tails = [set(resident(kv, b)) - set(resident(kv, parent)) for b in branches]
        for i in range(len(tails)):
            for j in range(i + 1, len(tails)):
                assert tails[i].isdisjoint(tails[j])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="write-isolation test needs GPU-backed pools")
class TestForkWriteIsolationGPU:
    """Tensor-level check of the commit-once assumption the fork relies on.

    The CPU tests verify the bookkeeping; this verifies the memory: after a
    fork, the shared physical slots hold identical bytes for both sessions,
    and a diverging child never rewrites the parent's committed K/V.
    """

    def test_child_divergence_does_not_touch_parent_bytes(self):
        device = torch.device("cuda")
        cfg = ARDiffusionKVConfig(enable=True, chunk_size=16, window_chunks=4)
        kv = ARDiffusionKVCache(cfg, max_model_len=4096, available_bytes=1 << 24, device=device, **DIMS)
        chunk_kv_shape = (1, cfg.chunk_size, DIMS["num_kv_heads"], DIMS["head_size"])

        def write_chunk(adapter, fill):
            kv.allocate_chunk(adapter)
            slots = kv.chunk_write_slots(adapter)
            for layer in range(DIMS["num_layers"]):
                k = torch.full(chunk_kv_shape, fill, dtype=DIMS["dtype"], device=device)
                v = torch.full(chunk_kv_shape, -fill, dtype=DIMS["dtype"], device=device)
                kv.write_chunk_kv(layer, k, v, adapter)
            kv.commit_chunk(adapter)
            return slots

        parent = kv.begin_request("parent")
        parent_slots = [write_chunk(parent, fill=float(i + 1)) for i in range(2)]

        def read_bytes(slot_list):
            return [
                (kv._k_pools[layer][slots].clone(), kv._v_pools[layer][slots].clone())
                for layer in range(DIMS["num_layers"])
                for slots in slot_list
            ]

        before = read_bytes(parent_slots)
        child = kv.fork_at_last_commit(parent, "child")

        # Shared prefix resolves to the same physical slots for the child.
        assert kv.window_block_ids(child) == kv.window_block_ids(parent)

        # The child diverges with different values; the parent's committed
        # bytes must be bit-identical afterwards.
        write_chunk(child, fill=99.0)
        after = read_bytes(parent_slots)
        for (k0, v0), (k1, v1) in zip(before, after):
            assert torch.equal(k0, k1)
            assert torch.equal(v0, v1)

        # And symmetrically: the parent rolling forward does not disturb the
        # child's shared view of the prefix.
        child_prefix_before = read_bytes(parent_slots)
        write_chunk(parent, fill=77.0)
        child_prefix_after = read_bytes(parent_slots)
        for (k0, v0), (k1, v1) in zip(child_prefix_before, child_prefix_after):
            assert torch.equal(k0, k1)
            assert torch.equal(v0, v1)


class TestSessionFork:
    def test_forks_both_cfg_streams(self):
        kv = make_cache()
        state = ARDiffusionKVState(kv, kv.begin_request("pos"), kv.begin_request("neg"), num_layers=2)
        run_chunks(kv, state.pos, 2)
        run_chunks(kv, state.neg, 2)
        state._cross_text_populated = {False: True, True: True}

        child = state.fork("pos.b0", "neg.b0")
        assert resident(kv, child.pos) == resident(kv, state.pos)
        assert resident(kv, child.neg) == resident(kv, state.neg)
        # Text conditioning is prompt-derived and branch-invariant: inherited.
        assert child._cross_text_populated == state._cross_text_populated

        child.close()
        state.close()

    def test_fork_invalidates_image_conditioning_on_both_branches(self):
        # The image cross-attn pool is engine-wide, not per session. If a
        # forked branch kept _cross_img_populated=True, it would skip
        # repopulation and silently read whichever branch last wrote the
        # shared pool. Fork must force BOTH branches to re-project image
        # conditioning from their own next observation.
        kv = make_cache()
        state = ARDiffusionKVState(kv, kv.begin_request("pos"), kv.begin_request("neg"), num_layers=2)
        run_chunks(kv, state.pos, 1)
        run_chunks(kv, state.neg, 1)
        state._cross_text_populated = {False: True, True: True}
        state._cross_img_populated = {False: True, True: True}

        child = state.fork("pos.b0", "neg.b0")
        assert child._cross_img_populated == {False: False, True: False}
        assert state._cross_img_populated == {False: False, True: False}

        child.close()
        state.close()

    def test_partial_fork_failure_rolls_back(self):
        kv = make_cache()
        state = ARDiffusionKVState(kv, kv.begin_request("pos"), kv.begin_request("neg"), num_layers=2)
        run_chunks(kv, state.pos, 1)
        run_chunks(kv, state.neg, 1)
        kv.begin_request("neg.b0")  # occupy the neg child id to force failure
        run_chunks(kv, kv._adapters["neg.b0"], 1)

        free_before = kv.manager.block_pool.get_num_free_blocks()
        with pytest.raises(ValueError):
            state.fork("pos.b0", "neg.b0")
        # The half-made pos fork was released; no leaked references.
        assert kv.manager.block_pool.get_num_free_blocks() == free_before
        assert "pos.b0" not in kv._adapters

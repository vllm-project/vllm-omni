# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for AR-Diffusion forward-context reuse and the paged-KV slot write.

The denoise loop calls ``get_kv_caches`` once per DiT forward, but inside the loop
the KV mapping cannot move: those forwards pass ``commit_current=False``, so
nothing is allocated or committed. ``ARDiffusionKVState`` therefore caches one
forward context per KV branch and hands the same one back until something that
could change the addressing changes.

Correctness here is entirely about *when the entry must be dropped*. A stale block
table or slot mapping does not crash -- it addresses the wrong blocks and returns a
plausible wrong answer -- so each invalidation path gets its own case:

* a committing forward is never cached and never reuses a cached entry;
* committing drops the branch's entry;
* ``close()`` and ``reset()`` drop everything, including on the
  ``keep_cross_attention`` path, which skips ``close()`` entirely;
* if the geometry moves under a reused context, ``prepare()`` raises rather than
  computing against the wrong table.

The last test covers the separate ``index_copy_`` slot write, which must stay
value-identical to the advanced indexing it replaces.

All CPU, no accelerator required.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64
POS = "positive"
NEG = "negative"
CROSS = "text"


def make_state(*, num_layers=2, window_chunks=2, cross=False):
    """A CPU session with two KV branches, mirroring tests/…/test_paged_attention.py."""
    cfg = ARDiffusionKVConfig(enable=True, chunk_size=BLOCK, window_chunks=window_chunks)
    kv = ARDiffusionKVCache(
        cfg,
        num_layers=num_layers,
        num_kv_heads=N_HEADS,
        head_size=HEAD_DIM,
        dtype=torch.float32,
        block_size=BLOCK,
        max_model_len=4096,
        available_bytes=1 << 26,
        kv_branches=(ARDiffusionKVBranchSpec(POS, 0), ARDiffusionKVBranchSpec(NEG, 1)),
        session_capacity=2,
        cross_attention_lengths={CROSS: BLOCK} if cross else None,
        frames_per_block=2,
        max_scratch_tokens_per_branch=BLOCK,
        device=torch.device("cpu"),
    )
    adapters = {POS: kv.begin_request("r-pos"), NEG: kv.begin_request("r-neg")}
    return kv, ARDiffusionKVState(kv, "s1", adapters, num_layers=num_layers)


def _read_only(st, kv_branch=POS):
    return st.get_kv_caches(kv_branch, seq_len=BLOCK, commit_current=False)


def test_denoise_loop_reuses_one_context():
    """Repeated read-only forwards share one context: 1 miss then all hits."""
    _, st = make_state()

    first = _read_only(st)
    repeats = [_read_only(st) for _ in range(15)]

    # The identical list object comes back, so the 40 layer contexts and every
    # metadata tensor behind them are built once, not once per denoise step.
    assert all(later is first for later in repeats)
    stats = st.fctx_cache_stats()
    assert stats["fctx_cache_miss"] == 1
    assert stats["fctx_cache_hit"] == 15
    assert stats["fctx_cache_invalidate"] == 0
    assert stats["fctx_cache_entries"] == 1


def test_committing_forward_is_never_cached_or_reused():
    """commit_current=True must miss every time, via the epoch token."""
    _, st = make_state()

    first = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=True)
    second = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=True)

    assert second is not first
    stats = st.fctx_cache_stats()
    assert stats["fctx_cache_hit"] == 0
    assert stats["fctx_cache_miss"] == 2


def test_commit_drops_the_branch_entry():
    """Committing advances the KV mapping, so the cached table is stale."""
    _, st = make_state()

    _read_only(st)
    assert st.fctx_cache_stats()["fctx_cache_entries"] == 1

    st.commit_paged_context(POS)
    assert st.fctx_cache_stats()["fctx_cache_entries"] == 0

    after = _read_only(st)
    assert st.fctx_cache_stats()["fctx_cache_miss"] == 2
    assert after is not None


def test_branches_are_cached_independently():
    """CFG-parallel runs two KV branches; each gets its own entry."""
    _, st = make_state()

    pos_first = _read_only(st, POS)
    neg_first = _read_only(st, NEG)
    assert neg_first is not pos_first
    assert st.fctx_cache_stats()["fctx_cache_entries"] == 2

    assert _read_only(st, POS) is pos_first
    assert _read_only(st, NEG) is neg_first
    assert st.fctx_cache_stats()["fctx_cache_hit"] == 2


def test_close_drops_entries():
    _, st = make_state()
    _read_only(st)
    st.close()
    assert st.fctx_cache_stats()["fctx_cache_entries"] == 0


def test_reset_drops_entries():
    _, st = make_state()
    _read_only(st)
    st.reset()
    assert st.fctx_cache_stats()["fctx_cache_entries"] == 0


def test_reset_with_keep_cross_attention_drops_entries():
    """The path where relying on close() alone would leave a stale entry.

    ``reset(keep_cross_attention=...)`` deliberately skips ``close()``, but it still
    installs brand new adapters and zeroes the committed counters, so every cached
    block table addresses blocks this session no longer owns.
    """
    _, st = make_state(cross=True)
    _read_only(st)
    assert st.fctx_cache_stats()["fctx_cache_entries"] == 1

    st.reset(keep_cross_attention=(CROSS,))

    assert st.fctx_cache_stats()["fctx_cache_entries"] == 0
    # And the rebuilt session still works, against the new adapters.
    assert _read_only(st) is not None


def test_reused_context_raises_when_geometry_moves():
    """The cache key does not cover action_len/query_len, so prepare() checks them.

    They arrive as arguments to ``prepare()`` rather than through the key, so a
    reused context whose geometry moved must fail loudly instead of computing
    against a block table built for a different query length.
    """
    _, st = make_state()
    device = torch.device("cpu")

    fctx = _read_only(st)[0].forward_ctx
    fctx.prepare(device=device, action_len=0, query_len=BLOCK)

    # Same call again is a no-op; the geometry matches.
    fctx.prepare(device=device, action_len=0, query_len=BLOCK)

    with pytest.raises(RuntimeError, match="reused with different geometry"):
        fctx.prepare(device=device, action_len=0, query_len=BLOCK + 1)
    with pytest.raises(RuntimeError, match="reused with different geometry"):
        fctx.prepare(device=device, action_len=4, query_len=BLOCK)


@pytest.mark.parametrize(
    "slots",
    [
        [0, 1, 2, 3],  # contiguous ascending, the prefill case
        [3, 2, 1, 0],  # descending, as the window block order produces
        [5, 0, 7, 2],  # scattered, after eviction
        [9],  # single slot
    ],
)
def test_index_copy_matches_advanced_indexing_for_unique_slots(slots):
    """``pool[slots] = v`` and ``pool.index_copy_(0, slots, v)`` must agree.

    Parametrized over the index patterns the production slot mapping actually
    produces, so this is not just the trivially contiguous case.
    """
    torch.manual_seed(0)
    index = torch.tensor(slots, dtype=torch.long)
    pool_a = torch.randn(12, N_HEADS, HEAD_DIM)
    pool_b = pool_a.clone()
    value = torch.randn(len(slots), N_HEADS, HEAD_DIM)

    pool_a[index] = value
    pool_b.index_copy_(0, index, value)

    assert torch.equal(pool_a, pool_b)

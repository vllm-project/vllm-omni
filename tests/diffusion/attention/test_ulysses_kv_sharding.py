# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the GQA K/V head reconciliation helpers in Ulysses-SP.

These cover the branches HunyuanImage3 does NOT exercise at runtime (it only
hits the ``kv_head_cnt % world_size == 0`` fast path). The helpers are pure
tensor functions -- ``world_size`` is just an int argument -- so no distributed
or GPU setup is required.

Branches under test for ``_prepare_kv_for_strict_sharding``:
  A. kv_head_cnt % world_size == 0 (and world_size<=1 / kv==0)  -> unchanged
  B. kv % world_size != 0, q % lcm(world_size, kv) == 0         -> minimal LCM repeat
  C. kv % world_size != 0, q % lcm != 0, q % kv == 0            -> full MHA expansion
  D. q % kv != 0                                                -> ValueError
"""

from __future__ import annotations

from math import lcm

import pytest
import torch

from vllm_omni.diffusion.attention.parallel.ulysses import (
    _expand_kv_to_match_q,
    _prepare_kv_for_strict_sharding,
    _repeat_kv_heads,
)

BATCH = 2
SEQ_LEN = 3
HEAD_DIM = 8


def _make_kv(num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K/V of shape (B, S, H, D) with each head filled by its head index.

    Head ``h`` is filled with the value ``h``, so the interleaved-repeat layout
    is trivially checkable (head ``h`` repeated ``r`` times -> r consecutive
    blocks all equal to ``h``).
    """
    torch.manual_seed(0)
    heads = torch.arange(num_heads, dtype=torch.float32)
    base = heads.view(1, 1, num_heads, 1).expand(BATCH, SEQ_LEN, num_heads, HEAD_DIM).contiguous()
    # k and v distinguishable so we catch any accidental swap.
    return base.clone(), (base + 100.0).clone()


def _assert_interleaved_repeat(out: torch.Tensor, src: torch.Tensor, r: int) -> None:
    """``out`` must equal ``src`` with each head interleave-repeated ``r`` times."""
    assert out.shape[2] == src.shape[2] * r
    torch.testing.assert_close(out, torch.repeat_interleave(src, r, dim=2))
    assert out.is_contiguous()


# --------------------------------------------------------------------------
# _repeat_kv_heads: the shared primitive
# --------------------------------------------------------------------------
def test_repeat_kv_heads_identity_when_n_rep_1() -> None:
    k, _ = _make_kv(4)
    out = _repeat_kv_heads(k, 1)
    assert out is k  # no copy on the no-op path


@pytest.mark.parametrize("n_rep", [2, 3, 4])
def test_repeat_kv_heads_matches_repeat_interleave(n_rep: int) -> None:
    k, _ = _make_kv(2)
    _assert_interleaved_repeat(_repeat_kv_heads(k, n_rep), k, n_rep)


# --------------------------------------------------------------------------
# Scenario A: kv_head_cnt already shards evenly -> returned unchanged
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kv_heads,world_size,q_heads",
    [
        (8, 4, 8),  # kv % world_size == 0
        (4, 4, 4),  # kv == world_size
        (2, 1, 8),  # world_size == 1 (SP disabled)
    ],
)
def test_strict_unchanged_fast_path(kv_heads: int, world_size: int, q_heads: int) -> None:
    k, v = _make_kv(kv_heads)
    out_k, out_v = _prepare_kv_for_strict_sharding(k, v, q_head_cnt=q_heads, world_size=world_size)
    # Same objects, not just equal -> proves we took the no-copy fast path.
    assert out_k is k
    assert out_v is v


def test_strict_unchanged_when_kv_zero() -> None:
    # Degenerate empty-KV guard (kv_head_cnt == 0).
    k = torch.empty(BATCH, SEQ_LEN, 0, HEAD_DIM)
    v = torch.empty(BATCH, SEQ_LEN, 0, HEAD_DIM)
    out_k, out_v = _prepare_kv_for_strict_sharding(k, v, q_head_cnt=8, world_size=4)
    assert out_k is k
    assert out_v is v


# --------------------------------------------------------------------------
# Scenario B: minimal repeat to lcm(world_size, kv_head_cnt)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kv_heads,world_size,q_heads",
    [
        (2, 4, 8),  # m = lcm(4,2) = 4 -> r = 2 (< full MHA r=4)
        (6, 4, 12),  # m = lcm(4,6) = 12 -> r = 2 (< full MHA r=2 happens to match here)
        (6, 4, 24),  # m = 12 -> r = 2, q is a larger multiple of m
    ],
)
def test_strict_minimal_lcm_repeat(kv_heads: int, world_size: int, q_heads: int) -> None:
    k, v = _make_kv(kv_heads)
    m = lcm(world_size, kv_heads)
    expected_r = m // kv_heads

    out_k, out_v = _prepare_kv_for_strict_sharding(k, v, q_head_cnt=q_heads, world_size=world_size)

    # Repeated to exactly m heads, which divides world_size (whole heads per rank)
    # and stays <= q_head_cnt (cheaper than full MHA expansion).
    assert out_k.shape[2] == m
    assert m % world_size == 0
    assert out_k.shape[2] <= q_heads
    _assert_interleaved_repeat(out_k, k, expected_r)
    _assert_interleaved_repeat(out_v, v, expected_r)


# --------------------------------------------------------------------------
# Scenario C: lcm doesn't divide q -> fall back to full MHA expansion
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kv_heads,world_size,q_heads",
    [
        (2, 4, 6),  # m = 4, 6 % 4 != 0, 6 % 2 == 0 -> r = 3, expand to 6 == q
        (3, 2, 9),  # m = 6, 9 % 6 != 0, 9 % 3 == 0 -> r = 3, expand to 9 == q
    ],
)
def test_strict_mha_fallback(kv_heads: int, world_size: int, q_heads: int) -> None:
    k, v = _make_kv(kv_heads)
    m = lcm(world_size, kv_heads)
    assert q_heads % m != 0  # guard: these inputs really hit the fallback
    expected_r = q_heads // kv_heads

    out_k, out_v = _prepare_kv_for_strict_sharding(k, v, q_head_cnt=q_heads, world_size=world_size)

    # Expanded all the way to full MHA (one KV head per Q head).
    assert out_k.shape[2] == q_heads
    _assert_interleaved_repeat(out_k, k, expected_r)
    _assert_interleaved_repeat(out_v, v, expected_r)


# --------------------------------------------------------------------------
# Scenario D: q_head_cnt not a multiple of kv_head_cnt -> ValueError
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kv_heads,world_size,q_heads",
    [
        (4, 3, 6),  # kv % ws != 0, q % kv != 0
        (3, 2, 8),  # 8 % 3 != 0
    ],
)
def test_strict_raises_when_q_not_multiple_of_kv(kv_heads: int, world_size: int, q_heads: int) -> None:
    k, v = _make_kv(kv_heads)
    with pytest.raises(ValueError, match="cannot reconcile kv_head_cnt"):
        _prepare_kv_for_strict_sharding(k, v, q_head_cnt=q_heads, world_size=world_size)


# --------------------------------------------------------------------------
# _expand_kv_to_match_q (advanced_uaa / non-strict path)
# --------------------------------------------------------------------------
def test_expand_kv_identity_when_already_mha() -> None:
    k, v = _make_kv(8)
    out_k, out_v = _expand_kv_to_match_q(k, v, q_head_cnt=8)
    assert out_k is k
    assert out_v is v


def test_expand_kv_identity_when_kv_zero() -> None:
    k = torch.empty(BATCH, SEQ_LEN, 0, HEAD_DIM)
    v = torch.empty(BATCH, SEQ_LEN, 0, HEAD_DIM)
    out_k, out_v = _expand_kv_to_match_q(k, v, q_head_cnt=8)
    assert out_k is k
    assert out_v is v


@pytest.mark.parametrize(
    "kv_heads,q_heads",
    [
        (2, 8),  # r = 4
        (4, 8),  # r = 2
        (3, 9),  # r = 3
    ],
)
def test_expand_kv_to_match_q(kv_heads: int, q_heads: int) -> None:
    k, v = _make_kv(kv_heads)
    expected_r = q_heads // kv_heads
    out_k, out_v = _expand_kv_to_match_q(k, v, q_head_cnt=q_heads)
    assert out_k.shape[2] == q_heads
    _assert_interleaved_repeat(out_k, k, expected_r)
    _assert_interleaved_repeat(out_v, v, expected_r)


@pytest.mark.parametrize("kv_heads,q_heads", [(3, 8), (5, 12)])
def test_expand_kv_raises_when_q_not_multiple_of_kv(kv_heads: int, q_heads: int) -> None:
    k, v = _make_kv(kv_heads)
    with pytest.raises(ValueError, match="not a multiple"):
        _expand_kv_to_match_q(k, v, q_head_cnt=q_heads)

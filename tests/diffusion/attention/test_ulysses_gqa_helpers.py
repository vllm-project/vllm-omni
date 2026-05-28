# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the GQA-aware K/V repetition helpers in Ulysses SP.

These cover the pure tensor-shape / arithmetic logic of
``_prepare_kv_for_strict_sharding`` and ``_expand_kv_to_match_q``. They do not
require a distributed environment.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.attention.parallel.ulysses import (
    _expand_kv_to_match_q,
    _prepare_kv_for_strict_sharding,
    _repeat_kv_heads,
)


def _make_kv(num_kv_heads: int, *, b: int = 1, s: int = 4, d: int = 8) -> torch.Tensor:
    # Use a head-distinct pattern so we can verify the interleaved repeat.
    base = torch.arange(num_kv_heads, dtype=torch.float32).view(1, 1, num_kv_heads, 1)
    return base.expand(b, s, num_kv_heads, d).contiguous()


def test_repeat_kv_heads_identity_on_n_rep_1() -> None:
    t = _make_kv(8)
    out = _repeat_kv_heads(t, 1)
    assert out is t  # no-op fast path


def test_repeat_kv_heads_interleaved() -> None:
    t = _make_kv(2, d=1)  # heads = [0, 1]
    out = _repeat_kv_heads(t, 3)
    assert out.shape == (1, 4, 6, 1)
    # interleaved → [0,0,0,1,1,1]
    expected = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)
    assert torch.equal(out[0, 0, :, 0], expected)
    assert out.is_contiguous()


def test_strict_no_repeat_when_kv_divides_world_size() -> None:
    k = _make_kv(8)
    v = _make_kv(8)
    k_out, v_out = _prepare_kv_for_strict_sharding(k, v, q_head_cnt=48, world_size=8)
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape
    assert k_out is k  # fast path returns the input


def test_strict_partial_repeat_when_kv_smaller_than_world_size() -> None:
    # HunyuanImage3-like: 48 Q heads, 8 KV heads, sp=16 -> repeat by 2 → 16 KV heads.
    k = _make_kv(8)
    v = _make_kv(8)
    k_out, _ = _prepare_kv_for_strict_sharding(k, v, q_head_cnt=48, world_size=16)
    assert k_out.shape[2] == 16
    # interleaved: every consecutive pair carries the same K-head id
    head_ids = k_out[0, 0, :, 0]
    expected = torch.arange(8, dtype=torch.float32).repeat_interleave(2)
    assert torch.equal(head_ids, expected)


def test_strict_falls_back_to_q_head_count_when_lcm_does_not_divide_q() -> None:
    # num_kv_heads=5, world_size=4, q=40.
    # lcm(4, 5) = 20; 40 % 20 == 0 → r = 4 (clean partial path).
    k = _make_kv(5)
    v = _make_kv(5)
    k_out, _ = _prepare_kv_for_strict_sharding(k, v, q_head_cnt=40, world_size=4)
    assert k_out.shape[2] == 20  # = lcm(4, 5)

    # num_kv_heads=5, world_size=4, q=20. lcm = 20; 20 % 20 == 0 → r = 4.
    k2 = _make_kv(5)
    v2 = _make_kv(5)
    k2_out, _ = _prepare_kv_for_strict_sharding(k2, v2, q_head_cnt=20, world_size=4)
    assert k2_out.shape[2] == 20

    # Pathological: num_kv_heads=3, world_size=4, q=12. lcm = 12; 12 % 12 == 0
    # → r = 4, partial path keeps it at q_head_cnt.
    k3 = _make_kv(3)
    v3 = _make_kv(3)
    k3_out, _ = _prepare_kv_for_strict_sharding(k3, v3, q_head_cnt=12, world_size=4)
    assert k3_out.shape[2] == 12


def test_strict_world_size_1_is_passthrough() -> None:
    k = _make_kv(8)
    k_out, _ = _prepare_kv_for_strict_sharding(k, k, q_head_cnt=48, world_size=1)
    assert k_out is k


def test_strict_raises_when_q_not_divisible_by_kv() -> None:
    k = _make_kv(5)  # 5 KV heads
    v = _make_kv(5)
    # q=7, kv=5: lcm(4,5)=20, 7%20!=0, fallback needs q%kv==0 (7%5!=0) → raise.
    with pytest.raises(ValueError, match="cannot reconcile"):
        _prepare_kv_for_strict_sharding(k, v, q_head_cnt=7, world_size=4)


def test_expand_kv_to_match_q_full_mha() -> None:
    k = _make_kv(8)
    v = _make_kv(8)
    k_out, v_out = _expand_kv_to_match_q(k, v, q_head_cnt=48)
    assert k_out.shape[2] == 48
    assert v_out.shape[2] == 48
    # interleaved by 6 (48/8)
    expected = torch.arange(8, dtype=torch.float32).repeat_interleave(6)
    assert torch.equal(k_out[0, 0, :, 0], expected)


def test_expand_kv_to_match_q_noop_when_already_matched() -> None:
    k = _make_kv(8)
    v = _make_kv(8)
    k_out, v_out = _expand_kv_to_match_q(k, v, q_head_cnt=8)
    assert k_out is k
    assert v_out is v


def test_expand_kv_to_match_q_raises_on_bad_ratio() -> None:
    k = _make_kv(7)
    v = _make_kv(7)
    with pytest.raises(ValueError, match="not a multiple"):
        _expand_kv_to_match_q(k, v, q_head_cnt=48)

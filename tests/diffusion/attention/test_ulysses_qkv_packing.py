# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_omni.diffusion.attention.parallel import ulysses


def _fake_sp_group(world_size: int = 2):
    return SimpleNamespace(
        ulysses_group=object(),
        ulysses_world_size=world_size,
        ulysses_rank=0,
        ring_world_size=1,
        ring_group=object(),
    )


def test_strict_pre_attention_packs_matching_qkv_into_one_5d_all_to_all(monkeypatch) -> None:
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda default="strict": "strict")

    calls_4d = []
    calls_5d = []

    def fake_4d_apply(pg, x, scatter_idx, gather_idx, use_sync):
        calls_4d.append((tuple(x.shape), scatter_idx, gather_idx, use_sync))
        bsz, shard_seq_len, head_cnt, head_size = x.shape
        return torch.empty(bsz, shard_seq_len * 2, head_cnt // 2, head_size, dtype=x.dtype)

    def fake_5d_apply(pg, x, scatter_idx, gather_idx, use_sync):
        calls_5d.append((tuple(x.shape), scatter_idx, gather_idx, use_sync))
        bsz, shard_seq_len, qkv_cnt, head_cnt, head_size = x.shape
        assert qkv_cnt == 3
        out = torch.empty(bsz, shard_seq_len * 2, qkv_cnt, head_cnt // 2, head_size, dtype=x.dtype)
        out[:, :, 0].fill_(1.0)
        out[:, :, 1].fill_(2.0)
        out[:, :, 2].fill_(3.0)
        return out

    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", staticmethod(fake_4d_apply))
    monkeypatch.setattr(ulysses, "SeqAllToAll5D", SimpleNamespace(apply=fake_5d_apply), raising=False)

    attn = ulysses.UlyssesParallelAttention(_fake_sp_group(), scatter_idx=2, gather_idx=1, use_sync=False)
    query = torch.randn(1, 4, 8, 16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    query_out, key_out, value_out, _, ctx = attn.pre_attention(query, key, value, attn_metadata=None)

    assert calls_5d == [((1, 4, 3, 8, 16), 3, 1, False)]
    assert calls_4d == []
    assert tuple(query_out.shape) == (1, 8, 4, 16)
    assert tuple(key_out.shape) == (1, 8, 4, 16)
    assert tuple(value_out.shape) == (1, 8, 4, 16)
    assert torch.all(query_out == 1.0)
    assert torch.all(key_out == 2.0)
    assert torch.all(value_out == 3.0)
    assert not ctx.use_uaa


def test_strict_pre_attention_falls_back_to_4d_all_to_all_for_mismatched_qkv(monkeypatch) -> None:
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda default="strict": "strict")

    calls_4d = []
    calls_5d = []

    def fake_4d_apply(pg, x, scatter_idx, gather_idx, use_sync):
        calls_4d.append((tuple(x.shape), scatter_idx, gather_idx, use_sync))
        bsz, shard_seq_len, head_cnt, head_size = x.shape
        return torch.empty(bsz, shard_seq_len * 2, head_cnt // 2, head_size, dtype=x.dtype)

    def fake_5d_apply(pg, x, scatter_idx, gather_idx, use_sync):
        calls_5d.append((tuple(x.shape), scatter_idx, gather_idx, use_sync))
        raise AssertionError("5D all-to-all should not run for mismatched Q/K/V shapes")

    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", staticmethod(fake_4d_apply))
    monkeypatch.setattr(ulysses, "SeqAllToAll5D", SimpleNamespace(apply=fake_5d_apply), raising=False)

    attn = ulysses.UlyssesParallelAttention(_fake_sp_group(), scatter_idx=2, gather_idx=1, use_sync=False)
    query = torch.randn(1, 4, 8, 16)
    key = torch.randn(1, 4, 4, 16)
    value = torch.randn_like(query)

    query_out, key_out, value_out, _, _ = attn.pre_attention(query, key, value, attn_metadata=None)

    assert calls_5d == []
    assert calls_4d == [
        ((1, 4, 8, 16), 2, 1, False),
        ((1, 4, 4, 16), 2, 1, False),
        ((1, 4, 8, 16), 2, 1, False),
    ]
    assert tuple(query_out.shape) == (1, 8, 4, 16)
    assert tuple(key_out.shape) == (1, 8, 2, 16)
    assert tuple(value_out.shape) == (1, 8, 4, 16)

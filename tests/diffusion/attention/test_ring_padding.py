# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.attention.backends import ring_flash_attn, ring_pytorch_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType
from vllm_omni.diffusion.attention.backends.ring.ring_utils import ring_kv_block_valid_length
from vllm_omni.diffusion.attention.parallel import ring as ring_parallel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _SingleRankComm:
    rank = 0
    world_size = 1


@dataclass(frozen=True)
class _FakeSequenceParallelGroup:
    ring_group: object


def _fake_attention_result(q, _k, _v, **_kwargs):
    return torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[2], q.shape[1])


@pytest.mark.parametrize(("block_rank", "expected"), [(0, 4), (1, 1), (2, 0)])
def test_global_prefix_is_partitioned_across_ring_blocks(block_rank, expected):
    assert ring_kv_block_valid_length(5, 4, block_rank, 3) == expected


def test_absent_prefix_length_keeps_the_full_ring_block():
    assert ring_kv_block_valid_length(None, 4, 2, 3) == 4


def test_pytorch_ring_trims_padding_from_each_kv_block(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_attention(q, k, v, **kwargs):
        captured.update(k=k, v=v, **kwargs)
        return _fake_attention_result(q, k, v, **kwargs)

    monkeypatch.setattr(ring_pytorch_attn, "RingComm", lambda _group: _SingleRankComm())
    monkeypatch.setattr(ring_pytorch_attn, "pytorch_attn_forward", fake_attention)
    qkv = torch.randn(1, 4, 2, 8)

    output = ring_pytorch_attn.ring_pytorch_attn_func(
        qkv,
        qkv,
        qkv,
        group=object(),
        valid_kv_length=3,
    )

    assert output.shape == qkv.shape
    assert captured["k"].shape[1] == 3
    assert captured["v"].shape[1] == 3


def test_pytorch_ring_trims_circulated_blocks_by_source_rank(monkeypatch):
    observed_lengths = []
    # Each ring step sends K and V separately.
    blocks = [torch.randn(1, 4, 2, 8) for _ in range(4)]

    class FakeRingComm:
        rank = 2
        world_size = 3

        def __init__(self, _group):
            self._next_block = 0

        def send_recv(self, _tensor):
            block = blocks[self._next_block]
            self._next_block += 1
            return block

        def commit(self):
            pass

        def wait(self):
            pass

    def fake_attention(q, k, v, **kwargs):
        observed_lengths.append(k.shape[1])
        return _fake_attention_result(q, k, v, **kwargs)

    monkeypatch.setattr(ring_pytorch_attn, "RingComm", FakeRingComm)
    monkeypatch.setattr(ring_pytorch_attn, "pytorch_attn_forward", fake_attention)
    qkv = torch.randn(1, 4, 2, 8)

    ring_pytorch_attn.ring_pytorch_attn_func(
        qkv,
        qkv,
        qkv,
        group=object(),
        valid_kv_length=5,
    )

    # Rank 2 starts with global block 2, then receives blocks 1 and 0.
    # The first block is empty and therefore does not launch attention.
    assert observed_lengths == [1, 4]


def test_flash_ring_trims_padding_from_each_kv_block(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_attention(q, k, v, **kwargs):
        captured.update(k=k, v=v, **kwargs)
        return _fake_attention_result(q, k, v, **kwargs)

    monkeypatch.setattr(ring_flash_attn, "RingComm", lambda _group: _SingleRankComm())
    monkeypatch.setattr(ring_flash_attn, "select_flash_attn_impl", lambda *_args, **_kwargs: fake_attention)
    qkv = torch.randn(1, 4, 2, 8)

    output, _ = ring_flash_attn.ring_flash_attn_forward(
        object(),
        qkv,
        qkv,
        qkv,
        softmax_scale=0.25,
        causal=False,
        attn_type=AttnType.FA,
        valid_kv_length=3,
    )

    assert output.shape == qkv.shape
    assert captured["k"].shape[1] == 3
    assert captured["v"].shape[1] == 3


def test_ring_dispatch_forwards_global_valid_kv_length(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_ring_attention(*_args, **kwargs):
        captured.update(kwargs)
        return "output"

    monkeypatch.setattr(ring_pytorch_attn, "ring_pytorch_attn_func", fake_ring_attention)
    strategy = ring_parallel.RingParallelAttention(
        _FakeSequenceParallelGroup(ring_group=object()),
        attn_backend_pref="sdpa",
    )
    metadata = AttentionMetadata(extra={"valid_kv_length": 3})
    query = torch.randn(1, 4, 2, 8)

    assert strategy.run_attention(query, query, query, metadata) == "output"
    assert captured["valid_kv_length"] == 3


def test_pytorch_ring_rejects_causal_with_a_valid_prefix(monkeypatch):
    """Trimming a K/V block under causal=True would silently shift the diagonal.

    The query block keeps the full padded length while the circulated K/V block
    is trimmed, and a causal mask over a rectangular tile is bottom-right
    aligned, so the combination must be refused rather than computed.
    """
    monkeypatch.setattr(ring_pytorch_attn, "RingComm", lambda _group: _SingleRankComm())
    monkeypatch.setattr(ring_pytorch_attn, "pytorch_attn_forward", _fake_attention_result)
    qkv = torch.randn(1, 4, 2, 8)

    with pytest.raises(ValueError, match="valid_kv_length is not supported with causal=True"):
        ring_pytorch_attn.ring_pytorch_attn_func(
            qkv,
            qkv,
            qkv,
            causal=True,
            group=object(),
            valid_kv_length=3,
        )


def test_flash_ring_rejects_causal_with_a_valid_prefix(monkeypatch):
    """Same refusal on the Flash-Attention ring path."""
    monkeypatch.setattr(ring_flash_attn, "RingComm", lambda _group: _SingleRankComm())
    monkeypatch.setattr(
        ring_flash_attn,
        "select_flash_attn_impl",
        lambda *_args, **_kwargs: _fake_attention_result,
    )
    qkv = torch.randn(1, 4, 2, 8)

    with pytest.raises(ValueError, match="valid_kv_length is not supported with causal=True"):
        ring_flash_attn.ring_flash_attn_forward(
            object(),
            qkv,
            qkv,
            qkv,
            softmax_scale=0.25,
            causal=True,
            attn_type=AttnType.FA,
            valid_kv_length=3,
        )


def test_ring_paths_still_accept_causal_without_a_valid_prefix(monkeypatch):
    """The guard is scoped to the padded case: plain causal Ring is unchanged."""
    monkeypatch.setattr(ring_flash_attn, "RingComm", lambda _group: _SingleRankComm())
    monkeypatch.setattr(
        ring_flash_attn,
        "select_flash_attn_impl",
        lambda *_args, **_kwargs: _fake_attention_result,
    )
    qkv = torch.randn(1, 4, 2, 8)

    output, _ = ring_flash_attn.ring_flash_attn_forward(
        object(),
        qkv,
        qkv,
        qkv,
        softmax_scale=0.25,
        causal=True,
        attn_type=AttnType.FA,
    )

    assert output.shape == qkv.shape

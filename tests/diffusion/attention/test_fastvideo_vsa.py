# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (
    FastVideoVSAImpl,
)
from vllm_omni.diffusion.attention.backends.registry import (
    DiffusionAttentionBackendEnum,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_fastvideo_vsa_backend_is_registered():
    assert DiffusionAttentionBackendEnum.FASTVIDEO_VSA.get_path().endswith("fastvideo_vsa.FastVideoVSABackend")


def test_fastvideo_vsa_tiles_3d_sequence_and_untiles(monkeypatch):
    calls: dict[str, Any] = {}
    fake_module = types.ModuleType("fastvideo_kernel")

    def fake_video_sparse_attn_bshd(
        q,
        k,
        v,
        variable_block_sizes,
        q_variable_block_sizes,
        compress_attn_weight,
        topk,
        block_size,
    ):
        calls["q_shape"] = tuple(q.shape)
        calls["vbs"] = variable_block_sizes.detach().cpu().tolist()
        calls["q_vbs"] = q_variable_block_sizes.detach().cpu().tolist()
        calls["compress_sum"] = float(compress_attn_weight.detach().cpu().sum())
        calls["compress_shape"] = tuple(compress_attn_weight.shape)
        calls["topk"] = topk
        calls["block_size"] = block_size
        return q + k + v

    setattr(fake_module, "video_sparse_attn_bshd", fake_video_sparse_attn_bshd)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", fake_module)

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={
            "topk": 1,
            "block_size": (4, 8, 8),
            "min_seq_len": 1,
            "disable_when_sp_active": False,
        },
    )
    query = torch.randn(1, 300, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attn_metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (3, 10, 10)})
    monkeypatch.setattr(impl, "_fallback_reason", lambda *args, **kwargs: None)

    output = impl.forward_cuda(query, key, value, attn_metadata)

    assert output.shape == query.shape
    assert calls["q_shape"] == (1, 1024, 2, 8)
    assert calls["vbs"] == [192, 48, 48, 12]
    assert calls["q_vbs"] == [192, 48, 48, 12]
    assert calls["compress_shape"] == (1, 1024, 2, 8)
    assert calls["compress_sum"] == 0.0
    assert calls["topk"] == 1
    assert calls["block_size"] == (4, 8, 8)


def test_fastvideo_vsa_uses_learned_gate_when_provided(monkeypatch):
    calls: dict[str, Any] = {}
    fake_module = types.ModuleType("fastvideo_kernel")

    def fake_video_sparse_attn_bshd(
        q,
        k,
        v,
        variable_block_sizes,
        q_variable_block_sizes,
        compress_attn_weight,
        topk,
        block_size,
    ):
        calls["compress_sum"] = float(compress_attn_weight.detach().cpu().sum())
        calls["compress_shape"] = tuple(compress_attn_weight.shape)
        return q + k + v

    setattr(fake_module, "video_sparse_attn_bshd", fake_video_sparse_attn_bshd)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", fake_module)

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={
            "topk": 1,
            "block_size": (4, 8, 8),
            "min_seq_len": 1,
            "disable_when_sp_active": False,
        },
    )
    query = torch.randn(1, 300, 2, 8)
    gate = torch.ones_like(query)
    metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (3, 10, 10), "gate_compress": gate})
    monkeypatch.setattr(impl, "_fallback_reason", lambda *args, **kwargs: None)

    output = impl.forward_cuda(query, query, query, metadata)

    assert output.shape == query.shape
    assert calls["compress_shape"] == (1, 1024, 2, 8)
    # Only the 300 real tokens carry ones; padded gate slots stay zero.
    assert calls["compress_sum"] == gate.numel()


def test_fastvideo_vsa_falls_back_without_dit_shape(monkeypatch):
    calls: dict[str, Any] = {}

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={
            "topk": 1,
            "block_size": (4, 8, 8),
            "min_seq_len": 1,
            "disable_when_sp_active": False,
        },
    )

    def fake_fallback(query, key, value, attn_metadata, reason):
        calls["reason"] = reason
        return torch.zeros_like(query)

    monkeypatch.setattr(impl, "_fallback", fake_fallback)

    query = torch.randn(1, 512, 2, 8)
    output = impl.forward_cuda(query, query, query, AttentionMetadata())

    assert output.shape == query.shape
    assert calls["reason"] == "vsa_dit_seq_shape metadata is required"


def test_fastvideo_vsa_falls_back_for_mask(monkeypatch):
    calls: dict[str, Any] = {}

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={"min_seq_len": 1},
    )

    def fake_fallback(query, key, value, attn_metadata, reason):
        calls["reason"] = reason
        return torch.zeros_like(query)

    monkeypatch.setattr(impl, "_fallback", fake_fallback)

    query = torch.randn(1, 512, 2, 8)
    mask = torch.ones(1, 512, dtype=torch.bool)
    output = impl.forward_cuda(query, query, query, AttentionMetadata(attn_mask=mask))

    assert output.shape == query.shape
    assert calls["reason"]


def test_fastvideo_vsa_allows_topk_equal_to_num_blocks():
    query = torch.randn(1, 512, 2, 8)
    metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (4, 8, 16)})

    all_blocks = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={"topk": 2, "block_size": (4, 8, 8), "min_seq_len": 1},
    )
    too_many = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={"topk": 3, "block_size": (4, 8, 8), "min_seq_len": 1},
    )

    # CPU/float32 is rejected later, but k=N itself must not trigger fallback.
    assert all_blocks._fallback_reason(query, query, query, metadata) == "dtype torch.float32 is not supported"
    assert too_many._fallback_reason(query, query, query, metadata) == "topk 3 > num_blocks 2"

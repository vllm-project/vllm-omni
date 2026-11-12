# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the MiniMax-H3 Video VAE NPU fused-infer attention gate."""

import sys
import types

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3 import vae_npu_fia as fia_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.training = False
        self.calls = []

        def original(query, key, value, pack_info):
            self.calls.append((query, key, value, pack_info))
            return query

        self.perform_attention = original
        self._original = original


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _FakeAttn()


class _FakeDecoder(nn.Module):
    def __init__(self, num_blocks=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_FakeBlock() for _ in range(num_blocks)])


@pytest.fixture
def decoder():
    return _FakeDecoder()


def _install(decoder, device):
    fia_module.install_vae_npu_fia(decoder, device, types.SimpleNamespace(info=lambda *a, **k: None))


def test_non_npu_installer_is_silent(decoder):
    _install(decoder, torch.device("cuda"))
    assert getattr(decoder, "_h3_npu_fia_stats", None) is None
    assert all(block.attn.perform_attention is block.attn._original for block in decoder.transformer_blocks)


def test_npu_installer_wraps_blocks_and_marks_attention(decoder):
    _install(decoder, torch.device("npu:0"))
    assert decoder._h3_npu_fia_stats == {"calls": 0, "fallbacks": 0}
    for block in decoder.transformer_blocks:
        assert block.attn.perform_attention is not block.attn._original
        assert block.attn._h3_decoder_fused_infer_attention is True


def test_environment_switch_disables_installation(decoder, monkeypatch):
    monkeypatch.setenv("H3_VIDEO_DECODER_FUSED_INFER_ATTENTION", "0")
    _install(decoder, torch.device("npu:0"))
    assert getattr(decoder, "_h3_npu_fia_stats", None) is None
    assert all(block.attn.perform_attention is block.attn._original for block in decoder.transformer_blocks)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"training": True}, "training"),
        ({"requires_grad": True}, "gradient"),
        ({"pack_info": {"cu_seqlens": torch.tensor([0, 1])}}, "varlen"),
        ({"pack_info": {"mask_mod": object()}}, "masked"),
        ({"pack_info": {"block_sparse": object()}}, "block-sparse"),
    ],
)
def test_unsupported_calls_fall_back(decoder, kwargs, reason):
    _install(decoder, torch.device("npu:0"))
    attn = decoder.transformer_blocks[0].attn
    attn.training = kwargs.get("training", False)
    query = torch.zeros(1, 2, 3, 4)
    if kwargs.get("requires_grad"):
        query.requires_grad_(True)
    pack_info = kwargs.get("pack_info", {})
    out = attn.perform_attention(query, query, query, pack_info)
    assert out is query
    assert attn.calls and attn.calls[-1][3] is pack_info
    assert decoder._h3_npu_fia_stats["fallbacks"] == 1
    assert reason  # documents which gate fired


def test_non_npu_tensor_falls_back(decoder):
    _install(decoder, torch.device("npu:0"))
    attn = decoder.transformer_blocks[0].attn
    query = torch.zeros(1, 2, 3, 4)
    out = attn.perform_attention(query, query, query, {})
    assert out is query
    assert decoder._h3_npu_fia_stats["fallbacks"] == 1


def test_fast_path_calls_fused_infer_attention(decoder, monkeypatch):
    _install(decoder, torch.device("npu:0"))
    attn = decoder.transformer_blocks[0].attn
    calls = []

    def fake_fia(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q, None

    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu.npu_fused_infer_attention_score = fake_fia
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setattr(fia_module, "_fallback_reason", lambda *args: None)

    query = torch.arange(1 * 2 * 3 * 4, dtype=torch.float16).reshape(1, 2, 3, 4)
    out = attn.perform_attention(query, query, query, {})

    assert len(calls) == 1
    q, k, v, kwargs = calls[0]
    assert q.shape == (1, 3, 2, 4)
    assert k.shape == q.shape and v.shape == q.shape
    assert kwargs["num_heads"] == 2
    assert kwargs["num_key_value_heads"] == 2
    assert kwargs["input_layout"] == "BNSD"
    assert kwargs["scale"] == pytest.approx(4**-0.5)
    assert kwargs["sparse_mode"] == 0
    assert kwargs["softmax_lse_flag"] is False
    assert out.shape == query.shape
    assert decoder._h3_npu_fia_stats["calls"] == 1
    assert not attn.calls

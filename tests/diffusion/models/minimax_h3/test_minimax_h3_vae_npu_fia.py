# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the MiniMax-H3 Video VAE NPU fused-infer attention gate."""

import ast
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3 import vae_npu_fia as fia_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_CONFIG_KEY = "video_vae_fused_infer_attention"


class _FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.training = False
        self.calls: list[tuple[object, object, object, dict[str, object]]] = []

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


class _NPUTensor:
    def __init__(self, tensor, device, *, requires_grad=False):
        self.tensor = tensor
        self.device = device
        self.shape = tensor.shape
        self.ndim = tensor.ndim
        self.dtype = tensor.dtype
        self.requires_grad = requires_grad

    def transpose(self, dim0, dim1):
        return self.tensor.transpose(dim0, dim1)


@pytest.fixture
def decoder():
    return _FakeDecoder()


@pytest.fixture
def npu_device():
    return Mock(spec=torch.device, type="npu")


def _install(decoder, device, *, enabled=True):
    fia_module.install_vae_npu_fia(decoder, device, Mock(), enabled=enabled)


def _assert_not_installed(decoder):
    assert getattr(decoder, "_h3_npu_fia_stats", None) is None
    assert all(block.attn.perform_attention is block.attn._original for block in decoder.transformer_blocks)


def test_non_npu_installer_is_silent(decoder):
    _install(decoder, torch.device("cuda"), enabled=False)
    _assert_not_installed(decoder)


def test_non_npu_installer_is_silent_when_enabled(decoder):
    _install(decoder, torch.device("cuda"))
    _assert_not_installed(decoder)


def test_npu_installer_wraps_blocks_and_marks_attention(decoder, npu_device):
    _install(decoder, npu_device, enabled=fia_module.resolve_video_vae_fused_infer_attention({_CONFIG_KEY: True}))
    assert decoder._h3_npu_fia_stats == {"calls": 0, "fallbacks": 0}
    for block in decoder.transformer_blocks:
        assert block.attn.perform_attention is not block.attn._original
        assert block.attn._h3_decoder_fused_infer_attention is True


def test_config_switch_disables_installation(decoder, npu_device):
    _install(decoder, npu_device, enabled=fia_module.resolve_video_vae_fused_infer_attention({_CONFIG_KEY: False}))
    _assert_not_installed(decoder)


def test_default_off_installs_nothing(decoder, npu_device):
    _install(decoder, npu_device, enabled=fia_module.resolve_video_vae_fused_infer_attention(None))
    _assert_not_installed(decoder)


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None])
def test_config_rejects_non_boolean(value):
    with pytest.raises(ValueError, match=_CONFIG_KEY):
        fia_module.resolve_video_vae_fused_infer_attention({_CONFIG_KEY: value})


@pytest.mark.parametrize("module_name", ["pipeline_minimax_h3", "pipeline_minimax_h3_decoder"])
def test_pipeline_passes_fia_config_to_video_vae(module_name):
    module = importlib.import_module(f"vllm_omni.diffusion.models.minimax_h3.{module_name}")
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "MiniMaxH3VideoVAE"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert (
        keywords["use_fused_infer_attention"] == "resolve_video_vae_fused_infer_attention(od_config.additional_config)"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"training": True},
        {"requires_grad": True},
        {"pack_info": {"cu_seqlens": torch.tensor([0, 1])}},
        {"pack_info": {"mask_mod": object()}},
        {"pack_info": {"block_sparse": object()}},
    ],
    ids=["training", "gradient", "varlen", "masked", "block-sparse"],
)
def test_unsupported_calls_fall_back(decoder, npu_device, kwargs, monkeypatch):
    fused = Mock()
    monkeypatch.setattr(fia_module, "_fused_infer_attention", fused)
    _install(decoder, npu_device)
    attn = decoder.transformer_blocks[0].attn
    attn.training = kwargs.get("training", False)
    query = _NPUTensor(
        torch.zeros(1, 2, 3, 4, dtype=torch.float16), npu_device, requires_grad=kwargs.get("requires_grad", False)
    )
    pack_info = kwargs.get("pack_info", {})
    out = attn.perform_attention(query, query, query, pack_info)
    assert out is query
    assert attn.calls and attn.calls[-1][3] is pack_info
    assert decoder._h3_npu_fia_stats == {"calls": 0, "fallbacks": 1}
    fused.assert_not_called()


def test_non_npu_tensor_falls_back(decoder, npu_device):
    _install(decoder, npu_device)
    attn = decoder.transformer_blocks[0].attn
    query = torch.zeros(1, 2, 3, 4)
    out = attn.perform_attention(query, query, query, {})
    assert out is query
    assert decoder._h3_npu_fia_stats["fallbacks"] == 1


def test_mismatched_qkv_shapes_fall_back(decoder, npu_device, monkeypatch):
    fused = Mock()
    monkeypatch.setattr(fia_module, "_fused_infer_attention", fused)
    _install(decoder, npu_device)
    attn = decoder.transformer_blocks[0].attn
    query = _NPUTensor(torch.zeros(1, 2, 3, 4, dtype=torch.float16), npu_device)
    key = _NPUTensor(torch.zeros(1, 3, 3, 4, dtype=torch.float16), npu_device)
    assert attn.perform_attention(query, key, query, {}) is query
    assert attn.calls[-1][1] is key
    assert decoder._h3_npu_fia_stats == {"calls": 0, "fallbacks": 1}
    fused.assert_not_called()


def test_empty_qkv_falls_back(decoder, npu_device, monkeypatch):
    fused = Mock()
    monkeypatch.setattr(fia_module, "_fused_infer_attention", fused)
    _install(decoder, npu_device)
    attn = decoder.transformer_blocks[0].attn
    query = _NPUTensor(torch.zeros(1, 0, 3, 4, dtype=torch.float16), npu_device)
    assert attn.perform_attention(query, query, query, {}) is query
    assert decoder._h3_npu_fia_stats == {"calls": 0, "fallbacks": 1}
    fused.assert_not_called()


def test_shape_envelope_accepts_qualifying_geometry():
    query = torch.zeros(2, 8, 3, 4)
    assert fia_module._shape_fallback_reason(query, query, query) is None


def test_qualifying_call_is_accepted(decoder, npu_device, monkeypatch):
    result = torch.ones(1, 8, 3, 4, dtype=torch.bfloat16)
    fused = Mock(return_value=result)
    monkeypatch.setattr(fia_module, "_fused_infer_attention", fused)
    _install(decoder, npu_device)
    attn = decoder.transformer_blocks[0].attn
    query = _NPUTensor(torch.zeros_like(result), npu_device)
    assert attn.perform_attention(query, query, query, {}) is result
    fused.assert_called_once_with(query, query, query)
    assert not attn.calls
    assert decoder._h3_npu_fia_stats == {"calls": 1, "fallbacks": 0}


def test_dispatch_supports_only_npu(npu_device):
    assert fia_module.supports_h3_vae_npu_fia(npu_device)
    assert not fia_module.supports_h3_vae_npu_fia(torch.device("cuda:0"))


def test_fast_path_calls_fused_infer_attention(decoder, npu_device, monkeypatch):
    _install(decoder, npu_device)
    attn = decoder.transformer_blocks[0].attn
    calls = []

    def fake_fia(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q, None

    fake_torch_npu = types.ModuleType("torch_npu")
    setattr(fake_torch_npu, "npu_fused_infer_attention_score", fake_fia)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    tensor = torch.arange(1 * 2 * 3 * 4, dtype=torch.float16).reshape(1, 2, 3, 4)
    query = _NPUTensor(tensor, npu_device)
    out = attn.perform_attention(query, query, query, {})

    assert len(calls) == 1
    q, k, v, kwargs = calls[0]
    assert q.shape == (1, 3, 2, 4)
    assert k.shape == q.shape and v.shape == q.shape
    assert kwargs["num_heads"] == 3
    assert kwargs["num_key_value_heads"] == 3
    assert kwargs["input_layout"] == "BNSD"
    assert kwargs["scale"] == pytest.approx(4**-0.5)
    assert kwargs["sparse_mode"] == 0
    assert kwargs["softmax_lse_flag"] is False
    assert out.shape == query.shape
    assert decoder._h3_npu_fia_stats["calls"] == 1
    assert not attn.calls


def test_fia_error_is_not_swallowed(decoder, npu_device, monkeypatch):
    fused = Mock(side_effect=RuntimeError("FIA failed"))
    monkeypatch.setattr(fia_module, "_fused_infer_attention", fused)
    _install(decoder, npu_device)
    attn = decoder.transformer_blocks[0].attn
    query = _NPUTensor(torch.zeros(1, 2, 3, 4, dtype=torch.float16), npu_device)
    with pytest.raises(RuntimeError, match="FIA failed"):
        attn.perform_attention(query, query, query, {})
    assert not attn.calls

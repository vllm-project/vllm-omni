# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU operator checks; overriding device admission is not NPU validation.

Load the standalone operator without importing the serving stack. The real
padding and convolution execute on CPU, with only NPU device admission
overridden in tests that exercise the candidate path.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_FLAG = "VLLM_OMNI_H3_VAE_TEMPORAL_PAD_FUSION"


@pytest.fixture
def ops() -> ModuleType:
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/ops/vae/npu_temporal_pad.py"
    spec = importlib.util.spec_from_file_location("h3_temporal_pad_operator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpatialParallelConv3d(nn.Conv3d):
    """Small replica of the public checkpoint's causal convolution ABI."""

    def __init__(self, padding=(1, 1, 1), stride=(1, 1, 1), pad_mode="reflect", bias=True):
        super().__init__(
            2,
            3,
            3,
            padding=padding,
            stride=stride,
            bias=bias,
            padding_mode="zeros" if pad_mode == "constant" else pad_mode,
        )
        self.pad_mode = pad_mode
        self.pad_mode_t = "constant"
        self.causal = True
        self.spatial_parallel = False
        self.original_calls = 0

    def _apply_padding(self, x):
        ph, pw = self.padding[1:]
        x = F.pad(x, (pw, pw, ph, ph, 0, 0), mode=self.pad_mode)
        if x.shape[2] > 1:
            return F.pad(x, (0, 0, 0, 0, 2 * self.padding[0], 0))
        zeros = torch.zeros_like(x[:, :, :1]).expand(-1, -1, self.kernel_size[0] - 1, -1, -1)
        return torch.cat((zeros, x), dim=2)

    def forward(self, x):
        self.original_calls += 1
        return F.conv3d(self._apply_padding(x), self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class EncoderFCN3D(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = SpatialParallelConv3d(**kwargs)


def _model(**kwargs) -> nn.Module:
    model = nn.Module()
    model.encoder = EncoderFCN3D(**kwargs).eval()
    model.decoder = nn.Conv3d(2, 2, 1)
    model.quant_conv = nn.Conv3d(2, 2, 1)
    return model


def _install(ops: ModuleType, model: nn.Module, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setattr(ops, "_is_npu_device", lambda device: device.type == "cpu")
    return ops.install_npu_temporal_pad_fusion(model, device=torch.device("cpu"))


@pytest.mark.parametrize("padding", [(1, 0, 0), (1, 1, 1)])
@pytest.mark.parametrize("stride", [(1, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("pad_mode", ["constant", "reflect"])
@pytest.mark.parametrize("bias", [True, False])
def test_integer_prefix_matches_reference(ops, monkeypatch, padding, stride, pad_mode, bias):
    model = _model(padding=padding, stride=stride, pad_mode=pad_mode, bias=bias)
    conv = model.encoder.conv
    x = torch.arange(2 * 2 * 17 * 6 * 7, dtype=torch.float32).reshape(2, 2, 17, 6, 7) % 5 - 2
    with torch.no_grad():
        conv.weight.copy_(torch.arange(conv.weight.numel()).reshape_as(conv.weight) % 3 - 1)
        if conv.bias is not None:
            conv.bias.fill_(1)
        expected = conv(x)
        calls = conv.original_calls
        assert _install(ops, model, monkeypatch) == 1
        actual = conv(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert conv.original_calls == calls
    assert actual.shape[2] == (17 if stride[0] == 1 else 9)


def test_random_fp32_prefix_matches_reference(ops, monkeypatch):
    model = _model()
    x = torch.randn(1, 2, 17, 7, 8, generator=torch.Generator().manual_seed(41))
    with torch.no_grad():
        expected = model.encoder.conv(x)
        assert _install(ops, model, monkeypatch) == 1
        actual = model.encoder.conv(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_spatial_pad_is_preserved_and_temporal_pad_is_removed(ops, monkeypatch):
    model = _model(padding=(1, 1, 1))
    assert _install(ops, model, monkeypatch) == 1
    calls = []
    original = F.pad

    def record(x, pad, *args, **kwargs):
        calls.append(pad)
        return original(x, pad, *args, **kwargs)

    monkeypatch.setattr(F, "pad", record)
    with torch.no_grad():
        model.encoder.conv(torch.ones(1, 2, 17, 6, 7))
    assert calls == [(1, 1, 1, 1, 0, 0)]


@pytest.mark.parametrize("frames", [1, 5, 9])
def test_other_frame_counts_use_original_forward(ops, monkeypatch, frames):
    model = _model()
    assert _install(ops, model, monkeypatch) == 1
    conv = model.encoder.conv
    with torch.no_grad():
        conv(torch.ones(1, 2, frames, 6, 7))
    assert conv.original_calls == 1


@pytest.mark.parametrize(
    "reason", ["training", "gradients", "compile", "autocast", "spatial_parallel", "dtype", "hooks"]
)
def test_execution_contract_falls_back(ops, monkeypatch, reason):
    model = _model()
    assert _install(ops, model, monkeypatch) == 1
    conv = model.encoder.conv
    x = torch.ones(1, 2, 17, 6, 7)
    if reason == "training":
        conv.train()
    elif reason == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif reason == "autocast":
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: True)
    elif reason == "spatial_parallel":
        conv.spatial_parallel = True
    elif reason == "dtype":
        conv.double()
        x = x.double()
    elif reason == "hooks":
        monkeypatch.setitem(conv._forward_hooks, 0, lambda module, args, output: output)
    with torch.set_grad_enabled(reason == "gradients"):
        conv(x)
    assert conv.original_calls == 1


def test_cpu_is_not_admitted(ops, monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    model = _model()
    assert ops.install_npu_temporal_pad_fusion(model, device=torch.device("cpu")) == 0
    assert "forward" not in vars(model.encoder.conv)


@pytest.mark.parametrize("value", [None, "0", "false", "off", ""])
def test_default_and_disabled_values_do_not_install(ops, monkeypatch, value):
    monkeypatch.setattr(ops, "_is_npu_device", lambda device: True)
    if value is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, value)
    assert ops.install_npu_temporal_pad_fusion(_model(), device=torch.device("cpu")) == 0


@pytest.mark.parametrize("value", ["1", "true", "ON"])
def test_enabled_values_install(ops, monkeypatch, value):
    monkeypatch.setattr(ops, "_is_npu_device", lambda device: True)
    monkeypatch.setenv(_FLAG, value)
    assert ops.install_npu_temporal_pad_fusion(_model(), device=torch.device("cpu")) == 1


def test_invalid_flag_raises(ops, monkeypatch):
    monkeypatch.setattr(ops, "_is_npu_device", lambda device: True)
    monkeypatch.setenv(_FLAG, "maybe")
    with pytest.raises(ValueError, match=_FLAG):
        ops.install_npu_temporal_pad_fusion(_model(), device=torch.device("cpu"))


def test_unknown_encoder_and_decode_only_model_are_unchanged(ops, monkeypatch):
    monkeypatch.setattr(ops, "_is_npu_device", lambda device: True)
    monkeypatch.setenv(_FLAG, "1")
    assert ops.install_npu_temporal_pad_fusion(nn.Module(), device=torch.device("cpu")) == 0
    model = nn.Module()
    model.encoder = nn.Sequential(SpatialParallelConv3d())
    assert ops.install_npu_temporal_pad_fusion(model, device=torch.device("cpu")) == 0


def test_padding_abi_mismatch_declines(ops, monkeypatch):
    model = _model()
    monkeypatch.setattr(SpatialParallelConv3d, "_apply_padding", lambda self, x: x)
    assert _install(ops, model, monkeypatch) == 0
    assert "forward" not in vars(model.encoder.conv)


def test_padding_probe_oom_propagates(ops, monkeypatch):
    model = _model()

    def oom(self, x):
        raise torch.OutOfMemoryError("probe")

    monkeypatch.setattr(SpatialParallelConv3d, "_apply_padding", oom)
    with pytest.raises(torch.OutOfMemoryError, match="probe"):
        _install(ops, model, monkeypatch)


def test_existing_instance_forward_is_preserved(ops, monkeypatch):
    model = _model()
    original = model.encoder.conv.forward
    model.encoder.conv.forward = original
    assert _install(ops, model, monkeypatch) == 0
    assert model.encoder.conv.forward is original


def test_installation_is_idempotent_and_encoder_only(ops, monkeypatch):
    model = _model()
    state = {name: value.data_ptr() for name, value in model.state_dict().items()}
    assert _install(ops, model, monkeypatch) == 1
    forward = model.encoder.conv.forward
    assert _install(ops, model, monkeypatch) == 1
    assert model.encoder.conv.forward is forward
    assert "forward" not in vars(model.decoder)
    assert "forward" not in vars(model.quant_conv)
    assert {name: value.data_ptr() for name, value in model.state_dict().items()} == state

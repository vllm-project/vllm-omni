# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU operator tests; the candidate's device guard is overridden explicitly.

No serving stack, model weights, or accelerator is loaded. Actual CPU tensor
operations check padding and downsampling equivalence, not NPU accuracy.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_FLAG = "VLLM_OMNI_H3_VAE_SKIP_NOOP_PAD"


@pytest.fixture
def ops() -> ModuleType:
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/ops/vae/npu_skip_noop_pad.py"
    spec = importlib.util.spec_from_file_location("h3_skip_noop_pad_operator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpatialParallelConv3d(nn.Conv3d):
    def __init__(self, padding=(1, 0, 0), stride=(1, 2, 2)):
        super().__init__(2, 3, 3, padding=padding, stride=stride, padding_mode="reflect")
        self.pad_mode = "reflect"
        self.pad_mode_t = "constant"
        self.causal = True
        self.spatial_parallel = False
        self.original_pad_calls = 0

    def _apply_padding(self, x):
        self.original_pad_calls += 1
        ph, pw = self.padding[1:]
        x = F.pad(x, (pw, pw, ph, ph, 0, 0), mode=self.pad_mode)
        if x.shape[2] > 1:
            return F.pad(x, (0, 0, 0, 0, 2 * self.padding[0], 0))
        zeros = torch.zeros_like(x[:, :, :1]).expand(-1, -1, self.kernel_size[0] - 1, -1, -1)
        return torch.cat((zeros, x), dim=2)

    def forward(self, x):
        return F.conv3d(self._apply_padding(x), self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class Downsample3D(nn.Module):
    def __init__(self, padding=(1, 0, 0), time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = SpatialParallelConv3d(padding=padding, stride=(time_stride, space_stride, space_stride))

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.conv.pad_mode)
        return self.conv(x)


class EncoderFCN3D(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.downsample = Downsample3D(**kwargs)


def _model(**kwargs) -> nn.Module:
    model = nn.Module()
    model.encoder = EncoderFCN3D(**kwargs).eval()
    model.decoder = nn.Conv3d(2, 2, 1)
    model.quant_conv = nn.Conv3d(2, 2, 1)
    return model


def _install(ops: ModuleType, model: nn.Module, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setattr(ops, "_is_npu_device", lambda device: device.type == "cpu")
    return ops.install_npu_skip_noop_pad(model, device=torch.device("cpu"))


@pytest.mark.parametrize("frames", [2, 5, 9, 17])
@pytest.mark.parametrize("time_stride", [1, 2])
@pytest.mark.parametrize("space_stride", [1, 2])
def test_video_downsampling_is_bitwise_equal(ops, monkeypatch, frames, time_stride, space_stride):
    model = _model(time_stride=time_stride, space_stride=space_stride)
    downsample = model.encoder.downsample
    x = torch.randn(2, 2, frames, 6, 7, generator=torch.Generator().manual_seed(53))
    with torch.no_grad():
        expected = downsample(x)
        assert _install(ops, model, monkeypatch) == 1
        calls = downsample.conv.original_pad_calls
        actual = downsample(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert downsample.conv.original_pad_calls == calls


def test_only_empty_spatial_pad_is_removed(ops, monkeypatch):
    model = _model()
    assert _install(ops, model, monkeypatch) == 1
    calls = []
    original = F.pad

    def record(x, pad, *args, **kwargs):
        calls.append(pad)
        return original(x, pad, *args, **kwargs)

    monkeypatch.setattr(F, "pad", record)
    with torch.no_grad():
        model.encoder.downsample(torch.ones(1, 2, 17, 6, 7))
    assert calls == [(0, 1, 0, 1, 0, 0), (0, 0, 0, 0, 2, 0)]


def test_images_keep_original_padding(ops, monkeypatch):
    model = _model()
    assert _install(ops, model, monkeypatch) == 1
    conv = model.encoder.downsample.conv
    calls = conv.original_pad_calls
    with torch.no_grad():
        conv(torch.ones(1, 2, 1, 6, 7))
    assert conv.original_pad_calls == calls + 1


@pytest.mark.parametrize(
    "reason", ["training", "gradients", "compile", "autocast", "spatial_parallel", "dtype", "strides", "hooks"]
)
def test_execution_contract_falls_back(ops, monkeypatch, reason):
    model = _model()
    assert _install(ops, model, monkeypatch) == 1
    conv = model.encoder.downsample.conv
    calls = conv.original_pad_calls
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
    elif reason == "strides":
        x = x.transpose(-1, -2)
        assert not x.is_contiguous()
    elif reason == "hooks":
        monkeypatch.setitem(conv._forward_hooks, 0, lambda module, args, output: output)
    with torch.set_grad_enabled(reason == "gradients"):
        conv(x)
    assert conv.original_pad_calls == calls + 1


def test_nonzero_spatial_padding_is_not_installed(ops, monkeypatch):
    model = _model(padding=(1, 1, 1))
    assert _install(ops, model, monkeypatch) == 0
    assert "_apply_padding" not in vars(model.encoder.downsample.conv)


def test_cpu_is_not_admitted(ops, monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    model = _model()
    assert ops.install_npu_skip_noop_pad(model, device=torch.device("cpu")) == 0
    assert "_apply_padding" not in vars(model.encoder.downsample.conv)


@pytest.mark.parametrize("value", [None, "0", "false", "off", ""])
def test_default_and_disabled_values_do_not_install(ops, monkeypatch, value):
    monkeypatch.setattr(ops, "_is_npu_device", lambda device: True)
    if value is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, value)
    assert ops.install_npu_skip_noop_pad(_model(), device=torch.device("cpu")) == 0


@pytest.mark.parametrize("value", ["1", "true", "ON"])
def test_enabled_values_install(ops, monkeypatch, value):
    monkeypatch.setattr(ops, "_is_npu_device", lambda device: True)
    monkeypatch.setenv(_FLAG, value)
    assert ops.install_npu_skip_noop_pad(_model(), device=torch.device("cpu")) == 1


def test_invalid_flag_raises(ops, monkeypatch):
    monkeypatch.setattr(ops, "_is_npu_device", lambda device: True)
    monkeypatch.setenv(_FLAG, "maybe")
    with pytest.raises(ValueError, match=_FLAG):
        ops.install_npu_skip_noop_pad(_model(), device=torch.device("cpu"))


def test_unknown_encoder_and_decode_only_model_are_unchanged(ops, monkeypatch):
    monkeypatch.setattr(ops, "_is_npu_device", lambda device: True)
    monkeypatch.setenv(_FLAG, "1")
    assert ops.install_npu_skip_noop_pad(nn.Module(), device=torch.device("cpu")) == 0
    model = nn.Module()
    model.encoder = nn.Sequential(SpatialParallelConv3d())
    assert ops.install_npu_skip_noop_pad(model, device=torch.device("cpu")) == 0


def test_encoder_from_unregistered_module_is_unchanged(ops, monkeypatch):
    model = _model()
    monkeypatch.setattr(EncoderFCN3D, "__module__", f"{__name__}.unregistered")
    assert _install(ops, model, monkeypatch) == 0
    assert "_apply_padding" not in vars(model.encoder.downsample.conv)


def test_changed_padding_abi_declines(ops, monkeypatch):
    model = _model()
    monkeypatch.setattr(SpatialParallelConv3d, "_apply_padding", lambda self, x: x)
    assert _install(ops, model, monkeypatch) == 0


def test_padding_probe_oom_propagates(ops, monkeypatch):
    model = _model()

    def oom(self, x):
        raise torch.OutOfMemoryError("probe")

    monkeypatch.setattr(SpatialParallelConv3d, "_apply_padding", oom)
    with pytest.raises(torch.OutOfMemoryError, match="probe"):
        _install(ops, model, monkeypatch)


def test_existing_instance_padding_method_is_preserved(ops, monkeypatch):
    model = _model()
    conv = model.encoder.downsample.conv
    original = conv._apply_padding
    conv._apply_padding = original
    assert _install(ops, model, monkeypatch) == 0
    assert conv._apply_padding is original


def test_original_forward_and_non_encoder_modules_are_preserved(ops, monkeypatch):
    model = _model()
    conv = model.encoder.downsample.conv
    forward = conv.forward
    state = {name: value.data_ptr() for name, value in model.state_dict().items()}
    assert _install(ops, model, monkeypatch) == 1
    patched = conv._apply_padding
    assert _install(ops, model, monkeypatch) == 1
    assert conv._apply_padding is patched
    assert conv.forward == forward
    assert "forward" not in vars(conv)
    assert "_apply_padding" not in vars(model.decoder)
    assert "_apply_padding" not in vars(model.quant_conv)
    assert {name: value.data_ptr() for name, value in model.state_dict().items()} == state

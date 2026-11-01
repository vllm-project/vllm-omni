# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the vLLM-Omni to MindIE-SD MXFP8 fallback adapter."""

from types import ModuleType, SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Module

from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _FakePolicy:
    def __init__(self):
        self.steps = ()

    def register(self, steps, strategy, target):
        assert strategy == "W16A16"
        assert target == "mxfp8_linear"
        self.steps = tuple(steps)


class _FakeFallback:
    def __init__(self, policy, master_mode="cpu"):
        self.policy = policy
        self.master_mode = master_mode

    def save_master(self, layer):
        master = layer.weight.detach().to(device="cpu", dtype=torch.bfloat16).transpose(0, 1).contiguous()
        if self.master_mode == "resident":
            master = master.to(layer.weight.device)
        layer._mxfp8_prefix_master = master
        layer._mxfp8_prefix_master_mode = self.master_mode

    @staticmethod
    def get_master(layer):
        return getattr(layer, "_mxfp8_prefix_master", None)

    def use_bf16(self, step):
        return step is not None and step in self.policy.steps

    def apply(self, layer, x, bias, ori_dtype, step):
        if not self.use_bf16(step):
            return None
        master = self.get_master(layer)
        if master is None:
            raise RuntimeError("BF16 prefix requested without the original master weight")
        output = torch.matmul(x, master.to(device=x.device, dtype=x.dtype))
        if bias is not None:
            output = output + bias.to(output.dtype)
        return output.to(ori_dtype)


def _install_fake_mindiesd(monkeypatch):
    import sys

    quantization = ModuleType("mindiesd.quantization")
    quantization.MXFP8BF16Fallback = _FakeFallback
    quantization.TimestepPolicyConfig = _FakePolicy
    mindiesd = ModuleType("mindiesd")
    mindiesd.quantization = quantization
    monkeypatch.setitem(sys.modules, "mindiesd", mindiesd)
    monkeypatch.setitem(sys.modules, "mindiesd.quantization", quantization)


def _set_step_context(monkeypatch, step, available=True):
    import sys

    context = SimpleNamespace(denoise_step_idx=step)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.forward_context",
        SimpleNamespace(get_forward_context=lambda: context, is_forward_context_available=lambda: available),
    )


def _online_method(steps=1, mode="cpu"):
    from vllm_omni.quantization.mxfp8_config import NPUMxfp8OnlineLinearMethod

    return NPUMxfp8OnlineLinearMethod(DiffusionMXFP8Config(prefix_bf16_steps=steps, prefix_bf16_master_mode=mode))


@pytest.mark.parametrize("value,expected", [(None, 0), ("0", 0), ("1", 1)])
def test_mxfp8_scale_alg(monkeypatch, value, expected):
    from vllm_omni.quantization.mxfp8_config import _mx_scale_alg

    monkeypatch.delenv("VLLM_OMNI_MXFP8_SCALE_ALG", raising=False)
    if value is not None:
        monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_ALG", value)
    assert _mx_scale_alg() == expected


def test_mxfp8_activation_passes_scale_alg(monkeypatch, mocker: MockerFixture):
    import sys

    from vllm_omni.quantization.mxfp8_config import NPUMxfp8LinearMethod

    monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_ALG", "1")
    quantize = mocker.Mock(return_value=("quantized", "scale"))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_dynamic_mx_quant=quantize, float8_e4m3fn="fp8"))
    method = NPUMxfp8LinearMethod(DiffusionMXFP8Config())
    x = torch.ones(2, 64, dtype=torch.bfloat16)
    assert method._quantize_activation(x) == ("quantized", "scale")
    quantize.assert_called_once_with(x, dst_type="fp8", scale_alg=1)


@pytest.mark.parametrize("value", [-1, 1.5, "1", True])
def test_mxfp8_prefix_rejects_invalid_steps(value):
    error = ValueError if isinstance(value, int) and not isinstance(value, bool) else TypeError
    with pytest.raises(error, match="prefix_bf16_steps"):
        DiffusionMXFP8Config(prefix_bf16_steps=value)


def test_mxfp8_prefix_rejects_unknown_mode_and_serialized_checkpoint():
    with pytest.raises(ValueError, match="Unknown MXFP8 prefix master mode"):
        DiffusionMXFP8Config(prefix_bf16_master_mode="invalid")
    with pytest.raises(ValueError, match="requires online MXFP8 weights"):
        DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=True, prefix_bf16_steps=1)


def test_mxfp8_prefix_config_roundtrip():
    from vllm_omni.quantization import build_quant_config

    assert DiffusionMXFP8Config.from_config({}).prefix_bf16_steps == 0
    assert DiffusionMXFP8Config.from_config({}).prefix_bf16_master_mode == "cpu"
    cfg = build_quant_config({"method": "mxfp8", "prefix_bf16_steps": 4, "prefix_bf16_master_mode": "resident"})
    assert isinstance(cfg, DiffusionMXFP8Config)
    assert cfg.prefix_bf16_steps == 4
    assert cfg.prefix_bf16_master_mode == "resident"


def test_mxfp8_prefix_saves_master_before_online_quantization(monkeypatch, mocker: MockerFixture):
    import sys

    _install_fake_mindiesd(monkeypatch)
    method = _online_method(steps=1)
    layer = Module()
    original = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    expected = original.transpose(0, 1).contiguous().clone()
    layer.register_parameter("weight", torch.nn.Parameter(original.clone(), requires_grad=False))

    def quantize(weight, **kwargs):
        assert kwargs["scale_alg"] == 1
        weight.zero_()
        return weight, torch.ones(4, 2, dtype=torch.uint8)

    quantize_mock = mocker.Mock(side_effect=quantize)
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_dynamic_mx_quant=quantize_mock, float8_e4m3fn="fp8")
    )
    monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_ALG", "1")
    method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer._mxfp8_prefix_master, expected, rtol=0, atol=0)
    assert layer._mxfp8_prefix_master.device.type == "cpu"
    assert "_mxfp8_prefix_master" not in layer.state_dict()
    quantize_mock.assert_called_once()


def test_mxfp8_prefix_dlo_uses_cpu_core_master(monkeypatch):
    _install_fake_mindiesd(monkeypatch)
    method = _online_method(steps=5, mode="dlo")
    layer = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
    expected = layer.weight.detach().transpose(0, 1).contiguous().clone()
    method._bf16_fallback.save_master(layer)
    layer._mxfp8_prefix_master_mode = method.quant_config.prefix_bf16_master_mode

    assert layer._mxfp8_prefix_master.device.type == "cpu"
    torch.testing.assert_close(layer._mxfp8_prefix_master, expected, rtol=0, atol=0)
    k, tensors, replaced = method.get_layerwise_prefix_weights(layer)
    assert k == 5
    assert replaced == ("weight", "weight_scale")
    assert tensors["_mxfp8_prefix_master"] is layer._mxfp8_prefix_master


def test_mxfp8_prefix_step_boundaries_and_missing_context(monkeypatch, mocker: MockerFixture):
    _install_fake_mindiesd(monkeypatch)
    method = _online_method(steps=1)
    layer = Module()
    layer._mxfp8_prefix_master = torch.eye(2, dtype=torch.bfloat16)
    layer._mxfp8_prefix_master_mode = "cpu"
    x = torch.eye(2, dtype=torch.bfloat16)
    mocker.patch.object(method, "_quantize_activation", return_value=(x, None))
    matmul = mocker.patch.object(method, "_quant_matmul", return_value=torch.zeros_like(x))

    for step, fallback in [(0, True), (0, True), (1, False), (-1, False), (None, False)]:
        _set_step_context(monkeypatch, step)
        output = method.apply(layer, x)
        if fallback:
            torch.testing.assert_close(output, x, rtol=0, atol=0)
        else:
            torch.testing.assert_close(output, torch.zeros_like(x), rtol=0, atol=0)
    assert matmul.call_count == 3


def test_mxfp8_prefix_dlo_requires_prefetched_master(monkeypatch):
    _install_fake_mindiesd(monkeypatch)
    method = _online_method(steps=1, mode="dlo")
    layer = Module()
    layer._mxfp8_prefix_master = torch.empty(0, dtype=torch.bfloat16)
    layer._mxfp8_prefix_master_mode = "dlo"
    layer._layerwise_prefix_managed = True
    _set_step_context(monkeypatch, 0)
    with pytest.raises(RuntimeError, match="was not prefetched"):
        method.apply(layer, torch.ones(1, 2, dtype=torch.bfloat16))


@pytest.mark.parametrize("mode,k", [("cpu", 5), ("resident", 5), ("dlo", 0)])
def test_provider_disabled_preserves_existing_paths(monkeypatch, mode, k):
    _install_fake_mindiesd(monkeypatch)
    assert _online_method(steps=k, mode=mode).get_layerwise_prefix_weights(Module()) is None

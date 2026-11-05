# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the vLLM-Omni to MindIE-SD MXFP8 fallback adapter."""

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class _FakeMaster:
    weight: torch.Tensor
    mode: str


class _FakeFallback:
    def __init__(self, policy, master_mode="cpu"):
        self.policy = policy
        self.master_mode = master_mode

    def prepare_master(self, weight):
        source = weight.detach()
        master = source.to(device="cpu", dtype=torch.bfloat16).transpose(0, 1).contiguous()
        if self.master_mode == "resident":
            if source.device.type != "npu":
                raise RuntimeError("Resident BF16 master must be prepared on the loading NPU")
            master = master.to(source.device)
        return _FakeMaster(master, self.master_mode)

    def use_bf16(self, step):
        return step is not None and step in self.policy.steps

    def apply(self, master, x, bias, ori_dtype, step):
        if not self.use_bf16(step):
            return None
        if master is None:
            raise RuntimeError("BF16 prefix requested without the original master weight")
        if master.weight.numel() == 0:
            raise RuntimeError("BF16 prefix master is not materialized")
        if master.mode == "resident" and master.weight.device != x.device:
            raise RuntimeError("Resident master and activation device mismatch")
        output = torch.matmul(x, master.weight.to(device=x.device, dtype=x.dtype))
        if bias is not None:
            output = output + bias.to(device=output.device, dtype=output.dtype)
        return output.to(ori_dtype)


def _install_fake_mindiesd(monkeypatch):
    quantization = ModuleType("mindiesd.quantization")
    setattr(quantization, "MXFP8BF16Fallback", _FakeFallback)
    setattr(quantization, "TimestepPolicyConfig", _FakePolicy)
    mindiesd = ModuleType("mindiesd")
    setattr(mindiesd, "quantization", quantization)
    monkeypatch.setitem(sys.modules, "mindiesd", mindiesd)
    monkeypatch.setitem(sys.modules, "mindiesd.quantization", quantization)


def _install_real_mindiesd(monkeypatch):
    """Load the real fallback and policy without importing MindIE-SD's NPU package init."""
    source = os.environ.get("MINDIE_SD_ROOT", "")
    if not source:
        pytest.skip("Set MINDIE_SD_ROOT to run the real MindIE-SD adapter contract")
    root = Path(source) / "mindiesd"
    mindiesd = ModuleType("mindiesd")
    mindiesd.__path__ = [str(root)]
    quantization = ModuleType("mindiesd.quantization")
    quantization.__path__ = [str(root / "quantization")]
    utils = ModuleType("mindiesd.utils")

    class ContractError(Exception):
        pass

    setattr(utils, "ModelInitError", ContractError)
    setattr(utils, "ParametersInvalid", ContractError)
    monkeypatch.setitem(sys.modules, "mindiesd", mindiesd)
    monkeypatch.setitem(sys.modules, "mindiesd.quantization", quantization)
    monkeypatch.setitem(sys.modules, "mindiesd.utils", utils)
    setattr(mindiesd, "quantization", quantization)
    for name in ("mode", "config", "mxfp8"):
        module_name = f"mindiesd.quantization.{name}"
        spec = importlib.util.spec_from_file_location(module_name, root / "quantization" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        setattr(quantization, name, module)
    config_module = sys.modules["mindiesd.quantization.config"]
    fallback_module = sys.modules["mindiesd.quantization.mxfp8"]
    setattr(quantization, "TimestepPolicyConfig", getattr(config_module, "TimestepPolicyConfig"))
    setattr(quantization, "MXFP8BF16Fallback", getattr(fallback_module, "MXFP8BF16Fallback"))
    setattr(quantization, "MXFP8BF16Master", getattr(fallback_module, "MXFP8BF16Master"))
    return quantization


def _set_step_context(monkeypatch, step, available=True):
    context = SimpleNamespace(denoise_step_idx=step)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.forward_context",
        SimpleNamespace(get_forward_context=lambda: context, is_forward_context_available=lambda: available),
    )


def _online_method(steps=1, mode="cpu"):
    from vllm_omni.quantization.mxfp8_config import NPUMxfp8OnlineLinearMethod

    return NPUMxfp8OnlineLinearMethod(DiffusionMXFP8Config(prefix_bf16_steps=steps, prefix_bf16_master_mode=mode))


@pytest.mark.parametrize("value,expected", [(None, 0), ("ocp", 0), ("amax_pow2_ceil", 1)])
def test_mxfp8_scale_method(monkeypatch, value, expected):
    from vllm_omni.quantization.mxfp8_config import _mx_scale_method_to_alg

    monkeypatch.delenv("VLLM_OMNI_MXFP8_SCALE_METHOD", raising=False)
    if value is not None:
        monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_METHOD", value)
    assert _mx_scale_method_to_alg() == expected


@pytest.mark.parametrize("scale_method,expected", [("ocp", 0), ("amax_pow2_ceil", 1)])
def test_mxfp8_activation_passes_scale_method(monkeypatch, mocker: MockerFixture, scale_method, expected):
    from vllm_omni.quantization.mxfp8_config import NPUMxfp8LinearMethod

    monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_METHOD", scale_method)
    quantize = mocker.Mock(return_value=("quantized", "scale"))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_dynamic_mx_quant=quantize, float8_e4m3fn="fp8"))
    method = NPUMxfp8LinearMethod(DiffusionMXFP8Config())
    x = torch.ones(2, 64, dtype=torch.bfloat16)
    assert method._quantize_activation(x) == ("quantized", "scale")
    quantize.assert_called_once_with(x, dst_type="fp8", scale_alg=expected)


@pytest.mark.parametrize("value", ["", "0", "1", "OCP", "unknown"])
def test_mxfp8_scale_method_rejects_invalid_values(monkeypatch, value):
    from vllm_omni.quantization.mxfp8_config import _mx_scale_method_to_alg

    monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_METHOD", value)
    with pytest.raises(ValueError, match="VLLM_OMNI_MXFP8_SCALE_METHOD"):
        _mx_scale_method_to_alg()


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


@pytest.mark.parametrize("scale_method,expected_alg", [(None, 0), ("ocp", 0), ("amax_pow2_ceil", 1)])
def test_mxfp8_prefix_saves_explicit_master_before_online_quantization(
    monkeypatch, mocker: MockerFixture, scale_method, expected_alg
):
    _install_fake_mindiesd(monkeypatch)
    method = _online_method(steps=1)
    layer = Module()
    original = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    expected = original.transpose(0, 1).contiguous().clone()
    layer.register_parameter("weight", torch.nn.Parameter(original.clone(), requires_grad=False))

    def quantize(weight, **kwargs):
        assert kwargs["scale_alg"] == expected_alg
        weight.zero_()
        return weight, torch.ones(4, 2, dtype=torch.uint8)

    quantize_mock = mocker.Mock(side_effect=quantize)
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_dynamic_mx_quant=quantize_mock, float8_e4m3fn="fp8")
    )
    monkeypatch.delenv("VLLM_OMNI_MXFP8_SCALE_METHOD", raising=False)
    if scale_method is not None:
        monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_METHOD", scale_method)
    method.process_weights_after_loading(layer)
    master = layer._mxfp8_prefix_master
    assert isinstance(master, _FakeMaster)
    assert master.mode == "cpu"
    torch.testing.assert_close(master.weight, expected, rtol=0, atol=0)
    assert master.weight.device.type == "cpu"
    assert "_mxfp8_prefix_master" not in layer.state_dict()
    quantize_mock.assert_called_once()


def test_mxfp8_prefix_dlo_hands_off_cpu_handle_tensor(monkeypatch):
    _install_fake_mindiesd(monkeypatch)
    method = _online_method(steps=5, mode="dlo")
    layer = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
    expected = layer.weight.detach().transpose(0, 1).contiguous().clone()
    layer._mxfp8_prefix_master = method._bf16_fallback.prepare_master(layer.weight)
    layer._mxfp8_prefix_master_mode = method.quant_config.prefix_bf16_master_mode

    master = layer._mxfp8_prefix_master
    assert master.mode == "cpu"
    assert master.weight.device.type == "cpu"
    torch.testing.assert_close(master.weight, expected, rtol=0, atol=0)
    k, tensors, replaced = method.get_layerwise_prefix_weights(layer)
    assert k == 5
    assert replaced == ("weight", "weight_scale")
    assert tensors["_mxfp8_prefix_master"] is master.weight


def test_mxfp8_prefix_step_boundaries_and_missing_context(monkeypatch, mocker: MockerFixture):
    _install_fake_mindiesd(monkeypatch)
    method = _online_method(steps=1)
    layer = Module()
    layer._mxfp8_prefix_master = method._bf16_fallback.prepare_master(torch.eye(2, dtype=torch.bfloat16))
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


def test_mxfp8_prefix_active_step_requires_master(monkeypatch):
    _install_fake_mindiesd(monkeypatch)
    method = _online_method(steps=1)
    _set_step_context(monkeypatch, 0)
    with pytest.raises(RuntimeError, match="without the original master weight"):
        method.apply(Module(), torch.ones(1, 2, dtype=torch.bfloat16))


def test_mxfp8_prefix_dlo_requires_prefetched_master(monkeypatch):
    _install_fake_mindiesd(monkeypatch)
    method = _online_method(steps=1, mode="dlo")
    layer = Module()
    layer._mxfp8_prefix_master = _FakeMaster(torch.empty(0, dtype=torch.bfloat16), mode="cpu")
    layer._mxfp8_prefix_master_mode = "dlo"
    layer._layerwise_prefix_managed = True
    _set_step_context(monkeypatch, 0)
    with pytest.raises(RuntimeError, match="was not prefetched"):
        method.apply(layer, torch.ones(1, 2, dtype=torch.bfloat16))


def test_mxfp8_prefix_resident_mismatch_is_preserved(monkeypatch):
    _install_fake_mindiesd(monkeypatch)
    method = _online_method(steps=1, mode="resident")
    layer = Module()
    layer._mxfp8_prefix_master = _FakeMaster(torch.eye(2, dtype=torch.bfloat16), mode="resident")
    _set_step_context(monkeypatch, 0)
    x = torch.empty(1, 2, dtype=torch.bfloat16, device="meta")
    with pytest.raises(RuntimeError, match="device mismatch"):
        method.apply(layer, x)


def test_real_mindiesd_handle_adapter_and_dlo_contract(monkeypatch, mocker: MockerFixture):
    real = _install_real_mindiesd(monkeypatch)
    method = _online_method(steps=2, mode="dlo")
    layer = torch.nn.Linear(3, 2, bias=True, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]]))
        layer.bias.copy_(torch.tensor([0.5, -0.25]))
    expected_master = layer.weight.detach().to(torch.bfloat16).transpose(0, 1).contiguous()
    quantize = mocker.Mock(side_effect=lambda weight, **kwargs: (weight, torch.ones(2, 2, dtype=torch.uint8)))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_dynamic_mx_quant=quantize, float8_e4m3fn="fp8"))

    method.process_weights_after_loading(layer)
    master = layer._mxfp8_prefix_master
    assert isinstance(master, real.MXFP8BF16Master)
    assert master.mode == "cpu"
    torch.testing.assert_close(master.weight, expected_master, rtol=0, atol=0)
    k, tensors, replaced = method.get_layerwise_prefix_weights(layer)
    assert (k, replaced) == (2, ("weight", "weight_scale"))
    assert tensors["_mxfp8_prefix_master"] is master.weight

    _set_step_context(monkeypatch, 0)
    x = torch.tensor([[[1.0, 2.0, -1.0], [0.5, -1.0, 4.0]]], dtype=torch.bfloat16)
    output = method.apply(layer, x, layer.bias)
    expected = (x @ expected_master + layer.bias.to(torch.bfloat16)).to(x.dtype)
    assert output.shape == (1, 2, 2)
    assert output.dtype == x.dtype
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode,k", [("cpu", 5), ("resident", 5), ("dlo", 0)])
def test_provider_disabled_preserves_existing_paths(monkeypatch, mode, k):
    _install_fake_mindiesd(monkeypatch)
    assert _online_method(steps=k, mode=mode).get_layerwise_prefix_weights(Module()) is None

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for online MXFP4 BF16-prefix fallback."""

import pytest
import torch
from torch.nn import Module

from vllm_omni.quantization.mxfp4_config import (
    DiffusionMXFP4Config,
    MXFP4BF16Fallback,
    MXFP4BF16Master,
    NPUMxfp4OnlineLinearMethod,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize("value", [-1, 1.5, "1", True])
def test_prefix_rejects_invalid_steps(value):
    error = ValueError if isinstance(value, int) and not isinstance(value, bool) else TypeError
    with pytest.raises(error, match="prefix_bf16_steps"):
        DiffusionMXFP4Config(prefix_bf16_steps=value)


def test_prefix_rejects_unknown_mode_and_serialized_checkpoint():
    with pytest.raises(ValueError, match="Unknown MXFP4 prefix master mode"):
        DiffusionMXFP4Config(prefix_bf16_master_mode="invalid")
    with pytest.raises(ValueError, match="requires online MXFP4"):
        DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True, prefix_bf16_steps=1)


def test_prefix_config_roundtrip():
    from vllm_omni.quantization import build_quant_config

    assert DiffusionMXFP4Config.from_config({}).prefix_bf16_steps == 0
    assert DiffusionMXFP4Config.from_config({}).prefix_bf16_master_mode == "cpu"
    cfg = build_quant_config({"method": "mxfp4", "prefix_bf16_steps": 5, "prefix_bf16_master_mode": "dlo"})
    assert isinstance(cfg, DiffusionMXFP4Config)
    assert cfg.prefix_bf16_steps == 5
    assert cfg.prefix_bf16_master_mode == "dlo"


def test_master_layout_and_dense_apply():
    original = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) / 16
    fallback = MXFP4BF16Fallback(2, "cpu")
    master = fallback.prepare_master(original)
    assert isinstance(master, MXFP4BF16Master)
    assert master.mode == "cpu"
    assert master.weight.device.type == "cpu"
    torch.testing.assert_close(master.weight, original.transpose(0, 1).contiguous(), rtol=0, atol=0)
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    bias = torch.randn(4, dtype=torch.bfloat16)
    expected = torch.matmul(x, original.transpose(0, 1)) + bias
    for step in (0, 1):
        actual = fallback.apply(master, x, bias, torch.bfloat16, step)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert fallback.apply(master, x, bias, torch.bfloat16, 2) is None
    assert fallback.apply(master, x, bias, torch.bfloat16, None) is None


def test_online_loader_saves_master_before_quantization(monkeypatch):
    import vllm_omni.quantization.mxfp4_config as module

    method = NPUMxfp4OnlineLinearMethod(DiffusionMXFP4Config(prefix_bf16_steps=5))
    layer = Module()
    original = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    expected = original.transpose(0, 1).contiguous().clone()
    layer.register_parameter("weight", torch.nn.Parameter(original.clone(), requires_grad=False))

    def quantize(weight, scale_alg):
        assert scale_alg == 0
        weight.zero_()
        return weight, torch.ones(4, 2, dtype=torch.uint8)

    monkeypatch.setattr(module, "_npu_quantize_mxfp4", quantize)
    method.process_weights_after_loading(layer)
    master = layer._mxfp4_prefix_master
    torch.testing.assert_close(master.weight, expected, rtol=0, atol=0)
    assert master.weight.device.type == "cpu"
    assert "_mxfp4_prefix_master" not in layer.state_dict()


def test_online_step_boundary_and_dlo_provider(monkeypatch):
    import vllm_omni.quantization.mxfp4_config as module

    method = NPUMxfp4OnlineLinearMethod(
        DiffusionMXFP4Config(prefix_bf16_steps=2, prefix_bf16_master_mode="dlo"), prefix="blocks.0.mlp.fc1"
    )
    layer = Module()
    original = torch.randn(4, 4, dtype=torch.bfloat16)
    layer._mxfp4_prefix_master = MXFP4BF16Master(original.transpose(0, 1).contiguous(), "cpu")
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, original)
    monkeypatch.setattr(module, "_current_denoise_step", lambda: 1)
    actual = method._apply_inner(layer, x, None, torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sentinel = object()
    monkeypatch.setattr(module.NPUMxfp4LinearMethod, "_apply_inner", lambda *args: sentinel)
    monkeypatch.setattr(module, "_current_denoise_step", lambda: 2)
    assert method._apply_inner(layer, x, None, torch.bfloat16) is sentinel
    steps, tensors, replaced = method.get_layerwise_prefix_weights(layer)
    assert steps == 2
    assert tensors == {"_mxfp4_prefix_master": layer._mxfp4_prefix_master.weight}
    assert replaced == ("weight", "weight_scale")


def test_requires_prefix_bf16_dlo_and_config_validation():
    from vllm_omni.diffusion.data import OmniDiffusionConfig, requires_prefix_bf16_dlo
    from vllm_omni.quantization import ComponentQuantizationConfig

    dlo_cfg = DiffusionMXFP4Config(prefix_bf16_steps=4, prefix_bf16_master_mode="dlo")
    cpu_cfg = DiffusionMXFP4Config(prefix_bf16_steps=4, prefix_bf16_master_mode="cpu")
    inactive_cfg = DiffusionMXFP4Config(prefix_bf16_steps=0, prefix_bf16_master_mode="dlo")

    assert requires_prefix_bf16_dlo(dlo_cfg) is True
    assert requires_prefix_bf16_dlo(cpu_cfg) is False
    assert requires_prefix_bf16_dlo(inactive_cfg) is False
    assert requires_prefix_bf16_dlo(ComponentQuantizationConfig({"transformer": dlo_cfg})) is True
    assert requires_prefix_bf16_dlo(ComponentQuantizationConfig({"transformer": cpu_cfg})) is False

    with pytest.raises(ValueError, match="requires distributed layerwise offload"):
        OmniDiffusionConfig(
            model="dummy",
            quantization_config={"method": "mxfp4", "prefix_bf16_steps": 4, "prefix_bf16_master_mode": "dlo"},
        )

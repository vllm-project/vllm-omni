# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for offline Smooth loading and the W4A8 dtype contract."""

import sys
from types import ModuleType

import pytest
import torch
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    override_forward_context,
    set_forward_context_denoise_step_idx,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.quantization.mxfp4_config import (
    DiffusionMXFP4Config,
    DiffusionMXFP4DualScaleMixedConfig,
    NPUMxfp4DualScaleLinearMethod,
    NPUMxfp4LinearMethod,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _patch_tp_state(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)


class _CheckpointLayer(torch.nn.Module):
    def __init__(self, method, dtype=torch.bfloat16):
        super().__init__()
        method.create_weights(
            self,
            input_size_per_partition=512,
            output_partition_sizes=[2],
            input_size=512,
            output_size=2,
            params_dtype=dtype,
            weight_loader=default_weight_loader,
        )

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            if name in params:
                params[name].weight_loader(params[name], value)
                loaded.add(name)
        return loaded


def _load_checkpoint(layer, config, weights, monkeypatch):
    od_config = OmniDiffusionConfig(model="", dtype=torch.bfloat16, quantization_config=config)
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    loader.counter_before_loading_weights = 0.0
    monkeypatch.setattr(loader, "get_all_weights", lambda model: iter(weights.items()))
    loader.load_weights(layer)


@pytest.mark.parametrize("require_smooth", [False, True])
@pytest.mark.parametrize("has_smooth", [False, True])
def test_single_scale_checkpoint_smooth_default_and_required(require_smooth, has_smooth, monkeypatch):
    config = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True, require_smooth_scale=require_smooth)
    layer = _CheckpointLayer(NPUMxfp4LinearMethod(config))
    weights = {name: torch.ones_like(param) for name, param in layer.named_parameters()}
    scale = torch.linspace(0.5, 3.0, 512)
    if has_smooth:
        weights["mul_scale"] = scale
    else:
        weights.pop("mul_scale")

    if require_smooth and not has_smooth:
        with pytest.raises(ValueError, match="mul_scale"):
            _load_checkpoint(layer, config, weights, monkeypatch)
    else:
        _load_checkpoint(layer, config, weights, monkeypatch)
        torch.testing.assert_close(layer.mul_scale, scale if has_smooth else torch.ones(512), rtol=0, atol=0)


@pytest.mark.parametrize("bad_scale", ["shape", "name"])
def test_declared_single_scale_smooth_rejects_malformed_checkpoint(bad_scale, monkeypatch):
    config = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True, require_smooth_scale=True)
    layer = _CheckpointLayer(NPUMxfp4LinearMethod(config))
    weights = {name: torch.ones_like(param) for name, param in layer.named_parameters()}
    if bad_scale == "shape":
        weights["mul_scale"] = torch.ones(511)
        with pytest.raises(AssertionError):
            _load_checkpoint(layer, config, weights, monkeypatch)
    else:
        weights["multiply_scale"] = weights.pop("mul_scale")
        with pytest.raises(ValueError, match="mul_scale"):
            _load_checkpoint(layer, config, weights, monkeypatch)


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 0.0, -1.0])
def test_single_scale_rejects_invalid_smooth_values(required, bad_value, monkeypatch):
    config = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True, require_smooth_scale=required)
    layer = _CheckpointLayer(NPUMxfp4LinearMethod(config))
    weights = {name: torch.ones_like(param) for name, param in layer.named_parameters()}
    weights["mul_scale"][7] = bad_value
    with pytest.raises(ValueError, match="finite, strictly positive"):
        _load_checkpoint(layer, config, weights, monkeypatch)


@pytest.mark.parametrize("bad_scale", [torch.ones(512, dtype=torch.int32), torch.ones(1, 512)])
def test_single_scale_rejects_invalid_smooth_type_or_rank(bad_scale, monkeypatch):
    config = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True)
    layer = _CheckpointLayer(NPUMxfp4LinearMethod(config))
    weights = {name: torch.ones_like(param) for name, param in layer.named_parameters()}
    weights["mul_scale"] = bad_scale
    with pytest.raises(ValueError, match="one-dimensional floating-point"):
        _load_checkpoint(layer, config, weights, monkeypatch)


@pytest.mark.parametrize("bad_value", [1e10, 1e-10])
def test_single_scale_rejects_smooth_overflow_or_underflow_after_cast(bad_value, monkeypatch):
    config = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True)
    method = NPUMxfp4LinearMethod(config)
    layer = _CheckpointLayer(method, torch.float16)
    layer.mul_scale.data.fill_(bad_value)
    # Validation must fail before any device packing or GEMM.
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    with pytest.raises(ValueError, match="finite, strictly positive"):
        method.process_weights_after_loading(layer)


@pytest.mark.parametrize(
    ("mode", "missing"),
    [
        ("single", "weight"),
        ("single", "weight_scale"),
        ("dual", "weight"),
        ("dual", "weight_scale"),
        ("dual", "weight_dual_scale"),
        ("dual", "mul_scale"),
    ],
)
def test_offline_mxfp4_required_tensors_do_not_use_legacy_scale_tolerance(mode, missing, monkeypatch):
    if mode == "single":
        config = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True)
        method = NPUMxfp4LinearMethod(config)
    else:
        config = DiffusionMXFP4DualScaleMixedConfig(is_checkpoint_serialized=True)
        method = NPUMxfp4DualScaleLinearMethod(config)
    layer = _CheckpointLayer(method)
    weights = {name: torch.ones_like(param) for name, param in layer.named_parameters() if name != missing}
    with pytest.raises(ValueError, match=missing):
        _load_checkpoint(layer, config, weights, monkeypatch)


@pytest.mark.parametrize("rank", [0, 1])
def test_single_scale_smooth_uses_real_row_parallel_loader(rank):
    from vllm.model_executor.layers.linear import RowParallelLinear

    layer = _CheckpointLayer(NPUMxfp4LinearMethod(DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True)))
    layer.tp_rank = rank
    checkpoint_scale = torch.arange(1024, dtype=torch.float32)
    RowParallelLinear.weight_loader(layer, layer.mul_scale, checkpoint_scale)
    torch.testing.assert_close(layer.mul_scale, checkpoint_scale[rank * 512 : (rank + 1) * 512], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("scale_alg", [0, 2])
def test_single_scale_smooth_precedes_quantization_and_preserves_output(
    dtype, fallback, with_bias, scale_alg, monkeypatch
):
    config = DiffusionMXFP4Config(
        is_checkpoint_mxfp4_serialized=True, w4a8_fallback_steps=[0], mxfp4_scale_alg=scale_alg
    )
    method = NPUMxfp4LinearMethod(config)
    layer = _CheckpointLayer(method, dtype)
    weights = {
        "weight": torch.linspace(-0.5, 1.5, 1024, dtype=dtype).reshape(2, 512),
        "weight_scale": torch.full((2, 16), 127, dtype=torch.uint8),
        "mul_scale": torch.linspace(0.5, 3.0, 512),
    }
    _load_checkpoint(layer, config, weights, monkeypatch)
    x = torch.linspace(-1.0, 2.0, 3072, dtype=dtype).reshape(2, 3, 512)
    bias = torch.tensor([0.5, -1.0], dtype=dtype) if with_bias else None
    quantized_inputs = []

    # Only the two device operators are substituted. Exercise real apply(),
    # step selection, Smooth multiplication, bias handling, and reshape logic.
    npu = ModuleType("torch_npu")

    def quantize(value, **kwargs):
        quantized_inputs.append(value.clone())
        expected = dict(
            dst_type=torch.float8_e4m3fn if fallback else "fp4",
            axis=-1,
            block_size=32,
            round_mode="rint",
            scale_alg=0 if fallback else scale_alg,
        )
        if not fallback and scale_alg == 2:
            expected["dst_type_max"] = 7.25
        assert kwargs == expected
        return value, torch.ones((6, 16), dtype=torch.uint8)

    def matmul(x_q, weight, weight_scale, *, bias, output_dtype, **kwargs):
        if fallback:
            assert output_dtype == torch.bfloat16
            if bias is not None:
                assert bias.dtype == output_dtype
                assert bias.shape == (1, 2)
        result = x_q.float() @ weight.float()
        if bias is not None:
            result += bias.float()
        return result.to(output_dtype)

    npu.__dict__.update(
        float8_e4m3fn=torch.float8_e4m3fn,
        float8_e8m0fnu=torch.uint8,
        float4_e2m1fn_x2="fp4",
        npu_dynamic_mx_quant=quantize,
        npu_quant_matmul=matmul,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    with override_forward_context(ForwardContext()):
        set_forward_context_denoise_step_idx(0 if fallback else 1)
        result = method.apply(layer, x, bias)

    expected_input = x.reshape(-1, 512) * weights["mul_scale"].to(dtype)
    assert len(quantized_inputs) == 1
    torch.testing.assert_close(quantized_inputs[0], expected_input, rtol=0, atol=0)
    expected = expected_input.float() @ weights["weight"].float().T
    if bias is not None:
        expected += bias.float()
    if fallback:
        expected = expected.bfloat16()
    torch.testing.assert_close(result, expected.to(dtype).reshape(2, 3, 2), rtol=0, atol=0)
    assert result.dtype == dtype

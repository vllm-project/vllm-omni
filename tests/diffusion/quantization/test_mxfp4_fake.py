# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.mxfp4_fake_config import (
    DiffusionMXFP4FakeConfig,
    MXFP4FakeLinearMethod,
    fake_quantize_mxfp4,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _scalar_reference(row):
    # Independent scalar oracle: frexp for exponent; exhaustive codebook search.
    maximum = max(abs(value) for value in row)
    scale = 2.0 ** max(-127, min(127, math.frexp(maximum)[1] - 3)) if maximum else 2.0**-127
    levels = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    return [
        math.copysign(levels[min(range(8), key=lambda i: (abs(abs(value) / scale - levels[i]), i % 2))] * scale, value)
        for value in row
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reference_matches_scalar_oracle_and_preserves_noncontiguous_input(dtype):
    generator = torch.Generator().manual_seed(123)
    x = (torch.randn(64, 5, generator=generator) * 3.1).to(dtype).T
    x[:, 32:] *= 16
    original = x.clone()
    expected = [_scalar_reference(row[:32]) + _scalar_reference(row[32:]) for row in x.float().tolist()]
    output = fake_quantize_mxfp4(x)
    torch.testing.assert_close(output, torch.tensor(expected, dtype=dtype), atol=0, rtol=0)
    torch.testing.assert_close(x, original)
    assert output.dtype == dtype
    assert torch.count_nonzero(output != original) > 0


def test_midpoints_use_even_codes_and_groups_have_independent_scales():
    x = torch.zeros(2, 32)
    x[:, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6])
    x[1] *= -8
    expected = torch.zeros_like(x)
    expected[:, :8] = torch.tensor([0, 1, 1, 2, 2, 4, 4, 6])
    expected[1] *= -8
    torch.testing.assert_close(fake_quantize_mxfp4(x), expected, atol=0, rtol=0)


def test_zero_empty_batch_and_nonfinite_values():
    assert torch.count_nonzero(fake_quantize_mxfp4(torch.zeros(2, 32))) == 0
    assert fake_quantize_mxfp4(torch.empty(0, 32)).shape == (0, 32)
    x = torch.ones(3, 32)
    x[0, 0], x[1, 0], x[2, 0] = float("nan"), float("inf"), -float("inf")
    y = fake_quantize_mxfp4(x)
    assert torch.isnan(y[0, 0]) and torch.isposinf(y[1, 0]) and torch.isneginf(y[2, 0])


@pytest.mark.parametrize("x", [torch.ones(31), torch.ones(2, 0), torch.tensor(1.0), torch.ones(32, dtype=torch.int32)])
def test_invalid_shape_and_dtype_fail(x):
    with pytest.raises(ValueError, match="mxfp4_fake requires"):
        fake_quantize_mxfp4(x)


def test_reference_chunk_boundary_and_idempotence():
    x = torch.full((65537, 32), 0.7)
    x[:, 0] = 6
    y = fake_quantize_mxfp4(x)
    assert torch.all(y[:, 1:] == 0.5)
    torch.testing.assert_close(fake_quantize_mxfp4(y), y, atol=0, rtol=0)


def test_factory_routes_only_quantizable_linears_and_rejects_unknown_recipe():
    config = build_quant_config({"method": "mxfp4_fake", "ignored_layers": ["token_refiner.blocks.0.attn.qkv_proj"]})
    assert isinstance(config, DiffusionMXFP4FakeConfig)
    layer = LinearBase.__new__(LinearBase)
    nn.Module.__init__(layer)
    assert isinstance(config.get_quant_method(layer, "blocks.0.attn.qkv_proj"), MXFP4FakeLinearMethod)
    assert type(config.get_quant_method(layer, "token_refiner.blocks.0.attn.qkv_proj")) is UnquantizedLinearMethod
    assert isinstance(config.get_quant_method(layer, "token_refiner.blocks.1.attn.qkv_proj"), MXFP4FakeLinearMethod)
    assert config.get_quant_method(nn.Linear(32, 32), "blocks.0.attn.sla_module.proj_l") is None
    with pytest.raises(ValueError, match="recipe must be"):
        build_quant_config({"method": "mxfp4_fake", "recipe": "unknown_c7_variant"})


def test_weight_and_activation_both_participate_in_floating_gemm():
    method = MXFP4FakeLinearMethod()
    layer = nn.Linear(32, 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.7)
        layer.weight[:, 0] = 6
    x = torch.full((2, 3, 32), 0.7)
    x[..., 0] = 6
    parameter = layer.weight
    method.process_weights_after_loading(layer)
    assert layer.weight is parameter
    # Both 0.7 values round to 0.5; 6*6 + 31*.5*.5 + .25 = 44.
    output = method.apply(layer, x, torch.tensor([0.25, 0.25]))
    torch.testing.assert_close(output, torch.full((2, 3, 2), 44.0), atol=0, rtol=0)
    assert layer.weight.is_floating_point()
    assert set(layer.state_dict()) == {"weight"}


def _c7_scalar_reference(row):
    # Independent finite FP32 oracle: choose the nearest code with ties away
    # from zero, rather than repeating the delivered private-exponent formula.
    maximum = max(abs(value) for value in row)
    exponent = max(-127, min(127, math.ceil(math.log2(max(maximum, 1.17e-38) / 7))))
    scale = 2.0**exponent
    levels = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    return [
        math.copysign(min(levels, key=lambda level: (abs(abs(value) / scale - level), -level)) * scale, value)
        for value in row
    ]


def test_c7_matches_independent_codebook_with_away_from_zero_ties_and_scale_boundary():
    x = torch.zeros(4, 64)
    x[:, :9] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7, 0.7])
    x[1] *= -8
    x[:, 32:] = x[:, :32] / 16
    x[2, 7] = 7.5  # C7 chooses scale 2 here; the RNE reference chooses 1.
    expected = [_c7_scalar_reference(row[:32]) + _c7_scalar_reference(row[32:]) for row in x.tolist()]
    output = fake_quantize_mxfp4(x, "mxfp4_c7")
    torch.testing.assert_close(output, torch.tensor(expected), atol=0, rtol=0)
    assert output[0, 0] == 0.5 and output[1, 0] == -4
    assert output[2, 7] == 8
    assert torch.count_nonzero(output != fake_quantize_mxfp4(x)) > 0


def test_c7_preserves_delivered_bf16_intermediate_rounding():
    x = torch.zeros(1, 32, dtype=torch.bfloat16)
    x[0, :2] = torch.tensor([6, 0.2490234375])
    actual = fake_quantize_mxfp4(x, "mxfp4_c7")
    promoted = fake_quantize_mxfp4(x.float(), "mxfp4_c7").bfloat16()
    # BF16 rounds 0.498046875 + 0.5 to 1 before floor. Promoting to FP32
    # silently changes the supplied recipe and rounds this element to zero.
    assert actual.dtype == x.dtype
    assert actual[0, 1] == 0.5
    assert promoted[0, 1] == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_c7_chunking_preserves_groups_noncontiguous_inputs_and_zero_batches(monkeypatch, dtype):
    from vllm_omni.quantization import mxfp4_fake_config as module

    x = torch.zeros(64, 5, dtype=dtype).T
    x[:, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7])
    x[:, 32:] = x[:, :32] * -8
    original = x.clone()
    monkeypatch.setattr(module, "_MXFP4_TORCH_CHUNK_ROWS", 2)
    chunked = fake_quantize_mxfp4(x, "mxfp4_c7")
    monkeypatch.setattr(module, "_MXFP4_TORCH_CHUNK_ROWS", 64)
    torch.testing.assert_close(chunked, fake_quantize_mxfp4(x, "mxfp4_c7"), atol=0, rtol=0)
    torch.testing.assert_close(x, original, atol=0, rtol=0)
    assert chunked.shape == x.shape and chunked.dtype == x.dtype
    assert fake_quantize_mxfp4(torch.empty(0, 32, dtype=dtype), "mxfp4_c7").shape == (0, 32)
    assert torch.count_nonzero(fake_quantize_mxfp4(torch.zeros(2, 32, dtype=dtype), "mxfp4_c7")) == 0


@pytest.mark.parametrize("x", [torch.ones(31), torch.ones(2, 0), torch.tensor(1.0), torch.ones(32).half()])
def test_c7_rejects_partial_groups_and_fp16_epsilon_underflow(x):
    with pytest.raises(ValueError, match="mxfp4_fake requires|mxfp4_c7 requires"):
        fake_quantize_mxfp4(x, "mxfp4_c7")


def test_c7_config_routes_weight_and_activation_to_delivered_recipe():
    config = build_quant_config({"method": "mxfp4_fake", "recipe": "mxfp4_c7"})
    parallel = LinearBase.__new__(LinearBase)
    nn.Module.__init__(parallel)
    method = config.get_quant_method(parallel, "blocks.0.attn.qkv_proj")
    assert isinstance(method, MXFP4FakeLinearMethod) and method.recipe == "mxfp4_c7"
    layer = nn.Linear(32, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[0, :2] = torch.tensor([6, 0.25])
    x = layer.weight.detach().clone()
    parameter = layer.weight
    method.process_weights_after_loading(layer)
    output = method.apply(layer, x)
    # Both weight and activation .25 become .5; reference RNE gives 36.
    torch.testing.assert_close(output, torch.tensor([[36.25]], dtype=torch.bfloat16), atol=0, rtol=0)
    assert layer.weight is parameter and set(layer.state_dict()) == {"weight"}
    assert config.get_quant_method(nn.Linear(32, 1), "blocks.0.attn.sla_module.proj_l") is None

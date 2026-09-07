# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    Magi2MHCConfig,
    Magi2MoEConfig,
    Magi2PreviewConfig,
)
from vllm_omni.diffusion.models.magi2.layers import (
    Magi2GroupedLinear,
    QuantizedLinear,
    make_grouped_linear,
    quantize_linear_modules_,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@dataclass
class _FakeQuantConfig:
    """Duck-typed stand-in for a vllm QuantizationConfig (`get_name()` only)."""

    name: str

    def get_name(self) -> str:
        return self.name


def _int8() -> _FakeQuantConfig:
    return _FakeQuantConfig("int8")


def _fp8() -> _FakeQuantConfig:
    return _FakeQuantConfig("fp8")


def _unsupported() -> _FakeQuantConfig:
    return _FakeQuantConfig("mxfp4")


def _tiny_config(quant_config: object | None) -> Magi2PreviewConfig:
    return Magi2PreviewConfig(
        num_layers=1,
        hidden_size=16,
        head_dim=8,
        num_query_groups=2,
        video_in_channels=4,
        audio_in_channels=4,
        text_in_channels=4,
        intermediate_factor=2,
        multimodal_layers=(0,),
        params_dtype=torch.float32,
        mhc=Magi2MHCConfig(num_streams=2),
        moe=Magi2MoEConfig(
            num_heads=2,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=8,
            shared_expert_intermediate_size=8,
            modality_shared_expert_intermediate_size=8,
            layers=(),  # keep layer 0 on the dense Magi2MLP path
        ),
        quant_config=quant_config,
    )


def _make_linear(quant_config: object | None, *, num_experts: int = 3) -> Magi2GroupedLinear:
    torch.manual_seed(0)
    linear = make_grouped_linear(
        in_features=16,
        out_features=12,
        num_experts=num_experts,
        bias=False,
        dtype=torch.float32,
        quant_config=quant_config,
    )
    with torch.no_grad():
        linear.weight.copy_(torch.randn_like(linear.weight))
    return linear


def test_unquantized_linear_is_unaffected_by_quant_config_field() -> None:
    linear = _make_linear(None)
    assert linear._quantized_dtype is None
    assert linear.weight is not None


def test_maybe_quantize_is_noop_without_supported_quant_config() -> None:
    for quant_config in (None, _unsupported()):
        linear = _make_linear(quant_config)
        linear.maybe_quantize_()
        assert linear._quantized_dtype is None
        assert linear.weight is not None


@pytest.mark.parametrize("quant_config,expected_dtype", [(_int8(), torch.int8), (_fp8(), torch.float8_e4m3fn)])
def test_maybe_quantize_replaces_weight_with_quantized_buffers(quant_config, expected_dtype) -> None:
    linear = _make_linear(quant_config)
    linear.maybe_quantize_()
    assert linear._quantized_dtype == expected_dtype
    assert linear.weight is None
    assert linear.weight_quantized.dtype == expected_dtype
    assert linear.weight_quantized.shape == (3, 12, 16)
    assert linear.weight_scale.shape == (3, 12, 1)

    # Idempotent: calling again must not raise or requantize.
    linear.maybe_quantize_()


@pytest.mark.parametrize(
    "quant_config,atol,rtol",
    [
        (_int8(), 5e-2, 5e-2),
        # e4m3 has only 3 mantissa bits (~1/8 relative step) vs int8's 7-bit
        # magnitude (~1/127 relative step), so fp8 needs a looser tolerance
        # for the same per-channel symmetric scheme -- this is the expected
        # quantization error, not a correctness bug.
        (_fp8(), 2e-1, 2e-1),
    ],
)
def test_quantized_forward_matches_unquantized_within_tolerance(quant_config, atol, rtol) -> None:
    baseline = _make_linear(None, num_experts=1)
    quantized = _make_linear(quant_config, num_experts=1)
    with torch.no_grad():
        quantized.weight.copy_(baseline.weight)
    quantized.maybe_quantize_()

    x = torch.randn(5, 16)
    expected = baseline(x)
    actual = quantized(x)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_quant_config_field_threads_from_model_config_into_dense_layers() -> None:
    from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2MLP

    quant_config = _int8()
    config = _tiny_config(quant_config)
    mlp = Magi2MLP(config, num_modality=1)
    assert mlp.up_gate_proj.quant_config is quant_config
    assert mlp.down_proj.quant_config is quant_config


def test_quant_config_field_threads_into_shared_expert_linears() -> None:
    from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2MultiHeadMoELayer

    quant_config = _int8()
    config = _tiny_config(quant_config)
    moe_layer = Magi2MultiHeadMoELayer(config)
    assert moe_layer.shared_expert_fc1.quant_config is quant_config
    assert moe_layer.shared_expert_fc2.quant_config is quant_config
    assert moe_layer.modality_specific_shared_expert_fc1.quant_config is quant_config
    assert moe_layer.modality_specific_shared_expert_fc2.quant_config is quant_config
    # The routed-expert gather/scatter path stays unquantized: it overlaps
    # the fused expert kernel work tracked separately under M2 in #7085.
    assert moe_layer.split_linear.quant_config is None
    assert moe_layer.merge_linear.quant_config is None


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=True)
        self.fc2 = nn.Linear(16, 8, bias=False)
        self.nested = nn.Sequential(nn.Linear(8, 8))


def test_quantize_linear_modules_is_noop_without_supported_quant_config() -> None:
    model = _TinyMLP()
    count = quantize_linear_modules_(model, None)
    assert count == 0
    assert isinstance(model.fc1, nn.Linear)


@pytest.mark.parametrize("quant_config,expected_dtype", [(_int8(), torch.int8), (_fp8(), torch.float8_e4m3fn)])
def test_quantize_linear_modules_replaces_all_linears(quant_config, expected_dtype) -> None:
    torch.manual_seed(0)
    model = _TinyMLP()
    count = quantize_linear_modules_(model, quant_config)
    assert count == 3
    assert isinstance(model.fc1, QuantizedLinear)
    assert isinstance(model.fc2, QuantizedLinear)
    assert isinstance(model.nested[0], QuantizedLinear)
    assert model.fc1.weight_quantized.dtype == expected_dtype
    assert model.fc2.bias is None
    assert model.fc1.bias is not None


@pytest.mark.parametrize(
    "quant_config,atol,rtol",
    [(_int8(), 5e-2, 5e-2), (_fp8(), 2e-1, 2e-1)],
)
def test_quantized_linear_forward_matches_unquantized_within_tolerance(quant_config, atol, rtol) -> None:
    torch.manual_seed(0)
    reference = nn.Linear(8, 16, bias=True)
    quantized_linear = QuantizedLinear(reference, {"int8": torch.int8, "fp8": torch.float8_e4m3fn}[quant_config.name])

    x = torch.randn(4, 8)
    expected = reference(x)
    actual = quantized_linear(x)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

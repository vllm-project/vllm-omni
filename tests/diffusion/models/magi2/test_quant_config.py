# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    Magi2MHCConfig,
    Magi2MoEConfig,
    Magi2PreviewConfig,
)
from vllm_omni.diffusion.models.magi2.layers import Magi2GroupedLinear, make_grouped_linear

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@dataclass
class _FakeInt8Config:
    """Duck-typed stand-in for vllm's DiffusionInt8Config (`get_name()` only)."""

    def get_name(self) -> str:
        return "int8"


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
    assert not linear._int8_quantized
    assert linear.weight is not None


def test_maybe_quantize_int8_is_noop_without_supported_quant_config() -> None:
    linear = _make_linear(None)
    linear.maybe_quantize_int8_()
    assert not linear._int8_quantized
    assert linear.weight is not None


def test_maybe_quantize_int8_replaces_weight_with_int8_buffers() -> None:
    linear = _make_linear(_FakeInt8Config())
    linear.maybe_quantize_int8_()
    assert linear._int8_quantized
    assert linear.weight is None
    assert linear.weight_int8.dtype == torch.int8
    assert linear.weight_int8.shape == (3, 12, 16)
    assert linear.weight_scale.shape == (3, 12, 1)

    # Idempotent: calling again must not raise or requantize.
    linear.maybe_quantize_int8_()


def test_quantized_forward_matches_unquantized_within_tolerance() -> None:
    baseline = _make_linear(None, num_experts=1)
    quantized = _make_linear(_FakeInt8Config(), num_experts=1)
    with torch.no_grad():
        quantized.weight.copy_(baseline.weight)
    quantized.maybe_quantize_int8_()

    x = torch.randn(5, 16)
    expected = baseline(x)
    actual = quantized(x)
    assert actual.shape == expected.shape
    # Per-channel symmetric int8 (127 levels) on N(0, 1) weights: relative
    # error is dominated by quantization step size, not a hard numerical bug.
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


def test_quant_config_field_threads_from_model_config_into_dense_layers() -> None:
    from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2MLP

    quant_config = _FakeInt8Config()
    config = _tiny_config(quant_config)
    mlp = Magi2MLP(config, num_modality=1)
    assert mlp.up_gate_proj.quant_config is quant_config
    assert mlp.down_proj.quant_config is quant_config

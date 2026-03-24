# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for diffusion GGUF quantization config and linear method."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.gguf import UNQUANTIZED_TYPES

from vllm_omni.diffusion.quantization import (
    DiffusionGgufConfig,
    get_diffusion_quant_config,
)
from vllm_omni.diffusion.quantization.gguf import (
    DiffusionGGUFLinearMethod,
    UnquantizedLinearMethod,
    _GGUFConfig,
    dequant_gemm_gguf,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_gguf_config_creation_and_delegation():
    config = DiffusionGgufConfig(
        gguf_model="weights.gguf",
        unquantized_modules=["proj_out"],
    )

    assert config.gguf_model == "weights.gguf"
    assert config.unquantized_modules == ["proj_out"]
    assert config.get_name() == "gguf"
    assert isinstance(config.get_vllm_quant_config(), _GGUFConfig)


def test_get_diffusion_quant_config_builds_gguf_config():
    config = get_diffusion_quant_config(
        "gguf",
        gguf_model="repo/model-Q4_0.gguf",
        unquantized_modules=["norm_out"],
    )

    assert isinstance(config, DiffusionGgufConfig)
    assert config.gguf_model == "repo/model-Q4_0.gguf"
    assert config.unquantized_modules == ["norm_out"]


def test_gguf_vllm_config_returns_diffusion_linear_method_for_linear_layers():
    linear = object.__new__(LinearBase)
    method = _GGUFConfig(unquantized_modules=[]).get_quant_method(linear, "transformer.img_in")

    assert isinstance(method, DiffusionGGUFLinearMethod)


def test_gguf_vllm_config_respects_unquantized_modules():
    linear = object.__new__(LinearBase)
    method = _GGUFConfig(unquantized_modules=["proj_out"]).get_quant_method(linear, "transformer.proj_out")

    assert isinstance(method, UnquantizedLinearMethod)


def test_gguf_vllm_config_returns_none_for_non_linear_layers():
    method = _GGUFConfig(unquantized_modules=[]).get_quant_method(torch.nn.LayerNorm(4), "norm")

    assert method is None


def test_dequant_gemm_gguf_uses_plain_matmul_for_unquantized_types():
    qweight_type = next(iter(UNQUANTIZED_TYPES))
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    qweight = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)

    out = dequant_gemm_gguf(x, qweight, qweight_type)

    assert torch.allclose(out, x @ qweight.T)


def test_diffusion_gguf_linear_method_applies_bias_on_unquantized_weight():
    qweight_type = next(iter(UNQUANTIZED_TYPES))
    method = DiffusionGGUFLinearMethod(_GGUFConfig(unquantized_modules=[]))
    layer = SimpleNamespace(
        qweight=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        qweight_type=SimpleNamespace(weight_type=qweight_type),
    )
    x = torch.tensor([[[1.0, 0.5]]], dtype=torch.float32)
    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)

    out = method.apply(layer, x, bias)

    expected = x @ layer.qweight.T + bias
    assert torch.allclose(out, expected)


def test_diffusion_gguf_linear_method_concatenates_sharded_outputs():
    qweight_type = next(iter(UNQUANTIZED_TYPES))
    method = DiffusionGGUFLinearMethod(_GGUFConfig(unquantized_modules=[]))
    qweight = torch.nn.Parameter(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    qweight.shard_id = ["left", "right"]
    qweight.shard_offset_map = {
        "left": (0, 2, 2),
        "right": (2, 4, 2),
    }
    layer = SimpleNamespace(
        qweight=qweight,
        qweight_type=SimpleNamespace(
            shard_weight_type={
                "left": qweight_type,
                "right": qweight_type,
            }
        ),
    )
    x = torch.tensor([[1.5, 2.0]], dtype=torch.float32)

    out = method.apply(layer, x)

    expected_q = x @ qweight[:2].T
    expected_k = x @ qweight[2:4].T
    assert torch.allclose(out, torch.cat([expected_q, expected_k], dim=-1))

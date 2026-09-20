# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Unit tests for the plain-tensor FP8 weight-only representation
# (``plain_fp8.py``): Float8Tensor unpacking, idempotency, numerical
# equivalence with the torchao runtime path, and fail-closed skipping of
# unsupported layouts.

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.diffusion.models.boogu_image.plain_fp8 import (
    PlainFloat8LinearMethod,
    repack_torchao_float8_linears,
    unpack_float8_linear,
)

_KO = pytest.importorskip("torchao")  # FP8 checkpoint support requires torchao


class _FakeQuantMethod:
    """Stand-in for vLLM's TorchAOLinearMethod on a loaded quant linear."""


def _make_float8_weight(rows: int = 135, cols: int = 64, seed: int = 0):
    """Quantize a real bf16 matrix with torchao's Float8WeightOnlyConfig.

    Mirrors what the official Boogu-Image FP8 checkpoints store: per-row
    block layout ``[1, cols]`` with an fp32 row scale.
    """
    torch.manual_seed(seed)
    hp = torch.randn(rows, cols, dtype=torch.bfloat16)

    from torchao.quantization import Float8WeightOnlyConfig
    from vllm.model_executor.layers.quantization.torchao import (
        torchao_quantize_param_data,
    )

    return torchao_quantize_param_data(nn.Parameter(hp), Float8WeightOnlyConfig(set_inductor_config=False))


def _make_linear_like(weight, quant_method: object = _FakeQuantMethod()):
    layer = nn.Module()
    layer.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
    layer.quant_method = quant_method
    layer.prefix = "dummy.linear"
    return layer


# ---------------------------------------------------------------------------
# Unpacking
# ---------------------------------------------------------------------------


def test_unpack_float8_linear_replaces_subclass_with_plain_tensors():
    w8 = _make_float8_weight()
    assert type(w8).__name__ == "Float8Tensor", "fixture must produce a Float8Tensor"
    layer = _make_linear_like(w8)

    assert unpack_float8_linear(layer) is True

    # weight becomes an ordinary fp8 parameter (no subclass)
    assert isinstance(layer.weight, nn.Parameter)
    assert type(layer.weight).__name__ == "Parameter"
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.shape == w8.shape

    # row scale is an ordinary fp32 parameter ([out, 1])
    assert isinstance(layer.weight_scale, nn.Parameter)
    assert layer.weight_scale.dtype == torch.float32
    assert tuple(layer.weight_scale.shape) == (w8.shape[0], 1)

    # original high-precision dtype recorded for the dequantize step
    assert layer.weight_dtype == w8.dtype == torch.bfloat16

    # quant method swapped to the plain implementation
    assert isinstance(layer.quant_method, PlainFloat8LinearMethod)


def test_unpack_is_idempotent():
    layer = _make_linear_like(_make_float8_weight())
    assert unpack_float8_linear(layer) is True
    assert unpack_float8_linear(layer) is False
    assert isinstance(layer.quant_method, PlainFloat8LinearMethod)


def test_unpack_preserves_loader_attributes():
    w8 = _make_float8_weight()
    layer = _make_linear_like(w8)
    # vLLM set_weight_attrs stamps input_dim/output_dim onto the *parameter*
    # (as LinearBase.__init__ does after create_weights).
    layer.weight.input_dim = 1  # type: ignore[attr-defined]
    layer.weight.output_dim = 0  # type: ignore[attr-defined]

    unpack_float8_linear(layer)

    assert layer.weight.input_dim == 1
    assert layer.weight.output_dim == 0


def test_plain_forward_is_bit_identical_to_torchao_runtime_path():
    w8 = _make_float8_weight()
    layer = _make_linear_like(w8)
    unpack_float8_linear(layer)

    x = torch.randn(3, w8.shape[1], dtype=torch.bfloat16)
    y_torchao = F.linear(x, w8.dequantize())  # what TorchAOLinearMethod.apply does
    y_plain = PlainFloat8LinearMethod().apply(layer, x)

    torch.testing.assert_close(y_plain, y_torchao, rtol=0.0, atol=0.0)


def test_plain_forward_applies_bias():
    w8 = _make_float8_weight()
    layer = _make_linear_like(w8)
    unpack_float8_linear(layer)

    x = torch.randn(2, w8.shape[1], dtype=torch.bfloat16)
    bias = torch.randn(w8.shape[0], dtype=torch.bfloat16)
    y_torchao = F.linear(x, w8.dequantize(), bias)
    y_plain = PlainFloat8LinearMethod().apply(layer, x, bias)

    torch.testing.assert_close(y_plain, y_torchao, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Fail-closed skipping
# ---------------------------------------------------------------------------


def test_skips_plain_bf16_weight():
    hp = torch.randn(8, 16, dtype=torch.bfloat16)
    layer = _make_linear_like(hp)
    assert unpack_float8_linear(layer) is False
    assert isinstance(layer.weight, nn.Parameter)
    assert not hasattr(layer, "weight_scale")
    assert isinstance(layer.quant_method, _FakeQuantMethod)


def test_skips_layer_without_quant_method():
    layer = nn.Module()
    layer.register_parameter("weight", nn.Parameter(torch.randn(8, 16)))
    assert unpack_float8_linear(layer) is False


def test_skips_non_per_row_block_layout():
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
        Float8Tensor,
    )

    w8 = _make_float8_weight()
    # Re-wrap the real qdata/scale under a non per-row block layout
    # (e.g. a 2-row fine-grained block), which the unpacker must reject.
    w_blocked = Float8Tensor(w8.qdata, w8.scale, block_size=[2, 8], dtype=torch.bfloat16)
    layer = _make_linear_like(w_blocked)

    assert unpack_float8_linear(layer) is False
    # left untouched: still a subclass, original quant method intact
    assert type(layer.weight).__name__ == "Float8Tensor"
    assert isinstance(layer.quant_method, _FakeQuantMethod)


def test_skips_layer_without_weight():
    layer = nn.Module()
    layer.quant_method = _FakeQuantMethod()
    assert unpack_float8_linear(layer) is False


# ---------------------------------------------------------------------------
# Model-wide scan
# ---------------------------------------------------------------------------


def test_repack_model_scan_repacks_only_float8_linears():
    model = nn.Module()
    quant = nn.Module()
    quant.linear = _make_linear_like(_make_float8_weight(seed=1))
    plain = nn.Module()
    plain.linear = _make_linear_like(torch.randn(8, 16, dtype=torch.bfloat16))
    model.quant, model.plain = quant, plain

    assert repack_torchao_float8_linears(model) == 1

    assert isinstance(quant.linear.quant_method, PlainFloat8LinearMethod)
    assert type(quant.linear.weight).__name__ == "Parameter"
    assert not isinstance(plain.linear.quant_method, PlainFloat8LinearMethod)
    assert type(plain.linear.weight).__name__ == "Parameter"


def test_repack_model_scan_reports_zero_when_nothing_to_do():
    model = nn.Module()
    model.linear = _make_linear_like(torch.randn(8, 16, dtype=torch.bfloat16))
    assert repack_torchao_float8_linears(model) == 0

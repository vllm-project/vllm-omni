# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the reusable PrimsTS fused QKV operation."""

import sys
from types import ModuleType

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.layers import prims_ts_fp8 as fp8_ops

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_fp8_weight_preparation_transposes_and_reorders_swiglu_channels():
    weight = torch.arange(8, dtype=torch.float32).reshape(2, 4).to(torch.float8_e4m3fn)
    weight_scale = torch.arange(1, 5, dtype=torch.float32)

    linear_weight, linear_scale = fp8_ops.prepare_prims_ts_fp8_weights(weight, weight_scale, layout="linear")
    swiglu_weight, swiglu_scale = fp8_ops.prepare_prims_ts_fp8_weights(weight, weight_scale, layout="swiglu")

    transposed = weight.t()
    expected_swiglu_weight = torch.stack((transposed[2:], transposed[:2]), dim=1).reshape(4, 2)
    torch.testing.assert_close(linear_weight, transposed.contiguous())
    torch.testing.assert_close(linear_scale, weight_scale)
    torch.testing.assert_close(swiglu_weight, expected_swiglu_weight.contiguous())
    torch.testing.assert_close(swiglu_scale, torch.tensor([3.0, 1.0, 4.0, 2.0]))


def _inputs():
    activation = torch.empty((4, 128), dtype=torch.float8_e4m3fn)
    weight = torch.empty((384, 128), dtype=torch.float8_e4m3fn)
    activation_scale = torch.ones((4,), dtype=torch.float32)
    weight_scale = torch.ones((384,), dtype=torch.float32)
    q_norm = torch.ones((128,), dtype=torch.bfloat16)
    k_norm = torch.ones((128,), dtype=torch.bfloat16)
    cos_sin = torch.empty((4, 128), dtype=torch.float32)
    positions = torch.arange(4, dtype=torch.int64)
    return activation, weight, activation_scale, weight_scale, q_norm, k_norm, cos_sin, positions


def _call_qkv(inputs, *, num_q_heads, num_kv_heads, head_dim):
    activation, weight, activation_scale, weight_scale, q_norm, k_norm, cos_sin, positions = inputs
    return fp8_ops.prims_ts_fp8_qkv_qknorm_rope(
        (activation, activation_scale),
        weight,
        weight_scale,
        q_norm,
        k_norm,
        cos_sin,
        positions,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def test_fp8_custom_op_has_model_independent_fake_shape() -> None:
    with FakeTensorMode() as mode:
        inputs = tuple(mode.from_tensor(tensor) for tensor in _inputs())
        output = _call_qkv(inputs, num_q_heads=1, num_kv_heads=1, head_dim=128)

    assert output.shape == (4, 384)
    assert output.dtype is torch.bfloat16


def test_fp8_linear_and_swiglu_fake_shapes() -> None:
    tensors = (
        torch.empty((4, 128), dtype=torch.float8_e4m3fn),
        torch.empty((256, 128), dtype=torch.float8_e4m3fn),
        torch.ones((4,), dtype=torch.float32),
        torch.ones((256,), dtype=torch.float32),
    )
    with FakeTensorMode() as mode:
        inputs = tuple(mode.from_tensor(tensor) for tensor in tensors)
        activation, weight, activation_scale, weight_scale = inputs
        activation_pair = (activation, activation_scale)
        linear_output = fp8_ops.prims_ts_fp8_linear(activation_pair, weight, weight_scale)
        swiglu_output = fp8_ops.prims_ts_fp8_swiglu(activation_pair, weight, weight_scale)

    assert linear_output.shape == (4, 256)
    assert swiglu_output.shape == (4, 128)
    assert linear_output.dtype is torch.bfloat16
    assert swiglu_output.dtype is torch.bfloat16


def test_fp8_entry_point_validates_qkv_geometry() -> None:
    with pytest.raises(ValueError, match="equal Q/K/V groups"):
        _call_qkv(_inputs(), num_q_heads=2, num_kv_heads=1, head_dim=128)
    with pytest.raises(ValueError, match="head_dim=128"):
        _call_qkv(_inputs(), num_q_heads=1, num_kv_heads=1, head_dim=64)


def test_fp8_linear_quantizes_floating_input_on_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm import _custom_ops as ops

    quantized = torch.empty((4, 128), dtype=torch.float8_e4m3fn)
    activation_scale = torch.ones((4,), dtype=torch.float32)
    quantize_calls = []
    op_calls = []
    monkeypatch.setattr(
        ops,
        "scaled_fp8_quant",
        lambda x, **kwargs: quantize_calls.append((x, kwargs)) or (quantized, activation_scale),
    )
    monkeypatch.setattr(
        fp8_ops,
        "_LINEAR_OP",
        lambda *args: op_calls.append(args) or torch.empty((4, 256)),
    )

    output = fp8_ops.prims_ts_fp8_linear(
        torch.empty((4, 128), dtype=torch.bfloat16),
        torch.empty((256, 128), dtype=torch.float8_e4m3fn),
        torch.ones((256,), dtype=torch.float32),
    )

    assert output.shape == (4, 256)
    assert len(quantize_calls) == len(op_calls) == 1
    assert quantize_calls[0][1] == {"use_per_token_if_dynamic": True}
    assert op_calls[0][0] is quantized
    assert torch.equal(op_calls[0][2], activation_scale)


@pytest.mark.parametrize(
    ("shape", "one_pass"),
    [((2, 8192, 3072), True), ((1, 8192, 3072), False), ((2, 8192, 9216), False)],
)
def test_fp8_quantize_dispatches_by_shape(monkeypatch: pytest.MonkeyPatch, shape, one_pass: bool) -> None:
    from vllm import _custom_ops as ops

    calls = []
    monkeypatch.setattr(fp8_ops, "per_token_fp8_quant", lambda x: calls.append(("one_pass", x.shape)) or (x, x))
    monkeypatch.setattr(ops, "scaled_fp8_quant", lambda x, **kwargs: calls.append(("vllm", x.shape)) or (x, x))

    fp8_ops.prims_ts_fp8_quantize(torch.empty(shape, dtype=torch.bfloat16))

    # Many rows of moderate width take the one-pass kernel, flattened to [M, K].
    assert calls == [("one_pass" if one_pass else "vllm", (shape[0] * shape[1], shape[2]))]


def test_fp8_prequantized_input_requires_a_scale_pair() -> None:
    with pytest.raises(TypeError, match="activation, scale"):
        fp8_ops.prims_ts_fp8_linear(
            torch.empty((4, 128), dtype=torch.float8_e4m3fn),
            torch.empty((256, 128), dtype=torch.float8_e4m3fn),
            torch.ones((256,), dtype=torch.float32),
        )


def test_fp8_qkv_impl_calls_shared_kernel_with_explicit_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    gemm = ModuleType("flashinfer.prims_ts.gemm")
    gemm.fp8_qkv_qknorm_rope = lambda *args, **kwargs: calls.append((args, kwargs)) or torch.empty((4, 384))
    prims_ts = ModuleType("flashinfer.prims_ts")
    prims_ts.gemm = gemm
    monkeypatch.setitem(sys.modules, "flashinfer.prims_ts", prims_ts)
    marker = object()
    monkeypatch.setitem(fp8_ops._CONFIG_PROVIDERS, "unit_test", lambda *_: marker)

    fp8_ops._fp8_qkv_qknorm_rope_impl(*_inputs(), 1, 1, 128, "unit_test")

    assert calls[0][1]["config"] is marker
    assert calls[0][1]["is_neox"] is False


def test_fp8_qkv_impl_builds_identity_positions_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    gemm = ModuleType("flashinfer.prims_ts.gemm")
    gemm.fp8_qkv_qknorm_rope = lambda *args, **kwargs: calls.append(args) or torch.empty((4, 384))
    prims_ts = ModuleType("flashinfer.prims_ts")
    prims_ts.gemm = gemm
    monkeypatch.setitem(sys.modules, "flashinfer.prims_ts", prims_ts)
    monkeypatch.setitem(fp8_ops._CONFIG_PROVIDERS, "unit_test", lambda *_: None)
    inputs = _inputs()[:-1]

    fp8_ops._fp8_qkv_qknorm_rope_impl(*inputs, None, 1, 1, 128, "unit_test")
    fp8_ops._fp8_qkv_qknorm_rope_impl(*inputs, None, 1, 1, 128, "unit_test")

    torch.testing.assert_close(calls[0][7], torch.arange(4, dtype=torch.int64))
    assert calls[1][7].data_ptr() == calls[0][7].data_ptr()


def test_fp8_qkv_fake_shape_without_positions() -> None:
    with FakeTensorMode() as mode:
        inputs = tuple(mode.from_tensor(tensor) for tensor in _inputs()[:-1])
        output = _call_qkv((*inputs, None), num_q_heads=1, num_kv_heads=1, head_dim=128)

    assert output.shape == (4, 384)


@pytest.mark.parametrize(
    ("impl", "kernel"),
    [(fp8_ops._fp8_linear_impl, "fp8_linear"), (fp8_ops._fp8_swiglu_impl, "fp8_linear_swiglu")],
)
def test_fp8_projection_impls_are_shared_custom_paths(monkeypatch: pytest.MonkeyPatch, impl, kernel: str) -> None:
    calls = []
    gemm = ModuleType("flashinfer.prims_ts.gemm")
    setattr(gemm, kernel, lambda *args, **kwargs: calls.append((args, kwargs)) or torch.empty((4, 256)))
    prims_ts = ModuleType("flashinfer.prims_ts")
    prims_ts.gemm = gemm
    monkeypatch.setitem(sys.modules, "flashinfer.prims_ts", prims_ts)
    marker = object()
    monkeypatch.setitem(fp8_ops._CONFIG_PROVIDERS, "unit_test", lambda *_: marker)

    impl(torch.empty((4, 128)), torch.empty((256, 128)), torch.ones(4), torch.ones(256), "unit_test")

    assert calls[0][1]["config"] is marker
    assert calls[0][0][2].ndim == 1

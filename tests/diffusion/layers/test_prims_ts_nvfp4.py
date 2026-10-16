# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.layers import prims_ts_nvfp4

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _test_config_provider(device, epilogue, k, n, m, head_dim, qkv_scale):
    return (device, epilogue, k, n, m, head_dim, qkv_scale)


def test_nvfp4_scale_layout_round_trips():
    logical = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8).to(torch.float8_e4m3fn)

    packed = prims_ts_nvfp4.nvfp4_scales_to_128x4(logical)
    restored = prims_ts_nvfp4.nvfp4_scales_to_linear(packed.view(torch.float8_e4m3fn).reshape_as(logical))

    torch.testing.assert_close(restored, logical)


def test_nvfp4_weight_preparation_reorders_swiglu_rows_and_scales():
    weight = torch.arange(128 * 16, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(128, 16)
    logical_scales = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4).to(torch.float8_e4m3fn)
    serialized_scales = prims_ts_nvfp4.nvfp4_scales_to_128x4(logical_scales)
    weight_scales = serialized_scales.view(torch.float8_e4m3fn).reshape_as(logical_scales)
    half = weight.shape[0] // 2
    expected_weight = torch.stack((weight[half:], weight[:half]), dim=1).reshape_as(weight)
    logical_rows = prims_ts_nvfp4.nvfp4_scales_to_linear(weight_scales)
    expected_logical_scales = torch.stack((logical_rows[half:], logical_rows[:half]), dim=1).reshape_as(logical_rows)

    prepared_weight, prepared_scales = prims_ts_nvfp4.prepare_prims_ts_nvfp4_weights(
        weight,
        weight_scales,
        layout="swiglu",
    )

    torch.testing.assert_close(prepared_weight, expected_weight)
    torch.testing.assert_close(prepared_scales, prims_ts_nvfp4.nvfp4_scales_to_128x4(expected_logical_scales))


def test_nvfp4_qkv_weight_preparation_preserves_complete_global_scales():
    weights = [torch.full((128, 8), value, dtype=torch.uint8) for value in (1, 2, 3)]
    logical_scales = [torch.full((128, 4), value, dtype=torch.float8_e4m3fn) for value in (1, 2, 3)]
    serialized_scales = [
        prims_ts_nvfp4.nvfp4_scales_to_128x4(scale).view(torch.float8_e4m3fn).reshape_as(scale)
        for scale in logical_scales
    ]

    weight, scale, qkv_scale, common_scale = prims_ts_nvfp4.prepare_prims_ts_nvfp4_qkv_weights(
        weights,
        serialized_scales,
        (2.0, 4.0, 8.0),
    )

    torch.testing.assert_close(weight, torch.cat(weights))
    logical_scale = torch.cat([prims_ts_nvfp4.nvfp4_scales_to_linear(item) for item in serialized_scales])
    torch.testing.assert_close(scale, prims_ts_nvfp4.nvfp4_scales_to_128x4(logical_scale))
    torch.testing.assert_close(qkv_scale, torch.tensor([2.0, 4.0, 8.0]))
    assert common_scale == 8.0


def test_registered_layer_uses_shared_nvfp4_custom_op_and_config_provider(monkeypatch):
    prims_ts_nvfp4.register_prims_ts_nvfp4_config_provider("test-consumer", _test_config_provider)
    owner = nn.Module()
    cache_id = prims_ts_nvfp4.register_prims_ts_nvfp4_layer(owner, config_provider="test-consumer")
    prepare_calls = []

    class _FakeGemm:
        def prepare_fp4_linear(self, weight, weight_scale, input_scale, global_scale, *, config):
            prepare_calls.append((weight, weight_scale, input_scale, global_scale, config))
            return lambda activation, activation_scale: torch.empty(
                (activation.shape[0], weight.shape[0]), dtype=torch.bfloat16
            )

    monkeypatch.setattr(prims_ts_nvfp4, "_gemm", lambda: _FakeGemm())
    activation = torch.empty((128, 8), dtype=torch.uint8)
    activation_scale = torch.empty((128,), dtype=torch.uint8)
    weight = torch.empty((16, 8), dtype=torch.uint8)
    weight_scale = torch.empty((16,), dtype=torch.uint8)

    # Call the custom-op implementation directly: the registered dispatcher
    # implementation is accelerator-only, while this test exercises CPU cache
    # and config-provider behavior with a fake GEMM.
    first = prims_ts_nvfp4._nvfp4_linear_impl(activation, activation_scale, weight, weight_scale, 0.25, 0.5, cache_id)
    second = prims_ts_nvfp4._nvfp4_linear_impl(activation, activation_scale, weight, weight_scale, 0.25, 0.5, cache_id)

    assert first.shape == second.shape == (128, 16)
    assert len(prepare_calls) == 1
    assert prepare_calls[0][2:5] == (0.25, 0.5, (torch.device("cpu"), "linear", 16, 16, 128, None, False))
    assert getattr(owner, prims_ts_nvfp4._PREPARED_ATTR)["linear"]


def test_nvfp4_linear_quantizes_floating_input_on_request(monkeypatch: pytest.MonkeyPatch):
    owner = nn.Module()
    cache_id = prims_ts_nvfp4.register_prims_ts_nvfp4_layer(owner)
    packed = torch.empty((6, 4), dtype=torch.uint8)
    activation_scale = torch.empty((6,), dtype=torch.uint8)
    quantize_calls = []
    linear_calls = []

    def _quantize(x, input_scale_inv):
        quantize_calls.append((x, input_scale_inv))
        return packed, activation_scale

    def _linear(x, scale, *_args):
        linear_calls.append((x, scale))
        return torch.empty((x.shape[0], 16), dtype=torch.bfloat16)

    monkeypatch.setattr(prims_ts_nvfp4, "_QUANTIZE_OP", _quantize)
    monkeypatch.setattr(prims_ts_nvfp4, "_LINEAR_OP", _linear)
    input_scale_inv = torch.ones((1,), dtype=torch.float32)
    output = prims_ts_nvfp4.prims_ts_nvfp4_linear(
        torch.empty((2, 3, 8), dtype=torch.bfloat16),
        torch.empty((16, 8), dtype=torch.uint8),
        torch.empty((16,), dtype=torch.uint8),
        0.25,
        0.5,
        cache_id,
        input_scale_inv=input_scale_inv,
    )

    assert output.shape == (6, 16)
    assert len(quantize_calls) == len(linear_calls) == 1
    assert quantize_calls[0][0].shape == (6, 8)
    assert quantize_calls[0][1] is input_scale_inv
    assert linear_calls[0][0] is packed
    assert linear_calls[0][1] is activation_scale


def test_nvfp4_swiglu_quantizes_floating_input_on_request(monkeypatch: pytest.MonkeyPatch):
    packed = torch.empty((6, 4), dtype=torch.uint8)
    activation_scale = torch.empty((6,), dtype=torch.uint8)
    quantize_calls = []
    swiglu_calls = []

    def _quantize(x, input_scale_inv):
        quantize_calls.append((x, input_scale_inv))
        return packed, activation_scale

    def _swiglu(x, scale, *_args):
        swiglu_calls.append((x, scale))
        return packed, activation_scale

    monkeypatch.setattr(prims_ts_nvfp4, "_QUANTIZE_OP", _quantize)
    monkeypatch.setattr(prims_ts_nvfp4, "_SWIGLU_OP", _swiglu)
    input_scale_inv = torch.ones((1,), dtype=torch.float32)
    output = prims_ts_nvfp4.prims_ts_nvfp4_swiglu(
        torch.empty((2, 3, 8), dtype=torch.bfloat16),
        torch.empty((64, 8), dtype=torch.uint8),
        torch.empty((64,), dtype=torch.uint8),
        0.25,
        0.5,
        1,
        input_scale_inv=input_scale_inv,
    )

    assert output[0] is packed
    assert output[1] is activation_scale
    assert len(quantize_calls) == len(swiglu_calls) == 1
    assert quantize_calls[0][0].shape == (6, 8)
    assert quantize_calls[0][1] is input_scale_inv
    assert swiglu_calls[0][0] is packed
    assert swiglu_calls[0][1] is activation_scale


def test_nvfp4_qkv_quantizes_floating_input_on_request(monkeypatch: pytest.MonkeyPatch):
    packed = torch.empty((6, 4), dtype=torch.uint8)
    activation_scale = torch.empty((6,), dtype=torch.uint8)
    quantize_calls = []
    qkv_calls = []

    def _quantize(x, input_scale_inv):
        quantize_calls.append((x, input_scale_inv))
        return packed, activation_scale

    def _qkv(*args):
        qkv_calls.append(args)
        return torch.empty((6, 24), dtype=torch.bfloat16)

    monkeypatch.setattr(prims_ts_nvfp4, "_QUANTIZE_OP", _quantize)
    monkeypatch.setattr(prims_ts_nvfp4, "_QKV_QKNORM_ROPE_OP", _qkv)
    input_scale_inv = torch.ones((1,), dtype=torch.float32)
    output = prims_ts_nvfp4.prims_ts_nvfp4_qkv_qknorm_rope(
        torch.empty((2, 3, 8), dtype=torch.bfloat16),
        torch.empty((24, 8), dtype=torch.uint8),
        torch.empty((24,), dtype=torch.uint8),
        0.25,
        0.5,
        torch.empty((24,), dtype=torch.float32),
        torch.empty((8,), dtype=torch.bfloat16),
        torch.empty((8,), dtype=torch.bfloat16),
        torch.empty((6, 8), dtype=torch.float32),
        torch.arange(6),
        1,
        input_scale_inv=input_scale_inv,
    )

    assert output.shape == (6, 24)
    assert len(quantize_calls) == len(qkv_calls) == 1
    assert quantize_calls[0][0].shape == (6, 8)
    assert quantize_calls[0][1] is input_scale_inv
    assert qkv_calls[0][0] is packed
    assert qkv_calls[0][1] is activation_scale


def test_nvfp4_explicit_quantization_can_be_reused_across_operations(monkeypatch: pytest.MonkeyPatch):
    packed = torch.empty((4, 4), dtype=torch.uint8)
    activation_scale = torch.empty((4,), dtype=torch.uint8)
    quantize_calls = []
    projection_inputs = []

    def _quantize(x, input_scale_inv):
        quantize_calls.append((x, input_scale_inv))
        return packed, activation_scale

    def _projection(x, scale, *_args):
        projection_inputs.append((x, scale))
        return torch.empty((4, 16), dtype=torch.bfloat16)

    monkeypatch.setattr(prims_ts_nvfp4, "_QUANTIZE_OP", _quantize)
    monkeypatch.setattr(prims_ts_nvfp4, "_LINEAR_OP", _projection)
    monkeypatch.setattr(prims_ts_nvfp4, "_SWIGLU_OP", _projection)
    activation = prims_ts_nvfp4.prims_ts_nvfp4_quantize(
        torch.empty((4, 8), dtype=torch.bfloat16), torch.ones((1,), dtype=torch.float32)
    )
    weight = torch.empty((16, 8), dtype=torch.uint8)
    weight_scale = torch.empty((16,), dtype=torch.uint8)

    prims_ts_nvfp4.prims_ts_nvfp4_linear(
        activation,
        weight,
        weight_scale,
        0.25,
        0.5,
        1,
    )
    prims_ts_nvfp4.prims_ts_nvfp4_swiglu(
        activation,
        weight,
        weight_scale,
        0.25,
        0.5,
        1,
    )

    assert len(quantize_calls) == 1
    assert len(projection_inputs) == 2
    assert all(x is packed and scale is activation_scale for x, scale in projection_inputs)


def test_nvfp4_prequantized_tensor_requires_a_scale_pair():
    with pytest.raises(TypeError, match="packed NVFP4 input as a"):
        prims_ts_nvfp4.prims_ts_nvfp4_linear(
            torch.empty((4, 8), dtype=torch.uint8),
            torch.empty((16, 8), dtype=torch.uint8),
            torch.empty((16,), dtype=torch.uint8),
            0.25,
            0.5,
            1,
        )


def test_nvfp4_fake_swiglu_returns_packed_output_shape():
    activation = torch.empty((32, 16), dtype=torch.uint8)
    weight = torch.empty((64, 16), dtype=torch.uint8)

    packed, scales = prims_ts_nvfp4._nvfp4_swiglu_fake(
        activation,
        torch.empty((32,), dtype=torch.uint8),
        weight,
        torch.empty((64,), dtype=torch.uint8),
        0.25,
        1.0,
        0,
    )

    assert packed.shape == (32, 16)
    assert scales.dtype is torch.uint8
    assert packed.dtype is torch.uint8


def test_nvfp4_provider_registration_rejects_a_conflicting_name():
    prims_ts_nvfp4.register_prims_ts_nvfp4_config_provider("test-consumer-conflict", _test_config_provider)

    def _other_provider(*_args):
        return SimpleNamespace()

    try:
        prims_ts_nvfp4.register_prims_ts_nvfp4_config_provider("test-consumer-conflict", _other_provider)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("conflicting config provider registration should fail")

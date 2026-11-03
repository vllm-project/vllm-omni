# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ConvRot rotated W8A8 / W4A4 diffusion quantization."""

import math

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.convrot_config import (
    ConvRotInt4OnlineLinearMethod,
    ConvRotInt8OnlineLinearMethod,
    pack_int4,
    quantize_int4,
    regular_hadamard,
    rotate,
    rotation_group_size,
    unpack_int4,
)
from vllm_omni.quantization.factory import SUPPORTED_QUANTIZATION_METHODS

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

cuda_available = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="GPU platform not available.")


@pytest.mark.parametrize("size", [4, 16, 64, 256, 1024])
def test_regular_hadamard_is_orthogonal_regular_and_exact(size):
    h = regular_hadamard(size).double()
    torch.testing.assert_close(h @ h.T, torch.eye(size, dtype=torch.float64))
    torch.testing.assert_close(h, h.T)
    # Regular: every row and column of sqrt(n) * H sums to +-sqrt(n).
    assert torch.allclose(h.sum(0).abs(), torch.ones(size, dtype=torch.float64))
    assert torch.allclose(h.sum(1).abs(), torch.ones(size, dtype=torch.float64))
    # Entries are +-2^-k, exact in half precision.
    assert torch.equal(h.abs().unique(), torch.tensor([1 / math.sqrt(size)], dtype=torch.float64))
    assert torch.equal(regular_hadamard(size).to(torch.bfloat16).float(), regular_hadamard(size))


@pytest.mark.parametrize("size", [0, 2, 8, 32, 48])
def test_rotation_group_size_must_be_a_power_of_four(size):
    with pytest.raises(ValueError, match="power of four"):
        regular_hadamard(size)


@pytest.mark.parametrize(
    "in_features,requested,expected",
    [(3072, 256, 256), (320, 256, 64), (5120, 1024, 1024), (30, 256, None), (36, 256, 4)],
)
def test_group_size_falls_back_to_a_power_of_four_that_tiles_the_input(in_features, requested, expected):
    assert rotation_group_size(in_features, requested) == expected


def test_rotation_preserves_the_product_and_spreads_outliers():
    torch.manual_seed(0)
    x = torch.randn(32, 512, dtype=torch.float64)
    x[:, 7] *= 100.0  # one outlier channel, as in DiT activations
    w = torch.randn(96, 512, dtype=torch.float64)
    h = regular_hadamard(256).double()
    torch.testing.assert_close(rotate(x, h) @ rotate(w, h).T, x @ w.T)
    assert rotate(x, h).abs().amax() < x.abs().amax() / 8


def test_int4_values_pack_two_per_byte():
    values = torch.arange(-8, 8, dtype=torch.int8).repeat(3, 2)
    packed = pack_int4(values)
    assert packed.dtype == torch.uint8 and packed.shape == (3, 16)
    assert torch.equal(unpack_int4(packed), values)
    with pytest.raises(ValueError, match="even"):
        pack_int4(torch.zeros(2, 3, dtype=torch.int8))


def test_int4_quantization_is_symmetric_per_row():
    x = torch.tensor([[0.0, 7.0, -7.0, 3.5], [0.0, 0.0, 0.0, 0.0]])
    q, scale = quantize_int4(x)
    assert torch.equal(q, torch.tensor([[0, 7, -7, 4], [0, 0, 0, 0]], dtype=torch.int8))
    assert scale.shape == (2, 1) and scale[0, 0] == 1.0 and scale[1, 0] > 0


def test_factory_builds_both_methods():
    assert {"convrot_int8", "convrot_int4"} <= set(SUPPORTED_QUANTIZATION_METHODS)
    int8 = build_quant_config("convrot_int8")
    assert int8.get_name() == "convrot_int8" and int8.rotation_group_size == 256
    int4 = build_quant_config(
        {"method": "convrot_int4", "rotation_group_size": 64, "ignored_layers": ["proj_out"], "int8_layers": ["to_out"]}
    )
    assert (int4.get_name(), int4.rotation_group_size, int4.ignored_layers, int4.int8_layers) == (
        "convrot_int4",
        64,
        ["proj_out"],
        ["to_out"],
    )
    with pytest.raises(ValueError, match="convrot_int4 only"):
        build_quant_config({"method": "convrot_int8", "int8_layers": ["to_out"]})
    with pytest.raises(ValueError, match="power of four"):
        build_quant_config({"method": "convrot_int4", "rotation_group_size": 128})


def test_layers_route_to_int4_int8_or_unquantized(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    for cls in (ConvRotInt8OnlineLinearMethod, ConvRotInt4OnlineLinearMethod):
        mocker.patch.object(cls, "__init__", lambda self, quant_config: None)
    layer = mocker.Mock(spec=LinearBase)
    config = build_quant_config(
        {"method": "convrot_int4", "ignored_layers": ["proj_out"], "int8_layers": ["attn.to_out"]}
    )
    assert isinstance(config.get_quant_method(layer, "blocks.0.ff.net.2"), ConvRotInt4OnlineLinearMethod)
    assert isinstance(config.get_quant_method(layer, "blocks.0.attn.to_out"), ConvRotInt8OnlineLinearMethod)
    assert isinstance(config.get_quant_method(layer, "proj_out"), UnquantizedLinearMethod)
    # Whole path components only; a subtree name covers every layer below it.
    subtree = build_quant_config(
        {"method": "convrot_int4", "ignored_layers": ["token_refiner"], "int8_layers": ["proj"]}
    )
    assert isinstance(subtree.get_quant_method(layer, "token_refiner.blocks.0.attn.qkv_proj"), UnquantizedLinearMethod)
    assert isinstance(subtree.get_quant_method(layer, "blocks.0.attn.out_proj"), ConvRotInt4OnlineLinearMethod)
    subtree.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    subtree.int8_layers = ["attn.v_proj"]
    assert isinstance(subtree.get_quant_method(layer, "blocks.0.attn.qkv_proj"), ConvRotInt8OnlineLinearMethod)
    assert config.get_quant_method(torch.nn.Module(), "norm") is None
    assert isinstance(build_quant_config("convrot_int8").get_quant_method(layer, "x"), ConvRotInt8OnlineLinearMethod)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    with pytest.raises(NotImplementedError, match="CUDA"):
        config.get_quant_method(layer, "blocks.0.ff.net.2")


@cuda_available
class TestCudaConvRot:
    """Real kernels: rotated quantization tracks BF16 and beats unrotated quantization on outliers."""

    @staticmethod
    def _layer(weight):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        layer.logical_widths = [weight.shape[0]]
        layer.input_size_per_partition = weight.shape[1]
        layer.output_size_per_partition = weight.shape[0]
        layer.orig_dtype = weight.dtype
        return layer

    @staticmethod
    def _inputs():
        torch.manual_seed(0)
        x = torch.randn(4, 64, 1024, dtype=torch.bfloat16, device="cuda")
        x[..., [3, 500, 777]] *= 80.0  # massive activation channels
        weight = torch.randn(768, 1024, dtype=torch.bfloat16, device="cuda") / 32
        weight[:, 11] *= 30.0  # an outlier input column in the weight
        bias = torch.randn(768, dtype=torch.bfloat16, device="cuda")
        return x, weight, bias

    @staticmethod
    def _error(out, ref):
        return ((out.float() - ref.float()).norm() / ref.float().norm()).item()

    def _quantized_error(self, method, weight, x, bias):
        layer = self._layer(weight)
        method.process_weights_after_loading(layer)
        out = method.apply(layer, x, bias)
        assert out.shape == (*x.shape[:-1], weight.shape[0]) and out.dtype == torch.bfloat16
        return layer, self._error(out, torch.nn.functional.linear(x, weight, bias))

    def test_int8_rotation_removes_the_outlier_error_of_plain_int8(self):
        from vllm_omni.quantization.int8_config import DiffusionInt8Config, Int8OnlineLinearMethod

        x, weight, bias = self._inputs()
        _, plain = self._quantized_error(Int8OnlineLinearMethod(DiffusionInt8Config()), weight, x, bias)
        layer, rotated = self._quantized_error(
            ConvRotInt8OnlineLinearMethod(build_quant_config("convrot_int8")), weight, x, bias
        )
        assert layer.weight.dtype == torch.int8 and layer.convrot_hadamard.shape == (256, 256)
        assert rotated < 0.02 and rotated < plain / 4

    def test_int4_rotation_bounds_w4a4_error_on_outliers(self):
        x, weight, bias = self._inputs()
        unrotated = ConvRotInt4OnlineLinearMethod(build_quant_config("convrot_int4"))
        unrotated.rotation_group_size = 1  # no power-of-four group: quantize without rotation
        _, plain = self._quantized_error(unrotated, weight, x, bias)
        layer, rotated = self._quantized_error(
            ConvRotInt4OnlineLinearMethod(build_quant_config("convrot_int4")), weight, x, bias
        )
        # Packed INT4 weights: a quarter of BF16 storage.
        assert layer.weight.dtype == torch.uint8 and layer.weight.shape == (768, 512)
        assert rotated < 0.25 and rotated < plain / 3

    def test_int8_layers_inside_convrot_int4_run_rotated_int8(self):
        x, weight, bias = self._inputs()
        config = build_quant_config({"method": "convrot_int4", "int8_layers": ["attn.to_out"]})
        layer, error = self._quantized_error(ConvRotInt8OnlineLinearMethod(config), weight, x, bias)
        assert layer.weight.dtype == torch.int8 and error < 0.02

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.quantization import svdquant_config as svdquant

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("settings", [{"activation_bits": 16}, {"rank": 16}, {"rank": 160}])
def test_native_backend_rejects_unsupported_contract(settings):
    with pytest.raises(ValueError, match="Native FlashInfer SVDQuant requires"):
        svdquant.DiffusionSVDQuantConfig(linear_backend="flashinfer", **settings)


def test_native_backend_metadata_roundtrip():
    config = svdquant.DiffusionSVDQuantConfig.from_config({"linear_backend": "flashinfer", "rank": 32})
    assert config.linear_backend == "flashinfer"
    assert "linear_backend='flashinfer'" in repr(config)
    with pytest.raises(ValueError, match="linear_backend must"):
        svdquant.DiffusionSVDQuantConfig(linear_backend="unknown")


def test_native_epilogue_preserves_original_input_channel_scale_and_bias(monkeypatch):
    monkeypatch.setattr(svdquant, "_assert_supported", lambda: None)
    generator = torch.Generator().manual_seed(6493)
    base = (torch.randn(128, 128, generator=generator) * 0.125).bfloat16()
    calls = []

    def native(x, weight, sf, alpha, pre_quant, down, up, global_scale, bias=None, backend=None):
        assert weight.dtype == sf.dtype == torch.uint8
        assert backend == "cute-dsl" and bias is None
        assert torch.equal(global_scale, torch.ones(1))
        calls.append((pre_quant.clone(), down.clone()))
        return (torch.addmm(torch.nn.functional.linear(x * pre_quant, base), x @ down, up.T) * alpha).bfloat16()

    monkeypatch.setattr(svdquant, "_flashinfer_svdquant", lambda: (native, lambda scales: scales.flatten(), "cute-dsl"))
    method = svdquant.DiffusionSVDQuantLinearMethod(svdquant.DiffusionSVDQuantConfig(linear_backend="flashinfer"))
    layer = torch.nn.Module()
    method.create_weights(layer, 128, [128], 128, 128, torch.bfloat16)
    layer.wtscale.data.fill_(2)
    layer.wcscales.data.copy_(torch.tensor([0.5, 1, 2, 4] * 32).bfloat16())
    layer.smooth_factor.data.copy_(torch.tensor([2, 4] * 64).bfloat16())
    layer.proj_down.data.copy_(torch.randn(128, 32, generator=generator).bfloat16() * 0.25)
    layer.proj_up.data.copy_(torch.randn(128, 32, generator=generator).bfloat16() * 0.25)
    original_down, original_up = layer.proj_down.detach().clone(), layer.proj_up.detach().clone()
    original_smooth, original_channels = layer.smooth_factor.detach().clone(), layer.wcscales.detach().clone()
    x = torch.randn(2, 128, 3, generator=generator).bfloat16().transpose(1, 2)
    bias = torch.linspace(-1, 1, 128).bfloat16()
    flat = x.reshape(-1, 128)
    expected = torch.nn.functional.linear(flat / original_smooth, base * 2) * original_channels
    expected = torch.addmm(expected, flat @ original_down, original_up.T) + bias
    method.process_weights_after_loading(layer)
    actual = method.apply(layer, x, bias)
    torch.testing.assert_close(actual, expected.reshape(2, 3, 128), rtol=0.02, atol=0.25)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], original_smooth.reciprocal())
    assert torch.equal(calls[0][1], original_down)
    assert not hasattr(layer, "wscales") and not hasattr(layer, "wcscales")


class _FakeNvfp4Kernel:
    def __init__(self, base_weight: torch.Tensor | None = None) -> None:
        self.base_weight = base_weight
        self.processed = False
        self.last_input: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        del layer
        self.processed = True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer
        assert bias is None
        assert self.base_weight is not None
        self.last_input = x.clone()
        return torch.mm(x, self.base_weight)


def _register_parameter(
    layer: torch.nn.Module,
    name: str,
    value: torch.Tensor,
) -> None:
    layer.register_parameter(
        name,
        torch.nn.Parameter(value, requires_grad=False),
    )


def test_supports_blackwell_nvfp4_targets() -> None:
    assert svdquant._supports_capability(SimpleNamespace(major=10, minor=0))
    assert svdquant._supports_capability(SimpleNamespace(major=10, minor=3))
    assert not svdquant._supports_capability(SimpleNamespace(major=11, minor=0))
    assert svdquant._supports_capability(SimpleNamespace(major=12, minor=0))
    assert not svdquant._supports_capability(None)


def test_rejects_nvfp4_backend_with_incompatible_layer_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svdquant._nvfp4_kernel.cache_clear()
    monkeypatch.setattr(
        svdquant,
        "init_nvfp4_linear_kernel",
        lambda: _FakeNvfp4Kernel(),
    )

    with pytest.raises(RuntimeError, match="incompatible with the SVDQuant checkpoint"):
        svdquant._nvfp4_kernel()

    svdquant._nvfp4_kernel.cache_clear()


def test_prepare_weights_uses_existing_nvfp4_layout_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _FakeNvfp4Kernel()
    monkeypatch.setattr(svdquant, "_assert_supported", lambda: None)
    monkeypatch.setattr(svdquant, "_nvfp4_kernel", lambda: kernel)
    method = svdquant.DiffusionSVDQuantLinearMethod(svdquant.DiffusionSVDQuantConfig())
    layer = torch.nn.Module()
    qweight = torch.arange(16, dtype=torch.int8).reshape(4, 4)
    wscales = torch.arange(8, dtype=torch.float32).to(torch.float8_e4m3fn).reshape(2, 4)
    _register_parameter(layer, "qweight", qweight)
    _register_parameter(layer, "wscales", wscales)
    _register_parameter(layer, "wtscale", torch.tensor([2.0], dtype=torch.bfloat16))
    _register_parameter(
        layer,
        "wcscales",
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16),
    )

    method.process_weights_after_loading(layer)

    assert kernel.processed
    assert not hasattr(layer, "qweight")
    assert not hasattr(layer, "wscales")
    assert layer.weight.dtype == torch.uint8
    assert torch.equal(layer.weight.view(torch.int8), qweight)
    assert torch.equal(layer.weight_scale, wscales.transpose(0, 1))
    torch.testing.assert_close(layer.alpha, torch.tensor([2.0]))
    assert layer.input_global_scale_inv.dtype == torch.float32
    assert layer.output_channel_scale.shape == (4,)


def test_prepare_weights_drops_identity_output_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _FakeNvfp4Kernel()
    monkeypatch.setattr(svdquant, "_assert_supported", lambda: None)
    monkeypatch.setattr(svdquant, "_nvfp4_kernel", lambda: kernel)
    method = svdquant.DiffusionSVDQuantLinearMethod(svdquant.DiffusionSVDQuantConfig())
    layer = torch.nn.Module()
    _register_parameter(layer, "qweight", torch.zeros(4, 4, dtype=torch.int8))
    _register_parameter(
        layer,
        "wscales",
        torch.ones(2, 4, dtype=torch.float8_e4m3fn),
    )
    _register_parameter(layer, "wtscale", torch.ones(1, dtype=torch.bfloat16))
    _register_parameter(layer, "wcscales", torch.ones(4, dtype=torch.bfloat16))

    method.process_weights_after_loading(layer)

    assert layer.output_channel_scale is None


def test_compatibility_path_applies_qkv_scale_and_rank_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_weight = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.bfloat16,
    )
    kernel = _FakeNvfp4Kernel(base_weight)
    monkeypatch.setattr(svdquant, "_assert_supported", lambda: None)
    monkeypatch.setattr(svdquant, "_nvfp4_kernel", lambda: kernel)
    method = svdquant.DiffusionSVDQuantLinearMethod(svdquant.DiffusionSVDQuantConfig())

    layer = torch.nn.Module()
    layer.smooth_factor = torch.tensor([2.0, 4.0], dtype=torch.bfloat16)
    layer.proj_down = torch.tensor([[1.0], [2.0]], dtype=torch.bfloat16)
    layer.proj_up = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.bfloat16)
    layer.output_channel_scale = torch.tensor([1.0, 2.0, 4.0], dtype=torch.bfloat16)
    layer.output_size_per_partition = 3
    x = torch.tensor([[2.0, 4.0], [4.0, 8.0]], dtype=torch.bfloat16)
    bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)

    actual = method.apply(layer, x, bias)

    smoothed = x / layer.smooth_factor
    expected = torch.mm(smoothed, base_weight)
    expected = expected * layer.output_channel_scale
    expected = expected + torch.mm(torch.mm(x, layer.proj_down), layer.proj_up.T)
    expected = expected + bias
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(kernel.last_input, smoothed)


def test_compatibility_path_rejects_fp16_activations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svdquant, "_assert_supported", lambda: None)
    method = svdquant.DiffusionSVDQuantLinearMethod(svdquant.DiffusionSVDQuantConfig())
    layer = torch.nn.Module()
    with pytest.raises(ValueError, match="requires BF16 activations"):
        method.apply(
            layer,
            torch.zeros(1, 16, dtype=torch.float16),
        )

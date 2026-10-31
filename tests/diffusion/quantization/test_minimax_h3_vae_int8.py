# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests; NPU operators below are numerical test doubles.

The adapter delegates to the real NPU INT8 method. These tests verify Python
routing and tensor contracts, not Ascend kernel support, quality or performance.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3 import vae_int8
from vllm_omni.quantization import int8_config
from vllm_omni.quantization.component_config import ComponentQuantizationConfig
from vllm_omni.quantization.int8_config import DiffusionInt8Config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def quant_config():
    return DiffusionInt8Config(is_checkpoint_int8_serialized=False, activation_scheme="dynamic")


@pytest.fixture
def cpu_kernels(monkeypatch):
    """Replace device execution only; keep the existing INT8 method intact."""
    state = SimpleNamespace(dynamic_inputs=[], matmuls=[], autocast=False, autocast_dtype=torch.bfloat16)

    def dynamic_quant(value):
        state.dynamic_inputs.append(value.detach().clone())
        scale = value.float().abs().amax(dim=-1).clamp_min(1e-12) / 127
        quantized = (value.float() / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
        return quantized, scale

    def quant_matmul(value, weight, weight_scale, *, bias, pertoken_scale, output_dtype):
        assert bias is None, "FP32 VAE bias must be added outside the INT8 kernel"
        assert not state.autocast, "INT8 execution must disable ambient NPU autocast"
        state.matmuls.append((value.shape, weight.shape, output_dtype))
        result = (value.float() * pertoken_scale[:, None]) @ (weight.float() * weight_scale[None, :])
        return result.to(output_dtype)

    @contextmanager
    def autocast(device_type, enabled=True, **kwargs):
        assert device_type == "npu"
        previous = state.autocast
        state.autocast = enabled
        try:
            yield
        finally:
            state.autocast = previous

    original_is_autocast_enabled = torch.is_autocast_enabled
    original_get_autocast_dtype = torch.get_autocast_dtype
    monkeypatch.setattr(vae_int8, "_require_npu_device", lambda device: None)
    monkeypatch.setattr(
        int8_config,
        "torch_npu",
        SimpleNamespace(npu_dynamic_quant=dynamic_quant, npu_quant_matmul=quant_matmul),
    )
    monkeypatch.setattr(torch, "autocast", autocast)
    monkeypatch.setattr(
        torch,
        "is_autocast_enabled",
        lambda device_type="cuda": state.autocast
        if device_type == "npu"
        else original_is_autocast_enabled(device_type),
    )
    monkeypatch.setattr(
        torch,
        "get_autocast_dtype",
        lambda device_type: state.autocast_dtype if device_type == "npu" else original_get_autocast_dtype(device_type),
    )
    return state


def _linear(source, quant_config, compute_dtype=torch.float16):
    return vae_int8.H3VAEInt8Linear.from_linear(
        source,
        quant_config,
        quant_device=torch.device("cpu"),
        storage_device=torch.device("cpu"),
        compute_dtype=compute_dtype,
    )


def _decoder(block_count=2):
    decoder = nn.Module()
    decoder.proj_in = nn.Linear(8, 8)
    decoder.proj_out = nn.Linear(8, 8)
    decoder.conv = nn.Conv2d(1, 1, 1)
    decoder.transformer_blocks = nn.ModuleList()
    for _ in range(block_count):
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.to_qkv = nn.Linear(8, 24)
        block.attn.to_out = nn.Linear(8, 8)
        block.ff = nn.Module()
        block.ff.w1 = nn.Linear(8, 32)
        block.ff.w2 = nn.Linear(16, 8)
        block.extra = nn.Linear(8, 8)
        decoder.transformer_blocks.append(block)
    return decoder


def _install(decoder, quant_config):
    return vae_int8.install_h3_video_vae_int8(
        decoder,
        quant_config,
        quant_device=torch.device("cpu"),
        storage_device=torch.device("cpu"),
        compute_dtype=torch.float16,
    )


def test_vae_requires_explicit_component_route(quant_config):
    assert vae_int8.resolve_h3_video_vae_quant_config(None) is None
    assert vae_int8.resolve_h3_video_vae_quant_config(quant_config) is None
    assert (
        vae_int8.resolve_h3_video_vae_quant_config(ComponentQuantizationConfig({"transformer": quant_config})) is None
    )
    assert (
        vae_int8.resolve_h3_video_vae_quant_config(ComponentQuantizationConfig({}, default_config=quant_config)) is None
    )
    assert (
        vae_int8.resolve_h3_video_vae_quant_config(
            ComponentQuantizationConfig({"video_vae": None}, default_config=quant_config)
        )
        is None
    )
    routed = ComponentQuantizationConfig({"transformer": None, "video_vae": quant_config})
    assert vae_int8.resolve_h3_video_vae_quant_config(routed) is quant_config
    assert routed.resolve("transformer") is None


def test_factory_component_json_builds_online_int8_without_quantizing_dit():
    from vllm_omni.quantization.factory import build_quant_config

    config = build_quant_config(
        {
            "transformer": None,
            "video_vae": {"method": "int8", "activation_scheme": "dynamic"},
        }
    )
    assert isinstance(config, ComponentQuantizationConfig)
    vae_config = vae_int8.resolve_h3_video_vae_quant_config(config)
    assert isinstance(vae_config, DiffusionInt8Config)
    assert not vae_config.is_checkpoint_int8_serialized
    assert vae_config.activation_scheme == "dynamic"
    assert config.resolve("transformer") is None


def test_exact_vae_route_does_not_use_longest_prefix(quant_config):
    routed = ComponentQuantizationConfig({"video": quant_config, "video_vae.decoder": quant_config})
    with pytest.raises(ValueError, match="exact video_vae"):
        vae_int8.resolve_h3_video_vae_quant_config(routed)


def test_cpu_execution_is_rejected_without_test_double(quant_config):
    with pytest.raises((ValueError, RuntimeError), match="NPU|npu"):
        vae_int8.validate_h3_video_vae_quant_config(quant_config, device=torch.device("cpu"))


@pytest.mark.parametrize(
    "kwargs",
    [{"encode_only": True}, {"decoder_source": "decoder-nvfp4.safetensors"}],
)
def test_unsupported_vae_combinations_are_rejected(cpu_kernels, quant_config, kwargs):
    with pytest.raises(ValueError):
        vae_int8.validate_h3_video_vae_quant_config(quant_config, device=torch.device("cpu"), **kwargs)


def test_serialized_int8_checkpoint_is_rejected(cpu_kernels):
    config = DiffusionInt8Config(is_checkpoint_int8_serialized=True, activation_scheme="dynamic")
    with pytest.raises(ValueError):
        vae_int8.validate_h3_video_vae_quant_config(config, device=torch.device("cpu"))


def test_wrong_quantization_method_is_rejected(cpu_kernels):
    config = SimpleNamespace(get_name=lambda: "fp8")
    with pytest.raises((TypeError, ValueError)):
        vae_int8.validate_h3_video_vae_quant_config(config, device=torch.device("cpu"))


def test_real_method_converts_registered_tensors_once(cpu_kernels, quant_config):
    source = nn.Linear(8, 6)
    layer = _linear(source, quant_config)
    assert isinstance(layer._int8_method, int8_config.NPUInt8OnlineLinearMethod)
    assert not hasattr(layer, "quant_method")
    assert not any(isinstance(child, nn.Linear) for child in layer.modules())
    parameters = dict(layer.named_parameters())
    assert parameters["weight"].shape == (8, 6)
    assert parameters["weight"].dtype == torch.int8
    assert parameters["weight"].is_contiguous()
    assert parameters["weight_scale"].shape == (6,)
    assert parameters["weight_scale"].dtype == torch.float32
    assert parameters["bias"].dtype == torch.float32
    assert all(not value.requires_grad for value in parameters.values())
    assert len(cpu_kernels.dynamic_inputs) == 1
    assert cpu_kernels.dynamic_inputs[0].dtype == torch.float16
    weight = layer.weight
    scale = layer.weight_scale
    layer._int8_method.process_weights_after_loading(layer)
    assert layer.weight is weight
    assert layer.weight_scale is scale
    assert len(cpu_kernels.dynamic_inputs) == 1


@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 3, 8), (2, 3, 4, 8)])
@pytest.mark.parametrize("compute_dtype", [torch.float16, torch.bfloat16])
def test_forward_preserves_leading_shape_and_parent_dtype(cpu_kernels, quant_config, shape, compute_dtype):
    torch.manual_seed(7)
    source = nn.Linear(8, 6)
    layer = _linear(source, quant_config, compute_dtype)
    value = torch.randn(shape)
    result = layer(value)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (*shape[:-1], 6)
    assert result.dtype == torch.float32
    assert cpu_kernels.dynamic_inputs[-1].dtype == compute_dtype
    assert cpu_kernels.matmuls[-1][2] == compute_dtype
    torch.testing.assert_close(result, source(value), atol=0.04, rtol=0.04)
    # Repeated execution does not reconvert weights.
    layer(value)
    assert len(cpu_kernels.dynamic_inputs) == 3


def test_noncontiguous_input_uses_real_method_reshape(cpu_kernels, quant_config):
    layer = _linear(nn.Linear(8, 6), quant_config)
    value = torch.randn(2, 8, 3).transpose(1, 2)
    assert not value.is_contiguous()
    assert layer(value).shape == (2, 3, 6)
    assert cpu_kernels.matmuls[-1][0] == (6, 8)


def test_fp32_bias_is_added_after_quantized_matmul(cpu_kernels, quant_config):
    source = nn.Linear(8, 6)
    with torch.no_grad():
        source.weight.zero_()
        source.bias.copy_(torch.tensor([1.0001, -2.0003, 0.0001, 3.14159, 0.12345, -0.98765]))
    layer = _linear(source, quant_config)
    result = layer(torch.randn(2, 8))
    torch.testing.assert_close(result, source.bias.expand(2, -1), atol=0, rtol=0)


def test_bias_free_linear_remains_bias_free(cpu_kernels, quant_config):
    layer = _linear(nn.Linear(8, 6, bias=False), quant_config)
    assert layer.bias is None
    assert "bias" not in dict(layer.named_parameters())
    assert layer(torch.randn(2, 8)).shape == (2, 6)


def test_npu_autocast_controls_public_dtype_and_is_restored(cpu_kernels, quant_config):
    layer = _linear(nn.Linear(8, 6), quant_config, compute_dtype=torch.float16)
    cpu_kernels.autocast = True
    cpu_kernels.autocast_dtype = torch.bfloat16
    result = layer(torch.randn(2, 8))
    assert result.dtype == torch.bfloat16
    assert cpu_kernels.autocast
    assert cpu_kernels.dynamic_inputs[-1].dtype == torch.bfloat16


@pytest.mark.parametrize("shape", [(0, 8), (2, 0, 8)])
def test_empty_input_does_not_invoke_npu_kernels(cpu_kernels, quant_config, shape):
    layer = _linear(nn.Linear(8, 6), quant_config)
    before = len(cpu_kernels.dynamic_inputs)
    assert layer(torch.empty(shape)).shape == (*shape[:-1], 6)
    assert len(cpu_kernels.dynamic_inputs) == before
    assert not cpu_kernels.matmuls


def test_installer_selects_only_four_linears_per_block(cpu_kernels, quant_config):
    decoder = _decoder()
    untouched = [decoder.proj_in, decoder.proj_out, decoder.conv]
    untouched.extend(block.extra for block in decoder.transformer_blocks)
    report = _install(decoder, quant_config)
    assert report.block_count == 2
    assert report.candidate_count == 8
    assert report.quantized_count == 8
    assert report.ignored_count == 0
    for block in decoder.transformer_blocks:
        for layer in (block.attn.to_qkv, block.attn.to_out, block.ff.w1, block.ff.w2):
            assert isinstance(layer, vae_int8.H3VAEInt8Linear)
    assert [decoder.proj_in, decoder.proj_out, decoder.conv] == untouched[:3]
    assert [block.extra for block in decoder.transformer_blocks] == untouched[3:]
    assert len(cpu_kernels.dynamic_inputs) == 8


def test_installer_is_idempotent(cpu_kernels, quant_config):
    decoder = _decoder(1)
    first_report = _install(decoder, quant_config)
    first_layer = decoder.transformer_blocks[0].attn.to_qkv
    count = len(cpu_kernels.dynamic_inputs)
    second_report = _install(decoder, quant_config)
    assert second_report.quantized_count == first_report.quantized_count
    assert decoder.transformer_blocks[0].attn.to_qkv is first_layer
    assert len(cpu_kernels.dynamic_inputs) == count


def test_explicit_ignored_layer_remains_original(cpu_kernels):
    config = DiffusionInt8Config(
        ignored_layers=["video_vae.decoder.transformer_blocks.0.attn.to_qkv"],
    )
    decoder = _decoder(1)
    original = decoder.transformer_blocks[0].attn.to_qkv
    report = _install(decoder, config)
    assert report.candidate_count == 4
    assert report.quantized_count == 3
    assert report.ignored_count == 1
    assert decoder.transformer_blocks[0].attn.to_qkv is original


def test_structural_failure_is_detected_before_any_conversion(cpu_kernels, quant_config):
    decoder = _decoder(2)
    original = decoder.transformer_blocks[0].attn.to_qkv
    decoder.transformer_blocks[1].ff.w2 = nn.Identity()
    with pytest.raises((TypeError, ValueError)):
        _install(decoder, quant_config)
    assert decoder.transformer_blocks[0].attn.to_qkv is original
    assert not cpu_kernels.dynamic_inputs


def test_overwide_linear_rejected_before_partial_install(cpu_kernels, quant_config):
    decoder = _decoder(2)
    original = decoder.transformer_blocks[0].attn.to_qkv
    decoder.transformer_blocks[1].ff.w2 = nn.Linear(1, 65536, bias=False)
    with pytest.raises(ValueError):
        _install(decoder, quant_config)
    assert decoder.transformer_blocks[0].attn.to_qkv is original
    assert not cpu_kernels.dynamic_inputs


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.float64])
def test_unsupported_compute_dtype_rejected(cpu_kernels, quant_config, compute_dtype):
    with pytest.raises(ValueError):
        _linear(nn.Linear(8, 6), quant_config, compute_dtype)


@pytest.mark.parametrize("invalid_state", ["meta", "int8", "nonfinite", "tied", "hook", "compiled"])
def test_unsafe_source_state_is_rejected_before_weight_conversion(cpu_kernels, quant_config, invalid_state):
    decoder = _decoder(2)
    original = decoder.transformer_blocks[0].attn.to_qkv
    target = decoder.transformer_blocks[1].attn.to_out
    if invalid_state == "meta":
        decoder.transformer_blocks[1].attn.to_out = nn.Linear(8, 8, device="meta")
    elif invalid_state == "int8":
        target.weight = nn.Parameter(target.weight.to(torch.int8), requires_grad=False)
    elif invalid_state == "nonfinite":
        with torch.no_grad():
            target.weight[0, 0] = float("nan")
    elif invalid_state == "tied":
        target.weight = decoder.transformer_blocks[0].attn.to_out.weight
    elif invalid_state == "hook":
        target.register_forward_hook(lambda module, args, output: output)
    else:
        decoder._compiled_call_impl = lambda *args, **kwargs: None
    with pytest.raises(ValueError):
        _install(decoder, quant_config)
    assert decoder.transformer_blocks[0].attn.to_qkv is original
    assert not cpu_kernels.dynamic_inputs


def test_excluding_all_candidates_fails_explicit_enable(cpu_kernels):
    ignored = [
        f"video_vae.decoder.transformer_blocks.0.{leaf}" for leaf in ("attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2")
    ]
    with pytest.raises(ValueError, match="every candidate"):
        _install(_decoder(1), DiffusionInt8Config(ignored_layers=ignored))
    assert not cpu_kernels.dynamic_inputs


def test_reinstall_with_different_exclusions_is_rejected(cpu_kernels, quant_config):
    decoder = _decoder(1)
    _install(decoder, quant_config)
    before = len(cpu_kernels.dynamic_inputs)
    different = DiffusionInt8Config(ignored_layers=["video_vae.decoder.transformer_blocks.0.attn.to_qkv"])
    with pytest.raises(ValueError, match="different configuration"):
        _install(decoder, different)
    assert len(cpu_kernels.dynamic_inputs) == before


def test_dtype_cast_after_install_is_rejected_before_quantized_matmul(cpu_kernels, quant_config):
    layer = _linear(nn.Linear(8, 6), quant_config)
    layer.to(dtype=torch.bfloat16)
    assert layer.weight.dtype == torch.int8
    with pytest.raises(RuntimeError, match="scale dtypes"):
        layer(torch.randn(2, 8))
    assert not cpu_kernels.matmuls

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for delivery-driven native mixed MX loading."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod

from vllm_omni.diffusion.models.minimax_h3 import native_mx

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _checkpoint(description):
    return SimpleNamespace(
        tensors=description,
        description=description,
        execution_mode="native_mx",
        diffusers_weights=True,
    )


def test_delivery_bits_and_fused_mapping_are_preserved(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    description = {f"transformer_blocks.2.attn.to_{role}.weight": "W4A4_MXFP4_DUALSCALE" for role in "qkv"}
    description["transformer_blocks.2.ff.net.2.weight"] = "W8A8_MXFP8"
    description["proj_in.weight"] = "FLOAT"
    config = native_mx.H3NativeMXConfig.from_checkpoint(_checkpoint(description))
    assert config.targets == {"blocks.2.attn.qkv_proj": 4, "blocks.2.mlp.fc2": 8}
    for prefix, bits in (("blocks.2.attn.qkv_proj", 4), ("transformer.blocks.2.mlp.fc2", 8)):
        layer = ReplicatedLinear(64, 32, quant_config=config, prefix=prefix)
        assert isinstance(layer.quant_method, native_mx.NativeMXLinearMethod)
        assert layer.quant_method.bits == bits
    ordinary = ReplicatedLinear(64, 32, quant_config=config, prefix="video_patch_proj")
    assert isinstance(ordinary.quant_method, UnquantizedLinearMethod)


def test_fused_mixed_schemes_fail_before_loading():
    description = {
        "transformer_blocks.2.attn.to_q.weight": "W4A4_MXFP4_DUALSCALE",
        "transformer_blocks.2.attn.to_k.weight": "W8A8_MXFP8",
    }
    with pytest.raises(ValueError, match="must share one delivery scheme"):
        native_mx.H3NativeMXConfig.from_checkpoint(_checkpoint(description))


@pytest.mark.parametrize("bits", [4, 8])
@torch.no_grad()
def test_lazy_loading_registers_runtime_scale_after_complete_weight(monkeypatch, bits):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm_omni.diffusion.distributed.utils.get_local_device", lambda: torch.device("cpu"))
    calls = []

    def quantize(weight, actual_bits, device):
        calls.append((weight.clone(), actual_bits, device))
        return torch.zeros(8, 32 if bits == 4 else 64, dtype=torch.uint8), torch.ones(8, 1, 2, dtype=torch.uint8)

    monkeypatch.setattr(native_mx, "quantize_mx_weight", quantize)
    method = native_mx.NativeMXLinearMethod(bits)
    layer = nn.Module()
    method.create_weights(layer, 24, [8], 24, 8, torch.bfloat16, weight_loader=lambda p, w: p.copy_(w))
    assert layer.weight.device.type == "meta"
    weight = torch.arange(8 * 24, dtype=torch.bfloat16).reshape(8, 24)
    layer.weight.weight_loader(layer.weight, weight)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], weight)
    assert calls[0][1] == bits
    assert layer.weight.dtype == layer.weight_scale.dtype == torch.uint8
    assert layer.weight.device.type == layer.weight_scale.device.type == "cpu"
    method.process_weights_after_loading(layer)
    assert len(calls) == 1


@pytest.mark.parametrize("bits", [4, 8])
def test_vae_byte_parameters_survive_outer_float_cast(monkeypatch, bits):
    width = 32 if bits == 4 else 64
    weight = torch.arange(3 * width, dtype=torch.uint8).reshape(3, width)
    scale = torch.arange(121, 127, dtype=torch.uint8).reshape(3, 1, 2)
    monkeypatch.setattr(native_mx, "quantize_mx_weight", lambda *_: (weight.clone(), scale.clone()))
    linear = nn.Linear(24, 3)
    replacement = native_mx.NativeMXLinear(linear, bits, torch.device("cpu")).to(dtype=torch.float32)
    assert replacement.weight.dtype == replacement.weight_scale.dtype == torch.uint8
    torch.testing.assert_close(replacement.weight, weight)
    torch.testing.assert_close(replacement.weight_scale, scale)
    assert replacement.bias is linear.bias


def test_native_kernel_never_silently_falls_back_to_cpu():
    pytest.importorskip("torch_npu")
    with pytest.raises(ValueError, match="same NPU"):
        native_mx.native_mx_linear(
            torch.ones(2, 64),
            torch.zeros(3, 64, dtype=torch.uint8),
            torch.zeros(3, 1, 2, dtype=torch.uint8),
            8,
            None,
        )


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int32, torch.bool, torch.complex64])
def test_native_kernel_rejects_nonfloating_activations_before_backend_import(dtype):
    with pytest.raises(ValueError, match="activations must be FP16, BF16, or FP32"):
        native_mx.native_mx_linear(
            torch.ones(2, 64, dtype=dtype),
            torch.zeros(3, 64, dtype=torch.uint8),
            torch.zeros(3, 1, 2, dtype=torch.uint8),
            8,
            None,
        )


@pytest.fixture
def fake_native_gemm(monkeypatch):
    """CPU caller-contract probe, not an emulation or validation of MX kernels."""
    monkeypatch.setitem(globals(), "_linear_with_module_dtype", _linear_with_module_dtype)
    dense_weights = {}
    calls = []

    def quantize(weight, bits, device):
        padded_k = (weight.shape[1] + 63) // 64 * 64
        width = padded_k // 2 if bits == 4 else padded_k
        packed = torch.arange(weight.shape[0] * width).to(torch.uint8).reshape(weight.shape[0], width)
        scale = torch.full((weight.shape[0], padded_k // 64, 2), 127, dtype=torch.uint8)
        dense_weights[packed.data_ptr()] = weight.detach().clone()
        return packed, scale

    def gemm(x, weight, scale, bits, bias):
        assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        assert weight.dtype == scale.dtype == torch.uint8
        calls.append((x.detach().clone(), bits, bias))
        dense = dense_weights[weight.data_ptr()]
        return nn.functional.linear(x, dense.to(x.dtype), None if bias is None else bias.to(x.dtype))

    monkeypatch.setattr(native_mx, "quantize_mx_weight", quantize)
    monkeypatch.setattr(native_mx, "native_mx_linear", gemm)
    return calls


def _linear_with_module_dtype(linear, tensor, out_dtype=None):
    """Checkpoint's original video VAE helper, including its byte-cast bug."""
    target_dtype = getattr(getattr(linear, "weight", None), "dtype", tensor.dtype)
    output = linear(tensor.to(target_dtype))
    return output if out_dtype is None else output.to(out_dtype)


class _VideoDtypeBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(5, 8)
        self.norm = nn.LayerNorm(8)
        self.proj_out = nn.Linear(8, 5)

    def forward(self, x):
        x = _linear_with_module_dtype(self.x_embedder, x, x.dtype)
        x = self.norm(x)
        return _linear_with_module_dtype(self.proj_out, x, x.dtype)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.no_grad()
def test_video_vae_preserves_floating_input_through_norm(monkeypatch, fake_native_gemm, bits, dtype):
    # Restore the shared checkpoint namespace after this test, like unloading a
    # remote model module. Installation must not alter ordinary float callers.
    monkeypatch.setitem(globals(), "_linear_with_module_dtype", _linear_with_module_dtype)
    torch.manual_seed(7)
    model = _VideoDtypeBoundary().to(dtype=dtype)
    x = torch.tensor([[[-1.75, -0.5, 0.25, 0.75, 1.5]]], dtype=dtype)
    expected = model(x)
    scheme = "W4A4_MXFP4_DUALSCALE" if bits == 4 else "W8A8_MXFP8"
    checkpoint = _checkpoint({"x_embedder.weight": scheme, "proj_out.weight": scheme})
    assert native_mx.install_native_mx_vae(model, checkpoint, torch.device("cpu")) == 2
    norm_inputs = []
    model.norm.register_forward_pre_hook(lambda _, args: norm_inputs.append(args[0].detach().clone()))
    weight = model.x_embedder.weight.clone()
    scale = model.x_embedder.weight_scale.clone()
    # Residency/outer casts must preserve the packed bytes as well as the
    # activation contract (the caller must never cast negatives to uint8).
    model.to(dtype=dtype)
    actual = model(x)
    assert len(fake_native_gemm) == 2
    torch.testing.assert_close(fake_native_gemm[0][0], x, rtol=0, atol=0)
    assert norm_inputs[0].dtype == actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.x_embedder.weight, weight, rtol=0, atol=0)
    torch.testing.assert_close(model.x_embedder.weight_scale, scale, rtol=0, atol=0)
    helper = _linear_with_module_dtype
    native_mx._install_native_mx_vae_callers(model)
    assert _linear_with_module_dtype is helper


def test_video_dtype_adapter_delegates_float_callers():
    calls = []

    def original(*args):
        calls.append(args)
        return torch.tensor([123.0])

    adapter = native_mx._with_native_mx_activation_dtype(original)
    linear = nn.Linear(2, 3)
    tensor = torch.ones(1, 2)
    assert adapter(linear, tensor, torch.float64).item() == 123.0
    assert calls == [(linear, tensor, torch.float64)]


@pytest.mark.parametrize("bits", [4, 8])
@torch.no_grad()
def test_video_dtype_adapter_casts_only_the_floating_output(fake_native_gemm, bits):
    linear = native_mx.NativeMXLinear(nn.Linear(5, 3), bits, torch.device("cpu"))
    x = torch.tensor([[-1.75, -0.5, 0.25, 0.75, 1.5]], dtype=torch.float32)
    expected = linear(x).half()
    fake_native_gemm.clear()
    adapter = native_mx._with_native_mx_activation_dtype(_linear_with_module_dtype)
    actual = adapter(linear, x, torch.float16)
    torch.testing.assert_close(fake_native_gemm[0][0], x, rtol=0, atol=0)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _AudioCausalAttention(nn.Module):
    """The checkpoint's external-bias QKV and all three output shape paths."""

    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.in_dim, self.out_dim, self.num_heads = in_dim, out_dim, num_heads
        width = max(in_dim, out_dim)
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(in_dim, width * 3, bias=False)
        self.q_bias = nn.Parameter(torch.randn(width))
        self.v_bias = nn.Parameter(torch.randn(width))
        self.register_buffer("zero_k_bias", torch.zeros(width))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        batch, sequence, _ = x.shape
        bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
        qkv = nn.functional.linear(x, self.qkv.weight, bias)
        q, k, v = qkv.reshape(batch, sequence, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(batch, sequence, -1)
        return self.proj(x)


@pytest.mark.parametrize("dims", [(8, 4, 2), (8, 2, 2), (4, 8, 2), (4, 4, 2)])
@pytest.mark.parametrize("bits", [4, 8])
@torch.no_grad()
def test_audio_vae_qkv_calls_native_module_and_adds_bias_once(fake_native_gemm, dims, bits):
    torch.manual_seed(19)
    model = _AudioCausalAttention(*dims)
    x = torch.randn(2, 5, dims[0])
    expected = model(x)
    scheme = "W4A4_MXFP4_DUALSCALE" if bits == 4 else "W8A8_MXFP8"
    assert (
        native_mx.install_native_mx_vae(
            model, _checkpoint({"qkv.weight": scheme, "proj.weight": scheme}), torch.device("cpu")
        )
        == 2
    )
    actual = model(x)
    assert len(fake_native_gemm) == 2
    torch.testing.assert_close(fake_native_gemm[0][0], x, rtol=0, atol=0)
    assert fake_native_gemm[0][2] is None
    assert actual.dtype == torch.float32
    assert actual.shape == (2, 5, dims[1])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_audio_vae_rejects_unknown_external_bias_contract(fake_native_gemm):
    model = _AudioCausalAttention(4, 8, 2)
    model.qkv.bias = nn.Parameter(torch.ones(24))
    with pytest.raises(ValueError, match="bias-free QKV"):
        native_mx.install_native_mx_vae(model, _checkpoint({"qkv.weight": "W8A8_MXFP8"}), torch.device("cpu"))

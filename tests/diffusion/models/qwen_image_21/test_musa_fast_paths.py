# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MUSA paths of the Qwen-Image-2.1 fast paths."""

from unittest.mock import Mock

import pytest
import torch
from diffusers.models.normalization import RMSNorm
from torch import nn
from torch.nn import functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import exact_bf16_rope_musa as musa_rope
from vllm_omni.diffusion.layers import vae_norm_epilogue, vae_rms_norm, vae_rms_norm_musa
from vllm_omni.diffusion.models.qwen_image_21 import autoencoder_kl_qwenimage21 as vae
from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _require_musa() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")


def _forbidden(name: str) -> Mock:
    return Mock(side_effect=AssertionError(f"{name} must not run here"))


def _freqs(sequence: int, generator: torch.Generator) -> torch.Tensor:
    angles = torch.randn(sequence, 64, generator=generator) * 3
    return torch.polar(torch.ones_like(angles), angles)


def _qkv_split(sequence: int, heads: int, generator: torch.Generator, device: str) -> tuple[torch.Tensor, ...]:
    """Q, K and V as the attention block produces them: views into one fused projection."""
    qkv = torch.randn(1, sequence, 3 * heads * 128, generator=generator).to(device=device, dtype=torch.bfloat16)
    return tuple(t.unflatten(-1, (heads, 128)) for t in qkv.split(heads * 128, dim=-1))


def _fp64_norm_rope(x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm then interleaved rotation in float64 on the CPU, with no intermediate rounding."""
    x = x.cpu().double()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.cpu().double()
    real, imag = normed.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs.real.cpu().double().unsqueeze(1)
    sin = freqs.imag.cpu().double().unsqueeze(1)
    return torch.stack((real * cos - imag * sin, imag * cos + real * sin), dim=-1).flatten(-2)


def _norm(generator: torch.Generator) -> RMSNorm:
    norm = RMSNorm(128, eps=1e-6).to("musa", torch.bfloat16)
    norm.weight.copy_((1 + 0.1 * torch.randn(128, generator=generator)).to("musa", torch.bfloat16))
    return norm


def _attention_without_init() -> transformer.QwenImage21Attention:
    # The rotation dispatch reads no module state.
    return transformer.QwenImage21Attention.__new__(transformer.QwenImage21Attention)


@pytest.mark.cpu
@torch.no_grad()
def test_cpu_tensors_keep_the_reference_rotation():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 8, 2, 128, generator=generator).to(torch.bfloat16)
    freqs = _freqs(8, generator)
    assert musa_rope.musa_rotary(x, freqs) is None
    weight = torch.ones(128, dtype=torch.bfloat16)
    assert musa_rope.musa_qk_rotary(x, x, weight, weight, freqs, 1e-6) is None
    native = transformer._apply_qwen_image21_rotary_emb_native(x, freqs)
    torch.testing.assert_close(_attention_without_init()._apply_rotary_emb(x, freqs), native, rtol=0, atol=0)
    # Inputs the MUSA kernel declines take the native rotation.
    torch.testing.assert_close(transformer._apply_qwen_image21_rotary_emb_musa(x, freqs), native, rtol=0, atol=0)


@pytest.mark.cpu
def test_musa_tensors_take_the_musa_rotation(monkeypatch):
    rotated = object()
    monkeypatch.setattr(transformer, "_apply_qwen_image21_rotary_emb_musa", lambda x, freqs: rotated)
    monkeypatch.setattr(transformer, "_apply_qwen_image21_rotary_emb_native", _forbidden("the native rotation"))
    x = Mock(spec=torch.Tensor, device=Mock(spec=torch.device, type="musa"))
    assert _attention_without_init()._apply_rotary_emb(x, None) is rotated


def _on_musa(tensor: torch.Tensor, device: Mock) -> Mock:
    """``tensor``'s metadata on a MUSA device: enough to pass every input check of the MUSA helpers."""
    fake = Mock(spec=torch.Tensor, device=device, dtype=tensor.dtype, shape=tensor.shape, ndim=tensor.ndim)
    fake.stride.side_effect = tensor.stride
    fake.numel.return_value = tensor.numel()
    fake.dim.return_value = tensor.dim()
    fake.is_contiguous.return_value = tensor.is_contiguous()
    return fake


@pytest.mark.cpu
@torch.no_grad()
def test_musa_helpers_decline_where_the_ops_are_not_registered(monkeypatch):
    device = Mock(spec=torch.device, type="musa")
    x = _on_musa(torch.empty(1, 8, 2, 128, dtype=torch.bfloat16), device)
    freqs = _on_musa(torch.empty(8, 64, dtype=torch.complex64), device)
    weight = _on_musa(torch.empty(128, dtype=torch.bfloat16), device)
    monkeypatch.setattr(torch.ops.vllm_omni, "musa_interleaved_rope", Mock(return_value="rope"), raising=False)
    monkeypatch.setattr(torch.ops.vllm_omni, "musa_qk_norm_rope", Mock(return_value="qk"), raising=False)
    # Registered, the ops take these inputs.
    monkeypatch.setattr(musa_rope, "_HAS_MUSA_OPS", True)
    assert musa_rope.musa_rotary(x, freqs) == "rope"
    assert musa_rope.musa_qk_rotary(x, x, weight, weight, freqs, 1e-6) == "qk"
    # Not registered, the helpers return None so the caller takes the native path.
    monkeypatch.setattr(musa_rope, "_HAS_MUSA_OPS", False)
    assert musa_rope.musa_rotary(x, freqs) is None
    assert musa_rope.musa_qk_rotary(x, x, weight, weight, freqs, 1e-6) is None


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_rope_kernel_matches_the_native_rotation(monkeypatch):
    _require_musa()
    generator = torch.Generator().manual_seed(1)
    query, _, _ = _qkv_split(512, 4, generator, "musa")
    freqs = _freqs(512, generator).to("musa")
    got = musa_rope.musa_rotary(query, freqs)
    assert got is not None
    expected = transformer._apply_qwen_image21_rotary_emb_native(query, freqs)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    monkeypatch.setattr(transformer, "_apply_qwen_image21_rotary_emb_native", _forbidden("the native rotation"))
    torch.testing.assert_close(_attention_without_init()._apply_rotary_emb(query, freqs), got, rtol=0, atol=0)


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_rope_runs_inside_a_compiled_region(monkeypatch):
    _require_musa()
    generator = torch.Generator().manual_seed(2)
    query, _, _ = _qkv_split(512, 4, generator, "musa")
    freqs = _freqs(512, generator).to("musa")
    eager = musa_rope.musa_rotary(query, freqs)
    compiled = torch.compile(musa_rope.musa_rotary, fullgraph=True)(query, freqs)
    assert compiled is not None
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
    # Tracing the native fallback would fail, so the compiled helper took the kernel.
    monkeypatch.setattr(transformer, "_apply_qwen_image21_rotary_emb_native", _forbidden("the native rotation"))
    helper = torch.compile(transformer._apply_qwen_image21_rotary_emb_musa, fullgraph=True)
    torch.testing.assert_close(helper(query, freqs), eager, rtol=0, atol=0)


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_fused_qk_is_as_accurate_as_the_reference_chain():
    _require_musa()
    generator = torch.Generator().manual_seed(3)
    query, key, value = _qkv_split(1024, 8, generator, "musa")
    freqs = _freqs(1024, generator).to("musa")
    norms = [_norm(generator), _norm(generator)]
    weights = (norms[0].weight, norms[1].weight)

    fused = musa_rope.musa_qk_rotary(query, key, *weights, freqs, 1e-6)
    assert fused is not None
    for got, x, norm in zip(fused, (query, key), norms):
        reference = transformer._apply_qwen_image21_rotary_emb_native(norm(x).to(value.dtype), freqs)
        # Only the variance reduction order differs from the eager chain, so only
        # elements that sit on a bf16 rounding boundary may move.
        assert (got != reference).float().mean().item() <= 1e-3
        # Against float64, the kernel is as accurate as the eager chain.
        truth = _fp64_norm_rope(x, norm.weight, freqs, 1e-6)
        got_error = (got.cpu().double() - truth).abs()
        reference_error = (reference.cpu().double() - truth).abs()
        assert got_error.mean() <= 1.01 * reference_error.mean()
        assert got_error.max() <= 1.5 * reference_error.max()


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_fused_qk_runs_inside_a_compiled_region():
    _require_musa()
    generator = torch.Generator().manual_seed(4)
    query, key, _ = _qkv_split(512, 4, generator, "musa")
    freqs = _freqs(512, generator).to("musa")
    weight_q = (1 + 0.1 * torch.randn(128, generator=generator)).to("musa", torch.bfloat16)
    weight_k = (1 + 0.1 * torch.randn(128, generator=generator)).to("musa", torch.bfloat16)

    def call(q, k):
        return musa_rope.musa_qk_rotary(q, k, weight_q, weight_k, freqs, 1e-6)

    eager = call(query, key)
    compiled = torch.compile(call, fullgraph=True)(query, key)
    assert eager is not None and compiled is not None
    for got, expected in zip(compiled, eager):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_kernels_take_the_batch_from_the_grid():
    _require_musa()
    generator = torch.Generator().manual_seed(8)
    qkv = torch.randn(2, 1037, 3 * 8 * 128, generator=generator).to("musa", torch.bfloat16)
    query, key, _ = (t.unflatten(-1, (8, 128)) for t in qkv.split(8 * 128, dim=-1))
    freqs = _freqs(1037, generator).to("musa")
    weight_q = (1 + 0.1 * torch.randn(128, generator=generator)).to("musa", torch.bfloat16)
    weight_k = (1 + 0.1 * torch.randn(128, generator=generator)).to("musa", torch.bfloat16)
    expected = transformer._apply_qwen_image21_rotary_emb_native(query, freqs)
    torch.testing.assert_close(musa_rope.musa_rotary(query, freqs), expected, rtol=0, atol=0)
    batched = musa_rope.musa_qk_rotary(query, key, weight_q, weight_k, freqs, 1e-6)
    for i in range(2):
        single = musa_rope.musa_qk_rotary(query[i : i + 1], key[i : i + 1], weight_q, weight_k, freqs, 1e-6)
        for got, expected in zip(batched, single):
            torch.testing.assert_close(got[i : i + 1], expected, rtol=0, atol=0)


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_attention_takes_the_fused_qk_path(monkeypatch):
    _require_musa()
    sequence = 17
    generator = torch.Generator().manual_seed(sequence)
    projection = torch.randn(2, sequence, 3, 32, 128, generator=generator).to("musa", torch.bfloat16)
    freqs = _freqs(sequence, generator).to("musa")
    attention = object.__new__(transformer.QwenImage21Attention)
    nn.Module.__init__(attention)
    attention.num_heads = attention.num_kv_heads = 32
    attention.head_dim = 128
    attention.norm_q = _norm(generator)
    attention.norm_k = _norm(generator)
    expected = musa_rope.musa_qk_rotary(
        projection[:, :, 0], projection[:, :, 1], attention.norm_q.weight, attention.norm_k.weight, freqs, 1e-6
    )
    attention.to_qkv = lambda _: (projection.flatten(2), None)
    observed = []

    def check_attention(q, k, v, metadata):
        torch.testing.assert_close(q, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(k, expected[1], rtol=0, atol=0)
        observed.append(True)
        return v

    attention.attn = check_attention
    attention.to_out = nn.Identity()
    # Neither the CUDA kernel entry nor the unfused norm + rotation chain may run.
    monkeypatch.setattr(transformer, "qk_rotary", _forbidden("the CUDA fused Q/K entry"))
    monkeypatch.setattr(transformer, "_apply_qwen_image21_rotary_emb_musa", _forbidden("the unfused chain"))
    result = attention(torch.empty(2, sequence, 4096, device="musa", dtype=torch.bfloat16), freqs)
    assert observed == [True]
    torch.testing.assert_close(result, projection[:, :, 2].flatten(2), rtol=0, atol=0)


class _ScaledAttention(nn.Module):
    def forward(self, values, *args, **kwargs):
        return values * 0.37


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_musa_broadcast_modulation_matches_the_per_token_selection():
    _require_musa()
    generator = torch.Generator().manual_seed(5)
    block = transformer.QwenImage21TransformerBlock.__new__(transformer.QwenImage21TransformerBlock)
    nn.Module.__init__(block)
    block.img_norm1 = nn.LayerNorm(512, elementwise_affine=False, eps=1e-6)
    block.img_norm2 = nn.LayerNorm(512, elementwise_affine=False, eps=1e-6)
    block.attn = _ScaledAttention()
    block.img_mlp = nn.SiLU()
    hidden_states = torch.randn(1, 256, 512, generator=generator).to("musa", torch.bfloat16)
    # One sample row and the trailing t = 0 row; in a decode step every query token is a target token.
    modulation = torch.randn(2, 4 * 512, generator=generator).to("musa", torch.bfloat16)
    target_token_mask = torch.ones(256, dtype=torch.bool, device="musa")
    scale1, gate1, scale2, gate2 = modulation.chunk(4, dim=-1)

    def prepare(select):
        return 1 + select(scale1), select(gate1).tanh(), 1 + select(scale2), select(gate2).tanh()

    selected = prepare(lambda value: transformer._select_modulation_rows(value, target_token_mask))
    broadcast = prepare(lambda value: value[:-1].unsqueeze(1))
    assert selected[0].shape == (1, 256, 512) and broadcast[0].shape == (1, 1, 512)
    expected = block(hidden_states, modulation, None, target_token_mask, prepared_modulation=selected)
    actual = block(hidden_states, modulation, None, target_token_mask, prepared_modulation=broadcast)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _vae_norm_reference(norm: vae.QwenImage21RMS_norm, x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1).to(x.dtype) * norm.scale * norm.gamma + norm.bias


@pytest.mark.cpu
@torch.no_grad()
def test_cpu_vae_norm_keeps_the_reference_path(monkeypatch):
    monkeypatch.setattr(vae, "rms_norm_fastpath", _forbidden("the Wan VAE norm fast path"))
    monkeypatch.setattr(vae, "rms_norm_fastpath_musa", _forbidden("the MUSA VAE norm fast path"))
    norm = vae.QwenImage21RMS_norm(16, images=False).to(torch.bfloat16)
    x = torch.randn(1, 16, 1, 8, 8, generator=torch.Generator().manual_seed(6)).to(torch.bfloat16)
    torch.testing.assert_close(norm(x), _vae_norm_reference(norm, x), rtol=0, atol=0)


@pytest.mark.cpu
@torch.no_grad()
def test_musa_vae_norm_declines_off_the_musa_platform(monkeypatch):
    device = Mock(spec=torch.device, type="musa")
    norm = vae.QwenImage21RMS_norm(8, images=False).to(torch.bfloat16)
    x = _on_musa(torch.empty(1, 8, 1, 4, 4, dtype=torch.bfloat16), device)
    launch = Mock()
    monkeypatch.setattr(vae_norm_epilogue, "rms_norm_scale", launch)
    # On MUSA, the dense input takes the epilogue.
    monkeypatch.setattr(vae_rms_norm_musa, "_HAS_MUSA_KERNEL", True)
    assert vae_rms_norm_musa.rms_norm_fastpath_musa(norm, x) is launch.return_value.view.return_value
    # Elsewhere the helper returns None so the caller takes the reference forward.
    monkeypatch.setattr(vae_rms_norm_musa, "_HAS_MUSA_KERNEL", False)
    assert vae_rms_norm_musa.rms_norm_fastpath_musa(norm, x) is None


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("channels_last", [False, True])
@torch.no_grad()
def test_musa_vae_norm_matches_the_native_norm(monkeypatch, channels_last, dtype):
    _require_musa()
    generator = torch.Generator().manual_seed(7)
    norm = vae.QwenImage21RMS_norm(96, images=False).to("musa", dtype)
    norm.gamma.copy_((1 + 0.1 * torch.randn(96, 1, 1, 1, generator=generator)).to("musa", dtype))
    x = (torch.randn(1, 96, 1, 256, 256, generator=generator) * 3).to("musa", dtype)
    if channels_last:
        x = x.squeeze(2).contiguous(memory_format=torch.channels_last).unsqueeze(2)
    launch = vae_norm_epilogue.rms_norm_scale
    calls = []

    def counted_launch(*args, **kwargs):
        calls.append(args)
        return launch(*args, **kwargs)

    monkeypatch.setattr(vae_norm_epilogue, "rms_norm_scale", counted_launch)
    # The MUSA path neither goes through the Wan fast path nor probes the reduction.
    monkeypatch.setattr(vae, "rms_norm_fastpath", _forbidden("the Wan VAE norm fast path"))
    monkeypatch.setattr(vae_rms_norm, "_fp32_norm_probe", _forbidden("the reduction probe"))
    torch.testing.assert_close(norm(x), _vae_norm_reference(norm, x), rtol=0, atol=0)
    # The epilogue kernel takes the dense layout; channels-last inputs take the reference forward.
    assert len(calls) == (0 if channels_last else 1)
    rows = x.contiguous().view(1, 96, -1)
    denom = rows.float().norm(p=2, dim=1, keepdim=True)
    assert launch(rows, denom, norm.gamma.reshape(-1), norm.scale, silu=True) is None

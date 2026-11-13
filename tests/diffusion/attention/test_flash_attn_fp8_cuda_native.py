# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real SM90 FA3 coverage for MHA, GQA and MQA FP8 scale contracts."""

import pytest
import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["H100", "H200"]}, num_cards=1)
@pytest.mark.parametrize("seq_len", [256, 19320, 75600])
@pytest.mark.parametrize("head_dim", [128, 256])
@torch.inference_mode()
def test_fp8_constant_v_native(seq_len, head_dim):
    """Match the NPU qualification test's constant-V inputs and tolerances."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3 FP8 requires SM90")

    torch.manual_seed(17)
    query = torch.randn(1, seq_len, 2, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    value = torch.tensor([0.5, -0.5], dtype=query.dtype, device=query.device)
    value = value.view(1, 1, 2, 1).expand_as(query).contiguous()
    impl = FlashAttentionImpl(num_heads=2, head_size=head_dim, softmax_scale=head_dim**-0.5)
    output = impl.forward_cuda(query, query, value, AttentionMetadata(extra={"kv_cache_dtype": "fp8"}))
    assert output.shape == query.shape and output.dtype == query.dtype
    assert torch.isfinite(output).all()
    max_abs_error = (output.float() - value.float()).abs().max().item()
    print(f"constant_v seq_len={seq_len} head_dim={head_dim} max_abs_error={max_abs_error}")
    torch.testing.assert_close(output.float(), value.float(), rtol=0.02, atol=0.02)


@hardware_test(res={"cuda": ["H100", "H200"]}, num_cards=1)
@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (4, 2), (8, 2), (4, 1)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [96, 128, 192, 256])
@torch.inference_mode()
def test_fp8_per_tensor_native(q_heads, kv_heads, causal, dtype, head_dim):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3 FP8 requires SM90")

    torch.manual_seed(17)
    batch, q_len, k_len = 2, 128, 192
    # Noncontiguous inputs and different head amplitudes exercise the tensor
    # flattening and scalar descale broadcasting for MHA/GQA/MQA.
    query = torch.randn(batch, q_heads, q_len, head_dim, device="cuda", dtype=dtype)
    query *= torch.linspace(0.25, 2.0, q_heads, device="cuda")[None, :, None, None]
    query = query.transpose(1, 2)
    key = torch.randn(batch, k_len, kv_heads, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    impl = FlashAttentionImpl(
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        head_size=head_dim,
        softmax_scale=head_dim**-0.5,
        causal=causal,
    )
    output = impl.forward_cuda(query, key, value, AttentionMetadata(extra={"kv_cache_dtype": "fp8"}))
    assert output.shape == query.shape and output.dtype == query.dtype
    assert torch.isfinite(output).all()

    # FA3 causal masking is bottom-right aligned for unequal Q/K lengths.
    mask = None
    if causal:
        q_pos = torch.arange(q_len, device="cuda")[:, None]
        k_pos = torch.arange(k_len, device="cuda")[None, :]
        mask = k_pos <= q_pos + k_len - q_len
    repeat = q_heads // kv_heads
    reference = F.scaled_dot_product_attention(
        query.float().transpose(1, 2),
        key.float().repeat_interleave(repeat, dim=2).transpose(1, 2),
        value.float().repeat_interleave(repeat, dim=2).transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2)
    # Numerical regression tolerance for this synthetic input, not an E2E quality gate.
    relative_l1 = (output.float() - reference).abs().mean(dim=(0, 1, 3)) / reference.abs().mean(dim=(0, 1, 3))
    print(
        f"dtype={dtype} heads={q_heads}/{kv_heads} head_dim={head_dim} "
        f"causal={causal} relative_l1_per_head={relative_l1.tolist()}"
    )
    assert (relative_l1 < 0.08).all(), relative_l1


@hardware_test(res={"cuda": ["H100", "H200"]}, num_cards=1)
@torch.inference_mode()
def test_fp8_fused_qkv_views_match_contiguous_inputs():
    """Row-strided QKV and broadcast scales preserve native FP8 output."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3 FP8 requires SM90")
    torch.manual_seed(17)
    batch, seq_len, heads, head_dim = 2, 256, 4, 128
    qkv = torch.randn(batch, seq_len, 3, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    query, key, value = qkv.unbind(dim=2)
    impl = FlashAttentionImpl(num_heads=heads, head_size=head_dim, softmax_scale=head_dim**-0.5)
    metadata = AttentionMetadata(extra={"kv_cache_dtype": "fp8"})
    actual = impl.forward_cuda(query, key, value, metadata)
    expected = impl.forward_cuda(query.contiguous(), key.contiguous(), value.contiguous(), metadata)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@hardware_test(res={"cuda": ["H100", "H200"]}, num_cards=1)
@pytest.mark.parametrize("head_dim", [16, 48, 96, 128, 192, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq", [1, 3, 5])
@torch.inference_mode()
def test_fp8_hadamard_normalization_and_input_ownership_native(mocker, head_dim, dtype, seq):
    """Validate the real upstream rotation on partial tiles and fused views."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3 FP8 requires SM90")

    torch.manual_seed(12)
    batch, heads = 1, 2
    projected = torch.randn(batch, seq, 3, heads, head_dim, device="cuda", dtype=dtype)
    original = projected.clone()
    query, key, value = projected.unbind(dim=2)
    rotated = []
    upstream_hadamard = ops.hadacore_transform

    def capture_rotation(x, inplace=False):
        result = upstream_hadamard(x, inplace=inplace)
        rotated.append(result.flatten()[: query.numel()].reshape(query.shape).float())
        return result

    mocker.patch.object(ops, "hadacore_transform", side_effect=capture_rotation)
    impl = FlashAttentionImpl(heads, head_dim, head_dim**-0.5)
    output = impl.forward_cuda(query, key, value, AttentionMetadata(extra={"kv_cache_dtype": "fp8"}))
    assert len(rotated) == 2
    actual_scores = torch.einsum("bqhd,bkhd->bhqk", *rotated)
    reference_scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float())
    # Normalize by input norms: a one-token QK score can be near zero even
    # when both vectors are large, making relative score error ill-conditioned.
    assert (actual_scores - reference_scores).norm() / (query.float().norm() * key.float().norm()) < 0.005
    for actual, source in zip(rotated, (query, key)):
        torch.testing.assert_close(actual.norm(dim=-1), source.float().norm(dim=-1), rtol=0.005, atol=0)
    torch.testing.assert_close(projected, original, rtol=0, atol=0)
    assert output.shape == query.shape and output.dtype == dtype
    assert torch.isfinite(output).all()


@hardware_test(res={"cuda": ["H100", "H200"]}, num_cards=1)
@pytest.mark.parametrize("kv_seq_len", [192, 2561])
@torch.inference_mode()
def test_fp8_fp16_attention_output_projection_native(mocker, kv_seq_len):
    """FP8 attention must feed an FP16 projection without a model-side cast."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3 FP8 requires SM90")

    torch.manual_seed(17)
    batch, q_len, q_heads, kv_heads, head_dim = 2, 128, 4, 2, 128
    config = OmniDiffusionConfig(
        dtype=torch.float16,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN")),
        diffusion_kv_cache_dtype="fp8",
    )
    with set_current_diffusion_config(config):
        attention = Attention(
            num_heads=q_heads,
            num_kv_heads=kv_heads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=False,
            skip_sequence_parallel=True,
        )
    quant_forward = mocker.spy(attention.attention, "forward_fa_quant_cuda")
    hidden_size = q_heads * head_dim
    projection = torch.nn.Linear(hidden_size, hidden_size, device="cuda", dtype=config.dtype)
    query = torch.randn(batch, q_len, q_heads, head_dim, device="cuda", dtype=config.dtype)
    key = torch.randn(batch, kv_seq_len, kv_heads, head_dim, device="cuda", dtype=config.dtype)
    value = torch.randn_like(key)

    attn_output = attention(query, key, value)
    # Keep the projection directly after attention so a dtype mismatch fails here.
    output = projection(attn_output.flatten(2))

    quant_forward.assert_called_once()
    assert attn_output.shape == query.shape and attn_output.dtype == query.dtype
    assert output.shape == (batch, q_len, hidden_size)
    assert output.dtype == projection.weight.dtype == torch.float16
    assert torch.isfinite(output).all()

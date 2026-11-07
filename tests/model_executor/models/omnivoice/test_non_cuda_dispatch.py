# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Non-CUDA tensors must avoid OmniVoice's CUDA Triton kernels."""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def non_cuda_kernels(monkeypatch):
    from vllm_omni.model_executor.models.omnivoice import omnivoice_generator

    def reject_cuda_kernel(*args, **kwargs):
        raise AssertionError("Non-CUDA tensor dispatched to a CUDA Triton kernel")

    monkeypatch.setattr(omnivoice_generator, "_TRITON_AVAILABLE", True)
    for name in ("triton_rms_norm", "triton_swiglu", "triton_fused_add_rms_norm"):
        monkeypatch.setattr(omnivoice_generator, name, reject_cuda_kernel, raising=False)
    return omnivoice_generator


@pytest.mark.core_model
@pytest.mark.cpu
def test_cpu_rmsnorm_with_triton_installed(non_cuda_kernels):
    torch.manual_seed(0)
    norm = non_cuda_kernels.OmniVoiceRMSNorm(16)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 16))
    x = torch.randn(2, 3, 16)
    expected = F.rms_norm(x, (16,), norm.weight, norm.eps)
    torch.testing.assert_close(norm(x), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.core_model
@pytest.mark.cpu
def test_cpu_mlp_with_triton_installed(non_cuda_kernels):
    from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

    torch.manual_seed(1)
    config = OmniVoiceConfig(llm_config={"hidden_size": 16, "intermediate_size": 32})
    mlp = non_cuda_kernels.OmniVoiceMLP(config)
    x = torch.randn(2, 3, 16)
    gate_weight, up_weight = mlp.gate_up_proj.weight.chunk(2, dim=0)
    expected = F.linear(F.silu(F.linear(x, gate_weight)) * F.linear(x, up_weight), mlp.down_proj.weight)
    torch.testing.assert_close(mlp(x), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("has_residual", [False, True])
def test_cpu_block_with_triton_installed(non_cuda_kernels, monkeypatch, has_residual):
    from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

    torch.manual_seed(2)
    # Attention backend selection is independent of the three kernel dispatches.
    monkeypatch.setattr(non_cuda_kernels, "OmniVoiceAttention", lambda *args: torch.nn.Identity())
    config = OmniVoiceConfig(llm_config={"hidden_size": 16, "intermediate_size": 32})
    block = non_cuda_kernels.OmniVoiceTransformerBlock(config, layer_idx=0)
    monkeypatch.setattr(block.self_attn, "forward", lambda hidden, *args: hidden.sin())
    x = torch.randn(3, 16)
    residual = torch.randn_like(x) if has_residual else None
    summed = x if residual is None else x + residual
    normed = F.rms_norm(summed, (16,), block.input_layernorm.weight, block.input_layernorm.eps)
    expected_residual = summed + normed.sin()
    post_norm = F.rms_norm(
        expected_residual, (16,), block.post_attention_layernorm.weight, block.post_attention_layernorm.eps
    )
    gate_weight, up_weight = block.mlp.gate_up_proj.weight.chunk(2, dim=0)
    expected = F.linear(
        F.silu(F.linear(post_norm, gate_weight)) * F.linear(post_norm, up_weight), block.mlp.down_proj.weight
    )
    actual, actual_residual = block(x, attn_metadata=None, residual=residual)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)

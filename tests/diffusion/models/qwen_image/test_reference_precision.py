# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.utils import default_unquantized_gemm

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.selector import get_attn_backend_for_capability
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer as qwen

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def single_rank(monkeypatch):
    # Real upstream linears and loaders; this test does not perform TP communication.
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(qwen.envs, "VLLM_BATCH_INVARIANT", False)


def _loaded_qkv(dtype, device, bias=True):
    generator = torch.Generator(device=device).manual_seed(2026)
    with torch.device(device):
        layer = linear.QKVParallelLinear(
            hidden_size=32,
            head_size=8,
            total_num_heads=4,
            total_num_kv_heads=2,
            params_dtype=dtype,
            bias=bias,
            disable_tp=True,
        )
    shards = []
    for name, size in zip(("q", "k", "v"), layer.output_partition_sizes):
        weight = torch.randn(size, 32, generator=generator, device=device, dtype=dtype) / 8
        offset = torch.randn(size, generator=generator, device=device, dtype=dtype) / 8 if bias else None
        layer.weight.weight_loader(layer.weight, weight, name)
        if offset is not None:
            layer.bias.weight_loader(layer.bias, offset, name)
        shards.append((weight, offset))
    # Exercise the upstream method with a noncontiguous, batched input.
    x = torch.randn(2, 3, 64, generator=generator, device=device, dtype=dtype)[..., ::2]
    return layer, x, shards


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_text_qkv_cpu_fallback_matches_real_layer(single_rank, monkeypatch, dtype):
    monkeypatch.setattr(linear, "dispatch_unquantized_gemm", lambda _backend: default_unquantized_gemm)
    layer, x, _ = _loaded_qkv(dtype, "cpu")
    original = layer.quant_method
    expected, _ = layer(x)
    layer.quant_method = qwen._ReferenceTextQKVMethod(original)

    assert layer.quant_method._gemm_impl is original._gemm_impl
    actual, _ = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_torch_sdpa_resolves_to_reference_backend_name():
    config = AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA"))
    backend = get_attn_backend_for_capability("self", attention_config=config)
    assert backend.get_name() == "SDPA"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_reference_text_qkv_cuda_preserves_provider_and_loaded_partitions(single_rank, mocker, dtype, bias):
    layer, x, shards = _loaded_qkv(dtype, "cuda", bias=bias)
    original = layer.quant_method
    packed, _ = layer(x)
    replacement = qwen._ReferenceTextQKVMethod(original)
    assert replacement._gemm_impl is original._gemm_impl
    # Observe calls to the actual configured provider; do not substitute its computation.
    calls = mocker.Mock(wraps=replacement._gemm_impl)
    replacement._gemm_impl = calls
    layer.quant_method = replacement
    actual, _ = layer(x)

    assert all(call.args[0] is layer and call.args[1] is x for call in calls.call_args_list)
    if dtype == torch.bfloat16:
        assert [call.args[2].shape[0] for call in calls.call_args_list] == layer.output_partition_sizes
        independent = torch.cat([F.linear(x, weight, offset) for weight, offset in shards], dim=-1)
        # Separate and packed BF16 GEMMs can round differently; neither is an FP32 reference.
        torch.testing.assert_close(actual, independent, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(actual, packed, rtol=1e-2, atol=1e-2)
    else:
        assert calls.call_count == 1
        torch.testing.assert_close(actual, packed, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("reference_eager", [False, True])
def test_qwen_reference_gate_uses_resolved_sdpa(single_rank, reference_eager):
    config = OmniDiffusionConfig(
        dtype=torch.bfloat16,
        enforce_eager=reference_eager,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    with set_current_diffusion_config(config):
        attention = qwen.QwenImageCrossAttention(
            dim=32,
            num_heads=4,
            head_dim=8,
            added_kv_proj_dim=32,
            reference_eager=reference_eager,
        )

    assert attention.attn.attn_backend.get_name() == "SDPA"
    assert attention.reference_precision is reference_eager
    expected_method = qwen._ReferenceTextQKVMethod if reference_eager else linear.UnquantizedLinearMethod
    assert type(attention.add_kv_proj.quant_method) is expected_method
    assert type(attention.to_qkv.quant_method) is linear.UnquantizedLinearMethod

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunYuanAttention,
    HunYuanRotary2DEmbedder,
)
from vllm_omni.diffusion.models.hunyuan_image3.layers.nvidia import autoencoder_blocks
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _check_qk_boundary(device, mode, use_norm, dtype, norm_dtype, nvidia, monkeypatch):
    head_dim = 8
    qkv = torch.arange(1, 3 * head_dim + 1, device=device, dtype=dtype).reshape(1, 1, -1)
    hidden = qkv[..., :head_dim]
    angles = torch.arange(1, 5, device=device, dtype=torch.float32).reshape(1, 4)
    rope = HunYuanRotary2DEmbedder(num_heads=1, num_kv_heads=1, head_dim=head_dim)
    norm = RMSNorm(head_dim, dtype=torch.bfloat16).to(device)
    if device == "cpu":
        rope.rope = rope.rope.forward_native
        norm._forward_method = norm.forward_native
    fused = Mock(wraps=norm._forward_fused)
    monkeypatch.setattr(norm, "_forward_fused", fused)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: nvidia)

    inputs = []
    norm.register_forward_pre_hook(lambda _, args: inputs.append(args[0]))
    attention = Mock(
        num_heads=1,
        num_kv_heads=1,
        head_dim=head_dim,
        q_size=head_dim,
        kv_size=head_dim,
        use_qk_norm=use_norm,
        image_rope2d_emb=rope,
        query_layernorm=norm,
        key_layernorm=norm,
    )
    attention.qkv_proj.return_value = qkv, None
    attention.rotary_emb.side_effect = lambda positions, q, k: (q, k)
    target = attention.image_attn if mode == "gen_image" else attention.attn
    target.return_value = hidden
    attention.o_proj.return_value = hidden, None
    HunYuanAttention.forward(
        attention,
        torch.tensor([0], device=device),
        hidden,
        mode=mode,
        custom_pos_emb=(angles.cos(), angles.sin()),
        query_lens=[1],
        first_step=True,
    )

    q, k, v = target.call_args.args
    assert v.dtype == dtype
    assert q.dtype == k.dtype == (torch.bfloat16 if mode == "gen_image" else dtype)
    assert [x.dtype for x in inputs] == ([norm_dtype, norm_dtype] if use_norm else [])
    if norm_dtype == torch.float32:
        fused.assert_not_called()
        for actual, x in zip((q, k), inputs, strict=True):
            assert (x != x.bfloat16().float()).any()
            expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.variance_epsilon)
            torch.testing.assert_close(actual, (expected * norm.weight.float()).to(v.dtype), rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize(
    ("mode", "use_norm", "dtype", "norm_dtype", "nvidia"),
    [
        ("gen_image", True, torch.bfloat16, torch.float32, True),
        ("gen_image", True, torch.bfloat16, torch.bfloat16, False),
        ("gen_image", True, torch.float16, torch.bfloat16, True),
        ("gen_image", False, torch.bfloat16, None, True),
        ("gen_text", True, torch.bfloat16, torch.bfloat16, True),
    ],
    ids=["cuda-image", "other-backend", "fp16-image", "no-qk-norm", "text"],
)
def test_cuda_attention_qk_precision(mode, use_norm, dtype, norm_dtype, nvidia, monkeypatch):
    _check_qk_boundary("cuda", mode, use_norm, dtype, norm_dtype, nvidia, monkeypatch)


@pytest.mark.cpu
def test_cpu_attention_qk_precision(monkeypatch):
    _check_qk_boundary("cpu", "gen_image", True, torch.bfloat16, torch.bfloat16, True, monkeypatch)


@pytest.mark.cpu
@pytest.mark.parametrize("benchmark", [False, True])
def test_vae_resnet_cudnn_flags(benchmark, monkeypatch):
    block = autoencoder_blocks.ResnetBlock(32, 32)
    monkeypatch.setattr(autoencoder_blocks, "fused_group_norm_silu", lambda x, *args, **kwargs: x)
    cudnn = torch.backends.cudnn

    def check_flags(module, args):
        assert cudnn.benchmark is benchmark
        assert cudnn.deterministic
        raise RuntimeError("convolution failed")

    block.conv1.register_forward_pre_hook(check_flags)
    with cudnn.flags(benchmark=benchmark, deterministic=False):
        with pytest.raises(RuntimeError, match="convolution failed"):
            block(torch.zeros(1, 32, 1, 1, 1))
        assert cudnn.benchmark is benchmark
        assert not cudnn.deterministic

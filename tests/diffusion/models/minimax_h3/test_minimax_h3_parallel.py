# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_grouped_qkv_checkpoint_reorder():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        _reorder_grouped_qkv_to_qkv,
    )

    # Two groups with rows [q, k, v] become [q0, q1, k0, k1, v0, v1].
    grouped = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    reordered = _reorder_grouped_qkv_to_qkv(
        grouped,
        num_query_groups=2,
        heads_per_group=1,
        head_dim=1,
    )

    assert reordered[:, 0].tolist() == [0, 3, 1, 4, 2, 5]


def test_transformer_declares_cache_sp_layerwise_offload_and_hsdp():
    from cache_dit import ForwardPattern

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    assert MiniMaxH3DiTModel._repeated_blocks == ["MiniMaxH3DiTBlock"]
    assert MiniMaxH3DiTModel._layerwise_offload_blocks_attrs == ["blocks"]
    assert MiniMaxH3DiTModel._cache_dit_adapter_config.block_forward_patterns["blocks"] == ForwardPattern.Pattern_3
    assert not MiniMaxH3DiTModel._cache_dit_adapter_config.has_separate_cfg
    assert set(MiniMaxH3DiTModel._sp_plan) == {"sp_prepare", "sp_gather"}

    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.token_refiner = nn.Module()
    model.token_refiner.blocks = nn.ModuleList([nn.Linear(4, 4)])
    model.final_layer = nn.Linear(4, 4)

    matched = [
        name
        for name, module in model.named_modules()
        if any(condition(name, module) for condition in MiniMaxH3DiTModel._hsdp_shard_conditions)
    ]
    assert matched == ["blocks.0", "blocks.1"]


def test_packed_attention_is_a_regional_compile_boundary():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    assert getattr(MiniMaxH3Attention._run_packed_attention, "_torchdynamo_disable", False)


def test_h3_fused_rope_matches_reference_and_preserves_unrotated_dims():
    from vllm_omni.diffusion.layers.rope import RotaryEmbedding
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    attention = object.__new__(MiniMaxH3Attention)
    nn.Module.__init__(attention)
    attention.rope = RotaryEmbedding(is_neox_style=True, half_head_dim=False)
    attention.rope._forward_method = attention.rope.forward_native

    x = torch.randn(11, 3, 128, dtype=torch.bfloat16)
    freqs_half = torch.randn(11, 48)
    freqs = torch.cat((freqs_half, freqs_half), dim=-1)
    actual = attention._apply_rope(x, freqs)

    cos = torch.cos(freqs).to(x.dtype).unsqueeze(1)
    sin = torch.sin(freqs).to(x.dtype).unsqueeze(1)
    x_rot = x[..., :96]
    x1, x2 = x_rot.chunk(2, dim=-1)
    expected_rot = x_rot * cos + torch.cat((-x2, x1), dim=-1) * sin
    expected = torch.cat((expected_rot, x[..., 96:]), dim=-1)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[..., 96:], x[..., 96:], atol=0, rtol=0)


def test_static_conditioning_matches_fallback_and_is_reused():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, values):
            self.calls += 1
            return values + 1, None

    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, values, *, cu_seqlens, max_seqlen):
            self.calls += 1
            assert cu_seqlens.tolist() == [0, 3, 3]
            assert max_seqlen == 3
            return values * 2

    class Rope(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, position_ids):
            self.calls += 1
            return position_ids.to(torch.float32).unsqueeze(-1)

    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.condition_proj = Projection()
    model.token_refiner = Refiner()
    model.rope = Rope()
    device = torch.device("cpu")
    prompt_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    img_position_ids = torch.arange(9).reshape(1, 3, 3)
    refiner_cu = torch.tensor([0, 3, 3], dtype=torch.int32)
    forward_hook_calls = 0

    def count_forward_hook(*_args):
        nonlocal forward_hook_calls
        forward_hook_calls += 1

    model.register_forward_pre_hook(count_forward_hook)

    prepared = model.prepare_static_conditioning(
        prompt_embeds=prompt_embeds,
        img_position_ids=img_position_ids,
        refiner_cu_seqlens=refiner_cu,
        refiner_max_seqlen=3,
    )
    assert forward_hook_calls == 1
    fallback = model._resolve_static_conditioning(
        static_conditioning=None,
        prompt_embeds=prompt_embeds,
        img_position_ids=img_position_ids,
        refiner_packed_seq_params={
            "cu_seqlens_q": refiner_cu,
            "max_seqlen_q": 3,
        },
        device=device,
    )

    torch.testing.assert_close(
        prepared.refined_prompt_embeds,
        fallback.refined_prompt_embeds,
    )
    torch.testing.assert_close(prepared.rope_freqs, fallback.rope_freqs)
    assert model.condition_proj.calls == 2
    assert model.token_refiner.calls == 2
    assert model.rope.calls == 2

    for _ in range(3):
        resolved = model._resolve_static_conditioning(
            static_conditioning=prepared,
            prompt_embeds=None,
            img_position_ids=None,
            refiner_packed_seq_params=None,
            device=device,
        )
        assert resolved is prepared

    assert model.condition_proj.calls == 2
    assert model.token_refiner.calls == 2
    assert model.rope.calls == 2


def test_denoise_branch_reuses_request_static_conditioning():
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MiniMaxH3DenoiseBranch,
    )
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MINIMAX_H3_STATIC_CONDITIONING_KWARG,
        MiniMaxH3StaticConditioning,
    )
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    packed = minimax_h3_packed_sequence(
        text_len=4,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        include_keyframe_cond=False,
    )
    static_conditioning = MiniMaxH3StaticConditioning(
        refined_prompt_embeds=torch.ones(4, 8),
        rope_freqs=torch.ones(64, 3),
    )
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.ones(4, 8),
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
        static_conditioning=static_conditioning,
    )

    assert branch.static_kwargs[MINIMAX_H3_STATIC_CONDITIONING_KWARG] is static_conditioning
    assert "prompt_embeds" not in branch.static_kwargs
    assert "img_position_ids" not in branch.static_kwargs
    assert "refiner_packed_seq_params" not in branch.static_kwargs

    for _ in range(3):
        kwargs = branch.forward_kwargs(
            video_rows=torch.zeros(branch.img_pos.shape[0], 96),
            audio_rows=torch.zeros(branch.audio_pos.shape[0], 32),
            t_video=0.5,
            t_audio=0.5,
            imgvid_cond_timestep=0.999,
            audio_ref_cond_timestep=1.0,
        )
        assert kwargs[MINIMAX_H3_STATIC_CONDITIONING_KWARG] is static_conditioning


@pytest.mark.parametrize(
    ("tp_size", "message"),
    [
        (3, "num_attention_heads"),
        (5, "num_attention_heads"),
    ],
)
def test_tp_rejects_non_divisible_head_counts(tp_size, message):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    with pytest.raises(ValueError, match=message):
        model._validate_tp_config(
            arch=MiniMaxH3DiTArchConfig(),
            tp_size=tp_size,
        )


def test_tp_accepts_checkpoint_supported_sizes():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    arch = MiniMaxH3DiTArchConfig()
    for tp_size in (1, 2, 4, 7):
        model._validate_tp_config(arch=arch, tp_size=tp_size)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
    assert set(MiniMaxH3DiTModel._sp_plan) == {
        "sp_prepare",
        "local_sp_prepare",
        "sp_gather",
    }
    assert set(MiniMaxH3DiTModel._sp_plan["sp_prepare"]) == {0, 1, 2}
    assert set(MiniMaxH3DiTModel._sp_plan["local_sp_prepare"]) == {2}

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


def test_denoise_branch_keeps_fast_h3_prefix_geometry_model_local():
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MiniMaxH3DenoiseBranch,
    )

    packed = {
        "seq_len": torch.tensor(9),
        "img_pos": torch.arange(4),
        "audio_pos": torch.empty(0, dtype=torch.long),
        "text_pos": torch.arange(2),
        "update_mask": torch.ones(4, dtype=torch.bool),
        "cu_seqlens": torch.tensor([0, 9, 9], dtype=torch.int32),
        "img_position_ids": torch.zeros(9, 3, dtype=torch.long),
        "video_spans": ({"start": 5, "latent_grid": (1, 2, 2), "role": "target"},),
    }
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(2, 4),
        token_tags=torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2]),
        device=torch.device("cpu"),
    )

    layout = branch.static_kwargs["video_token_layout"]
    assert not hasattr(layout, "prefix_segments")
    assert branch.static_kwargs["packed_seq_params"]["vsa_prefix_segments"] == (2, 3)


def test_h3_fused_rope_matches_reference_and_preserves_unrotated_dims():
    from vllm_omni.diffusion.layers.rope import RotaryEmbedding
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    attention = object.__new__(MiniMaxH3Attention)
    nn.Module.__init__(attention)
    attention.rot_dim = 96
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


def test_h3_rope_table_materializes_local_rows_in_fused_kernel_layout():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        _build_rope_table,
    )

    local_freqs_half = torch.randn(2, 48)
    local_freqs = torch.cat((local_freqs_half, local_freqs_half), dim=-1)

    actual = _build_rope_table(local_freqs)
    expected = torch.cat(
        (torch.cos(local_freqs_half), torch.sin(local_freqs_half)),
        dim=-1,
    ).to(torch.bfloat16)

    assert actual.shape == (2, 96)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_h3_prepared_rope_table_is_rank_local_and_validated(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    class Rope(nn.Module):
        def forward(self, position_ids):
            freqs_half = position_ids[0].float()
            return torch.cat((freqs_half, freqs_half), dim=-1)

    model = object.__new__(h3.MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.arch = h3.MiniMaxH3DiTArchConfig(rope_inv_freq_len=1)
    model.rope = Rope()
    model.local_sp_prepare = nn.Identity()
    monkeypatch.setattr(h3, "_sequence_parallel_local_span", lambda *args, **kwargs: (1, 2))

    position_ids = torch.arange(12, dtype=torch.long).reshape(1, 4, 3)
    actual = model.prepare_rope_table(position_ids, seq_len=4)
    expected = h3._build_rope_table(model.rope(position_ids[:, 1:3]))

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    model._validate_prepared_rope_table(actual, local_len=2, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="local_seq_len"):
        model._validate_prepared_rope_table(actual[:1], local_len=2, device=torch.device("cpu"))


def test_denoise_branch_prepares_rope_table_once_per_npu_run(monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.diffusion.models.minimax_h3 import denoise_loop
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MiniMaxH3DenoiseBranch,
        minimax_h3_denoise_loop,
    )

    monkeypatch.setattr(denoise_loop.current_omni_platform, "is_npu", lambda: True)

    packed = {
        "seq_len": torch.tensor(3),
        "img_pos": torch.tensor([1]),
        "audio_pos": torch.tensor([2]),
        "text_pos": torch.tensor([0]),
        "update_mask": torch.tensor([True]),
        "cu_seqlens": torch.tensor([0, 3, 3], dtype=torch.int32),
        "img_position_ids": torch.zeros(3, 3, dtype=torch.long),
        "latent_grid": torch.tensor([1, 1, 1]),
        "video_row_start": torch.tensor(1),
    }
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(1, 2),
        token_tags=torch.zeros(3, dtype=torch.long),
        device=torch.device("cpu"),
    )

    class Model:
        def __init__(self):
            self.rope_table = torch.zeros(3, 6, dtype=torch.bfloat16)
            self.prepare_calls = 0
            self.forward_calls = 0

        def prepare_rope_table(self, img_position_ids, *, seq_len):
            assert img_position_ids is branch.static_kwargs["img_position_ids"]
            assert seq_len == 3
            self.prepare_calls += 1
            return self.rope_table

        def __call__(self, **kwargs):
            assert kwargs["rope_table"] is self.rope_table
            self.forward_calls += 1
            return torch.zeros(1, 96), torch.zeros(1, 32)

    model = Model()
    minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=torch.zeros(1, 96),
        initial_audio_rows=torch.zeros(1, 32),
        keyframe_cond_rows=None,
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.5, 0.0],
        device=torch.device("cpu"),
    )

    assert model.prepare_calls == 1
    assert model.forward_calls == 2


def test_denoise_branch_keeps_rope_construction_on_the_reference_path_off_npu(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_omni.diffusion.models.minimax_h3 import denoise_loop
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MiniMaxH3DenoiseBranch,
        minimax_h3_denoise_loop,
    )

    monkeypatch.setattr(denoise_loop.current_omni_platform, "is_npu", lambda: False)
    packed = {
        "seq_len": torch.tensor(3),
        "img_pos": torch.tensor([1]),
        "audio_pos": torch.tensor([2]),
        "text_pos": torch.tensor([0]),
        "update_mask": torch.tensor([True]),
        "cu_seqlens": torch.tensor([0, 3, 3], dtype=torch.int32),
        "img_position_ids": torch.zeros(3, 3, dtype=torch.long),
        "latent_grid": torch.tensor([1, 1, 1]),
        "video_row_start": torch.tensor(1),
    }
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(1, 2),
        token_tags=torch.zeros(3, dtype=torch.long),
        device=torch.device("cpu"),
    )

    class Model:
        def __init__(self):
            self.forward_calls = 0

        def prepare_rope_table(self, *_args, **_kwargs):
            raise AssertionError("the reference path must not pre-build a rope_table")

        def __call__(self, **kwargs):
            assert "rope_table" not in kwargs
            self.forward_calls += 1
            return torch.zeros(1, 96), torch.zeros(1, 32)

    model = Model()
    minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=torch.zeros(1, 96),
        initial_audio_rows=torch.zeros(1, 32),
        keyframe_cond_rows=None,
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.5, 0.0],
        device=torch.device("cpu"),
    )

    assert model.forward_calls == 2


def test_strict_sp_local_span_uses_rank_owned_rows(monkeypatch):
    from vllm_omni.diffusion import forward_context
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_ulysses_mode", lambda **kwargs: "strict")
    monkeypatch.setattr(parallel_state, "get_sequence_parallel_world_size", lambda: 8)
    monkeypatch.setattr(parallel_state, "get_sequence_parallel_rank", lambda: 3)
    monkeypatch.setattr(parallel_state, "get_ulysses_parallel_world_size", lambda: 8)
    monkeypatch.setattr(parallel_state, "get_ring_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_allgather_parallel_world_size", lambda: 1)

    assert h3._sequence_parallel_local_span(16, hooks_applied=True) == (6, 2)


@pytest.mark.parametrize(
    ("hooks_applied", "mode", "seq_len", "ulysses", "ring", "allgather"),
    [
        (False, "strict", 16, 8, 1, 1),
        (True, "advanced_uaa", 16, 8, 1, 1),
        (True, "strict", 17, 8, 1, 1),
        (True, "strict", 16, 4, 2, 1),
        (True, "strict", 16, 1, 1, 8),
    ],
)
def test_local_span_falls_back_when_local_embedding_is_unsafe(
    monkeypatch,
    hooks_applied,
    mode,
    seq_len,
    ulysses,
    ring,
    allgather,
):
    from vllm_omni.diffusion import forward_context
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_ulysses_mode", lambda **kwargs: mode)
    monkeypatch.setattr(parallel_state, "get_sequence_parallel_world_size", lambda: 8)
    monkeypatch.setattr(parallel_state, "get_sequence_parallel_rank", lambda: 3)
    monkeypatch.setattr(
        parallel_state,
        "get_ulysses_parallel_world_size",
        lambda: ulysses,
    )
    monkeypatch.setattr(parallel_state, "get_ring_parallel_world_size", lambda: ring)
    monkeypatch.setattr(
        parallel_state,
        "get_allgather_parallel_world_size",
        lambda: allgather,
    )

    assert h3._sequence_parallel_local_span(seq_len, hooks_applied=hooks_applied) == (
        0,
        seq_len,
    )


def test_local_embedding_spans_reassemble_multimodal_rows():
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    class _IdentityProjection(nn.Module):
        def forward(self, x):
            return x, None

    class _IdentityRefiner(nn.Module):
        def forward(self, x, **kwargs):
            return x

    model = object.__new__(h3.MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.hidden_size = 2
    model.video_patch_proj = _IdentityProjection()
    model.audio_patch_proj = _IdentityProjection()
    model.condition_proj = _IdentityProjection()
    model.token_refiner = _IdentityRefiner()
    model.time_embedder = nn.Identity()

    seq_len = 16
    x = torch.arange(seq_len * 2, dtype=torch.float32).reshape(1, seq_len, 2)
    audio_x = x + 100
    text = torch.tensor(
        [[200.0, 201.0], [202.0, 203.0], [204.0, 205.0], [206.0, 207.0]],
        dtype=torch.bfloat16,
    )
    common = {
        "x": x,
        "audio_x": audio_x,
        "text_embeddings_selected": text,
        "unique_timesteps": torch.tensor([0.2, 0.3]),
        "img_pos": torch.tensor([1, 2, 6, 7, 11, 12]),
        "audio_pos": torch.tensor([3, 4, 8, 9, 13, 14]),
        "text_pos": torch.tensor([0, 5, 10, 15]),
        "refiner_cu_seqlens": torch.tensor([0, 4, 4], dtype=torch.int32),
        "refiner_max_seqlen": 4,
        "seq_len": seq_len,
        "device": torch.device("cpu"),
        "local_span": (0, seq_len),
    }

    full, full_t_emb = model._embed(**common)
    local_spans = []
    for rank in range(8):
        local_span = (rank * (seq_len // 8), seq_len // 8)
        local, local_t_emb = model._embed(
            **{
                **common,
                "local_span": local_span,
            }
        )
        assert local.shape == (2, 2)
        torch.testing.assert_close(local_t_emb, full_t_emb)
        local_spans.append(local)

    torch.testing.assert_close(torch.cat(local_spans), full, atol=0, rtol=0)


@pytest.mark.parametrize(
    "local_projection",
    [False, True],
    ids=["fallback", "strict-local"],
)
def test_sp_output_boundary_preserves_logits_and_minimizes_gather(
    monkeypatch,
    local_projection,
):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    class _StaticRope(nn.Module):
        input_rows = None

        def forward(self, position_ids):
            self.input_rows = position_ids.shape[1]
            return torch.zeros(position_ids.shape[1], 6)

    class _Prepare(nn.Module):
        def __init__(self, shard):
            super().__init__()
            self.shard = shard

        def forward(self, hidden, rope, combined):
            local_len = hidden.shape[0] // 2 if self.shard else hidden.shape[0]
            return hidden[:local_len], rope[:local_len], combined[:local_len]

    class _Gather(nn.Module):
        def __init__(self, remote_rows, replace):
            super().__init__()
            self.remote_rows = remote_rows
            self.replace = replace
            self.input_width = None

        def forward(self, tensor):
            self.input_width = tensor.shape[-1]
            if self.replace:
                return self.remote_rows
            return torch.cat((tensor, self.remote_rows), dim=0)

    class _FinalLayer(nn.Module):
        def forward(self, hidden, *, t_emb, inverse_indices):
            del t_emb
            inverse = inverse_indices.to(torch.float32).unsqueeze(-1)
            return (
                hidden[:, :2].to(torch.float32) + inverse,
                hidden[:, 2:3].to(torch.float32) - inverse,
            )

    seq_len = 8
    full_hidden = torch.arange(seq_len * 4, dtype=torch.bfloat16).reshape(seq_len, 4)
    inverse_indices = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    final_layer = _FinalLayer()
    remote_video, remote_audio = final_layer(
        full_hidden[seq_len // 2 :],
        t_emb=torch.empty(0),
        inverse_indices=inverse_indices[seq_len // 2 :],
    )
    if local_projection:
        remote_rows = torch.cat((remote_video, remote_audio), dim=-1)
    else:
        remote_rows = full_hidden
    gather = _Gather(remote_rows, replace=not local_projection)

    model = object.__new__(h3.MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.arch = h3.MiniMaxH3DiTArchConfig(
        latents_dim=2,
        audio_latents_dim=1,
        patch_size=(1, 1, 1),
    )
    rope = _StaticRope()
    model.rope = rope
    model.sp_prepare = _Prepare(shard=True)
    model.local_sp_prepare = _Prepare(shard=False)
    model.sp_gather = gather
    model.blocks = nn.ModuleList()
    model.final_layer = final_layer
    monkeypatch.setattr(
        model,
        "_embed",
        lambda **kwargs: (
            full_hidden[: seq_len // 2] if local_projection else full_hidden,
            torch.tensor([[0.5]], dtype=torch.float32),
        ),
    )

    def _local_span(*args, **kwargs):
        del args, kwargs
        return (0, seq_len // 2) if local_projection else (0, seq_len)

    monkeypatch.setattr(
        h3,
        "_sequence_parallel_local_span",
        _local_span,
    )

    img_pos = torch.tensor([0, 2, 4, 6])
    audio_pos = torch.tensor([1, 3, 5, 7])
    video, audio = model(
        x=torch.zeros(1, seq_len, 2),
        audio_x=torch.zeros(1, seq_len, 1),
        img_position_ids=torch.zeros(1, seq_len, 3, dtype=torch.long),
        unique_timesteps=torch.tensor([0.5]),
        inverse_indices=inverse_indices,
        update_mask=torch.ones(img_pos.numel()),
        token_tags=torch.zeros(seq_len, dtype=torch.long),
        prompt_embeds=torch.empty(0, 2),
        img_pos_info={"position_ids": img_pos},
        audio_pos_info={"position_ids": audio_pos},
        text_pos_info={"position_ids": torch.empty(0, dtype=torch.long)},
        img_pos_for_infer_output_info={"position_ids": img_pos},
        packed_seq_params={
            "cu_seqlens_q": torch.tensor([0, seq_len, seq_len], dtype=torch.int32),
            "max_seqlen_q": seq_len,
        },
        refiner_packed_seq_params={
            "cu_seqlens_q": torch.tensor([0, 0, 0], dtype=torch.int32),
            "max_seqlen_q": 0,
        },
    )

    expected_video, expected_audio = final_layer(
        full_hidden,
        t_emb=torch.empty(0),
        inverse_indices=inverse_indices,
    )
    torch.testing.assert_close(video, expected_video.index_select(0, img_pos))
    torch.testing.assert_close(audio, expected_audio.index_select(0, audio_pos))
    expected_gather_width = 3 if local_projection else full_hidden.shape[-1]
    expected_rope_rows = seq_len // 2 if local_projection else seq_len
    assert gather.input_width == expected_gather_width
    assert rope.input_rows == expected_rope_rows


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

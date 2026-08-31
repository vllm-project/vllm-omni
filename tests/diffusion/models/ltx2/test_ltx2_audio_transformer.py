# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the independent LTX audio-only Transformer."""

import pytest
import torch
from cache_dit import ForwardPattern
from torch import nn

from vllm_omni.diffusion.models.ltx2 import ltx2_audio_transformer
from vllm_omni.diffusion.models.ltx2.ltx2_audio_transformer import (
    LTX2AudioStaticConditioning,
    LTX2AudioTransformerBlock,
    LTX2AudioTransformerModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_ltx_audio_block_contains_only_audio_and_text_paths(monkeypatch):
    class FakeAttention(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

    class FakeFeedForward(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    monkeypatch.setattr(ltx2_audio_transformer, "LTX2Attention", FakeAttention)
    monkeypatch.setattr(ltx2_audio_transformer, "LTX2FeedForward", FakeFeedForward)

    block = LTX2AudioTransformerBlock(
        audio_dim=8,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
    )

    names = set(dict(block.named_modules()))
    assert {"audio_norm1", "audio_attn1", "audio_norm2", "audio_attn2", "audio_norm3", "audio_ff"} <= names
    assert not any(name in names for name in ("attn1", "attn2", "ff", "audio_to_video_attn", "video_to_audio_attn"))


def test_ltx_audio_transformer_parameter_tree_has_no_video_branch(monkeypatch):
    class FakeBlock(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.audio_weight = nn.Parameter(torch.ones(1))

        def forward(self, audio_hidden_states, *_args, **_kwargs):
            return audio_hidden_states

    monkeypatch.setattr(ltx2_audio_transformer, "LTX2AudioTransformerBlock", FakeBlock)
    model = LTX2AudioTransformerModel(
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        caption_channels=8,
        num_layers=2,
        use_prompt_embeddings=False,
    )

    parameter_names = set(dict(model.named_parameters()))
    assert "audio_proj_in.weight" in parameter_names
    assert "audio_proj_out.weight" in parameter_names
    assert "audio_scale_shift_table" in parameter_names
    assert not any(
        name.startswith(("proj_in", "proj_out", "caption_projection", "time_embed", "rope")) for name in parameter_names
    )
    assert not any("audio_to_video" in name or "video_to_audio" in name for name in parameter_names)


def test_ltx_audio_transformer_sp_plan_only_shards_audio_inputs_and_output():
    plan = LTX2AudioTransformerModel._build_sp_plan("interleaved")

    assert set(plan[""]) == {
        "audio_hidden_states",
        "audio_encoder_hidden_states",
        "audio_timestep",
    }
    assert set(plan) == {"", "audio_rope", "audio_proj_out"}


def test_ltx_audio_transformer_declares_single_stream_cache_dit_pattern():
    config = LTX2AudioTransformerModel._cache_dit_adapter_config

    assert config.block_forward_patterns == {"transformer_blocks": ForwardPattern.Pattern_2}
    assert config.has_separate_cfg is False
    assert config.check_forward_pattern is False


def test_ltx_audio_transformer_exposes_hsdp_blocks(monkeypatch):
    class FakeBlock(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

    monkeypatch.setattr(ltx2_audio_transformer, "LTX2AudioTransformerBlock", FakeBlock)
    model = LTX2AudioTransformerModel(
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        caption_channels=8,
        num_layers=2,
        use_prompt_embeddings=False,
    )

    matched = [
        name
        for name, module in model.named_modules()
        if any(condition(name, module) for condition in model._hsdp_shard_conditions)
    ]
    assert matched == ["transformer_blocks.0", "transformer_blocks.1"]


def test_ltx_audio_transformer_forward_has_audio_only_signature(monkeypatch):
    calls = []
    prompt_timesteps = []

    class FakeBlock(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

        def forward(self, audio_hidden_states, audio_encoder_hidden_states, **kwargs):
            calls.append((audio_hidden_states.shape, audio_encoder_hidden_states.shape, kwargs))
            return audio_hidden_states

    monkeypatch.setattr(ltx2_audio_transformer, "LTX2AudioTransformerBlock", FakeBlock)
    model = LTX2AudioTransformerModel(
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        caption_channels=8,
        num_layers=1,
        use_prompt_embeddings=False,
        audio_cross_attn_mod=True,
    )
    model.audio_prompt_adaln.register_forward_pre_hook(
        lambda _module, args: prompt_timesteps.append(args[0].detach().clone())
    )
    audio = torch.randn(2, 3, 4)
    context = torch.randn(2, 5, 8)

    output = model(
        audio_hidden_states=audio,
        audio_encoder_hidden_states=context,
        audio_timestep=torch.full((2, 3), 500.0),
        audio_sigma=torch.full((2,), 0.5),
        audio_num_frames=3,
    )

    assert output.shape == audio.shape
    assert calls[0][0] == (2, 3, 8)
    assert calls[0][1] == context.shape
    assert "audio_rotary_emb" in calls[0][2]
    assert torch.equal(prompt_timesteps[0], torch.full((2,), 500.0))


def test_ltx_audio_static_conditioning_matches_raw_path_and_is_request_scoped(monkeypatch):
    block_calls = []

    class FakeBlock(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

        def forward(self, audio_hidden_states, audio_encoder_hidden_states, **kwargs):
            block_calls.append(
                (
                    audio_encoder_hidden_states.detach().clone(),
                    tuple(value.detach().clone() for value in kwargs["audio_rotary_emb"]),
                )
            )
            return audio_hidden_states

    monkeypatch.setattr(ltx2_audio_transformer, "LTX2AudioTransformerBlock", FakeBlock)
    model = LTX2AudioTransformerModel(
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        caption_channels=8,
        num_layers=1,
    )
    projection_calls = []
    rope_calls = []
    model.audio_caption_projection.register_forward_hook(lambda *_args: projection_calls.append(True))
    model.audio_rope.register_forward_hook(lambda *_args: rope_calls.append(True))
    audio = torch.randn(2, 3, 4)
    context = torch.randn(2, 5, 8)
    coords = model.audio_rope.prepare_audio_coords(2, 3, audio.device)
    timestep = torch.full((2, 3), 500.0)

    raw_output = model(audio, context, timestep, audio_coords=coords)
    prepared = model.prepare_static_conditioning(context, coords, hidden_dtype=audio.dtype)
    calls_after_prepare = (len(projection_calls), len(rope_calls))
    prepared_output = model(
        audio,
        context,
        timestep,
        audio_static_conditioning=prepared,
    )

    assert isinstance(prepared, LTX2AudioStaticConditioning)
    assert (len(projection_calls), len(rope_calls)) == calls_after_prepare == (2, 2)
    assert not hasattr(model, "_audio_static_conditioning")
    torch.testing.assert_close(raw_output, prepared_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(block_calls[0][0], block_calls[1][0], rtol=0.0, atol=0.0)
    for raw_rotary, prepared_rotary in zip(block_calls[0][1], block_calls[1][1], strict=True):
        torch.testing.assert_close(raw_rotary, prepared_rotary, rtol=0.0, atol=0.0)

    other = model.prepare_static_conditioning(context + 1, coords, hidden_dtype=audio.dtype)
    assert other is not prepared
    assert not torch.equal(other.encoder_hidden_states, prepared.encoder_hidden_states)


def test_ltx_audio_weight_loader_ignores_video_weights_and_loads_audio_weights(monkeypatch):
    class FakeBlock(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

    monkeypatch.setattr(ltx2_audio_transformer, "LTX2AudioTransformerBlock", FakeBlock)
    model = LTX2AudioTransformerModel(
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        caption_channels=8,
        num_layers=0,
        use_prompt_embeddings=False,
    )
    replacement = torch.full_like(model.audio_proj_in.weight, 3.0)

    loaded = model.load_weights(
        [
            ("proj_in.weight", torch.zeros(4, 4)),
            ("audio_proj_in.weight", replacement),
            ("audio_to_video_attn.to_q.weight", torch.zeros(4, 4)),
        ]
    )

    assert loaded == {"audio_proj_in.weight"}
    torch.testing.assert_close(model.audio_proj_in.weight, replacement)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan2.2 S2V audio injection under sequence parallelism.

``after_transformer_block`` injects the audio residual into each rank's own shard instead of gathering the full
sequence on every rank and re-sharding. These tests check, on CPU, that the per-shard result equals the full-sequence
computation for shard boundaries that fall inside a frame, padded shards and shards holding no noisy tokens.
"""

import os

import pytest
import torch
from einops import rearrange

import vllm_omni.diffusion.distributed.parallel_state as parallel_state
from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import WanS2VTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_DIM = 32
_FRAMES = 5
_FRAME_LEN = 6  # tokens per frame: shards of 10 / 13 / 19 start mid-frame
_EXTRA = 7  # ref / motion tokens after the noisy ones


@pytest.fixture
def tp1_group(monkeypatch):
    """Real single-process TP group so the parallel linear layers construct on CPU; torch SDPA attention, which
    runs on CPU tensors whatever the host platform's default backend is."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni.diffusion.attention import layer

    monkeypatch.setattr(layer, "get_attn_backend_for_role", lambda **kwargs: (layer.SDPABackend, None))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29503")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


def _model(enable_adain: bool) -> WanS2VTransformer3DModel:
    torch.manual_seed(0)
    model = WanS2VTransformer3DModel(
        audio_dim=16,
        num_audio_token=4,
        enable_adain=enable_adain,
        audio_inject_layers=[0, 1],
        motion_token_num=16,
        dim=_DIM,
        ffn_dim=64,
        freq_dim=16,
        text_dim=16,
        text_len=8,
        num_heads=2,
        num_layers=2,
    ).eval()
    with torch.no_grad():  # random injector weights: zero-init to_out would make every residual 0
        for p in model.audio_injector.parameters():
            p.normal_(0.0, 0.2)
    model.original_seq_len = _FRAMES * _FRAME_LEN
    model.merged_audio_emb = torch.randn(1, _FRAMES, 4, _DIM)
    model.audio_emb_global = torch.randn(1, _FRAMES, 1, _DIM)
    return model


def _reference(model: WanS2VTransformer3DModel, hidden_states: torch.Tensor, attn_id: int) -> torch.Tensor:
    """The full-sequence injection the gather path computed."""
    out = hidden_states.clone()
    x = rearrange(out[:, : model.original_seq_len], "b (t n) c -> (b t) n c", t=_FRAMES)
    if model.enable_adain and model.adain_mode == "attn_norm":
        temb = rearrange(model.audio_emb_global, "b t n c -> (b t) n c")[:, 0]
        x = model.audio_injector.injector_adain_layers[attn_id](x, temb=temb)
    else:
        x = model.audio_injector.injector_pre_norm_feat[attn_id](x)
    audio = rearrange(model.merged_audio_emb, "b t n c -> (b t) n c")
    lens = torch.full((x.shape[0],), audio.shape[1], dtype=torch.long)
    residual = model.audio_injector.injector[attn_id](x=x, context=audio, context_lens=lens)
    out[:, : model.original_seq_len].add_(rearrange(residual, "(b t) n c -> b (t n) c", t=_FRAMES))
    return out


@pytest.mark.parametrize("enable_adain", [True, False])
@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 7])
def test_s2v_audio_injection_per_shard_matches_full_sequence(tp1_group, monkeypatch, enable_adain, world_size):
    model = _model(enable_adain)
    total = model.original_seq_len + _EXTRA
    chunk = -(-total // world_size)
    full = torch.randn(1, chunk * world_size, _DIM)  # padded like blocks_prepare(auto_pad=True)
    expected = _reference(model, full, attn_id=0)

    monkeypatch.setattr(model, "_is_sp_active", lambda: world_size > 1)
    shards = []
    with torch.no_grad():
        for rank in range(world_size):
            monkeypatch.setattr(parallel_state, "get_sequence_parallel_rank", lambda rank=rank: rank)
            local = full[:, rank * chunk : (rank + 1) * chunk].clone()
            shards.append(model.after_transformer_block(0, local))
    torch.testing.assert_close(torch.cat(shards, dim=1), expected, rtol=1e-5, atol=1e-5)


def test_s2v_audio_injection_skips_non_injected_blocks(tp1_group):
    model = _model(enable_adain=True)
    model.audio_injector.injected_block_id = {0: 0}
    hidden_states = torch.randn(1, model.original_seq_len + _EXTRA, _DIM)
    assert model.after_transformer_block(1, hidden_states) is hidden_states

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Sync-free correctness tests for the vendored audex causal speech decoder.

Covers the two host-sync removals:
1. RoPE cache sizing from a host-side ``pos_max`` instead of
   ``int(input_pos.max().item())`` (24 device syncs per decode call before).
2. ``AudexSpeechTokenEmbedder`` radix constants (levels/basis) hoisted to
   per-device cached tensors instead of per-call ``torch.tensor`` H2D builds.

Both changes must be bit-identical to the previous expressions.
"""

from __future__ import annotations

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.audex.speech_decoder import (
    AudexCausalSpeechDecoderConfig,
    AudexCausalSpeechDecoderModel,
)
from vllm_omni.model_executor.models.audex.speech_decoder.modeling_audex_causal_speech_decoder import (
    AudexSpeechTokenEmbedder,
    CausalCodecDecoderCache,
    CausalVocosBackbone,
    RotaryPositionalEmbeddings,
)

pytestmark = [pytest.mark.core_model]

# small geometry for fast CPU tests
ROPE_DIM = 16
BACKBONE_KW = dict(hidden_dim=64, depth=2, heads=4, pos_meb_dim=ROPE_DIM)


def _small_config() -> AudexCausalSpeechDecoderConfig:
    return AudexCausalSpeechDecoderConfig(
        hidden_dim=64,
        depth=2,
        heads=4,
        pos_meb_dim=ROPE_DIM,
        hop_length=80,
        vq_dim=64,
        lookahead_steps=2,
        codebook_levels=[3, 4, 5, 4],
        token_embed_dim=4,
    )


def _packed_codes(n_frames: int, levels: list[int], seed: int = 0) -> list[list[int]]:
    import math

    bound = math.prod(levels)
    gen = torch.Generator().manual_seed(seed)
    return [[int(c)] for c in torch.randint(0, bound, (n_frames,), generator=gen).tolist()]


# ---------------------------------------------------------------------------
# RoPE: host pos_max vs device max
# ---------------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize("start,length", [(0, 5), (10, 6), (40, 3)])
def test_rope_pos_max_matches_device_max_path(start: int, length: int) -> None:
    torch.manual_seed(0)
    rotary = RotaryPositionalEmbeddings(dim=ROPE_DIM, max_seq_len=64)
    x = torch.randn(1, length, 4, ROPE_DIM)
    input_pos = torch.arange(start, start + length).unsqueeze(0)

    with torch.inference_mode():
        via_host = rotary(x, input_pos=input_pos, pos_max=start + length - 1)
        via_device = rotary(x, input_pos=input_pos)  # falls back to .item()

    assert torch.equal(via_host, via_device)


@pytest.mark.cpu
def test_rope_pos_max_grows_cache_identically() -> None:
    torch.manual_seed(0)
    rotary = RotaryPositionalEmbeddings(dim=ROPE_DIM, max_seq_len=8)
    x = torch.randn(1, 6, 4, ROPE_DIM)
    input_pos = torch.arange(20, 26).unsqueeze(0)

    with torch.inference_mode():
        via_host = rotary(x, input_pos=input_pos, pos_max=25)
        via_device = rotary(x, input_pos=input_pos)

    assert rotary.cache.size(0) >= 26
    assert torch.equal(via_host, via_device)


@pytest.mark.cpu
def test_backbone_threads_host_pos_max() -> None:
    torch.manual_seed(0)
    backbone = CausalVocosBackbone(**BACKBONE_KW)
    rotary = backbone.transformers[0].att.rotary_embed
    original_forward = rotary.forward
    seen: list[int | None] = []

    def recording_forward(*args, **kwargs):
        seen.append(kwargs.get("pos_max"))
        return original_forward(*args, **kwargs)

    rotary.forward = recording_forward
    try:
        cache = CausalCodecDecoderCache()
        with torch.inference_mode():
            for _ in range(3):
                backbone(torch.randn(1, 5, BACKBONE_KW["hidden_dim"]), cache=cache)
    finally:
        del rotary.forward

    # twice per layer per step (q + k) across both depth-2 blocks, always the
    # known host value, never the device-max fallback (None)
    assert seen == [4, 4, 4, 4, 9, 9, 9, 9, 14, 14, 14, 14]
    assert cache.position == 15


# ---------------------------------------------------------------------------
# Embedder: hoisted radix constants
# ---------------------------------------------------------------------------


@pytest.mark.cpu
def test_embedder_cached_constants_bit_exact() -> None:
    torch.manual_seed(0)
    levels = [3, 4, 5, 4]
    embedder = AudexSpeechTokenEmbedder(output_dim=32, token_embed_dim=4, codebook_levels=levels)
    indices = torch.randint(0, 3 * 4 * 5 * 4, (1, 7, 1))

    # the pre-change expression, built from fresh tensors
    level_t = torch.tensor(levels, dtype=torch.long)
    basis = torch.cumprod(torch.cat([level_t.new_ones(1), level_t[:-1]]), dim=0)
    level_indices = (indices.long() // basis) % level_t
    codes = level_indices.to(dtype=embedder.project_out.weight.dtype)
    level_t = level_t.to(dtype=codes.dtype)
    expected = embedder.project_out(codes * (2.0 / (level_t - 1.0)) - 1.0)

    with torch.inference_mode():
        first = embedder(indices)
        second = embedder(indices)  # cache hit path

    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


# ---------------------------------------------------------------------------
# Session end-to-end sanity after both changes
# ---------------------------------------------------------------------------


@pytest.mark.cpu
def test_session_decode_deterministic_and_shaped() -> None:
    config = _small_config()
    model = AudexCausalSpeechDecoderModel(config).eval()
    frames = _packed_codes(20, config.codebook_levels)

    def run() -> torch.Tensor:
        session = model.create_session(chunk_frames=3, return_numpy=False)
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(frames), 7):
                for _sr, chunk in session.push(frames[i : i + 7]):
                    chunks.append(chunk.reshape(-1))
            for _sr, chunk in session.flush():
                chunks.append(chunk.reshape(-1))
        return torch.cat(chunks).to(torch.float32)

    wav_a, wav_b = run(), run()
    assert wav_a.numel() == 20 * config.hop_length
    assert torch.isfinite(wav_a).all()
    assert torch.equal(wav_a, wav_b)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("start,length", [(0, 5), (10, 6), (40, 3)])
def test_rope_pos_max_matches_device_max_path_cuda(start: int, length: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    rotary = RotaryPositionalEmbeddings(dim=ROPE_DIM, max_seq_len=64).to(device)
    x = torch.randn(1, length, 4, ROPE_DIM, device=device)
    input_pos = torch.arange(start, start + length, device=device).unsqueeze(0)

    with torch.inference_mode():
        via_host = rotary(x, input_pos=input_pos, pos_max=start + length - 1)
        via_device = rotary(x, input_pos=input_pos)

    assert torch.equal(via_host, via_device)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_embedder_cached_constants_bit_exact_cuda() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    levels = [3, 4, 5, 4]
    embedder = AudexSpeechTokenEmbedder(output_dim=32, token_embed_dim=4, codebook_levels=levels).to(device)
    indices = torch.randint(0, 3 * 4 * 5 * 4, (1, 7, 1), device=device)

    with torch.inference_mode():
        out = embedder(indices)
        assert out.device.type == "cuda"
        # cross-check against the same computation on freshly built constants
        level_t = torch.tensor(levels, dtype=torch.long, device=device)
        basis = torch.cumprod(torch.cat([level_t.new_ones(1), level_t[:-1]]), dim=0)
        level_indices = (indices.long() // basis) % level_t
        codes = level_indices.to(dtype=embedder.project_out.weight.dtype)
        level_t = level_t.to(dtype=codes.dtype)
        expected = embedder.project_out(codes * (2.0 / (level_t - 1.0)) - 1.0)
        assert torch.equal(out, expected)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_session_decode_real_geometry_cuda() -> None:
    torch.manual_seed(0)
    config = AudexCausalSpeechDecoderConfig()
    model = AudexCausalSpeechDecoderModel(config).to(device="cuda", dtype=torch.float32).eval()
    frames = _packed_codes(45, config.codebook_levels, seed=3)

    session = model.create_session(chunk_frames=5, return_numpy=False)
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(frames), 15):
            for _sr, chunk in session.push(frames[i : i + 15]):
                chunks.append(chunk.reshape(-1))
        for _sr, chunk in session.flush():
            chunks.append(chunk.reshape(-1))
    wav = torch.cat(chunks).to(torch.float32)

    assert wav.numel() == 45 * config.hop_length
    assert torch.isfinite(wav).all()

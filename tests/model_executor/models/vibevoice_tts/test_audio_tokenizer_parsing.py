# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for parse_batched_latent_input and apply_ctx_frames_cutting.

These functions parse the [ctx_frames, context_length, ...latents] wire
format produced by the VibeVoice stage input processor and cut leading
context samples from decoded audio arrays.
"""

import pytest
import torch

from vllm_omni.model_executor.models.vibevoice_tts.vibevoice_tts import (
    apply_ctx_frames_cutting,
    parse_batched_latent_input,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LATENT_DIM = 64


# ──────────────────────────────────────────────────────────────────
# parse_batched_latent_input tests
# ──────────────────────────────────────────────────────────────────


def _make_input(
    *requests: tuple[int, int, torch.Tensor],
) -> torch.Tensor:
    """Build a flat float tensor from (ctx_frames, context_length, latents)."""
    parts = []
    for ctx_frames, context_length, latents in requests:
        parts.append(torch.tensor([float(ctx_frames), float(context_length)]))
        parts.append(latents.flatten().float())
    return torch.cat(parts)


def test_parse_single_request():
    """Single request is parsed correctly."""
    ctx_frames = 3
    context_length = 5
    total = ctx_frames + context_length
    latents = torch.randn(total, LATENT_DIM)
    flat = _make_input((ctx_frames, context_length, latents))

    result, ctx_list = parse_batched_latent_input(flat, LATENT_DIM)

    assert len(result) == 1
    assert len(ctx_list) == 1
    assert ctx_list[0] == ctx_frames
    assert result[0].shape == (total, LATENT_DIM)
    torch.testing.assert_close(result[0], latents)


def test_parse_multiple_requests():
    """Multiple requests in a batch are parsed correctly."""
    lat1 = torch.ones(6, LATENT_DIM)
    lat2 = torch.full((3, LATENT_DIM), 7.0)
    flat = _make_input((2, 4, lat1), (0, 3, lat2))

    result, ctx_list = parse_batched_latent_input(flat, LATENT_DIM)

    assert len(result) == 2
    assert ctx_list == [2, 0]
    assert result[0].shape == (6, LATENT_DIM)
    assert result[1].shape == (3, LATENT_DIM)
    torch.testing.assert_close(result[0], lat1)
    torch.testing.assert_close(result[1], lat2)


def test_parse_zero_ctx_frames():
    """ctx_frames=0 means no context."""
    n = 10
    latents = torch.randn(n, LATENT_DIM)
    flat = _make_input((0, n, latents))

    result, ctx_list = parse_batched_latent_input(flat, LATENT_DIM)

    assert ctx_list[0] == 0
    assert result[0].shape == (n, LATENT_DIM)


def test_parse_preserves_values():
    """Parsed latents match the original values exactly."""
    latents = torch.arange(4 * LATENT_DIM, dtype=torch.float32).view(4, LATENT_DIM)
    flat = _make_input((1, 3, latents))

    result, _ = parse_batched_latent_input(flat, LATENT_DIM)

    torch.testing.assert_close(result[0], latents)


def test_parse_different_latent_dim():
    """Works with a non-default latent dimension."""
    dim = 16
    latents = torch.randn(5, dim)
    flat = _make_input((2, 3, latents))

    result, ctx_list = parse_batched_latent_input(flat, dim)

    assert result[0].shape == (5, dim)
    assert ctx_list[0] == 2


# ──────────────────────────────────────────────────────────────────
# apply_ctx_frames_cutting tests
# ──────────────────────────────────────────────────────────────────


def test_cut_removes_leading_samples():
    """Context frames are removed from the front of the audio array."""
    downsample_factor = 3200  # VibeVoice: 24000 / 7.5
    ctx_frames = 5
    total_samples = 32000  # 10 frames * 3200
    audio = torch.arange(total_samples, dtype=torch.float32)

    result = apply_ctx_frames_cutting([audio], [ctx_frames], downsample_factor)

    expected_cut = ctx_frames * downsample_factor
    assert len(result) == 1
    assert result[0].shape[0] == total_samples - expected_cut
    torch.testing.assert_close(result[0], audio[expected_cut:])


def test_cut_zero_ctx_frames_unchanged():
    """ctx_frames=0 leaves the audio unchanged."""
    downsample_factor = 3200
    audio = torch.randn(32000)

    result = apply_ctx_frames_cutting([audio], [0], downsample_factor)

    assert result[0].shape == audio.shape
    torch.testing.assert_close(result[0], audio)


def test_cut_multiple_requests():
    """Each request in the batch gets its own ctx_frames cut."""
    downsample_factor = 100
    audio1 = torch.arange(1000, dtype=torch.float32)
    audio2 = torch.arange(500, dtype=torch.float32)

    result = apply_ctx_frames_cutting(
        [audio1, audio2], [3, 0], downsample_factor,
    )

    assert len(result) == 2
    assert result[0].shape[0] == 1000 - 300
    assert result[1].shape[0] == 500
    torch.testing.assert_close(result[0], audio1[300:])
    torch.testing.assert_close(result[1], audio2)


def test_cut_all_frames_returns_empty():
    """Cutting all frames returns an empty tensor."""
    downsample_factor = 100
    audio = torch.randn(500)  # 5 frames

    result = apply_ctx_frames_cutting([audio], [5], downsample_factor)

    assert result[0].numel() == 0

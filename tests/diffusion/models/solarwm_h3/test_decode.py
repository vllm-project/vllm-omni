# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.minimax_h3.temporal_chunks import decode_temporal_chunks
from vllm_omni.diffusion.models.solarwm_h3.decode import SolarH3DecodeState, SolarH3StreamingDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _TemporalVAE:
    """Input-observable clip decoder: temporal convolution needs both neighbors.

    A clip-relative offset also makes moving a temporal boundary observable;
    a constant-output fake would miss wrong overlap, latent slicing, and trim.
    """

    use_3d_conv = True
    tokens_chunk_size = 5
    token_drop = 3
    token_overlap = 2
    vae_ratio_t = 4
    vae_ratio = 2
    frame_pre_padding = 3
    frame_overlap = 5
    clip_length = 17
    isolated_first_frame = isolated_last_frame = False

    def __init__(self):
        self.clips = []
        self.processor = SimpleNamespace(revert_tensor=lambda pixels: (pixels + 2).clamp(0, 100))

    def _adaptive_decode(self, clip):
        self.clips.append(clip.clone())
        kernel = clip.new_tensor([0.2, 0.5, 0.3]).view(1, 1, 3, 1, 1)
        pixels = F.conv3d(F.pad(clip, (0, 0, 0, 0, 1, 1), mode="replicate"), kernel)
        pixels = pixels.repeat_interleave(4, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        return pixels.repeat(1, 3, 1, 1, 1) + torch.arange(pixels.shape[2]).view(1, 1, -1, 1, 1) / 100

    @staticmethod
    def blend(overlap, part, frame_overlap, dim):
        assert dim == -3
        result = part.clone()
        weights = torch.arange(frame_overlap, dtype=part.dtype).view(1, 1, -1, 1, 1) / frame_overlap
        result[:, :, :frame_overlap] = overlap * (1 - weights) + part[:, :, :frame_overlap] * weights
        return result

    def _decode_temporal_output_frame_plan(self, z, z_head, z_tail, num_chunks, pad_tokens):
        assert z_head is z_tail is None
        total = num_chunks * 17 + 5
        original = z.shape[2] - pad_tokens
        trim = sum(1 if (original + k) % 5 == 0 else 4 for k in range(pad_tokens))
        return total, trim, total - trim


class _Host:
    config_dict = {"latent_channels": 1}

    def __init__(self):
        self.model = _TemporalVAE()
        self.tile_contexts = 0

    def _decode_tiling_context(self, clip):
        assert clip.dtype == torch.float32
        self.tile_contexts += 1
        return nullcontext()

    @staticmethod
    def _denormalize_latent(latent):
        return latent * 0.7 + 0.13

    @staticmethod
    def _normalize_decoded_frames(decoded):
        return decoded.float()


def _latents(count):
    return torch.linspace(-1, 1, count * 4).reshape(1, 1, count, 2, 2)


def _reference(host, latents):
    raw = decode_temporal_chunks(host.model, host._denormalize_latent(latents.float()), None)
    return host.model.processor.revert_tensor(raw).float()


@pytest.mark.parametrize("length", range(3, 33))
@pytest.mark.parametrize("push_size", [1, 5, 11])
def test_incremental_decode_matches_complete_native_temporal_loop(length, push_size):
    host = _Host()
    decoder = SolarH3StreamingDecoder(host, device="cpu")
    state = SolarH3DecodeState("session")
    latents = _latents(length)
    chunks = []
    for part in latents.split(push_size, dim=2):
        decoded = decoder.push(state, part)
        if decoded is not None:
            chunks.append(decoded)
        assert state.pending.shape[2] <= 6
        assert state.nbytes() <= decoder.state_bytes(4, 4)
    decoded = decoder.flush(state)
    if decoded is not None:
        chunks.append(decoded)
    actual = torch.cat(chunks, dim=2)
    torch.testing.assert_close(actual, _reference(_Host(), latents), rtol=0, atol=0)
    assert state.closed and not state.failed and state.nbytes() == 0
    assert host.tile_contexts == state.decoded_clips


def test_five_latent_blocks_delay_pixels_and_use_true_future_lookahead():
    host = _Host()
    decoder = SolarH3StreamingDecoder(host, device="cpu")
    state = SolarH3DecodeState("session")
    latents = _latents(10)
    assert decoder.push(state, latents[:, :, :5]) is None
    first = decoder.push(state, latents[:, :, 5:])
    assert first.shape[2] == 17
    torch.testing.assert_close(host.model.clips[0], host._denormalize_latent(latents[:, :, :7]))
    last = decoder.flush(state)
    assert last.shape[2] == 17
    torch.testing.assert_close(torch.cat((first, last), dim=2), _reference(_Host(), latents), rtol=0, atol=0)


def test_requested_horizon_is_trimmed_across_push_and_flush():
    host = _Host()
    decoder = SolarH3StreamingDecoder(host, device="cpu")
    state = SolarH3DecodeState("session", max_frames=23)
    latents = _latents(15)
    first = decoder.push(state, latents)
    assert first.shape[2] == 23
    assert decoder.flush(state) is None
    torch.testing.assert_close(first, _reference(_Host(), latents)[:, :, :23], rtol=0, atol=0)
    assert state.emitted_frames == 23


def test_session_interleaving_reset_and_owned_storage():
    host = _Host()
    decoder = SolarH3StreamingDecoder(host, device="cpu")
    a, b = SolarH3DecodeState("a"), SolarH3DecodeState("b")
    source = _latents(10)
    decoder.push(a, source[:, :, :5])
    assert a.pending.untyped_storage().nbytes() == a.pending.numel() * a.pending.element_size()
    decoder.push(b, source[:, :, :5] + 10)
    a_first = decoder.push(a, source[:, :, 5:])
    b_first = decoder.push(b, source[:, :, 5:] + 10)
    assert not torch.equal(a_first, b_first)
    assert a.overlap.untyped_storage().nbytes() == a.overlap.numel() * a.overlap.element_size()
    a.reset()
    assert a.nbytes() == 0 and a.latent_frames == a.emitted_frames == 0
    assert not a.closed
    b.close()
    with pytest.raises(RuntimeError, match="closed or failed"):
        decoder.push(b, source)


def test_decode_failure_drops_history_and_requires_reset():
    host = _Host()
    decoder = SolarH3StreamingDecoder(host, device="cpu")
    state = SolarH3DecodeState("session")
    decoder.push(state, _latents(5))

    def fail(_clip):
        raise RuntimeError("native decode failed")

    decoder._native_decode = fail
    with pytest.raises(RuntimeError, match="native decode failed"):
        decoder.push(state, _latents(5))
    assert state.failed and state.closed and state.nbytes() == 0
    with pytest.raises(RuntimeError, match="reset"):
        decoder.flush(state)
    state.reset()
    assert not state.failed and not state.closed


@pytest.mark.parametrize("length", [0, 1, 2])
def test_empty_or_too_short_final_decode_fails_and_releases_state(length):
    decoder = SolarH3StreamingDecoder(_Host(), device="cpu")
    state = SolarH3DecodeState("session")
    if length:
        decoder.push(state, _latents(length))
    with pytest.raises(ValueError, match="empty session|complete temporal clip"):
        decoder.flush(state)
    assert state.failed and state.nbytes() == 0


def test_geometry_change_fails_before_reusing_another_shape():
    decoder = SolarH3StreamingDecoder(_Host(), device="cpu")
    state = SolarH3DecodeState("session")
    decoder.push(state, _latents(5))
    with pytest.raises(ValueError, match="geometry changed"):
        decoder.push(state, torch.zeros(1, 1, 5, 3, 2))
    assert state.nbytes() == 0


def test_reduced_real_diffusers_vae_matches_whole_decode():
    from diffusers import AutoencoderKLMiniMaxH3

    from vllm_omni.diffusion.models.solarwm_h3.vae import SolarH3VideoVAE

    torch.manual_seed(3)
    model = AutoencoderKLMiniMaxH3(
        latent_channels=24,
        block_out_channels=(8, 8),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(2, 2),
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=16,
        latents_mean=tuple(index / 100 for index in range(24)),
        latents_std=tuple(0.7 + index / 100 for index in range(24)),
    ).eval()
    vae = SolarH3VideoVAE(model)
    decoder = SolarH3StreamingDecoder(vae, device="cpu")
    for length in (7, 8, 10):
        latent = torch.randn(1, 24, length, 2, 2)
        state = SolarH3DecodeState("real-vae")
        chunks = []
        for part in latent.split(5, dim=2):
            frames = decoder.push(state, part)
            if frames is not None:
                chunks.append(frames)
        frames = decoder.flush(state)
        if frames is not None:
            chunks.append(frames)
        torch.testing.assert_close(torch.cat(chunks, dim=2), vae.decode_latent(latent), rtol=0, atol=0)

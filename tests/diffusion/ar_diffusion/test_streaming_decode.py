# SPDX-License-Identifier: Apache-2.0

"""Streaming VAE decode against the real Wan autoencoder, on CPU.

The autoencoder is built from a config rather than a checkpoint -- weights are
random, only shapes and the temporal cache protocol matter -- so these run
without a device or a download.

The load-bearing test is equivalence: decoding a session chunk by chunk must
produce exactly what decoding the whole clip produces. Everything else about
streaming is worthless if that does not hold.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.streaming_decode import (
    StreamingDecodeState,
    SupportsStreamingDecode,
    WanStreamingDecoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Small enough to decode on CPU in a test, same architecture as the shipped
# checkpoint: base_dim 96 / dim_mult [1,2,4,4] would be far too slow, so the
# widths are scaled down while the causal structure is unchanged.
VAE_CONFIG = {
    "base_dim": 8,
    "z_dim": 4,
    "dim_mult": [1, 2],
    "num_res_blocks": 1,
    "temperal_downsample": [True],
    "attn_scales": [],
    "dropout": 0.0,
}
LATENT_H = LATENT_W = 4


@pytest.fixture(scope="module")
def vae():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    model = diffusers.AutoencoderKLWan(**VAE_CONFIG).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def decoder(vae):
    return WanStreamingDecoder(vae)


def _latent(num_frames: int, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, VAE_CONFIG["z_dim"], num_frames, LATENT_H, LATENT_W, generator=generator)


@torch.no_grad()
def _decode_whole(vae, latent: torch.Tensor) -> torch.Tensor:
    """The non-streaming reference: one call, cache cleared at both ends."""
    return vae._decode(latent, return_dict=False)[0]


# --------------------------------------------------------------------------
# Equivalence: the property that makes streaming worth doing at all
# --------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("chunk_sizes", [(3, 3, 3), (1, 1, 1, 1, 1), (2, 4, 3), (9,)])
def test_streaming_chunks_reproduce_the_whole_clip_decode(vae, decoder, chunk_sizes) -> None:
    """Chunked decode must equal one-shot decode, for any chunking."""
    total = sum(chunk_sizes)
    latent = _latent(total)
    reference = _decode_whole(vae, latent)

    state = decoder.new_decode_state("s")
    pieces = []
    offset = 0
    for size in chunk_sizes:
        pieces.append(decoder.decode_chunk(latent[:, :, offset : offset + size], state))
        offset += size
    streamed = torch.cat(pieces, dim=2)

    assert streamed.shape == reference.shape
    torch.testing.assert_close(streamed, reference, rtol=0, atol=0)


@torch.no_grad()
def test_a_restarted_stream_does_not_match_a_continued_one(vae, decoder) -> None:
    """Negative control: the equivalence test above must be discriminating.

    Clearing state between chunks -- what the non-streaming path does on every
    call -- has to produce a different result, otherwise the temporal cache
    carries nothing and the assertion above is vacuous.
    """
    latent = _latent(6)
    reference = _decode_whole(vae, latent)

    restarted = torch.cat(
        [
            decoder.decode_chunk(latent[:, :, :3], decoder.new_decode_state("a")),
            decoder.decode_chunk(latent[:, :, 3:], decoder.new_decode_state("b")),
        ],
        dim=2,
    )
    assert restarted.shape != reference.shape or not torch.equal(restarted, reference)


# --------------------------------------------------------------------------
# Cross-session isolation: the failure that timing metrics cannot see
# --------------------------------------------------------------------------


@torch.no_grad()
def test_interleaved_sessions_keep_independent_temporal_context(vae, decoder) -> None:
    """Two sessions ticking alternately must each match their solo decode.

    A session that reads another's temporal cache produces a perfectly
    plausible but wrong video, and every latency metric stays green, so this
    has to be asserted against a recorded solo run rather than inferred.
    """
    latent_a = _latent(6, seed=1)
    latent_b = _latent(6, seed=2)
    solo_a = _decode_whole(vae, latent_a)
    solo_b = _decode_whole(vae, latent_b)
    assert not torch.equal(solo_a, solo_b), "negative control: the two sessions must differ"

    state_a = decoder.new_decode_state("a")
    state_b = decoder.new_decode_state("b")
    out_a, out_b = [], []
    for start in (0, 3):
        window = slice(start, start + 3)
        out_a.append(decoder.decode_chunk(latent_a[:, :, window], state_a))
        out_b.append(decoder.decode_chunk(latent_b[:, :, window], state_b))

    torch.testing.assert_close(torch.cat(out_a, dim=2), solo_a, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(out_b, dim=2), solo_b, rtol=0, atol=0)


@torch.no_grad()
def test_the_module_cache_is_untouched_so_sessions_can_share_one_decoder(vae, decoder) -> None:
    vae.clear_cache()
    before = [entry for entry in vae._feat_map]
    decoder.decode_chunk(_latent(3), decoder.new_decode_state("s"))
    assert vae._feat_map == before, "streaming decode must not write the module-owned cache"


# --------------------------------------------------------------------------
# Causal geometry and boundedness
# --------------------------------------------------------------------------


@torch.no_grad()
def test_opening_chunk_is_shorter_and_later_chunks_are_full_length(vae, decoder) -> None:
    """(n - 1) * factor + 1 for the opening chunk, n * factor after it."""
    factor = 2 ** sum(VAE_CONFIG["temperal_downsample"])
    state = decoder.new_decode_state("s")
    first = decoder.decode_chunk(_latent(3), state)
    second = decoder.decode_chunk(_latent(3, seed=5), state)

    assert first.shape[2] == (3 - 1) * factor + 1
    assert second.shape[2] == 3 * factor
    assert state.chunks_decoded == 2
    assert state.frames_decoded == 6


@torch.no_grad()
def test_resident_state_does_not_grow_with_session_length(vae, decoder) -> None:
    """Each causal convolution retains at most CACHE_T temporal slices."""
    state = decoder.new_decode_state("s")
    decoder.decode_chunk(_latent(3), state)
    baseline = state.nbytes()
    assert baseline > 0

    for step in range(8):
        decoder.decode_chunk(_latent(3, seed=step + 10), state)
        assert state.nbytes() == baseline

    assert sum(state.nbytes_by_device().values()) == baseline


@torch.no_grad()
def test_release_returns_the_session_to_its_pre_stream_state(vae, decoder) -> None:
    state = decoder.new_decode_state("s")
    decoder.decode_chunk(_latent(3), state)
    assert state.started and state.nbytes() > 0

    decoder.release(state)
    assert state.nbytes() == 0
    assert not state.started
    assert state.chunks_decoded == 0

    # A released session restarts rather than resumes, so its first chunk is
    # short again -- the same semantics reset() has for AR KV.
    factor = 2 ** sum(VAE_CONFIG["temperal_downsample"])
    assert decoder.decode_chunk(_latent(3), state).shape[2] == (3 - 1) * factor + 1


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_wan_decoder_satisfies_the_protocol(decoder) -> None:
    assert isinstance(decoder, SupportsStreamingDecode)


def test_state_from_another_decoder_is_rejected(decoder) -> None:
    foreign = StreamingDecodeState(session_id="s", feat_map=[None])
    with pytest.raises(ValueError, match="does not belong to this decoder"):
        decoder.decode_chunk(_latent(1), foreign)


def test_a_non_wan_module_is_rejected() -> None:
    with pytest.raises(TypeError, match="Wan-family autoencoder"):
        WanStreamingDecoder(object())


def test_empty_and_malformed_latents_are_rejected(decoder) -> None:
    state = decoder.new_decode_state("s")
    with pytest.raises(ValueError, match=r"\[B, C, T, H, W\]"):
        decoder.decode_chunk(torch.zeros(1, 4, 4, 4), state)
    with pytest.raises(ValueError, match="at least one frame"):
        decoder.decode_chunk(torch.zeros(1, 4, 0, LATENT_H, LATENT_W), state)


def test_session_id_must_be_meaningful(decoder) -> None:
    with pytest.raises(ValueError, match="session_id"):
        decoder.new_decode_state("  ")


def test_declared_state_bytes_scales_with_area_and_dtype(decoder) -> None:
    # Measured for the shipped checkpoint: 37832 KiB of fp32 cache at 64x64.
    decoder._bytes_per_pixel_fp32 = 37832 * 1024 / (64 * 64)
    fp32 = decoder.declared_state_bytes(height=480, width=832, dtype=torch.float32)
    bf16 = decoder.declared_state_bytes(height=480, width=832, dtype=torch.bfloat16)
    half_area = decoder.declared_state_bytes(height=240, width=832, dtype=torch.float32)
    assert bf16 == fp32 // 2
    assert half_area == fp32 // 2
    assert fp32 / 2**20 == pytest.approx(3602.0, abs=5.0)

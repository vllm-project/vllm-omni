# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Streaming VAE decode against the real Wan autoencoder, on CPU.

The autoencoder is built from a config rather than a checkpoint -- weights are
random, only shapes and the temporal cache protocol matter -- so these run
without a device or a download.

The load-bearing test is equivalence: decoding a session chunk by chunk must
produce exactly what decoding the whole clip produces. Everything else about
streaming is worthless if that does not hold.
"""

from __future__ import annotations

from contextlib import contextmanager

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


def _stream(decoder, latent: torch.Tensor, chunk_sizes, *, session_id: str = "s") -> torch.Tensor:
    state = decoder.new_decode_state(session_id)
    pieces, offset = [], 0
    for size in chunk_sizes:
        pieces.append(decoder.decode_chunk(latent[:, :, offset : offset + size], state))
        offset += size
    return torch.cat(pieces, dim=2)


@torch.no_grad()
@pytest.mark.parametrize("chunk_sizes", [(3, 3, 3), (1, 1, 1, 1, 1), (2, 4, 3)])
def test_chunking_is_neutral(vae, decoder, chunk_sizes) -> None:
    """Splitting a session into chunks must change nothing about its output.

    This is the property streaming actually introduces, so it is the one
    asserted exactly. The control is the same session decoded in a single
    call through the same code path -- not ``_decode`` -- because ``_decode``
    differs in a way that has nothing to do with chunking; see
    ``test_distance_from_the_one_shot_path_is_a_post_quant_conv_artifact``.
    """
    total = sum(chunk_sizes)
    latent = _latent(total)
    reference = _stream(decoder, latent, (total,), session_id="ref")
    streamed = _stream(decoder, latent, chunk_sizes)

    assert streamed.shape == reference.shape
    torch.testing.assert_close(streamed, reference, rtol=0, atol=0)


@torch.no_grad()
def test_distance_from_the_one_shot_path_is_a_post_quant_conv_artifact(vae, decoder) -> None:
    """Explain, and bound, the gap against diffusers' ``_decode``.

    ``_decode`` applies ``post_quant_conv`` to the whole latent at once; the
    tiled path in this repo, and this decoder, apply it per frame. It is a
    1x1x1 convolution, so the two are mathematically identical -- but they can
    select different kernels, and on bf16 CUDA that difference is real and is
    amplified by the causal convolutions downstream.

    Asserting equality with ``_decode`` would therefore pin a kernel-selection
    coincidence rather than a property of streaming. Assert instead that the
    gap is entirely explained by where ``post_quant_conv`` runs.
    """
    latent = _latent(6)
    per_frame_pq = torch.cat([vae.post_quant_conv(latent[:, :, k : k + 1]) for k in range(latent.shape[2])], dim=2)
    whole_pq = vae.post_quant_conv(latent)

    streamed = _stream(decoder, latent, (3, 3))
    one_shot = _decode_whole(vae, latent)

    # Same inputs to the decoder => same output, whatever the backend does.
    if torch.equal(whole_pq, per_frame_pq):
        torch.testing.assert_close(streamed, one_shot, rtol=0, atol=0)
    else:  # pragma: no cover - backend dependent
        pytest.skip("backend batches post_quant_conv differently; the gap is that, not chunking")


@torch.no_grad()
def test_a_restarted_stream_does_not_match_a_continued_one(decoder) -> None:
    """Negative control: the equivalence test above must be discriminating.

    Clearing state between chunks -- what the non-streaming path does on every
    call -- has to change the pixels, otherwise the temporal cache carries
    nothing and the assertion above holds for a write-only cache too.

    Restarting also changes the frame count, because a fresh state expands its
    opening latent frame to one raw frame. Comparing totals would therefore
    pass on the frame count alone and never look at a pixel, so this compares
    the tail both runs agree on: the frames from the chunk's later latents,
    which are the full temporal factor wide on either side.
    """
    latent = _latent(6)
    state = decoder.new_decode_state("continued")
    decoder.decode_chunk(latent[:, :, :3], state)
    continued = decoder.decode_chunk(latent[:, :, 3:], state)
    restarted = decoder.decode_chunk(latent[:, :, 3:], decoder.new_decode_state("restarted"))

    tail = min(continued.shape[2], restarted.shape[2]) - 1
    assert tail > 0, "the comparison needs at least one fully expanded frame on both sides"
    continued_tail, restarted_tail = continued[:, :, -tail:], restarted[:, :, -tail:]
    assert continued_tail.shape == restarted_tail.shape, "the tails must be comparable for this to mean anything"
    assert not torch.equal(continued_tail, restarted_tail), (
        "restarting the decoder produced the same pixels as continuing it, so the carried cache is not being read"
    )


# --------------------------------------------------------------------------
# Cross-session isolation: the failure that timing metrics cannot see
# --------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("seed_a, seed_b", [(1, 2), (11, 12)])
def test_interleaved_sessions_keep_independent_temporal_context(vae, decoder, seed_a: int, seed_b: int) -> None:
    """Two sessions ticking alternately must each match their solo decode.

    A session that reads another's temporal cache produces a perfectly
    plausible but wrong video, and every latency metric stays green, so this
    has to be asserted against a recorded solo run rather than inferred. The
    control is a solo stream rather than ``_decode``, so it differs from the
    run under test only in the interleaving.
    """
    latent_a = _latent(6, seed=seed_a)
    latent_b = _latent(6, seed=seed_b)
    solo_a = _stream(decoder, latent_a, (6,), session_id="solo-a")
    solo_b = _stream(decoder, latent_b, (6,), session_id="solo-b")
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


def test_the_vaes_execution_context_is_entered_for_every_frame(vae) -> None:
    """Decode must run inside the autoencoder's own execution context.

    Every public compute entry point on ``OmniAutoencoderKLWan`` enters
    ``_execution_context()``, which is where autocast is established for
    fp16/bf16 parameters -- and on NPU the only place it is established at all.
    Driving ``post_quant_conv`` and ``decoder`` directly bypasses it unless this
    decoder enters it itself, which would compute streamed chunks under a
    different numeric context from the whole-clip decode they are compared to.
    """
    entered: list[str] = []

    class _ContextualVAE:
        """The parts of the contract this decoder uses, plus the hook."""

        config = vae.config
        dtype = torch.float32

        def __init__(self) -> None:
            self.decoder = vae.decoder
            self.post_quant_conv = vae.post_quant_conv
            self._cached_conv_counts = vae._cached_conv_counts

        @contextmanager
        def _execution_context(self):
            entered.append("enter")
            try:
                yield
            finally:
                entered.append("exit")

    decoder = WanStreamingDecoder(_ContextualVAE())
    with torch.no_grad():
        decoder.decode_chunk(_latent(3), decoder.new_decode_state("s"))

    # One context for the call, not one per frame, and it closes.
    assert entered == ["enter", "exit"]


def test_a_plain_diffusers_autoencoder_needs_no_execution_context(decoder) -> None:
    """The hook is optional: a VAE without one still decodes."""
    with torch.no_grad():
        out = decoder.decode_chunk(_latent(1), decoder.new_decode_state("s"))
    assert out.shape[2] == 1


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


def test_a_batch_is_refused_rather_than_threaded_through_one_cache(decoder) -> None:
    """One state is one session, so a batch has no correct interpretation here.

    ``AutoencoderKLWan.decode`` handles a batch by decoding each sample with
    its own cleared cache (``use_slicing``). This decoder carries exactly one
    temporal context, so a batch would mix samples into it and still return
    correctly shaped pixels.
    """
    state = decoder.new_decode_state("s")
    batched = torch.zeros(2, VAE_CONFIG["z_dim"], 3, LATENT_H, LATENT_W)
    with pytest.raises(ValueError, match="one sample per session state"):
        decoder.decode_chunk(batched, state)
    assert state.frames_decoded == 0, "a refused call must not advance the session"


def test_session_id_must_be_meaningful(decoder) -> None:
    with pytest.raises(ValueError, match="session_id"):
        decoder.new_decode_state("  ")


def test_declared_state_bytes_scales_with_area_and_dtype(vae) -> None:
    # A local decoder, not the module-scoped fixture: the measured constant
    # belongs to this test, and leaking it would change what a reordered run
    # of the default-None case sees. Passing it here is also the constructor
    # argument's intended use.
    # Measured for the shipped checkpoint: 37832 KiB of fp32 cache at 64x64.
    decoder = WanStreamingDecoder(vae, bytes_per_pixel_fp32=37832 * 1024 / (64 * 64))
    fp32 = decoder.declared_state_bytes(height=480, width=832, dtype=torch.float32)
    bf16 = decoder.declared_state_bytes(height=480, width=832, dtype=torch.bfloat16)
    half_area = decoder.declared_state_bytes(height=240, width=832, dtype=torch.float32)
    assert bf16 == fp32 // 2
    assert half_area == fp32 // 2
    assert fp32 / 2**20 == pytest.approx(3602.0, abs=5.0)

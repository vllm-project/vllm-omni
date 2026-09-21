# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Correctness + boundedness tests for the incremental (windowed) HiFT path.

The streaming vocoder runs HiFT over a bounded mel window instead of the full
cumulative history, carrying harmonic phase and noise-buffer offset across
chunks.

* **Correctness** — windowed output must match the full cumulative re-run.
  With full history the match is bit-exact (proves the phase/noise carry is
  right). With a bounded window, float64 still agrees to ~1e-16 (the window
  is algebraically exact); float32 differs by ~2e-7 absolute (reassociation
  rounding, not truncation), so tests assert a tight atol/rtol rather than
  PCM16 byte-identity, since a sub-LSB perturbation can flip a boundary sample.
* **Boundedness** — the per-chunk window must not grow with mel history.

Tests are parametrized across REAL_CONFIG (24000 Hz -> SineGen2, what
CosyVoice3 ships) and LEGACY_CONFIG (22050 Hz -> SineGen) so a class-selection
regression in either path is caught.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    CausalConvRNNF0Predictor,
    CausalHiFTGenerator,
    SineGen,
    SineGen2,
    _wrapped_slice,
)
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHUNK_LEN = 24
TOTAL_MEL = 96
SPM = 480  # samples per mel frame for the small test HiFT (8*5*3*4), matching the real model

LONG_TOTAL_MEL = 400  # long enough that the 64-frame window actually truncates
PCM16_LSB = 1.0 / 32767.0  # smallest representable PCM16 difference

# Worst observed deviation is 1.97e-7 (~2x float32 eps); atol=1e-6 keeps 5x headroom.
ATOL = 1e-6
RTOL = 1e-5

# What CosyVoice3 ships (-> SineGen2). nb_harmonics=8 is required, not a free
# choice: SineGen2's causal buffers are hardcoded to dim 9 (= harmonic_num+1).
REAL_CONFIG = {"sampling_rate": 24000, "nb_harmonics": 8}
LEGACY_CONFIG = {"sampling_rate": 22050, "nb_harmonics": 4}  # -> SineGen, still shipped code

CONFIGS = [
    pytest.param(REAL_CONFIG, id="sinegen2_real_config"),
    pytest.param(LEGACY_CONFIG, id="sinegen1_legacy_config"),
]


def _make_hift(config: dict = REAL_CONFIG) -> CausalHiFTGenerator:
    """Small CausalHiFTGenerator; defaults to the config CosyVoice3 ships."""
    torch.manual_seed(0)
    f0_predictor = CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=16)
    return CausalHiFTGenerator(
        in_channels=80,
        base_channels=32,
        nb_harmonics=config["nb_harmonics"],
        sampling_rate=config["sampling_rate"],
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=f0_predictor,
    ).eval()


def test_real_config_resolves_to_sinegen2():
    """Guard against silently testing the wrong SineGen class again."""
    hift = _make_hift(REAL_CONFIG)
    assert isinstance(hift.m_source.l_sin_gen, SineGen2)

    hift_legacy = _make_hift(LEGACY_CONFIG)
    assert isinstance(hift_legacy.m_source.l_sin_gen, SineGen)


def _make_model(hift: CausalHiFTGenerator, window_len: int) -> CosyVoice3Code2Wav:
    """Build a CosyVoice3Code2Wav shell wired to the given HiFT."""
    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.hift = hift
    model._hift_window_len = window_len
    return model


def _chunks(total_mel: int = TOTAL_MEL, chunk_len: int = CHUNK_LEN) -> list[torch.Tensor]:
    torch.manual_seed(1)
    mel = torch.randn(1, 80, total_mel)
    return [mel[:, :, i : i + chunk_len] for i in range(0, total_mel, chunk_len)]


def _variable_chunks(total_mel: int, first_len: int, step_len: int) -> list[torch.Tensor]:
    """First chunk sized like a real prompt prefill, then small fixed-size steps."""
    torch.manual_seed(1)
    mel = torch.randn(1, 80, total_mel)
    chunks = [mel[:, :, :first_len]]
    pos = first_len
    while pos < total_mel:
        chunks.append(mel[:, :, pos : pos + step_len])
        pos += step_len
    return chunks


def _full_reference(model: CosyVoice3Code2Wav, chunks: list[torch.Tensor]) -> torch.Tensor:
    """Re-run HiFT over the full cumulative mel each chunk (pre-windowing behavior).

    Uses non-finalize inference per chunk, matching the old streaming vocoder; a
    single finalize pass would emit the look-right tail streaming holds back.
    """
    torch.manual_seed(0)
    emitted = []
    cache: dict[str, torch.Tensor] | None = None
    for chunk in chunks:
        cached = None if cache is None else cache.get("mel")
        if cached is not None and cached.numel() > 0:
            tts_mel = torch.cat([cached, chunk], dim=-1)
        else:
            tts_mel = chunk
        speech, _, _ = model.hift.inference(speech_feat=tts_mel, finalize=False)
        speech = speech.reshape(speech.shape[0], -1)
        offset = 0 if cache is None else cache.get("speech_offset")
        emitted.append(speech[:, offset:])
        cache = {"mel": tts_mel.detach().cpu(), "speech_offset": int(speech.shape[-1])}
    return torch.cat(emitted, dim=-1)


def _pcm16(x: torch.Tensor) -> torch.Tensor:
    """Quantize float audio to PCM16, the format the server streams to clients."""
    return (x.clamp(-1, 1) * 32767).round().to(torch.int16)


def _incremental(model: CosyVoice3Code2Wav, chunks: list[torch.Tensor]) -> torch.Tensor:
    """Run the new windowed path via _stream_hift_from_feat."""
    torch.manual_seed(0)
    emitted = []
    cache: dict[str, torch.Tensor] | None = None
    for chunk in chunks:
        speech, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=False)
        emitted.append(speech.reshape(speech.shape[0], -1))
    return torch.cat(emitted, dim=-1)


def _rel_divergence(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().mean() / a.abs().mean().clamp_min(1e-6)).item()


@pytest.mark.parametrize("config", CONFIGS)
def test_incremental_hift_matches_full_reference_exactly_with_full_history(config):
    """With a window covering the full history, the windowed path must match the
    full cumulative re-run exactly. This proves the phase/noise carry is right."""
    hift = _make_hift(config)
    model = _make_model(hift, window_len=TOTAL_MEL)  # window >= history
    chunks = _chunks()

    full = _full_reference(model, chunks)
    incr = _incremental(model, chunks)

    assert full.shape == incr.shape, f"{full.shape} vs {incr.shape}"
    assert _rel_divergence(full, incr) < 1e-6


@pytest.mark.parametrize("config", CONFIGS)
def test_incremental_hift_bounded_window_is_close(config):
    """With a bounded window, the windowed path stays close to the full re-run;
    only the truncated deep history diverges, and it is small."""
    hift = _make_hift(config)
    model = _make_model(hift, window_len=48)
    chunks = _chunks()

    full = _full_reference(model, chunks)
    incr = _incremental(model, chunks)

    assert full.shape == incr.shape
    torch.testing.assert_close(incr, full, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", CONFIGS)
def test_incremental_hift_matches_streaming_reference_within_tolerance(config):
    """Windowed output matches the full-cumulative reference within float32 rounding.

    The load-bearing correctness test: 400 mel frames far exceeds the 64-frame
    window, so truncation genuinely exercises the phase/noise carry. Float64
    agrees to ~1e-16, confirming the residual here is rounding, not truncation.
    """
    hift = _make_hift(config)
    model = _make_model(hift, window_len=64)  # the real _hift_window_len
    chunks = _chunks(total_mel=LONG_TOTAL_MEL)

    trim = int(hift.f0_predictor.condnet[0].causal_padding)
    assert LONG_TOTAL_MEL > 64 + trim + CHUNK_LEN  # window must actually truncate

    full = _full_reference(model, chunks)
    incr = _incremental(model, chunks)

    assert full.shape == incr.shape, f"{full.shape} vs {incr.shape}"
    torch.testing.assert_close(incr, full, atol=ATOL, rtol=RTOL)

    max_dev = (full - incr).abs().max().item()  # stays far below one PCM16 step
    assert max_dev < PCM16_LSB / 10, f"max deviation {max_dev:.3e} vs PCM16 LSB {PCM16_LSB:.3e}"


def test_incremental_hift_window_is_bounded():
    """Per-chunk window must not grow with cumulative mel history."""
    hift = _make_hift()
    model = _make_model(hift, window_len=32)
    chunks = _chunks(total_mel=200, chunk_len=24)

    trim = int(hift.f0_predictor.condnet[0].causal_padding)
    f0_margin = int(hift.f0_predictor.left_context_frames)

    cache = None
    window_sizes = []
    for chunk in chunks:
        _, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=False)
        window_sizes.append(int(cache["mel"].shape[-1]))

    assert max(window_sizes) <= 32 + trim + f0_margin + 24  # bounded regardless of utterance length
    steady = max(window_sizes)
    assert window_sizes.count(steady) >= 2  # steady-state value repeats once history exceeds window


def test_incremental_hift_finalize_releases_tail():
    """Finalize must emit the released look-right tail without crashing."""
    hift = _make_hift()
    model = _make_model(hift, window_len=32)
    chunks = _chunks(total_mel=72, chunk_len=24)

    cache = None
    emitted = []
    for i, chunk in enumerate(chunks):
        finalize = i == len(chunks) - 1
        speech, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=finalize)
        emitted.append(speech.reshape(speech.shape[0], -1))
        if finalize:
            assert cache is None

    full = torch.cat(emitted, dim=-1)
    assert full.shape[-1] > 0
    assert torch.isfinite(full).all()


def test_incremental_hift_emits_full_audio_length():
    """Streamed output covers the full mel history at samples-per-mel resolution.

    Guards against emission arithmetic dropping an upsample stage, which
    silently truncates audio (e.g. 1/3 of expected length with [8,5,3]).
    """
    hift = _make_hift()
    model = _make_model(hift, window_len=32)
    chunks = _chunks(total_mel=72, chunk_len=24)

    cache = None
    emitted = []
    for i, chunk in enumerate(chunks):
        finalize = i == len(chunks) - 1
        speech, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=finalize)
        emitted.append(speech.reshape(speech.shape[0], -1))
        if finalize:
            assert cache is None

    full = torch.cat(emitted, dim=-1)
    # total_mel * SPM, +slack for finalize's look-right tail, -slack for a dropped upsample stage
    total_mel = sum(c.shape[-1] for c in chunks)
    assert full.shape[-1] > total_mel * SPM * 0.8, f"{full.shape[-1]} vs {total_mel * SPM}"
    assert full.shape[-1] < total_mel * SPM * 1.5, f"{full.shape[-1]} vs {total_mel * SPM}"


@pytest.mark.parametrize("config", CONFIGS)
def test_finalize_matches_single_pass_length_exactly(config):
    """Streamed total (chunked, last=finalize) must exactly match a single
    non-streaming finalize=True call over the whole mel -- not just approximately.

    The finalize release must include `trim` (the F0 predictor's own held-back
    frames), or the last 3 mel frames (1,440 samples / 60ms at 24kHz) of every
    stream's audio go missing.
    """
    hift = _make_hift(config)
    chunks = _chunks(total_mel=200, chunk_len=24)

    torch.manual_seed(0)
    full_mel = torch.cat(chunks, dim=-1)
    ground_truth, _, _ = hift.inference(speech_feat=full_mel, finalize=True)
    ground_truth = ground_truth.reshape(ground_truth.shape[0], -1)

    hift2 = _make_hift(config)
    model = _make_model(hift2, window_len=64)
    torch.manual_seed(0)
    cache = None
    emitted = []
    for i, chunk in enumerate(chunks):
        finalize = i == len(chunks) - 1
        speech, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=finalize)
        emitted.append(speech.reshape(speech.shape[0], -1))
    streamed = torch.cat(emitted, dim=-1)

    assert streamed.shape[-1] == ground_truth.shape[-1], (
        f"{streamed.shape[-1]} vs {ground_truth.shape[-1]} (diff={ground_truth.shape[-1] - streamed.shape[-1]} samples)"
    )
    torch.testing.assert_close(streamed, ground_truth, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", CONFIGS)
def test_incremental_hift_matches_reference_for_voiced_f0(config):
    """Forces F0 above nsf_voiced_threshold (untrained weights never trigger voiced,
    so no other test here does) and checks windowed output against the full-
    cumulative reference, which exercises condnet's left-causal convs and
    SineGen/SineGen2's phase integration under genuinely voiced harmonics.

    Tolerance is looser than ATOL/RTOL: summing this sustained constant-F0 phase
    in per-window chunks vs. one recompute reassociates float32 adds differently;
    real speech's varying/unvoiced F0 keeps this far smaller in practice.
    """
    hift = _make_hift(config)
    with torch.no_grad():
        hift.f0_predictor.classifier.bias.fill_(150.0)  # clearly voiced (threshold is 10)
    chunks = _chunks(total_mel=200, chunk_len=24)

    full = _full_reference(_make_model(hift, window_len=400), chunks)
    incr = _incremental(_make_model(hift, window_len=64), chunks)

    assert full.shape == incr.shape
    torch.testing.assert_close(incr, full, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("first_len,step_len", [(10, 6), (10, 3), (20, 3), (10, 1)])
@pytest.mark.skipif(
    current_omni_platform.is_rocm(),
    reason="exhaustive CPU reference matrix exceeds the AMD CI time budget",
)
def test_incremental_hift_matches_reference_for_voiced_f0_small_chunks(config, first_len, step_len):
    """Same as test_incremental_hift_matches_reference_for_voiced_f0, but with small,
    sub-receptive-field chunk sizes (down to 1 mel frame) instead of the uniform
    24-frame chunks used elsewhere. cache_state must carry the F0 predictor's own
    left receptive field across calls regardless of how small a single chunk is,
    or the next call's margin comes up short and substitutes zero-padding for real
    history -- corrupting SineGen's phase integration the same way as an absent
    margin entirely.
    """
    hift = _make_hift(config)
    with torch.no_grad():
        hift.f0_predictor.classifier.bias.fill_(150.0)  # clearly voiced (threshold is 10)
    chunks = _variable_chunks(total_mel=200, first_len=first_len, step_len=step_len)

    full = _full_reference(_make_model(hift, window_len=400), chunks)
    incr = _incremental(_make_model(hift, window_len=64), chunks)

    assert full.shape == incr.shape
    torch.testing.assert_close(incr, full, atol=2e-4, rtol=2e-4)


def test_wrapped_slice_wraps_position_deterministically():
    """SineGen/SineGen2/SourceModuleHnNSF's noise buffers are a fixed 300s at
    24kHz; a stream running past that must keep repeating the realization
    (position-deterministic) rather than the slice coming back short and
    failing to broadcast against the rest of that call's tensors.
    """
    buf = torch.arange(10).reshape(1, 10, 1)

    assert _wrapped_slice(buf, 2, 3).flatten().tolist() == [2, 3, 4]
    assert _wrapped_slice(buf, 8, 4).flatten().tolist() == [8, 9, 0, 1]
    assert _wrapped_slice(buf, 12, 3).flatten().tolist() == [2, 3, 4]  # offset past the end
    assert _wrapped_slice(buf, 5, 13).flatten().tolist() == [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7]


def test_incremental_hift_empty_middle_chunk_emits_nothing():
    """An empty middle chunk (reachable from forward_streaming_batch when a
    row's post-trim feat comes out empty near the lookahead boundary) must
    emit zero samples and leave cache_state's phase_acc unchanged, not
    re-emit the re-decoded overlap window as if it were new audio.
    """
    hift = _make_hift(REAL_CONFIG)
    model = _make_model(hift, window_len=64)
    chunks = _chunks(total_mel=96, chunk_len=24)
    empty = chunks[0][:, :, :0]

    cache = None
    for chunk in chunks[:2]:
        _, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=False)
    assert cache is not None
    phase_acc_before = cache["phase_acc"]

    speech, cache = model._stream_hift_from_feat(empty, cache_state=cache, finalize=False)

    assert speech.shape[-1] == 0
    if phase_acc_before is None:
        assert cache["phase_acc"] is None
    else:
        torch.testing.assert_close(cache["phase_acc"], phase_acc_before)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("mel_frames", [1, 2, 3, 4])
def test_incremental_hift_short_finalize_chunk_does_not_crash(config, mel_frames):
    """A one-token (or otherwise sub-trim) utterance that finalizes on its very
    first chunk must not crash. On finalize the F0 predictor consumes the whole
    window (it does not hold back `trim` frames), so the phase-carry index must
    not be inflated by `trim` -- it would point past the end of a short window.
    """
    hift = _make_hift(config)
    model = _make_model(hift, window_len=32)
    chunk = torch.randn(1, 80, mel_frames)

    speech, cache = model._stream_hift_from_feat(chunk, cache_state=None, finalize=True)

    assert cache is None
    assert speech.shape[0] == 1
    assert speech.shape[-1] > 0
    assert torch.isfinite(speech).all()

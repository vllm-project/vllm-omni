# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for ``Qwen3OmniMoeCode2Wav.chunked_decode_streaming``.

Background
----------
The Code2Wav decoder is a causal conv / transposed-conv stack whose forward
emits a fixed number ``C`` *fewer* samples than ``frames * total_upsample`` (the
right-edge trim, ``C == 555`` for Qwen3-Omni-30B-A3B). ``chunked_decode_streaming``
used to slice ``[left_context * up : code_seq_len * up]``, assuming the full
length. Because the real output is ``C`` short, the upper bound clamped and every
streaming chunk silently dropped its last ``C`` samples -> a gap/click at every
chunk boundary plus ~1.2% cumulative time compression.

The fix measures ``tail`` from the actual decoder output and shifts each chunk's
start back by ``tail`` so the chunk refills the gap the previous chunk left.

Why this is path-agnostic (eager *and* CUDA graph)
--------------------------------------------------
Originally the CUDA-graph wrapper returned a *surplus* (nominal ``actual * up``,
with stale padded-tail samples), so ``tail`` evaluated to 0 and the graph path
looked immune. Since #4466 (``CUDAGraphDecoderWrapper._trim_replay_output``) the
graph wrapper returns the eager-equivalent ``actual * up - C`` instead, so the
same per-chunk tail drop happens on the graph branch too and the same fix
repairs it. These tests cover both branches; ``_FakeWrapper`` models the
post-#4466 contract (``decode`` returns ``actual * up - C``).

These are pure length/slice-arithmetic tests: CPU-only, no weights, no GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import (
    Qwen3OmniMoeCode2Wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_UP = 8  # total_upsample
_TRIM = 5  # causal right-edge trim C, 0 < C < up (mirrors 555 < 1920)
_Q = 4  # num_quantizers


class _ShortTrimDecoder:
    """Minimal stand-in exposing only what ``chunked_decode_streaming`` touches.

    Models a *shift-equivariant* causal decoder: the waveform value at an
    absolute output sample ``q`` is ``floor(q / up)`` == the index of the frame
    that sample belongs to. With codes set so frame ``i`` carries value ``i``,
    any window covering a frame (with enough right context) reproduces the same
    global waveform -- which is exactly the property that lets the next chunk
    refill the tail the previous chunk dropped. The forward emits ``trim`` fewer
    samples than ``frames * up`` (the right-edge trim that lacks right context).
    """

    def __init__(self, *, up=_UP, trim=_TRIM, num_quantizers=_Q, cudagraph=False):
        self.total_upsample = up
        self.trim = trim
        self.config = SimpleNamespace(num_quantizers=num_quantizers)
        self._cudagraph_enabled = cudagraph
        self._cudagraph_wrapper = _FakeWrapper(self) if cudagraph else None

    def _decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [B, Q, W]; frame value (row 0) == absolute frame index.
        batch, _, width = codes.shape
        out_len = max(0, width * self.total_upsample - self.trim)
        frame_of = torch.arange(out_len) // self.total_upsample  # [out_len]
        vals = codes[:, 0, :].to(torch.float32)  # [B, W]
        out = vals.gather(1, frame_of.unsqueeze(0).expand(batch, -1))  # [B, out_len]
        return out.unsqueeze(1)  # [B, 1, out_len]

    # ``chunked_decode_streaming`` calls ``self(codes)`` on the eager branch.
    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        return self._decode(codes)


class _FakeWrapper:
    """Models ``CUDAGraphDecoderWrapper.decode`` after #4466: returns the
    eager-equivalent short length (``actual * up - C``), values correct."""

    def __init__(self, decoder: _ShortTrimDecoder):
        self.decoder = decoder

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder._decode(codes)


def _make_codes(frames: int, q: int = _Q) -> torch.Tensor:
    """codes[0, :, i] == i so each frame carries its own absolute index."""
    idx = torch.arange(frames, dtype=torch.long)
    return idx.view(1, 1, frames).expand(1, q, frames).contiguous()


def _decode_chunk(model, window, lc):
    """Call the real production method for one streaming window."""
    win_frames = window.shape[-1]
    out = Qwen3OmniMoeCode2Wav.chunked_decode_streaming(
        model,
        window,
        left_context_size=[lc],
        seq_token_counts=[win_frames * model.config.num_quantizers],
    )
    return out[0].reshape(-1)


def _drive_async(model, codes, *, initial=4, chunk=25, left_context=25):
    """Replicate qwen3_omni's ``talker2code2wav_async_chunk`` 4/25/25 schedule."""
    total = codes.shape[-1]
    chunks = []
    emitted = 0
    first = True
    while emitted < total:
        target = min(initial, total) if first else min(emitted + chunk, total)
        lc = min(emitted, left_context)
        window = codes[:, :, emitted - lc : target]
        chunks.append(_decode_chunk(model, window, lc))
        emitted = target
        first = False
    return torch.cat(chunks, dim=-1)


def _single_shot(model, codes):
    """Ground-truth: one full-utterance forward (length T*up - trim)."""
    return model(codes)[0].reshape(-1)


def test_streaming_reconstructs_single_shot_eager():
    """Eager: concatenated streaming chunks == the single-shot full decode."""
    model = _ShortTrimDecoder()
    codes = _make_codes(116)

    gt = _single_shot(model, codes)
    streamed = _drive_async(model, codes)

    assert streamed.shape[-1] == gt.shape[-1]
    torch.testing.assert_close(streamed, gt)


def test_streaming_reconstructs_single_shot_cudagraph():
    """CUDA-graph branch (post-#4466 contract): identical exact reconstruction.

    Answers the first-principles question: the non-eager path is NOT immune on
    current main -- it drops the same tail without the fix, and the fix repairs
    it the same way.
    """
    model = _ShortTrimDecoder(cudagraph=True)
    codes = _make_codes(116)

    gt = _single_shot(model, codes)  # forward() is eager regardless of the branch
    streamed = _drive_async(model, codes)

    assert streamed.shape[-1] == gt.shape[-1]
    torch.testing.assert_close(streamed, gt)


def test_eager_and_cudagraph_streaming_match():
    """Both decode branches must yield bit-identical streamed audio."""
    codes = _make_codes(116)
    eager = _drive_async(_ShortTrimDecoder(cudagraph=False), codes)
    graph = _drive_async(_ShortTrimDecoder(cudagraph=True), codes)
    torch.testing.assert_close(eager, graph)


def test_per_chunk_tail_drop_is_repaired_not_present():
    """The defect is exactly ``C`` lost per internal boundary; the fix removes it.

    Without the fix the streamed length would be short by
    ``(num_internal_boundaries) * trim``; assert the boundary count and that the
    repaired stream has the full single-shot length (no deficit).
    """
    model = _ShortTrimDecoder()
    codes = _make_codes(116)
    gt = _single_shot(model, codes)
    streamed = _drive_async(model, codes)

    # 116 frames @ initial=4, chunk=25 -> targets 4,29,54,79,104,116 => 5 boundaries
    num_internal_boundaries = 5
    buggy_len = gt.shape[-1] - num_internal_boundaries * _TRIM
    assert streamed.shape[-1] == gt.shape[-1]
    assert streamed.shape[-1] != buggy_len  # the pre-fix slice produced buggy_len


def test_first_chunk_start_is_not_shifted_below_zero():
    """First chunk has left_context==0 -> start = max(0, 0 - tail) = 0 (unchanged)."""
    model = _ShortTrimDecoder()
    codes = _make_codes(4)  # single initial chunk, lc=0
    out = _decode_chunk(model, codes, lc=0)
    # length == frames*up - trim, values == frame indices (GT prefix)
    assert out.shape[-1] == 4 * _UP - _TRIM
    torch.testing.assert_close(out, _single_shot(model, codes))


def test_no_shift_when_decoder_returns_nominal_length():
    """C==0 decoder (e.g. Qwen3-TTS): tail==0 -> genuine no-op.

    With no trim, the fixed slice must equal the original
    ``[left_context*up : code_seq_len*up]`` -- every chunk emits exactly
    ``ctx_len*up`` samples and streaming reconstructs the full (untrimmed) audio.
    """
    model = _ShortTrimDecoder(trim=0)
    codes = _make_codes(116)
    gt = _single_shot(model, codes)
    streamed = _drive_async(model, codes)

    assert gt.shape[-1] == 116 * _UP  # no trim -> nominal full length
    assert streamed.shape[-1] == gt.shape[-1]
    torch.testing.assert_close(streamed, gt)


@pytest.mark.parametrize("cudagraph", [False, True])
@pytest.mark.parametrize("trim", [_TRIM, 0])
@pytest.mark.parametrize("short_frames,left_context", [(1, 0), (4, 0), (29, 4), (26, 25), (37, 25)])
@pytest.mark.parametrize("reverse", [False, True])
def test_streaming_batch_matches_independent_requests(cudagraph, trim, short_frames, left_context, reverse):
    """Padding a request beside a longer one cannot supply valid future codec frames.

    Compare values as well as lengths: an incorrect start and end can shift
    the whole chunk while preserving its duration. The independent request
    is the oracle, rather than another copy of the slicing arithmetic.
    """
    model = _ShortTrimDecoder(trim=trim, cudagraph=cudagraph)
    short = _make_codes(short_frames)
    long = _make_codes(50) + 100
    windows = [(short, left_context), (long, 25)]
    if reverse:
        windows.reverse()
    batch = torch.zeros(2, _Q, 50, dtype=torch.long)
    for i, (window, _) in enumerate(windows):
        batch[i, :, : window.shape[-1]] = window[0]
    actual = Qwen3OmniMoeCode2Wav.chunked_decode_streaming(
        model,
        batch,
        left_context_size=[lc for _, lc in windows],
        seq_token_counts=[window.numel() for window, _ in windows],
    )
    assert len(actual) == len(windows)
    for output, (window, lc) in zip(actual, windows):
        expected = _decode_chunk(model, window, lc)
        torch.testing.assert_close(output.reshape(-1), expected, rtol=0, atol=0)


@pytest.mark.parametrize("cudagraph", [False, True])
def test_streaming_preserves_samples_when_batch_composition_changes(cudagraph):
    """A singleton -> mixed batch -> singleton stream cannot skip then repeat samples."""
    model = _ShortTrimDecoder(cudagraph=cudagraph)
    codes = _make_codes(54)
    first = _decode_chunk(model, codes[:, :, :4], 0)
    middle_batch = torch.zeros(2, _Q, 50, dtype=torch.long)
    middle_batch[0, :, :29] = codes[0, :, :29]
    middle_batch[1] = _make_codes(50)[0] + 100
    middle = Qwen3OmniMoeCode2Wav.chunked_decode_streaming(
        model, middle_batch, left_context_size=[4, 25], seq_token_counts=[29 * _Q, 50 * _Q]
    )[0].reshape(-1)
    last = _decode_chunk(model, codes[:, :, 4:54], 25)
    streamed = torch.cat((first, middle, last))
    torch.testing.assert_close(streamed, _single_shot(model, codes), rtol=0, atol=0)

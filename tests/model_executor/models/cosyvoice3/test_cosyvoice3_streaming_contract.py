# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for CosyVoice 3 streaming code2wav / vocoder.

Pins:
1. Bounded mel cache invariant (cache_state['mel'] <= mel_cache_len).
2. Exact duration parity between multi-chunk streaming emission and offline forward (eliminating boundary sample inflation).
3. Tail withholding on non-final chunks and complete emission on finalize.
4. Equivalence between bucketed batched vocoder execution and sequential execution.
5. Backward test compatibility seam preservation when _stream_hift_from_feat is monkeypatched.
"""

from __future__ import annotations

import math
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ContractDummyHiFT(nn.Module):
    """Linear mock HiFT generator with CosyVoice 3 upsample architecture."""

    def __init__(self, upsample_rates: list[int] | None = None, hop_len: int = 4):
        super().__init__()
        self.upsample_rates = upsample_rates or [8, 5, 3]
        self.istft_params = {"hop_len": hop_len}
        self.upsample_scale = int(math.prod(self.upsample_rates) * hop_len)
        self.param = nn.Parameter(torch.zeros(1))
        self.m_source = SimpleNamespace(l_linear=SimpleNamespace(weight=self.param))

    def inference(self, speech_feat: torch.Tensor, finalize: bool = True) -> tuple[torch.Tensor, None]:
        b, _, t = speech_feat.shape
        waveform = speech_feat.mean(dim=1, keepdim=True).repeat_interleave(self.upsample_scale, dim=-1)
        return waveform, None


def _build_test_code2wav(mel_cache_len: int = 20) -> CosyVoice3Code2Wav:
    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.hift = _ContractDummyHiFT()
    model.mel_cache_len = mel_cache_len
    upsample_rates = getattr(model.hift, "upsample_rates", [8, 5, 3])
    istft_hop_len = getattr(model.hift, "istft_params", {}).get("hop_len", 4)
    upsample_scale = int(math.prod(upsample_rates) * istft_hop_len)
    model.source_cache_len = int(model.mel_cache_len * upsample_scale)
    return model


def test_streaming_mel_cache_bounded():
    """Verify that cached mel frames never exceed mel_cache_len."""
    model = _build_test_code2wav(mel_cache_len=20)
    chunk_lens = [10, 25, 30, 15]

    state = None
    for length in chunk_lens:
        feat = torch.randn(1, 80, length)
        _, state = model._stream_hift_from_feat(feat, cache_state=state, finalize=False)
        assert state is not None
        assert state["mel"].shape[-1] <= model.mel_cache_len


@pytest.mark.parametrize(
    "chunk_lens",
    [
        [25, 30, 20, 35],
        [5, 10, 15, 20],
        [40, 50],
        [10, 10, 10, 10, 10],
    ],
)
def test_streaming_duration_length_parity(chunk_lens: list[int]):
    """Verify exact bit-for-bit sample length parity between streaming and offline forward."""
    model = _build_test_code2wav(mel_cache_len=20)
    total_mel_frames = sum(chunk_lens)
    expected_total_samples = total_mel_frames * model.hift.upsample_scale

    emitted_chunks: list[torch.Tensor] = []
    state = None
    for i, length in enumerate(chunk_lens):
        finalize = i == len(chunk_lens) - 1
        feat = torch.randn(1, 80, length)
        audio, state = model._stream_hift_from_feat(feat, cache_state=state, finalize=finalize)
        emitted_chunks.append(audio)

    total_streaming_samples = sum(chunk.shape[-1] for chunk in emitted_chunks)
    assert total_streaming_samples == expected_total_samples
    assert state is None


def test_streaming_tail_withholding_and_finalize():
    """Verify that intermediate chunks withhold source_cache_len samples and finalize emits tail."""
    model = _build_test_code2wav(mel_cache_len=20)
    feat1 = torch.randn(1, 80, 25)
    audio1, state1 = model._stream_hift_from_feat(feat1, cache_state=None, finalize=False)

    assert state1 is not None
    assert state1["speech"].shape[-1] == model.source_cache_len
    assert audio1.shape[-1] == 25 * model.hift.upsample_scale - model.source_cache_len

    feat2 = torch.randn(1, 80, 15)
    audio2, state2 = model._stream_hift_from_feat(feat2, cache_state=state1, finalize=True)

    assert state2 is None
    assert audio2.shape[-1] == 15 * model.hift.upsample_scale + model.source_cache_len


def test_batched_vs_single_stream_equivalence():
    """Verify that equal-length bucketing batched HiFT matches single-item streaming."""
    model = _build_test_code2wav(mel_cache_len=20)

    f1 = torch.randn(1, 80, 25)
    f2 = torch.randn(1, 80, 25)
    f3 = torch.randn(1, 80, 30)

    items = [(0, f1, None), (1, f2, None), (2, f3, None)]
    batch_results = dict(model._stream_hift_from_feat_batch(items, finalize=False))

    single_0 = model._stream_hift_from_feat(f1, cache_state=None, finalize=False)
    single_1 = model._stream_hift_from_feat(f2, cache_state=None, finalize=False)
    single_2 = model._stream_hift_from_feat(f3, cache_state=None, finalize=False)

    torch.testing.assert_close(batch_results[0][0], single_0[0])
    torch.testing.assert_close(batch_results[0][1]["speech"], single_0[1]["speech"])
    torch.testing.assert_close(batch_results[0][1]["mel"], single_0[1]["mel"])

    torch.testing.assert_close(batch_results[1][0], single_1[0])
    torch.testing.assert_close(batch_results[1][1]["speech"], single_1[1]["speech"])
    torch.testing.assert_close(batch_results[1][1]["mel"], single_1[1]["mel"])

    torch.testing.assert_close(batch_results[2][0], single_2[0])
    torch.testing.assert_close(batch_results[2][1]["speech"], single_2[1]["speech"])
    torch.testing.assert_close(batch_results[2][1]["mel"], single_2[1]["mel"])


def test_seam_preservation_when_stream_hift_patched():
    """Verify that forward_streaming_batch preserves patched _stream_hift_from_feat seams."""
    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)

    patched_calls: list[tuple[tuple[int, ...], bool]] = []

    def fake_stream(self, feat: torch.Tensor, *, cache_state=None, finalize: bool = False):
        patched_calls.append((tuple(feat.shape), finalize))
        return feat, None

    model._stream_hift_from_feat = types.MethodType(fake_stream, model)

    items = [(0, torch.zeros(1, 80, 10), None), (1, torch.zeros(1, 80, 15), None)]
    results = model._stream_hift_from_feat_batch(items, finalize=True)

    assert len(patched_calls) == 2
    assert patched_calls[0] == ((1, 80, 10), True)
    assert patched_calls[1] == ((1, 80, 15), True)
    assert results[0][1][0].shape == (1, 80, 10)
    assert results[1][1][0].shape == (1, 80, 15)

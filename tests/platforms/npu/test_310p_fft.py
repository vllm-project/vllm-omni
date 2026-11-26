# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Correctness tests for the 310P pure-PyTorch FFT/STFT re-implementation.

All results are compared against CPU-native torch references. These tests run
on any device — the module under test only needs ``npu`` availability for its
fast paths, so on CPU-only CI the same math is exercised through the CPU
reference path of the operators.

Run:
    pytest tests/platforms/npu/test_310p_fft.py -v
"""

from __future__ import annotations

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms.npu._310p import fft_ops

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"npu": "310P"}, num_cards=1),
]

TOL = 5e-4


def _pair(t: torch.Tensor) -> torch.Tensor:
    if t.is_complex():
        return torch.view_as_real(t.resolve_conj())
    return t


@pytest.mark.parametrize("n", [8, 512, 1024])
def test_fft_pow2_matches_cpu(n):
    x = torch.randn(2, 3, n)
    got = fft_ops.fft_pow2(x, inverse=False)
    want = torch.fft.fft(x)
    assert (_pair(got) - _pair(want)).abs().max().item() / _pair(want).abs().max().item() < TOL


@pytest.mark.parametrize("n", [500, 1000])
def test_fft_bluestein_matches_cpu(n):
    x = torch.randn(4, n)
    got = fft_ops.fft_dft(x, inverse=False)
    want = torch.fft.fft(x)
    assert (_pair(got) - _pair(want)).abs().max().item() / _pair(want).abs().max().item() < TOL


def test_ifft_roundtrip():
    x = torch.randn(2, 3, 1024)
    rt = fft_ops.fft_pow2(fft_ops.fft_pow2(x, inverse=False), inverse=True).real
    assert (_pair(rt) - x).abs().max().item() < 1e-5


def test_rfft_irfft_roundtrip():
    x = torch.randn(2, 16000)
    spec = fft_ops.rfft(x)
    want_spec = torch.fft.rfft(x)
    assert (_pair(spec) - _pair(want_spec)).abs().max().item() / _pair(want_spec).abs().max().item() < TOL
    rt = fft_ops.irfft(spec, n=16000)
    assert (rt - x).abs().max().item() < 1e-4


def test_stft_matches_cpu_mel_frontend_params():
    """Qwen3-TTS / IndexTTS mel front-end parameters: n_fft=1024, hop=256."""
    wav = torch.randn(1, 22050) * 0.1
    kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )
    got = fft_ops.stft(wav, window=fft_ops.hann_window(1024), **kwargs)
    want = torch.stft(wav, window=torch.hann_window(1024), return_complex=True, **kwargs)
    assert got.shape == want.shape
    assert (_pair(got) - _pair(want)).abs().max().item() / _pair(want).abs().max().item() < TOL


def test_istft_matches_cpu():
    wav = torch.randn(1, 22050) * 0.1
    spec = torch.stft(
        wav,
        n_fft=1024,
        hop_length=256,
        window=torch.hann_window(1024),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    got = fft_ops.istft(spec, n_fft=1024, hop_length=256, window=fft_ops.hann_window(1024), center=True)
    want = torch.istft(spec, n_fft=1024, hop_length=256, window=torch.hann_window(1024), center=True)
    assert got.shape == want.shape
    assert (got - want).abs().max().item() / want.abs().max().item() < TOL


def test_stft_istft_waveform_roundtrip():
    wav = torch.randn(1, 22050) * 0.1
    kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )
    rt = fft_ops.istft(
        fft_ops.stft(wav, window=fft_ops.hann_window(1024), **kwargs),
        n_fft=1024,
        hop_length=256,
        window=fft_ops.hann_window(1024),
        center=True,
    )
    assert (rt - wav[..., : rt.shape[-1]]).abs().max().item() < 1e-5


def test_hann_window_matches_cpu():
    got = fft_ops.hann_window(1024)
    want = torch.hann_window(1024)
    assert (got - want).abs().max().item() == 0.0


def test_long_audio_real_load():
    """15s x2 at 24kHz — the Qwen3-TTS speaker-encoder workload."""
    wav = torch.randn(2, 24000 * 15) * 0.1
    kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )
    got = fft_ops.stft(wav, window=fft_ops.hann_window(1024), **kwargs)
    want = torch.stft(wav, window=torch.hann_window(1024), return_complex=True, **kwargs)
    assert got.shape == want.shape
    assert (_pair(got) - _pair(want)).abs().max().item() / _pair(want).abs().max().item() < TOL

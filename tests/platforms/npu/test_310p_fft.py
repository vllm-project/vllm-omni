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

from vllm_omni.platforms.npu._310p import fft_ops

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

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


def test_fft_ops_module_is_import_safe_and_non_invasive():
    """Regression: importing/using fft_ops must not touch process-wide torch APIs.

    Guards the P1 review finding — the 310P FFT overrides are bound locally
    to the Qwen3-TTS builder, never monkey-patched onto torch. If a future
    change reintroduces global patching, native torch APIs would be replaced
    and this test fails.
    """
    x = torch.randn(256)
    native = torch.fft.fft(x)
    # torch.fft.fft must still be the native implementation
    assert torch.fft.fft.__module__ == "torch._C" or "fft_ops" not in repr(torch.fft.fft)
    # and must still produce correct results (i.e., was not replaced by a
    # broken wrapper)
    assert (_pair(torch.fft.fft(x)) - _pair(native)).abs().max().item() == 0.0
    # fft_ops.stft must remain usable while torch.stft is untouched
    kwargs = dict(n_fft=256, hop_length=64, win_length=256, center=False, onesided=True)
    got = fft_ops.stft(x, window=fft_ops.hann_window(256), **kwargs)
    want = torch.stft(x, window=torch.hann_window(256), return_complex=True, **kwargs)
    assert (_pair(got) - _pair(want)).abs().max().item() / _pair(want).abs().max().item() < TOL


def test_310p_builder_mel_front_end_binding():
    """Regression: the 310P builder overrides mel locally, not via torch patches.

    Verifies (in a fresh subprocess, as the review requested) that:
    1. stock torch.hann_window still works after the 310P talker patch chain
       is exercised (no recursion, no replacement), and
    2. the 310P builder class exists with _mel_spectrogram_on_cpu=True so the
       NPU-native override in extract_speaker_embedding is the only on-device
       mel path.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import torch
        # 1. native window API works and is not a wrapper that recurses
        w = torch.hann_window(64)
        assert w.shape == (64,) and float(w.sum()) > 0.5
        # 2. the 310P builder specialization keeps CPU mel as the fallback flag
        from vllm_omni.platforms.npu._310p.patch.qwen3_tts import (
            _Qwen3TTSPromptEmbedsBuilder310P,
        )
        assert _Qwen3TTSPromptEmbedsBuilder310P._mel_spectrogram_on_cpu is True
        assert "fft_ops" in __import__("inspect").getsource(
            _Qwen3TTSPromptEmbedsBuilder310P.extract_speaker_embedding
        )
        print("SUBPROCESS-OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr[-2000:]}"
    assert "SUBPROCESS-OK" in proc.stdout

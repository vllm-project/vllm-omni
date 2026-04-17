# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for vllm_omni.utils.audio."""

import numpy as np
import pytest
import torch

from vllm_omni.utils.audio import mel_filter_bank, peak_normalize

# Parameter combinations used across the codebase.
_PARAM_SETS = [
    # Qwen3-TTS talker / speaker encoder (sr=24000)
    dict(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000),
    # CosyVoice3 whisper encoder, Qwen3-TTS 25Hz tokenizer (sr=16000, 80 mels)
    dict(sr=16000, n_fft=400, n_mels=80),
    # CosyVoice3 whisper encoder (sr=16000, 128 mels)
    dict(sr=16000, n_fft=400, n_mels=128),
]

_parametrize_params = pytest.mark.parametrize(
    "params", _PARAM_SETS, ids=lambda p: f"{p['sr']}_{p['n_fft']}_{p['n_mels']}"
)


class TestMelFilterBank:
    @_parametrize_params
    def test_output_shape(self, params):
        fb = mel_filter_bank(**params)
        n_freqs = params["n_fft"] // 2 + 1
        assert fb.shape == (params["n_mels"], n_freqs)

    @_parametrize_params
    def test_non_negative(self, params):
        fb = mel_filter_bank(**params)
        assert (fb >= 0).all()

    def test_dtype_is_float(self):
        fb = mel_filter_bank(sr=16000, n_fft=400, n_mels=80)
        assert fb.dtype == torch.float32

    def test_fmax_defaults_to_nyquist(self):
        """When fmax is omitted it should equal sr / 2."""
        fb_default = mel_filter_bank(sr=16000, n_fft=400, n_mels=80)
        fb_explicit = mel_filter_bank(sr=16000, n_fft=400, n_mels=80, fmax=8000.0)
        torch.testing.assert_close(fb_default, fb_explicit)

    def test_each_mel_band_has_nonzero_energy(self):
        """Every mel band should have at least one nonzero frequency bin."""
        fb = mel_filter_bank(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)
        for i in range(fb.shape[0]):
            assert fb[i].sum() > 0, f"mel band {i} is all zeros"

    def test_higher_fmax_extends_coverage(self):
        """A higher fmax should produce nonzero weights at higher frequency bins."""
        fb_low = mel_filter_bank(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=6000)
        fb_high = mel_filter_bank(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)
        # The highest nonzero column should be larger for fb_high.
        last_nonzero_low = (fb_low.sum(dim=0) > 0).nonzero()[-1].item()
        last_nonzero_high = (fb_high.sum(dim=0) > 0).nonzero()[-1].item()
        assert last_nonzero_high > last_nonzero_low


class TestPeakNormalize:
    def test_silence_unchanged(self):
        """All-zero input should remain all-zero."""
        audio = np.zeros(1600, dtype=np.float32)
        result = peak_normalize(audio, db_level=-6.0)
        np.testing.assert_array_equal(result, audio)

    def test_peak_reaches_target(self):
        """After normalization, peak amplitude should be at target dB."""
        rng = np.random.default_rng(7)
        audio = rng.uniform(-0.4, 0.4, size=16000).astype(np.float32)

        result = peak_normalize(audio, db_level=-6.0)
        peak_db = 20 * np.log10(np.abs(result).max())
        np.testing.assert_allclose(peak_db, -6.0, atol=1e-4)

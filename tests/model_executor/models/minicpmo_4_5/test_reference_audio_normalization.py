# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the reference-audio normalization grid (S1b).

Every request's reference audio is folded onto one sample rate and one fixed
length so the CFM attention cache origin (L0) is a single constant; that is
what collapsed the CFM CUDA-graph key space back from 58 shapes to a handful
and cut the capture storm behind the #6628 regression. These tests pin the
pure math so future edits cannot silently widen L0 again.
"""

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    _REF_MAX_SECONDS,
    _REF_TARGET_SAMPLE_RATE,
    _normalize_reference,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TARGET = _REF_TARGET_SAMPLE_RATE
MAX_SAMPLES = int(_REF_MAX_SECONDS * TARGET)  # 6 s @ 24 kHz


def test_mono_downmix_averages_channels_without_interleaving():
    """(2, T) stereo must become channel-mean mono, not a reshape interleave.

    reshape(-1) on (2, T) would alternate 1/0 samples; mean(dim=0) yields a
    constant 0.5 waveform. The whole-output equality fails on any interleave.
    """
    stereo = torch.zeros(2, 16)
    stereo[0] = 1.0
    waveform, sr = _normalize_reference(stereo, TARGET)
    assert sr == TARGET
    assert waveform.shape == (MAX_SAMPLES,)
    assert torch.all(waveform[:16] == 0.5)  # mono body
    assert torch.all(waveform[16:] == 0.0)  # zero-padded tail


def test_resample_48k_to_24k_halves_sample_count():
    """Off-grid sample rates are resampled onto 24 kHz before padding."""
    wav_48k = torch.arange(24000, dtype=torch.float32)
    waveform, sr = _normalize_reference(wav_48k, 48000)
    assert sr == TARGET
    assert waveform.shape == (MAX_SAMPLES,)
    assert torch.all(waveform[:12000] > 0)  # resampled body, then silence
    assert torch.all(waveform[12000:] == 0)


def test_long_reference_is_truncated_to_6s():
    samples_7s = torch.arange(7 * TARGET, dtype=torch.float32)
    waveform, sr = _normalize_reference(samples_7s, TARGET)
    assert sr == TARGET
    assert waveform.shape == (MAX_SAMPLES,)
    assert torch.equal(waveform, samples_7s[:MAX_SAMPLES])


def test_short_reference_is_zero_padded_to_6s():
    """The zero-pad is the load-bearing half of S1b.

    Truncation alone only caps long references; short ones would keep
    distinct L0 values (verified experimentally: rtf 3.375 with truncation
    only, 1.2 with zero-padding). Pin the padding behavior.
    """
    samples_1s = torch.arange(TARGET, dtype=torch.float32)
    waveform, sr = _normalize_reference(samples_1s, TARGET)
    assert sr == TARGET
    assert waveform.shape == (MAX_SAMPLES,)
    assert torch.equal(waveform[:TARGET], samples_1s)
    assert torch.all(waveform[TARGET:] == 0)


def test_exact_length_reference_passes_through_unchanged():
    samples_6s = torch.arange(MAX_SAMPLES, dtype=torch.float32)
    waveform, sr = _normalize_reference(samples_6s, TARGET)
    assert sr == TARGET
    assert torch.equal(waveform, samples_6s)


def test_every_output_shares_one_length():
    """Whatever comes in, one request's L0 equals any other's."""
    for length in (TARGET // 4, TARGET, 3 * TARGET, MAX_SAMPLES + 1):
        waveform, _ = _normalize_reference(torch.ones(length), TARGET)
        assert waveform.numel() == MAX_SAMPLES


def test_empty_reference_stays_empty():
    """An empty waveform must not be padded into a valid-length reference.

    ``_materialize_runtime_prompt`` relies on numel() == 0 to raise
    "empty_ref_audio" and reject the request; zero-padding would turn a
    broken input into a silently accepted all-silence reference.
    """
    waveform, sr = _normalize_reference(torch.zeros(0), TARGET)
    assert waveform.numel() == 0
    assert sr == TARGET

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared HiFT must retain its native path unless explicitly opted in."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core import hifigan
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_shared_hift_keeps_native_istft_without_opt_in(monkeypatch):
    n_fft, hop = 16, 4
    window = torch.hann_window(n_fft)
    hift = SimpleNamespace(
        istft_params={"n_fft": n_fft, "hop_len": hop},
        _get_stft_window=lambda tensor: window,
    )

    def unexpected_cache(*args):
        raise AssertionError("shared HiFT must not opt in implicitly")

    monkeypatch.setattr(hifigan, "_istft_without_host_sync", unexpected_cache)
    magnitude, phase = torch.rand(2, 9, 20), torch.randn(2, 9, 20)
    assert HiFTGenerator._istft(hift, magnitude, phase).shape == (2, 76)
    assert not hasattr(hift, "_istft_envelopes")

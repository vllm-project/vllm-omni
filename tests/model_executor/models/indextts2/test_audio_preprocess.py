# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm_omni.model_executor.models.indextts2 import preprocess_utils


def test_reference_audio_speaker_mode_matches_official_resample_order(monkeypatch):
    calls: list[tuple[int, int, int]] = []

    def fake_resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        calls.append((orig_sr, target_sr, wav.numel()))
        if orig_sr == target_sr:
            return wav
        out_len = int(round(wav.numel() * target_sr / orig_sr))
        return torch.zeros(out_len)

    monkeypatch.setattr(preprocess_utils, "_resample", fake_resample)

    wav = torch.zeros(20 * 48000)
    wav_16k, wav_22k = preprocess_utils.load_reference_audio((wav, 48000), torch.device("cpu"), mode="speaker")

    assert calls == [
        (48000, 22050, 20 * 48000),
        (22050, 16000, 15 * 22050),
    ]
    assert wav_22k.numel() == 15 * 22050
    assert wav_16k.numel() == 15 * 16000


def test_reference_audio_emotion_mode_matches_official_resample_order(monkeypatch):
    calls: list[tuple[int, int, int]] = []

    def fake_resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        calls.append((orig_sr, target_sr, wav.numel()))
        if orig_sr == target_sr:
            return wav
        out_len = int(round(wav.numel() * target_sr / orig_sr))
        return torch.zeros(out_len)

    monkeypatch.setattr(preprocess_utils, "_resample", fake_resample)

    wav = torch.zeros(20 * 48000)
    wav_16k, wav_22k = preprocess_utils.load_reference_audio((wav, 48000), torch.device("cpu"), mode="emotion")

    assert calls == [
        (48000, 16000, 20 * 48000),
        (16000, 22050, 15 * 16000),
    ]
    assert wav_16k.numel() == 15 * 16000
    assert wav_22k.numel() == int(round(15 * 16000 * 22050 / 16000))

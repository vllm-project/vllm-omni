# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in memory switches: S3Tokenizer placement in StepAudio2Token2WavCore."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav as code2wav_module
import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav as token2wav_module
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav
from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import StepAudio2Token2WavCore

pytestmark = [pytest.mark.core_model]


def _code2wav_config(tmp_path: Path, extra: dict) -> SimpleNamespace:
    (tmp_path / "assets" / "token2wav").mkdir(parents=True)
    sf.write(tmp_path / "assets" / "HT_ref_audio.wav", np.zeros(1600, dtype=np.float32), 16000)
    return SimpleNamespace(model_config=SimpleNamespace(model=str(tmp_path), stage_connector_config={"extra": extra}))


@pytest.mark.cpu
@pytest.mark.parametrize(("extra", "expected"), [({}, None), ({"token2wav_s3tokenizer_device": "cpu"}, "cpu")])
def test_code2wav_passes_tokenizer_device_only_when_set(monkeypatch, tmp_path, extra, expected):
    seen: list[dict] = []

    class _FakeToken2wav:
        def __init__(self, path, **kwargs):
            seen.append(kwargs)

    monkeypatch.setattr(token2wav_module, "MiniCPMO45Token2wav", _FakeToken2wav)
    monkeypatch.setattr(code2wav_module, "BatchedToken2Wav", lambda token2wav, **_: SimpleNamespace())
    model = MiniCPMO45Code2Wav(vllm_config=_code2wav_config(tmp_path, extra))
    model._build_backend()

    assert len(seen) == 1
    if expected is None:
        assert "audio_tokenizer_device" not in seen[0]
    else:
        assert seen[0]["audio_tokenizer_device"] == expected


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.devices: list[torch.device] = []

    def quantize(self, mels: torch.Tensor, mels_lens: torch.Tensor):
        self.devices.append(mels.device)
        tokens = torch.arange(4, device=mels.device).unsqueeze(0)
        return tokens, torch.tensor([4], device=mels.device)


def _core(tmp_path: Path, device: str, tokenizer_device: str | None):
    core = StepAudio2Token2WavCore(str(tmp_path), device=device, audio_tokenizer_device=tokenizer_device)
    tokenizer = _RecordingTokenizer()
    core._audio_tokenizer = tokenizer
    core._spk_model = lambda feat: torch.zeros(1, 192)
    core._flow = SimpleNamespace(up_rate=20)
    core._hift = None
    core._models_loaded = True
    wav = tmp_path / "ref.wav"
    sf.write(wav, (0.1 * np.sin(np.linspace(0, 200, 16000))).astype(np.float32), 16000)
    return core, tokenizer, str(wav)


@pytest.mark.cpu
def test_core_tokenizer_device_defaults_to_vocoder_device(tmp_path):
    core, tokenizer, wav = _core(tmp_path, "cpu", None)
    assert core.audio_tokenizer_device == core.device
    tokens, lens, *_ = core._prepare_prompt(wav)
    assert tokenizer.devices == [torch.device("cpu")]
    assert tokens.device == torch.device("cpu")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_core_cpu_tokenizer_returns_tokens_on_vocoder_device(tmp_path):
    core, tokenizer, wav = _core(tmp_path, "cuda", "cpu")
    tokens, lens, spk, mels, mels_lens = core._prepare_prompt(wav)
    assert tokenizer.devices == [torch.device("cpu")]
    assert tokens.device.type == "cuda" and lens.device.type == "cuda"
    assert mels.device.type == "cuda"

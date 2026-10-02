# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in memory switches: S3Tokenizer placement and the session-sized S0 audio graph grid.

Both are off by default and must leave the default path unchanged.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav as code2wav_module
import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm as omni_llm_module
import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav as token2wav_module
import vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder_graph as graph_module
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder_graph import batch_sizes_for_sessions
from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import StepAudio2Token2WavCore

pytestmark = [pytest.mark.core_model]


# --- S0 audio graph grid ---------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("sessions", "grid"),
    [
        (1, (1,)),
        (2, (1, 2)),
        (4, (1, 2, 4)),
        (12, (1, 2, 4, 8, 12)),
        (16, (1, 2, 4, 8, 16)),
        (20, (1, 2, 4, 8, 16, 20)),
        (32, (1, 2, 4, 8, 16, 32)),
        (0, (1,)),
    ],
)
def test_batch_sizes_for_sessions(sessions, grid):
    assert batch_sizes_for_sessions(sessions) == grid


class _FakeGraphEncoder:
    built: list[dict] = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.batch_sizes = tuple(kwargs["batch_sizes"])
        self.cache_buckets = tuple(kwargs["cache_buckets"])
        self.pinned_h2d = kwargs["pinned_h2d"]
        _FakeGraphEncoder.built.append(kwargs)

    def capture(self) -> None:
        pass


def _thinker(config: SimpleNamespace, max_sessions: int):
    thinker = MiniCPMO45OmniLLMForConditionalGeneration.__new__(MiniCPMO45OmniLLMForConditionalGeneration)
    torch.nn.Module.__init__(thinker)
    weight = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    object.__setattr__(thinker, "config", config)
    object.__setattr__(thinker, "apm", SimpleNamespace(conv1=SimpleNamespace(weight=weight)))
    object.__setattr__(thinker, "audio_projection_layer", None)
    object.__setattr__(thinker, "audio_avg_pooler", None)
    object.__setattr__(thinker, "_duplex_max_sessions", max_sessions)
    object.__setattr__(thinker, "supports_streaming_audio_batch", lambda: True)
    return thinker


def _build(monkeypatch, config: SimpleNamespace, max_sessions: int) -> dict:
    _FakeGraphEncoder.built.clear()
    monkeypatch.setattr(graph_module, "StreamingAudioGraphEncoder", _FakeGraphEncoder)
    monkeypatch.setattr(omni_llm_module.torch.cuda, "memory_allocated", lambda *_: 0)
    thinker = _thinker(config, max_sessions)
    assert thinker.build_streaming_audio_graph_encoder(unit_frames=104) is True
    assert len(_FakeGraphEncoder.built) == 1
    return _FakeGraphEncoder.built[0]


@pytest.mark.cpu
def test_grid_default_and_explicit_list_unchanged(monkeypatch):
    kwargs = _build(monkeypatch, SimpleNamespace(audio_pool_step=5), max_sessions=16)
    assert tuple(kwargs["batch_sizes"]) == graph_module.DEFAULT_GRAPH_BATCH_SIZES

    explicit = SimpleNamespace(audio_pool_step=5, duplex_audio_encoder_cuda_graph_batch_sizes=[1, 2, 4, 8, 16, 20])
    kwargs = _build(monkeypatch, explicit, max_sessions=16)
    assert kwargs["batch_sizes"] == [1, 2, 4, 8, 16, 20]


@pytest.mark.cpu
@pytest.mark.parametrize(("sessions", "grid"), [(16, [1, 2, 4, 8, 16]), (20, [1, 2, 4, 8, 16, 20])])
def test_grid_follows_max_sessions_when_enabled(monkeypatch, sessions, grid):
    config = SimpleNamespace(
        audio_pool_step=5,
        duplex_audio_encoder_cuda_graph_batch_sizes=[1, 2, 4, 8, 16, 20],
        duplex_audio_encoder_cuda_graph_batch_sizes_from_sessions=True,
    )
    kwargs = _build(monkeypatch, config, max_sessions=sessions)
    assert kwargs["batch_sizes"] == grid


@pytest.mark.cpu
def test_grid_flag_needs_literal_true(monkeypatch):
    config = SimpleNamespace(audio_pool_step=5, duplex_audio_encoder_cuda_graph_batch_sizes_from_sessions="false")
    kwargs = _build(monkeypatch, config, max_sessions=4)
    assert tuple(kwargs["batch_sizes"]) == graph_module.DEFAULT_GRAPH_BATCH_SIZES


# --- S3Tokenizer placement ----------------------------------------------------


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

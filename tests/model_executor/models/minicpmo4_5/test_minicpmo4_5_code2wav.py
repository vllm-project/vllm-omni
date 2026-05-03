# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import soundfile as sf
import torch

import vllm_omni.model_executor.models.minicpmo4_5.minicpmo4_5_code2wav as code2wav_mod
from vllm_omni.model_executor.models.minicpmo4_5.minicpmo4_5_code2wav import MiniCPMO4_5Code2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _minimal_model():
    model = object.__new__(MiniCPMO4_5Code2Wav)
    model.config = SimpleNamespace(s3_stream_n_timesteps=10)
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=False))
    model.model_path = "/tmp/nonexistent-minicpm-model"
    model._token2wav = None
    model._output_sample_rate = 24000
    model._audio_prompt_sample_rate = 16000
    model._audio_eos_token_id = 6561
    return model


def _wav_bytes(waveform, sr=24000):
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    return buf.getvalue()


def test_forward_splits_batched_requests_and_decodes_independently(monkeypatch):
    model = _minimal_model()
    monkeypatch.setattr(model, "_ensure_token2wav_loaded", lambda: None)

    calls = []

    def fake_decode(token_ids, ref_audio):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        calls.append((token_ids, ref_audio))
        idx = len(calls)
        return torch.tensor([float(idx)], dtype=torch.float32), torch.tensor(24000 + idx, dtype=torch.int32)

    monkeypatch.setattr(model, "_decode_one", fake_decode)

    out = model.forward(
        input_ids=torch.tensor([11, 12, 21, 22, 23], dtype=torch.long),
        seq_token_counts=[2, 3],
        model_intermediate_buffer=[
            {"ref_audio": {"wav": [0.1, -0.1], "sr": 16000}},
            {},
        ],
    )

    assert calls == [
        ([11, 12], {"wav": [0.1, -0.1], "sr": 16000}),
        ([21, 22, 23], None),
    ]
    assert out.multimodal_outputs["model_outputs"][0].tolist() == [1.0]
    assert out.multimodal_outputs["model_outputs"][1].tolist() == [2.0]
    assert [int(x.item()) for x in out.multimodal_outputs["sr"]] == [24001, 24002]


def test_decode_one_writes_prompt_wav_and_strips_trailing_eos(tmp_path):
    model = _minimal_model()
    observed = {}

    def fake_token2wav(token_ids, prompt_wav_path):
        observed["token_ids"] = token_ids
        observed["prompt_wav_path"] = prompt_wav_path
        assert prompt_wav_path is not None
        assert Path(prompt_wav_path).exists()
        return _wav_bytes([0.25, -0.25], sr=22050)

    model._token2wav = fake_token2wav

    audio, sr = model._decode_one(
        [5, 6, model._audio_eos_token_id, model._audio_eos_token_id],
        {"wav": [0.0, 0.1, -0.1], "sr": 16000},
    )

    assert observed["token_ids"] == [5, 6]
    assert observed["prompt_wav_path"] is not None
    assert audio.dtype == torch.float32
    assert audio.shape == (2,)
    assert int(sr.item()) == 22050


def test_decode_one_uses_silence_prompt_when_ref_audio_is_missing():
    model = _minimal_model()
    observed = {}

    def fake_token2wav(token_ids, prompt_wav_path):
        observed["token_ids"] = token_ids
        observed["prompt_wav_path"] = prompt_wav_path
        assert prompt_wav_path is not None
        assert Path(prompt_wav_path).exists()
        return _wav_bytes([0.1, 0.2, 0.3], sr=16000)

    model._token2wav = fake_token2wav

    audio, sr = model._decode_one([7], None)

    assert observed["token_ids"] == [7]
    assert observed["prompt_wav_path"] is not None
    assert audio.shape == (3,)
    assert int(sr.item()) == 16000


def test_ensure_token2wav_loaded_requires_local_assets(tmp_path, monkeypatch):
    model = _minimal_model()
    model.model_path = str(tmp_path)

    with pytest.raises(FileNotFoundError, match="token2wav assets"):
        model._ensure_token2wav_loaded()


def test_ensure_token2wav_loaded_constructs_local_core(tmp_path, monkeypatch):
    model = _minimal_model()
    model.model_path = str(tmp_path)
    asset_dir = tmp_path / "assets" / "token2wav"
    asset_dir.mkdir(parents=True)

    observed = {}

    class DummyToken2wav:
        def __init__(self, model_path, *, float16, n_timesteps, device):
            observed["model_path"] = model_path
            observed["float16"] = float16
            observed["n_timesteps"] = n_timesteps
            observed["device"] = device

    monkeypatch.setattr(code2wav_mod, "MiniCPMToken2wavCore", DummyToken2wav)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    model._ensure_token2wav_loaded()

    assert isinstance(model._token2wav, DummyToken2wav)
    assert observed == {
        "model_path": str(asset_dir),
        "float16": False,
        "n_timesteps": 10,
        "device": "cpu",
    }


def test_normalize_ref_audio_rejects_non_canonical_payload():
    model = _minimal_model()

    with pytest.raises(TypeError, match="canonical ref_audio dict"):
        model._normalize_ref_audio(([0.1, 0.2], 16000))


def test_decode_one_returns_empty_audio_when_only_eos_tokens():
    model = _minimal_model()

    audio, sr = model._decode_one([model._audio_eos_token_id, model._audio_eos_token_id], None)

    assert audio.dtype == torch.float32
    assert audio.numel() == 0
    assert int(sr.item()) == 24000


def test_forward_empty_input_returns_empty_audio_without_loading_token2wav(monkeypatch):
    model = _minimal_model()

    def fail_if_loaded():
        raise AssertionError("Token2wav should not load when there are no audio tokens")

    monkeypatch.setattr(model, "_ensure_token2wav_loaded", fail_if_loaded)

    out = model.forward(
        input_ids=torch.empty((0,), dtype=torch.long),
        model_intermediate_buffer=None,
    )

    assert [audio.numel() for audio in out.multimodal_outputs["model_outputs"]] == [0]
    assert [int(sr.item()) for sr in out.multimodal_outputs["sr"]] == [24000]

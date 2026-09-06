# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.breeze_tts_2.audio_tokenizer import (
    BreezeReferenceAudioTokenizer,
    resolve_audio_tokenizer_path,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Tokenizer:
    def encode(self, audio, sr=None):
        assert isinstance(audio, np.ndarray)
        assert sr == 16000
        return {"audio_codes": [torch.arange(32).reshape(16, 2)]}


def test_reference_audio_tokenizer_normalizes_codebook_major_output():
    adapter = BreezeReferenceAudioTokenizer(_Tokenizer())

    codes = adapter.encode(np.zeros(32, dtype=np.float32), 16000)

    assert tuple(codes.shape) == (2, 16)
    assert codes.dtype == torch.int16
    assert codes.device.type == "cpu"


def test_reference_audio_tokenizer_rejects_waveform_without_sample_rate():
    adapter = BreezeReferenceAudioTokenizer(_Tokenizer())

    try:
        adapter.encode(np.zeros(32, dtype=np.float32))
    except ValueError as exc:
        assert "sample_rate is required" in str(exc)
    else:
        raise AssertionError("missing sample rate should be rejected")


def test_resolve_prefers_local_directory_without_hub_access(tmp_path, monkeypatch):
    bundled = tmp_path / "audio_tokenizer"
    bundled.mkdir()

    def _no_hub(*_args, **_kwargs):
        raise AssertionError("local directories must not touch the hub")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _no_hub)

    assert resolve_audio_tokenizer_path(str(tmp_path)) == bundled


def test_resolve_local_directory_without_bundled_tokenizer_returns_none(tmp_path, monkeypatch):
    def _no_hub(*_args, **_kwargs):
        raise AssertionError("local directories must not touch the hub")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _no_hub)

    assert resolve_audio_tokenizer_path(str(tmp_path)) is None


def test_resolve_repo_id_uses_hf_snapshot(tmp_path, monkeypatch):
    snapshot_root = tmp_path / "snapshot"
    (snapshot_root / "audio_tokenizer").mkdir(parents=True)
    calls = []

    def _fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(snapshot_root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)

    resolved = resolve_audio_tokenizer_path("BreezeBlue/Breeze-TTS-2")

    assert resolved == snapshot_root / "audio_tokenizer"
    assert calls == [("BreezeBlue/Breeze-TTS-2", {"allow_patterns": ["audio_tokenizer/*"]})]


def test_resolve_repo_id_without_bundled_tokenizer_returns_none(tmp_path, monkeypatch):
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_args, **_kwargs: str(snapshot_root),
    )

    assert resolve_audio_tokenizer_path("BreezeBlue/Breeze-TTS-2") is None

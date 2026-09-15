# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.breeze_tts_2.audio_tokenizer import (
    BreezeReferenceAudioTokenizer,
    resolve_audio_tokenizer_path,
)


def _patch_hf_api(monkeypatch, snapshot_download):
    """Point the resolver's tagged HF helper at a fake API object."""
    from vllm.transformers_utils import repo_utils

    monkeypatch.setattr(repo_utils, "hf_api", lambda: SimpleNamespace(snapshot_download=snapshot_download))


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

    _patch_hf_api(monkeypatch, _no_hub)

    assert resolve_audio_tokenizer_path(str(tmp_path)) == bundled


def test_resolve_local_directory_without_bundled_tokenizer_returns_none(tmp_path, monkeypatch):
    def _no_hub(*_args, **_kwargs):
        raise AssertionError("local directories must not touch the hub")

    _patch_hf_api(monkeypatch, _no_hub)

    assert resolve_audio_tokenizer_path(str(tmp_path)) is None


def test_resolve_repo_id_uses_hf_snapshot(tmp_path, monkeypatch):
    snapshot_root = tmp_path / "snapshot"
    (snapshot_root / "audio_tokenizer").mkdir(parents=True)
    calls = []

    def _fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("cold cache")
        return str(snapshot_root)

    _patch_hf_api(monkeypatch, _fake_snapshot_download)

    resolved = resolve_audio_tokenizer_path("BreezeBlue/Breeze-TTS-2")

    assert resolved == snapshot_root / "audio_tokenizer"
    # The cold-cache probe runs first; the online fetch carries the pattern.
    assert calls[0] == (
        "BreezeBlue/Breeze-TTS-2",
        {"allow_patterns": ["audio_tokenizer/*"], "local_files_only": True},
    )
    assert calls[1] == ("BreezeBlue/Breeze-TTS-2", {"allow_patterns": ["audio_tokenizer/*"]})


def test_resolve_repo_id_prefers_complete_cached_snapshot(tmp_path, monkeypatch):
    snapshot_root = tmp_path / "snapshot"
    (snapshot_root / "audio_tokenizer").mkdir(parents=True)
    calls = []

    def _fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        assert kwargs.get("local_files_only"), "cached snapshot must not hit the network"
        return str(snapshot_root)

    _patch_hf_api(monkeypatch, _fake_snapshot_download)

    resolved = resolve_audio_tokenizer_path("BreezeBlue/Breeze-TTS-2")

    assert resolved == snapshot_root / "audio_tokenizer"
    assert len(calls) == 1


def test_resolve_repo_id_without_bundled_tokenizer_returns_none(tmp_path, monkeypatch):
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()

    def _fake_snapshot_download(repo_id, **kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("cold cache")
        return str(snapshot_root)

    _patch_hf_api(monkeypatch, _fake_snapshot_download)

    assert resolve_audio_tokenizer_path("BreezeBlue/Breeze-TTS-2") is None

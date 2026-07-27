# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiniCPM-o 4.5 code2wav model-dir resolution (#5442).

In hub/CI deployments ``model_config.model`` is a repo id rather than a local
directory, so asset lookups must resolve to the downloaded snapshot. The
resolution must stay lazy: constructing the model with a fake path (as the
CPU unit tests do) must not touch the hub.
"""

from __future__ import annotations

from types import SimpleNamespace

import huggingface_hub
import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
    _resolve_model_dir,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _no_hub(monkeypatch):
    def _fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("snapshot_download must not be called here")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fail)


def test_local_directory_is_returned_unchanged(tmp_path, monkeypatch):
    _no_hub(monkeypatch)
    assert _resolve_model_dir(str(tmp_path)) == str(tmp_path)


def test_repo_id_resolves_via_snapshot_download(tmp_path, monkeypatch):
    calls = {}

    def _fake_snapshot_download(model_ref, revision=None, allow_patterns=None):
        calls["model_ref"] = model_ref
        calls["revision"] = revision
        calls["allow_patterns"] = allow_patterns
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)
    resolved = _resolve_model_dir("openbmb/MiniCPM-o-4_5", revision="abc123")
    assert resolved == str(tmp_path / "snapshot")
    assert calls["model_ref"] == "openbmb/MiniCPM-o-4_5"
    assert calls["revision"] == "abc123"
    assert calls["allow_patterns"] == ["assets/*"]


def test_snapshot_download_failure_propagates(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("offline and not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _raise)
    with pytest.raises(FileNotFoundError):
        _resolve_model_dir("openbmb/MiniCPM-o-4_5")


def test_init_with_fake_path_does_not_resolve(monkeypatch):
    """Mirrors the CPU-test construction: fake model path, no ``revision``."""
    _no_hub(monkeypatch)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/fake/model",
            stage_connector_config=None,
        )
    )
    model = MiniCPMO45Code2Wav(vllm_config=config)
    assert model.model_path == "/fake/model"
    assert model._default_prompt_wav == "/fake/model/assets/HT_ref_audio.wav"


def test_default_prompt_wav_follows_resolved_model_path(monkeypatch):
    _no_hub(monkeypatch)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="openbmb/MiniCPM-o-4_5",
            stage_connector_config=None,
        )
    )
    model = MiniCPMO45Code2Wav(vllm_config=config)
    model.model_path = "/resolved/snapshot"
    assert model._default_prompt_wav == "/resolved/snapshot/assets/HT_ref_audio.wav"

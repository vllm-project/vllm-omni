# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for MiniCPM-o 4.5 code2wav model-dir resolution (#5442).

In hub/CI deployments ``model_config.model`` is a repo id rather than a local
directory, so asset lookups must resolve to the downloaded snapshot. The
resolution must stay lazy: constructing the model with a fake path (as the
CPU unit tests do) must not touch the hub.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.helpers.mock import patch_hf_snapshot_download
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
    _resolve_model_dir,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _no_hub(monkeypatch, tmp_path):
    def _fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("snapshot_download must not be called here")

    patch_hf_snapshot_download(monkeypatch, _fail, hf_home=tmp_path)


def test_local_directory_is_returned_unchanged(tmp_path, monkeypatch):
    _no_hub(monkeypatch, tmp_path)
    assert _resolve_model_dir(str(tmp_path)) == str(tmp_path)


def test_repo_id_resolves_via_snapshot_download(tmp_path, monkeypatch):
    calls = {}

    def _fake_snapshot_download(model_ref, revision=None, allow_patterns=None):
        calls["model_ref"] = model_ref
        calls["revision"] = revision
        calls["allow_patterns"] = allow_patterns
        return str(tmp_path / "snapshot")

    patch_hf_snapshot_download(monkeypatch, _fake_snapshot_download, hf_home=tmp_path)
    resolved = _resolve_model_dir("openbmb/MiniCPM-o-4_5", revision="abc123")
    assert resolved == str(tmp_path / "snapshot")
    assert calls["model_ref"] == "openbmb/MiniCPM-o-4_5"
    assert calls["revision"] == "abc123"
    assert calls["allow_patterns"] == ["assets/*"]


def test_snapshot_download_failure_propagates(monkeypatch, tmp_path):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("offline and not cached")

    patch_hf_snapshot_download(monkeypatch, _raise, hf_home=tmp_path)
    with pytest.raises(FileNotFoundError):
        _resolve_model_dir("openbmb/MiniCPM-o-4_5")


def test_init_with_fake_path_does_not_resolve(monkeypatch, tmp_path):
    """Mirrors the CPU-test construction: fake model path, no ``revision``."""
    _no_hub(monkeypatch, tmp_path)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/fake/model",
            stage_connector_config=None,
        )
    )
    model = MiniCPMO45Code2Wav(vllm_config=config)
    assert model.model_path == "/fake/model"
    assert model._default_prompt_wav == "/fake/model/assets/HT_ref_audio.wav"


def test_default_prompt_wav_follows_resolved_model_path(monkeypatch, tmp_path):
    _no_hub(monkeypatch, tmp_path)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="openbmb/MiniCPM-o-4_5",
            stage_connector_config=None,
        )
    )
    model = MiniCPMO45Code2Wav(vllm_config=config)
    model.model_path = "/resolved/snapshot"
    assert model._default_prompt_wav == "/resolved/snapshot/assets/HT_ref_audio.wav"


def _config_with_extra(extra: dict, *, max_num_seqs=None):
    scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs) if max_num_seqs is not None else None
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="/fake/model",
            stage_connector_config={"extra": extra},
        ),
        scheduler_config=scheduler_config,
    )


@pytest.mark.parametrize(
    ("extra", "max_num_seqs", "sizes"),
    [
        ({}, None, [1, 2, 4, 8, 16, 32]),
        ({}, 6, [1, 2, 4, 6]),
        ({"hift_graph_capture_batch_sizes": [1, 3]}, None, [1, 3]),
    ],
)
def test_hift_capture_batch_sizes(monkeypatch, tmp_path, extra, max_num_seqs, sizes):
    """By default every vocoder batch up to min(max_num_seqs, 32) rounds up to a captured size."""
    _no_hub(monkeypatch, tmp_path)
    config = _config_with_extra({"enable_hift_graph": True, **extra}, max_num_seqs=max_num_seqs)
    assert MiniCPMO45Code2Wav(vllm_config=config)._hift_graph_config["capture_batch_sizes"] == sizes


@pytest.mark.parametrize(("extra", "frames"), [({}, []), ({"hift_graph_codec_chunk_frames": [25, 75]}, [25, 75])])
def test_hift_graph_codec_chunk_frames_reach_the_connector_config(monkeypatch, tmp_path, extra, frames):
    _no_hub(monkeypatch, tmp_path)
    config = _config_with_extra({"enable_hift_graph": True, **extra})
    assert MiniCPMO45Code2Wav(vllm_config=config)._connector_config["hift_graph_codec_chunk_frames"] == frames

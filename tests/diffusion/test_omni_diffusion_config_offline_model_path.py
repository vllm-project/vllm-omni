# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniDiffusionConfig HF-offline model-path resolution."""

import os

import huggingface_hub
import pytest

import vllm_omni.diffusion.data as diffusion_data
from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_LOCAL_SNAPSHOT = "/root/.cache/huggingface/hub/models--foo--bar/snapshots/deadbeef"


@pytest.fixture
def record_get_model_path(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str | None]] = []

    def fake_get_model_path(model, revision=None):
        calls.append((model, revision))
        if os.path.exists(model):
            return model
        return _LOCAL_SNAPSHOT

    monkeypatch.setattr(diffusion_data, "get_model_path", fake_get_model_path)
    return calls


class TestOmniDiffusionConfigOfflineModelPath:
    def test_offline_repo_id_replaced_with_local_path(
        self, monkeypatch: pytest.MonkeyPatch, record_get_model_path
    ) -> None:
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
        config = OmniDiffusionConfig(model="foo/bar")
        assert config.model == _LOCAL_SNAPSHOT
        assert record_get_model_path == [("foo/bar", None)]

    def test_online_leaves_model_untouched(self, monkeypatch: pytest.MonkeyPatch, record_get_model_path) -> None:
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
        config = OmniDiffusionConfig(model="foo/bar")
        assert config.model == "foo/bar"
        assert record_get_model_path == []

    def test_offline_local_path_unchanged(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
        local = str(tmp_path)
        config = OmniDiffusionConfig(model=local)
        assert config.model == local

    @pytest.mark.parametrize("model", [None, ""])
    def test_offline_empty_model_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, record_get_model_path, model: str | None
    ) -> None:
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
        config = OmniDiffusionConfig(model=model)
        assert config.model == model
        assert record_get_model_path == []

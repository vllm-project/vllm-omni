# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for stateless Pi-family pipeline helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.pi.common import pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("value,expected", [(None, None), (1, 1), (25, 25), (np.int64(3), 3)])
def test_resolve_num_inference_steps_accepts_positive_top_level_integers(value, expected):
    params = SimpleNamespace(num_inference_steps=value, extra_args={"num_inference_steps": 999})

    assert pipeline.resolve_num_inference_steps(params) == expected


def test_resolve_num_inference_steps_ignores_extra_args():
    params = SimpleNamespace(extra_args={"num_inference_steps": 7})

    assert pipeline.resolve_num_inference_steps(params) is None


@pytest.mark.parametrize("value", [0, -1, 2.5, True, "4"])
def test_resolve_num_inference_steps_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="num_inference_steps must be a positive integer"):
        pipeline.resolve_num_inference_steps(SimpleNamespace(num_inference_steps=value))


def test_identity_post_process_factory_returns_picklable_module_function():
    post_process = pipeline.get_identity_post_process_func(SimpleNamespace())
    value = object()

    assert post_process is pipeline.identity_post_process
    assert post_process(value) is value


def test_resolve_model_dir_preserves_none_and_local_directory(tmp_path):
    assert pipeline.resolve_model_dir(None) is None
    assert pipeline.resolve_model_dir(str(tmp_path)) == str(tmp_path)


def test_resolve_model_dir_downloads_only_checkpoint_artifacts(monkeypatch):
    observed = {}

    class FakeApi:
        def snapshot_download(self, **kwargs):
            observed.update(kwargs)
            return "/downloaded/model"

    monkeypatch.setattr("vllm_omni.transformers_utils.repo_utils.hf_api", lambda: FakeApi())

    assert pipeline.resolve_model_dir("owner/model") == "/downloaded/model"
    assert observed == {
        "repo_id": "owner/model",
        "allow_patterns": ["*.json", "*.safetensors", "*.model", "tokenizer*"],
    }


def test_resolve_tokenizer_source_prefers_checkpoint_metadata(tmp_path):
    fallback = "google/paligemma-3b-pt-224"
    assert pipeline.resolve_tokenizer_source(str(tmp_path), fallback) == fallback

    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    assert pipeline.resolve_tokenizer_source(str(tmp_path), fallback) == str(tmp_path)


def test_has_safetensors_checkpoint(tmp_path):
    assert not pipeline.has_safetensors_checkpoint(None)
    assert not pipeline.has_safetensors_checkpoint(str(tmp_path))

    (tmp_path / "model.safetensors").touch()

    assert pipeline.has_safetensors_checkpoint(str(tmp_path))


def test_resolve_device_uses_worker_local_device(monkeypatch):
    expected = torch.device("cpu")
    monkeypatch.setattr("vllm_omni.diffusion.distributed.utils.get_local_device", lambda: expected)

    assert pipeline.resolve_device() == expected


def test_resolve_device_falls_back_when_worker_context_is_absent(monkeypatch):
    def unavailable():
        raise RuntimeError("worker context unavailable")

    monkeypatch.setattr("vllm_omni.diffusion.distributed.utils.get_local_device", unavailable)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert pipeline.resolve_device() == torch.device("cpu")

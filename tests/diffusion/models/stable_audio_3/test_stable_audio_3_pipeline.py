# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import math
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import resolve_model_class_name
from vllm_omni.diffusion.models.stable_audio_3.pipeline_stable_audio_3 import (
    StableAudio3Pipeline,
    resolve_checkpoint_root,
    seeded_global_rng,
)
from vllm_omni.diffusion.utils.hf_utils import resolve_native_diffusion_model_class
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "window",
    [
        {"audio_start_in_s": 5.0, "audio_end_in_s": 5.0},  # zero length
        {"audio_start_in_s": 5.0, "audio_end_in_s": 1.0},  # end before start
        {"audio_start_in_s": 0.0, "audio_end_in_s": 1e-6},  # shorter than one sample
        {"audio_start_in_s": 0.0, "audio_end_in_s": math.nan},
        {"audio_start_in_s": 0.0, "audio_end_in_s": math.inf},
        {"audio_start_in_s": 0.0, "audio_end_in_s": 381.0},
        {"audio_start_in_s": -1.0},
        {"audio_start_in_s": math.nan},
    ],
)
def test_forward_rejects_invalid_audio_window(window):
    # Validation runs before any model component is touched, so a bare stub is enough;
    # if it is skipped, forward reaches the stub's missing attributes and raises AttributeError.
    pipeline = SimpleNamespace(max_audio_seconds=StableAudio3Pipeline.max_audio_seconds, sample_rate=44100)
    req = SimpleNamespace(sampling_params=OmniDiffusionSamplingParams(extra_args=window), prompts=["rain"])
    with pytest.raises(ValueError):
        StableAudio3Pipeline.forward(pipeline, req)


def test_decode_noise_follows_the_request_seed_and_leaves_the_global_rng():
    """The SAME decoder draws noise from the global RNG; it must follow the request generator."""
    state = torch.random.get_rng_state()
    draws = []
    for seed in (7, 7, 8):
        with seeded_global_rng(torch.Generator().manual_seed(seed), torch.device("cpu")):
            draws.append(torch.randn(4))
    assert torch.equal(draws[0], draws[1]) and not torch.equal(draws[0], draws[2])
    assert torch.equal(torch.random.get_rng_state(), state)


def _write_model_config(directory, conditioner_type):
    config = {"model": {"conditioning": {"configs": [{"id": "prompt", "type": conditioner_type, "config": {}}]}}}
    (directory / "model_config.json").write_text(json.dumps(config))


_ACCEPTED_HUB_IDS = [
    "stabilityai/stable-audio-3-medium",
    "StabilityAI/Stable-Audio-3-Medium/",
    " stabilityai/stable-audio-3-medium ",
]


@pytest.fixture
def snapshot_calls(tmp_path, monkeypatch):
    calls = []

    class _FakeApi:
        def snapshot_download(self, *, repo_id, revision):
            calls.append((repo_id, revision))
            return str(tmp_path / "snapshot")

    monkeypatch.setattr("vllm_omni.transformers_utils.repo_utils.hf_api", lambda: _FakeApi())
    return calls


@pytest.mark.parametrize("model", _ACCEPTED_HUB_IDS)
def test_bare_hub_id_resolves_and_loads_without_index_files(model, snapshot_calls, tmp_path):
    """The Hub repo has no model_index.json / config.json (#11018); every accepted
    spelling must resolve and download the canonical repo at the requested revision."""
    assert resolve_native_diffusion_model_class(model) == "StableAudio3Pipeline"
    assert resolve_model_class_name(model) == "StableAudio3Pipeline"
    assert resolve_checkpoint_root(model, "abc123") == str(tmp_path / "snapshot")
    assert snapshot_calls == [("stabilityai/stable-audio-3-medium", "abc123")]


def test_local_native_checkpoint_is_detected_by_its_t5gemma_conditioner(tmp_path):
    _write_model_config(tmp_path, "t5gemma")
    assert resolve_native_diffusion_model_class(str(tmp_path)) == "StableAudio3Pipeline"


def test_other_native_stability_checkpoints_are_not_claimed(tmp_path):
    # Stable Audio Open also ships model_config.json, but conditions on T5.
    _write_model_config(tmp_path, "t5")
    assert resolve_native_diffusion_model_class(str(tmp_path)) is None
    (tmp_path / "model_config.json").write_text("{not json")
    assert resolve_native_diffusion_model_class(str(tmp_path)) is None
    assert resolve_native_diffusion_model_class("stabilityai/stable-audio-open-1.0") is None


def test_checkpoint_root_keeps_local_dirs_and_other_ids_unchanged(snapshot_calls, tmp_path):
    local = str(tmp_path) + "/"
    assert resolve_checkpoint_root(local) == local
    assert snapshot_calls == []
    # A fork selected with --model-class-name is passed through as given.
    resolve_checkpoint_root("someone/sa3-finetune", None)
    assert snapshot_calls == [("someone/sa3-finetune", None)]

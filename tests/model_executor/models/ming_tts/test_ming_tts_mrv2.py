# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming TTS on Model Runner V2: default profiles, request-id contract, state release."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.ming_tts import ming_tts as ming_tts_module
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_DECODE_STEP,
    KEY_LATENT_HISTORY,
    KEY_NEXT_EMBEDS,
    KEY_REQUEST_ID,
)
from vllm_omni.model_executor.models.ming_tts.ming_tts import MingTTSForConditionalGeneration
from vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae import MingAudioVAEModel
from vllm_omni.model_executor.models.ming_tts.pipeline import MING_TTS_MOE_PIPELINE, MING_TTS_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

_PROFILES = [("ming_tts.yaml", MING_TTS_PIPELINE), ("ming_tts_moe.yaml", MING_TTS_MOE_PIPELINE)]


def _resolve_stages(monkeypatch, deploy_name, pipeline, platform):
    # Resolve the platform's overrides regardless of the host running the test.
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(deploy_name)), platform=platform)
    return merge_pipeline_deploy(pipeline, config)


@pytest.mark.parametrize("deploy_name, pipeline", _PROFILES)
def test_ming_tts_cuda_profiles_run_both_stages_on_native_mrv2(monkeypatch, deploy_name, pipeline):
    stages = _resolve_stages(monkeypatch, deploy_name, pipeline, "cuda")
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [True, True]
    assert [s.yaml_engine_args["supports_native_mrv2_data_plane"] for s in stages] == [True, True]
    # Stage-0 decode state is not in the token stream, so recompute must fail loudly.
    assert stages[0].yaml_engine_args["recompute_preemption"] == "fail"


@pytest.mark.parametrize("platform", ["rocm", "npu", "xpu", "musa"])
@pytest.mark.parametrize("deploy_name, pipeline", _PROFILES)
def test_ming_tts_profiles_keep_non_cuda_on_v1(monkeypatch, deploy_name, pipeline, platform):
    stages = _resolve_stages(monkeypatch, deploy_name, pipeline, platform)
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [False, False]


class _RecordingStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.finished = []

    def on_requests_finished(self, finished_req_ids):
        self.finished.append(set(finished_req_ids))


@pytest.mark.parametrize("model_stage", ["llm", "audio_vae"])
def test_ming_tts_wrapper_declares_runner_hooks(monkeypatch, model_stage):
    inner = _RecordingStage()
    fake_config = SimpleNamespace(from_hf_config=lambda _hf: SimpleNamespace(validate=lambda: None))
    monkeypatch.setattr(ming_tts_module, "MingTTSConfig", fake_config)
    monkeypatch.setattr(ming_tts_module, "init_vllm_registered_model", lambda **_: inner)
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=None, model_stage=model_stage))

    model = MingTTSForConditionalGeneration(vllm_config=vllm_config)

    # Stage 0 forward returns OmniOutput, which MRV2 PIECEWISE capture must unwrap.
    assert getattr(model, "_returns_tuple", False) is (model_stage == "llm")
    # Runners pass request ids, and so match finish notifications, only on request.
    assert getattr(model, "requires_request_ids", False) is (model_stage == "audio_vae")
    model.on_requests_finished({"req-0"})
    assert inner.finished == [{"req-0"}]


class _PendingLLM:
    latent_patch_key = "ming_latent_patch"
    stop_probability_key = "ming_stop_prob"

    def __init__(self, req_id: str):
        self.pending = {
            req_id: {
                self.latent_patch_key: torch.zeros(1, 4, 64),
                KEY_LATENT_HISTORY: torch.ones(1, 8, 64),
                KEY_NEXT_EMBEDS: torch.full((1, 1, 16), 2.0),
                self.stop_probability_key: torch.tensor([0.1]),
            }
        }

    def pop_postprocess_update(self, req_id):
        return self.pending.pop(req_id, {}) if isinstance(req_id, str) else {}


def test_ming_tts_postprocess_finds_pending_update_by_mrv2_req_id():
    # The MRV2 intermediate buffer carries ``req_id``, not V1's ``request_id``.
    # Without it the decode step never advances and requests run to max_tokens.
    model = object.__new__(MingTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "llm"
    model.model = _PendingLLM("req-0")

    # An explicit None under an earlier key must not hide ``req_id``.
    update = model.postprocess(torch.zeros(1, 16), req_id="req-0", request_id=None, **{KEY_DECODE_STEP: 3})

    assert update[KEY_REQUEST_ID] == "req-0"
    assert update[KEY_DECODE_STEP] == 4
    assert torch.equal(update[KEY_LATENT_HISTORY], torch.ones(1, 8, 64))


class _FakeAudioVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def decode(self, latent_patch, *, past_key_values, use_cache, stream_state, last_chunk):
        del past_key_values, use_cache, stream_state, last_chunk
        return torch.ones(latent_patch.shape[1] * 2), ("stream",), ("kv",)


def _vae_stage() -> MingAudioVAEModel:
    vae = object.__new__(MingAudioVAEModel)
    nn.Module.__init__(vae)
    vae.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    vae.ming_config = SimpleNamespace(sample_rate=44100, latent_dim=64, patch_size=4, llm_hidden_size=16)
    vae.audio = _FakeAudioVAE()
    vae._past_key_values, vae._stream_state = {}, {}
    vae._state_access_times = OrderedDict()
    return vae


def test_ming_audio_vae_releases_aborted_request_state():
    # An aborted stream never delivers its final chunk, so the runner's finish
    # notification is the only signal that frees the decoder state.
    vae = _vae_stage()
    chunk = {KEY_REQUEST_ID: "payload-id", "latent": torch.zeros(2, 4, 64), "meta": {"stream_finished": False}}

    vae(model_intermediate_buffer=[chunk, dict(chunk)], request_ids=["runner-a", "runner-b"])
    assert set(vae._past_key_values) == {"runner-a", "runner-b"}

    vae.on_requests_finished({"runner-a"})
    assert set(vae._past_key_values) == set(vae._stream_state) == {"runner-b"}

    # A payload without a runner id could never be freed, so it must fail loudly.
    with pytest.raises(RuntimeError, match="no runner request id"):
        vae(model_intermediate_buffer=[chunk, dict(chunk)], request_ids=["runner-c"])

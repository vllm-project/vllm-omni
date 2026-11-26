# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming TTS on Model Runner V2: default profiles, request-id contract, state release."""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config, merge_pipeline_deploy
from vllm_omni.engine.stage_init_utils import build_engine_args_dict_from_omni_stage_config
from vllm_omni.model_executor.models.common.ming.audio_vae import StreamingLinearUpsample
from vllm_omni.model_executor.models.ming3_tts.batched_vae_decoder import BatchedStreamingVAEDecoder, _StreamState
from vllm_omni.model_executor.models.ming_tts import ming_tts as ming_tts_module
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_DECODE_STEP,
    KEY_LATENT_HISTORY,
    KEY_NEXT_EMBEDS,
    KEY_REQUEST_ID,
    KEY_TEXT_MODE,
    MingTTSConfig,
)
from vllm_omni.model_executor.models.ming_tts.ming_tts import MingTTSForConditionalGeneration, _feedback_tensor
from vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae import MingAudioVAEModel
from vllm_omni.model_executor.models.ming_tts.ming_tts_llm import _flow_host_stops
from vllm_omni.model_executor.models.ming_tts.pipeline import MING_TTS_MOE_PIPELINE, MING_TTS_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

_PROFILES = [("ming_tts.yaml", MING_TTS_PIPELINE), ("ming_tts_moe.yaml", MING_TTS_MOE_PIPELINE)]


@pytest.mark.parametrize("platform", ["cuda", "rocm", "npu", "xpu", "musa"])
@pytest.mark.parametrize("deploy_name, pipeline", _PROFILES)
def test_ming_tts_bundled_profiles_resolve_on_native_mrv2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deploy_name: str, pipeline, platform: str
):
    # Resolve through both the production stage launcher path and legacy merge
    # regardless of the host running the test.
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    (tmp_path / "config.json").write_text("{}")
    model = str(tmp_path)
    deploy_path = get_deploy_config_path(deploy_name)

    resolution = resolve_omni_config(
        model,
        trust_remote_code=None,
        deploy_config_path=deploy_path,
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    assert resolution.pipeline_config is pipeline
    assert [s.model_config.use_v2_model_runner for s in resolution.stage_configs] == [True, True]
    assert [s.model_config.supports_native_mrv2_data_plane for s in resolution.stage_configs] == [True, True]
    # Stage-0 decode state is not in the token stream, so recompute must fail loudly.
    assert resolution.stage_by_id(0).model_config.recompute_preemption == "fail"

    engine_args = [build_engine_args_dict_from_omni_stage_config(stage, model) for stage in resolution.stage_configs]
    assert [args["use_v2_model_runner"] for args in engine_args] == [True, True]
    assert [args["supports_native_mrv2_data_plane"] for args in engine_args] == [True, True]
    assert engine_args[0]["recompute_preemption"] == "fail"

    deploy = _apply_platform_overrides(load_deploy_config(deploy_path), platform=platform)
    legacy_stages = merge_pipeline_deploy(pipeline, deploy)
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in legacy_stages] == [True, True]
    assert [s.yaml_engine_args["supports_native_mrv2_data_plane"] for s in legacy_stages] == [True, True]
    assert legacy_stages[0].yaml_engine_args["recompute_preemption"] == "fail"


@pytest.mark.parametrize("deploy_name, pipeline", _PROFILES)
def test_ming_tts_v1_override_is_rejected(monkeypatch: pytest.MonkeyPatch, deploy_name: str, pipeline):
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    deploy = load_deploy_config(get_deploy_config_path(deploy_name))
    deploy.model_runner = "v1"

    with pytest.raises(ValueError, match="Model Runner V1 is not supported"):
        VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)
    with pytest.raises(ValueError, match="Model Runner V1 is not supported"):
        merge_pipeline_deploy(pipeline, deploy)


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


@pytest.mark.parametrize("patch_size", [1, 4])
def test_flow_host_stops_and_gpu_resident_feedback_support_dense_patch_sizes(patch_size: int):
    history_size = 1 if patch_size == 1 else 32
    hidden = torch.zeros((2, 8), dtype=torch.bfloat16)
    latent = torch.zeros((2, patch_size, 4), dtype=torch.bfloat16)
    embeds = torch.arange(16, dtype=torch.bfloat16).reshape(2, 1, 8)
    stops = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)

    assert _flow_host_stops([([1, 0], hidden, latent, embeds, stops)]) == {1: 0.25, 0: 0.75}
    latent[1, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="Non-finite Ming flow-head output"):
        _flow_host_stops([([1, 0], hidden, latent, embeds, stops)])

    model = object.__new__(MingTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model._resident_feedback = True
    model.model_stage = "llm"
    model.ming_config = MingTTSConfig(
        llm_hidden_size=8, history_patch_size=history_size, latent_dim=4, patch_size=patch_size
    )
    backbone = nn.Module()
    backbone.embed_input_ids = nn.Embedding(4, 8, dtype=torch.bfloat16)
    model.model = backbone

    ids = torch.tensor([2, 1])
    history = torch.ones((history_size, 4))
    flat_embeds = embeds.squeeze(1)
    infos = [
        {KEY_NEXT_EMBEDS: row, KEY_LATENT_HISTORY: history, "req_id": str(i), KEY_DECODE_STEP: i}
        for i, row in enumerate(flat_embeds.flip(0))
    ]
    _, out, updates = model._decode_feedback_batch(input_ids=ids, input_embeds=flat_embeds, req_infos=infos)
    assert torch.equal(out, flat_embeds.flip(0))
    assert updates == [{KEY_REQUEST_ID: str(i), KEY_DECODE_STEP: i} for i in range(2)]
    assert _feedback_tensor(model, history).device == history.device

    # Mixed text-mode cohort falls back cleanly to per-request preprocess.
    infos[0] = {KEY_TEXT_MODE: True, "req_id": "text"}
    scalar = [model.preprocess(ids[i : i + 1], None, **info) for i, info in enumerate(infos)]
    _, out, updates = model._decode_feedback_batch(input_ids=ids, input_embeds=flat_embeds, req_infos=infos)
    assert torch.equal(out, torch.cat([s[1] for s in scalar]))
    assert updates[0] == scalar[0][2]


@pytest.mark.parametrize("patch_size", [1, 4])
def test_batched_vae_orders_chunks_flushes_held_back_frame_and_cleans_up(patch_size: int):
    upsampling = StreamingLinearUpsample(scale_factor=4)
    upsampling.frame_holdback = True
    stub_decoder = nn.Module()
    stub_decoder.fc1 = nn.Identity()
    stub_decoder.register_parameter("weight", nn.Parameter(torch.zeros(1)))
    stub_decoder.upsampling = upsampling
    stub_decoder.head = lambda hidden, **kw: (
        hidden.reshape(-1) + (100.0 if kw["last_chunk"] else 0.0),
        None,
        None,
        None,
    )

    batched = object.__new__(BatchedStreamingVAEDecoder)
    batched.decoder = stub_decoder
    batched._packed = None
    batched._free = [1, 0]
    batched._streams = {}
    batched._forward_rows = lambda states, hiddens: hiddens

    vae = _vae_stage()
    vae.ming_config.patch_size = patch_size
    vae._batched_vae_enabled = True
    vae._batched_vae_decoder = batched

    chunk = {KEY_REQUEST_ID: "p", "latent": torch.ones(2, patch_size, 64), "meta": {"stream_finished": False}}
    end = {KEY_REQUEST_ID: "p", "meta": {"stream_finished": True}}
    result = vae(model_intermediate_buffer=[chunk, chunk, end], request_ids=["a", "b", "a"])
    outs = result.multimodal_outputs["model_outputs"]
    assert outs[0].numel() > 0 and outs[1].numel() > 0
    # Empty terminal flush emits the held-back frame and frees stream "a" while keeping "b" live.
    assert outs[2].numel() > 0 and outs[2].min().item() >= 100.0
    assert set(batched._streams) == {"b"} and isinstance(batched._streams["b"], _StreamState)
    vae.on_requests_finished({"b"})
    assert not batched._streams and sorted(batched._free) == [0, 1]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import resolve_deploy_yaml
from vllm_omni.model_executor.models.lfm3_audio.model import _TIED_DEPTH_HEAD, Lfm3AudioForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_training_export_names_map_to_shared_modules():
    exported = {
        "lfm.model.layers.0.conv.in_proj.weight": "model.language_model.layers.0.conv.in_proj.weight",
        "lfm.model.embed_tokens.weight": "model.language_model.embed_tokens.weight",
        "encoder.downsample.conv_in.weight": "model.audio_encoder.downsample.conv_in.weight",
        "encoder.lfm.lfm.layers.16.feed_forward.w1.weight": "model.audio_encoder.lfm.layers.16.feed_forward.w1.weight",
        "mlp.model.0.weight": "model.audio_projector.layer_norm.weight",
        "mlp.model.1.bias": "model.audio_projector.linear_1.bias",
        "mlp.model.3.weight": "model.audio_projector.linear_2.weight",
        "codebook_embedding.weight": "model.codebook_embedding.weight",
        "depth_linear.weight": "model.depth_decoder.depth_linear.weight",
        "mask_embedding": "model.depth_decoder.mask_embedding",
        "depthformer.model.layers.2.self_attn.q_proj.weight": "model.depth_decoder.model.layers.2.self_attn.q_proj.weight",
        "depth_embeddings.11.embedding.weight": "model.depth_decoder.depth_embeddings.11.embedding.weight",
    }
    weights = [(name, torch.empty(0)) for name in exported]
    mapped = [name for name, _ in Lfm3AudioForConditionalGeneration.hf_to_vllm_mapper.apply(weights)]
    assert mapped == list(exported.values())
    assert _TIED_DEPTH_HEAD.search("model.depth_decoder.depth_embeddings.3.to_logits.weight")
    assert not any(_TIED_DEPTH_HEAD.search(name) for name in mapped)


def test_load_weights_maps_exported_names_and_skips_tied_depth_heads(monkeypatch):
    model = Lfm3AudioForConditionalGeneration.__new__(Lfm3AudioForConditionalGeneration)
    params = {
        "model.language_model.embed_tokens.weight": torch.zeros(2),
        "model.depth_decoder.depth_embeddings.0.embedding.weight": torch.zeros(2),
        "model.depth_decoder.mask_embedding": torch.zeros(2),
    }
    monkeypatch.setattr(model, "named_parameters", lambda: iter(params.items()), raising=False)
    weights = [
        ("lfm.model.embed_tokens.weight", torch.full((2,), 1.0)),
        ("depth_embeddings.0.embedding.weight", torch.full((2,), 2.0)),
        # The export's copy of the tied head must be skipped, not rejected.
        ("depth_embeddings.0.to_logits.weight", torch.full((2,), 9.0)),
        ("mask_embedding", torch.full((2,), 3.0)),
    ]
    assert model.load_weights(weights) == set(params)
    assert [float(p[0]) for p in params.values()] == [1.0, 2.0, 3.0]


def test_audio_pipelines_reuse_codec_and_override_speech_sampling():
    speech, text = OMNI_PIPELINES["lfm3_audio"], OMNI_PIPELINES["lfm3_audio_text"]
    assert speech.model_arch == text.model_arch == "Lfm3AudioForConditionalGeneration"
    assert [stage.model_stage for stage in speech.stages] == ["lfm3_temporal", "lfm3_code2wav"]
    assert speech.stages[1].model_arch == "Lfm3OmniCode2Wav"
    assert [stage.model_stage for stage in text.stages] == ["lfm3_temporal"]

    deploy = resolve_deploy_yaml(get_deploy_config_path("lfm3_audio.yaml"))
    sampling = deploy["stages"][0]["default_sampling_params"]
    assert sampling["extra_args"] == {
        "audio_temperature": 1.0,
        "audio_top_k": 128,
        "audio_repetition_penalty": 1.1,
        "audio_repetition_window": 4096,
    }
    # Inherited from the LFM3-Omni profile.
    assert sampling["seed"] == 42 and deploy["stages"][0]["enforce_eager"] is True
    assert resolve_deploy_yaml(get_deploy_config_path("lfm3_audio_text.yaml"))["pipeline"] == "lfm3_audio_text"

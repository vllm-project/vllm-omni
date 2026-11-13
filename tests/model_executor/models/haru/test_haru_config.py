# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Haru-4B config, registration, pipeline and deploy wiring (CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from transformers import AutoConfig
from vllm.sampling_params import SamplingParams

from tests.model_executor.models.haru.conftest import CHECKPOINT_CONFIG
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.engine.duplex.plugin import load_duplex_plugin, validate_duplex_plugin_sampling
from vllm_omni.model_executor.models.haru.configuration_haru import HaruConfig, HaruSpec
from vllm_omni.model_executor.models.haru.duplex.plugin import HaruDuplexPlugin
from vllm_omni.model_executor.models.haru.pipeline import HARU_PIPELINE
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEPLOY_PATH = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "haru.yaml"


def _encode_audio(audio, _sample_rate, _response_format, _speed):
    return None if audio is None else f"audio-{int(np.asarray(audio).size)}"


def test_checkpoint_config_builds_the_qwen35_backbone_and_the_audio_spec() -> None:
    config = HaruConfig(**CHECKPOINT_CONFIG)
    text = config.get_text_config()
    assert (text.hidden_size, text.num_hidden_layers, text.vocab_size) == (2560, 32, 248320)
    assert text.layer_types[:4] == ["linear_attention"] * 3 + ["full_attention"]

    spec = config.spec
    assert spec.frame_samples == spec.codec.frame_samples == 3200
    assert spec.frame_rate == 7.5
    assert spec.hidden_size == 2560
    assert (spec.pad_token_id, spec.eop_token_id, spec.eos_token_id) == (248076, 248071, 248044)
    assert spec.codec.ratios == (8, 5, 5, 4, 2, 2)
    assert spec.head.guidance_range == (1.0, 2.0)


def test_spec_rejects_codec_ratios_that_do_not_make_one_frame() -> None:
    params = {**CHECKPOINT_CONFIG, "codec": {**CHECKPOINT_CONFIG["codec"], "ratios": [8, 5, 5, 4, 2]}}
    with pytest.raises(ValueError, match="frame_samples"):
        HaruSpec.from_dict(params)


def test_spec_rejects_other_checkpoints() -> None:
    with pytest.raises(ValueError, match="not a Haru checkpoint"):
        HaruSpec.from_dict({**CHECKPOINT_CONFIG, "model_type": "qwen3_5"})


def test_config_and_architecture_are_registered() -> None:
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Qwen3_5ForCausalLMConfig

    import vllm_omni.transformers_utils.configs  # noqa: F401  (registers AutoConfig + MODELS_CONFIG_MAP)

    params = {k: v for k, v in CHECKPOINT_CONFIG.items() if k != "model_type"}
    assert isinstance(AutoConfig.for_model("haru", **params), HaruConfig)
    assert MODELS_CONFIG_MAP["HaruForConditionalGeneration"] is Qwen3_5ForCausalLMConfig
    assert _OMNI_MODELS["HaruForConditionalGeneration"] == ("haru", "haru", "HaruForConditionalGeneration")
    assert OMNI_PIPELINES["haru"] is HARU_PIPELINE


def test_pipeline_binds_the_duplex_plugin() -> None:
    assert HARU_PIPELINE.default_deploy_config_name == DEPLOY_PATH.name
    plugin = load_duplex_plugin(HARU_PIPELINE.duplex_plugin, _encode_audio)
    assert isinstance(plugin, HaruDuplexPlugin)
    assert plugin.plugin_id == "haru"
    validate_duplex_plugin_sampling(plugin, sampling_defaults=(SamplingParams(),))


def test_deploy_is_one_duplex_stage_without_unsupported_engine_options() -> None:
    deploy = load_deploy_config(DEPLOY_PATH)
    (stage,) = merge_pipeline_deploy(HARU_PIPELINE, deploy)

    assert deploy.session_mode == "duplex"
    assert deploy.duplex_session.max_sessions == stage.yaml_engine_args["max_num_seqs"] == 2
    # HaruForConditionalGeneration refuses both at startup (_check_supported).
    assert stage.yaml_engine_args["async_scheduling"] is False
    assert stage.yaml_engine_args["enable_prefix_caching"] is False
    assert stage.yaml_engine_args["skip_tokenizer_init"] is True

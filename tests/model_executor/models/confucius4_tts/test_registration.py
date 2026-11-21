# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from transformers import AutoConfig
from vllm.transformers_utils.config import _CONFIG_REGISTRY

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    _DEPLOY_DIR,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.engine.arg_utils import _register_omni_hf_configs
from vllm_omni.model_executor.models.confucius4_tts.configuration_confucius4_tts import (
    Confucius4Config,
    Confucius4TTSConfig,
)
from vllm_omni.model_executor.models.confucius4_tts.pipeline import (
    CONFUCIUS4_TTS_PIPELINE,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_config_defaults() -> None:
    config = Confucius4TTSConfig()
    assert config.model_type == "confucius4_tts"
    assert config.max_text_tokens == 520
    assert config.max_semantic_tokens == 1520
    assert config.bos_token_id == 8192
    assert config.eos_token_id == 8193
    assert config.output_sample_rate == 22050
    assert config.max_reference_audio_seconds == 30.0


def test_official_config_alias_is_registered() -> None:
    _register_omni_hf_configs()
    config = AutoConfig.for_model("confucius4")
    assert isinstance(config, Confucius4Config)
    assert config.model_type == "confucius4"
    assert config.t2s_checkpoint == "t2s_model.safetensors"
    assert config.s2a_checkpoint == "s2a_model.pt"
    assert config.architectures == [
        "Confucius4TTSTalkerForConditionalGeneration",
        "Confucius4TTSS2ADecoder",
    ]
    assert _CONFIG_REGISTRY["confucius4"] is Confucius4Config


def test_pipeline_and_architectures_are_registered() -> None:
    pipeline = OMNI_PIPELINES["confucius4_tts"]
    assert OMNI_PIPELINES["confucius4"] is pipeline
    assert pipeline.model_arch == ("Confucius4TTSTalkerForConditionalGeneration")
    assert [stage.model_stage for stage in pipeline.stages] == [
        "confucius4_tts_talker",
        "confucius4_tts_decoder",
    ]
    assert "tokenizer" not in pipeline.stages[0].extras
    assert _OMNI_MODELS["Confucius4TTSTalkerForConditionalGeneration"][1] == "confucius4_tts_talker"
    assert _OMNI_MODELS["Confucius4TTSS2ADecoder"][1] == ("confucius4_tts_decoder")


def test_official_local_config_resolves_pipeline(tmp_path) -> None:
    _register_omni_hf_configs()
    (tmp_path / "config.json").write_text(
        '{"model_type": "confucius4"}',
        encoding="utf-8",
    )
    StageConfigFactory.try_infer_model_type.cache_clear()

    pipeline = StageConfigFactory.get_pipeline_config(
        str(tmp_path),
        trust_remote_code=False,
    )

    assert pipeline is CONFUCIUS4_TTS_PIPELINE


def test_packaged_deploy_config_merges_two_stage_pipeline() -> None:
    deploy = load_deploy_config(_DEPLOY_DIR / "confucius4_tts.yaml")
    stages = merge_pipeline_deploy(CONFUCIUS4_TTS_PIPELINE, deploy)

    assert CONFUCIUS4_TTS_PIPELINE.default_deploy_config_name == ("confucius4_tts.yaml")
    assert deploy.pipeline == "confucius4_tts"
    assert deploy.trust_remote_code is False
    assert len(stages) == 2
    assert [stage.yaml_engine_args["model_arch"] for stage in stages] == [
        "Confucius4TTSTalkerForConditionalGeneration",
        "Confucius4TTSS2ADecoder",
    ]
    assert stages[0].yaml_engine_args["custom_process_next_stage_input_func"].endswith(".talker2decoder_full_payload")
    assert stages[1].custom_process_input_func.endswith(".talker2decoder_token_only")
    assert stages[0].yaml_extras["output_connectors"] == {"to_stage_1": "confucius4_tts_shm"}
    assert stages[1].yaml_extras["input_connectors"] == {"from_stage_0": "confucius4_tts_shm"}
    assert "dtype" not in stages[0].yaml_engine_args
    assert all(stage.yaml_engine_args["supports_native_mrv2_data_plane"] for stage in stages)
    assert stages[1].yaml_engine_args["dtype"] == "float32"
    assert stages[0].yaml_extras["default_sampling_params"]["max_tokens"] == 1518


def test_decode_counter_is_a_snapshot_not_concatenated_content() -> None:
    # Stage 0 re-emits ``meta.semantic_code_count`` as a 0-d tensor every step;
    # the default CONCAT_DIM0 would raise when the output processor consolidates.
    from vllm_omni.outputs.output_modality import (
        OutputModality,
        TensorAccumulationStrategy,
        get_accumulation_strategy,
    )

    assert (
        get_accumulation_strategy(OutputModality.LATENT, "meta.semantic_code_count")
        is TensorAccumulationStrategy.REPLACE
    )

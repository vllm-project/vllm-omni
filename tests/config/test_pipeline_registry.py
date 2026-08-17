# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for out of tree registration to OMNI_PIPELINES."""

import pytest
from transformers import PretrainedConfig

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline
from vllm_omni.config.stage_config import (
    DiffusionStageRole,
    PipelineConfig,
    load_deploy_config,
    pipeline_cfg_resolver,
    resolve_diffusion_stage_role,
)
from vllm_omni.diffusion.models.interface import stage_component_groups

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def custom_resolver():
    """Build a reusable custom resolver for PipelineConfigs."""

    class CustomConfigType(PretrainedConfig):
        pass

    @pipeline_cfg_resolver(config_type=CustomConfigType)
    def custom_resolver(
        hf_config: CustomConfigType,
    ) -> PipelineConfig:
        return PipelineConfig(model_type="resolved_type")

    return custom_resolver


def test_register_pipeline_config(clean_pipeline_registry):
    """Ensure that we can register a custom pipeline config to OMNI_PIPELINES."""
    new_model_type = "new_model_type"
    pipe_cfg = PipelineConfig(model_type=new_model_type)
    assert new_model_type not in OMNI_PIPELINES
    register_pipeline(pipe_cfg)
    assert new_model_type in OMNI_PIPELINES
    assert OMNI_PIPELINES[new_model_type] is pipe_cfg


def test_register_pipeline_config_with_model_type(clean_pipeline_registry):
    """Ensure that we can register a custom pipeline config with an explicit model_type to OMNI_PIPELINES."""
    new_model_type = "new_model_type"
    unused_model_type = "foo"
    pipe_cfg = PipelineConfig(model_type=unused_model_type)
    assert new_model_type not in OMNI_PIPELINES
    assert unused_model_type not in OMNI_PIPELINES

    # Registering with an explicitly provided model_type uses
    # the passed value instead of the pipeline_cfg.model_type
    register_pipeline(pipe_cfg, new_model_type)
    assert new_model_type in OMNI_PIPELINES
    assert unused_model_type not in OMNI_PIPELINES
    assert OMNI_PIPELINES[new_model_type] is pipe_cfg


def test_register_resolver(custom_resolver, clean_pipeline_registry):
    """Ensure that we can register a custom resolver to OMNI_PIPELINES."""
    new_model_type = "new_model_type"
    assert new_model_type not in OMNI_PIPELINES
    register_pipeline(custom_resolver, new_model_type)
    assert new_model_type in OMNI_PIPELINES
    assert OMNI_PIPELINES[new_model_type] is custom_resolver


def test_register_resolver_requires_model_type(custom_resolver, clean_pipeline_registry):
    """Ensure that registering a custom resolver to OMNI_PIPELINES requires an explicit model_type."""
    with pytest.raises(ValueError):
        register_pipeline(custom_resolver)


def test_minimax_h3_disaggregation_is_explicit_opt_in():
    assert "minimax_h3" not in OMNI_PIPELINES
    pipeline = OMNI_PIPELINES["minimax_h3_disaggregated"]
    assert isinstance(pipeline, PipelineConfig)
    assert pipeline.model_type == "minimax_h3_disaggregated"

def test_wan_eg_preserves_fused_denoise_decode_role():
    pipeline = OMNI_PIPELINES["wan2_2_eg"]

    assert isinstance(pipeline, PipelineConfig)
    assert [stage.stage_role for stage in pipeline.stages] == [
        DiffusionStageRole.ENCODE,
        DiffusionStageRole.DENOISE_DECODE,
    ]
    assert resolve_diffusion_stage_role(None, "dit") is DiffusionStageRole.DENOISE_DECODE
    assert stage_component_groups("denoise") == frozenset({"dit"})
    assert stage_component_groups("denoise_decode") == frozenset({"dit", "vae"})


def test_wan_egd_topology_and_deploy_wiring():
    pipeline = OMNI_PIPELINES["wan2_2_egd"]

    assert isinstance(pipeline, PipelineConfig)
    assert [stage.stage_role for stage in pipeline.stages] == [
        DiffusionStageRole.ENCODE,
        DiffusionStageRole.DENOISE,
        DiffusionStageRole.DECODE,
    ]
    assert [stage.stage_input_payload_keys for stage in pipeline.stages] == [
        (),
        ("prompt_embeds", "negative_prompt_embeds"),
        ("latents",),
    ]
    assert [stage.stage_output_payload_keys for stage in pipeline.stages] == [
        ("prompt_embeds", "negative_prompt_embeds"),
        ("latents",),
        (),
    ]
    assert [stage.input_sources for stage in pipeline.stages] == [(), (0,), (1,)]
    assert [stage.final_output for stage in pipeline.stages] == [False, False, True]

    deploy = load_deploy_config("vllm_omni/deploy/wan2_2_egd.yaml")
    assert deploy.pipeline == "wan2_2_egd"
    assert deploy.connectors is not None
    assert set(deploy.connectors) == {"wan_encode_connector", "wan_latent_connector"}
    assert [stage.devices for stage in deploy.stages] == ["0", "1", "2"]
    assert deploy.stages[0].output_connectors == {"to_stage_1": "wan_encode_connector"}
    assert deploy.stages[1].input_connectors == {"from_stage_0": "wan_encode_connector"}
    assert deploy.stages[1].output_connectors == {"to_stage_2": "wan_latent_connector"}
    assert deploy.stages[2].input_connectors == {"from_stage_1": "wan_latent_connector"}

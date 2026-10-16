# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for out of tree registration to OMNI_PIPELINES."""

import pytest
from transformers import PretrainedConfig

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline
from vllm_omni.config.stage_config import (
    DiffusionStageRole,
    PipelineConfig,
    StagePipelineConfig,
    load_deploy_config,
    pipeline_cfg_resolver,
    resolve_deploy_yaml,
    resolve_diffusion_stage_role,
)
from vllm_omni.diffusion.models.interface import stage_component_groups

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("topology,ports", [("eg", [50081]), ("egd", [50091, 50092])])
@pytest.mark.parametrize("nixl", [False, True], ids=["shared_memory", "nixl"])
def test_wan_deploy_transport_is_explicit_opt_in(topology, ports, nixl):
    from vllm_omni.distributed.omni_connectors.utils.initialization import load_omni_transfer_config

    base_path = f"vllm_omni/deploy/wan2_2_{topology}.yaml"
    deploy_path = base_path.replace(".yaml", "_nixl.yaml") if nixl else base_path
    base = resolve_deploy_yaml(base_path)
    config = resolve_deploy_yaml(deploy_path)
    deploy = load_deploy_config(deploy_path)
    assert deploy.pipeline == f"wan2_2_{topology}"
    assert config["stages"] == base["stages"]
    assert config["async_chunk"] is False
    transfer = load_omni_transfer_config(config_dict=config)
    assert transfer is not None
    assert set(transfer.connectors) == {(str(stage), str(stage + 1)) for stage in range(len(ports))}
    expected_name = "NixlConnector" if nixl else "SharedMemoryConnector"
    for stage, port in enumerate(ports):
        edge = transfer.connectors[(str(stage), str(stage + 1))]
        assert edge.name == expected_name
        if nixl:
            assert edge.extra["host"] == "auto"
            assert edge.extra["zmq_port"] == port
            assert edge.extra["backends"] == ["UCX"]
            assert edge.extra["lease_seconds"] == 300
            assert edge.extra["transfer_timeout_s"] == 300
        else:
            assert set(edge.extra) == {"wakeup_scope"}
            assert isinstance(edge.extra["wakeup_scope"], str)
            assert edge.extra["wakeup_scope"]


def build_fake_pipeline_config(model_type: str) -> PipelineConfig:
    return PipelineConfig(
        model_type=model_type, stages=(StagePipelineConfig(stage_id=0, model_stage="a", final_output=True),)
    )


@pytest.fixture
def custom_resolver():
    """Build a reusable custom resolver for PipelineConfigs."""

    class CustomConfigType(PretrainedConfig):
        pass

    @pipeline_cfg_resolver(config_type=CustomConfigType)
    def custom_resolver(
        hf_config: CustomConfigType,
    ) -> PipelineConfig:
        return build_fake_pipeline_config("resolved_type")

    return custom_resolver


def test_register_pipeline_config(clean_pipeline_registry):
    """Ensure that we can register a custom pipeline config to OMNI_PIPELINES."""
    new_model_type = "new_model_type"
    pipe_cfg = build_fake_pipeline_config(new_model_type)
    assert new_model_type not in OMNI_PIPELINES
    register_pipeline(pipe_cfg)
    assert new_model_type in OMNI_PIPELINES
    assert OMNI_PIPELINES[new_model_type] is pipe_cfg


def test_register_pipeline_config_with_model_type(clean_pipeline_registry):
    """Ensure that we can register a custom pipeline config with an explicit model_type to OMNI_PIPELINES."""
    new_model_type = "new_model_type"
    unused_model_type = "foo"
    pipe_cfg = build_fake_pipeline_config(unused_model_type)
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


@pytest.mark.parametrize("model_type", ["wan2_2", "wan2_2_eg", "wan2_2_egd"])
def test_wan_explicit_topologies_do_not_shadow_ti2v_discovery(model_type):
    pipeline = OMNI_PIPELINES[model_type]

    assert pipeline.diffusers_class_name is None
    assert pipeline.diffusers_class_aliases == ()
    assert OMNI_PIPELINES["wan2_2_ti2v"].diffusers_class_name == "WanPipeline"


def test_wan_eg_preserves_fused_denoise_decode_role():
    pipeline = OMNI_PIPELINES["wan2_2_eg"]

    assert isinstance(pipeline, PipelineConfig)
    assert [stage.stage_role for stage in pipeline.stages] == [
        DiffusionStageRole.ENCODE,
        DiffusionStageRole.DENOISE_DECODE,
    ]
    assert resolve_diffusion_stage_role(None, "dit") is DiffusionStageRole.FULL
    assert resolve_diffusion_stage_role("denoise_decode", "dit") is DiffusionStageRole.DENOISE_DECODE
    assert (
        resolve_diffusion_stage_role(None, OMNI_PIPELINES["wan2_2_ti2v"].stages[0].model_stage)
        is DiffusionStageRole.FULL
    )
    assert stage_component_groups("denoise") == frozenset({"dit"})
    assert stage_component_groups("denoise_decode") == frozenset({"dit", "vae"})

    keys = ("prompt_embeds", "negative_prompt_embeds", "wan_image_condition", "wan_conditioning_metadata")
    assert pipeline.stages[0].stage_output_payload_keys == keys
    assert pipeline.stages[1].stage_input_payload_keys == keys


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
        ("prompt_embeds", "negative_prompt_embeds", "wan_image_condition", "wan_conditioning_metadata"),
        ("latents",),
    ]
    assert [stage.stage_output_payload_keys for stage in pipeline.stages] == [
        ("prompt_embeds", "negative_prompt_embeds", "wan_image_condition", "wan_conditioning_metadata"),
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


def test_omni_pipelines_sorted_alphabetically():
    """OMNI_PIPELINES keys must stay sorted (case-insensitive) by model_type."""
    assert list(OMNI_PIPELINES) == sorted(OMNI_PIPELINES, key=str.lower)

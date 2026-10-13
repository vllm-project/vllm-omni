# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SolarWM discovery must select its AR runtime and video/image capability."""

from __future__ import annotations

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config import config_factory as factory_module
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import StageExecutionType, load_deploy_config, merge_pipeline_deploy
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, _NO_CACHE_ACCELERATION

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("model", ["SolarWM/SolarWM-H3", "/models/SolarWM-H3", "/models/solarwm_h3"])
def test_name_discovery_selects_solar_topology_without_hf_metadata(monkeypatch, model):
    monkeypatch.setattr(StageConfigFactory, "get_hf_config", classmethod(lambda cls, **kwargs: None))
    monkeypatch.setattr(factory_module, "get_hf_file_to_dict", lambda *args, **kwargs: None)
    assert StageConfigFactory._try_infer_model_type(model, trust_remote_code=False) == "solarwm_h3"


def test_native_registration_does_not_inherit_minimax_audio_reference_capabilities():
    assert _DIFFUSION_MODELS["SolarWMH3Pipeline"] == ("solarwm_h3", "pipeline_solarwm_h3", "SolarWMH3Pipeline")
    metadata = get_diffusion_model_metadata("SolarWMH3Pipeline")
    assert metadata.supports_multimodal_inputs
    assert metadata.max_multimodal_image_inputs == 1
    assert metadata.final_output_type == "video"
    assert not metadata.supports_mixed_reference_inputs
    assert not metadata.supports_latent_mask_editing
    assert "SolarWMH3Pipeline" in _NO_CACHE_ACCELERATION


def test_explicit_ar_deployment_keeps_solar_on_the_ar_diffusion_engine():
    topology = OMNI_PIPELINES["solarwm_h3"]
    assert topology.default_deploy_config_name is None
    deploy = load_deploy_config(get_deploy_config_path("solarwm_h3.yaml"))
    stages = merge_pipeline_deploy(topology, deploy)
    assert len(stages) == 1
    assert topology.stages[0].execution_type is StageExecutionType.DIFFUSION
    assert topology.stages[0].final_output_type == "video"
    stage = stages[0]
    assert stage.yaml_engine_args["max_num_seqs"] == 1
    assert stage.yaml_engine_args["model_class_name"] == "SolarWMH3Pipeline"
    assert stage.yaml_engine_args["engine_backend"].endswith("ar_diffusion.engine.ARDiffusionEngine")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3 disaggregated topology registration and deploy YAML.

``cosmos3_omni_disagg`` shares its checkpoint with the co-located deployment, so it
must not be auto-detectable: it is reachable only through a deploy-YAML
``pipeline:`` key.
"""

from pathlib import Path

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.config_factory import StageConfigFactory, _materialize_object_storage_configs
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import (
    StageExecutionType,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_DISAGG_PIPELINE,
    COSMOS3_GENERATOR_ARCH,
    COSMOS3_REASONER_ARCH,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DISAGG_YAML = "cosmos3_super_t2i_disagg.yaml"


@pytest.fixture(autouse=True)
def clear_config_factory_caches():
    """Cached classmethods leak resolved model types between tests otherwise."""
    yield
    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()
    _materialize_object_storage_configs.cache_clear()


def test_disagg_topology_registered():
    assert resolve_pipeline_config("cosmos3_omni_disagg") is COSMOS3_DISAGG_PIPELINE


def test_disagg_topology_is_one_stage_per_tower():
    reasoner, generator = COSMOS3_DISAGG_PIPELINE.stages
    assert COSMOS3_DISAGG_PIPELINE.default_deploy_config_name == DISAGG_YAML

    assert (reasoner.stage_id, reasoner.model_stage) == (0, "reasoner")
    assert reasoner.execution_type is StageExecutionType.DIFFUSION
    assert reasoner.input_sources == ()
    assert reasoner.final_output is False
    assert reasoner.model_arch == COSMOS3_REASONER_ARCH

    assert (generator.stage_id, generator.model_stage) == (1, "generator")
    assert generator.input_sources == (0,)
    assert generator.final_output is True
    assert generator.final_output_type == "image"
    assert generator.model_arch == COSMOS3_GENERATOR_ARCH
    # The handoff travels in the stage payload, not the AR KV-transfer path.
    assert generator.omni_kv_config == {"need_recv_cache": False}


def test_declares_no_auto_detect_hooks():
    """Every Cosmos3 checkpoint shares one arch/class name; claiming it would hijack them all."""
    assert COSMOS3_DISAGG_PIPELINE.hf_architectures == ()
    assert COSMOS3_DISAGG_PIPELINE.diffusers_class_name is None


def test_deploy_pipeline_key_selects_the_topology(tmp_path):
    deploy_path = tmp_path / "deploy.yaml"
    deploy_path.write_text("pipeline: cosmos3_omni_disagg\n", encoding="utf-8")

    pipeline = StageConfigFactory.get_pipeline_config(
        model=str(tmp_path),
        trust_remote_code=False,
        deploy_config_path=str(deploy_path),
    )

    assert pipeline is COSMOS3_DISAGG_PIPELINE


def test_shipped_yaml_merges_into_reasoner_plus_generator():
    deploy = load_deploy_config(Path(get_deploy_config_path(DISAGG_YAML)))
    stages = merge_pipeline_deploy(COSMOS3_DISAGG_PIPELINE, deploy)

    assert [s.model_stage for s in stages] == ["reasoner", "generator"]
    # Logical indexes into one shared visible set: they must differ.
    assert [s.yaml_runtime["devices"] for s in stages] == ["0", "1"]
    assert stages[1].custom_process_input_func.endswith(".reasoner2generator")
    assert all(s.yaml_engine_args["model_config"]["guardrails"] is False for s in stages)
    # Generation defaults on one stage only would make the branch fingerprints diverge.
    assert not any(s.default_sampling_params for s in deploy.stages)

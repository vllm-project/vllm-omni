# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.resolver import _build_registered_resolution
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.model_executor.models.hunyuan_image3.pipeline import HUNYUAN_IMAGE3_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
DEPLOY = Path(__file__).resolve().parents[2] / "vllm_omni/deploy/hunyuan-image-3-weight-sharing.yaml"


def test_shared_deploy_reaches_structured_resolver_without_losing_kv_config():
    deploy = load_deploy_config(DEPLOY)
    assert deploy.weight_sharing is not None and deploy.model_runner == "v2"
    assert deploy.weight_sharing.gpu_memory_utilization == 0.95
    assert deploy.weight_sharing.timeout == 120.0
    stages = {stage.stage_id: stage for stage in deploy.stages}
    assert stages[0].devices == stages[1].devices == "0,1,2,3"
    assert all(stage.num_replicas == 1 for stage in stages.values())
    assert stages[0].engine_extras["kv_cache_memory_bytes"] == 2 * 1024**3
    assert stages[1].engine_extras["kv_cache_memory_bytes"] == 1024**3
    structured = VllmOmniConfig.from_pipeline_config(HUNYUAN_IMAGE3_PIPELINE, user_deploy_config=deploy)
    resolved = _build_registered_resolution(structured)
    assert resolved.weight_sharing == deploy.weight_sharing
    assert resolved.stage_by_id(0).cache_config.kv_cache_memory_bytes == 2 * 1024**3
    assert resolved.stage_by_id(1).cache_config.kv_cache_memory_bytes == 1024**3
    for stage in resolved.stage_configs:
        assert stage.parallel_config.tensor_parallel_size == 4
        assert not stage.parallel_config.enable_expert_parallel
        assert stage.model_config.enforce_eager
        assert stage.connector_config.kv_transfer_config is not None


def test_yaml_null_disables_sharing_without_changing_inherited_stages(tmp_path):
    overlay = tmp_path / "ordinary.yaml"
    overlay.write_text(f"base_config: '{DEPLOY.as_posix()}'\nweight_sharing: null\n", encoding="utf-8")
    config = load_deploy_config(overlay)
    assert config.weight_sharing is None
    assert len(config.stages) == 2
    assert config.model_runner == "v2"

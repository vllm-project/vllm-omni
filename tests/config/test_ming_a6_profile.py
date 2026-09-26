# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ming_a6_single_gpu_profile_is_explicitly_colocated():
    path = Path(get_deploy_config_path("ming_flash_omni_image_single_gpu.yaml"))
    deploy = load_deploy_config(path)
    stages = merge_pipeline_deploy(resolve_pipeline_config("ming_flash_omni_image"), deploy)

    assert [stage.yaml_runtime["devices"] for stage in stages] == ["0", "0"]
    assert stages[0].yaml_engine_args["tensor_parallel_size"] == 1
    assert stages[1].yaml_engine_args["parallel_config"]["cfg_parallel_size"] == 1
    assert stages[1].yaml_engine_args["inline_diffusion"] is True


def test_ming_a6_cfg_parallel_profile_assigns_two_diffusion_ranks():
    path = Path(get_deploy_config_path("ming_flash_omni_image_cfg_parallel.yaml"))
    deploy = load_deploy_config(path)
    stages = merge_pipeline_deploy(resolve_pipeline_config("ming_flash_omni_image"), deploy)

    assert [stage.yaml_runtime["devices"] for stage in stages] == ["0,1,2,3", "4,5"]
    assert stages[1].yaml_engine_args["parallel_config"]["cfg_parallel_size"] == 2

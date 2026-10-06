# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
    assert stages[0].yaml_engine_args["max_num_seqs"] == 2
    assert stages[0].yaml_engine_args["tensor_parallel_size"] == 1
    assert stages[1].yaml_engine_args["parallel_config"]["cfg_parallel_size"] == 1
    assert stages[1].yaml_engine_args["inline_diffusion"] is True


@pytest.mark.parametrize(
    ("profile", "expected_inline"),
    [("ming_flash_omni_image_single_gpu.yaml", True), ("ming_flash_omni_image_stepwise.yaml", False)],
)
def test_ming_profile_reaches_typed_runtime_with_declared_client_topology(profile, expected_inline):
    """Regression: production config resolution must preserve inline selection.

    Input source: the shipped Ming deploy profiles used by ``vllm serve --omni``.
    Expected source: the documented single-GPU inline client contract versus the
    separate-process stepwise profile. StageRuntime consumes the typed topology,
    so inspecting only the legacy YAML dictionary misses this startup regression.
    """
    from vllm_omni.config.omni_config import VllmOmniConfig

    pipeline = resolve_pipeline_config("ming_flash_omni_image")
    config = VllmOmniConfig.from_pipeline_config(pipeline, deploy_config_path=str(get_deploy_config_path(profile)))
    image_stage = config.stage_by_id(1)
    assert image_stage.stage_pipeline_config.inline_diffusion is expected_inline
    assert pipeline.stages[1].inline_diffusion is False


def test_ming_a6_cfg_parallel_profile_assigns_two_diffusion_ranks():
    path = Path(get_deploy_config_path("ming_flash_omni_image_cfg_parallel.yaml"))
    deploy = load_deploy_config(path)
    stages = merge_pipeline_deploy(resolve_pipeline_config("ming_flash_omni_image"), deploy)

    assert [stage.yaml_runtime["devices"] for stage in stages] == ["0,1,2,3", "4,5"]
    assert stages[0].yaml_engine_args["max_num_seqs"] == 2
    assert stages[1].yaml_engine_args["parallel_config"]["cfg_parallel_size"] == 2


def test_ming_high_throughput_profile_enables_request_waves():
    path = Path(get_deploy_config_path("ming_flash_omni_image_high_throughput.yaml"))
    deploy = load_deploy_config(path)
    stages = merge_pipeline_deploy(resolve_pipeline_config("ming_flash_omni_image"), deploy)

    assert stages[0].yaml_engine_args["max_num_seqs"] == 8
    assert stages[1].yaml_engine_args["max_num_seqs"] == 4
    assert stages[1].yaml_engine_args["request_batch_max_wait_ms"] == 20


def test_ming_default_image_profile_remains_low_latency():
    path = Path(get_deploy_config_path("ming_flash_omni_image.yaml"))
    deploy = load_deploy_config(path)
    stages = merge_pipeline_deploy(resolve_pipeline_config("ming_flash_omni_image"), deploy)

    assert stages[1].yaml_engine_args.get("max_num_seqs", 1) == 1
    assert stages[1].yaml_engine_args.get("request_batch_max_wait_ms", 0.0) == 0.0


def test_ming_stepwise_profile_enables_continuous_batching():
    path = Path(get_deploy_config_path("ming_flash_omni_image_stepwise.yaml"))
    deploy = load_deploy_config(path)
    stages = merge_pipeline_deploy(resolve_pipeline_config("ming_flash_omni_image"), deploy)

    assert stages[1].yaml_engine_args["step_execution"] is True
    assert stages[1].yaml_engine_args["max_num_seqs"] == 4
    assert stages[1].yaml_engine_args.get("request_batch_max_wait_ms", 0.0) == 0.0

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deployment discovery for Cosmos3-Nano-Sim-Auto across the registries."""

import importlib
from pathlib import Path

import pytest
import yaml

from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.diffusion.registry import (
    _DIFFUSION_IR_OP_PRIORITY_FUNCS,
    _DIFFUSION_MODELS,
    _DIFFUSION_POST_PROCESS_FUNCS,
    _DIFFUSION_PRE_PROCESS_FUNCS,
    _NO_CACHE_ACCELERATION,
)
from vllm_omni.model_extras import get_extra_body_params

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_deployment_discovery() -> None:
    pipeline = resolve_pipeline_config("cosmos3_nano_sim_auto")
    assert pipeline is not None
    deploy_path = Path(__file__).resolve().parents[4] / "vllm_omni/deploy" / pipeline.default_deploy_config_name
    deploy = yaml.safe_load(deploy_path.read_text())
    assert deploy["pipeline"] == "cosmos3_nano_sim_auto"
    assert pipeline.model_arch == pipeline.diffusers_class_name == "Cosmos3NanoSimAutoPipeline"
    assert deploy["stages"][0]["model_class_name"] == pipeline.model_arch
    assert deploy["stages"][0]["engine_backend"].endswith("ARDiffusionEngine")

    folder, module_name, class_name = _DIFFUSION_MODELS[pipeline.model_arch]
    module = importlib.import_module(f"vllm_omni.diffusion.models.{folder}.{module_name}")
    assert getattr(module, class_name).__name__ == pipeline.model_arch
    for registry in (_DIFFUSION_PRE_PROCESS_FUNCS, _DIFFUSION_POST_PROCESS_FUNCS, _DIFFUSION_IR_OP_PRIORITY_FUNCS):
        assert callable(getattr(module, registry[pipeline.model_arch]))
    assert pipeline.model_arch in _NO_CACHE_ACCELERATION
    extras = get_extra_body_params(pipeline.model_arch)
    assert {"session_id", "reset", "close_session", "multiview", "lidar", "wsm"} <= set(extras)

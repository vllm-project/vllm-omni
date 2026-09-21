# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for the additive SenseNova-Vision single-stage topology."""

from __future__ import annotations

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType
from vllm_omni.model_executor.models.sensenova_vision.pipeline import SENSENOVA_VISION_SINGLE_STAGE_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_single_stage_topology_is_one_diffusion_stage() -> None:
    stage = SENSENOVA_VISION_SINGLE_STAGE_PIPELINE.get_stage(0)
    assert stage is not None
    assert stage.execution_type == StageExecutionType.DIFFUSION
    assert stage.input_sources == ()


def test_single_stage_topology_resolves_from_registry() -> None:
    assert OMNI_PIPELINES["sensenova_vision_single_stage"] is SENSENOVA_VISION_SINGLE_STAGE_PIPELINE
    resolved = resolve_pipeline_config("sensenova_vision_single_stage")
    assert isinstance(resolved, PipelineConfig)
    assert resolved is SENSENOVA_VISION_SINGLE_STAGE_PIPELINE
    assert resolved.default_deploy_config_name == "sensenova_vision_single_stage.yaml"

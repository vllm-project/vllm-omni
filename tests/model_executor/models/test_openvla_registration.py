# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 registration tests for OpenVLA (CPU, no weights).

Without a pipeline entry, `vllm-omni serve openvla/openvla-7b` does not fall
through to the model registry — it resolves no pipeline, takes the default
single-stage *diffusion* config and dies in the diffusion loader. These tests
pin the registration and the one property that makes it an AR stage.
"""

from importlib import import_module
from pathlib import Path

import pytest
import yaml

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.model_executor.models.openvla.pipeline import (
    OPENVLA_ACTION_DIM,
    OPENVLA_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ARCH = "OpenVLAForActionPrediction"


def test_model_type_resolves_to_the_openvla_pipeline():
    assert OMNI_PIPELINES["openvla"] is OPENVLA_PIPELINE


def test_topology_is_one_autoregressive_stage():
    assert OPENVLA_PIPELINE.validate() == []
    assert len(OPENVLA_PIPELINE.stages) == 1
    stage = OPENVLA_PIPELINE.stages[0]
    assert stage.execution_type is StageExecutionType.LLM_AR
    assert stage.final_output is True
    assert stage.final_output_type == "actions"
    assert stage.requires_multimodal_data is True
    assert stage.owns_tokenizer is True


def test_engine_output_type_is_unset():
    """`actions` is not an OutputModality; setting it crashes engine init."""
    assert OPENVLA_PIPELINE.stages[0].engine_output_type is None


def test_sampling_is_pinned_to_a_fixed_length_greedy_decode():
    constraints = OPENVLA_PIPELINE.stages[0].sampling_constraints
    assert constraints["temperature"] == 0.0
    assert constraints["max_tokens"] == OPENVLA_ACTION_DIM
    assert constraints["min_tokens"] == OPENVLA_ACTION_DIM
    assert constraints["ignore_eos"] is True
    # The tokens are bin indices, so there is nothing to detokenise.
    assert constraints["detokenize"] is False


def test_stage_declares_its_robot_adapter():
    """The OpenPI endpoint finds the adapter through the stage, not a side table."""
    from vllm_omni.model_executor.models.openvla.robot_adapter import OpenVLARobotAdapter

    path = OPENVLA_PIPELINE.stages[0].robot_adapter
    assert path == ("vllm_omni.model_executor.models.openvla.robot_adapter.OpenVLARobotAdapter")
    module_path, attr = path.rsplit(".", 1)
    assert getattr(import_module(module_path), attr) is OpenVLARobotAdapter


def test_architecture_resolves_through_the_merged_model_registry():
    """The model class itself is upstream vLLM's; we add no model code."""
    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    assert _ARCH in OmniModelRegistry.get_supported_archs()
    assert OPENVLA_PIPELINE.model_arch == _ARCH
    assert OPENVLA_PIPELINE.stages[0].model_arch == _ARCH


def test_deploy_config_exists_and_points_back_at_this_pipeline():
    name = OPENVLA_PIPELINE.default_deploy_config_name
    assert name == "openvla.yaml"
    path = Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy" / name
    deploy = yaml.safe_load(path.read_text())
    assert deploy["pipeline"] == "openvla"
    # A single-stage pipeline has no next-stage processor, so chunking must be off.
    assert deploy["async_chunk"] is False
    assert len(deploy["stages"]) == 1

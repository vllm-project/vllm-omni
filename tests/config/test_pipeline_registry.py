# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for out of tree registration to OMNI_PIPELINES."""

import pytest
from transformers import PretrainedConfig

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline
from vllm_omni.config.stage_config import PipelineConfig, pipeline_cfg_resolver

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

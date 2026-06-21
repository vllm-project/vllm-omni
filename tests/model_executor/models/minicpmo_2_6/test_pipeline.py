# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MiniCPM-o 2.6 pipeline registration.

Covers:
  - pipeline declared in the central registry
  - lazy loader returns the expected ``PipelineConfig``
  - 2-stage topology (thinker LLM_AR + talker LLM_AR with audio output)
  - stage 1 routes through ``llm2tts`` custom input processor
  - ``hf_architectures`` covers both the shared ``MiniCPMO`` alias and the
    explicit 2.6 arch
  - ``hf_config_predicate`` selects MiniCPM-o 2.6 only and rejects 4.5
    checkpoints (regression guard for the shared-arch routing collision).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.config.pipeline_registry import _OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    _PIPELINE_REGISTRY,
    PipelineConfig,
    StageExecutionType,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PIPELINE_KEY = "minicpmo_2_6"


class TestRegistryDeclaration:
    def test_declared_in_omni_pipelines(self) -> None:
        assert _PIPELINE_KEY in _OMNI_PIPELINES

    def test_visible_in_central_registry(self) -> None:
        assert _PIPELINE_KEY in _PIPELINE_REGISTRY

    def test_lazy_load_returns_pipeline_config(self) -> None:
        pipeline = _PIPELINE_REGISTRY[_PIPELINE_KEY]
        assert isinstance(pipeline, PipelineConfig)
        assert pipeline.model_type == _PIPELINE_KEY
        assert pipeline.model_arch == "MiniCPMO26OmniForConditionalGeneration"


class TestPipelineTopology:
    @pytest.fixture(scope="class")
    def pipeline(self) -> PipelineConfig:
        return _PIPELINE_REGISTRY[_PIPELINE_KEY]

    def test_two_stages(self, pipeline: PipelineConfig) -> None:
        assert len(pipeline.stages) == 2
        assert [s.stage_id for s in pipeline.stages] == [0, 1]

    def test_topology_validates(self, pipeline: PipelineConfig) -> None:
        assert pipeline.validate() == []

    def test_thinker_stage(self, pipeline: PipelineConfig) -> None:
        thinker = pipeline.get_stage(0)
        assert thinker is not None
        assert thinker.model_stage == "llm"
        assert thinker.execution_type == StageExecutionType.LLM_AR
        assert thinker.input_sources == ()
        assert thinker.final_output is True
        assert thinker.final_output_type == "text"
        assert thinker.owns_tokenizer is True
        assert thinker.requires_multimodal_data is True

    def test_talker_stage(self, pipeline: PipelineConfig) -> None:
        talker = pipeline.get_stage(1)
        assert talker is not None
        assert talker.model_stage == "tts"
        assert talker.execution_type == StageExecutionType.LLM_AR
        # talker consumes thinker output
        assert talker.input_sources == (0,)
        assert talker.final_output is True
        assert talker.final_output_type == "audio"
        assert talker.engine_output_type == "audio"
        # scope KV cache / mrope sizing to talker sub-config
        assert talker.hf_config_name == "tts_config"

    def test_talker_routes_through_llm2tts(self, pipeline: PipelineConfig) -> None:
        talker = pipeline.get_stage(1)
        assert talker is not None
        assert talker.custom_process_input_func == (
            "vllm_omni.model_executor.stage_input_processors.minicpmo_2_6_omni.llm2tts"
        )


class TestArchAliases:
    """``hf_architectures`` must cover both the shared and explicit names."""

    @pytest.fixture(scope="class")
    def pipeline(self) -> PipelineConfig:
        return _PIPELINE_REGISTRY[_PIPELINE_KEY]

    def test_shared_minicpmo_alias_present(self, pipeline: PipelineConfig) -> None:
        assert "MiniCPMO" in pipeline.hf_architectures

    def test_explicit_2_6_arch_present(self, pipeline: PipelineConfig) -> None:
        assert "MiniCPMO26OmniForConditionalGeneration" in pipeline.hf_architectures


class TestHfConfigPredicate:
    """Regression guard for the 2.6 / 4.5 shared-arch routing collision.

    Both MiniCPM-o 2.6 and 4.5 ship ``architectures=["MiniCPMO"]`` in HF
    config. The 2.6 pipeline uses ``hf_config_predicate`` to opt in only
    when ``config.version == "2.6"``.
    """

    @pytest.fixture(scope="class")
    def predicate(self):
        pipeline = _PIPELINE_REGISTRY[_PIPELINE_KEY]
        assert pipeline.hf_config_predicate is not None, (
            "MiniCPM-o 2.6 pipeline must declare hf_config_predicate to "
            "avoid misrouting MiniCPM-o 4.5 checkpoints into the 2.6 path."
        )
        return pipeline.hf_config_predicate

    def test_accepts_2_6_string(self, predicate) -> None:
        assert predicate(SimpleNamespace(version="2.6")) is True

    def test_rejects_4_5_string(self, predicate) -> None:
        assert predicate(SimpleNamespace(version="4.5")) is False

    def test_rejects_missing_version(self, predicate) -> None:
        assert predicate(SimpleNamespace()) is False

    def test_rejects_empty_version(self, predicate) -> None:
        assert predicate(SimpleNamespace(version="")) is False

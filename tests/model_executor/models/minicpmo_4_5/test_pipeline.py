# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MiniCPM-o 4.5 pipeline registration.

Covers:
  - pipeline declared in the central registry
  - lazy loader returns the expected ``PipelineConfig``
  - 2-stage topology (thinker LLM_AR + talker LLM_AR with audio output)
  - stage 1 routes through ``llm2tts`` custom input processor
  - ``hf_architectures`` covers both the shared ``MiniCPMO`` alias and the
    explicit 4.5 arch
  - ``hf_config_predicate`` selects MiniCPM-o 4.5 only and rejects 2.6
    checkpoints (regression guard for the shared-arch routing collision).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_PIPELINE_KEY = "minicpmo_4_5"


class TestRegistryDeclaration:
    def test_declared_in_omni_pipelines(self) -> None:
        assert _PIPELINE_KEY in OMNI_PIPELINES

    def test_visible_in_central_registry(self) -> None:
        assert _PIPELINE_KEY in OMNI_PIPELINES

    def test_lazy_load_returns_pipeline_config(self) -> None:
        pipeline = OMNI_PIPELINES[_PIPELINE_KEY]
        assert isinstance(pipeline, PipelineConfig)
        assert pipeline.model_type == _PIPELINE_KEY
        assert pipeline.model_arch == "MiniCPMO45OmniForConditionalGeneration"


class TestPipelineTopology:
    @pytest.fixture(scope="class")
    def pipeline(self) -> PipelineConfig:
        return OMNI_PIPELINES[_PIPELINE_KEY]

    def test_two_stages(self, pipeline: PipelineConfig) -> None:
        assert len(pipeline.stages) == 2
        assert [s.stage_id for s in pipeline.stages] == [0, 1]

    def test_topology_validates(self, pipeline: PipelineConfig) -> None:
        # ``validate`` returns a list of structural errors; empty == valid.
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
        # stage 1's custom_process_input_func is what bridges thinker
        # hidden_states + token ids into the talker; if this drifts the
        # talker silently goes through the dummy path.
        assert talker.custom_process_input_func == (
            "vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni.llm2tts"
        )


class TestArchAliases:
    """``hf_architectures`` must cover both the shared and explicit names."""

    @pytest.fixture(scope="class")
    def pipeline(self) -> PipelineConfig:
        return OMNI_PIPELINES[_PIPELINE_KEY]

    def test_shared_minicpmo_alias_present(self, pipeline: PipelineConfig) -> None:
        # MiniCPM-o 4.5 ships ``architectures=["MiniCPMO"]`` in its HF config.
        # Without this alias the arch-fallback path in StageConfigFactory
        # cannot resolve the pipeline.
        assert "MiniCPMO" in pipeline.hf_architectures

    def test_explicit_4_5_arch_present(self, pipeline: PipelineConfig) -> None:
        # Reserve the explicit arch name for future repos that opt into it.
        assert "MiniCPMO45OmniForConditionalGeneration" in pipeline.hf_architectures


class TestHfConfigPredicate:
    """Regression guard for the 2.6 / 4.5 shared-arch routing collision.

    Both MiniCPM-o 2.6 and 4.5 ship ``architectures=["MiniCPMO"]`` in HF
    config. The 4.5 pipeline uses ``hf_config_predicate`` to opt in only
    when ``config.version == "4.5"``; without it, a 2.6 checkpoint would
    intersect on the shared arch and get misrouted into the 4.5 pipeline.
    """

    @pytest.fixture(scope="class")
    def predicate(self):
        pipeline = OMNI_PIPELINES[_PIPELINE_KEY]
        assert pipeline.hf_config_predicate is not None, (
            "MiniCPM-o 4.5 pipeline must declare hf_config_predicate to "
            "avoid misrouting MiniCPM-o 2.6 checkpoints into the 4.5 path."
        )
        return pipeline.hf_config_predicate

    def test_accepts_4_5_string(self, predicate) -> None:
        assert predicate(SimpleNamespace(version="4.5")) is True

    def test_rejects_2_6_string(self, predicate) -> None:
        assert predicate(SimpleNamespace(version="2.6")) is False

    def test_rejects_missing_version(self, predicate) -> None:
        # 1.0 / older checkpoints do not carry ``version`` at all.
        assert predicate(SimpleNamespace()) is False

    def test_rejects_empty_version(self, predicate) -> None:
        assert predicate(SimpleNamespace(version="")) is False

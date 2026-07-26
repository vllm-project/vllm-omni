# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MiniCPM-o 4.5 pipeline registration.

Covers:
  - pipeline declared in the central registry
  - lazy loader returns the expected ``PipelineConfig``
  - split 3-stage topology with no fused compatibility registration
  - stage 1 routes through ``llm2tts`` and the async Code2Wav producer
  - ``hf_architectures`` covers both the shared ``MiniCPMO`` alias and the
    explicit 4.5 arch
  - ``hf_config_predicate`` selects MiniCPM-o 4.5 only and rejects 2.6
    checkpoints (regression guard for the shared-arch routing collision).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_PIPELINE_KEY = "minicpmo_4_5"
_DEPLOY_DIR = Path(__file__).resolve().parents[4] / "vllm_omni" / "deploy"
_DEPLOY_LAYOUTS = {
    "minicpmo_4_5.yaml": ["0", "0", "0"],
    "minicpmo_4_5_batching.yaml": ["0", "1", "1"],
    "minicpmo_4_5_2gpu.yaml": ["0", "1", "1"],
    "minicpmo_4_5_3gpu.yaml": ["0", "1", "2"],
    "minicpmo_4_5_8x4090.yaml": ["0,1,2,3", "4", "5"],
}


class TestRegistryDeclaration:
    def test_declared_in_omni_pipelines(self) -> None:
        assert _PIPELINE_KEY in OMNI_PIPELINES
        assert "minicpmo_4_5_fused" not in OMNI_PIPELINES

    def test_lazy_load_returns_pipeline_config(self) -> None:
        pipeline = OMNI_PIPELINES[_PIPELINE_KEY]
        assert isinstance(pipeline, PipelineConfig)
        assert pipeline.model_type == _PIPELINE_KEY
        assert pipeline.model_arch == "MiniCPMO45OmniForConditionalGeneration"

    def test_native_duplex_control_is_explicit_without_a_fixed_session_cap(self) -> None:
        pipeline = OMNI_PIPELINES[_PIPELINE_KEY]
        assert pipeline.duplex_control_enabled is True
        assert pipeline.duplex_serving_adapter == (
            "vllm_omni.experimental.fullduplex.minicpmo45.serving_adapter.MiniCPMO45ServingRuntimeAdapter"
        )
        assert not hasattr(pipeline, "max_native_duplex_sessions")

    def test_ordinary_pipeline_defaults_to_no_duplex_control(self) -> None:
        pipeline = PipelineConfig(model_type="ordinary")
        assert pipeline.duplex_control_enabled is False
        assert pipeline.duplex_serving_adapter is None
        assert not hasattr(pipeline, "max_native_duplex_sessions")


class TestPipelineTopology:
    @pytest.fixture(scope="class")
    def pipeline(self) -> PipelineConfig:
        return OMNI_PIPELINES[_PIPELINE_KEY]

    def test_three_stages(self, pipeline: PipelineConfig) -> None:
        assert len(pipeline.stages) == 3
        assert [s.stage_id for s in pipeline.stages] == [0, 1, 2]

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
        assert talker.final_output is False
        assert talker.final_output_type is None
        assert talker.engine_output_type == "latent"
        # scope KV cache / mrope sizing to talker sub-config
        assert talker.hf_config_name == "tts_config"
        assert talker.custom_process_next_stage_input_func is None
        assert talker.async_chunk_process_next_stage_input_func == (
            "vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni.tts2code2wav_async_chunk"
        )

    def test_talker_routes_through_llm2tts(self, pipeline: PipelineConfig) -> None:
        talker = pipeline.get_stage(1)
        assert talker is not None
        # stage 1's custom_process_input_func is what bridges thinker
        # hidden_states + token ids into the talker; if this drifts the
        # talker silently goes through the dummy path.
        assert talker.custom_process_input_func == (
            "vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni.llm2tts"
        )

    def test_code2wav_stage(self, pipeline: PipelineConfig) -> None:
        code2wav = pipeline.get_stage(2)
        assert code2wav is not None
        assert code2wav.model_stage == "code2wav"
        assert code2wav.execution_type == StageExecutionType.LLM_GENERATION
        assert code2wav.input_sources == (1,)
        assert code2wav.final_output is True
        assert code2wav.final_output_type == "audio"
        assert code2wav.engine_output_type == "audio"
        assert code2wav.model_arch == "MiniCPMO45Code2Wav"
        assert code2wav.sync_process_input_func is None


class TestDeployTopology:
    @pytest.mark.parametrize(("filename", "devices"), _DEPLOY_LAYOUTS.items())
    def test_deploy_resolves_three_stage_pipeline(self, filename: str, devices: list[str]) -> None:
        deploy = load_deploy_config(_DEPLOY_DIR / filename)
        pipeline = OMNI_PIPELINES[deploy.pipeline]
        stages = merge_pipeline_deploy(pipeline, deploy)

        assert deploy.pipeline == _PIPELINE_KEY
        assert deploy.async_chunk is True
        assert [stage.stage_id for stage in stages] == [0, 1, 2]
        assert [stage.yaml_runtime["devices"] for stage in stages] == devices
        assert "additional_config" not in stages[1].yaml_engine_args
        assert "skip_mm_profiling" not in stages[0].yaml_engine_args
        assert stages[1].yaml_engine_args["skip_mm_profiling"] is True
        assert stages[2].yaml_engine_args["skip_mm_profiling"] is True
        assert stages[1].yaml_extras["output_connectors"]["to_stage_2"] == "connector_of_shared_memory"
        assert stages[2].yaml_extras["input_connectors"]["from_stage_1"] == "connector_of_shared_memory"
        connector = deploy.connectors["connector_of_shared_memory"]
        assert connector["name"] == "SharedMemoryConnector"
        assert connector["extra"]["codec_chunk_frames"] == 25
        assert connector["extra"]["codec_left_context_frames"] == 3
        assert connector["extra"]["connector_get_max_wait_first_chunk"] == 3000
        assert connector["extra"]["connector_get_max_wait"] == 300
        if filename == "minicpmo_4_5.yaml":
            assert [stage.yaml_engine_args["max_num_seqs"] for stage in stages] == [4, 4, 4]
            assert [stage.yaml_engine_args["gpu_memory_utilization"] for stage in stages] == [
                0.65,
                0.15,
                0.15,
            ]
        elif filename in {"minicpmo_4_5_batching.yaml", "minicpmo_4_5_2gpu.yaml"}:
            assert [stage.yaml_engine_args["gpu_memory_utilization"] for stage in stages] == [
                0.9,
                0.55,
                0.35,
            ]

    def test_pipeline_exposes_no_full_payload_or_token_placeholder_hooks(self) -> None:
        pipeline = OMNI_PIPELINES[_PIPELINE_KEY]
        talker = pipeline.get_stage(1)
        code2wav = pipeline.get_stage(2)
        assert talker is not None
        assert code2wav is not None
        assert talker.custom_process_next_stage_input_func is None
        assert code2wav.sync_process_input_func is None


def test_code2wav_model_is_lazily_registered() -> None:
    assert _OMNI_MODELS["MiniCPMO45Code2Wav"] == (
        "minicpmo_4_5",
        "minicpmo_4_5_code2wav",
        "MiniCPMO45Code2Wav",
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

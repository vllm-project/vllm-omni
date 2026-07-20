# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for StageConfigFactory and related classes.
"""

import importlib
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import PretrainedConfig, Qwen3OmniMoeConfig

from tests.helpers.stage_config import get_deploy_config_path, get_deploy_config_stage
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.endpoint_policy import EndpointRestriction, OmniServingCapability
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline, resolve_pipeline_config
from vllm_omni.config.stage_config import (
    _DEPLOY_DIR,
    DeployConfig,
    PipelineConfig,
    StageConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    StageType,
    _apply_platform_overrides,
    _deep_merge_stage,
    _resolve_scheduler,
    build_stage_runtime_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
    pipeline_cfg_resolver,
    strip_parent_engine_args,
)
from vllm_omni.engine.arg_utils import SHARED_FIELDS, EngineArgs, internal_blacklist_keys

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _stable_test_platform(monkeypatch):
    from vllm_omni import platforms

    platform = platforms.current_omni_platform
    monkeypatch.setattr(platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(platform, "device_type", "cpu", raising=False)


@pytest.fixture(autouse=True)
def clear_config_factory_caches():
    """Clear cached classmethods from the StageConfigFactory to prevent test pollution."""
    yield
    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()


Q3_OMNI_ALL_STAGES_HF_CONFIG = Qwen3OmniMoeConfig(enable_audio_output=True)
Q3_OMNI_THINKER_HF_CONFIG = Qwen3OmniMoeConfig(enable_audio_output=False)


class TestStageType:
    """Tests for StageType enum."""

    def test_stage_type_values(self):
        """Test StageType enum values."""
        assert StageType.LLM.value == "llm"
        assert StageType.DIFFUSION.value == "diffusion"

    def test_stage_type_from_string(self):
        """Test creating StageType from string."""
        assert StageType("llm") == StageType.LLM
        assert StageType("diffusion") == StageType.DIFFUSION


class TestStageConfig:
    """Tests for StageConfig dataclass."""

    def test_minimal_config(self):
        """Test creating StageConfig with minimal required fields."""
        config = StageConfig(stage_id=0, model_stage="thinker")
        assert config.stage_id == 0
        assert config.model_stage == "thinker"
        assert config.stage_type == StageType.LLM
        assert config.input_sources == []
        assert config.final_output is False
        assert config.worker_type is None

    def test_full_config(self):
        """Test creating StageConfig with all fields."""
        config = StageConfig(
            stage_id=1,
            model_stage="talker",
            stage_type=StageType.LLM,
            input_sources=[0],
            custom_process_input_func="module.path.func",
            final_output=True,
            final_output_type="audio",
            worker_type="ar",
            scheduler_cls="path.to.Scheduler",
            hf_config_name="talker_config",
            is_comprehension=False,
        )
        assert config.stage_id == 1
        assert config.model_stage == "talker"
        assert config.input_sources == [0]
        assert config.final_output_type == "audio"
        assert config.worker_type == "ar"

    def test_to_omegaconf_basic(self):
        """Test converting StageConfig to OmegaConf format."""
        config = StageConfig(
            stage_id=0,
            model_stage="thinker",
            stage_type=StageType.LLM,
            worker_type="ar",
            final_output=True,
            final_output_type="text",
        )
        omega_config = config.to_omegaconf()

        assert omega_config.stage_id == 0
        assert omega_config.stage_type == "llm"
        assert omega_config.engine_args.model_stage == "thinker"
        assert omega_config.engine_args.worker_type == "ar"
        assert omega_config.final_output is True
        assert omega_config.final_output_type == "text"
        assert "max_num_seqs" not in omega_config.engine_args
        # Legacy field name for backward compatibility
        assert omega_config.engine_input_source == []

    def test_to_omegaconf_with_runtime_overrides(self):
        """Test that runtime overrides are applied to OmegaConf output."""
        config = StageConfig(
            stage_id=0,
            model_stage="thinker",
            runtime_overrides={
                "gpu_memory_utilization": 0.9,
                "tensor_parallel_size": 2,
                "devices": "0,1",
                "max_batch_size": 64,
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            omega_config = config.to_omegaconf()

        assert omega_config.engine_args.gpu_memory_utilization == 0.9
        assert omega_config.engine_args.tensor_parallel_size == 2
        assert omega_config.runtime.devices == "0,1"
        # max_batch_size is migrated to engine_args.max_num_seqs
        assert omega_config.engine_args.max_num_seqs == 64

    def test_to_omegaconf_max_batch_size_deprecation(self):
        """Test that runtime.max_batch_size emits a FutureWarning."""
        config = StageConfig(
            stage_id=0,
            model_stage="thinker",
            runtime_overrides={"max_batch_size": 8},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config.to_omegaconf()
            deprecation_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
            assert len(deprecation_warnings) == 1
            assert "max_batch_size" in str(deprecation_warnings[0].message)

    def test_to_omegaconf_max_num_seqs_in_engine_args(self):
        """Test that max_num_seqs in yaml_engine_args takes precedence."""
        config = StageConfig(
            stage_id=0,
            model_stage="thinker",
            yaml_engine_args={"max_num_seqs": 32},
        )
        omega_config = config.to_omegaconf()
        assert omega_config.engine_args.max_num_seqs == 32

    def test_to_omegaconf_diffusion_parallel_overrides_replace_nested_values(self):
        config = StageConfig(
            stage_id=1,
            model_stage="diffusion",
            stage_type=StageType.DIFFUSION,
            yaml_engine_args={
                "parallel_config": {
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 1,
                    "tensor_parallel_size": 4,
                    "enable_expert_parallel": False,
                    "ulysses_degree": 1,
                    "ring_degree": 1,
                    "ulysses_mode": "strict",
                    "sequence_parallel_size": 1,
                    "cfg_parallel_size": 1,
                    "vae_patch_parallel_size": 1,
                    "use_hsdp": False,
                    "hsdp_shard_size": -1,
                    "hsdp_replicate_size": 1,
                }
            },
            runtime_overrides={
                "pipeline_parallel_size": 2,
                "data_parallel_size": 3,
                "tensor_parallel_size": 8,
                "enable_expert_parallel": True,
                "ulysses_degree": 2,
                "ring_degree": 4,
                "ulysses_mode": "advanced_uaa",
                "sequence_parallel_size": 8,
                "cfg_parallel_size": 2,
                "vae_patch_parallel_size": 2,
                "use_hsdp": True,
                "hsdp_shard_size": 8,
                "hsdp_replicate_size": 2,
            },
        )

        omega_config = config.to_omegaconf()

        assert omega_config.engine_args.parallel_config.pipeline_parallel_size == 2
        assert omega_config.engine_args.parallel_config.data_parallel_size == 3
        assert omega_config.engine_args.parallel_config.tensor_parallel_size == 8
        assert omega_config.engine_args.parallel_config.enable_expert_parallel is True
        assert omega_config.engine_args.parallel_config.ulysses_degree == 2
        assert omega_config.engine_args.parallel_config.ring_degree == 4
        assert omega_config.engine_args.parallel_config.ulysses_mode == "advanced_uaa"
        assert omega_config.engine_args.parallel_config.sequence_parallel_size == 8
        assert omega_config.engine_args.parallel_config.cfg_parallel_size == 2
        assert omega_config.engine_args.parallel_config.vae_patch_parallel_size == 2
        assert omega_config.engine_args.parallel_config.use_hsdp is True
        assert omega_config.engine_args.parallel_config.hsdp_shard_size == 8
        assert omega_config.engine_args.parallel_config.hsdp_replicate_size == 2
        assert "pipeline_parallel_size" not in omega_config.engine_args
        assert "data_parallel_size" not in omega_config.engine_args
        assert "tensor_parallel_size" not in omega_config.engine_args
        assert "enable_expert_parallel" not in omega_config.engine_args
        assert "ulysses_degree" not in omega_config.engine_args
        assert "ring_degree" not in omega_config.engine_args
        assert "ulysses_mode" not in omega_config.engine_args
        assert "sequence_parallel_size" not in omega_config.engine_args
        assert "cfg_parallel_size" not in omega_config.engine_args
        assert "vae_patch_parallel_size" not in omega_config.engine_args
        assert "use_hsdp" not in omega_config.engine_args
        assert "hsdp_shard_size" not in omega_config.engine_args
        assert "hsdp_replicate_size" not in omega_config.engine_args

    def test_to_omegaconf_diffusion_parallel_overrides_create_parallel_config(self):
        config = StageConfig(
            stage_id=1,
            model_stage="diffusion",
            stage_type=StageType.DIFFUSION,
            runtime_overrides={
                "pipeline_parallel_size": 2,
                "data_parallel_size": 3,
                "tensor_parallel_size": 8,
                "enable_expert_parallel": True,
                "ulysses_degree": 2,
                "ring_degree": 4,
                "ulysses_mode": "advanced_uaa",
                "sequence_parallel_size": 8,
                "cfg_parallel_size": 2,
                "vae_patch_parallel_size": 2,
                "use_hsdp": True,
                "hsdp_shard_size": 8,
                "hsdp_replicate_size": 2,
            },
        )

        omega_config = config.to_omegaconf()

        assert omega_config.engine_args.parallel_config.pipeline_parallel_size == 2
        assert omega_config.engine_args.parallel_config.data_parallel_size == 3
        assert omega_config.engine_args.parallel_config.tensor_parallel_size == 8
        assert omega_config.engine_args.parallel_config.enable_expert_parallel is True
        assert omega_config.engine_args.parallel_config.ulysses_degree == 2
        assert omega_config.engine_args.parallel_config.ring_degree == 4
        assert omega_config.engine_args.parallel_config.ulysses_mode == "advanced_uaa"
        assert omega_config.engine_args.parallel_config.sequence_parallel_size == 8
        assert omega_config.engine_args.parallel_config.cfg_parallel_size == 2
        assert omega_config.engine_args.parallel_config.vae_patch_parallel_size == 2
        assert omega_config.engine_args.parallel_config.use_hsdp is True
        assert omega_config.engine_args.parallel_config.hsdp_shard_size == 8
        assert omega_config.engine_args.parallel_config.hsdp_replicate_size == 2
        assert "pipeline_parallel_size" not in omega_config.engine_args
        assert "data_parallel_size" not in omega_config.engine_args
        assert "tensor_parallel_size" not in omega_config.engine_args
        assert "enable_expert_parallel" not in omega_config.engine_args
        assert "ulysses_degree" not in omega_config.engine_args
        assert "ring_degree" not in omega_config.engine_args
        assert "ulysses_mode" not in omega_config.engine_args
        assert "sequence_parallel_size" not in omega_config.engine_args
        assert "cfg_parallel_size" not in omega_config.engine_args
        assert "vae_patch_parallel_size" not in omega_config.engine_args
        assert "use_hsdp" not in omega_config.engine_args
        assert "hsdp_shard_size" not in omega_config.engine_args
        assert "hsdp_replicate_size" not in omega_config.engine_args

    def test_to_omegaconf_diffusion_parallel_degree_overrides_recompute_sequence_parallel_size(self):
        config = StageConfig(
            stage_id=1,
            model_stage="diffusion",
            stage_type=StageType.DIFFUSION,
            yaml_engine_args={
                "parallel_config": {
                    "sequence_parallel_size": 1,
                    "ulysses_degree": 1,
                    "ring_degree": 1,
                }
            },
            runtime_overrides={
                "ulysses_degree": 2,
                "ring_degree": 4,
            },
        )

        omega_config = config.to_omegaconf()

        assert omega_config.engine_args.parallel_config.ulysses_degree == 2
        assert omega_config.engine_args.parallel_config.ring_degree == 4
        assert omega_config.engine_args.parallel_config.sequence_parallel_size == 8
        assert "ulysses_degree" not in omega_config.engine_args
        assert "ring_degree" not in omega_config.engine_args
        assert "sequence_parallel_size" not in omega_config.engine_args

    def test_to_omegaconf_diffusion_parallel_explicit_sequence_parallel_size_is_preserved(self):
        config = StageConfig(
            stage_id=1,
            model_stage="diffusion",
            stage_type=StageType.DIFFUSION,
            yaml_engine_args={
                "parallel_config": {
                    "sequence_parallel_size": 1,
                    "ulysses_degree": 1,
                    "ring_degree": 1,
                }
            },
            runtime_overrides={
                "ulysses_degree": 2,
                "ring_degree": 4,
                "sequence_parallel_size": 16,
            },
        )

        omega_config = config.to_omegaconf()

        assert omega_config.engine_args.parallel_config.ulysses_degree == 2
        assert omega_config.engine_args.parallel_config.ring_degree == 4
        assert omega_config.engine_args.parallel_config.sequence_parallel_size == 16

    def test_to_omegaconf_llm_parallel_overrides_remain_top_level(self):
        config = StageConfig(
            stage_id=0,
            model_stage="thinker",
            stage_type=StageType.LLM,
            runtime_overrides={
                "pipeline_parallel_size": 2,
                "data_parallel_size": 3,
                "tensor_parallel_size": 8,
            },
        )

        omega_config = config.to_omegaconf()

        assert omega_config.engine_args.pipeline_parallel_size == 2
        assert omega_config.engine_args.data_parallel_size == 3
        assert omega_config.engine_args.tensor_parallel_size == 8
        assert "pipeline_parallel_size" in omega_config.engine_args
        assert "data_parallel_size" in omega_config.engine_args
        assert "tensor_parallel_size" in omega_config.engine_args
        assert "parallel_config" not in omega_config.engine_args


class TestStageConfigFactory:
    """Tests for StageConfigFactory class."""

    def test_default_diffusion_no_yaml(self):
        """Test single-stage diffusion works without YAML config (@ZJY0516)."""
        kwargs = {
            "cache_backend": "none",
            "cache_config": None,
            "dtype": "bfloat16",
        }
        configs = StageConfigFactory.create_default_diffusion(kwargs)

        assert len(configs) == 1
        cfg = configs[0]
        assert cfg["stage_id"] == 0
        assert cfg["stage_type"] == "diffusion"
        assert cfg["final_output"] is True
        assert cfg["final_output_type"] == "image"

    def test_default_diffusion_with_parallel_config(self):
        """Test diffusion config calculates devices from parallel_config."""

        @dataclass
        class MockParallelConfig:
            world_size: int = 4

        kwargs = {
            "parallel_config": MockParallelConfig(),
            "cache_backend": "tea_cache",
        }
        configs = StageConfigFactory.create_default_diffusion(kwargs)

        assert configs[0]["runtime"]["devices"] == "0,1,2,3"

    def test_per_stage_override_precedence(self):
        """Test that --stage-0-gpu-memory-utilization overrides global."""
        stage = StageConfig(stage_id=0, model_stage="thinker", input_sources=[])
        cli_overrides = {
            "gpu_memory_utilization": 0.5,  # Global
            "stage_0_gpu_memory_utilization": 0.9,  # Per-stage override
        }

        overrides = StageConfigFactory._merge_cli_overrides(stage, cli_overrides)

        # Per-stage should override global
        assert overrides["gpu_memory_utilization"] == 0.9

    def test_cli_override_forwards_engine_registered_args(self):
        """Test that any engine-registered CLI arg is forwarded (@wuhang2014)."""
        stage = StageConfig(stage_id=0, model_stage="thinker", input_sources=[])
        cli_overrides = {
            "gpu_memory_utilization": 0.9,  # Well-known param
            "custom_engine_flag": True,  # Not orchestrator-owned, so forwarded
        }

        overrides = StageConfigFactory._merge_cli_overrides(stage, cli_overrides)

        assert overrides["gpu_memory_utilization"] == 0.9
        assert overrides["custom_engine_flag"] is True

    def test_cli_override_excludes_internal_keys(self):
        """Test that internal/orchestrator keys are not forwarded."""
        stage = StageConfig(stage_id=0, model_stage="thinker", input_sources=[])
        cli_overrides = {
            "gpu_memory_utilization": 0.9,
            "model": "some_model",  # Internal
            "stage_configs_path": "/path",  # Internal
            "batch_timeout": 10,  # Internal
        }

        overrides = StageConfigFactory._merge_cli_overrides(stage, cli_overrides)

        assert overrides["gpu_memory_utilization"] == 0.9
        assert "model" not in overrides
        assert "stage_configs_path" not in overrides
        assert "batch_timeout" not in overrides

    def test_per_stage_override_excludes_internal_keys(self):
        """Test that per-stage overrides also skip internal keys."""
        stage = StageConfig(stage_id=0, model_stage="thinker", input_sources=[])
        cli_overrides = {
            "stage_0_gpu_memory_utilization": 0.9,
            "stage_0_model": "override_model",  # Internal, should be skipped
            "stage_0_batch_timeout": 5,  # Internal, should be skipped
        }

        overrides = StageConfigFactory._merge_cli_overrides(stage, cli_overrides)

        assert overrides["gpu_memory_utilization"] == 0.9
        assert "model" not in overrides
        assert "batch_timeout" not in overrides


class TestStageResolutionHelpers:
    """Tests for shared stage override / filtering helpers."""

    def test_build_stage_runtime_overrides_ignores_other_stage_and_internal_keys(self):
        # Pass the same filter set the function uses by default
        # (orchestrator-only fields plus SHARED_FIELDS so ``model`` is
        # treated as not-per-stage-overridable).
        overrides = build_stage_runtime_overrides(
            0,
            {
                "gpu_memory_utilization": 0.5,
                "stage_0_gpu_memory_utilization": 0.9,
                "stage_1_gpu_memory_utilization": 0.1,
                "stage_0_model": "should_be_ignored",
                "parallel_config": {"world_size": 2},
            },
            internal_keys=internal_blacklist_keys() | SHARED_FIELDS,
        )

        assert overrides["gpu_memory_utilization"] == 0.9
        assert "model" not in overrides
        assert "parallel_config" not in overrides

    def test_strip_parent_engine_args_reports_only_surprising_parent_overrides(self):
        parent_fields = {f.name: f for f in fields(EngineArgs)}
        filtered, overridden = strip_parent_engine_args(
            {
                "model": "some/model",
                "stage_configs_path": "/tmp/stages.yaml",
                "tensor_parallel_size": 4,
                "worker_extension_cls": "some.Extension",
                "custom_pipeline_args": {"pipeline_class": "demo.Pipeline"},
            },
            parent_fields=parent_fields,
            keep_keys={"worker_extension_cls"},
            strip_keys={"stage_configs_path"},
            no_warn_keys={"model"},
        )

        assert filtered == {
            "worker_extension_cls": "some.Extension",
            "custom_pipeline_args": {"pipeline_class": "demo.Pipeline"},
        }
        assert overridden == ["tensor_parallel_size"]

    def test_strip_parent_engine_args_keeps_allowed_media_access_controls(self):
        parent_fields = {f.name: f for f in fields(EngineArgs)}
        filtered, overridden = strip_parent_engine_args(
            {
                "model": "some/model",
                "stage_configs_path": "/tmp/stages.yaml",
                "allowed_local_media_path": "/data/qwentts",
                "allowed_media_domains": ["example.com"],
            },
            parent_fields=parent_fields,
            keep_keys={"allowed_local_media_path", "allowed_media_domains"},
            strip_keys={"stage_configs_path"},
            no_warn_keys={"model"},
        )

        assert filtered == {
            "allowed_local_media_path": "/data/qwentts",
            "allowed_media_domains": ["example.com"],
        }
        assert overridden == []


class TestPipelineDiscovery:
    """Tests for the central pipeline registry (``OMNI_PIPELINES``)."""

    def test_registry_has_known_models(self):
        """Check that specific models are in OMNI_PIPELINES."""
        assert "qwen2_5_omni" in OMNI_PIPELINES
        assert "qwen3_omni_moe" in OMNI_PIPELINES
        assert "qwen3_tts" in OMNI_PIPELINES

    def test_registry_resolver_qwen3_omni_all_stages(self):
        """Test that providing the HF config for qwen3 omni with audio enabled uses all stages."""
        pipeline = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(pipeline, PipelineConfig)
        assert pipeline.model_type == "qwen3_omni_moe"
        assert pipeline.default_deploy_config_name == "qwen3_omni_moe.yaml"
        assert len(pipeline.stages) == 3  # thinker + talker + code2wav

    def test_registry_resolver_qwen3_omni_thinker_only(self):
        """Test that providing the HF config for qwen3 omni without audio is thinker only."""
        pipeline = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_THINKER_HF_CONFIG,
        )
        assert isinstance(pipeline, PipelineConfig)
        assert pipeline.model_type == "qwen3_omni_moe_thinker_only"
        assert pipeline.default_deploy_config_name is None
        assert len(pipeline.stages) == 1  # thinker only

    @pytest.mark.parametrize(
        "pipeline",
        [registered for registered in OMNI_PIPELINES.values() if isinstance(registered, PipelineConfig)],
        ids=lambda pipeline: pipeline.model_type,
    )
    def test_registered_default_deploy_config_exists(self, pipeline):
        if pipeline.default_deploy_config_name is None:
            return
        assert Path(get_deploy_config_path(pipeline.default_deploy_config_name)).is_file()

    def test_registry_returns_none_for_unknown(self):
        """Unknown model_types aren't found and resolve to `None`."""
        assert "definitely_not_a_real_model" not in OMNI_PIPELINES
        assert OMNI_PIPELINES.get("definitely_not_a_real_model") is None
        assert resolve_pipeline_config("definitely_not_a_real_model") is None

    def test_pipeline_config_supports_hf_architectures(self):
        """PipelineConfig accepts hf_architectures for HF-arch fallback
        (replaces the old _ARCHITECTURE_MODELS dict)."""
        p = PipelineConfig(
            model_type="custom_collide",
            hf_architectures=("SomeCollidingArch",),
        )
        assert p.hf_architectures == ("SomeCollidingArch",)


class TestStagePipelineConfig:
    def test_frozen(self):
        s = StagePipelineConfig(stage_id=0, model_stage="a")
        with pytest.raises(AttributeError):
            s.model_stage = "changed"

    def test_defaults(self):
        s = StagePipelineConfig(stage_id=0, model_stage="a")
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.input_sources == ()
        assert s.final_output is False
        assert s.sampling_constraints == {}
        assert s.engine_output_type is None
        assert s.scheduler_cls is None


class TestPipelineConfigNew:
    def test_frozen(self):
        p = PipelineConfig(model_type="t", model_arch="A")
        with pytest.raises(AttributeError):
            p.model_type = "changed"

    def test_validate_valid(self):
        p = PipelineConfig(
            model_type="t",
            model_arch="A",
            stages=(
                StagePipelineConfig(stage_id=0, model_stage="a"),
                StagePipelineConfig(stage_id=1, model_stage="b", input_sources=(0,)),
            ),
        )
        assert p.validate() == []

    def test_validate_no_stages(self):
        p = PipelineConfig(model_type="t", model_arch="A")
        assert any("no stages" in e.lower() for e in p.validate())


class TestPipelineRegistration:
    def test_resolve_pipeline_prefers_deploy_pipeline_key(self, clean_pipeline_registry, tmp_path):
        deploy_key = "deploy_selected_pipeline"
        model_type_key = "hf_model_type_pipeline"
        deploy_pipe = PipelineConfig(model_type=deploy_key)
        model_type_pipe = PipelineConfig(model_type=model_type_key)
        register_pipeline(deploy_pipe)
        register_pipeline(model_type_pipe)

        deploy_path = tmp_path / "deploy.yaml"
        deploy_path.write_text(f"pipeline: {deploy_key}\n", encoding="utf-8")

        class FakeConfig(PretrainedConfig):
            model_type = model_type_key

        fake_config = FakeConfig()
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=str(deploy_path),
            )

        assert pipeline_cfg is deploy_pipe

    def test_resolve_pipeline_rejects_invalid_deploy_pipeline_key(
        self,
        clean_pipeline_registry,
        tmp_path,
    ):
        model_type_key = "registered_model_type_pipeline"
        register_pipeline(PipelineConfig(model_type=model_type_key))

        deploy_path = tmp_path / "deploy.yaml"
        deploy_path.write_text("pipeline: missing_pipeline\n", encoding="utf-8")

        class FakeConfig(PretrainedConfig):
            model_type = model_type_key

        with patch("vllm_omni.config.config_factory.get_config", return_value=FakeConfig()):
            with pytest.raises(KeyError, match="missing_pipeline"):
                StageConfigFactory.get_pipeline_config(
                    "fake/model",
                    trust_remote_code=True,
                    deploy_config_path=str(deploy_path),
                )
            with pytest.raises(KeyError, match="missing_pipeline"):
                StageConfigFactory.create_from_model(
                    "fake/model",
                    trust_remote_code=False,
                    cli_overrides={},
                    deploy_config_path=str(deploy_path),
                )
            with pytest.raises(KeyError, match="missing_pipeline"):
                StageConfigFactory.create_legacy_stage_configs_from_model(
                    "fake/model",
                    trust_remote_code=False,
                    cli_overrides={},
                    deploy_config_path=str(deploy_path),
                )

    def test_resolve_pipeline_matches_hf_architecture_fallback(self, clean_pipeline_registry):
        pipeline_key = "architecture_fallback_pipeline"
        pipe_cfg = PipelineConfig(
            model_type=pipeline_key,
            hf_architectures=("ArchitectureFallbackForTest",),
        )
        register_pipeline(pipe_cfg)

        class FakeConfig(PretrainedConfig):
            model_type = "unregistered_model_type"

        fake_config = FakeConfig()
        fake_config.architectures = ["ArchitectureFallbackForTest"]
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=None,
            )

        assert pipeline_cfg is pipe_cfg

    def test_resolve_pipeline_architecture_fallback_respects_predicates(self, clean_pipeline_registry):
        shared_arch = "SharedPredicateArchitectureForTest"

        def rejecting_predicate(_hf_config):
            return False

        def raising_predicate(_hf_config):
            raise ValueError("predicate failure")

        reject_cfg = PipelineConfig(
            model_type="rejected_by_predicate",
            hf_architectures=(shared_arch,),
            hf_config_predicate=rejecting_predicate,
        )
        raises_cfg = PipelineConfig(
            model_type="raises_in_predicate",
            hf_architectures=(shared_arch,),
            hf_config_predicate=raising_predicate,
        )
        accept_cfg = PipelineConfig(
            model_type="accepted_by_predicate",
            hf_architectures=(shared_arch,),
        )
        register_pipeline(reject_cfg)
        register_pipeline(raises_cfg)
        register_pipeline(accept_cfg)

        class FakeConfig(PretrainedConfig):
            model_type = "unregistered_model_type"

        fake_config = FakeConfig()
        fake_config.architectures = [shared_arch]
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=None,
            )

        assert pipeline_cfg is accept_cfg

    def test_resolve_pipeline_architecture_fallback_rejects_non_matching_architecture(
        self,
        clean_pipeline_registry,
    ):
        predicate_calls = []

        def predicate(_hf_config):
            predicate_calls.append(_hf_config)
            return True

        pipe_cfg = PipelineConfig(
            model_type="predicate_without_arch_match",
            hf_architectures=("DifferentArchitectureForTest",),
            hf_config_predicate=predicate,
        )
        register_pipeline(pipe_cfg)

        class FakeConfig(PretrainedConfig):
            model_type = "unregistered_model_type"

        fake_config = FakeConfig()
        fake_config.architectures = ["RequestedArchitectureForTest"]
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=None,
            )

        assert pipeline_cfg is None
        assert predicate_calls == [fake_config]

    def test_resolve_pipeline_architecture_fallback_resolves_callable_pipeline(self, clean_pipeline_registry):
        pipeline_key = "callable_architecture_fallback"
        resolved_cfg = PipelineConfig(
            model_type="callable_resolved_pipeline",
            hf_architectures=("CallableArchitectureForTest",),
        )
        seen_hf_configs = []

        def resolver(hf_config):
            seen_hf_configs.append(hf_config)
            return resolved_cfg

        register_pipeline(resolver, model_type=pipeline_key)

        class FakeConfig(PretrainedConfig):
            model_type = "unregistered_model_type"

        fake_config = FakeConfig()
        fake_config.architectures = ["CallableArchitectureForTest"]
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=None,
            )

        assert pipeline_cfg is resolved_cfg
        assert seen_hf_configs == [fake_config]

    def test_resolve_pipeline_returns_none_without_registry_or_architecture_match(self, clean_pipeline_registry):
        class FakeConfig(PretrainedConfig):
            model_type = "unregistered_model_type"

        fake_config = FakeConfig()
        fake_config.architectures = ["UnmatchedArchitectureForTest"]
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=None,
            )

        assert pipeline_cfg is None

    def test_create_from_model_returns_structured_omni_config(self):
        class FakeConfig(PretrainedConfig):
            model_type = "qwen3_tts"

        with patch("vllm_omni.config.config_factory.get_config", return_value=FakeConfig()):
            omni_config = StageConfigFactory.create_from_model(
                "fake/model",
                trust_remote_code=False,
                cli_overrides={},
                deploy_config_path=None,
            )

        assert isinstance(omni_config, VllmOmniConfig)
        assert omni_config.pipeline_config is OMNI_PIPELINES["qwen3_tts"]
        assert len(omni_config.stage_configs) == 2

    def test_create_from_model_preserves_model_on_structured_diffusion_stage(self):
        class FakeConfig(PretrainedConfig):
            model_type = "dreamzero"

        with patch("vllm_omni.config.config_factory.get_config", return_value=FakeConfig()):
            omni_config = StageConfigFactory.create_from_model(
                "fake/model",
                trust_remote_code=False,
                cli_overrides={},
                deploy_config_path=None,
            )

        assert isinstance(omni_config, VllmOmniConfig)
        assert omni_config.stage_by_id(0).diffusion_config.model == "fake/model"

    def test_pipeline_registration(self, clean_pipeline_registry):
        """Ensure that we can register and create a custom pipeline config."""
        new_model_type = "new_model_type"
        pipe_cfg = PipelineConfig(model_type=new_model_type)

        # Register the new PipelineConfig
        assert new_model_type not in OMNI_PIPELINES
        register_pipeline(pipe_cfg)
        assert new_model_type in OMNI_PIPELINES
        assert OMNI_PIPELINES[new_model_type] is pipe_cfg

        class FakeConfig(PretrainedConfig):
            model_type = new_model_type

        # Create the model
        fake_config = FakeConfig()
        with (
            patch("vllm_omni.config.config_factory.get_config", return_value=fake_config),
            patch.object(VllmOmniConfig, "from_pipeline_config") as mock_create,
        ):
            StageConfigFactory.create_from_model(
                "fake/model",
                trust_remote_code=False,
                cli_overrides={},
                deploy_config_path=None,
            )
            mock_create.assert_called_once()
            assert mock_create.call_args.args == (pipe_cfg,)
            assert mock_create.call_args.kwargs["cli_overrides"] == {
                "trust_remote_code": False,
                "model": "fake/model",
            }
        assert pipe_cfg.model_type == new_model_type

    def test_resolver_registration(self, clean_pipeline_registry):
        """Ensure that we can register and create a custom resolver."""
        new_model_type = "new_model_type"
        resolved_type = "resolved_type"

        class FakeConfig(PretrainedConfig):
            model_type = new_model_type

        @pipeline_cfg_resolver(config_type=FakeConfig)
        def custom_resolver(
            hf_config: FakeConfig,
        ) -> PipelineConfig:
            return PipelineConfig(model_type=resolved_type)

        # Register the new PipelineConfig
        assert new_model_type not in OMNI_PIPELINES

        register_pipeline(custom_resolver, model_type=new_model_type)
        assert new_model_type in OMNI_PIPELINES
        assert OMNI_PIPELINES[new_model_type] is custom_resolver

        # Create the model
        fake_config = FakeConfig()
        with (
            patch("vllm_omni.config.config_factory.get_config", return_value=fake_config) as mock_get_config,
            patch.object(VllmOmniConfig, "from_pipeline_config") as mock_create,
        ):
            StageConfigFactory.create_from_model(
                "fake/model",
                trust_remote_code=True,
                cli_overrides={},
                deploy_config_path=None,
            )
            mock_create.assert_called_once()
            assert mock_create.call_args.args[0].model_type == resolved_type
            assert mock_create.call_args.kwargs["cli_overrides"] == {
                "trust_remote_code": True,
                "model": "fake/model",
            }
            mock_get_config.assert_called_once_with("fake/model", trust_remote_code=True)

        with patch.object(StageConfigFactory, "_create_legacy_from_registry", return_value=([], None)) as mock_legacy:
            StageConfigFactory.create_legacy_stage_configs_from_model(
                "fake/model",
                trust_remote_code=True,
                cli_overrides={},
                deploy_config_path=None,
            )

        assert mock_legacy.call_args.args[0].model_type == resolved_type
        assert mock_legacy.call_args.args[1]["trust_remote_code"] is True

    def test_resolve_when_autodetect_resolves_none(self):
        """Regression test for: https://github.com/vllm-project/vllm-omni/issues/4726"""
        deploy_path = get_deploy_config_path("ming_tts.yaml")
        resolved_config = StageConfigFactory.create_from_model(
            model="inclusionAI/Ming-omni-tts-0.5B",
            trust_remote_code=False,
            cli_overrides={},
            deploy_config_path=deploy_path,
        )
        assert resolved_config is not None
        assert len(resolved_config.stage_configs) > 0

    def test_legacy_and_structured_paths_share_deploy_pipeline_override(self, clean_pipeline_registry, tmp_path):
        pipeline_key = "deploy_only_pipeline"
        pipe_cfg = PipelineConfig(
            model_type=pipeline_key,
            model_arch="DeployOnlyArch",
            stages=(StagePipelineConfig(stage_id=0, model_stage="diffusion"),),
        )
        register_pipeline(pipe_cfg)

        deploy_path = tmp_path / "deploy_only.yaml"
        deploy_path.write_text(f"pipeline: {pipeline_key}\n", encoding="utf-8")

        class FakeConfig(PretrainedConfig):
            model_type = "dense"
            architectures = ("UnregisteredArch",)

        with patch("vllm_omni.config.config_factory.get_config", return_value=FakeConfig()):
            omni_config = StageConfigFactory.create_from_model(
                "fake/model",
                trust_remote_code=False,
                cli_overrides={},
                deploy_config_path=str(deploy_path),
            )
            legacy_configs, _ = StageConfigFactory.create_legacy_stage_configs_from_model(
                "fake/model",
                trust_remote_code=False,
                cli_overrides={},
                deploy_config_path=str(deploy_path),
            )

        assert omni_config is not None
        assert omni_config.pipeline_config is pipe_cfg
        assert legacy_configs is not None
        assert len(legacy_configs) == 1
        assert legacy_configs[0].yaml_engine_args["model_arch"] == "DeployOnlyArch"

    def test_structured_path_loads_explicit_deploy_config_once(self, clean_pipeline_registry, tmp_path):
        pipeline_key = "single_load_pipeline"
        register_pipeline(PipelineConfig(model_type=pipeline_key))
        deploy_path = tmp_path / "single_load.yaml"
        deploy_path.write_text(f"pipeline: {pipeline_key}\n", encoding="utf-8")

        class FakeConfig(PretrainedConfig):
            model_type = "unregistered"

        with (
            patch("vllm_omni.config.config_factory.get_config", return_value=FakeConfig()),
            patch(
                "vllm_omni.config.config_factory.load_deploy_config",
                wraps=load_deploy_config,
            ) as mock_load,
        ):
            StageConfigFactory.create_from_model(
                "fake/model",
                trust_remote_code=False,
                cli_overrides={},
                deploy_config_path=str(deploy_path),
            )

        mock_load.assert_called_once_with(deploy_path)

    def test_legacy_default_deploy_name_ignores_cwd(self, monkeypatch, tmp_path):
        default_name = "pipeline_default.yaml"
        (tmp_path / default_name).write_text("stages: []\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        pipeline = PipelineConfig(
            model_type="pipeline_with_default",
            default_deploy_config_name=default_name,
        )

        with patch(
            "vllm_omni.config.config_factory.load_deploy_config",
            return_value=DeployConfig(),
        ) as mock_load:
            StageConfigFactory._create_legacy_from_registry(pipeline, {})

        mock_load.assert_called_once_with(_DEPLOY_DIR / default_name)

    def test_deploy_override_uses_correct_endpoint_restrictions(self, clean_pipeline_registry, tmp_path):
        """Ensure endpoint restrictions must come from the final pipeline
        after deploy config overrides, not the auto-detected pipeline.
        """
        # Register two pipeline configs, where one has an endpoint restriction, and one doesn't
        restriction = EndpointRestriction(
            OmniServingCapability.COMPLETIONS,
            "pipeline_a blocks completions",
        )
        pipe_a = PipelineConfig(model_type="detect_type", endpoint_restrictions=(restriction,))
        pipe_b = PipelineConfig(model_type="override_type", endpoint_restrictions=())
        register_pipeline(pipe_a)
        register_pipeline(pipe_b)

        # Create a config with the autodetected type, and write the
        # deploy config specifying the override type to a temp path
        class FakeConfig(PretrainedConfig):
            model_type = "detect_type"

        deploy_yaml = tmp_path / "override.yaml"
        deploy_yaml.write_text("pipeline: override_type\n")

        # Get the endpoint restrictions, passing the deploy config with the override
        # type + patching the config for the detected type. Ensure that the endpoint
        # restrictions correspond to the type in the deploy config.
        with patch(
            "vllm_omni.config.config_factory.get_config",
            return_value=FakeConfig(),
        ):
            restrictions = StageConfigFactory.get_pipeline_endpoint_restrictions(
                model="fake/model",
                trust_remote_code=False,
                deploy_config_path=str(deploy_yaml),
            )

        assert restrictions == ()


class TestResolveScheduler:
    def test_all_execution_types_handled(self):
        for et in StageExecutionType:
            _resolve_scheduler(et)

    def test_ar_sync_when_false(self):
        cls = _resolve_scheduler(StageExecutionType.LLM_AR, async_scheduling=False)
        assert cls is not None
        assert "Async" not in cls.__name__

    def test_ar_async_when_true(self):
        cls = _resolve_scheduler(StageExecutionType.LLM_AR, async_scheduling=True)
        assert cls is not None
        assert "Async" in cls.__name__

    def test_generation(self):
        cls = _resolve_scheduler(StageExecutionType.LLM_GENERATION)
        assert cls is not None
        assert "Generation" in cls.__name__

    def test_diffusion_returns_none(self):
        assert _resolve_scheduler(StageExecutionType.DIFFUSION) is None


class TestDeployConfigLoading:
    def test_custom_voice_dir_is_pipeline_wide_engine_arg(self, tmp_path):
        deploy_path = tmp_path / "qwen3_tts_custom_voice.yaml"
        custom_voice_dir = tmp_path / "voices"
        deploy_path.write_text(
            f"""
async_chunk: true
custom_voice_dir: {custom_voice_dir}
stages:
  - stage_id: 0
    devices: "0"
  - stage_id: 1
    devices: "0"
""",
            encoding="utf-8",
        )

        deploy = load_deploy_config(deploy_path)
        pipeline = resolve_pipeline_config("qwen3_tts")
        stages = merge_pipeline_deploy(pipeline, deploy)
        assert isinstance(pipeline, PipelineConfig)

        assert deploy.custom_voice_dir == str(custom_voice_dir)
        assert {s.yaml_engine_args.get("custom_voice_dir") for s in stages} == {str(custom_voice_dir)}

    def test_load_qwen3_omni_moe_deploy_config(self):
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        deploy = load_deploy_config(deploy_path)
        assert len(deploy.stages) == 3
        assert deploy.async_chunk is True
        assert deploy.connectors is not None
        assert deploy.platforms is not None

    def test_load_voxtral_tts_deploy_config_schema_fields(self):
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "voxtral_tts.yaml"
        deploy = load_deploy_config(deploy_path)
        assert deploy.stages[0].config_format == "mistral"
        assert deploy.stages[0].load_format == "mistral"
        assert deploy.stages[0].tokenizer_mode == "mistral"
        assert not any(
            name in deploy.stages[0].engine_extras for name in ("config_format", "load_format", "tokenizer_mode")
        )

    def test_load_ming_flash_omni_deploy_config_schema_fields(self):
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "ming_flash_omni.yaml"
        deploy = load_deploy_config(deploy_path)
        assert deploy.stages[0].compilation_config == {"pass_config": {"fuse_allreduce_rms": False}}
        assert "compilation_config" not in deploy.stages[0].engine_extras

    def test_load_voxcpm2_deploy_config_preserves_engine_extras(self):
        deploy_path = get_deploy_config_path("voxcpm2.yaml")
        raw_stage = get_deploy_config_stage("voxcpm2.yaml", 0)
        expected_runtime_config = raw_stage["engine_extras"]["hf_overrides"]["voxcpm2_runtime_config"]

        deploy = load_deploy_config(deploy_path)
        runtime_config = deploy.stages[0].engine_extras["hf_overrides"]["voxcpm2_runtime_config"]
        assert runtime_config == expected_runtime_config

    @pytest.mark.parametrize(
        ("deploy_name", "pipeline_name", "stage_count", "final_output_type", "declared_pipeline"),
        [
            ("mammoth_moda2.yaml", "mammoth_moda2", 2, "image", "mammoth_moda2"),
            ("mammoth_moda2_ar.yaml", "mammoth_moda2_ar", 1, "text", "mammoth_moda2_ar"),
            ("omnivoice.yaml", "omnivoice", 1, "audio", "omnivoice"),
            ("mimo_audio.yaml", "mimo_audio", 2, "audio", None),
            ("step_audio_2.yaml", "step_audio_2", 2, "audio", None),
            ("step_audio_2_asr.yaml", "step_audio_2_asr", 1, "text", "step_audio_2_asr"),
            ("step_audio_2_async_chunk.yaml", "step_audio_2", 2, "audio", None),
            ("hunyuan_video_15.yaml", "hunyuan_video_15", 1, "video", None),
            ("wan2_2_ti2v.yaml", "wan2_2_ti2v", 1, "video", None),
        ],
    )
    def test_load_new_registry_backed_deploy_configs(
        self,
        deploy_name: str,
        pipeline_name: str,
        stage_count: int,
        final_output_type: str,
        declared_pipeline: str | None,
    ):
        deploy_path = Path(get_deploy_config_path(deploy_name))
        deploy = load_deploy_config(deploy_path)
        assert deploy.pipeline == declared_pipeline

        with patch("vllm_omni.platforms.current_omni_platform") as platform:
            platform.device_name = "cuda"
            stages = merge_pipeline_deploy(OMNI_PIPELINES[pipeline_name], deploy)
        assert len(stages) == stage_count
        assert stages[-1].final_output is True
        assert stages[-1].final_output_type == final_output_type
        if deploy.trust_remote_code is not None:
            assert {s.yaml_engine_args.get("trust_remote_code") for s in stages} == {deploy.trust_remote_code}

    @pytest.mark.parametrize(
        ("config_json", "model_index", "expected_pipeline"),
        [
            ({"model_type": "step_audio_2"}, None, "step_audio_2"),
            (None, {"_class_name": "HunyuanVideo15Pipeline"}, "hunyuan_video_15"),
            (None, {"_class_name": "WanPipeline"}, "wan2_2_ti2v"),
        ],
    )
    def test_migrated_models_are_discovered_without_explicit_deploy(
        self,
        config_json,
        model_index,
        expected_pipeline,
    ):
        def get_model_file(filename, _model, revision=None):
            del revision
            if filename == "config.json":
                return config_json
            if filename == "model_index.json":
                return model_index
            return None

        with (
            patch.object(StageConfigFactory, "get_hf_config", return_value=None),
            patch("vllm_omni.config.config_factory.get_hf_file_to_dict", side_effect=get_model_file),
        ):
            pipeline = StageConfigFactory.get_pipeline_config(
                model="/models/unrelated-checkpoint-name",
                trust_remote_code=False,
            )

        assert pipeline is not None
        assert pipeline.model_type == expected_pipeline

    @pytest.mark.parametrize("deploy_name", ["step_audio_2.yaml", "step_audio_2_async_chunk.yaml"])
    def test_step_audio2_deploy_configs_fit_two_gpus(self, deploy_name):
        deploy = load_deploy_config(Path(get_deploy_config_path(deploy_name)))

        assert deploy.stages[0].devices == "0,1"
        assert deploy.stages[0].tensor_parallel_size == 2
        assert deploy.stages[1].devices == "1"

    def test_step_audio2_dispatches_sync_and_async_chunk_processors(self):
        pipeline = resolve_pipeline_config("step_audio_2")
        assert isinstance(pipeline, PipelineConfig)

        sync_stages = merge_pipeline_deploy(pipeline, DeployConfig(async_chunk=False))
        assert "custom_process_next_stage_input_func" not in sync_stages[0].yaml_engine_args
        assert sync_stages[1].custom_process_input_func.endswith("thinker2token2wav")

        async_stages = merge_pipeline_deploy(pipeline, DeployConfig(async_chunk=True))
        assert (
            async_stages[0]
            .yaml_engine_args["custom_process_next_stage_input_func"]
            .endswith("thinker2token2wav_async_chunk")
        )
        assert async_stages[1].custom_process_input_func is None

    def test_no_bundled_legacy_stage_config_yamls(self):
        stage_config_dir = Path(__file__).parent.parent / "vllm_omni" / "model_executor" / "stage_configs"
        assert not list(stage_config_dir.glob("*.yaml"))

    def test_merge_pipeline_deploy(self):
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        pipeline = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(pipeline, PipelineConfig)

        deploy = load_deploy_config(deploy_path)
        stages = merge_pipeline_deploy(pipeline, deploy)

        assert len(stages) == 3
        s0 = stages[0]
        assert s0.model_stage == "thinker"
        assert s0.yaml_engine_args["model_arch"] == "Qwen3OmniMoeForConditionalGeneration"
        assert s0.yaml_engine_args["engine_output_type"] == "latent"
        assert s0.yaml_extras["default_sampling_params"]["detokenize"] is True

    def test_merge_pipeline_deploy_preserves_num_replicas(self, tmp_path):
        pipeline = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(pipeline, PipelineConfig)

        base = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not base.exists():
            pytest.skip("Deploy config not found")

        overlay = tmp_path / "multi_replicas.yaml"
        overlay.write_text(f'base_config: {base}\nstages:\n  - stage_id: 1\n    devices: "1,2"\n    num_replicas: 2\n')

        deploy = load_deploy_config(overlay)
        assert deploy.stages[1].num_replicas == 2

        stages = merge_pipeline_deploy(pipeline, deploy)
        assert stages[1].yaml_runtime["devices"] == "1,2"
        assert stages[1].yaml_runtime["num_replicas"] == 2

    def test_merge_pipeline_deploy_preserves_requires_multimodal_data(self):
        pipeline = PipelineConfig(
            model_type="test_mm",
            model_arch="TestModel",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="ar",
                    execution_type=StageExecutionType.LLM_AR,
                    requires_multimodal_data=True,
                ),
            ),
        )
        deploy = DeployConfig(async_chunk=False, stages=[StageDeployConfig(stage_id=0)])

        stages = merge_pipeline_deploy(pipeline, deploy)

        assert stages[0].yaml_runtime["requires_multimodal_data"] is True

    def test_merge_pipeline_deploy_preserves_pipeline_scheduler_cls(self):
        scheduler_cls = "tests.fake.CustomScheduler"
        pipeline = PipelineConfig(
            model_type="test_scheduler",
            model_arch="TestModel",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="ar",
                    execution_type=StageExecutionType.LLM_AR,
                    scheduler_cls=scheduler_cls,
                ),
            ),
        )
        deploy = DeployConfig(async_chunk=False, stages=[StageDeployConfig(stage_id=0)])

        stages = merge_pipeline_deploy(pipeline, deploy)

        assert stages[0].scheduler_cls == scheduler_cls

    def test_mixed_schema_preserves_flat_fields(self):
        """Ensure flat fields are not dropped when engine_args are present."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "gpu_memory_utilization": 0.7,
                    "max_num_seqs": 8,
                    "engine_args": {
                        "tensor_parallel_size": 4,
                        "enforce_eager": True,
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert len(deploy.stages) == 1
        stage = deploy.stages[0]
        # Check that the engine args are set
        assert stage.tensor_parallel_size == 4
        assert stage.enforce_eager is True
        # Check that the other settings are also preserved
        assert stage.gpu_memory_utilization == 0.7
        assert stage.max_num_seqs == 8

    def test_engine_parse_engine_fields(self):
        """Test that we correctly parse & recursively merge stage deploy fields."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "compilation_config": {
                        "encoder_cudagraph_token_budgets": [1024, 2048],
                        "pass_config": {"fuse_norm_quant": True},
                    },
                    "engine_args": {
                        "compilation_config": {
                            "cudagraph_mm_encoder": True,
                            "pass_config": {"fuse_allreduce_rms": False},
                        },
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert len(deploy.stages) == 1
        stage = deploy.stages[0]
        assert stage.compilation_config == {
            "encoder_cudagraph_token_budgets": [1024, 2048],
            "pass_config": {
                "fuse_norm_quant": True,
                "fuse_allreduce_rms": False,
            },
            "cudagraph_mm_encoder": True,
        }

    def test_engine_extras_deep_merges_dicts_simple(self):
        """Ensure dictionary valued keys merge properly for top level dicts."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "foo": {"a": 111, "b": 1},
                    "engine_args": {
                        "foo": {"b": 2, "c": 3},
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert len(deploy.stages) == 1
        stage = deploy.stages[0]
        assert "foo" in stage.engine_extras
        assert stage.engine_extras["foo"] == {"a": 111, "b": 2, "c": 3}

    def test_engine_extras_dict_type_mismatch(self):
        """Ensure that we handle type mismatches with nested dicts correctly."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "foo": {"b": {1: 1}},
                    "engine_args": {
                        "foo": {"b": 2},
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert len(deploy.stages) == 1
        stage = deploy.stages[0]
        assert "foo" in stage.engine_extras
        assert stage.engine_extras["foo"] == {"b": 2}

    def test_mixed_engine_extras_deep_merges_dicts(self):
        """Ensure dictionary valued keys merge properly for nested dicts."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "foo": {"a": 111, "b": {"e": 199}},
                    "engine_args": {
                        "foo": {"b": {"d": 9}, "c": 3},
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert len(deploy.stages) == 1
        stage = deploy.stages[0]
        assert "foo" in stage.engine_extras
        assert stage.engine_extras["foo"] == {"a": 111, "b": {"d": 9, "e": 199}, "c": 3}

    def test_explicit_engine_extras_merge_with_flat_and_engine_args(self):
        """Explicit engine_extras should be preserved with existing pass-through styles."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "engine_extras": {
                        "hf_overrides": {
                            "runtime_config": {
                                "explicit_only": True,
                                "shared": "explicit",
                                "nested": {"a": 1},
                            }
                        }
                    },
                    "hf_overrides": {
                        "runtime_config": {
                            "flat_only": True,
                            "shared": "flat",
                            "nested": {"b": 2},
                        }
                    },
                    "engine_args": {
                        "hf_overrides": {
                            "runtime_config": {
                                "engine_args_only": True,
                                "shared": "engine_args",
                                "nested": {"c": 3},
                            }
                        }
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert deploy.stages[0].engine_extras["hf_overrides"] == {
            "runtime_config": {
                "engine_args_only": True,
                "explicit_only": True,
                "flat_only": True,
                "nested": {"a": 1, "b": 2, "c": 3},
                "shared": "engine_args",
            }
        }

    def test_deep_merge_does_not_mutate_inputs(self):
        """Merging engine_args must not mutate the base stage dict."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "foo": {"a": 1, "b": {"x": 10, "y": 20}},
                    "engine_args": {
                        "foo": {"b": {"y": 99, "z": 30}, "c": 3},
                    },
                },
            ]
        }
        original_foo = fake_config["stages"][0]["foo"]
        original_b = original_foo["b"]

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        # Values should merge into the new dict correctly
        assert deploy.stages[0].engine_extras["foo"] == {
            "a": 1,
            "b": {"x": 10, "y": 99, "z": 30},
            "c": 3,
        }
        # But the original nested dicts foo / b should be unchanged, since
        # we recursively shallow copy to avoid mutating in place.
        assert original_foo == {"a": 1, "b": {"x": 10, "y": 20}}
        assert original_b == {"x": 10, "y": 20}

    def test_mixed_schema_engine_args_wins_scalars(self):
        """engine_args takes precedence over flat fields for scalar conflicts."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "gpu_memory_utilization": 0.7,
                    "engine_args": {
                        "gpu_memory_utilization": 0.5,
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert deploy.stages[0].gpu_memory_utilization == 0.5


class TestQwen3OmniPipeline:
    def test_registered(self):
        p = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(p, PipelineConfig)
        assert p.model_arch == "Qwen3OmniMoeForConditionalGeneration"
        assert len(p.stages) == 3
        assert p.validate() == []

    def test_thinker(self):
        p = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.owns_tokenizer is True
        assert s.engine_output_type == "latent"
        assert s.sampling_constraints["detokenize"] is True

    def test_talker(self):
        p = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(1)
        assert isinstance(s, StagePipelineConfig)
        assert s.input_sources == (0,)
        assert s.sampling_constraints["stop_token_ids"] == [2150]
        # thinker2talker was removed: sync_process_input_func always wins.
        assert s.custom_process_input_func is None
        assert s.sync_process_input_func is not None
        assert s.sync_process_input_func.endswith("thinker2talker_token_only")
        assert s.custom_process_next_stage_input_func is not None

    def test_code2wav(self):
        p = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(2)
        assert isinstance(s, StagePipelineConfig)
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"


class TestQwen2_5OmniPipeline:
    def test_registered(self):
        p = resolve_pipeline_config("qwen2_5_omni")
        assert isinstance(p, PipelineConfig)
        assert p.model_arch == "Qwen2_5OmniForConditionalGeneration"
        assert len(p.stages) == 3
        assert p.validate() == []

    def test_thinker(self):
        p = resolve_pipeline_config("qwen2_5_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.owns_tokenizer is True
        assert s.engine_output_type == "latent"
        assert s.requires_multimodal_data is True

    def test_talker(self):
        p = resolve_pipeline_config("qwen2_5_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(1)
        assert isinstance(s, StagePipelineConfig)
        assert s.input_sources == (0,)
        assert s.sampling_constraints["stop_token_ids"] == [8294]
        # thinker2talker was removed: qwen2_5_omni has no async_chunk support,
        # so sync_process_input_func always wins and custom_process_input_func
        # was dead code.
        assert s.custom_process_input_func is None
        assert s.sync_process_input_func is not None

    def test_code2wav(self):
        p = resolve_pipeline_config("qwen2_5_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(2)
        assert isinstance(s, StagePipelineConfig)
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"


class TestQwen3TTSPipeline:
    def test_registered(self):
        p = resolve_pipeline_config("qwen3_tts")
        assert isinstance(p, PipelineConfig)
        assert p is not None
        assert p.model_arch == "Qwen3TTSTalkerForConditionalGeneration"
        assert len(p.stages) == 2
        assert p.validate() == []

    def test_talker_stage(self):
        p = resolve_pipeline_config("qwen3_tts")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "qwen3_tts"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.owns_tokenizer is True
        assert s.engine_output_type == "latent"
        assert s.sampling_constraints["stop_token_ids"] == [2150]
        # Stage 0 inherits the pipeline-level model_arch
        assert s.model_arch is None

    def test_code2wav_stage_has_per_stage_model_arch(self):
        p = resolve_pipeline_config("qwen3_tts")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(1)
        assert isinstance(s, StagePipelineConfig)
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"
        # Per-stage model_arch override (different from pipeline-level talker)
        assert s.model_arch == "Qwen3TTSCode2Wav"
        # tts_args is passed through via extras
        assert s.extras["tts_args"]["max_instructions_length"] == 500

    def test_per_stage_model_arch_flows_through_merge(self, tmp_path):
        """Verify the new ps.model_arch override survives merge_pipeline_deploy."""
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_tts.yaml"
        if not deploy_path.exists():
            pytest.skip("qwen3_tts deploy yaml not found")

        deploy = load_deploy_config(deploy_path)
        pipeline = OMNI_PIPELINES["qwen3_tts"]
        stages = merge_pipeline_deploy(pipeline, deploy)

        # Stage 0 inherits pipeline-level model_arch
        assert stages[0].yaml_engine_args["model_arch"] == "Qwen3TTSTalkerForConditionalGeneration"
        # Stage 1 uses its per-stage override
        assert stages[1].yaml_engine_args["model_arch"] == "Qwen3TTSCode2Wav"

    def test_subtalker_sampling_params_deep_merge_preserves_base_keys(self):
        """Verify subtalker sampling params participate in stage deep-merge."""
        base = {
            "stage_id": 0,
            "subtalker_sampling_params": {
                "do_sample": True,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 1.0,
            },
        }
        overlay = {
            "stage_id": 0,
            "subtalker_sampling_params": {
                "temperature": 0.7,
                "top_k": 32,
            },
        }

        merged = _deep_merge_stage(base, overlay)

        assert merged["subtalker_sampling_params"] == {
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 32,
            "top_p": 1.0,
        }


class TestMingFlashOmniPipeline:
    def test_registered(self):
        p = resolve_pipeline_config("ming_flash_omni")
        assert isinstance(p, PipelineConfig)
        assert p.model_arch == "MingFlashOmniForConditionalGeneration"
        assert len(p.stages) == 2
        assert p.validate() == []

    def test_thinker_stage(self):
        p = resolve_pipeline_config("ming_flash_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.owns_tokenizer is True
        assert s.requires_multimodal_data is True
        assert s.engine_output_type == "text"
        assert s.hf_config_name == "llm_config"
        assert s.sampling_constraints["detokenize"] is True

    def test_talker_stage(self):
        p = resolve_pipeline_config("ming_flash_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(1)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "ming_tts"
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.input_sources == (0,)
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"
        assert s.hf_config_name == "talker_config"
        # Per-stage model_arch override (Ming talker has its own self-contained LLM)
        assert s.model_arch == "MingFlashOmniTalkerForConditionalGeneration"
        assert s.tokenizer_subdir == "talker/llm"
        # thinker2talker was removed: ming_flash_omni has no async_chunk support
        # and both thinker2talker / thinker2talker_token_only called _build_talker_inputs
        # identically, so custom_process_input_func was dead code.
        assert s.custom_process_input_func is None
        assert s.sync_process_input_func is not None

    def test_talker_stage_processor_wiring_resolves(self):
        """The sync_process_input_func string must point to a real callable.

        Lazy string references only fail at first inference otherwise — this
        catches typos in the pipeline declaration at import / registration time.
        """
        p = resolve_pipeline_config("ming_flash_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(1)
        assert isinstance(s, StagePipelineConfig)
        module_path, _, attr = s.sync_process_input_func.rpartition(".")
        module = importlib.import_module(module_path)
        assert callable(getattr(module, attr))

    def test_tts_pipeline_registered(self):
        p = resolve_pipeline_config("ming_flash_omni_tts")
        assert isinstance(p, PipelineConfig)
        assert p.model_arch == "MingFlashOmniTalkerForConditionalGeneration"
        assert len(p.stages) == 1
        assert p.validate() == []

    def test_tts_stage(self):
        p = resolve_pipeline_config("ming_flash_omni_tts")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "ming_tts"
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.input_sources == ()
        assert s.owns_tokenizer is True
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"
        assert s.hf_config_name == "talker_config"
        assert s.tokenizer_subdir == "talker/llm"

    def test_full_yaml_loads_and_merges(self):
        """deploy/ming_flash_omni.yaml parses and merges with the registered pipeline."""
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "ming_flash_omni.yaml"
        if not deploy_path.exists():
            pytest.skip("ming_flash_omni deploy yaml not found")

        deploy = load_deploy_config(deploy_path)
        assert len(deploy.stages) == 2
        assert deploy.async_chunk is False
        assert deploy.pipeline == "ming_flash_omni"
        # We won't test stage 0/1 colocation contract here,
        # as there could exist more variant of custom device setup

        pipeline = resolve_pipeline_config("ming_flash_omni")
        assert isinstance(pipeline, PipelineConfig)
        stages = merge_pipeline_deploy(pipeline, deploy)
        assert len(stages) == 2
        assert stages[0].yaml_engine_args["model_arch"] == "MingFlashOmniForConditionalGeneration"
        assert stages[1].yaml_engine_args["model_arch"] == "MingFlashOmniTalkerForConditionalGeneration"

    def test_tts_yaml_loads_and_merges(self):
        """deploy/ming_flash_omni_tts.yaml parses and routes to the TTS-only pipeline."""
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "ming_flash_omni_tts.yaml"
        if not deploy_path.exists():
            pytest.skip("ming_flash_omni_tts deploy yaml not found")

        deploy = load_deploy_config(deploy_path)
        assert len(deploy.stages) == 1
        assert deploy.pipeline == "ming_flash_omni_tts"

        pipeline = OMNI_PIPELINES["ming_flash_omni_tts"]
        stages = merge_pipeline_deploy(pipeline, deploy)
        assert len(stages) == 1
        assert stages[0].yaml_engine_args["model_arch"] == "MingFlashOmniTalkerForConditionalGeneration"

    def test_thinker_only_pipeline_registered(self):
        p = resolve_pipeline_config("ming_flash_omni_thinker_only")
        assert isinstance(p, PipelineConfig)
        assert p.model_arch == "MingFlashOmniForConditionalGeneration"
        assert len(p.stages) == 1
        assert p.validate() == []

    def test_thinker_only_stage(self):
        p = resolve_pipeline_config("ming_flash_omni_thinker_only")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.input_sources == ()
        assert s.owns_tokenizer is True
        assert s.requires_multimodal_data is True
        assert s.final_output_type == "text"
        assert s.engine_output_type == "text"
        assert s.hf_config_name == "llm_config"
        assert s.sampling_constraints["detokenize"] is True

    def test_thinker_only_yaml_loads_and_merges(self):
        """deploy/ming_flash_omni_thinker_only.yaml parses and routes to the thinker-only pipeline."""
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "ming_flash_omni_thinker_only.yaml"
        if not deploy_path.exists():
            pytest.skip("ming_flash_omni_thinker_only deploy yaml not found")

        deploy = load_deploy_config(deploy_path)
        assert len(deploy.stages) == 1
        assert deploy.pipeline == "ming_flash_omni_thinker_only"

        pipeline = resolve_pipeline_config("ming_flash_omni_thinker_only")
        assert isinstance(pipeline, PipelineConfig)
        stages = merge_pipeline_deploy(pipeline, deploy)
        assert len(stages) == 1
        assert stages[0].yaml_engine_args["model_arch"] == "MingFlashOmniForConditionalGeneration"

    def test_image_pipeline_registered(self):
        p = OMNI_PIPELINES.get("ming_flash_omni_image")
        assert p is not None
        assert p.model_arch == "MingFlashOmniForConditionalGeneration"
        assert len(p.stages) == 2
        assert p.validate() == []

    def test_image_thinker_stage(self):
        s = resolve_pipeline_config("ming_flash_omni_image").get_stage(0)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.input_sources == ()
        assert s.final_output is False
        assert s.owns_tokenizer is True
        assert s.requires_multimodal_data is True
        # Image variant exports hidden states for the diffusion stage.
        assert s.engine_output_type == "latent"
        assert s.hf_config_name == "thinker_config"
        assert s.sampling_constraints["detokenize"] is False
        assert s.prompt_expand_func is not None

    def test_image_dit_stage(self):
        s = resolve_pipeline_config("ming_flash_omni_image").get_stage(1)
        assert s.model_stage == "dit"
        assert s.execution_type == StageExecutionType.DIFFUSION
        assert s.input_sources == (0,)
        assert s.final_output is True
        assert s.final_output_type == "image"
        assert s.hf_config_name == "image_gen_config"
        assert s.model_arch == "MingImagePipeline"
        assert s.custom_process_input_func is not None

    def test_image_processor_wiring_resolves(self):
        """The prompt_expand_func and custom_process_input_func strings must point to real callables."""
        pipeline = resolve_pipeline_config("ming_flash_omni_image")
        assert isinstance(pipeline, PipelineConfig)

        thinker = pipeline.get_stage(0)
        dit = pipeline.get_stage(1)
        for ref in (thinker.prompt_expand_func, dit.custom_process_input_func):
            module_path, _, attr = ref.rpartition(".")
            module = importlib.import_module(module_path)
            assert callable(getattr(module, attr))

    def test_image_yaml_loads_and_merges(self):
        """deploy/ming_flash_omni_image.yaml parses and routes to the image pipeline."""
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "ming_flash_omni_image.yaml"
        if not deploy_path.exists():
            pytest.skip("ming_flash_omni_image deploy yaml not found")

        deploy = load_deploy_config(deploy_path)
        assert len(deploy.stages) == 2
        assert deploy.async_chunk is False
        assert deploy.pipeline == "ming_flash_omni_image"
        assert deploy.connectors is not None
        assert "shared_memory_connector" in deploy.connectors

        pipeline = resolve_pipeline_config("ming_flash_omni_image")
        assert isinstance(pipeline, PipelineConfig)
        stages = merge_pipeline_deploy(pipeline, deploy)
        assert len(stages) == 2
        # Stage 0 thinker: AR worker that emits latents.
        assert stages[0].yaml_engine_args["model_arch"] == "MingFlashOmniForConditionalGeneration"
        assert stages[0].yaml_engine_args["engine_output_type"] == "latent"
        assert stages[0].yaml_extras["default_sampling_params"]["detokenize"] is False
        assert stages[0].yaml_extras["prompt_expand_func"] is not None
        # Stage 1 dit: diffusion stage with MingImagePipeline.
        assert stages[1].yaml_engine_args["model_arch"] == "MingImagePipeline"
        assert stages[1].custom_process_input_func is not None
        assert stages[1].final_output is True
        assert stages[1].final_output_type == "image"


class TestBaseConfigInheritance:
    """Test deploy YAML base_config inheritance."""

    def test_ci_inherits_from_main(self):
        ci_path = Path(get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
        if not ci_path.exists():
            pytest.skip("CI deploy config not found")

        deploy = load_deploy_config(ci_path)
        assert len(deploy.stages) == 3
        # CI overrides
        assert deploy.stages[0].load_format is None
        assert "load_format" not in deploy.stages[0].engine_extras
        assert deploy.stages[0].max_num_seqs == 5
        # Inherited from base
        assert deploy.stages[0].gpu_memory_utilization == 0.9
        assert deploy.connectors is not None
        assert "connector_of_shared_memory" in deploy.connectors
        # CI overlay explicitly sets async_chunk: False (see
        # tests.helpers.stage_config._CI_OVERLAYS and PR #2383 discussion). Overlay
        # bool overrides base even when the base yaml has async_chunk: true.
        assert deploy.async_chunk is False

    def test_ci_sampling_merge(self):
        ci_path = Path(get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
        if not ci_path.exists():
            pytest.skip("CI deploy config not found")

        deploy = load_deploy_config(ci_path)
        s0 = deploy.stages[0].default_sampling_params
        # CI overrides max_tokens
        assert s0["max_tokens"] == 150

    def test_pure_inheritance_overlay(self, tmp_path):
        """An overlay with only ``base_config`` inherits everything."""
        base = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not base.exists():
            pytest.skip("Base deploy config not found")

        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(f"base_config: {base}\n")

        deploy = load_deploy_config(overlay)
        assert len(deploy.stages) == 3
        assert deploy.stages[0].gpu_memory_utilization == 0.9

    def test_single_field_overlay(self, tmp_path):
        """An overlay overriding one stage field merges with the base."""
        base = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not base.exists():
            pytest.skip("Base deploy config not found")

        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(f"base_config: {base}\nstages:\n  - stage_id: 2\n    max_num_batched_tokens: 1000000\n")

        deploy = load_deploy_config(overlay)
        assert deploy.stages[2].max_num_batched_tokens == 1000000
        # Rest inherited
        assert deploy.stages[0].gpu_memory_utilization == 0.9


class TestPlatformOverrides:
    """Test platform-specific deploy config overrides."""

    def test_npu_overrides(self):
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        deploy = _apply_platform_overrides(deploy, platform="npu")

        assert deploy.stages[0].gpu_memory_utilization == 0.6
        assert deploy.stages[0].tensor_parallel_size == 2
        assert deploy.stages[0].devices == "0,1"
        # Stage 2 unaffected fields stay at base
        assert deploy.stages[2].enforce_eager is False

    def test_xpu_overrides(self):
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        deploy = _apply_platform_overrides(deploy, platform="xpu")

        assert deploy.stages[0].tensor_parallel_size == 4
        assert deploy.stages[0].devices == "0,1,2,3"
        assert deploy.stages[0].engine_extras.get("max_cudagraph_capture_size") == 0

    def test_unknown_platform_noop(self):
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        original_mem = deploy.stages[0].gpu_memory_utilization
        deploy = _apply_platform_overrides(deploy, platform="unknown_hw")
        assert deploy.stages[0].gpu_memory_utilization == original_mem

    def test_platforms_deep_merge_inheritance(self, tmp_path):
        """Overlay's platforms: block layers onto base's, per-stage."""
        base = tmp_path / "base.yaml"
        base.write_text(
            "stages:\n"
            "  - stage_id: 0\n"
            "    gpu_memory_utilization: 0.9\n"
            "platforms:\n"
            "  rocm:\n"
            "    stages:\n"
            "      - stage_id: 0\n"
            "        enforce_eager: true\n"
        )
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(
            f"base_config: {base.name}\n"
            "platforms:\n"
            "  rocm:\n"
            "    stages:\n"
            "      - stage_id: 0\n"
            "        max_num_seqs: 1\n"
        )

        deploy = load_deploy_config(overlay)
        deploy = _apply_platform_overrides(deploy, platform="rocm")
        # Both base's enforce_eager and overlay's max_num_seqs should apply.
        assert deploy.stages[0].enforce_eager is True
        assert deploy.stages[0].max_num_seqs == 1
        # Inherited stage default not touched by overlay platforms section.
        assert deploy.stages[0].gpu_memory_utilization == 0.9


class TestCLIOverrideFlow:
    """Test --stage-overrides JSON merge into StageConfig."""

    def test_stage_overrides_merge(self):
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        pipeline = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(pipeline, PipelineConfig)
        stages = merge_pipeline_deploy(pipeline, deploy)

        # Simulate --stage-overrides '{"0": {"gpu_memory_utilization": 0.5}}'
        overrides = {"stage_0_gpu_memory_utilization": 0.5}
        stages[0].runtime_overrides = StageConfigFactory._merge_cli_overrides(stages[0], overrides)
        assert stages[0].runtime_overrides["gpu_memory_utilization"] == 0.5

    def test_global_override_applies_to_all(self):
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        pipeline = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(pipeline, PipelineConfig)
        stages = merge_pipeline_deploy(pipeline, deploy)

        overrides = {"enforce_eager": True}
        for s in stages:
            s.runtime_overrides = StageConfigFactory._merge_cli_overrides(s, overrides)
            assert s.runtime_overrides["enforce_eager"] is True


class TestAuraOmniDeploy:
    def test_aura_omni_deploy_forces_pipeline_override(self):
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "aura_omni.yaml"
        deploy = load_deploy_config(deploy_path)

        assert deploy.pipeline == "aura_omni"

    def test_aura_omni_deploy_resolves_four_native_stages(self):
        pipeline_cfg = resolve_pipeline_config("aura_omni")

        stages, _ = StageConfigFactory._create_legacy_from_registry(
            pipeline_cfg,
            cli_overrides={},
            deploy_config_path=str(Path(__file__).parent.parent / "vllm_omni" / "deploy" / "aura_omni.yaml"),
        )

        assert [stage.model_stage for stage in stages] == [
            "asr",
            "aura",
            "qwen3_tts",
            "code2wav",
        ]
        assert [stage.final_output for stage in stages] == [False, True, False, True]
        assert [stage.final_output_type for stage in stages] == [None, "text", None, "audio"]
        assert stages[0].yaml_engine_args["model_arch"] == "Qwen3ASRForConditionalGeneration"
        assert stages[1].yaml_engine_args["model_arch"] == "AuraQwen3VLForConditionalGeneration"
        assert stages[2].yaml_engine_args["model_arch"] == "Qwen3TTSTalkerForConditionalGeneration"
        assert stages[3].yaml_engine_args["model_arch"] == "Qwen3TTSCode2Wav"


class TestDeployCliOverrideFlow:
    """Test deploy-YAML baselines overridden by CLI runtime overrides."""

    def test_diffusion_deploy_fields_can_be_overridden_by_cli(self, tmp_path):
        deploy_path = tmp_path / "diffusion_stage.yaml"
        deploy_path.write_text(
            """
async_chunk: false
stages:
  - stage_id: 0
    devices: "4,5"
    parallel_config:
      pipeline_parallel_size: 1
      data_parallel_size: 1
      tensor_parallel_size: 1
      enable_expert_parallel: false
      sequence_parallel_size: 1
      ulysses_degree: 1
      ring_degree: 1
      cfg_parallel_size: 1
      vae_patch_parallel_size: 1
      use_hsdp: false
      hsdp_shard_size: -1
      hsdp_replicate_size: 1
    engine_args:
      cache_backend: cache_dit
      diffusion_attention_backend: FLASH_ATTN
      diffusion_kv_cache_dtype: auto
      step_execution: false
      vae_use_tiling: false
      enable_cpu_offload: false
      max_generated_image_size: 1048576
      tts_max_instructions_length: 1000
""",
            encoding="utf-8",
        )

        pipeline = PipelineConfig(
            model_type="test_diffusion",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="dit",
                    execution_type=StageExecutionType.DIFFUSION,
                    final_output=True,
                    final_output_type="image",
                ),
            ),
        )
        deploy = load_deploy_config(deploy_path)
        stages = merge_pipeline_deploy(pipeline, deploy)
        stage = stages[0]

        assert stage.yaml_engine_args["parallel_config"]["pipeline_parallel_size"] == 1
        assert stage.yaml_engine_args["parallel_config"]["data_parallel_size"] == 1
        assert stage.yaml_engine_args["parallel_config"]["tensor_parallel_size"] == 1
        assert stage.yaml_engine_args["parallel_config"]["enable_expert_parallel"] is False
        assert stage.yaml_engine_args["parallel_config"]["sequence_parallel_size"] == 1
        assert stage.yaml_engine_args["parallel_config"]["ulysses_degree"] == 1
        assert stage.yaml_engine_args["parallel_config"]["ring_degree"] == 1
        assert stage.yaml_engine_args["parallel_config"]["cfg_parallel_size"] == 1
        assert stage.yaml_engine_args["parallel_config"]["vae_patch_parallel_size"] == 1
        assert stage.yaml_engine_args["cache_backend"] == "cache_dit"
        assert stage.yaml_engine_args["parallel_config"]["use_hsdp"] is False
        assert stage.yaml_engine_args["parallel_config"]["hsdp_shard_size"] == -1
        assert stage.yaml_engine_args["parallel_config"]["hsdp_replicate_size"] == 1
        assert stage.yaml_engine_args["diffusion_attention_backend"] == "FLASH_ATTN"
        assert stage.yaml_engine_args["diffusion_kv_cache_dtype"] == "auto"
        assert stage.yaml_engine_args["step_execution"] is False
        assert stage.yaml_engine_args["vae_use_tiling"] is False
        assert stage.yaml_engine_args["enable_cpu_offload"] is False
        assert stage.yaml_engine_args["max_generated_image_size"] == 1048576
        assert stage.yaml_engine_args["tts_max_instructions_length"] == 1000

        stage.runtime_overrides = StageConfigFactory._merge_cli_overrides(
            stage,
            {
                "pipeline_parallel_size": 2,
                "data_parallel_size": 3,
                "tensor_parallel_size": 4,
                "enable_expert_parallel": True,
                "sequence_parallel_size": 24,
                "ulysses_degree": 2,
                "ring_degree": 4,
                "cfg_parallel_size": 2,
                "vae_patch_parallel_size": 2,
                "cache_backend": "tea_cache",
                "use_hsdp": True,
                "hsdp_shard_size": 8,
                "hsdp_replicate_size": 2,
                "diffusion_attention_backend": "SAGE_ATTN",
                "diffusion_kv_cache_dtype": "fp8",
                "step_execution": True,
                "vae_use_tiling": True,
                "enable_cpu_offload": True,
                "max_generated_image_size": 2097152,
                "tts_max_instructions_length": 2000,
            },
        )

        omega_config = stage.to_omegaconf()

        assert omega_config.engine_args.cache_backend == "tea_cache"
        assert omega_config.engine_args.diffusion_attention_backend == "SAGE_ATTN"
        assert omega_config.engine_args.diffusion_kv_cache_dtype == "fp8"
        assert omega_config.engine_args.step_execution is True
        assert omega_config.engine_args.vae_use_tiling is True
        assert omega_config.engine_args.enable_cpu_offload is True
        assert omega_config.engine_args.max_generated_image_size == 2097152
        assert omega_config.engine_args.tts_max_instructions_length == 2000
        assert omega_config.engine_args.parallel_config.pipeline_parallel_size == 2
        assert omega_config.engine_args.parallel_config.data_parallel_size == 3
        assert omega_config.engine_args.parallel_config.tensor_parallel_size == 4
        assert omega_config.engine_args.parallel_config.enable_expert_parallel is True
        assert omega_config.engine_args.parallel_config.sequence_parallel_size == 24
        assert omega_config.engine_args.parallel_config.ulysses_degree == 2
        assert omega_config.engine_args.parallel_config.ring_degree == 4
        assert omega_config.engine_args.parallel_config.cfg_parallel_size == 2
        assert omega_config.engine_args.parallel_config.vae_patch_parallel_size == 2
        assert omega_config.engine_args.parallel_config.use_hsdp is True
        assert omega_config.engine_args.parallel_config.hsdp_shard_size == 8
        assert omega_config.engine_args.parallel_config.hsdp_replicate_size == 2

    def test_llm_deploy_fields_can_be_overridden_by_cli(self, tmp_path):
        deploy_path = tmp_path / "llm_stage.yaml"
        deploy_path.write_text(
            """
async_chunk: false
stages:
  - stage_id: 0
    devices: "0,1"
    engine_args:
      tensor_parallel_size: 1
      enable_expert_parallel: false
      gpu_memory_utilization: 0.5
      max_num_seqs: 16
      max_num_batched_tokens: 1024
      max_model_len: 4096
      enforce_eager: false
""",
            encoding="utf-8",
        )

        pipeline = PipelineConfig(
            model_type="test_llm",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="thinker",
                    execution_type=StageExecutionType.LLM_AR,
                    final_output=True,
                    final_output_type="text",
                ),
            ),
        )
        deploy = load_deploy_config(deploy_path)
        stages = merge_pipeline_deploy(pipeline, deploy)
        stage = stages[0]

        assert stage.yaml_engine_args["tensor_parallel_size"] == 1
        assert stage.yaml_engine_args["enable_expert_parallel"] is False
        assert stage.yaml_engine_args["gpu_memory_utilization"] == 0.5
        assert stage.yaml_engine_args["max_num_seqs"] == 16
        assert stage.yaml_engine_args["max_num_batched_tokens"] == 1024
        assert stage.yaml_engine_args["max_model_len"] == 4096
        assert stage.yaml_engine_args["enforce_eager"] is False

        stage.runtime_overrides = StageConfigFactory._merge_cli_overrides(
            stage,
            {
                "tensor_parallel_size": 2,
                "enable_expert_parallel": True,
                "gpu_memory_utilization": 0.9,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 2048,
                "max_model_len": 8192,
                "enforce_eager": True,
            },
        )

        omega_config = stage.to_omegaconf()

        assert omega_config.engine_args.tensor_parallel_size == 2
        assert omega_config.engine_args.enable_expert_parallel is True
        assert omega_config.engine_args.gpu_memory_utilization == 0.9
        assert omega_config.engine_args.max_num_seqs == 32
        assert omega_config.engine_args.max_num_batched_tokens == 2048
        assert omega_config.engine_args.max_model_len == 8192
        assert omega_config.engine_args.enforce_eager is True


class TestSentinelDefaultPrecedence:
    """Caller-typed (non-None) values win over YAML; None values fall through
    to YAML / dataclass defaults (#3035)."""

    def _stages(self, cli_overrides):
        model_type = "qwen3_omni_moe"
        pipeline_cfg = resolve_pipeline_config(
            model_type,
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        stages, _ = StageConfigFactory._create_legacy_from_registry(
            pipeline_cfg,
            cli_overrides=cli_overrides,
        )
        return stages

    def test_typed_kwarg_overrides_yaml(self):
        stages = self._stages({"max_num_seqs": 999})
        assert stages[2].runtime_overrides.get("max_num_seqs") == 999

    def test_none_value_skipped_yaml_wins(self):
        stages = self._stages({"max_num_seqs": None})
        assert stages[2].runtime_overrides.get("max_num_seqs") is None
        assert stages[2].yaml_engine_args.get("max_num_seqs") == 64

    def test_empty_kwargs_yaml_only(self):
        stages = self._stages({})
        for stage in stages:
            assert stage.runtime_overrides == {}

    def test_typed_kwarg_equal_to_dataclass_default_still_overrides(self):
        # Caller intent honored regardless of value coincidence (no heuristic).
        stages = self._stages({"gpu_memory_utilization": 0.9})
        assert stages[2].runtime_overrides.get("gpu_memory_utilization") == 0.9

    def test_per_stage_kwarg_routed_to_correct_stage(self):
        stages = self._stages({"stage_0_gpu_memory_utilization": 0.42})
        assert stages[0].runtime_overrides.get("gpu_memory_utilization") == 0.42
        assert stages[2].runtime_overrides.get("gpu_memory_utilization") is None

    def test_async_chunk_false_overrides_yaml_true(self):
        stages = self._stages({"async_chunk": False})
        for stage in stages:
            assert stage.yaml_engine_args.get("async_chunk") is not True

    def test_async_chunk_none_keeps_yaml_true(self):
        stages = self._stages({"async_chunk": None})
        for stage in stages:
            assert stage.yaml_engine_args.get("async_chunk") is True

    def test_enable_prefix_caching_typed_overrides_yaml(self):
        stages = self._stages({"enable_prefix_caching": True})
        for stage in stages:
            assert stage.runtime_overrides.get("enable_prefix_caching") is True

    def test_omni_with_vars_args_anti_pattern_is_safe(self):
        # Omni(**vars(args)) with mostly-None namespace must not clobber YAML.
        simulated_vars_args = {
            "gpu_memory_utilization": None,
            "max_num_seqs": None,
            "async_chunk": None,
            "enable_prefix_caching": None,
            "dtype": None,
        }
        stages = self._stages(simulated_vars_args)
        for stage in stages:
            assert stage.runtime_overrides == {}

    def test_async_chunk_dispatches_processors(self):
        """A single ``qwen3_tts`` pipeline picks per-chunk vs end-to-end
        processors based on ``deploy.async_chunk``, without needing a
        separate variant pipeline registration."""
        pipeline = resolve_pipeline_config("qwen3_tts")
        assert isinstance(pipeline, PipelineConfig)

        # async_chunk=True → stage 0's per-chunk processor wires up, stage 1
        # has no sync input processor.
        async_stages = merge_pipeline_deploy(pipeline, DeployConfig(async_chunk=True))
        assert (
            async_stages[0]
            .yaml_engine_args.get("custom_process_next_stage_input_func", "")
            .endswith("talker2code2wav_async_chunk")
        )
        assert async_stages[1].custom_process_input_func is None

        # async_chunk=False → stage 0 ships the bulk codec via the
        # worker-connector full-payload producer; stage 1 wires the
        # ``_token_only`` placeholder so the orchestrator emits no
        # legacy ``additional_information``-shaped input (PR3 sync-
        # via-connector data plane).
        sync_stages = merge_pipeline_deploy(pipeline, DeployConfig(async_chunk=False))
        assert (
            sync_stages[0]
            .yaml_engine_args["custom_process_next_stage_input_func"]
            .endswith("talker2code2wav_full_payload")
        )
        assert sync_stages[1].custom_process_input_func is not None
        assert sync_stages[1].custom_process_input_func.endswith("talker2code2wav_token_only")

    def test_async_chunk_dispatches_qwen3_omni_processors(self):
        import runpy
        from pathlib import Path

        from vllm_omni.config.stage_config import DeployConfig, merge_pipeline_deploy

        pipeline_path = (
            Path(__file__).parent.parent / "vllm_omni" / "model_executor" / "models" / "qwen3_omni" / "pipeline.py"
        )
        pipeline = runpy.run_path(str(pipeline_path))["QWEN3_OMNI_PIPELINE"]

        async_stages = merge_pipeline_deploy(pipeline, DeployConfig(async_chunk=True))
        assert (
            async_stages[0]
            .yaml_engine_args["custom_process_next_stage_input_func"]
            .endswith("thinker2talker_async_chunk")
        )
        assert (
            async_stages[1]
            .yaml_engine_args["custom_process_next_stage_input_func"]
            .endswith("talker2code2wav_async_chunk")
        )

        sync_stages = merge_pipeline_deploy(pipeline, DeployConfig(async_chunk=False))
        assert (
            sync_stages[0]
            .yaml_engine_args["custom_process_next_stage_input_func"]
            .endswith("thinker2talker_full_payload")
        )
        assert (
            sync_stages[1]
            .yaml_engine_args["custom_process_next_stage_input_func"]
            .endswith("talker2code2wav_full_payload")
        )

    def test_ming_flash_omni_topology(self):
        """Guard ming_flash_omni's SIP cleanup: stage 0 has no full-payload
        producer hook (arch is not in ``_FULL_PAYLOAD_INPUT_STAGES``), and
        stage 1 uses only ``thinker2talker_token_only`` (sync_process_input_func).
        The dead ``thinker2talker`` (custom_process_input_func) was removed
        because ming_flash_omni has no async_chunk support and both functions
        called ``_build_talker_inputs`` identically.
        Merge under either async_chunk mode must not re-introduce a
        stage-0 full-payload hook."""
        pipeline = resolve_pipeline_config("ming_flash_omni")
        assert isinstance(pipeline, PipelineConfig)

        stage0, stage1 = pipeline.stages
        assert stage0.custom_process_next_stage_input_func is None, (
            "ming_flash_omni stage 0 must not declare a full-payload producer "
            "(connector path is not active for this arch)."
        )
        assert stage1.custom_process_input_func is None
        assert stage1.sync_process_input_func is not None
        assert stage1.sync_process_input_func.endswith("thinker2talker_token_only")

        # async_chunk=True must now be rejected: removing the fake hook means
        # there is no next-stage input processor for the validator to accept.
        # (Positive consequence -- users can't accidentally enable async_chunk
        # on an arch that doesn't actually support it.)
        import pytest as _pytest

        with _pytest.raises(ValueError, match="async_chunk=True"):
            merge_pipeline_deploy(pipeline, DeployConfig(async_chunk=True))

        # async_chunk=False merges cleanly and stage-0 yaml_engine_args carries
        # no spurious full-payload hook.
        merged = merge_pipeline_deploy(pipeline, DeployConfig(async_chunk=False))
        assert "custom_process_next_stage_input_func" not in merged[0].yaml_engine_args, (
            "stage-0 full-payload hook unexpectedly re-appeared in yaml_engine_args"
        )


class TestSamplingConstraintsPrecedence:
    """Test that pipeline sampling_constraints override deploy defaults."""

    def test_constraints_win(self):
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        pipeline = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(pipeline, PipelineConfig)
        stages = merge_pipeline_deploy(pipeline, deploy)

        # Pipeline says detokenize=True for thinker, deploy can't override
        assert stages[0].yaml_extras["default_sampling_params"]["detokenize"] is True
        # Pipeline says stop_token_ids=[2150] for talker
        assert stages[1].yaml_extras["default_sampling_params"]["stop_token_ids"] == [2150]


class TestPipelineConfigResolvers:
    @pytest.mark.parametrize("resolver", [obj for obj in OMNI_PIPELINES.values() if callable(obj)])
    def test_all_resolvers_reject_bad_types(self, resolver):
        """Ensure that all resolvers registered reject incorrect config types."""

        class NotTheRightHfConfig(PretrainedConfig):
            pass

        assert resolver(NotTheRightHfConfig()) is None

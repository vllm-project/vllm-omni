# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for StageConfigFactory and related classes.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from vllm_omni.config.stage_config import (
    _EXECUTION_TYPE_TO_SCHEDULER,
    _PIPELINE_REGISTRY,
    ModelPipeline,
    PipelineConfig,
    StageConfig,
    StageConfigFactory,
    StageExecutionType,
    StagePipelineConfig,
    StageType,
    build_stage_runtime_overrides,
    register_pipeline,
    strip_parent_engine_args,
)
from vllm_omni.engine.arg_utils import SHARED_FIELDS, internal_blacklist_keys


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
        # Legacy field name for backward compatibility
        assert omega_config.engine_input_source == []

    def test_to_omegaconf_with_runtime_overrides(self):
        """Test that runtime overrides are applied to OmegaConf output."""
        import warnings

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
        import warnings

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


class TestModelPipeline:
    """Tests for ModelPipeline class."""

    def test_valid_linear_dag(self):
        """Test validation of a valid linear DAG."""
        stages = [
            StageConfig(stage_id=0, model_stage="thinker", input_sources=[]),
            StageConfig(stage_id=1, model_stage="talker", input_sources=[0]),
            StageConfig(stage_id=2, model_stage="code2wav", input_sources=[1]),
        ]
        pipeline = ModelPipeline(model_type="test", stages=stages)
        errors = pipeline.validate_pipeline()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_valid_branching_dag(self):
        """Test validation of a valid branching DAG."""
        stages = [
            StageConfig(stage_id=0, model_stage="input", input_sources=[]),
            StageConfig(stage_id=1, model_stage="branch_a", input_sources=[0]),
            StageConfig(stage_id=2, model_stage="branch_b", input_sources=[0]),
        ]
        pipeline = ModelPipeline(model_type="test", stages=stages)
        errors = pipeline.validate_pipeline()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_entry_point(self):
        """Test that missing entry point is detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="stage_a", input_sources=[1]),
            StageConfig(stage_id=1, model_stage="stage_b", input_sources=[0]),
        ]
        pipeline = ModelPipeline(model_type="test", stages=stages)
        errors = pipeline.validate_pipeline()
        assert any("entry point" in e.lower() for e in errors)

    def test_missing_dependency(self):
        """Test that missing stage reference is detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="input", input_sources=[]),
            StageConfig(stage_id=1, model_stage="output", input_sources=[99]),  # Invalid
        ]
        pipeline = ModelPipeline(model_type="test", stages=stages)
        errors = pipeline.validate_pipeline()
        assert any("non-existent" in e.lower() for e in errors)

    def test_duplicate_stage_ids(self):
        """Test that duplicate stage IDs are detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="stage_a", input_sources=[]),
            StageConfig(stage_id=0, model_stage="stage_b", input_sources=[]),  # Duplicate
        ]
        pipeline = ModelPipeline(model_type="test", stages=stages)
        errors = pipeline.validate_pipeline()
        assert any("duplicate" in e.lower() for e in errors)

    def test_self_reference(self):
        """Test that self-references are detected."""
        stages = [
            StageConfig(stage_id=0, model_stage="entry", input_sources=[]),
            StageConfig(stage_id=1, model_stage="self_ref", input_sources=[1]),  # Self
        ]
        pipeline = ModelPipeline(model_type="test", stages=stages)
        errors = pipeline.validate_pipeline()
        assert any("itself" in e.lower() for e in errors)

    def test_get_stage_by_id(self):
        """Test getting stage by ID."""
        stages = [
            StageConfig(stage_id=0, model_stage="thinker", input_sources=[]),
            StageConfig(stage_id=1, model_stage="talker", input_sources=[0]),
        ]
        pipeline = ModelPipeline(model_type="test", stages=stages)

        stage = pipeline.get_stage(1)
        assert stage is not None
        assert stage.model_stage == "talker"

        missing = pipeline.get_stage(99)
        assert missing is None

    def test_empty_pipeline(self):
        """Test validation of empty pipeline."""
        pipeline = ModelPipeline(model_type="test", stages=[])
        errors = pipeline.validate_pipeline()
        assert any("no stages" in e.lower() for e in errors)


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
        from dataclasses import fields as dc_fields

        from vllm.engine.arg_utils import EngineArgs

        parent_fields = {f.name: f for f in dc_fields(EngineArgs)}
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


class TestPipelineYamlParsing:
    """Tests for pipeline YAML file parsing (@ZJY0516)."""

    def test_parse_qwen3_omni_moe_yaml(self, tmp_path):
        """Test parsing the qwen3_omni_moe pipeline YAML."""
        yaml_content = """\
model_type: qwen3_omni_moe
async_chunk: false

stages:
  - stage_id: 0
    model_stage: thinker
    stage_type: llm
    input_sources: []
    worker_type: ar
    scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
    hf_config_name: thinker_config
    final_output: true
    final_output_type: text
    is_comprehension: true

  - stage_id: 1
    model_stage: talker
    stage_type: llm
    input_sources: [0]
    worker_type: ar
    scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
    hf_config_name: talker_config
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker

  - stage_id: 2
    model_stage: code2wav
    stage_type: llm
    input_sources: [1]
    worker_type: generation
    scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
    hf_config_name: thinker_config
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav
    final_output: true
    final_output_type: audio
"""
        yaml_file = tmp_path / "qwen3_omni_moe.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = StageConfigFactory._parse_pipeline_yaml(yaml_file, "qwen3_omni_moe")

        assert pipeline.model_type == "qwen3_omni_moe"
        assert len(pipeline.stages) == 3

        # Stage 0: thinker
        s0 = pipeline.stages[0]
        assert s0.stage_id == 0
        assert s0.model_stage == "thinker"
        assert s0.stage_type == StageType.LLM
        assert s0.input_sources == []
        assert s0.worker_type == "ar"
        assert s0.final_output is True
        assert s0.final_output_type == "text"
        assert s0.is_comprehension is True

        # Stage 1: talker
        s1 = pipeline.stages[1]
        assert s1.stage_id == 1
        assert s1.input_sources == [0]
        assert s1.custom_process_input_func == (
            "vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker"
        )
        assert s1.final_output is False

        # Stage 2: code2wav
        s2 = pipeline.stages[2]
        assert s2.stage_id == 2
        assert s2.input_sources == [1]
        assert s2.worker_type == "generation"
        assert s2.final_output_type == "audio"

    def test_parse_yaml_with_legacy_engine_input_source(self, tmp_path):
        """Test backward compatibility with engine_input_source field."""
        yaml_content = """\
model_type: legacy_model

stages:
  - stage_id: 0
    model_stage: entry
    stage_type: llm
  - stage_id: 1
    model_stage: downstream
    stage_type: llm
    engine_input_source: [0]
"""
        yaml_file = tmp_path / "legacy.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = StageConfigFactory._parse_pipeline_yaml(yaml_file, "legacy_model")
        assert pipeline.stages[1].input_sources == [0]

    def test_parse_yaml_with_connectors_and_edges(self, tmp_path):
        """Test parsing pipeline with optional connectors and edges."""
        yaml_content = """\
model_type: test_model

stages:
  - stage_id: 0
    model_stage: entry
    stage_type: llm
    input_sources: []

connectors:
  type: ray

edges:
  - from: 0
    to: 1
"""
        yaml_file = tmp_path / "with_connectors.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = StageConfigFactory._parse_pipeline_yaml(yaml_file, "test_model")
        assert pipeline.connectors == {"type": "ray"}
        assert pipeline.edges == [{"from": 0, "to": 1}]

    def test_parsed_pipeline_passes_validation(self, tmp_path):
        """Test that a well-formed YAML produces a valid pipeline."""
        yaml_content = """\
model_type: valid_model

stages:
  - stage_id: 0
    model_stage: entry
    stage_type: llm
    input_sources: []
    final_output: true
    final_output_type: text
  - stage_id: 1
    model_stage: next
    stage_type: llm
    input_sources: [0]
"""
        yaml_file = tmp_path / "valid.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = StageConfigFactory._parse_pipeline_yaml(yaml_file, "valid_model")
        errors = pipeline.validate_pipeline()
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_parse_yaml_migrates_runtime_max_batch_size(self, tmp_path):
        """Test that runtime.max_batch_size is migrated to engine_args.max_num_seqs."""
        yaml_content = """\
model_type: legacy_batch
stages:
  - stage_id: 0
    model_stage: entry
    stage_type: llm
    input_sources: []
    runtime:
      devices: "0"
      max_batch_size: 16
    engine_args:
      model_arch: SomeModel
"""
        yaml_file = tmp_path / "legacy_batch.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = StageConfigFactory._parse_pipeline_yaml(yaml_file, "legacy_batch")
        s0 = pipeline.stages[0]
        assert "max_batch_size" not in s0.yaml_runtime
        assert s0.yaml_engine_args.get("max_num_seqs") == 16

    def test_parse_diffusion_stage_type(self, tmp_path):
        """Test parsing a diffusion stage type from YAML."""
        yaml_content = """\
model_type: diff_model

stages:
  - stage_id: 0
    model_stage: dit
    stage_type: diffusion
    input_sources: []
    final_output: true
    final_output_type: image
"""
        yaml_file = tmp_path / "diffusion.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = StageConfigFactory._parse_pipeline_yaml(yaml_file, "diff_model")
        assert pipeline.stages[0].stage_type == StageType.DIFFUSION

    def test_parse_mimo_audio_yaml(self, tmp_path):
        """Test parsing the MiMo Audio pipeline YAML."""
        yaml_content = """\
model_type: mimo_audio

stages:
  - stage_id: 0
    model_stage: fused_thinker_talker
    stage_type: llm
    input_sources: []
    worker_type: ar
    scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
    is_comprehension: true
    final_output: true
    final_output_type: text

  - stage_id: 1
    model_stage: code2wav
    stage_type: llm
    input_sources: [0]
    worker_type: generation
    scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.mimo_audio.llm2code2wav
    final_output: true
    final_output_type: audio
"""
        yaml_file = tmp_path / "mimo_audio.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = StageConfigFactory._parse_pipeline_yaml(yaml_file, "mimo_audio")

        assert pipeline.model_type == "mimo_audio"
        assert len(pipeline.stages) == 2

        # Stage 0: fused_thinker_talker
        s0 = pipeline.stages[0]
        assert s0.stage_id == 0
        assert s0.model_stage == "fused_thinker_talker"
        assert s0.input_sources == []
        assert s0.worker_type == "ar"
        assert s0.is_comprehension is True
        assert s0.final_output is True
        assert s0.final_output_type == "text"

        # Stage 1: code2wav
        s1 = pipeline.stages[1]
        assert s1.stage_id == 1
        assert s1.input_sources == [0]
        assert s1.worker_type == "generation"
        assert s1.custom_process_input_func == (
            "vllm_omni.model_executor.stage_input_processors.mimo_audio.llm2code2wav"
        )
        assert s1.final_output is True
        assert s1.final_output_type == "audio"

        # Pipeline validation
        errors = pipeline.validate_pipeline()
        assert errors == [], f"Unexpected errors: {errors}"


class TestAsyncChunk:
    """Tests for async_chunk pipeline flag."""

    def test_model_pipeline_async_chunk_default(self):
        """Test ModelPipeline defaults async_chunk to False."""
        stages = [StageConfig(stage_id=0, model_stage="entry", input_sources=[])]
        pipeline = ModelPipeline(model_type="test", stages=stages)
        assert pipeline.async_chunk is False

    def test_model_pipeline_async_chunk_set(self):
        """Test ModelPipeline with async_chunk=True."""
        stages = [StageConfig(stage_id=0, model_stage="entry", input_sources=[])]
        pipeline = ModelPipeline(model_type="test", stages=stages, async_chunk=True)
        assert pipeline.async_chunk is True

    def test_parse_async_chunk_from_yaml(self, tmp_path):
        """Test that async_chunk is parsed from pipeline YAML."""
        yaml_content = """\
model_type: test_async
async_chunk: true

stages:
  - stage_id: 0
    model_stage: entry
    stage_type: llm
    input_sources: []
"""
        yaml_file = tmp_path / "async.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = StageConfigFactory._parse_pipeline_yaml(yaml_file, "test_async")
        assert pipeline.async_chunk is True

    def test_parse_missing_async_chunk_defaults_false(self, tmp_path):
        """Test that missing async_chunk defaults to False."""
        yaml_content = """\
model_type: test_no_async

stages:
  - stage_id: 0
    model_stage: entry
    stage_type: llm
    input_sources: []
"""
        yaml_file = tmp_path / "no_async.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = StageConfigFactory._parse_pipeline_yaml(yaml_file, "test_no_async")
        assert pipeline.async_chunk is False


class TestPipelineDiscovery:
    """Tests for the central pipeline registry (``pipeline_registry._VLLM_OMNI_PIPELINES``)."""

    def test_registry_has_known_models(self):
        """Built-in pipelines are lazy-loaded from the central declaration
        on first access; no eager import or discovery walk needed."""
        # ``in`` triggers the lazy-map lookup without forcing a load.
        assert "qwen2_5_omni" in _PIPELINE_REGISTRY
        assert "qwen3_omni_moe" in _PIPELINE_REGISTRY
        assert "qwen3_tts" in _PIPELINE_REGISTRY

    def test_registry_loads_pipeline_on_getitem(self):
        """Looking up a registered model_type returns the matching PipelineConfig."""
        pipeline = _PIPELINE_REGISTRY["qwen3_omni_moe"]
        assert pipeline.model_type == "qwen3_omni_moe"
        assert len(pipeline.stages) == 3  # thinker + talker + code2wav

    def test_registry_returns_none_for_unknown(self):
        """Unknown model_types aren't found; ``get()`` returns None."""
        assert "definitely_not_a_real_model" not in _PIPELINE_REGISTRY
        assert _PIPELINE_REGISTRY.get("definitely_not_a_real_model") is None

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

    def test_get_scheduler_cls(self):
        p = PipelineConfig(
            model_type="t",
            model_arch="A",
            stages=(
                StagePipelineConfig(stage_id=0, model_stage="a", execution_type=StageExecutionType.LLM_AR),
                StagePipelineConfig(
                    stage_id=1, model_stage="b", execution_type=StageExecutionType.LLM_GENERATION, input_sources=(0,)
                ),
            ),
        )
        assert "OmniARScheduler" in p.get_scheduler_cls(0)
        assert "OmniGenerationScheduler" in p.get_scheduler_cls(1)


class TestExecutionTypeToScheduler:
    def test_all_types_mapped(self):
        for et in StageExecutionType:
            assert et in _EXECUTION_TYPE_TO_SCHEDULER


class TestPipelineRegistry:
    def test_register_and_lookup(self):
        p = PipelineConfig(
            model_type="__test_only__",
            model_arch="A",
            stages=(StagePipelineConfig(stage_id=0, model_stage="a"),),
        )
        register_pipeline(p)
        assert _PIPELINE_REGISTRY["__test_only__"] is p
        del _PIPELINE_REGISTRY["__test_only__"]


class TestDeployConfigLoading:
    def test_load_deploy_config(self):
        from pathlib import Path

        from vllm_omni.config.stage_config import load_deploy_config

        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        assert len(deploy.stages) == 3
        assert deploy.async_chunk is True
        assert deploy.connectors is not None
        assert deploy.platforms is not None

    def test_merge_pipeline_deploy(self):
        from pathlib import Path

        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401
        from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

        pipeline = _PIPELINE_REGISTRY["qwen3_omni_moe"]
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        stages = merge_pipeline_deploy(pipeline, deploy)

        assert len(stages) == 3
        s0 = stages[0]
        assert s0.model_stage == "thinker"
        assert s0.yaml_engine_args["model_arch"] == "Qwen3OmniMoeForConditionalGeneration"
        assert s0.yaml_engine_args["engine_output_type"] == "latent"
        assert s0.yaml_extras["default_sampling_params"]["detokenize"] is True


class TestQwen3OmniPipeline:
    def test_registered(self):
        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        p = _PIPELINE_REGISTRY.get("qwen3_omni_moe")
        assert p is not None
        assert p.model_arch == "Qwen3OmniMoeForConditionalGeneration"
        assert len(p.stages) == 3
        assert p.validate() == []

    def test_thinker(self):
        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen3_omni_moe"].get_stage(0)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.owns_tokenizer is True
        assert s.engine_output_type == "latent"
        assert s.sampling_constraints["detokenize"] is True

    def test_talker(self):
        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen3_omni_moe"].get_stage(1)
        assert s.input_sources == (0,)
        assert s.sampling_constraints["stop_token_ids"] == [2150]
        assert s.custom_process_input_func is not None
        assert s.custom_process_next_stage_input_func is not None

    def test_code2wav(self):
        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen3_omni_moe"].get_stage(2)
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.final_output_type == "audio"
        assert s.custom_process_input_func is not None


class TestQwen2_5OmniPipeline:
    def test_registered(self):
        import vllm_omni.model_executor.models.qwen2_5_omni.pipeline  # noqa: F401

        p = _PIPELINE_REGISTRY.get("qwen2_5_omni")
        assert p is not None
        assert p.model_arch == "Qwen2_5OmniForConditionalGeneration"
        assert len(p.stages) == 3
        assert p.validate() == []

    def test_thinker(self):
        import vllm_omni.model_executor.models.qwen2_5_omni.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen2_5_omni"].get_stage(0)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.owns_tokenizer is True
        assert s.engine_output_type == "latent"
        assert s.requires_multimodal_data is True

    def test_talker(self):
        import vllm_omni.model_executor.models.qwen2_5_omni.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen2_5_omni"].get_stage(1)
        assert s.input_sources == (0,)
        assert s.sampling_constraints["stop_token_ids"] == [8294]
        assert s.custom_process_input_func is not None

    def test_code2wav(self):
        import vllm_omni.model_executor.models.qwen2_5_omni.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen2_5_omni"].get_stage(2)
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"


class TestQwen3TTSPipeline:
    def test_registered(self):
        import vllm_omni.model_executor.models.qwen3_tts.pipeline  # noqa: F401

        p = _PIPELINE_REGISTRY.get("qwen3_tts")
        assert p is not None
        assert p.model_arch == "Qwen3TTSTalkerForConditionalGeneration"
        assert len(p.stages) == 2
        assert p.validate() == []

    def test_talker_stage(self):
        import vllm_omni.model_executor.models.qwen3_tts.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen3_tts"].get_stage(0)
        assert s.model_stage == "qwen3_tts"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.owns_tokenizer is True
        assert s.engine_output_type == "latent"
        assert s.sampling_constraints["stop_token_ids"] == [2150]
        # Stage 0 inherits the pipeline-level model_arch
        assert s.model_arch is None

    def test_code2wav_stage_has_per_stage_model_arch(self):
        import vllm_omni.model_executor.models.qwen3_tts.pipeline  # noqa: F401

        s = _PIPELINE_REGISTRY["qwen3_tts"].get_stage(1)
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"
        # Per-stage model_arch override (different from pipeline-level talker)
        assert s.model_arch == "Qwen3TTSCode2Wav"
        # tts_args is passed through via extras
        assert s.extras["tts_args"]["max_instructions_length"] == 500

    def test_per_stage_model_arch_flows_through_merge(self, tmp_path):
        """Verify the new ps.model_arch override survives merge_pipeline_deploy."""
        import vllm_omni.model_executor.models.qwen3_tts.pipeline  # noqa: F401
        from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_tts.yaml"
        if not deploy_path.exists():
            pytest.skip("qwen3_tts deploy yaml not found")

        deploy = load_deploy_config(deploy_path)
        pipeline = _PIPELINE_REGISTRY["qwen3_tts"]
        stages = merge_pipeline_deploy(pipeline, deploy)

        # Stage 0 inherits pipeline-level model_arch
        assert stages[0].yaml_engine_args["model_arch"] == "Qwen3TTSTalkerForConditionalGeneration"
        # Stage 1 uses its per-stage override
        assert stages[1].yaml_engine_args["model_arch"] == "Qwen3TTSCode2Wav"

    def test_subtalker_sampling_params_deep_merge_preserves_base_keys(self):
        """Verify subtalker sampling params participate in stage deep-merge."""
        from vllm_omni.config.stage_config import _deep_merge_stage

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


class TestBaseConfigInheritance:
    """Test deploy YAML base_config inheritance."""

    def test_ci_inherits_from_main(self):
        from tests.helpers.stage_config import get_deploy_config_path
        from vllm_omni.config.stage_config import load_deploy_config

        ci_path = Path(get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
        if not ci_path.exists():
            pytest.skip("CI deploy config not found")

        deploy = load_deploy_config(ci_path)
        assert len(deploy.stages) == 3
        # CI overrides
        assert deploy.stages[0].engine_extras.get("load_format") == "dummy"
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
        from tests.helpers.stage_config import get_deploy_config_path
        from vllm_omni.config.stage_config import load_deploy_config

        ci_path = Path(get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
        if not ci_path.exists():
            pytest.skip("CI deploy config not found")

        deploy = load_deploy_config(ci_path)
        s0 = deploy.stages[0].default_sampling_params
        # CI overrides max_tokens
        assert s0["max_tokens"] == 150
        # Inherited from base
        assert s0["temperature"] == 0.4
        assert s0["seed"] == 42

    def test_pure_inheritance_overlay(self, tmp_path):
        """An overlay with only ``base_config`` inherits everything."""
        from vllm_omni.config.stage_config import load_deploy_config

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
        from vllm_omni.config.stage_config import load_deploy_config

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
        from pathlib import Path

        from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config

        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        deploy = _apply_platform_overrides(deploy, platform="npu")

        assert deploy.stages[0].gpu_memory_utilization == 0.6
        assert deploy.stages[0].tensor_parallel_size == 2
        assert deploy.stages[0].devices == "0,1"
        # Stage 2 unaffected fields stay at base
        assert deploy.stages[2].enforce_eager is True

    def test_xpu_overrides(self):
        from pathlib import Path

        from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config

        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        deploy = _apply_platform_overrides(deploy, platform="xpu")

        assert deploy.stages[0].tensor_parallel_size == 4
        assert deploy.stages[0].devices == "0,1,2,3"
        assert deploy.stages[0].engine_extras.get("max_cudagraph_capture_size") == 0

    def test_unknown_platform_noop(self):
        from pathlib import Path

        from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config

        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        original_mem = deploy.stages[0].gpu_memory_utilization
        deploy = _apply_platform_overrides(deploy, platform="unknown_hw")
        assert deploy.stages[0].gpu_memory_utilization == original_mem

    def test_platforms_deep_merge_inheritance(self, tmp_path):
        """Overlay's platforms: block layers onto base's, per-stage."""
        from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config

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
        from pathlib import Path

        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401
        from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

        pipeline = _PIPELINE_REGISTRY["qwen3_omni_moe"]
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        stages = merge_pipeline_deploy(pipeline, deploy)

        # Simulate --stage-overrides '{"0": {"gpu_memory_utilization": 0.5}}'
        overrides = {"stage_0_gpu_memory_utilization": 0.5}
        stages[0].runtime_overrides = StageConfigFactory._merge_cli_overrides(stages[0], overrides)
        assert stages[0].runtime_overrides["gpu_memory_utilization"] == 0.5

    def test_global_override_applies_to_all(self):
        from pathlib import Path

        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401
        from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

        pipeline = _PIPELINE_REGISTRY["qwen3_omni_moe"]
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        stages = merge_pipeline_deploy(pipeline, deploy)

        overrides = {"enforce_eager": True}
        for s in stages:
            s.runtime_overrides = StageConfigFactory._merge_cli_overrides(s, overrides)
            assert s.runtime_overrides["enforce_eager"] is True


class TestCLIExplicitPrecedence:
    """Verify YAML > argparse defaults; explicit CLI args > YAML."""

    def _stages(self, cli_overrides, cli_explicit_keys):
        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401

        return StageConfigFactory._create_from_registry(
            "qwen3_omni_moe",
            cli_overrides=cli_overrides,
            cli_explicit_keys=cli_explicit_keys,
        )

    def test_explicit_cli_overrides_yaml(self):
        """User-typed --max-num-seqs wins over the deploy YAML value."""
        stages = self._stages(
            cli_overrides={"max_num_seqs": 999},
            cli_explicit_keys={"max_num_seqs"},
        )
        # Stage 2 yaml has max_num_seqs=1; explicit CLI must beat it.
        assert stages[2].runtime_overrides.get("max_num_seqs") == 999

    def test_default_cli_does_not_override_yaml(self):
        """Argparse defaults must NOT clobber values that are present in YAML."""
        stages = self._stages(
            cli_overrides={"max_num_seqs": 256},
            cli_explicit_keys=set(),  # user typed nothing
        )
        # Stage 2's YAML value (1) should win because the user didn't type --max-num-seqs.
        assert stages[2].runtime_overrides.get("max_num_seqs") != 256

    def test_default_cli_fills_missing_yaml_field(self):
        """Argparse defaults still fill fields the YAML doesn't set."""
        stages = self._stages(
            cli_overrides={"some_unrelated_knob": "fallback"},
            cli_explicit_keys=set(),
        )
        # Field absent from YAML → CLI default flows through as a fallback.
        assert stages[0].runtime_overrides.get("some_unrelated_knob") == "fallback"

    def test_per_stage_overrides_always_explicit(self):
        """``stage_<id>_*`` keys are always treated as explicit."""
        stages = self._stages(
            cli_overrides={"stage_0_gpu_memory_utilization": 0.42},
            cli_explicit_keys=set(),  # not in the explicit set, but per-stage
        )
        assert stages[0].runtime_overrides.get("gpu_memory_utilization") == 0.42

    def test_none_explicit_set_treats_all_as_explicit(self):
        """Programmatic Omni() callers (cli_explicit_keys=None) keep current behavior."""
        stages = self._stages(
            cli_overrides={"max_num_seqs": 999},
            cli_explicit_keys=None,
        )
        assert stages[2].runtime_overrides.get("max_num_seqs") == 999

    def test_explicit_async_chunk_false_overrides_yaml(self):
        """``--no-async-chunk`` flips the deploy-level async_chunk to False even
        when the YAML sets it to True. Verifies that the per-stage
        ``async_chunk: True`` injection in ``merge_pipeline_deploy`` is skipped
        and that ``async_chunk`` does not leak through ``_merge_cli_overrides``.
        """
        stages = self._stages(
            cli_overrides={"async_chunk": False},
            cli_explicit_keys={"async_chunk"},
        )
        # qwen3_omni_moe.yaml has `async_chunk: true`, so by default every
        # stage's engine_args would carry it. With the explicit override, it
        # must NOT show up.
        for stage in stages:
            assert stage.yaml_engine_args.get("async_chunk") is not True
            assert stage.runtime_overrides.get("async_chunk") is None

    def test_default_async_chunk_leaves_yaml_alone(self):
        """An unset ``--async-chunk`` (default None) must leave the YAML's True
        in force on every stage."""
        stages = self._stages(
            cli_overrides={"async_chunk": None},
            cli_explicit_keys=set(),
        )
        # qwen3_omni_moe.yaml: `async_chunk: true` → injected on every stage.
        for stage in stages:
            assert stage.yaml_engine_args.get("async_chunk") is True

    def test_explicit_enable_prefix_caching_overrides_yaml(self):
        """``--enable-prefix-caching`` (global) flips every stage's
        ``enable_prefix_caching`` to True regardless of the YAML default."""
        stages = self._stages(
            cli_overrides={"enable_prefix_caching": True},
            cli_explicit_keys={"enable_prefix_caching"},
        )
        for stage in stages:
            assert stage.runtime_overrides.get("enable_prefix_caching") is True

    def test_async_chunk_dispatches_processors(self):
        """A single ``qwen3_tts`` pipeline picks per-chunk vs end-to-end
        processors based on ``deploy.async_chunk``, without needing a
        separate variant pipeline registration."""
        import vllm_omni.model_executor.models.qwen3_tts.pipeline  # noqa: F401
        from vllm_omni.config.stage_config import (
            _PIPELINE_REGISTRY,
            DeployConfig,
            merge_pipeline_deploy,
        )

        pipeline = _PIPELINE_REGISTRY["qwen3_tts"]

        # async_chunk=True → stage 0's per-chunk processor wires up, stage 1
        # has no sync input processor.
        async_stages = merge_pipeline_deploy(pipeline, DeployConfig(async_chunk=True))
        assert (
            async_stages[0]
            .yaml_engine_args.get("custom_process_next_stage_input_func", "")
            .endswith("talker2code2wav_async_chunk")
        )
        assert async_stages[1].custom_process_input_func is None

        # async_chunk=False → stage 0 has no streaming processor, stage 1's
        # batch-end processor wires up.
        sync_stages = merge_pipeline_deploy(pipeline, DeployConfig(async_chunk=False))
        assert "custom_process_next_stage_input_func" not in sync_stages[0].yaml_engine_args
        assert sync_stages[1].custom_process_input_func is not None
        assert sync_stages[1].custom_process_input_func.endswith("talker2code2wav")


class TestSamplingConstraintsPrecedence:
    """Test that pipeline sampling_constraints override deploy defaults."""

    def test_constraints_win(self):
        from pathlib import Path

        import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401
        from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

        pipeline = _PIPELINE_REGISTRY["qwen3_omni_moe"]
        deploy_path = Path(__file__).parent.parent / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        stages = merge_pipeline_deploy(pipeline, deploy)

        # Pipeline says detokenize=True for thinker, deploy can't override
        assert stages[0].yaml_extras["default_sampling_params"]["detokenize"] is True
        # Pipeline says stop_token_ids=[2150] for talker
        assert stages[1].yaml_extras["default_sampling_params"]["stop_token_ids"] == [2150]
        # Deploy temperature still flows through
        assert stages[0].yaml_extras["default_sampling_params"]["temperature"] == 0.4

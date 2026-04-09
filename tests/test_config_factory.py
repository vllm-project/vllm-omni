# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for StageConfigFactory and related classes.
"""

from vllm_omni.config.stage_config import (
    ModelPipeline,
    StageConfig,
    StageConfigFactory,
    StageType,
)


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

        class MockParallelConfig:
            world_size = 4

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
            "custom_engine_flag": True,  # Not in _INTERNAL_KEYS, so forwarded
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


class TestArchitectureFallback:
    """Tests for architecture-based model detection fallback."""

    def test_architecture_models_mapping_exists(self):
        """Test that _ARCHITECTURE_MODELS contains expected entries."""
        assert "MiMoAudioForConditionalGeneration" in StageConfigFactory._ARCHITECTURE_MODELS
        assert StageConfigFactory._ARCHITECTURE_MODELS["MiMoAudioForConditionalGeneration"] == "mimo_audio"
        assert "HunyuanImage3ForCausalMM" in StageConfigFactory._ARCHITECTURE_MODELS
        assert StageConfigFactory._ARCHITECTURE_MODELS["HunyuanImage3ForCausalMM"] == "hunyuan_image3"

    def test_mimo_audio_in_pipeline_models(self):
        """Test that mimo_audio is registered in PIPELINE_MODELS."""
        assert "mimo_audio" in StageConfigFactory.PIPELINE_MODELS

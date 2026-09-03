# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy
import sys
import types
from dataclasses import fields, replace
from pathlib import Path

import pytest
from pydantic.fields import FieldInfo
from vllm.config import CacheConfig as VllmCacheConfig
from vllm.config import CompilationConfig as VllmCompilationConfig
from vllm.config import LoadConfig as VllmLoadConfig
from vllm.config import ParallelConfig as VllmParallelConfig
from vllm.config import ProfilerConfig as VllmProfilerConfig
from vllm.config import SchedulerConfig as VllmSchedulerConfig
from vllm.engine.arg_utils import EngineArgs

from vllm_omni.config.omni_config import (
    _LLM_STAGE_ENGINE_FIELDS,
    OmniStageCacheConfig,
    OmniStageLoadConfig,
    OmniStageParallelConfig,
    OmniStageSchedulerConfig,
    VllmOmniARStageConfig,
    VllmOmniConfig,
)
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    StageType,
    build_stage_runtime_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.stage_init_utils import (
    build_engine_args_dict,
    build_engine_args_dict_from_omni_stage_config,
    build_legacy_engine_args_dict,
)
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY_DIR = Path(__file__).parents[2] / "vllm_omni" / "deploy"


def _effective_backend_values(config_cls: type, engine_args: dict) -> dict[str, object]:
    backend_fields = fields(config_cls)
    backend_field_names = {backend_field.name for backend_field in backend_fields}
    backend_config = config_cls(
        **{name: copy.deepcopy(value) for name, value in engine_args.items() if name in backend_field_names}
    )
    effective_values: dict[str, object] = {}
    for backend_field in backend_fields:
        value = getattr(backend_config, backend_field.name)
        if isinstance(value, FieldInfo):
            value = value.get_default(call_default_factory=True)
        effective_values[backend_field.name] = value
    return effective_values


_LLM_BACKEND_FIELDS = frozenset(field.name for field in fields(OmniEngineArgs))
_DIFFUSION_BACKEND_FIELDS = frozenset(field.name for field in fields(OmniDiffusionConfig))
_TOPOLOGY_ONLY_ENGINE_ARGS = frozenset({"inline_diffusion"})
_OMNI_ONLY_LLM_STAGE_ENGINE_FIELDS = frozenset(
    {
        "active_stream_window",
        "codec_frame_rate_hz",
        "custom_voice_dir",
        "devices",
        "disable_autocast",
        "enable_multithread_weight_load",
        "env",
        "has_sampling_extra_args",
        "log_level",
        "log_stats",
        "model_arch",
        "model_subdir",
        "num_gpus",
        "num_replicas",
        "num_weight_load_threads",
        "omni_kv_config",
        "parallel_config",
        "silence_ban_frames",
        "subtalker_sampling_params",
        "task_type",
        "tokenizer_subdir",
    }
)


@pytest.fixture(autouse=True)
def _stable_engine_arg_environment(monkeypatch, tmp_path):
    from vllm_omni import platforms

    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP16", raising=False)
    xcodec_model = tmp_path / "xcodec-model"
    xcodec_model.mkdir()
    monkeypatch.setenv("XCODEC1_PATH", str(xcodec_model))
    platform = platforms.current_omni_platform
    monkeypatch.setattr(platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(platform, "device_type", "cpu", raising=False)
    monkeypatch.setattr(platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        OmniDiffusionConfig,
        "_resolve_master_port",
        lambda config: config.master_port or 29500,
    )

    def _resolve_test_worker(engine_args):
        worker_type = engine_args.get("worker_type")
        if worker_type is not None and engine_args.get("worker_cls") in (None, "auto"):
            engine_args["worker_cls"] = f"test.{worker_type}.Worker"

    monkeypatch.setattr(stage_init_utils, "resolve_worker_cls", _resolve_test_worker)

    class _TestConnectorRunner(OmniConnectorModelRunnerMixin):
        pass

    class _TestWorker:
        model_runner_cls = _TestConnectorRunner

    for worker_type in ("ar", "generation"):
        worker_module = types.ModuleType(f"test.{worker_type}")
        setattr(worker_module, "Worker", _TestWorker)
        monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)


def _engine_arg_inputs(tmp_path: Path) -> tuple[PipelineConfig, DeployConfig, str]:
    parent_model = tmp_path / "parent-model"
    stage_model = tmp_path / "stage-model"
    diffusion_model = tmp_path / "diffusion-model"
    for root in (parent_model, stage_model, diffusion_model):
        root.mkdir()
    (stage_model / "ar-model").mkdir()
    (stage_model / "ar-tokenizer").mkdir()

    pipeline = PipelineConfig(
        model_type="typed-engine-args",
        model_arch="PipelineModel",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="thinker",
                execution_type=StageExecutionType.LLM_AR,
                requires_multimodal_data=True,
                hf_config_name="thinker_config",
                engine_output_type="hidden_states",
                model_arch="Qwen3OmniMoeForConditionalGeneration",
                model_subdir="ar-model",
                tokenizer_subdir="ar-tokenizer",
                retains_state_across_chunks=True,
                async_chunk_process_next_stage_input_func="operator.add",
            ),
            StagePipelineConfig(
                stage_id=1,
                model_stage="talker",
                execution_type=StageExecutionType.LLM_GENERATION,
                input_sources=(0,),
                engine_output_type="audio_tokens",
            ),
            StagePipelineConfig(
                stage_id=2,
                model_stage="dit",
                execution_type=StageExecutionType.DIFFUSION,
                input_sources=(1,),
                final_output=True,
                final_output_type="image",
                model_arch="TypedDiffusionPipeline",
            ),
        ),
    )
    deploy = DeployConfig(
        async_chunk=True,
        trust_remote_code=True,
        dtype="float16",
        distributed_executor_backend="mp",
        stages=[
            StageDeployConfig(
                stage_id=0,
                tensor_parallel_size=2,
                max_num_seqs=8,
                default_sampling_params={"extra_args": {"voice": "test"}},
                engine_extras={
                    "model": str(stage_model),
                    "attention_backend": "FLASHINFER",
                    "attention_config": {"use_trtllm_attention": False},
                    "hf_overrides": {"rope_scaling": {"factor": 2}},
                    "limit_mm_per_prompt": {"audio": 1},
                    "logits_processors": ["test.custom.LogitsProcessor"],
                    "kv_cache_memory_bytes": 1024,
                    "omni_kv_config": {"need_send_cache": True},
                    "max_cudagraph_capture_size": 0,
                    "worker_cls": "test.custom.Worker",
                },
            ),
            StageDeployConfig(
                stage_id=1,
                max_num_seqs=4,
            ),
            StageDeployConfig(
                stage_id=2,
                tensor_parallel_size=2,
                vae_parallel_mode="spatial_shard_height",
                diffusion_attention_config={
                    "default": {"backend": "FLASH_ATTN"},
                    "per_role": {"cross": {"backend": "TORCH_SDPA"}},
                },
                engine_extras={
                    "model": str(diffusion_model),
                    "engine_backend": "test.diffusion.Engine",
                    "enable_session_state_manager": True,
                },
            ),
        ],
    )
    return pipeline, deploy, str(parent_model)


def _legacy_and_typed_stages(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    model: str,
    cli_overrides: dict[str, object] | None = None,
):
    resolved_cli_overrides = {"model": model, **(cli_overrides or {})}
    legacy_stage_configs = merge_pipeline_deploy(
        pipeline,
        copy.deepcopy(deploy),
    )
    for stage in legacy_stage_configs:
        stage.runtime_overrides = build_stage_runtime_overrides(
            stage.stage_id,
            resolved_cli_overrides,
        )
    legacy_stages = [stage.to_omegaconf() for stage in legacy_stage_configs]
    omni_config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=copy.deepcopy(deploy),
        cli_overrides=resolved_cli_overrides,
    )
    return legacy_stages, omni_config


def test_llm_stage_engine_field_schema_tracks_upstream_engine_args():
    upstream_engine_fields = frozenset(field.name for field in fields(EngineArgs))

    assert _LLM_STAGE_ENGINE_FIELDS - upstream_engine_fields == _OMNI_ONLY_LLM_STAGE_ENGINE_FIELDS


def test_typed_llm_projection_discovers_explicit_upstream_config_fields():
    stage_config = VllmOmniARStageConfig(
        stage_pipeline_config=StagePipelineConfig(stage_id=0, model_stage="test"),
        load_config=OmniStageLoadConfig(ignore_patterns=["*.bin"]),
        cache_config=OmniStageCacheConfig(block_size=32, cache_dtype="fp8"),
        scheduler_config=OmniStageSchedulerConfig(
            prefill_schedule_interval=2,
            policy="priority",
        ),
        parallel_config=OmniStageParallelConfig(
            data_parallel_master_ip="10.0.0.1",
            enable_dbo=True,
        ),
    )

    engine_args = stage_init_utils._project_omni_stage_engine_args(stage_config)

    assert engine_args["ignore_patterns"] == ["*.bin"]
    assert engine_args["block_size"] == 32
    assert engine_args["kv_cache_dtype"] == "fp8"
    assert engine_args["prefill_schedule_interval"] == 2
    assert engine_args["scheduling_policy"] == "priority"
    assert engine_args["data_parallel_address"] == "10.0.0.1"
    assert engine_args["enable_dbo"] is True


def test_typed_llm_projection_does_not_emit_inherited_upstream_defaults():
    stage_config = VllmOmniARStageConfig(
        stage_pipeline_config=StagePipelineConfig(stage_id=0, model_stage="test"),
    )

    engine_args = stage_init_utils._project_omni_stage_engine_args(stage_config)

    inherited_defaults = {
        "ignore_patterns",
        "block_size",
        "kv_cache_dtype",
        "prefill_schedule_interval",
        "scheduling_policy",
        "data_parallel_address",
        "data_parallel_rank",
        "data_parallel_size_local",
        "data_parallel_rpc_port",
        "enable_dbo",
    }
    assert inherited_defaults.isdisjoint(engine_args)


def test_typed_llm_projection_rejects_explicit_fields_owned_by_another_boundary():
    stage_config = VllmOmniARStageConfig(
        stage_pipeline_config=StagePipelineConfig(stage_id=0, model_stage="test"),
        scheduler_config=OmniStageSchedulerConfig(scheduler_cls="test.CustomScheduler"),
    )

    with pytest.raises(ValueError, match=r"no EngineArgs projection: scheduler_cls"):
        stage_init_utils._project_omni_stage_engine_args(stage_config)


@pytest.mark.parametrize("stage_id", [0, 1], ids=["ar", "generation"])
def test_typed_llm_engine_args_preserve_upstream_config_objects(tmp_path, stage_id):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    compilation_config = VllmCompilationConfig(backend="eager")
    profiler_config = VllmProfilerConfig(profiler="cuda")
    stage_prefix = f"stage_{stage_id}_"
    omni_config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=copy.deepcopy(deploy),
        cli_overrides={
            "model": model,
            f"{stage_prefix}compilation_config": compilation_config,
            f"{stage_prefix}profiler_config": profiler_config,
        },
    )

    stage_config = omni_config.stage_by_id(stage_id)
    typed_args = build_engine_args_dict_from_omni_stage_config(stage_config, model)

    assert isinstance(stage_config.compilation_config, VllmCompilationConfig)
    assert isinstance(typed_args["compilation_config"], VllmCompilationConfig)
    assert typed_args["compilation_config"] is not stage_config.compilation_config
    assert typed_args["compilation_config"].backend == "eager"

    assert isinstance(stage_config.profiler_config, VllmProfilerConfig)
    assert isinstance(typed_args["profiler_config"], VllmProfilerConfig)
    assert typed_args["profiler_config"] is not stage_config.profiler_config
    assert typed_args["profiler_config"].profiler == "cuda"


def test_typed_llm_engine_args_preserve_legacy_adapter_behavior(tmp_path):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    legacy_stages, omni_config = _legacy_and_typed_stages(pipeline, deploy, model)
    connector_spec = {"name": "SharedMemoryConnector", "extra": {"mode": "test"}}
    cli_tokenizer = "/external/tokenizer"

    typed_args_by_stage = {}
    for stage_id in (0, 1):
        legacy_args = build_legacy_engine_args_dict(
            legacy_stages[stage_id],
            model,
            stage_connector_spec=connector_spec,
            cli_tokenizer=cli_tokenizer,
        )
        typed_args = build_engine_args_dict_from_omni_stage_config(
            omni_config.stage_by_id(stage_id),
            model,
            stage_connector_spec=connector_spec,
            cli_tokenizer=cli_tokenizer,
        )
        typed_args_by_stage[stage_id] = typed_args

        expected_args = {name: value for name, value in legacy_args.items() if name not in _TOPOLOGY_ONLY_ENGINE_ARGS}
        assert {name: typed_args[name] for name in expected_args} == expected_args
        assert _TOPOLOGY_ONLY_ENGINE_ARGS.isdisjoint(typed_args)

    thinker_args = typed_args_by_stage[0]
    inherited_vllm_fields = {
        field.name
        for config_cls in (VllmLoadConfig, VllmCacheConfig, VllmSchedulerConfig, VllmParallelConfig)
        for field in fields(config_cls)
    }
    topology_projected_fields = {"scheduler_cls"}
    assert (inherited_vllm_fields - _LLM_STAGE_ENGINE_FIELDS - topology_projected_fields).isdisjoint(thinker_args)
    assert thinker_args["model"] == str(tmp_path / "stage-model" / "ar-model")
    assert thinker_args["tokenizer"] == str(tmp_path / "stage-model" / "ar-tokenizer")
    assert thinker_args["stage_id"] == 0
    assert thinker_args["stage_connector_spec"] == connector_spec
    assert thinker_args["has_sampling_extra_args"] is True
    assert thinker_args["omni_kv_config"] == {"need_send_cache": True}
    assert thinker_args["worker_cls"] == "test.custom.Worker"
    assert thinker_args["attention_backend"] == "FLASHINFER"
    assert thinker_args["attention_config"] == {"use_trtllm_attention": False}
    assert thinker_args["trust_remote_code"] is True
    assert thinker_args["dtype"] == "float16"
    assert thinker_args["distributed_executor_backend"] == "mp"
    assert thinker_args["tensor_parallel_size"] == 2
    assert thinker_args["hf_overrides"] == {"rope_scaling": {"factor": 2}}
    assert thinker_args["limit_mm_per_prompt"] == {"audio": 1}
    assert thinker_args["logits_processors"] == ["test.custom.LogitsProcessor"]
    assert thinker_args["retains_state_across_chunks"] is True
    assert thinker_args["kv_cache_memory_bytes"] == 1024
    assert thinker_args["max_cudagraph_capture_size"] == 0

    talker_args = typed_args_by_stage[1]
    assert talker_args["model"] == model
    assert talker_args["tokenizer"] == cli_tokenizer
    assert talker_args["worker_cls"] == "test.generation.Worker"
    assert talker_args["disable_hybrid_kv_cache_manager"] is True
    assert talker_args["enable_prefix_caching"] is False


def test_typed_diffusion_engine_args_use_structured_diffusion_config(tmp_path):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    legacy_stages, omni_config = _legacy_and_typed_stages(pipeline, deploy, model)

    legacy_args = build_legacy_engine_args_dict(legacy_stages[2], model)
    typed_args = build_engine_args_dict_from_omni_stage_config(
        omni_config.stage_by_id(2),
        model,
    )

    assert typed_args["stage_id"] == legacy_args["stage_id"] == 2
    assert typed_args["model"] == legacy_args["model"] == str(tmp_path / "diffusion-model")
    assert omni_config.stage_by_id(2).diffusion_config.model == str(tmp_path / "diffusion-model")
    assert typed_args["model_stage"] == legacy_args["model_stage"] == "dit"
    assert typed_args["model_arch"] == legacy_args["model_arch"] == "TypedDiffusionPipeline"
    assert typed_args["model_class_name"] == legacy_args["model_class_name"] == "TypedDiffusionPipeline"
    assert typed_args["engine_backend"] == legacy_args["engine_backend"] == "test.diffusion.Engine"
    assert typed_args["enable_session_state_manager"] is legacy_args["enable_session_state_manager"] is True
    assert typed_args["parallel_config"]["tensor_parallel_size"] == 2
    assert typed_args["parallel_config"]["vae_parallel_mode"] == "spatial_shard_height"
    assert isinstance(typed_args["diffusion_attention_config"], AttentionConfig)
    assert typed_args["diffusion_attention_config"].default.backend == "FLASH_ATTN"
    assert typed_args["diffusion_attention_config"].per_role["cross"].backend == "TORCH_SDPA"


def test_engine_args_consume_stage_diffusion_attention_shorthand(tmp_path):
    """A stage-level ``diffusion_attention_backend`` shorthand is folded into the structured config."""
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    deploy.stages[2] = replace(
        deploy.stages[2],
        diffusion_attention_config=None,
        diffusion_attention_backend="TORCH_SDPA",
    )
    legacy_stages, omni_config = _legacy_and_typed_stages(pipeline, deploy, model)

    legacy_args = build_legacy_engine_args_dict(legacy_stages[2], model)
    typed_args = build_engine_args_dict_from_omni_stage_config(
        omni_config.stage_by_id(2),
        model,
    )

    for engine_args in (legacy_args, typed_args):
        assert engine_args.get("diffusion_attention_backend") is None
        assert isinstance(engine_args["diffusion_attention_config"], AttentionConfig)
        assert engine_args["diffusion_attention_config"].default.backend == "TORCH_SDPA"
        od_config = OmniDiffusionConfig.from_kwargs(**engine_args)
        assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"


@pytest.mark.parametrize(
    "yaml_attention_config",
    [
        {"default": {"backend": "FLASH_ATTN"}, "per_role": {"cross": {"backend": "SAGE_ATTN"}}},
        {"per_role": {"cross": {"backend": "SAGE_ATTN"}}},
    ],
    ids=["yaml-default", "yaml-per-role-only"],
)
def test_engine_args_apply_cli_attention_shorthand_over_yaml_config(tmp_path, yaml_attention_config):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    deploy.stages[2] = replace(deploy.stages[2], diffusion_attention_config=yaml_attention_config)
    legacy_stages, omni_config = _legacy_and_typed_stages(
        pipeline,
        deploy,
        model,
        cli_overrides={"stage_2_diffusion_attention_backend": "TORCH_SDPA"},
    )

    legacy_args = build_legacy_engine_args_dict(legacy_stages[2], model)
    typed_args = build_engine_args_dict_from_omni_stage_config(
        omni_config.stage_by_id(2),
        model,
    )

    for engine_args in (legacy_args, typed_args):
        assert engine_args.get("diffusion_attention_backend") is None
        attention_config = engine_args["diffusion_attention_config"]
        assert attention_config.default.backend == "TORCH_SDPA"
        assert attention_config.per_role["cross"].backend == "SAGE_ATTN"
        od_config = OmniDiffusionConfig.from_kwargs(**engine_args)
        assert od_config.diffusion_attention_config.default.backend == "TORCH_SDPA"


def test_typed_engine_args_preserve_explicit_backend_default_overrides(tmp_path):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    deploy.enable_prefix_caching = False
    deploy.enable_chunked_prefill = False
    thinker_deploy = deploy.stages[0]
    thinker_deploy.gpu_memory_utilization = 0.75
    thinker_deploy.disable_hybrid_kv_cache_manager = False
    thinker_deploy.async_scheduling = False
    legacy_stages, omni_config = _legacy_and_typed_stages(pipeline, deploy, model)

    legacy_args = build_legacy_engine_args_dict(legacy_stages[0], model)
    typed_args = build_engine_args_dict_from_omni_stage_config(
        omni_config.stage_by_id(0),
        model,
    )
    expected = {
        "gpu_memory_utilization": 0.75,
        "enable_prefix_caching": False,
        "disable_hybrid_kv_cache_manager": False,
        "max_num_seqs": 8,
        "enable_chunked_prefill": False,
        "async_scheduling": False,
    }

    assert {name: legacy_args[name] for name in expected} == expected
    assert {name: typed_args[name] for name in expected} == expected


@pytest.mark.parametrize("stage_scoped", [False, True], ids=["global", "stage-scoped"])
def test_typed_engine_args_preserve_inherited_model_and_load_cli_fields(tmp_path, stage_scoped):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    expected = {
        "revision": "model-rev",
        "tokenizer_revision": "tokenizer-rev",
        "code_revision": "code-rev",
        "seed": 123,
        "download_dir": str(tmp_path / "downloads"),
    }
    cli_overrides = {f"stage_0_{name}" if stage_scoped else name: value for name, value in expected.items()}
    legacy_stages, omni_config = _legacy_and_typed_stages(
        pipeline,
        deploy,
        model,
        cli_overrides,
    )

    legacy_args = build_legacy_engine_args_dict(legacy_stages[0], model)
    typed_args = build_engine_args_dict_from_omni_stage_config(
        omni_config.stage_by_id(0),
        model,
    )

    assert {name: legacy_args[name] for name in expected} == expected
    assert {name: typed_args[name] for name in expected} == expected


def test_typed_engine_args_preserve_deploy_subdirectory_precedence(tmp_path):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    stage_model = tmp_path / "stage-model"
    (stage_model / "override-model").mkdir()
    (stage_model / "override-tokenizer").mkdir()
    deploy.stages[0].engine_extras.update(
        {
            "model_subdir": "override-model",
            "tokenizer_subdir": "override-tokenizer",
        }
    )
    legacy_stages, omni_config = _legacy_and_typed_stages(pipeline, deploy, model)

    typed_stage = omni_config.stage_by_id(0)
    legacy_args = build_legacy_engine_args_dict(legacy_stages[0], model)
    typed_args = build_engine_args_dict_from_omni_stage_config(typed_stage, model)

    assert typed_stage.model_config.model_subdir == "override-model"
    assert typed_stage.model_config.tokenizer_subdir == "override-tokenizer"
    assert typed_args["model"] == legacy_args["model"] == str(stage_model / "override-model")
    assert typed_args["tokenizer"] == legacy_args["tokenizer"] == str(stage_model / "override-tokenizer")


@pytest.mark.parametrize(
    ("cli_overrides", "expected"),
    [
        ({}, {"need_recv_cache": True}),
        (
            {"stage_0_omni_kv_config": {"need_send_cache": False}},
            {"need_send_cache": False},
        ),
    ],
    ids=["topology-over-deploy", "cli-over-topology"],
)
def test_typed_engine_args_preserve_legacy_omni_kv_precedence(tmp_path, cli_overrides, expected):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    thinker = replace(
        pipeline.stages[0],
        omni_kv_config={"need_recv_cache": True},
    )
    pipeline = replace(pipeline, stages=(thinker, *pipeline.stages[1:]))
    legacy_stages, omni_config = _legacy_and_typed_stages(
        pipeline,
        deploy,
        model,
        cli_overrides,
    )

    legacy_args = build_legacy_engine_args_dict(legacy_stages[0], model)
    typed_stage = omni_config.stage_by_id(0)
    typed_args = build_engine_args_dict_from_omni_stage_config(typed_stage, model)

    assert typed_stage.connector_config.omni_kv_config == expected
    assert typed_args["omni_kv_config"] == legacy_args["omni_kv_config"] == expected


def test_build_engine_args_dict_preserves_legacy_api(tmp_path):
    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    legacy_stages, _ = _legacy_and_typed_stages(pipeline, deploy, model)
    connector_spec = {"name": "SharedMemoryConnector", "extra": {}}

    compatibility_args = build_engine_args_dict(
        copy.deepcopy(legacy_stages[0]),
        model,
        stage_connector_spec=connector_spec,
        cli_tokenizer="/external/tokenizer",
    )
    explicitly_legacy_args = build_legacy_engine_args_dict(
        copy.deepcopy(legacy_stages[0]),
        model,
        stage_connector_spec=connector_spec,
        cli_tokenizer="/external/tokenizer",
    )

    assert compatibility_args == explicitly_legacy_args


def test_legacy_engine_args_drop_none_tp_without_mutating_stage_config():
    stage_config = types.SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args={"tensor_parallel_size": None},
        default_sampling_params={},
    )

    engine_args = build_legacy_engine_args_dict(
        stage_config,
        model="test-model",
    )

    assert "tensor_parallel_size" not in engine_args
    assert stage_config.engine_args == {"tensor_parallel_size": None}


def test_typed_engine_args_own_rocm_attention_default(monkeypatch, tmp_path):
    from vllm_omni import platforms

    pipeline, deploy, model = _engine_arg_inputs(tmp_path)
    legacy_stages, omni_config = _legacy_and_typed_stages(pipeline, deploy, model)
    monkeypatch.setattr(platforms.current_omni_platform, "is_rocm", lambda: True)
    aiter_module = types.ModuleType("vllm._aiter_ops")
    setattr(
        aiter_module,
        "rocm_aiter_ops",
        types.SimpleNamespace(is_enabled=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", aiter_module)

    legacy_args = build_legacy_engine_args_dict(legacy_stages[1], model)
    typed_args = build_engine_args_dict_from_omni_stage_config(
        omni_config.stage_by_id(1),
        model,
    )

    assert "attention_backend" not in legacy_args
    assert typed_args["attention_backend"] == "TRITON_ATTN"


def test_typed_ming_image_engine_args_defer_diffusion_batch_default():
    pipeline = resolve_pipeline_config("ming_flash_omni_image")
    assert pipeline is not None
    deploy = load_deploy_config(_DEPLOY_DIR / "ming_flash_omni_image.yaml")
    legacy_stages, omni_config = _legacy_and_typed_stages(
        pipeline,
        deploy,
        model="/tmp",
    )

    legacy_args = build_legacy_engine_args_dict(legacy_stages[1], model="/tmp")
    typed_args = build_engine_args_dict_from_omni_stage_config(
        omni_config.stage_by_id(1),
        model="/tmp",
    )
    typed_backend_args = {name: value for name, value in typed_args.items() if name in _DIFFUSION_BACKEND_FIELDS}

    assert "max_num_seqs" not in legacy_args
    assert "max_num_seqs" not in typed_args
    assert OmniDiffusionConfig(**typed_backend_args).max_num_seqs == 1


def _fake_model_root(pipeline, tmp_path):
    """A model directory carrying every subfolder the pipeline's stages declare.

    Stage init fails closed when a declared ``model_subdir``/``tokenizer_subdir``
    is not a real directory, because the joined path would otherwise reach
    HuggingFace as a malformed repo id (issue #6638).
    """
    root = tmp_path / "model"
    root.mkdir(exist_ok=True)
    for stage in pipeline.stages:
        for subdir in (stage.model_subdir, stage.tokenizer_subdir):
            if subdir:
                (root / subdir).mkdir(parents=True, exist_ok=True)
    return str(root)


@pytest.mark.parametrize("model_type", sorted(OMNI_PIPELINES))
def test_typed_engine_args_match_current_registry_backend_semantics(model_type, tmp_path):
    pipeline = resolve_pipeline_config(model_type)
    if pipeline is None:
        pytest.skip(f"Pipeline {model_type!r} requires an HF config to resolve")

    deploy = (
        load_deploy_config(_DEPLOY_DIR / pipeline.default_deploy_config_name)
        if pipeline.default_deploy_config_name is not None
        else DeployConfig()
    )
    model = _fake_model_root(pipeline, tmp_path)
    legacy_stages, omni_config = _legacy_and_typed_stages(
        pipeline,
        deploy,
        model=model,
    )

    for legacy_stage in legacy_stages:
        stage_id = legacy_stage.stage_id
        legacy_args = build_legacy_engine_args_dict(legacy_stage, model=model)
        typed_args = build_engine_args_dict_from_omni_stage_config(
            omni_config.stage_by_id(stage_id),
            model=model,
        )
        if legacy_stage.stage_type == StageType.DIFFUSION:
            backend_fields = _DIFFUSION_BACKEND_FIELDS
            backend_config_cls = OmniDiffusionConfig
        else:
            backend_fields = _LLM_BACKEND_FIELDS
            backend_config_cls = OmniEngineArgs

        backend_fields -= _TOPOLOGY_ONLY_ENGINE_ARGS
        missing_fields = (legacy_args.keys() & backend_fields) - typed_args.keys()
        assert not missing_fields, f"{model_type} stage {stage_id} lost backend fields: {sorted(missing_fields)}"

        legacy_effective_args = _effective_backend_values(backend_config_cls, legacy_args)
        typed_effective_args = _effective_backend_values(backend_config_cls, typed_args)
        assert typed_effective_args == legacy_effective_args, (
            f"{model_type} stage {stage_id} changed effective backend arguments"
        )

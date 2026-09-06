# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""E2E tests that quant/stage CLI args resolve onto each stage's built config."""

import json
import os
from contextlib import contextmanager
from unittest import mock

import pytest
from transformers import LlamaConfig
from vllm import SamplingParams
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

import vllm_omni.diffusion.stage_diffusion_client as stage_diffusion_client
import vllm_omni.engine.async_omni_engine as async_omni_engine
import vllm_omni.engine.stage_init_utils as stage_init_utils
import vllm_omni.engine.stage_runtime as stage_runtime
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.entrypoints.omni import Omni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# A serialized quantization_config as it appears in a checkpoint's config.json.
_SERIALIZED_FP8 = {
    "quant_method": "fp8",
    "is_checkpoint_fp8_serialized": True,
    "activation_scheme": "static",
}

# Keep the LLM build off the tokenizer/weight/compile paths (none exist on disk).
_LLM_ENGINE_ARGS = {
    "worker_cls": "auto",
    "skip_tokenizer_init": True,
    "enforce_eager": True,
}

# Minimal patched LLM only pipeline config that we can use for testing the AR path.
_LLM_PIPELINE = PipelineConfig(
    model_type="llama",
    model_arch="LlamaForCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _write_llm_model_dir(path, quantization_config: dict | None = None) -> str:
    """A LlamaConfig dir that will be resolved to an LLM stage."""
    kwargs: dict = {}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    LlamaConfig(**kwargs).save_pretrained(path)
    return str(path)


def _write_diffusion_model_dir(path, quantization_config: dict | None = None) -> str:
    """A Gr00tN1d7 dir that will be resolved to a Diffusion stage."""
    config: dict = {"model_type": "Gr00tN1d7"}
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    return str(path)


@pytest.fixture
def llm_model_dir(tmp_path):
    return _write_llm_model_dir(tmp_path)


@pytest.fixture
def diffusion_model_dir(tmp_path):
    return _write_diffusion_model_dir(tmp_path)


class _FakeStageClient:
    """Stand-in for a launched stage client without the worker process."""

    def __init__(self, metadata):
        self.default_sampling_params = SamplingParams()
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self.stage_type = metadata.stage_type
        self.model_stage = metadata.model_stage
        self.prompt_expand_func = None
        self.is_comprehension = getattr(metadata, "is_comprehension", False)

    def shutdown(self):
        pass


@contextmanager
def built_stage_configs(model: str, **kwargs):
    """Run the real resolver & stub only worker spawn. This records each stage's
    built config & the Omni instance, but uses as much as the real build path as
    possible (i.e., Vllmconfig for LLM & OmniDiffusionConfig for diffusion).
    """
    built: list = []

    real_build_vllm = stage_runtime.build_vllm_config
    real_build_diffusion = stage_init_utils.build_diffusion_config

    def _record_vllm(*args, **kw):
        vllm_config, executor_class = real_build_vllm(*args, **kw)
        built.append(vllm_config)
        return vllm_config, executor_class

    def _record_diffusion(*args, **kw):
        od_config = real_build_diffusion(*args, **kw)
        built.append(od_config)
        return od_config

    def _skip_llm_spawn(runtime, plan, stage_init_timeout):
        return _FakeStageClient(plan.metadata)

    def _skip_diffusion_client(_model, _od_config, metadata, *args, **kw):
        return _FakeStageClient(metadata)

    async def _noop_run(self):
        return None

    with (
        mock.patch.dict(OMNI_PIPELINES, {"llama": _LLM_PIPELINE}),
        mock.patch.object(stage_runtime, "build_vllm_config", side_effect=_record_vllm),
        mock.patch.object(stage_init_utils, "build_diffusion_config", side_effect=_record_diffusion),
        mock.patch.object(stage_runtime.StageRuntime, "_initialize_local_llm_replica", _skip_llm_spawn),
        mock.patch.object(stage_diffusion_client, "create_diffusion_client", side_effect=_skip_diffusion_client),
        mock.patch.object(async_omni_engine.Orchestrator, "run", _noop_run),
        mock.patch.object(stage_init_utils.current_omni_platform, "get_device_count", return_value=1),
    ):
        omni = Omni(model=model, **kwargs)
        try:
            yield omni, built
        finally:
            try:
                omni.shutdown()
            except Exception:
                pass


class TestQuantization:
    """Ensure kwargs + serialized-checkpoint quant both resolve onto the built config,
    for LLM and diffusion stages."""

    def test_llm_cli_quantization_is_online_fp8(self, llm_model_dir):
        with built_stage_configs(llm_model_dir, quantization="fp8", **_LLM_ENGINE_ARGS) as (_, built):
            vllm_config = built[0]
            assert vllm_config.quant_config is not None
            assert vllm_config.quant_config.get_name() == "fp8"
            assert vllm_config.quant_config.is_checkpoint_fp8_serialized is False
            assert vllm_config.model_config.quantization == "fp8"

    def test_diffusion_cli_quantization_is_online_fp8(self, diffusion_model_dir):
        with built_stage_configs(diffusion_model_dir, quantization="fp8") as (_, built):
            qc = built[0].quantization_config
            assert qc is not None
            assert qc.get_name() == "fp8"
            assert qc.is_checkpoint_fp8_serialized is False

    def test_llm_serialized_checkpoint_is_serialized_fp8(self, tmp_path):
        """Serialized fp8 checkpoint, no CLI flag: must build a serialized fp8 config."""
        model = _write_llm_model_dir(tmp_path, quantization_config=_SERIALIZED_FP8)
        with built_stage_configs(model, **_LLM_ENGINE_ARGS) as (_, built):
            vllm_config = built[0]
            assert vllm_config.quant_config is not None
            assert vllm_config.quant_config.get_name() == "fp8"
            assert vllm_config.quant_config.is_checkpoint_fp8_serialized is True
            assert vllm_config.model_config.quantization == "fp8"

    def test_diffusion_serialized_checkpoint_is_serialized_fp8(self, tmp_path):
        """Serialized fp8 checkpoint, no CLI flag: must carry quant to the built config."""
        model = _write_diffusion_model_dir(tmp_path, quantization_config=_SERIALIZED_FP8)
        with built_stage_configs(model) as (_, built):
            qc = built[0].quantization_config
            assert qc is not None
            assert qc.get_name() == "fp8"
            assert qc.is_checkpoint_fp8_serialized is True

    def test_llm_builds_offline_without_quant(self, llm_model_dir):
        with built_stage_configs(llm_model_dir, **_LLM_ENGINE_ARGS) as (_, built):
            assert built[0].model_config.quantization is None
            assert built[0].quant_config is None


class TestDiffusionStageKwargs:
    """Diffusion kwargs actually land on the built OmniDiffusionConfig."""

    def test_cli_kwargs_reach_built_config(self, diffusion_model_dir):
        with built_stage_configs(
            diffusion_model_dir,
            lora_path="/tmp/lora",
            diffusion_kv_cache_dtype="fp8",
            static_lora_scale=0.7,
        ) as (_, built):
            od = built[0]
            assert od.lora_path == "/tmp/lora"
            assert od.diffusion_kv_cache_dtype == "fp8"
            # Tests static_lora_scale aliasing to lora_scale
            assert od.lora_scale == 0.7

    def test_enable_ar_profiler_reaches_top_level_not_config(self, diffusion_model_dir):
        """enable_ar_profiler is a top-level Omni flag, not a model-config field."""
        with built_stage_configs(diffusion_model_dir, enable_ar_profiler=True) as (omni, built):
            assert omni._enable_ar_profiler is True
            assert not hasattr(built[0], "enable_ar_profiler")


class _StopAfterOmniConfigError(Exception):
    """Abort Omni init once the VllmOmniConfig is built (consumers unplumbed)."""


@contextmanager
def built_omni_config(model: str, **kwargs):
    """Route Omni() through the new create_from_model path & yield the VllmOmniConfig.

    This should be used for testing the initialization path to create the VllmOmniConfig
    and make assertions about the initialized state. Note that we currently abort after
    we create the config, since it hasn't been wired through the stage creation yet.
    """
    captured: dict = {}

    def _build_omni_config(m, *, trust_remote_code, cli_overrides, deploy_config_path, strategy_specs=None):
        captured["config"] = StageConfigFactory.create_from_model(
            m,
            trust_remote_code=trust_remote_code,
            cli_overrides=cli_overrides,
            deploy_config_path=deploy_config_path,
        )
        raise _StopAfterOmniConfigError

    with (
        mock.patch.dict(OMNI_PIPELINES, {"llama": _LLM_PIPELINE}),
        mock.patch.object(
            StageConfigFactory,
            "create_legacy_stage_configs_from_model",
            side_effect=_build_omni_config,
        ),
    ):
        try:
            Omni(model=model, **kwargs)
        except _StopAfterOmniConfigError:
            # For now, we blow up after we create the omni model config, since it's
            # not pushed through the rest of initialization quite yet.
            pass
    yield captured["config"]


class TestOmniConfigQuantization:
    """Test initialization of quantization configs through the VllmOmniConfig path."""

    def test_llm_cli_quantization_is_preformed_fp8(self, llm_model_dir):
        with built_omni_config(llm_model_dir, quantization="fp8") as cfg:
            qc = cfg.stage_configs[0].quantization_config
            assert isinstance(qc, QuantizationConfig)
            assert qc.get_name() == "fp8"
            assert qc.is_checkpoint_fp8_serialized is False

    def test_llm_serialized_checkpoint_is_preformed_fp8(self, tmp_path):
        model = _write_llm_model_dir(tmp_path, quantization_config=_SERIALIZED_FP8)
        with built_omni_config(model) as cfg:
            qc = cfg.stage_configs[0].quantization_config
            assert isinstance(qc, QuantizationConfig)
            assert qc.get_name() == "fp8"
            assert qc.is_checkpoint_fp8_serialized is True

    def test_diffusion_cli_quantization_is_preformed_fp8(self, diffusion_model_dir):
        with built_omni_config(diffusion_model_dir, quantization="fp8") as cfg:
            qc = cfg.stage_configs[0].quantization_config
            assert isinstance(qc, QuantizationConfig)
            assert qc.get_name() == "fp8"
            assert qc.is_checkpoint_fp8_serialized is False

    def test_diffusion_serialized_checkpoint_is_preformed_fp8(self, tmp_path):
        model = _write_diffusion_model_dir(tmp_path, quantization_config=_SERIALIZED_FP8)
        with built_omni_config(model) as cfg:
            qc = cfg.stage_configs[0].quantization_config
            assert isinstance(qc, QuantizationConfig)
            assert qc.get_name() == "fp8"
            assert qc.is_checkpoint_fp8_serialized is True

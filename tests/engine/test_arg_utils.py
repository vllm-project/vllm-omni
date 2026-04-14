"""
Tests for Omni config utils. For stability, these tests should largely be
invariant to the specific attributes of vLLM config except in cases where we
explicitly patch values that differ from vLLM.
"""

import argparse
import inspect
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from transformers import PretrainedConfig
from vllm.engine.arg_utils import EngineArgs

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_sync_config_is_omni():
    """Ensure create_model_config gives the right type."""
    cfg = OmniEngineArgs().create_model_config()
    assert isinstance(cfg, OmniModelConfig)


def test_default_stage_id_is_concrete_int():
    """Ensure `stage_id` stays safe for downstream arithmetic/indexing."""
    engine_args = OmniEngineArgs()

    assert engine_args.stage_id == 0
    assert isinstance(engine_args.stage_id, int)
    assert engine_args.log_stats is False

    cfg = engine_args.create_model_config()
    assert cfg.stage_id == 0


def test_multimodal_kwarg_overrides():
    """Ensure that overrides in the multimodal config are preserved."""
    # Get a different value than the default for a multimodal field
    sig = inspect.signature(OmniEngineArgs)
    default_mm_cache = sig.parameters["mm_processor_cache_gb"].default
    override_val = default_mm_cache + 1

    # NOTE: This needs to be a model that resolves to supports_multimodal=True
    # in vLLM, otherwise we won't have an MM config
    cfg = OmniEngineArgs(
        model="Qwen/Qwen2-VL-2B-Instruct",
        mm_processor_cache_gb=override_val,
    ).create_model_config()

    # Ensure that the override was applied correctly
    assert cfg.multimodal_config is not None
    assert cfg.multimodal_config.mm_processor_cache_gb == override_val


def test_from_vllm_config_validates_invalid_omni_kwargs():
    """Ensure omni-specific field validation catches invalid keys."""
    model_config = EngineArgs().create_model_config()
    with pytest.raises(ValueError, match="Unexpected omni kwarg"):
        OmniModelConfig.from_vllm_model_config(model_config, foo="bar")


def test_from_vllm_config_validates_bad_omni_kwarg_types():
    """Ensure omni-specific field validation catches type errors."""
    model_config = EngineArgs().create_model_config()
    with pytest.raises(ValidationError):
        OmniModelConfig.from_vllm_model_config(model_config, stage_id="not_an_int")


def test_default_all_values_are_initialized():
    """Ensure omni-specific field initializes all fields"""
    model_config = EngineArgs().create_model_config()
    cfg = OmniModelConfig.from_vllm_model_config(model_config)

    # Test a primitive
    assert cfg.model_stage == "thinker"
    # Test a field initialized with a default factory
    assert cfg.stage_connector_config == {
        "name": "SharedMemoryConnector",
        "extra": {},
    }

    # Ensure that hf_config is initialized on model_config in the vLLM by ModelConfig's
    # __post_init__, and that the hf_config is copied over to the OmniModelConfig;
    # we explicitly set this since the field sets init=False
    assert isinstance(model_config.hf_config, PretrainedConfig)
    assert cfg.hf_config is model_config.hf_config

    # Ensure that we can convert it to a string; this will convert
    # all attributes, so should raise if we have attributes that are
    # not initialized correctly, e.g., due to default factories
    str(cfg)


def test_qwen3_tts_codec_frame_rate_patching():
    """Ensure the patch for qwen3 tts is applied correctly when creating the omni config."""
    # Create a vLLM ModelConfig
    vllm_config = EngineArgs().create_model_config()

    # Create a mock talking config with a dummy value for position_id_per_seconds
    mock_talker_config = SimpleNamespace()
    mock_talker_config.position_id_per_seconds = 12.3
    vllm_config.hf_config.talker_config = mock_talker_config

    # Ensure creating the config for a Qwen3TTSTalkerForConditionalGenerationARVLLM
    # model calls the patch func to apply position_id_per_seconds from the talker
    # config to the config's codec_frame_rate_hz
    omni_config = OmniModelConfig.from_vllm_model_config(
        vllm_config,
        model_arch="Qwen3TTSTalkerForConditionalGenerationARVLLM",
    )

    # Verify codec_frame_rate_hz was patched
    assert omni_config.codec_frame_rate_hz == 12.3


def test_from_cli_args_picks_up_stage_configs_path():
    """from_cli_args should pick up stage_configs_path from namespace."""
    ns = argparse.Namespace(
        model="facebook/opt-125m",
        stage_configs_path="/some/path.yaml",
        custom_pipeline_args=None,
    )

    args = OmniEngineArgs.from_cli_args(ns)
    assert args.stage_configs_path == "/some/path.yaml"
    assert args.custom_pipeline_args is None


def test_stage_specific_text_config_override():
    """Ensure dependent attributes are updated when using stage-specific config."""
    vllm_config = EngineArgs().create_model_config()
    vllm_config.disable_sliding_window = True

    # Switch the created hf text config with a mock whose
    # values we want to pull through the text config helper
    stage_text_config = vllm_config.hf_text_config
    vllm_config.hf_text_config = SimpleNamespace()
    stage_text_config.sliding_window = 4096
    stage_text_config.attention_chunk_size = 2048

    # Move the stage config's text config getter & thinker config
    mock_stage_config = SimpleNamespace(get_text_config=lambda: stage_text_config)
    vllm_config.hf_config.thinker_config = mock_stage_config

    # Ensure that create from a vLLM config correctly pulls the
    # expected values off of the thinker config & swaps the text config
    omni_config = OmniModelConfig.from_vllm_model_config(
        vllm_config,
        hf_config_name="thinker_config",
    )

    assert omni_config.hf_text_config is stage_text_config
    assert omni_config.attention_chunk_size == 2048
    assert omni_config.max_model_len == 4096
    assert omni_config.hf_text_config.sliding_window is None


def test_stage_configs_path_field():
    """OmniEngineArgs with stage_configs_path should construct without error."""
    args = OmniEngineArgs(stage_configs_path="/some/path.yaml")
    assert args.stage_configs_path == "/some/path.yaml"


def test_strip_single_engine_args():
    """_strip_single_engine_args should remove EngineArgs fields but keep omni fields."""
    kwargs = {
        # Parent EngineArgs fields — should be stripped
        "compilation_config": '{"cudagraph_mode": "FULL_AND_PIECEWISE"}',
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9,
        "model": "some/model",
        # Parent field that should be kept (allowlisted)
        "worker_extension_cls": "some.Extension",
        # OmniEngineArgs-only / non-engine fields — should pass through
        "stage_configs_path": "/path/to/yaml",
        "custom_pipeline_args": {"pipeline_class": "my.Pipeline"},
        "mode": "text-to-image",
        "lora_path": "/some/lora",
    }

    filtered = AsyncOmniEngine._strip_single_engine_args(kwargs)

    # Stripped — parent EngineArgs fields
    assert "compilation_config" not in filtered
    assert "tensor_parallel_size" not in filtered
    assert "gpu_memory_utilization" not in filtered
    assert "model" not in filtered

    # Stripped — orchestrator-level OmniEngineArgs field
    assert "stage_configs_path" not in filtered

    # Kept
    assert filtered["worker_extension_cls"] == "some.Extension"
    assert filtered["custom_pipeline_args"] == {"pipeline_class": "my.Pipeline"}
    assert filtered["mode"] == "text-to-image"
    assert filtered["lora_path"] == "/some/lora"


def test_strip_single_engine_args_model_does_not_trigger_warning(mocker):
    """model is always in kwargs (callers set it via from_cli_args/asdict),
    so it should not cause the override warning by itself or appear in it."""
    mock_warn = mocker.patch("vllm_omni.engine.async_omni_engine.logger.warning")

    # Typical caller kwargs: model is always present, no other parent
    # EngineArgs fields are explicitly overridden.
    AsyncOmniEngine._strip_single_engine_args(
        {
            "model": "some/model",
            "custom_pipeline_args": {"pipeline_class": "my.Pipeline"},
        }
    )
    mock_warn.assert_not_called()

    # When there *are* genuinely surprising overrides alongside model,
    # the warning should mention them but not model.
    AsyncOmniEngine._strip_single_engine_args(
        {
            "model": "some/model",
            "tensor_parallel_size": 4,
            "custom_pipeline_args": {"pipeline_class": "my.Pipeline"},
        }
    )
    mock_warn.assert_called_once()
    warned_args = mock_warn.call_args[0][-1]  # the formatted arg list
    assert "tensor_parallel_size" in warned_args
    assert "model" not in warned_args

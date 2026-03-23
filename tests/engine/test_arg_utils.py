"""
Tests for Omni config utils. For stability, these tests should largely be
invariant to the specific attributes of vLLM config except in cases where we
explicitly patch values that differ from vLLM.
"""

import inspect
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from transformers import PretrainedConfig
from vllm.engine.arg_utils import EngineArgs

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_sync_config_is_omni():
    """Ensure create_model_config gives the right type."""
    cfg = OmniEngineArgs().create_model_config()
    assert isinstance(cfg, OmniModelConfig)


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
    mock_talker_config = Mock()
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


def test_stage_specific_text_config_override():
    """Ensure dependent attributes are updated when using stage-specific config."""
    vllm_config = EngineArgs().create_model_config()
    vllm_config.disable_sliding_window = True

    # Switch the created hf text config with a mock whose
    # values we want to pull through the text config helper
    stage_text_config = vllm_config.hf_text_config
    vllm_config.hf_text_config = Mock()
    stage_text_config.sliding_window = 4096
    stage_text_config.attention_chunk_size = 2048

    # Move the stage config's text config getter & thinker config
    mock_stage_config = Mock()
    mock_stage_config.get_text_config.return_value = stage_text_config
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

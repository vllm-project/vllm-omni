# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for OmniOpenAIServingChat sampling params handling.

Tests that standard OpenAI API parameters (max_tokens, temperature, etc.)
are correctly applied to the thinker stage while preserving YAML defaults.
"""

import pytest
from unittest.mock import MagicMock

from vllm.sampling_params import SamplingParams


@pytest.fixture
def mock_thinker_stage():
    """Create a mock thinker stage with is_comprehension=True."""
    stage = MagicMock()
    stage.is_comprehension = True
    stage.model_stage = "thinker"
    return stage


@pytest.fixture
def mock_other_stage():
    """Create a mock non-thinker stage."""
    stage = MagicMock()
    stage.is_comprehension = False
    stage.model_stage = "talker"
    return stage


@pytest.fixture
def default_thinker_params():
    """Default sampling params for thinker stage (from YAML)."""
    return SamplingParams(
        temperature=0.4,
        top_p=0.9,
        top_k=1,
        max_tokens=2048,
        seed=42,
        repetition_penalty=1.05,
    )


@pytest.fixture
def default_other_params():
    """Default sampling params for non-thinker stage (from YAML)."""
    return SamplingParams(
        temperature=0.9,
        top_k=50,
        max_tokens=4096,
        seed=42,
    )


@pytest.fixture
def mock_engine_client(mock_thinker_stage, mock_other_stage,
                       default_thinker_params, default_other_params):
    """Create mock engine client with stage_list and default_sampling_params_list."""
    engine_client = MagicMock()
    engine_client.stage_list = [mock_thinker_stage, mock_other_stage]
    engine_client.default_sampling_params_list = [
        default_thinker_params,
        default_other_params,
    ]
    return engine_client


@pytest.fixture
def serving_chat(mock_engine_client):
    """Create OmniOpenAIServingChat instance with mocked dependencies."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    # Create instance without calling __init__
    instance = object.__new__(OmniOpenAIServingChat)
    instance.engine_client = mock_engine_client
    return instance


def test_preserves_yaml_defaults_when_no_request_params(serving_chat):
    """Test that YAML defaults are preserved when request has no params."""
    request = MagicMock()
    request.max_tokens = None
    request.temperature = None
    request.top_p = None
    request.top_k = None
    request.seed = None
    request.repetition_penalty = None

    def mock_to_sampling_params(max_tokens, logits_processor_pattern, default_sampling_params):
        return SamplingParams(
            temperature=default_sampling_params.get("temperature", 1.0),
            top_p=default_sampling_params.get("top_p", 1.0),
            top_k=default_sampling_params.get("top_k", 0),
            max_tokens=max_tokens,
            seed=default_sampling_params.get("seed"),
            repetition_penalty=default_sampling_params.get("repetition_penalty", 1.0),
        )

    request.to_sampling_params = mock_to_sampling_params

    result = serving_chat._build_sampling_params_list_from_request(request)

    assert len(result) == 2
    thinker_params = result[0]
    assert thinker_params.temperature == 0.4
    assert thinker_params.top_p == 0.9
    assert thinker_params.top_k == 1
    assert thinker_params.max_tokens == 2048
    assert thinker_params.seed == 42
    assert thinker_params.repetition_penalty == 1.05


def test_request_temperature_overrides_yaml_default(serving_chat):
    """Test that request temperature overrides YAML default."""
    request = MagicMock()
    request.max_tokens = None
    request.temperature = 0.8  # Override

    def mock_to_sampling_params(max_tokens, logits_processor_pattern, default_sampling_params):
        return SamplingParams(
            temperature=0.8,
            top_p=default_sampling_params.get("top_p", 1.0),
            max_tokens=max_tokens,
            seed=default_sampling_params.get("seed"),
        )

    request.to_sampling_params = mock_to_sampling_params

    result = serving_chat._build_sampling_params_list_from_request(request)

    thinker_params = result[0]
    assert thinker_params.temperature == 0.8
    assert thinker_params.seed == 42  # Preserved from YAML


def test_request_top_p_overrides_yaml_default(serving_chat):
    """Test that request top_p overrides YAML default."""
    request = MagicMock()
    request.max_tokens = None
    request.top_p = 0.95  # Override

    def mock_to_sampling_params(max_tokens, logits_processor_pattern, default_sampling_params):
        return SamplingParams(
            temperature=default_sampling_params.get("temperature", 1.0),
            top_p=0.95,
            max_tokens=max_tokens,
            seed=default_sampling_params.get("seed"),
        )

    request.to_sampling_params = mock_to_sampling_params

    result = serving_chat._build_sampling_params_list_from_request(request)

    thinker_params = result[0]
    assert thinker_params.top_p == 0.95
    assert thinker_params.temperature == 0.4  # Preserved from YAML


def test_request_max_tokens_overrides_yaml_default(serving_chat):
    """Test that request max_tokens overrides YAML default."""
    request = MagicMock()
    request.max_tokens = 100  # Override

    def mock_to_sampling_params(max_tokens, logits_processor_pattern, default_sampling_params):
        return SamplingParams(
            temperature=default_sampling_params.get("temperature", 1.0),
            max_tokens=max_tokens,
        )

    request.to_sampling_params = mock_to_sampling_params

    result = serving_chat._build_sampling_params_list_from_request(request)

    assert result[0].max_tokens == 100


def test_max_tokens_uses_yaml_default_when_not_specified(serving_chat):
    """Test that max_tokens falls back to YAML default when not in request."""
    request = MagicMock()
    request.max_tokens = None

    def mock_to_sampling_params(max_tokens, logits_processor_pattern, default_sampling_params):
        return SamplingParams(max_tokens=max_tokens)

    request.to_sampling_params = mock_to_sampling_params

    result = serving_chat._build_sampling_params_list_from_request(request)

    assert result[0].max_tokens == 2048


def test_non_thinker_stages_use_cloned_defaults(serving_chat):
    """Test that non-thinker stages always use cloned YAML defaults."""
    request = MagicMock()
    request.max_tokens = 50
    request.temperature = 0.1

    def mock_to_sampling_params(max_tokens, logits_processor_pattern, default_sampling_params):
        return SamplingParams(temperature=0.1, max_tokens=max_tokens)

    request.to_sampling_params = mock_to_sampling_params

    result = serving_chat._build_sampling_params_list_from_request(request)

    other_params = result[1]
    assert other_params.temperature == 0.9  # YAML default
    assert other_params.max_tokens == 4096  # YAML default
    assert other_params.top_k == 50  # YAML default
    assert other_params.seed == 42  # YAML default


def test_multiple_params_override_together(serving_chat):
    """Test that multiple request params can override together."""
    request = MagicMock()
    request.max_tokens = 200
    request.temperature = 0.7
    request.top_p = 0.85

    def mock_to_sampling_params(max_tokens, logits_processor_pattern, default_sampling_params):
        return SamplingParams(
            temperature=0.7,
            top_p=0.85,
            top_k=default_sampling_params.get("top_k", 0),
            max_tokens=max_tokens,
            seed=default_sampling_params.get("seed"),
            repetition_penalty=default_sampling_params.get("repetition_penalty", 1.0),
        )

    request.to_sampling_params = mock_to_sampling_params

    result = serving_chat._build_sampling_params_list_from_request(request)

    thinker_params = result[0]
    # Overridden
    assert thinker_params.temperature == 0.7
    assert thinker_params.top_p == 0.85
    assert thinker_params.max_tokens == 200
    # Preserved from YAML
    assert thinker_params.top_k == 1
    assert thinker_params.seed == 42
    assert thinker_params.repetition_penalty == 1.05


def test_get_thinker_stage_index_finds_first_stage(mock_engine_client):
    """Test finding thinker stage when it's at index 0."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)
    instance.engine_client = mock_engine_client

    assert instance._get_thinker_stage_index() == 0


def test_get_thinker_stage_index_finds_second_stage():
    """Test finding thinker stage when it's at index 1."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)

    other = MagicMock()
    other.is_comprehension = False
    thinker = MagicMock()
    thinker.is_comprehension = True

    instance.engine_client = MagicMock()
    instance.engine_client.stage_list = [other, thinker]

    assert instance._get_thinker_stage_index() == 1


def test_get_thinker_stage_index_raises_when_not_found():
    """Test that ValueError is raised when no thinker stage exists."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)

    stage1 = MagicMock()
    stage1.is_comprehension = False
    stage2 = MagicMock()
    stage2.is_comprehension = False

    instance.engine_client = MagicMock()
    instance.engine_client.stage_list = [stage1, stage2]

    with pytest.raises(ValueError, match="No thinker stage"):
        instance._get_thinker_stage_index()


def test_sampling_params_to_dict_converts_all_fields(serving_chat):
    """Test that _sampling_params_to_dict converts all fields."""
    params = SamplingParams(
        temperature=0.5,
        top_p=0.9,
        top_k=10,
        max_tokens=100,
        seed=123,
        repetition_penalty=1.1,
    )

    result = serving_chat._sampling_params_to_dict(params)

    assert isinstance(result, dict)
    assert result["temperature"] == 0.5
    assert result["top_p"] == 0.9
    assert result["top_k"] == 10
    assert result["max_tokens"] == 100
    assert result["seed"] == 123
    assert result["repetition_penalty"] == 1.1

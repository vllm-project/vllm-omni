# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for OmniOpenAIServingChat sampling params handling.

Tests that standard OpenAI API parameters (max_tokens, temperature, etc.)
are correctly applied to the comprehension stage while preserving YAML defaults.
"""

from unittest.mock import MagicMock

import pytest
from vllm.sampling_params import SamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def mock_comprehension_stage():
    """Create a mock comprehension stage with is_comprehension=True."""
    stage = MagicMock()
    stage.is_comprehension = True
    stage.model_stage = "comprehension"
    return stage


@pytest.fixture
def mock_other_stage():
    """Create a mock non-comprehension stage."""
    stage = MagicMock()
    stage.is_comprehension = False
    stage.model_stage = "other"
    return stage


@pytest.fixture
def default_comprehension_params():
    """Default sampling params for comprehension stage (from YAML)."""
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
    """Default sampling params for non-comprehension stage (from YAML)."""
    return SamplingParams(
        temperature=0.9,
        top_k=50,
        max_tokens=4096,
        seed=42,
    )


@pytest.fixture
def mock_engine_client(mock_comprehension_stage, mock_other_stage, default_comprehension_params, default_other_params):
    """Create mock engine client with stage_list and default_sampling_params_list."""
    engine_client = MagicMock()
    engine_client.stage_list = [mock_comprehension_stage, mock_other_stage]
    engine_client.default_sampling_params_list = [
        default_comprehension_params,
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


@pytest.fixture
def mock_request():
    """Create a mock request with all OpenAI sampling params set to None."""
    request = MagicMock()
    # OpenAI standard sampling fields
    request.temperature = None
    request.top_p = None
    request.top_k = None
    request.max_tokens = None
    request.min_tokens = None
    request.seed = None
    request.ignore_eos = None
    request.stop = None
    request.stop_token_ids = None
    request.frequency_penalty = None
    request.presence_penalty = None
    return request


# =============================================================================
# Tests for _OPENAI_SAMPLING_FIELDS constant
# =============================================================================


def test_openai_sampling_fields_contains_expected_fields():
    """Test that _OPENAI_SAMPLING_FIELDS contains all expected OpenAI params."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    expected_fields = {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "min_tokens",
        "seed",
        "ignore_eos",
        "stop",
        "stop_token_ids",
        "frequency_penalty",
        "presence_penalty",
    }
    assert OmniOpenAIServingChat._OPENAI_SAMPLING_FIELDS == expected_fields


# =============================================================================
# Tests for _build_sampling_params_list_from_request
# =============================================================================


def test_preserves_yaml_defaults_when_no_request_params(serving_chat, mock_request):
    """Test that YAML defaults are preserved when request has no params."""
    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    assert len(result) == 2
    comprehension_params = result[0]
    assert comprehension_params.temperature == 0.4
    assert comprehension_params.top_p == 0.9
    assert comprehension_params.top_k == 1  # YAML custom param preserved
    assert comprehension_params.max_tokens == 2048
    assert comprehension_params.seed == 42
    assert comprehension_params.repetition_penalty == 1.05  # YAML custom param preserved


def test_request_temperature_overrides_yaml_default(serving_chat, mock_request):
    """Test that request temperature overrides YAML default."""
    mock_request.temperature = 0.8

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    comprehension_params = result[0]
    assert comprehension_params.temperature == 0.8  # Overridden
    assert comprehension_params.seed == 42  # Preserved from YAML
    assert comprehension_params.top_k == 1  # YAML custom param preserved


def test_request_top_p_overrides_yaml_default(serving_chat, mock_request):
    """Test that request top_p overrides YAML default."""
    mock_request.top_p = 0.95

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    comprehension_params = result[0]
    assert comprehension_params.top_p == 0.95  # Overridden
    assert comprehension_params.temperature == 0.4  # Preserved from YAML


def test_request_max_tokens_overrides_yaml_default(serving_chat, mock_request):
    """Test that request max_tokens overrides YAML default."""
    mock_request.max_tokens = 100

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    assert result[0].max_tokens == 100


def test_max_tokens_uses_yaml_default_when_not_specified(serving_chat, mock_request):
    """Test that max_tokens falls back to YAML default when not in request."""
    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    assert result[0].max_tokens == 2048


def test_request_seed_overrides_yaml_default(serving_chat, mock_request):
    """Test that request seed overrides YAML default."""
    mock_request.seed = 123

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    comprehension_params = result[0]
    assert comprehension_params.seed == 123  # Overridden
    assert comprehension_params.temperature == 0.4  # Preserved from YAML


def test_request_frequency_penalty_overrides(serving_chat, mock_request):
    """Test that request frequency_penalty is applied."""
    mock_request.frequency_penalty = 0.5

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    assert result[0].frequency_penalty == 0.5


def test_request_presence_penalty_overrides(serving_chat, mock_request):
    """Test that request presence_penalty is applied."""
    mock_request.presence_penalty = 0.3

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    assert result[0].presence_penalty == 0.3


def test_non_comprehension_stages_use_cloned_defaults(serving_chat, mock_request):
    """Test that non-comprehension stages always use cloned YAML defaults."""
    mock_request.max_tokens = 50
    mock_request.temperature = 0.1

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    other_params = result[1]
    assert other_params.temperature == 0.9  # YAML default (not affected by request)
    assert other_params.max_tokens == 4096  # YAML default (not affected by request)
    assert other_params.top_k == 50  # YAML default
    assert other_params.seed == 42  # YAML default


def test_multiple_params_override_together(serving_chat, mock_request):
    """Test that multiple request params can override together."""
    mock_request.max_tokens = 200
    mock_request.temperature = 0.7
    mock_request.top_p = 0.85
    mock_request.seed = 999

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    comprehension_params = result[0]
    # Overridden by request
    assert comprehension_params.temperature == 0.7
    assert comprehension_params.top_p == 0.85
    assert comprehension_params.max_tokens == 200
    assert comprehension_params.seed == 999
    # Preserved from YAML (not in _OPENAI_SAMPLING_FIELDS)
    assert comprehension_params.top_k == 1
    assert comprehension_params.repetition_penalty == 1.05


# =============================================================================
# Tests for _apply_request_overrides
# =============================================================================


def test_apply_request_overrides_clones_params(serving_chat, mock_request, default_comprehension_params):
    """Test that _apply_request_overrides returns a cloned object."""
    result = serving_chat._apply_request_overrides(default_comprehension_params, mock_request)

    assert result is not default_comprehension_params  # Different object


def test_apply_request_overrides_preserves_defaults(serving_chat, mock_request, default_comprehension_params):
    """Test that _apply_request_overrides preserves defaults when request has None."""
    result = serving_chat._apply_request_overrides(default_comprehension_params, mock_request)

    assert result.temperature == 0.4
    assert result.top_p == 0.9
    assert result.seed == 42
    assert result.top_k == 1  # YAML custom param


def test_apply_request_overrides_applies_values(serving_chat, mock_request, default_comprehension_params):
    """Test that _apply_request_overrides applies non-None request values."""
    mock_request.temperature = 0.8
    mock_request.seed = 123

    result = serving_chat._apply_request_overrides(default_comprehension_params, mock_request)

    assert result.temperature == 0.8  # Overridden
    assert result.seed == 123  # Overridden
    assert result.top_p == 0.9  # Preserved from default
    assert result.top_k == 1  # YAML custom param preserved


# =============================================================================
# Tests for _get_comprehension_stage_index
# =============================================================================


def test_get_comprehension_stage_index_finds_first_stage(mock_engine_client):
    """Test finding comprehension stage when it's at index 0."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)
    instance.engine_client = mock_engine_client

    assert instance._get_comprehension_stage_index() == 0


def test_get_comprehension_stage_index_finds_second_stage():
    """Test finding comprehension stage when it's at index 1."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)

    other = MagicMock()
    other.is_comprehension = False
    comprehension = MagicMock()
    comprehension.is_comprehension = True

    instance.engine_client = MagicMock()
    instance.engine_client.stage_list = [other, comprehension]

    assert instance._get_comprehension_stage_index() == 1


def test_get_comprehension_stage_index_raises_when_not_found():
    """Test that ValueError is raised when no comprehension stage exists."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)

    stage1 = MagicMock()
    stage1.is_comprehension = False
    stage2 = MagicMock()
    stage2.is_comprehension = False

    instance.engine_client = MagicMock()
    instance.engine_client.stage_list = [stage1, stage2]

    with pytest.raises(ValueError, match="No comprehension stage"):
        instance._get_comprehension_stage_index()

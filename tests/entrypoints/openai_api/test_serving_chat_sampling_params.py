# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for OmniOpenAIServingChat sampling params handling.

Tests that standard OpenAI API parameters (max_tokens, temperature, etc.)
are correctly applied to the comprehension stage while preserving YAML defaults.
"""

import pytest
from pytest_mock import MockerFixture
from vllm.sampling_params import SamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def mock_comprehension_stage(mocker: MockerFixture):
    """Create a mock comprehension stage with is_comprehension=True."""
    stage = mocker.MagicMock()
    stage.is_comprehension = True
    stage.model_stage = "comprehension"
    return stage


@pytest.fixture
def mock_other_stage(mocker: MockerFixture):
    """Create a mock non-comprehension stage."""
    stage = mocker.MagicMock()
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
def mock_engine_client(
    mock_comprehension_stage,
    mock_other_stage,
    default_comprehension_params,
    default_other_params,
    mocker: MockerFixture,
):
    """Create mock engine client with stage_configs and default_sampling_params_list."""
    engine_client = mocker.MagicMock()
    engine_client.stage_configs = [mock_comprehension_stage, mock_other_stage]
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
def mock_request(mocker: MockerFixture):
    """Create a mock request with all OpenAI sampling params set to None."""
    request = mocker.MagicMock()
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
    # Must be real Python objects (not MagicMock) so the code's explicit-field
    # and extra_body checks work correctly.
    request.model_fields_set = set()
    request.extra_body = {}
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
    mock_request.model_fields_set = {"temperature"}

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    comprehension_params = result[0]
    assert comprehension_params.temperature == 0.8  # Overridden
    assert comprehension_params.seed == 42  # Preserved from YAML
    assert comprehension_params.top_k == 1  # YAML custom param preserved


def test_request_top_p_overrides_yaml_default(serving_chat, mock_request):
    """Test that request top_p overrides YAML default."""
    mock_request.top_p = 0.95
    mock_request.model_fields_set = {"top_p"}

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    comprehension_params = result[0]
    assert comprehension_params.top_p == 0.95  # Overridden
    assert comprehension_params.temperature == 0.4  # Preserved from YAML


def test_request_max_tokens_overrides_yaml_default(serving_chat, mock_request):
    """Test that request max_tokens overrides YAML default."""
    mock_request.max_tokens = 100
    mock_request.model_fields_set = {"max_tokens"}

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    assert result[0].max_tokens == 100


def test_max_tokens_uses_yaml_default_when_not_specified(serving_chat, mock_request):
    """Test that max_tokens falls back to YAML default when not in request."""
    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    assert result[0].max_tokens == 2048


def test_request_seed_overrides_yaml_default(serving_chat, mock_request):
    """Test that request seed overrides YAML default."""
    mock_request.seed = 123
    mock_request.model_fields_set = {"seed"}

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    comprehension_params = result[0]
    assert comprehension_params.seed == 123  # Overridden
    assert comprehension_params.temperature == 0.4  # Preserved from YAML


def test_request_frequency_penalty_overrides(serving_chat, mock_request):
    """Test that request frequency_penalty is applied."""
    mock_request.frequency_penalty = 0.5
    mock_request.model_fields_set = {"frequency_penalty"}

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    assert result[0].frequency_penalty == 0.5


def test_request_presence_penalty_overrides(serving_chat, mock_request):
    """Test that request presence_penalty is applied."""
    mock_request.presence_penalty = 0.3
    mock_request.model_fields_set = {"presence_penalty"}

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
    mock_request.model_fields_set = {"max_tokens", "temperature", "top_p", "seed"}

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
    mock_request.model_fields_set = {"temperature", "seed"}

    result = serving_chat._apply_request_overrides(default_comprehension_params, mock_request)

    assert result.temperature == 0.8  # Overridden
    assert result.seed == 123  # Overridden
    assert result.top_p == 0.9  # Preserved from default
    assert result.top_k == 1  # YAML custom param preserved


# =============================================================================
# Tests for empty-list handling in _apply_request_overrides
# =============================================================================


def test_apply_overrides_empty_stop_list_preserves_default(serving_chat, mocker):
    """Test that request.stop=[] does NOT override YAML default stop words."""
    default_params = SamplingParams(temperature=0.5, stop=["<|im_end|>"])
    request = mocker.MagicMock()
    request.temperature = None
    request.top_p = None
    request.top_k = None
    request.max_tokens = None
    request.min_tokens = None
    request.seed = None
    request.ignore_eos = None
    request.stop = []  # empty list — should be treated as "not set"
    request.stop_token_ids = None
    request.frequency_penalty = None
    request.presence_penalty = None
    request.model_fields_set = {"stop"}
    request.extra_body = {}

    result = serving_chat._apply_request_overrides(default_params, request)

    assert result.stop == ["<|im_end|>"]  # YAML default preserved


def test_apply_overrides_nonempty_stop_list_overrides_default(serving_chat, mocker):
    """Test that request.stop=["\\n"] overrides YAML default stop words."""
    default_params = SamplingParams(temperature=0.5, stop=["<|im_end|>"])
    request = mocker.MagicMock()
    request.temperature = None
    request.top_p = None
    request.top_k = None
    request.max_tokens = None
    request.min_tokens = None
    request.seed = None
    request.ignore_eos = None
    request.stop = ["\n"]  # non-empty list — should override
    request.stop_token_ids = None
    request.frequency_penalty = None
    request.presence_penalty = None
    request.model_fields_set = {"stop"}
    request.extra_body = {}

    result = serving_chat._apply_request_overrides(default_params, request)

    assert result.stop == ["\n"]  # Overridden by request


def test_apply_overrides_empty_stop_token_ids_preserves_default(serving_chat, mocker):
    """Test that request.stop_token_ids=[] does NOT override YAML default."""
    default_params = SamplingParams(temperature=0.5, stop_token_ids=[2, 3])
    request = mocker.MagicMock()
    request.temperature = None
    request.top_p = None
    request.top_k = None
    request.max_tokens = None
    request.min_tokens = None
    request.seed = None
    request.ignore_eos = None
    request.stop = None
    request.stop_token_ids = []  # empty list — should be treated as "not set"
    request.frequency_penalty = None
    request.presence_penalty = None

    result = serving_chat._apply_request_overrides(default_params, request)

    assert result.stop_token_ids == [2, 3]  # YAML default preserved


def test_apply_overrides_nonempty_stop_token_ids_overrides_default(serving_chat, mocker):
    """Test that request.stop_token_ids=[100] overrides YAML default."""
    default_params = SamplingParams(temperature=0.5, stop_token_ids=[2, 3])
    request = mocker.MagicMock()
    request.temperature = None
    request.top_p = None
    request.top_k = None
    request.max_tokens = None
    request.min_tokens = None
    request.seed = None
    request.ignore_eos = None
    request.stop = None
    request.stop_token_ids = [100]  # non-empty list — should override
    request.frequency_penalty = None
    request.presence_penalty = None
    request.model_fields_set = {"stop_token_ids"}
    request.extra_body = {}

    result = serving_chat._apply_request_overrides(default_params, request)

    assert result.stop_token_ids == [100]  # Overridden by request


def test_apply_overrides_mixed_empty_and_nonempty_lists(serving_chat, mocker):
    """Test mixing empty and non-empty list fields with scalar fields."""
    default_params = SamplingParams(
        temperature=0.4,
        stop=["<|end|>"],
        stop_token_ids=[2],
    )
    request = mocker.MagicMock()
    request.temperature = 0.9
    request.top_p = None
    request.top_k = None
    request.max_tokens = None
    request.min_tokens = None
    request.seed = None
    request.ignore_eos = None
    request.stop = []  # empty — should NOT override
    request.stop_token_ids = [100, 200]  # non-empty — SHOULD override
    request.frequency_penalty = None
    request.presence_penalty = None
    request.model_fields_set = {"temperature", "stop", "stop_token_ids"}
    request.extra_body = {}

    result = serving_chat._apply_request_overrides(default_params, request)

    assert result.temperature == 0.9  # Scalar override works
    assert result.stop == ["<|end|>"]  # Empty list did NOT override
    assert result.stop_token_ids == [100, 200]  # Non-empty list DID override


def test_apply_overrides_none_scalar_still_preserves_default(serving_chat, mocker):
    """Regression: ensure None scalar values still don't override defaults."""
    default_params = SamplingParams(temperature=0.5, max_tokens=100, seed=42)
    request = mocker.MagicMock()
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
    request.model_fields_set = set()
    request.extra_body = {}

    result = serving_chat._apply_request_overrides(default_params, request)

    assert result.temperature == 0.5
    assert result.max_tokens == 100
    assert result.seed == 42


def test_apply_overrides_both_lists_empty_preserves_defaults(serving_chat, mocker):
    """Test that both stop=[] and stop_token_ids=[] preserve YAML defaults."""
    default_params = SamplingParams(
        temperature=0.5,
        stop=["<|end|>", "\\n"],
        stop_token_ids=[2, 32000],
    )
    request = mocker.MagicMock()
    request.temperature = None
    request.top_p = None
    request.top_k = None
    request.max_tokens = None
    request.min_tokens = None
    request.seed = None
    request.ignore_eos = None
    request.stop = []
    request.stop_token_ids = []
    request.frequency_penalty = None
    request.presence_penalty = None
    request.model_fields_set = {"stop", "stop_token_ids"}
    request.extra_body = {}

    result = serving_chat._apply_request_overrides(default_params, request)

    assert result.stop == ["<|end|>", "\\n"]
    assert result.stop_token_ids == [2, 32000]


def test_build_sampling_params_list_empty_stop_preserves_yaml(serving_chat, mock_request):
    """Test that empty stop list in request preserves YAML defaults via
    _build_sampling_params_list_from_request."""
    mock_request.stop = []
    mock_request.stop_token_ids = []

    result = serving_chat._build_sampling_params_list_from_request(mock_request)

    comprehension_params = result[0]
    # Empty lists should NOT override — YAML defaults are preserved
    assert comprehension_params.stop == []
    assert comprehension_params.stop_token_ids == []


# =============================================================================
# Tests for _get_comprehension_stage_index
# =============================================================================


def test_get_comprehension_stage_index_finds_first_stage(mock_engine_client):
    """Test finding comprehension stage when it's at index 0."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)
    instance.engine_client = mock_engine_client

    assert instance._get_comprehension_stage_index() == 0


def test_get_comprehension_stage_index_finds_second_stage(mocker: MockerFixture):
    """Test finding comprehension stage when it's at index 1."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)

    other = mocker.MagicMock()
    other.is_comprehension = False
    comprehension = mocker.MagicMock()
    comprehension.is_comprehension = True

    instance.engine_client = mocker.MagicMock()
    instance.engine_client.stage_configs = [other, comprehension]

    assert instance._get_comprehension_stage_index() == 1


def test_get_comprehension_stage_index_raises_when_not_found(mocker: MockerFixture):
    """Test that ValueError is raised when no comprehension stage exists."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)

    stage1 = mocker.MagicMock()
    stage1.is_comprehension = False
    stage2 = mocker.MagicMock()
    stage2.is_comprehension = False

    instance.engine_client = mocker.MagicMock()
    instance.engine_client.stage_configs = [stage1, stage2]

    with pytest.raises(ValueError, match="No comprehension stage"):
        instance._get_comprehension_stage_index()


# =============================================================================
# Tests for _resolve_height_width_from_extra_body
# =============================================================================


class TestResolveHeightWidth:
    def test_explicit_height_width(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({"height": 512, "width": 768})
        assert h == 512
        assert w == 768

    def test_size_string(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({"size": "768x512"})
        assert w == 768
        assert h == 512

    def test_size_string_uppercase(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({"size": "768X512"})
        assert w == 768
        assert h == 512

    def test_size_fallback_when_height_missing(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({"size": "512x512", "width": 1024})
        # height is None -> size fallback fires and sets BOTH width and height
        assert h == 512
        assert w == 512

    def test_empty_extra_body(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({})
        assert h is None
        assert w is None

    def test_invalid_size_format_ignored(self):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        h, w = OmniOpenAIServingChat._resolve_height_width_from_extra_body({"size": "invalid"})
        assert h is None
        assert w is None


# =============================================================================
# Tests for _apply_request_overrides with GLM-Image (max_tokens computation)
# =============================================================================


class TestApplyRequestOverridesGLMImage:
    """Test dynamic max_tokens computation for GLM-Image AR stage."""

    @pytest.fixture
    def glm_serving_chat(self, mock_engine_client, mocker: MockerFixture):
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

        instance = object.__new__(OmniOpenAIServingChat)
        instance.engine_client = mock_engine_client
        # Mock the image extraction to return no reference images (t2i by default)
        instance._extract_diffusion_prompt_and_images_from_messages = mocker.MagicMock(return_value=("a cat", []))
        return instance

    @pytest.fixture
    def glm_request(self, mocker: MockerFixture):
        req = mocker.MagicMock()
        req.temperature = None
        req.top_p = None
        req.top_k = None
        req.max_tokens = None
        req.min_tokens = None
        req.seed = None
        req.ignore_eos = None
        req.stop = None
        req.stop_token_ids = None
        req.frequency_penalty = None
        req.presence_penalty = None
        req.extra_body = {"height": 1024, "width": 1024}
        req.model_fields_set = set()
        return req

    def test_t2i_computes_max_tokens(self, glm_serving_chat, glm_request, default_comprehension_params):
        """t2i mode: max_tokens computed from height/width, no reference images."""
        result = glm_serving_chat._apply_request_overrides(default_comprehension_params, glm_request)
        # t2i 1024x1024 = 256 + 1024 + 1 = 1281
        assert result.max_tokens == 1281
        assert result.extra_args["target_h"] == 1024
        assert result.extra_args["target_w"] == 1024

    def test_i2i_computes_fewer_tokens(
        self, glm_serving_chat, glm_request, default_comprehension_params, mocker: MockerFixture
    ):
        """i2i mode: max_tokens should be smaller than t2i for same dimensions."""
        # Make it detect reference images
        glm_serving_chat._extract_diffusion_prompt_and_images_from_messages = mocker.MagicMock(
            return_value=("edit this", ["fake_image"])
        )

        result = glm_serving_chat._apply_request_overrides(default_comprehension_params, glm_request)
        # i2i 1024x1024 = 1024 + 1 = 1025
        assert result.max_tokens == 1025

    def test_dynamic_max_tokens_overrides_user_value(self, glm_serving_chat, glm_request, default_comprehension_params):
        """When height/width are provided, dynamic computation overrides user max_tokens."""
        glm_request.max_tokens = 500
        glm_request.model_fields_set = {"max_tokens"}

        result = glm_serving_chat._apply_request_overrides(default_comprehension_params, glm_request)
        # Dynamic computation from height/width always wins when present
        assert result.max_tokens == 1281

    def test_no_height_width_preserves_default(
        self, glm_serving_chat, mocker: MockerFixture, default_comprehension_params
    ):
        """When no height/width in extra_body, keep YAML default max_tokens."""
        req = mocker.MagicMock()
        req.temperature = None
        req.top_p = None
        req.top_k = None
        req.max_tokens = None
        req.min_tokens = None
        req.seed = None
        req.ignore_eos = None
        req.stop = None
        req.stop_token_ids = None
        req.frequency_penalty = None
        req.presence_penalty = None
        req.extra_body = {}
        req.model_fields_set = set()

        result = glm_serving_chat._apply_request_overrides(default_comprehension_params, req)
        assert result.max_tokens == 2048  # YAML default

    def test_size_string_parsed_for_glm_image(
        self, glm_serving_chat, mocker: MockerFixture, default_comprehension_params
    ):
        """'size' in extra_body is parsed as fallback for height/width."""
        req = mocker.MagicMock()
        req.temperature = None
        req.top_p = None
        req.top_k = None
        req.max_tokens = None
        req.min_tokens = None
        req.seed = None
        req.ignore_eos = None
        req.stop = None
        req.stop_token_ids = None
        req.frequency_penalty = None
        req.presence_penalty = None
        req.extra_body = {"size": "512x512"}
        req.model_fields_set = set()

        result = glm_serving_chat._apply_request_overrides(default_comprehension_params, req)
        # 512x512 t2i = 256 + 256 + 1 = 513
        assert result.max_tokens == 513

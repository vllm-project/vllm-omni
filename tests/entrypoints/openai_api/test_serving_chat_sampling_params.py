# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for OmniOpenAIServingChat sampling params handling.

Tests that standard OpenAI API parameters (max_tokens, temperature, etc.)
are correctly applied to the comprehension stage while preserving YAML defaults.
"""

import asyncio
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
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
        max_tokens=4353,
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


def test_serving_boundary_normalizes_declared_root_and_nested_extras(serving_chat):
    serving_chat._diffusion_extra_body_params = frozenset({"cfg_text_scale", "negative_prompt"})
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        modalities=["image"],
        num_inference_steps=7,
        size="768x512",
        negative_prompt="avoid blur",
        lora={"name": "adapter"},
        cfg_text_scale=7.0,
        extra_body={"extra_args": {"sample_solver": "euler"}},
    )

    normalized_extra_args, diffusion_request_args = serving_chat._normalize_diffusion_request_args(request)

    assert normalized_extra_args == {
        "cfg_text_scale": 7.0,
        "sample_solver": "euler",
    }
    assert diffusion_request_args["cfg_text_scale"] == 7.0
    assert diffusion_request_args["num_inference_steps"] == 7
    assert diffusion_request_args["size"] == "768x512"
    assert diffusion_request_args["negative_prompt"] == "avoid blur"
    assert diffusion_request_args["lora"] == {"name": "adapter"}
    assert diffusion_request_args["modalities"] == ["image"]


def test_unknown_root_extra_does_not_claim_canonical_extra(serving_chat):
    serving_chat._diffusion_extra_body_params = frozenset()
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        pipeline_option="ignored-root-value",
        extra_args={"pipeline_option": "canonical-value"},
    )

    normalized_extra_args, diffusion_request_args = serving_chat._normalize_diffusion_request_args(request)

    assert normalized_extra_args == {
        "pipeline_option": "canonical-value",
    }
    assert "pipeline_option" not in diffusion_request_args


def test_unregistered_cfg_scale_aliases_common_true_cfg_scale(serving_chat):
    serving_chat._diffusion_extra_body_params = frozenset()
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        cfg_scale=7.0,
    )

    normalized_extra_args, diffusion_request_args = serving_chat._normalize_diffusion_request_args(request)

    assert normalized_extra_args == {}
    assert diffusion_request_args == {"true_cfg_scale": 7.0}


@pytest.mark.parametrize(
    ("registered", "request_kwargs"),
    [
        ({"cfg_scale"}, {"cfg_scale": 7.0, "extra_args": {"cfg_scale": 8.0}}),
        (set(), {"cfg_scale": 7.0, "true_cfg_scale": 2.0}),
    ],
    ids=["registered-model-extra", "unregistered-common-alias"],
)
def test_cfg_scale_owner_conflicts_are_rejected(serving_chat, registered, request_kwargs):
    serving_chat._diffusion_extra_body_params = frozenset(registered)
    request = ChatCompletionRequest(model="test", messages=[], **request_kwargs)

    with pytest.raises(ValueError, match="provided more than once"):
        serving_chat._normalize_diffusion_request_args(request)


@pytest.mark.parametrize("diffusion_mode", [True, False], ids=["pure", "mixed"])
def test_duplicate_extras_return_the_same_bad_request_before_dispatch(
    serving_chat,
    mocker: MockerFixture,
    diffusion_mode: bool,
):
    serving_chat._diffusion_mode = diffusion_mode
    serving_chat._diffusion_extra_body_params = frozenset({"sample_solver"})
    serving_chat.engine_client.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    check_model = mocker.patch.object(serving_chat, "_check_model", new=mocker.AsyncMock())
    pure_dispatch = mocker.patch.object(
        serving_chat,
        "_create_diffusion_chat_completion",
        new=mocker.AsyncMock(),
    )
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        sample_solver="euler",
        extra_args={"sample_solver": "ddim"},
    )

    response = asyncio.run(serving_chat._create_chat_completion(request))

    assert response.error.code == 400
    assert response.error.message == (
        'Diffusion request parameters were provided more than once: "sample_solver": '
        "request.sample_solver, request.extra_args.sample_solver."
    )
    check_model.assert_not_awaited()
    pure_dispatch.assert_not_awaited()


def test_pure_consumer_preserves_defaults_and_separate_cfg_owners(serving_chat, mocker: MockerFixture):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    captured: dict[str, object] = {}

    async def generate(**kwargs):
        captured.update(kwargs)
        if False:
            yield None

    serving_chat._diffusion_model_name = "test"
    serving_chat._diffusion_engine = SimpleNamespace(
        stage_configs=[SimpleNamespace(stage_type="diffusion")],
        default_sampling_params_list=[
            OmniDiffusionSamplingParams(extra_args={"solver": "euler", "stage_default": True}),
        ],
        generate=generate,
    )
    serving_chat._diffusion_mode = True
    serving_chat._diffusion_extra_body_params = frozenset({"cfg_scale"})
    serving_chat._extract_diffusion_prompt_and_media = mocker.Mock(return_value=("prompt", [], [], []))
    request = ChatCompletionRequest(
        model="test",
        messages=[],
        modalities=["image"],
        num_inference_steps=7,
        size="768x512",
        negative_prompt="avoid blur",
        cfg_scale=7.0,
        true_cfg_scale=2.0,
        extra_body={"extra_args": {"solver": "ddim"}},
    )

    response = asyncio.run(serving_chat._create_chat_completion(request))

    (sampling_params,) = captured["sampling_params_list"]
    assert sampling_params.extra_args == {
        "cfg_scale": 7.0,
        "solver": "ddim",
        "stage_default": True,
    }
    assert sampling_params.true_cfg_scale == 2.0
    assert sampling_params.num_inference_steps == 7
    assert (sampling_params.height, sampling_params.width) == (512, 768)
    assert captured["prompt"]["negative_prompt"] == "avoid blur"
    assert response.error.message == "No output generated from AsyncOmni"


def test_mixed_consumer_keeps_root_common_args_with_nested_extras(serving_chat, mocker: MockerFixture):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    serving_chat.engine_client.stage_configs = [
        SimpleNamespace(stage_type="llm", is_comprehension=True),
        SimpleNamespace(stage_type="diffusion", is_comprehension=False),
    ]
    serving_chat.engine_client.default_sampling_params_list = [
        SamplingParams(),
        OmniDiffusionSamplingParams(),
    ]
    serving_chat.engine_client.output_modalities = ["image"]
    serving_chat.engine_client.errored = False

    captured: dict[str, object] = {}

    async def results():
        if False:
            yield None

    def generate(**kwargs):
        captured.update(kwargs)
        return results()

    serving_chat.engine_client.generate = generate
    serving_chat.__dict__.update(
        _diffusion_mode=False,
        _diffusion_extra_body_params=frozenset(),
        models=SimpleNamespace(model_name=lambda _: "test"),
        renderer=SimpleNamespace(get_tokenizer=lambda: object()),
        online_renderer=SimpleNamespace(validate_chat_template=lambda **_: None),
        parser_cls=None,
        use_harmony=False,
        enable_auto_tools=False,
        exclude_tools_when_tool_choice_none=False,
        trust_request_chat_template=True,
        chat_template=None,
        chat_template_content_format="auto",
    )
    mocker.patch.multiple(
        serving_chat,
        _check_model=mocker.AsyncMock(return_value=None),
        _maybe_get_adapters=mocker.Mock(return_value=None),
        _effective_chat_template_kwargs=mocker.Mock(return_value={}),
        _preprocess_chat=mocker.AsyncMock(return_value=([], [{"prompt": "raw"}])),
        _base_request_id=mocker.Mock(return_value="test"),
        _extract_diffusion_prompt_and_images_from_messages=mocker.Mock(return_value=("prompt", [])),
        _log_inputs=mocker.Mock(),
        chat_completion_full_generator=mocker.AsyncMock(return_value="done"),
    )
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "prompt"}],
        modalities=["image"],
        num_inference_steps=7,
        size="768x512",
        negative_prompt="avoid blur",
        extra_body={"extra_args": {"solver": "euler"}},
    )

    assert asyncio.run(serving_chat._create_chat_completion(request)) == "done"

    sampling_params_list = captured["sampling_params_list"]
    assert sampling_params_list[0].extra_args == {"target_h": 512, "target_w": 768}
    diffusion_params = sampling_params_list[1]
    assert diffusion_params.num_inference_steps == 7
    assert (diffusion_params.height, diffusion_params.width) == (512, 768)
    assert diffusion_params.extra_args == {"solver": "euler"}
    assert captured["prompt"]["negative_prompt"] == "avoid blur"


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
    assert comprehension_params.max_tokens == 4353
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

    assert result[0].max_tokens == 4353


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


def test_to_sampling_params_list_pads_missing_tail_stage_with_defaults():
    """AURA callers may pass 3 semantic model params for a 4-stage engine pipeline."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)
    default_params = [
        SamplingParams(max_tokens=10),
        SamplingParams(max_tokens=20),
        SamplingParams(max_tokens=30),
        SamplingParams(max_tokens=40),
    ]
    instance.engine_client = SimpleNamespace(
        stage_configs=[SimpleNamespace(stage_type="llm") for _ in range(4)],
        default_sampling_params_list=default_params,
    )

    result = instance._to_sampling_params_list(
        [
            {"max_tokens": 1},
            {"max_tokens": 2},
            {"max_tokens": 3},
        ]
    )

    assert len(result) == 4
    assert [params.max_tokens for params in result] == [1, 2, 3, 40]
    assert result[3] is not default_params[3]


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

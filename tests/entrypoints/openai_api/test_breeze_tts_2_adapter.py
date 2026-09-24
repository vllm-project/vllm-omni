# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regression coverage for Breeze's public speech request compatibility."""

from concurrent.futures import Executor
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from pydantic import ValidationError
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.breeze_tts_2 import BreezeTTS2Adapter
from vllm_omni.model_executor.models.breeze_tts_2.prompt import DEFAULT_INSTRUCTION

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
REFERENCE = "data:audio/wav;base64,cHVibGljLXRlc3QtZGF0YQ=="


@dataclass
class ModelConfigStub:
    max_model_len: int


@dataclass
class EngineStub:
    model_config: ModelConfigStub


@dataclass
class ServerStub:
    _tts_executor: Executor | None
    _validate_ref_audio_format: Mock
    _resolve_ref_audio: AsyncMock


@pytest.fixture
def adapter():
    engine = EngineStub(model_config=ModelConfigStub(max_model_len=32))
    server = ServerStub(
        _tts_executor=None,
        _validate_ref_audio_format=Mock(return_value=None),
        _resolve_ref_audio=AsyncMock(return_value=(np.ones(1920, dtype=np.float32), 24000, None)),
    )
    result = BreezeTTS2Adapter(SpeechServingContext(server=server, engine_client=engine))
    result.tokenizer = Mock()
    result.tokenizer.encode.side_effect = lambda text, **kwargs: list(text.encode("utf-8"))
    return result


@pytest.mark.parametrize("voice,tag", [(None, "S0"), ("default", "S0"), ("", "S0"), ("S3", "S3"), ("[S2]", "S2")])
def test_speaker_tags_reach_the_conditioning_prompt(adapter, voice, tag):
    request = OpenAICreateSpeechRequest(input="Hello.", voice=voice)
    assert adapter.validate(request) is None
    prompt = adapter._build_prompt(request, SamplingParams(), None)
    target = bytes(prompt["additional_information"]["breeze_prompt"]["target_ids"]).decode()
    assert target == f"[{tag}]<ins_bos>{DEFAULT_INSTRUCTION}<ins_eos>Hello."


def test_explicit_empty_instruction_remains_empty(adapter):
    request = OpenAICreateSpeechRequest(input="Hello.", instructions="")
    prompt = adapter._build_prompt(request, SamplingParams(), None)
    target = bytes(prompt["additional_information"]["breeze_prompt"]["target_ids"]).decode()
    assert target == "[S0]<ins_bos><ins_eos>Hello."


@pytest.mark.parametrize(
    "extra,expected",
    [({"cfg_scale": 4.0}, 4.0), ({"guidance_scale": 2.0, "cfg_scale": 4.0}, 2.0)],
)
def test_cfg_alias_and_explicit_guidance_precedence(adapter, extra, expected):
    request = OpenAICreateSpeechRequest(input="Hello.", extra_params=extra)
    assert adapter.validate(request) is None
    prompt = adapter._build_prompt(request, SamplingParams(), None)
    assert prompt["additional_information"]["breeze_prompt"]["guidance_scale"] == expected
    assert prompt["additional_information"]["cfg_group"]["role"] == "cond"


@pytest.mark.parametrize("reference", [None, REFERENCE, [REFERENCE]])
def test_custom_voice_and_false_x_vector_are_compatible(adapter, reference):
    request = OpenAICreateSpeechRequest(
        input="Hello.",
        task_type="CustomVoice",
        x_vector_only_mode=False,
        ref_audio=reference,
        ref_text="Reference words." if reference is not None else None,
    )
    assert adapter.validate(request) is None
    if reference is not None:
        adapter.ctx.server._validate_ref_audio_format.assert_called_once_with(REFERENCE)


@pytest.mark.asyncio
async def test_single_reference_list_is_resolved_once_and_preserved_in_prompt(adapter):
    request = OpenAICreateSpeechRequest(
        input="New words.", voice="S1", ref_audio=[REFERENCE], ref_text="Reference words.", task_type="Base"
    )
    assert adapter.validate(request) is None
    prepared = await adapter.build(request, [SamplingParams()], has_inline_ref_audio=True)
    adapter.ctx.server._resolve_ref_audio.assert_awaited_once_with(REFERENCE)
    info = prepared.prompt["additional_information"]
    assert bytes(info["breeze_prompt"]["reference_ids"]).decode() == "[S1]Reference words."
    assert info["breeze_prompt"]["reference_frames"] == 1
    np.testing.assert_array_equal(info["reference_waveform"].numpy(), np.ones(1920, dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"input": " \n"}, "empty"),
        ({"speed": 1.2}, "speed"),
        ({"language": "English"}, "language"),
        ({"ref_audio_2": REFERENCE}, "one reference"),
        ({"speaker_embedding": [0.1, 0.2]}, "one reference"),
        ({"x_vector_only_mode": True}, "x_vector_only_mode"),
        ({"ref_audio": [REFERENCE, REFERENCE], "ref_text": "words"}, "exactly one"),
        ({"ref_audio": REFERENCE}, "transcript"),
        ({"ref_audio": REFERENCE, "ref_text": " "}, "transcript"),
        ({"ref_text": "words"}, "requires ref_audio"),
        ({"task_type": "Base"}, "VoiceDesign"),
        ({"ref_audio": REFERENCE, "ref_text": "words", "task_type": "VoiceDesign"}, "Base"),
        ({"extra_params": {"unknown": 1}}, "Unsupported"),
        ({"extra_params": {"temperature": True}}, "finite number"),
        ({"extra_params": {"temperature": float("nan")}}, "finite number"),
        ({"extra_params": {"cfg_scale": float("inf")}}, "finite number"),
        ({"extra_params": {"cfg_scale": 0}}, "positive"),
        ({"extra_params": {"guidance_scale": 0, "cfg_scale": 4}}, "positive"),
        ({"extra_params": {"repetition_penalty": 0}}, "positive"),
        ({"extra_params": {"temperature": -0.1}}, "temperature"),
        ({"extra_params": {"top_p": 0}}, "top_p"),
        ({"extra_params": {"top_p": 1.1}}, "top_p"),
        ({"extra_params": {"top_k": 1.5}}, "top_k"),
        ({"extra_params": {"top_k": -2}}, "top_k"),
    ],
)
def test_unsupported_or_invalid_inputs_fail_before_build(adapter, kwargs, fragment):
    request = OpenAICreateSpeechRequest(**{"input": "Hello.", **kwargs})
    error = adapter.validate(request)
    assert error is not None and fragment in error
    adapter.ctx.server._resolve_ref_audio.assert_not_awaited()


def test_empty_reference_list_is_rejected_by_request_schema():
    with pytest.raises(ValidationError, match="list cannot be empty"):
        OpenAICreateSpeechRequest(input="Hello.", ref_audio=[])


def test_reference_format_validation_error_is_preserved(adapter):
    adapter.ctx.server._validate_ref_audio_format.return_value = "Unsupported reference format"
    request = OpenAICreateSpeechRequest(input="Hello.", ref_audio=[REFERENCE], ref_text="words")
    assert adapter.validate(request) == "Unsupported reference format"


@pytest.mark.parametrize("requested,expected", [(None, 9), (4, 4), (200, 9)])
def test_sampling_override_clones_defaults_and_caps_remaining_context(adapter, requested, expected):
    defaults = [SamplingParams(max_tokens=20, top_k=50), SamplingParams(max_tokens=100)]
    request = OpenAICreateSpeechRequest(
        input="Hello.", max_new_tokens=requested, extra_params={"repetition_penalty": 1.7}
    )
    result = adapter.apply_sampling_overrides(defaults, request, {"prompt_token_ids": [0] * 23})
    assert result is not defaults and result[0] is not defaults[0]
    assert result[0].max_tokens == expected
    assert result[0].repetition_penalty == 1.7
    assert (defaults[0].max_tokens, defaults[0].repetition_penalty) == (20, 1.0)
    assert result[1].max_tokens == defaults[1].max_tokens == 100


def test_sampling_override_normalizes_disabled_top_k_without_mutating_default(adapter):
    default = SamplingParams(max_tokens=20)
    default.top_k = 0
    result = adapter.apply_sampling_overrides([default], OpenAICreateSpeechRequest(input="Hello."))
    assert result[0].top_k == -1
    assert default.top_k == 0


@pytest.mark.parametrize("prompt_length", [32, 33])
def test_filled_context_is_rejected_without_mutating_sampling_defaults(adapter, prompt_length):
    default = SamplingParams(max_tokens=20)
    request = OpenAICreateSpeechRequest(input="Hello.", max_new_tokens=4)
    with pytest.raises(ValueError, match="fills the model context"):
        adapter.apply_sampling_overrides([default], request, {"prompt_token_ids": [0] * prompt_length})
    assert default.max_tokens == 20

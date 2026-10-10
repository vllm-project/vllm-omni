# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox's public speech request and conditioning contracts."""

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.chatterbox import ChatterboxAdapter
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioning

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def adapter():
    server = SimpleNamespace(
        _tts_executor=None,
        _apply_uploaded_speaker=lambda request: None,
        _validate_ref_audio_format=lambda reference: None,
    )
    engine = SimpleNamespace(model="ResembleAI/chatterbox-turbo")
    return ChatterboxAdapter(SpeechServingContext(server=server, engine_client=engine))


def test_entry_stage_selects_chatterbox():
    assert detect_tts_model_type("chatterbox_t3", None) == "chatterbox"


@pytest.mark.parametrize(
    "fields,error",
    [
        ({"input": "  "}, "empty"),
        ({"language": "French"}, "English"),
        ({"instructions": "Whisper"}, "instructions"),
        ({"voice": "alloy"}, "default"),
        ({"ref_audio": ["a", "b"]}, "one reference"),
        ({"ref_audio_2": "a"}, "one reference"),
    ],
)
def test_unsupported_request_is_rejected(adapter, fields, error):
    request = OpenAICreateSpeechRequest(**{"input": "Hello.", **fields})
    assert error in adapter.validate(request)


@pytest.mark.parametrize("fields", [{}, {"voice": "default"}, {"language": "en"}, {"ref_audio": "clip.wav"}])
def test_builtin_and_reference_requests_are_accepted(adapter, fields):
    assert adapter.validate(OpenAICreateSpeechRequest(input="Hello.", **fields)) is None


def test_builtin_voice_loads_checkpoint_tensors(tmp_path):
    t3 = {"cond_prompt_speech_tokens": torch.tensor([[1, 2, 3]]), "speaker_emb": torch.randn(1, 256)}
    gen = {
        "prompt_token": torch.tensor([[4, 5]]),
        "prompt_feat": torch.randn(1, 4, 80),
        "embedding": torch.randn(1, 192),
    }
    torch.save({"t3": t3, "gen": gen}, tmp_path / "conds.pt")
    voice = VoiceConditioning.from_builtin(str(tmp_path))
    assert torch.equal(voice.cond_tokens, t3["cond_prompt_speech_tokens"])
    assert torch.equal(voice.speaker_emb, t3["speaker_emb"])
    assert torch.equal(voice.prompt_token, gen["prompt_token"])
    assert torch.equal(voice.prompt_feat, gen["prompt_feat"])
    assert torch.equal(voice.embedding, gen["embedding"])


@pytest.mark.asyncio
async def test_builtin_request_uses_shared_prompt_builder_without_reference_encoders(adapter):
    adapter.builtin_voice = VoiceConditioning(
        cond_tokens=torch.tensor([[1, 2, 3]]),
        speaker_emb=torch.randn(1, 256),
        prompt_token=torch.tensor([[4, 5]]),
        prompt_feat=torch.randn(1, 4, 80),
        embedding=torch.randn(1, 192),
    )
    normalized = []

    def encode(text, *, add_special_tokens):
        normalized.append((text, add_special_tokens))
        return [42, 43]

    adapter.tokenizer = SimpleNamespace(encode=encode)
    request = OpenAICreateSpeechRequest(input="hello")
    prepared = await adapter.build(request, [SamplingParams()], False)
    assert normalized == [("Hello.", False)]
    assert prepared.prompt["prompt_token_ids"] == [6561] * 7
    assert prepared.prompt["additional_information"]["ids"] == {"prompt": [42, 43], "speech_token": [1, 2, 3]}
    assert adapter.conditioner is None
    assert prepared.model_type == "chatterbox"


def test_request_token_cap_does_not_mutate_deploy_defaults(adapter):
    defaults = [SamplingParams(max_tokens=1000), SamplingParams(max_tokens=1)]
    result = adapter.apply_sampling_overrides(defaults, OpenAICreateSpeechRequest(input="Hi.", max_new_tokens=50))
    assert [params.max_tokens for params in result] == [50, 1]
    assert defaults[0].max_tokens == 1000

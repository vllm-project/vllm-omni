# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox's public speech request and conditioning contracts."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.chatterbox import ChatterboxAdapter, ChatterboxOriginalAdapter
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
        ({"task_type": "VoiceDesign"}, "VoiceDesign"),
        ({"x_vector_only_mode": True}, "full reference"),
        ({"extra_params": {"cfg_weight": 0.5}}, "Original"),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cls", [ChatterboxAdapter, ChatterboxOriginalAdapter])
async def test_reference_encoding_overlaps_without_mixing_requests(adapter, adapter_cls):
    rendezvous = Barrier(2, timeout=10)

    class Conditioner:
        def prepare(self, wav, sample_rate):
            rendezvous.wait()
            return VoiceConditioning(
                cond_tokens=torch.tensor([[int(wav[0])]]),
                speaker_emb=torch.zeros(1, 256),
                prompt_token=torch.tensor([[1]]),
                prompt_feat=torch.zeros(1, 2, 80),
                embedding=torch.zeros(1, 192),
            )

    with ThreadPoolExecutor(max_workers=adapter_cls.preprocessing_workers) as executor:
        adapter.ctx.server._tts_executor = executor
        concurrent = adapter_cls(adapter.ctx)
        concurrent.conditioner = Conditioner()
        concurrent.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [42])
        concurrent.original_tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[42]))
        prompts = await asyncio.gather(
            concurrent.build_async("Hello.", (np.array([1.0]), 24000)),
            concurrent.build_async("Hello.", (np.array([2.0]), 24000)),
        )
    assert [prompt["additional_information"]["ids"]["speech_token"] for prompt in prompts] == [[1], [2]]


@pytest.mark.parametrize("value", [-1, float("inf"), "0.5"])
def test_original_rejects_invalid_guidance_controls(adapter, value):
    original = ChatterboxOriginalAdapter(adapter.ctx)
    request = OpenAICreateSpeechRequest(input="Hello.", extra_params={"cfg_weight": value})
    assert "finite nonnegative" in original.validate(request)


def test_original_owns_its_stage_and_controls(adapter):
    assert detect_tts_model_type("chatterbox_original_t3", None) == "chatterbox_original"
    original = ChatterboxOriginalAdapter(adapter.ctx)
    assert original.config.variant == "original"
    request = OpenAICreateSpeechRequest(input="Hello.", extra_params={"cfg_weight": 0, "exaggeration": 0.7})
    assert original.validate(request) is None

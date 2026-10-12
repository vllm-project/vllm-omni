# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox's public speech request and conditioning contracts."""

import asyncio
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


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cls", [ChatterboxAdapter, ChatterboxOriginalAdapter])
async def test_64_distinct_references_are_batched_and_cached(adapter, adapter_cls):
    concurrent = adapter_cls(adapter.ctx)
    calls = []
    active = 0

    async def resolve(source):
        return np.array([int(source)], dtype=np.float32), 24000, source

    async def encode(references):
        nonlocal active
        active += 1
        assert active == 1
        calls.append(len(references))
        await asyncio.sleep(0)
        active -= 1
        return [
            VoiceConditioning(
                torch.tensor([[int(wav[0])]]),
                torch.zeros(1, 256),
                torch.zeros(1, 1),
                torch.zeros(1, 2, 80),
                torch.zeros(1, 192),
            )
            for wav, _ in references
        ]

    concurrent.ctx.server._resolve_ref_audio_array = resolve
    concurrent.ctx.server._get_resolved_ref_audio_artifact_key = lambda key: key
    concurrent.condition_batch_async = encode
    voices = await asyncio.gather(*(concurrent.resolve_voice(str(index)) for index in range(64)))
    assert [voice.cond_tokens.item() for voice in voices] == list(range(64))
    assert sum(calls) == 64
    assert max(calls) <= 8
    assert len(calls) < 64
    cached = await asyncio.gather(*(concurrent.resolve_voice(str(index)) for index in range(64)))
    assert all(left is right for left, right in zip(voices, cached, strict=True))
    assert sum(calls) == 64
    assert not concurrent.conditioning_pending


@pytest.mark.asyncio
async def test_reference_singleflight_survives_waiter_cancellation(adapter):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def resolve(source):
        return np.array([1], dtype=np.float32), 24000, source

    voice = VoiceConditioning(
        torch.ones(1, 1), torch.zeros(1, 256), torch.zeros(1, 1), torch.zeros(1, 2, 80), torch.zeros(1, 192)
    )

    async def encode(references):
        calls.append(len(references))
        started.set()
        await release.wait()
        return [voice]

    adapter.ctx.server._resolve_ref_audio_array = resolve
    adapter.ctx.server._get_resolved_ref_audio_artifact_key = lambda key: "same-waveform"
    adapter.condition_batch_async = encode
    first = asyncio.create_task(adapter.resolve_voice("first-url"))
    await started.wait()
    second = asyncio.create_task(adapter.resolve_voice("other-url"))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    assert await second is voice
    assert calls == [1]


@pytest.mark.asyncio
async def test_bad_reference_does_not_poison_batch(adapter):
    voice = VoiceConditioning(
        torch.ones(1, 1), torch.zeros(1, 256), torch.zeros(1, 1), torch.zeros(1, 2, 80), torch.zeros(1, 192)
    )

    async def resolve(source):
        return np.array([1], dtype=np.float32), 24000, source

    async def encode(references):
        return [ValueError("too short"), voice]

    adapter.ctx.server._resolve_ref_audio_array = resolve
    adapter.ctx.server._get_resolved_ref_audio_artifact_key = lambda key: key
    adapter.condition_batch_async = encode
    results = await asyncio.gather(adapter.resolve_voice("bad"), adapter.resolve_voice("good"), return_exceptions=True)
    assert isinstance(results[0], ValueError)
    assert results[1] is voice
    assert "bad" not in adapter.conditioning_cache
    assert not adapter.conditioning_pending


@pytest.mark.asyncio
async def test_cancelled_wave_cannot_overfill_encoder_queue(adapter):
    started = asyncio.Event()
    release = asyncio.Event()
    voice = VoiceConditioning(
        torch.ones(1, 1), torch.zeros(1, 256), torch.zeros(1, 1), torch.zeros(1, 2, 80), torch.zeros(1, 192)
    )

    async def resolve(source):
        return np.array([1], dtype=np.float32), 24000, source

    async def encode(references):
        started.set()
        await release.wait()
        return [voice] * len(references)

    adapter.ctx.server._resolve_ref_audio_array = resolve
    adapter.ctx.server._get_resolved_ref_audio_artifact_key = lambda key: key
    adapter.condition_batch_async = encode
    first = [asyncio.create_task(adapter.resolve_voice(str(index))) for index in range(64)]
    await started.wait()
    for request in first:
        request.cancel()
    await asyncio.gather(*first, return_exceptions=True)
    second = [asyncio.create_task(adapter.resolve_voice(str(index))) for index in range(64, 128)]
    await asyncio.sleep(0)
    assert len(adapter.conditioning_pending) == 64
    assert adapter.conditioning_queue.qsize() <= 64
    assert not any(request.done() for request in second)
    release.set()
    assert all(result is voice for result in await asyncio.wait_for(asyncio.gather(*second), 10))
    assert not adapter.conditioning_pending


@pytest.mark.asyncio
async def test_shutdown_cancels_encoder_waiters(adapter):
    started = asyncio.Event()

    async def resolve(source):
        return np.ones(1, dtype=np.float32), 24000, source

    async def encode(references):
        started.set()
        await asyncio.Event().wait()

    adapter.ctx.server._resolve_ref_audio_array = resolve
    adapter.ctx.server._get_resolved_ref_audio_artifact_key = lambda key: key
    adapter.condition_batch_async = encode
    request = asyncio.create_task(adapter.resolve_voice("voice"))
    await started.wait()
    adapter.shutdown()
    adapter.shutdown()
    with pytest.raises(asyncio.CancelledError):
        await request
    with pytest.raises(RuntimeError, match="shut down"):
        await adapter.resolve_voice("new")
    assert not adapter.conditioning_pending
    assert adapter.conditioning_queue.empty()


@pytest.mark.parametrize("config_field", ["model_config", "engine_args"])
def test_conditioning_device_is_explicit_in_stage_config(adapter, config_field):
    adapter.ctx.server._tts_stage = SimpleNamespace(
        **{config_field: SimpleNamespace(hf_overrides={"conditioning_device": "cuda:2"})}
    )
    configured = ChatterboxAdapter(adapter.ctx)
    assert configured.conditioning_device == torch.device("cuda:2")


def test_unsupported_conditioning_device_is_rejected(adapter):
    adapter.ctx.server._tts_stage = SimpleNamespace(
        model_config=SimpleNamespace(hf_overrides={"conditioning_device": "meta"})
    )
    with pytest.raises(ValueError, match="must be cpu or cuda"):
        ChatterboxAdapter(adapter.ctx)


@pytest.mark.parametrize("adapter_cls", [ChatterboxAdapter, ChatterboxOriginalAdapter])
def test_public_speech_token_cap_is_enforced_before_admission(adapter, adapter_cls):
    configured = adapter_cls(adapter.ctx)
    assert configured.validate(OpenAICreateSpeechRequest(input="Hello.", max_new_tokens=4096)) is None
    assert "4096" in configured.validate(OpenAICreateSpeechRequest(input="Hello.", max_new_tokens=4097))

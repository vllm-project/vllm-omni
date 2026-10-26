# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.ming_tts import MingTTSAdapter
from vllm_omni.model_executor.models.ming_tts.constants import (
    KEY_PROMPT_LATENTS,
    KEY_SPEAKER_SAMPLE_RATES,
    KEY_SPEAKER_WAVEFORM,
    SAMPLE_RATE,
    SPEAKER_EMBEDDING_DIM,
)
from vllm_omni.model_executor.models.ming_tts.speaker_extractor import _resolve_speaker_embeddings
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


class _FakeTokenizer:
    unk_token_id = None

    def convert_tokens_to_ids(self, token):
        return 100 if token == "<audioPatch>" else None

    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(char) for char in text]


class _FakeSpeakerExtractor:
    def __init__(self):
        self.call_count = 0

    def extract_from_waveform(self, waveform, sample_rate):
        self.call_count += 1
        return torch.full((SPEAKER_EMBEDDING_DIM,), float(waveform[0, 0]), dtype=torch.float32)


def _wrapper(*, model="checkpoint-a", cache=None):
    return SimpleNamespace(
        _speaker_cache=cache if cache is not None else SpeakerEmbeddingCache(max_bytes=1024 * 1024),
        _speaker_extractor=_FakeSpeakerExtractor(),
        ming_config=SimpleNamespace(sample_rate=SAMPLE_RATE),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(model=model)),
    )


def _request(*, ref_audio="reference-a", text="First utterance", voice=None):
    return OpenAICreateSpeechRequest(input=text, ref_audio=ref_audio, voice=voice)


def _adapter(references):
    """Each reference supplies decoded samples, rate, and resolver artifact key."""
    artifact_lookups = []

    async def resolve(source):
        samples, sample_rate, _ = references[source]
        return samples, sample_rate, f"locator:{source}"

    def artifact_key(locator):
        artifact_lookups.append(locator)
        return references[locator.removeprefix("locator:")][2]

    server = SimpleNamespace(
        _tts_tokenizer=_FakeTokenizer(),
        uploaded_speakers={},
        _resolve_ref_audio=resolve,
        _get_resolved_ref_audio_artifact_key=artifact_key,
    )
    return MingTTSAdapter(SpeechServingContext(server=server)), server, artifact_lookups


async def _build_and_roundtrip(adapter, request):
    prepared = await adapter.build(request, [], has_inline_ref_audio=request.ref_audio is not None)
    return deserialize_additional_information(serialize_additional_information(prepared.tts_params))


@pytest.mark.asyncio
async def test_inline_reference_artifact_key_survives_wire_roundtrip():
    adapter, _, lookups = _adapter({"reference-a": ([0.25, 0.5], 16000, "content-a")})

    info = await _build_and_roundtrip(adapter, _request())

    assert lookups == ["locator:reference-a"]
    assert info["ming_speaker_audio_keys"] == ["content-a"]
    assert "speaker_audio_keys" not in info
    assert "voice_name" not in info
    assert KEY_PROMPT_LATENTS not in info
    torch.testing.assert_close(info[KEY_SPEAKER_WAVEFORM], torch.tensor([[0.25, 0.5]]))
    assert info[KEY_SPEAKER_SAMPLE_RATES].tolist() == [16000]


@pytest.mark.asyncio
async def test_inline_reference_reuses_embedding_across_text_and_locator_changes():
    adapter, _, _ = _adapter(
        {
            "reference-a": ([0.25, 0.5], 16000, "content-a"),
            "same-content-other-locator": ([0.25, 0.5], 16000, "content-a"),
        }
    )
    wrapper = _wrapper()
    first = await _build_and_roundtrip(adapter, _request())
    second = await _build_and_roundtrip(
        adapter, _request(ref_audio="same-content-other-locator", text="A different utterance")
    )

    first_embeddings = _resolve_speaker_embeddings(wrapper, first)
    second_embeddings = _resolve_speaker_embeddings(wrapper, second)

    assert wrapper._speaker_extractor.call_count == 1
    torch.testing.assert_close(first_embeddings[0], second_embeddings[0])
    assert wrapper._speaker_cache.stats()["entries"] == 1


@pytest.mark.asyncio
async def test_changed_content_at_same_locator_does_not_reuse_embedding():
    references = {"reference-a": ([0.25, 0.5], 16000, "content-a")}
    adapter, _, _ = _adapter(references)
    wrapper = _wrapper()
    first = _resolve_speaker_embeddings(wrapper, await _build_and_roundtrip(adapter, _request()))

    references["reference-a"] = ([0.75, 0.5], 16000, "content-b")
    second = _resolve_speaker_embeddings(wrapper, await _build_and_roundtrip(adapter, _request()))
    repeated = _resolve_speaker_embeddings(wrapper, await _build_and_roundtrip(adapter, _request()))

    assert wrapper._speaker_extractor.call_count == 2
    assert not torch.equal(first[0], second[0])
    torch.testing.assert_close(second[0], repeated[0])


@pytest.mark.asyncio
async def test_changed_sample_rate_does_not_reuse_embedding():
    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

    samples = np.array([0.25, 0.5], dtype=np.float32)
    references = {
        str(rate): (
            samples.tolist(),
            rate,
            OmniOpenAIServingSpeech._make_ref_audio_artifact_cache_key(samples, rate),
        )
        for rate in (16000, 24000)
    }
    adapter, _, _ = _adapter(references)
    wrapper = _wrapper()

    for rate, expected_calls in ((16000, 1), (24000, 2), (16000, 2)):
        info = await _build_and_roundtrip(adapter, _request(ref_audio=str(rate)))
        _resolve_speaker_embeddings(wrapper, info)
        assert wrapper._speaker_extractor.call_count == expected_calls

    assert wrapper._speaker_cache.stats()["entries"] == 2


@pytest.mark.asyncio
async def test_disabled_reference_cache_extracts_each_time():
    adapter, _, _ = _adapter({"reference-a": ([0.25, 0.5], 16000, "content-a")})
    wrapper = _wrapper(cache=SpeakerEmbeddingCache(max_bytes=0))

    for expected_calls in (1, 2):
        info = await _build_and_roundtrip(adapter, _request())
        embeddings = _resolve_speaker_embeddings(wrapper, info)
        assert embeddings[0].shape == (SPEAKER_EMBEDDING_DIM,)
        assert wrapper._speaker_extractor.call_count == expected_calls

    assert wrapper._speaker_cache.stats()["entries"] == 0


@pytest.mark.asyncio
async def test_evicted_reference_is_extracted_again():
    adapter, _, _ = _adapter(
        {
            "reference-a": ([0.25, 0.5], 16000, "content-a"),
            "reference-b": ([0.75, 0.5], 16000, "content-b"),
        }
    )
    embedding_bytes = torch.empty(SPEAKER_EMBEDDING_DIM, dtype=torch.float32).nbytes
    wrapper = _wrapper(cache=SpeakerEmbeddingCache(max_bytes=embedding_bytes))

    for expected_calls, source in enumerate(("reference-a", "reference-b", "reference-a"), start=1):
        info = await _build_and_roundtrip(adapter, _request(ref_audio=source))
        embeddings = _resolve_speaker_embeddings(wrapper, info)
        expected = 0.25 if source == "reference-a" else 0.75
        torch.testing.assert_close(embeddings[0], torch.full((SPEAKER_EMBEDDING_DIM,), expected))
        assert wrapper._speaker_extractor.call_count == expected_calls
        assert wrapper._speaker_cache.stats()["entries"] == 1
        assert wrapper._speaker_cache.memory_bytes() == embedding_bytes


@pytest.mark.asyncio
async def test_inline_reference_without_artifact_key_stays_uncached():
    adapter, _, _ = _adapter({"reference-a": ([0.25, 0.5], 16000, None)})
    wrapper = _wrapper()

    for _ in range(2):
        embeddings = _resolve_speaker_embeddings(wrapper, await _build_and_roundtrip(adapter, _request()))
        assert embeddings[0].shape == (SPEAKER_EMBEDDING_DIM,)

    assert wrapper._speaker_extractor.call_count == 2
    assert wrapper._speaker_cache.stats()["entries"] == 0


@pytest.mark.asyncio
async def test_inline_reference_cache_is_scoped_to_model():
    adapter, _, _ = _adapter({"reference-a": ([0.25, 0.5], 16000, "content-a")})
    info = await _build_and_roundtrip(adapter, _request())
    first = _wrapper()
    second = _wrapper(model="checkpoint-b", cache=first._speaker_cache)

    _resolve_speaker_embeddings(first, info)
    _resolve_speaker_embeddings(second, info)
    _resolve_speaker_embeddings(first, info)
    _resolve_speaker_embeddings(second, info)

    assert first._speaker_extractor.call_count == second._speaker_extractor.call_count == 1
    assert first._speaker_cache.stats()["entries"] == 2


def test_zero_speaker_embedding_bypasses_reference_cache_and_extractor():
    wrapper = _wrapper()

    embeddings = _resolve_speaker_embeddings(
        wrapper,
        {
            "use_zero_spk_emb": True,
            "ming_speaker_audio_keys": ["content-a"],
            KEY_SPEAKER_WAVEFORM: torch.ones((1, 2)),
        },
    )

    torch.testing.assert_close(embeddings[0], torch.zeros(SPEAKER_EMBEDDING_DIM))
    assert wrapper._speaker_extractor.call_count == 0
    assert wrapper._speaker_cache.stats()["entries"] == 0


@pytest.mark.asyncio
async def test_uploaded_voice_retains_versioned_cache_without_duplicate_artifact_entry():
    adapter, server, lookups = _adapter({"reference-a": ([0.25, 0.5], 16000, "content-a")})
    server.uploaded_speakers = {"uploaded": {"embedding_source": "audio"}}
    server._get_uploaded_audio_data = lambda voice: "reference-a"
    created_at = 123
    server._voice_created_at = lambda voice: created_at
    wrapper = _wrapper()

    info = await _build_and_roundtrip(adapter, _request(ref_audio=None, voice="Uploaded"))
    assert not lookups
    assert "ming_speaker_audio_keys" not in info
    # Named voice identity must take precedence even if both forms arrive.
    info["ming_speaker_audio_keys"] = ["content-a"]
    _resolve_speaker_embeddings(wrapper, info)
    _resolve_speaker_embeddings(wrapper, info)
    assert wrapper._speaker_extractor.call_count == 1
    assert wrapper._speaker_cache.stats()["entries"] == 1

    created_at = 124
    changed = await _build_and_roundtrip(adapter, _request(ref_audio=None, voice="Uploaded"))
    _resolve_speaker_embeddings(wrapper, changed)
    assert wrapper._speaker_extractor.call_count == 2
    assert wrapper._speaker_cache.clear("uploaded") == 2
    assert wrapper._speaker_cache.stats()["entries"] == 0

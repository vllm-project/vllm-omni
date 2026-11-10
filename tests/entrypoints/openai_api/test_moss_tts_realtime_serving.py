# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import moss_tts as adapter_module
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.moss_tts import MossTTSAdapter
from vllm_omni.model_executor.models.moss_tts import reference_encoder
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("hf_config", "expected_codec_path"),
    [
        (SimpleNamespace(), "OpenMOSS-Team/MOSS-Audio-Tokenizer"),
        (
            SimpleNamespace(codec_model_name_or_path="/models/custom-moss-codec"),
            "/models/custom-moss-codec",
        ),
    ],
    ids=["default-codec", "configured-codec"],
)
def test_realtime_components_use_the_realtime_model_and_codec(
    monkeypatch: pytest.MonkeyPatch,
    hf_config: SimpleNamespace,
    expected_codec_path: str,
) -> None:
    engine_client = SimpleNamespace(
        model_config=SimpleNamespace(
            model="OpenMOSS-Team/MOSS-TTS-Realtime",
            hf_config=hf_config,
        )
    )

    server = MossTTSAdapter(SpeechServingContext(server=object(), engine_client=engine_client))

    tokenizer = object()
    codec = type("Codec", (), {"to": lambda self, device: self, "eval": lambda self: self})()
    processor_calls = []

    class Processor:
        def __init__(self, *, tokenizer):
            processor_calls.append(tokenizer)

    class_calls = []
    tokenizer_calls = []
    codec_calls = []

    def fake_get_class(class_reference, model_id):
        class_calls.append((class_reference, model_id))
        return Processor

    def fake_load_tokenizer(model_id, *, trust_remote_code):
        tokenizer_calls.append((model_id, trust_remote_code))
        return tokenizer

    def fake_load_codec(model_id, *, trust_remote_code):
        codec_calls.append((model_id, trust_remote_code))
        return codec

    monkeypatch.setattr(adapter_module, "get_class_from_dynamic_module", fake_get_class)
    monkeypatch.setattr(adapter_module.AutoTokenizer, "from_pretrained", fake_load_tokenizer)
    monkeypatch.setattr(adapter_module.AutoModel, "from_pretrained", fake_load_codec)

    components = server._get_moss_realtime_components()

    assert components[0] is tokenizer
    assert components[2] is codec
    assert server._get_moss_realtime_components() is components
    assert class_calls == [
        (
            "processing_mossttsrealtime.MossTTSRealtimeProcessor",
            "OpenMOSS-Team/MOSS-TTS-Realtime",
        )
    ]
    assert tokenizer_calls == [("OpenMOSS-Team/MOSS-TTS-Realtime", True)]
    assert codec_calls == [(expected_codec_path, True)]
    assert processor_calls == [tokenizer]


def test_realtime_serving_builds_the_talker_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    server = MossTTSAdapter(SpeechServingContext(server=SimpleNamespace(uploaded_speakers={})))
    server._moss_variant = "realtime"
    tokenizer = object()
    processor = object()
    codec = object()
    request_thread = threading.get_ident()

    def load_components():
        assert threading.get_ident() != request_thread
        return tokenizer, processor, codec

    server._get_moss_realtime_components = load_components

    reference_codes = torch.arange(64, dtype=torch.int64).reshape(4, 16)
    encode = AsyncMock(return_value=([reference_codes], {0: "reference-cache-key"}))
    monkeypatch.setattr(server, "_encode_moss_references", encode)

    build_call = None

    def fake_build(actual_tokenizer, actual_processor, text, actual_codes):
        nonlocal build_call
        build_call = (actual_tokenizer, actual_processor, text, actual_codes)
        return {
            "prompt_token_ids": [10, 11],
            "codes": {"ref": actual_codes},
            "ids": {"all": [12]},
        }

    monkeypatch.setattr(adapter_module, "build_realtime_prompt", fake_build)

    params = asyncio.run(
        server._build_moss_tts_params(
            OpenAICreateSpeechRequest(
                input="speak this text",
                ref_audio="data:audio/wav;base64,AAAA",
                max_new_tokens=50,
            )
        )
    )

    encode.assert_awaited_once()
    assert encode.call_args.kwargs == {"has_inline_ref_audio": False, "two_speaker": False}
    assert encode.call_args.args[0].ref_audio == "data:audio/wav;base64,AAAA"
    assert build_call == (tokenizer, processor, "speak this text", reference_codes)
    assert params == {
        "prompt_token_ids": [10, 11],
        "codes": {"ref": reference_codes},
        "ids": {"all": [12]},
        "max_new_frames": [50],
        "ref_audio_cache_key": "reference-cache-key",
    }
    assert "prompt_audio_array" not in params


def test_cancelled_component_waiter_does_not_duplicate_codec_load(monkeypatch):
    adapter = MossTTSAdapter(
        SpeechServingContext(
            server=object(),
            engine_client=SimpleNamespace(
                model_config=SimpleNamespace(model="OpenMOSS-Team/MOSS-TTS-Realtime", hf_config=SimpleNamespace())
            ),
        )
    )
    started = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    loads = []
    codec = type("Codec", (), {"to": lambda self, device: self, "eval": lambda self: self})()

    def load_codec(*args, **kwargs):
        loads.append(args)
        started.set()
        assert release.wait(5), "codec loader was not released"
        return codec

    monkeypatch.setattr(adapter_module, "get_class_from_dynamic_module", lambda *args: lambda **kwargs: object())
    monkeypatch.setattr(adapter_module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(adapter_module.AutoModel, "from_pretrained", load_codec)

    def second_load():
        second_started.set()
        return adapter._get_moss_realtime_components()

    async def run():
        first = asyncio.create_task(asyncio.to_thread(adapter._get_moss_realtime_components))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(asyncio.to_thread(second_load))
            try:
                assert await asyncio.to_thread(second_started.wait, 5)
            finally:
                release.set()
            components = await asyncio.wait_for(second, 5)
            assert components[2] is codec
            assert adapter._get_moss_realtime_components() is components
            assert len(loads) == 1
        finally:
            release.set()

    asyncio.run(run())


@pytest.mark.parametrize("voice_name", [None, "speaker"])
def test_realtime_reference_cache_reuses_codes(voice_name):
    cache = SpeakerEmbeddingCache()
    resolve = AsyncMock(return_value=([0.0], 24000, "audio-key"))
    request_thread = threading.get_ident()
    encode_calls = []
    expected = torch.arange(64).reshape(4, 16)

    def batch_encode(wavs, *, num_quantizers):
        assert threading.get_ident() != request_thread
        assert num_quantizers == 16
        encode_calls.append(wavs)
        return SimpleNamespace(audio_codes=expected.T.unsqueeze(1), audio_codes_lengths=torch.tensor([4]))

    encoder = reference_encoder.build_reference_encoder(
        SimpleNamespace(batch_encode=batch_encode), variant="realtime", speaker_cache=cache
    )

    async def run():
        kwargs = dict(
            resolve_ref_audio=resolve,
            get_artifact_key=lambda key: "content-key",
            voice_name=voice_name,
            voice_created_at=1,
        )
        try:
            first, first_key = await encoder.encode("reference", **kwargs)
            second, second_key = await encoder.encode("reference", **kwargs)
            assert torch.equal(first, expected)
            assert torch.equal(second, expected)
            assert first.dtype == torch.int64
            assert first_key == "audio-key"
            assert second_key == (None if voice_name else "audio-key")
            second.zero_()
            third, _ = await encoder.encode("another-reference", **kwargs)
            assert torch.equal(third, expected)
            assert len(encode_calls) == 1
            assert resolve.await_count == (1 if voice_name else 3)
            key = cache.make_cache_key(
                voice_name or "ref:content-key", "moss_tts_realtime_nq16", 1 if voice_name else 0
            )
            assert cache.get(key)["codes"].dtype == torch.int32
            if voice_name:
                kwargs["voice_created_at"] = 2
                await encoder.encode("reference", **kwargs)
                assert len(encode_calls) == 2
                assert resolve.await_count == 2
        finally:
            await encoder.aclose()

    asyncio.run(run())


def test_realtime_reference_batch_trims_each_item():
    cache = SpeakerEmbeddingCache()
    codes = torch.arange(16 * 2 * 5).reshape(16, 2, 5)
    calls = []

    def batch_encode(wavs, *, num_quantizers):
        calls.append(wavs)
        assert num_quantizers == 16
        assert [wav.shape for wav in wavs] == [torch.Size([8]), torch.Size([12])]
        return SimpleNamespace(audio_codes=codes, audio_codes_lengths=torch.tensor([3, 5]))

    encoder = reference_encoder.build_reference_encoder(
        SimpleNamespace(batch_encode=batch_encode), variant="realtime", speaker_cache=cache
    )

    async def resolve(ref):
        return [0.0] * (8 if ref == "short" else 12), 24000, ref

    async def run():
        try:
            short, long = await asyncio.gather(
                encoder.encode("short", resolve_ref_audio=resolve, get_artifact_key=lambda key: key),
                encoder.encode("long", resolve_ref_audio=resolve, get_artifact_key=lambda key: key),
            )
            torch.testing.assert_close(short[0], codes[:, 0, :3].T)
            torch.testing.assert_close(long[0], codes[:, 1, :5].T)
            assert len(calls) == 1
        finally:
            await encoder.aclose()

    asyncio.run(run())


def test_failed_realtime_reference_encoding_does_not_populate_cache():
    cache = SpeakerEmbeddingCache()
    resolve = AsyncMock(return_value=([0.0], 24000, "audio-key"))

    def fail(*args, **kwargs):
        raise ValueError("invalid reference")

    encoder = reference_encoder.build_reference_encoder(
        SimpleNamespace(batch_encode=fail), variant="realtime", speaker_cache=cache
    )

    async def run():
        try:
            with pytest.raises(ValueError, match="invalid reference"):
                await encoder.encode("reference", resolve_ref_audio=resolve, get_artifact_key=lambda key: None)
            assert cache.memory_bytes() == 0
        finally:
            await encoder.aclose()

    asyncio.run(run())

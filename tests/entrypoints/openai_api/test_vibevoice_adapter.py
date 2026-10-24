# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VibeVoice API conditioning, validation and sampling ownership (CPU)."""

import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest
from vllm import SamplingParams
from vllm.exceptions import VLLMUnprocessableEntityError, VLLMValidationError

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.vibevoice import VibeVoiceTTSAdapter
from vllm_omni.model_executor.models.vibevoice.prompt import AUDIO_TOKEN, render_vibevoice_prompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
REF = "data:audio/wav;base64,AAAA"


@pytest.fixture
def adapter(mocker):
    server = SimpleNamespace(
        uploaded_speakers={"alice": {"ref_text": "Stored transcript."}, "embedding": {"embedding_source": "direct"}},
        _tts_model_type="vibevoice",
        model_config=SimpleNamespace(allowed_local_media_path=""),
        _validate_ref_audio_format=OmniOpenAIServingSpeech._validate_ref_audio_format,
        _resolve_ref_audio_array=mocker.AsyncMock(return_value=(np.zeros(3200, dtype=np.float32), 24000, "key")),
        _get_uploaded_audio_data=mocker.Mock(return_value=REF),
        _generate_audio_bytes=mocker.AsyncMock(return_value=(b"wav", "audio/wav")),
    )
    server._validate_ref_audio_format = lambda ref: OmniOpenAIServingSpeech._validate_ref_audio_format(server, ref)
    server._apply_uploaded_speaker = lambda request: OmniOpenAIServingSpeech._apply_uploaded_speaker(server, request)
    return VibeVoiceTTSAdapter(SpeechServingContext(server=server))


@pytest.mark.parametrize(
    "fields",
    [
        {"input": ""},
        {"input": "Speaker 0:"},
        {"input": AUDIO_TOKEN},
        {"voice": "unknown"},
        {"voice": "embedding"},
        {"voice": "alice", "ref_audio": REF},
        {"voice": "alice", "input": "Speaker 0: A\nSpeaker 1: B"},
        {"ref_audio": [REF, REF]},
        {"ref_audio": "ftp://example/a.wav"},
        {"seed": 0},
        {"extra_params": {"guidance_scale": 1.3}},
        {"extra_params": {"temperature": 1}},
        {"max_new_tokens": 16385},
        {"sample_rate": 16000},
        {"speed": 1.1},
        {"instructions": ""},
        {"language": "Auto"},
        {"ref_text": ""},
        {"task_type": "Base"},
        {"ref_audio_2": REF},
        {"word_timestamps": True},
        {"x_vector_only_mode": False},
        {"non_streaming_mode": False},
        {"initial_codec_chunk_frames": 0},
        {"speaker_embedding": [0.0]},
        {"duration_seconds": 1},
        {"ambient_sound": "rain"},
        {"ref_audio_codes": [[1]], "ref_text": "Transcript."},
    ],
)
def test_reject_unsupported_conditioning(adapter, fields):
    request = OpenAICreateSpeechRequest(**{"input": "Hello.", **fields})
    assert adapter.validate(request) is not None
    adapter.ctx.server._resolve_ref_audio_array.assert_not_called()


@pytest.mark.parametrize("scheme", ["file", "FILE"])
def test_local_reference_requires_explicit_permission(adapter, tmp_path, scheme):
    ref = tmp_path / "reference.wav"
    ref.write_bytes(b"fake")  # adapter doesn't read it; a real path is portable and robust.
    source = ref.as_uri()
    if scheme == "FILE":
        source = "FILE" + source[len("file") :]  # uppercase scheme tests case-insensitivity
    request = OpenAICreateSpeechRequest(input="Hello.", ref_audio=source)
    assert "--allowed-local-media-path" in adapter.validate(request)
    adapter.ctx.server._resolve_ref_audio_array.assert_not_called()
    adapter.ctx.server.model_config.allowed_local_media_path = str(tmp_path)
    assert adapter.validate(request) is None  # MediaConnector still owns path containment checks.


@pytest.mark.parametrize("error", [VLLMValidationError("Invalid audio"), VLLMUnprocessableEntityError("HTTP 404")])
async def test_media_client_errors_are_bad_requests(adapter, error):
    adapter.ctx.server._resolve_ref_audio_array.side_effect = error
    with pytest.raises(ValueError, match="reference audio") as caught:
        await adapter.build(OpenAICreateSpeechRequest(input="Hello.", ref_audio=REF), [], True)
    assert caught.value.__cause__ is error


async def test_internal_resolver_errors_are_not_hidden(adapter):
    error = RuntimeError("Internal failure")
    adapter.ctx.server._resolve_ref_audio_array.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        await adapter.build(OpenAICreateSpeechRequest(input="Hello.", ref_audio=REF), [], True)
    assert caught.value is error


def test_empty_reference_rejected_by_protocol():
    with pytest.raises(ValueError, match="list cannot be empty"):
        OpenAICreateSpeechRequest(input="Hello.", ref_audio=[])


@pytest.mark.parametrize("ref", [None, REF])
async def test_single_speaker_matches_offline(adapter, ref):
    request = OpenAICreateSpeechRequest(input="Hello.", ref_audio=ref)
    assert adapter.validate(request) is None
    prepared = await adapter.build(request, [], ref is not None)
    assert prepared.prompt["prompt"] == render_vibevoice_prompt("Hello.", reference=ref is not None)
    assert ("multi_modal_data" in prepared.prompt) == (ref is not None)


@pytest.mark.parametrize("count", [0, 1, 2, 4])
async def test_ordered_multispeaker_references(adapter, count):
    refs = [f"{REF}{n}" for n in range(count)]
    request = OpenAICreateSpeechRequest(
        input="Speaker 3: A\nSpeaker 1: B\nSpeaker 0: C\nSpeaker 2: D\nSpeaker 3: E",
        ref_audio=refs if count else None,
    )
    assert adapter.validate(request) is None
    prepared = await adapter.build(request, [], bool(count))
    prompt = prepared.prompt["prompt"]
    assert prompt.count(AUDIO_TOKEN) == count
    for index, speaker in enumerate([3, 1, 0, 2]):
        assert (f"Speaker {speaker}:<|vision_start|>" in prompt) == (index < count)
    assert [call.args[0] for call in adapter.ctx.server._resolve_ref_audio_array.await_args_list] == refs


async def test_uploaded_voice_keeps_caller_unchanged(adapter):
    request = OpenAICreateSpeechRequest(input="Hello.", voice="Alice")
    assert adapter.validate(request) is None
    prepared = await adapter.build(request, [], False)
    assert prepared.prompt["prompt"] == render_vibevoice_prompt("Hello.")
    assert request.voice == "Alice" and request.ref_audio is None and request.ref_text is None
    adapter.ctx.server._resolve_ref_audio_array.assert_awaited_once_with(REF)
    adapter.ctx.server._get_uploaded_audio_data.return_value = None
    with pytest.raises(ValueError, match="missing"):
        await adapter.build(request, [], False)


@pytest.mark.parametrize(
    "wav,sr", [(np.empty(0), 24000), (np.array([np.nan]), 24000), (np.zeros(61), 1), (np.zeros(2), 0)]
)
async def test_bad_resolved_reference_fails_before_engine(adapter, wav, sr):
    adapter.ctx.server._resolve_ref_audio_array.return_value = (wav, sr, "key")
    with pytest.raises(ValueError, match="reference audio"):
        await adapter.build(OpenAICreateSpeechRequest(input="Hello.", ref_audio=REF), [], True)


def test_detection_capabilities_and_sampling(adapter):
    assert detect_tts_model_type("vibevoice", None) == "vibevoice"
    assert detect_tts_model_type(None, "VibeVoiceForConditionalGeneration") == "vibevoice"
    assert not adapter.load_capabilities().supported_speakers
    assert adapter.capabilities.codec_frame_rate == 7.5
    params = [SamplingParams(max_tokens=100)]
    result = adapter.apply_sampling_overrides(params, OpenAICreateSpeechRequest(input="Hello.", max_new_tokens=5))
    assert result[0].max_tokens == 5 and params[0].max_tokens == 100
    assert result[0].temperature == params[0].temperature  # pipeline owns constraints


async def test_bounded_warmup(adapter):
    await adapter.warmup()
    call = adapter.ctx.server._generate_audio_bytes.await_args
    assert call.args[0].max_new_tokens == 8 and call.args[0].ref_audio is None
    adapter.ctx.server._generate_audio_bytes.side_effect = RuntimeError("test failure")
    await adapter.warmup()  # startup is not failed by a best-effort warmup


@pytest.mark.parametrize("seconds,ok", [(31, True), (60, True), (61, False), (0.5, False)])
def test_reference_duration_uses_adapter_cap(seconds, ok, adapter):
    # The shared resolver calls _finalize_fetched_ref_audio before the adapter;
    # the adapter's max_reference_duration=60 must override the shared 30s cap.
    server = SimpleNamespace(_adapter=adapter, _make_ref_audio_artifact_cache_key=lambda *_: "key")
    server._ref_audio_max_duration = lambda: OmniOpenAIServingSpeech._ref_audio_max_duration(server)
    sr = 24000
    wav = np.zeros(int(seconds * sr), dtype=np.float32)
    if ok:
        OmniOpenAIServingSpeech._finalize_fetched_ref_audio(server, wav, sr)
    else:
        with pytest.raises(ValueError, match="too short" if seconds < 1 else "too long"):
            OmniOpenAIServingSpeech._finalize_fetched_ref_audio(server, wav, sr)


def test_default_adapter_keeps_shared_30s_cap():
    # No adapter (or None max_reference_duration) keeps the shared 30s default.
    server = SimpleNamespace(_adapter=None, _make_ref_audio_artifact_cache_key=lambda *_: "key")
    server._ref_audio_max_duration = lambda: OmniOpenAIServingSpeech._ref_audio_max_duration(server)
    sr = 24000
    with pytest.raises(ValueError, match="too long"):
        OmniOpenAIServingSpeech._finalize_fetched_ref_audio(server, np.zeros(31 * sr, dtype=np.float32), sr)


class _FakeUploadFile:
    """Minimal UploadFile stand-in for upload_voice (file seek/tell + async read)."""

    def __init__(self, data: bytes, filename: str = "voice.wav"):
        self.file = io.BytesIO(data)
        self.filename = filename
        self.content_type = "audio/wav"

    async def read(self) -> bytes:
        return self.file.getvalue()


def _upload_server(tmp_path, adapter):
    """Bare-bones server namespace for upload_voice with the real duration helper."""
    server = SimpleNamespace(
        _adapter=adapter,
        uploaded_speakers={},
        uploaded_speakers_dir=tmp_path,
        _voice_registry_lock=asyncio.Lock,
        _check_registration_allowed=lambda *_: None,
        _check_upload_cap=lambda: None,
        _next_upload_timestamp=lambda: "ts",
        _save_voice_file=lambda *a, **k: None,
        _speaker_metadata_to_header=lambda d: {},
        _evict_existing_upload=lambda *a: None,
        _remember_written_voice=lambda *a: None,
    )
    server._ref_audio_max_duration = lambda: OmniOpenAIServingSpeech._ref_audio_max_duration(server)
    return server


@pytest.mark.parametrize(("seconds", "ok"), [(45.0, True), (61.0, False)])
def test_upload_voice_uses_adapter_cap(tmp_path, adapter, mocker, seconds, ok):
    """POST /v1/audio/voices must honor the same per-adapter cap as ref_audio."""
    sr = 24000
    mocker.patch("soundfile.read", return_value=(np.zeros(int(seconds * sr), dtype=np.float32), sr))
    server = _upload_server(tmp_path, adapter)
    if ok:
        result = asyncio.run(OmniOpenAIServingSpeech.upload_voice(server, _FakeUploadFile(b"x"), "c", "voice_up"))
        assert result["name"] == "voice_up"
    else:
        with pytest.raises(ValueError, match="too long"):
            asyncio.run(OmniOpenAIServingSpeech.upload_voice(server, _FakeUploadFile(b"x"), "c", "voice_up"))


def test_upload_voice_default_cap_without_adapter(tmp_path, mocker):
    """No adapter (or None max_reference_duration) keeps the shared 30s upload cap."""
    sr = 24000
    mocker.patch("soundfile.read", return_value=(np.zeros(31 * sr, dtype=np.float32), sr))
    server = _upload_server(tmp_path, None)
    with pytest.raises(ValueError, match="too long"):
        asyncio.run(OmniOpenAIServingSpeech.upload_voice(server, _FakeUploadFile(b"x"), "c", "voice_up"))


def test_upload_voice_size_cap_scales_with_adapter(tmp_path, adapter, mocker):
    """The 10MB/30s upload-size pre-filter scales with the per-adapter duration cap."""
    sr = 24000
    mocker.patch("soundfile.read", return_value=(np.zeros(45 * sr, dtype=np.float32), sr))
    server = _upload_server(tmp_path, adapter)
    big = _FakeUploadFile(b"\0" * (11 * 1024 * 1024))  # > 10MB, < 20MB
    result = asyncio.run(OmniOpenAIServingSpeech.upload_voice(server, big, "c", "voice_big"))
    assert result["name"] == "voice_big"
    assert result["file_size"] == 11 * 1024 * 1024


@pytest.mark.parametrize(
    ("duration_cap", "size_mib", "limit_text"), [(None, 11, "10 MiB"), (60, 21, "20 MiB"), (40, 14, "13.3333 MiB")]
)
def test_upload_voice_size_cap_rejects_before_decode(tmp_path, adapter, mocker, duration_cap, size_mib, limit_text):
    """Default and scaled limits reject before decoding, with accurate MiB messages."""
    if duration_cap is not None:
        mocker.patch.object(adapter, "max_reference_duration", duration_cap)
    server = _upload_server(tmp_path, adapter if duration_cap is not None else None)
    decode = mocker.patch("soundfile.read")
    file_size = size_mib * 1024 * 1024
    big = _FakeUploadFile(b"\0" * file_size)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(OmniOpenAIServingSpeech.upload_voice(server, big, "c", "voice_big"))
    assert str(exc_info.value) == f"File size exceeds maximum limit of {limit_text}. Got {file_size} bytes."
    decode.assert_not_called()

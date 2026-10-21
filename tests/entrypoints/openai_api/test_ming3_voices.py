# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Persisted model voices and Realtime/Chat/Talker voice selection, CPU only."""

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import soundfile as sf
from starlette.datastructures import Headers, UploadFile

from vllm_omni.config.speech_cache import SpeechCacheConfig
from vllm_omni.entrypoints.openai.ming3_realtime import Ming3RealtimeConnection
from vllm_omni.entrypoints.openai.ming3_voices import register_model_voice_presets, resolve_stored_voice, voice_id
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.ming3_audio_tts import Ming3AudioTTSAdapter
from vllm_omni.model_executor.stage_input_processors.ming_talker import build_ming_talker_additional_info

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "stored"))
    server = object.__new__(OmniOpenAIServingSpeech)
    server.speech_cache_config = SpeechCacheConfig()
    server._init_speaker_storage()
    server._adapter = None
    server._tts_model_type = "ming3_audio_tts"
    return server


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "model"
    presets = root / "voice_presets"
    (presets / "voice_audio").mkdir(parents=True)
    for name in ("a", "b"):
        sf.write(presets / "voice_audio" / f"{name}.wav", np.full(20000, 0.125, dtype=np.float32), 16000)
    (presets / "voice_prompts.json").write_text(
        json.dumps(
            {
                "TEST_VOICE_A": {"text": "感谢你的认可。", "audio": "voice_audio/a.wav"},
                "test_voice_b": {"text": "你好", "audio": "voice_audio/b.wav"},
            },
            ensure_ascii=False,
        )
    )
    return root


def chat_for(store):
    chat = object.__new__(OmniOpenAIServingChat)
    chat.engine_client = SimpleNamespace(
        stage_configs=[
            {"engine_args": {"model_arch": "BailingMoeV3AudioForConditionalGeneration", "model_stage": "thinker"}},
            {"engine_args": {"model_arch": "BailingTalker2", "model_stage": "ming3_tts"}},
        ]
    )
    chat._ming3_speech_service = store
    return chat


async def test_model_import_persists_audio_text_and_reuses_on_restart(store, model):
    before = {str(p.relative_to(model)): p.read_bytes() for p in model.rglob("*") if p.is_file()}
    await register_model_voice_presets(store, str(model))
    assert set(store.uploaded_speakers) == {"test_voice_a", "test_voice_b"}
    assert store.uploaded_speakers["test_voice_a"]["ref_text"] == "感谢你的认可。"
    files = sorted(store.uploaded_speakers_dir.glob("*.safetensors"))
    assert len(files) == 2
    timestamps = {p: p.stat().st_mtime_ns for p in files}
    restored = object.__new__(OmniOpenAIServingSpeech)
    restored.speech_cache_config = SpeechCacheConfig()
    restored._init_speaker_storage()
    restored._adapter = None
    await register_model_voice_presets(restored, str(model))
    assert {p: p.stat().st_mtime_ns for p in files} == timestamps
    audio, sr = restored._load_uploaded_audio("TEST_VOICE_A")
    assert sr == 16000
    np.testing.assert_allclose(audio, 0.125)
    assert {str(p.relative_to(model)): p.read_bytes() for p in model.rglob("*") if p.is_file()} == before


async def test_manifest_change_refreshes_existing_preset(store, model):
    await register_model_voice_presets(store, str(model))
    previous = store.uploaded_speakers["test_voice_a"]["file_path"]
    manifest = model / "voice_presets/voice_prompts.json"
    data = json.loads(manifest.read_text())
    data["TEST_VOICE_A"]["text"] = "新文本"
    manifest.write_text(json.dumps(data))
    await register_model_voice_presets(store, str(model))
    assert store.uploaded_speakers["test_voice_a"]["ref_text"] == "新文本"
    assert not Path(previous).exists()
    assert len(list(store.uploaded_speakers_dir.glob("*.safetensors"))) == 2


async def test_runtime_upload_takes_precedence(store, model):
    store.uploaded_speakers["test_voice_a"] = {"name": "TEST_VOICE_A", "ref_text": "user"}
    await register_model_voice_presets(store, str(model))
    assert store.uploaded_speakers["test_voice_a"]["ref_text"] == "user"
    assert "test_voice_b" in store.uploaded_speakers


@pytest.mark.parametrize(
    "entry",
    [
        None,
        {},
        {"text": 1, "audio": "voice_audio/a.wav"},
        {"text": "x", "audio": "../outside.wav"},
        {"text": "x", "audio": "/tmp/outside.wav"},
        {"text": "x", "audio": "missing.wav"},
    ],
)
async def test_bad_manifest_fails_before_registration(store, model, entry):
    (model / "voice_presets/voice_prompts.json").write_text(json.dumps({"bad": entry}))
    with pytest.raises((ValueError, FileNotFoundError)):
        await register_model_voice_presets(store, str(model))
    assert not store.uploaded_speakers


async def test_duplicate_casefolded_names_rejected(store, model):
    entry = {"text": "x", "audio": "voice_audio/a.wav"}
    (model / "voice_presets/voice_prompts.json").write_text(json.dumps({"A": entry, "a": entry}))
    with pytest.raises(ValueError, match="Duplicate"):
        await register_model_voice_presets(store, str(model))
    assert not store.uploaded_speakers


async def test_missing_manifest_is_optional(store, tmp_path):
    await register_model_voice_presets(store, str(tmp_path))
    assert not store.uploaded_speakers


async def test_named_chat_voice_reaches_talker_with_audio_and_text(store, model):
    await register_model_voice_presets(store, str(model))
    chat = chat_for(store)
    prompt: dict = {}
    request = SimpleNamespace(audio={"voice": {"id": "TEST_VOICE_A"}})
    await chat._attach_ming3_reference_audio(prompt, request)
    info = build_ming_talker_additional_info("你好", prompt)
    assert info["voice_name"] == "test_voice_a"
    assert info["prompt_text"] == "感谢你的认可。"
    assert info["prompt_audio_mode"] == "continuation"
    assert info["speaker_conditioning"] == "prompt"
    from vllm_omni.model_executor.models.ming_flash_omni.reference_audio import unpack_reference_audio

    reference = unpack_reference_audio(info)[0]
    assert reference["sample_rate"] == 16000
    assert reference["samples"][0] == pytest.approx(0.125)
    assert info["instruction"] is None


async def test_ip_voice_takes_precedence_over_stored_selection(store):
    chat = chat_for(store)
    prompt: dict = {}
    await chat._attach_ming3_reference_audio(prompt, SimpleNamespace(audio={"voice": {"id": "灵小柔"}}))
    info = build_ming_talker_additional_info("你好", prompt)
    assert json.loads(info["instruction"])["audio_sequence"][0]["IP"] == "灵小柔"
    assert info["reference_audio"] is None
    assert info["prompt_audio_mode"] == "none"


async def test_unknown_voice_and_conflicting_reference_rejected(store):
    with pytest.raises(ValueError, match="Unknown voice"):
        await resolve_stored_voice(store, {"id": "unknown"})
    with pytest.raises(ValueError, match="either a stored voice"):
        await chat_for(store)._attach_ming3_reference_audio(
            {"additional_information": {"voice_name": "x", "ref_audio": "AA=="}}
        )


@pytest.fixture
def realtime(store):
    socket = SimpleNamespace(query_params={"model": "ming"}, send_text=AsyncMock())
    chat = Mock()
    chat._is_model_supported.return_value = True
    chat._ming3_speech_service = store
    conn = Ming3RealtimeConnection(socket, chat)
    conn._is_connected = True
    return conn


def sent(conn):
    return [json.loads(call.args[0]) for call in conn.websocket.send_text.call_args_list]


async def update(conn, voice="TEST_VOICE_A", **extra):
    await conn.handle_event(
        {
            "type": "session.update",
            "session": {
                "audio": {"output": {"voice": voice}},
                **extra,
            },
        }
    )


async def test_realtime_official_session_voice_and_unknown_error(realtime, store, model):
    await register_model_voice_presets(store, str(model))
    await update(realtime, {"id": "TEST_VOICE_A"})
    assert sent(realtime)[-1]["type"] == "session.updated"
    assert realtime._session["audio"]["output"]["voice"] == "test_voice_a"
    await update(realtime, "unknown")
    assert sent(realtime)[-1]["type"] == "error"
    assert realtime._session["audio"]["output"]["voice"] == "test_voice_a"


async def test_realtime_voice_locked_after_audio(realtime, store, model):
    await register_model_voice_presets(store, str(model))
    await update(realtime)
    realtime._audio_started = True
    realtime._spoken_voice = ("reference", "test_voice_a")
    await update(realtime, {"id": "TEST_VOICE_A"})
    assert sent(realtime)[-1]["type"] == "session.updated"
    await update(realtime, "test_voice_b")
    assert sent(realtime)[-1]["type"] == "error"


async def test_realtime_ip_mode_needs_no_reference(realtime):
    await update(realtime, {"id": "US_Stella"})
    assert sent(realtime)[-1]["type"] == "session.updated"


async def test_realtime_audio_buffer_commit_is_a_user_turn(realtime):
    pcm = np.zeros(24000, dtype="<i2").tobytes()
    await realtime.handle_event({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
    await realtime.handle_event({"type": "input_audio_buffer.commit"})
    assert sent(realtime)[-1]["type"] == "input_audio_buffer.committed"
    assert not realtime._input_audio
    assert realtime.generation_task is None  # Official manual-turn commit waits for response.create.
    audio = realtime._messages[0]["content"][0]["input_audio"]
    waveform, sr = sf.read(io.BytesIO(base64.b64decode(audio["data"])))
    assert sr == 24000 and len(waveform) == 24000


def audio_part(audio, fmt=None):
    part = {"type": "input_audio", "audio": base64.b64encode(audio).decode()}
    if fmt is not None:
        part["format"] = fmt
    return part


@pytest.mark.parametrize("fmt", [None, "pcm16", "wav", "WAVEX", "RF64"])
async def test_realtime_direct_audio_matches_buffer_commit(realtime, fmt):
    samples = np.array([0, 100, -100, 32767], dtype="<i2")
    audio = samples.tobytes()
    if fmt in {"wav", "WAVEX", "RF64"}:
        buffer = io.BytesIO()
        sf.write(buffer, samples, 24000, format=fmt.upper(), subtype="PCM_16")
        audio = buffer.getvalue()
        fmt = "wav"
    await realtime._add_item({"type": "message", "role": "user", "content": [audio_part(audio, fmt)]})
    data = realtime._messages[-1]["content"][0]["input_audio"]["data"]
    actual, rate = sf.read(io.BytesIO(base64.b64decode(data)), dtype="int16")
    np.testing.assert_array_equal(actual, samples)
    assert rate == 24000


@pytest.mark.parametrize(
    "part,error",
    [
        (audio_part(b""), "empty"),
        (audio_part(b"", "wav"), "empty"),
        (audio_part(b"\x00"), "PCM16"),
        (audio_part(b"\x00\x00", "mp3"), "format"),
        ({"type": "input_text", "text": 123}, "string"),
    ],
)
async def test_realtime_direct_item_rejects_invalid_parts_atomically(realtime, part, error):
    realtime._messages = [{"role": "user", "content": "previous"}]
    await realtime.handle_event(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "valid first part"}, part],
            },
        }
    )
    assert sent(realtime)[-1]["type"] == "error"
    assert error in sent(realtime)[-1]["error"]["message"]
    assert realtime._messages == [{"role": "user", "content": "previous"}]


async def test_realtime_empty_wav_rejected(realtime):
    buffer = io.BytesIO()
    sf.write(buffer, np.empty(0, dtype="<i2"), 24000, format="WAV", subtype="PCM_16")
    with pytest.raises(ValueError, match="empty"):
        await realtime._add_item({"type": "message", "role": "user", "content": [audio_part(buffer.getvalue(), "wav")]})
    assert not realtime._messages


async def test_realtime_non_wav_container_cannot_claim_wav_format(realtime):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(4, dtype="<i2"), 24000, format="FLAC")
    with pytest.raises(ValueError, match="WAV"):
        await realtime._add_item({"type": "message", "role": "user", "content": [audio_part(buffer.getvalue(), "wav")]})
    assert not realtime._messages


@pytest.mark.parametrize("fmt", [None, "wav"])
async def test_realtime_direct_audio_size_limit(realtime, fmt):
    realtime._max_audio_filesize_mb = 4 / 1024**2
    with pytest.raises(ValueError, match="maximum size"):
        await realtime._add_item({"type": "message", "role": "user", "content": [audio_part(b"\x00" * 6, fmt)]})
    assert not realtime._messages


@pytest.mark.parametrize("route", ["direct", "buffer"])
@pytest.mark.parametrize("size", ["fits", "header_overflow", "full_raw_limit"])
async def test_realtime_pcm_admission_matches_native_wav_file_limit(realtime, monkeypatch, route, size):
    from vllm.exceptions import VLLMValidationError
    from vllm.multimodal.media.audio import AudioMediaIO

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(2, dtype="<i2"), 24000, format="WAV", subtype="PCM_16")
    limit = len(buffer.getvalue())  # Exact WAV boundary, including the container.
    realtime._max_audio_filesize_mb = limit / 1024**2
    monkeypatch.setattr(AudioMediaIO, "get_max_bytes", lambda self: limit)
    native = AudioMediaIO(audio_backend="soundfile")
    pcm = b"\x00" * {"fits": 4, "header_overflow": 6, "full_raw_limit": limit}[size]
    fits = size == "fits"
    if not fits:
        buffer = io.BytesIO()
        sf.write(buffer, np.frombuffer(pcm, dtype="<i2"), 24000, format="WAV", subtype="PCM_16")
        with pytest.raises(VLLMValidationError):
            native.load_base64("audio/wav", base64.b64encode(buffer.getvalue()).decode())
    realtime._messages = [{"role": "user", "content": "previous"}]
    if route == "buffer":
        await realtime.handle_event({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
        await realtime.handle_event({"type": "input_audio_buffer.commit"})
    else:
        await realtime.handle_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [audio_part(pcm)]},
            }
        )
    if not fits:
        assert sent(realtime)[-1]["type"] == "error"
        assert "maximum size" in sent(realtime)[-1]["error"]["message"]
        assert realtime._messages == [{"role": "user", "content": "previous"}]
        assert not any(
            e["type"] in {"conversation.item.created", "input_audio_buffer.committed"} for e in sent(realtime)
        )
        if route == "buffer":
            assert realtime._input_audio == pcm
    else:
        assert sent(realtime)[-1]["type"] == (
            "input_audio_buffer.committed" if route == "buffer" else "conversation.item.created"
        )
        audio = realtime._messages[-1]["content"][0]["input_audio"]
        assert len(base64.b64decode(audio["data"])) == limit
        waveform, rate = native.load_base64("audio/wav", audio["data"])
        assert rate == 24000 and len(waveform) == len(pcm) // 2
        assert not realtime._input_audio


async def test_realtime_audio_locks_voice_once_after_first_nonempty_chunk(realtime, monkeypatch):
    key = Mock(wraps=realtime._voice_key)
    monkeypatch.setattr(realtime, "_voice_key", key)
    assert not await realtime._emit_audio(np.empty(0), {}, realtime._session)
    key.assert_not_called()
    assert not realtime._audio_started
    for _ in range(3):
        assert await realtime._emit_audio(np.zeros(10), {}, realtime._session)
    key.assert_called_once_with(realtime._session)
    assert realtime._spoken_voice == ("reference", "灵小柔")


@pytest.mark.parametrize(
    "selected,finish_reason",
    [(None, "stop"), ("TEST_VOICE_A", "stop"), ("US_Stella", "stop"), ("灵小柔", "stop"), (None, "length")],
)
async def test_realtime_response_uses_selected_voice_and_emits_pcm24(realtime, store, model, selected, finish_reason):
    await register_model_voice_presets(store, str(model))
    await update(realtime, {"id": selected} if selected is not None else None)
    await realtime.handle_event(
        {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "你好"}]},
        }
    )
    buffer = io.BytesIO()
    sf.write(buffer, np.full(4410, 0.125), 44100, format="WAV")
    audio = base64.b64encode(buffer.getvalue()).decode()
    captured = []
    closed = []

    async def events():
        try:
            for modality, content in [("text", "你好"), ("audio", audio)]:
                choice: dict[str, object] = {"delta": {"content": content}}
                if modality == "audio":
                    choice["finish_reason"] = finish_reason
                yield "data: " + json.dumps({"modality": modality, "choices": [choice]}) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            closed.append(True)

    async def create(request):
        captured.append(request)
        # Exercise actual voice lookup and thinker-to-talker handoff too.
        prompt = {"additional_information": dict(request.additional_information)}
        await chat_for(store)._attach_ming3_reference_audio(prompt, request)
        info = build_ming_talker_additional_info("你好", prompt)
        if selected == "TEST_VOICE_A":
            assert info["prompt_text"] == "感谢你的认可。"
        else:
            assert json.loads(info["instruction"])["audio_sequence"][0]["IP"] == (selected or "灵小柔")
            assert info["reference_audio"] is None
            assert info["prompt_audio_mode"] == "none"
        return events()

    realtime.serving.create_chat_completion = create
    await realtime.handle_event({"type": "response.create"})
    await realtime.generation_task
    assert captured[0].additional_information.get("voice_name") == (selected.lower() if selected else None)
    events_sent = sent(realtime)
    delta = next(e for e in events_sent if e["type"] == "response.output_audio.delta")
    assert "audio" not in delta  # Official event uses delta.
    assert (
        sum(len(base64.b64decode(e["delta"])) for e in events_sent if e["type"] == "response.output_audio.delta")
        == 2400 * 2
    )
    response = events_sent[-1]["response"]
    assert response["status"] == ("incomplete" if finish_reason == "length" else "completed")
    assert response.get("status_details") == (
        {"type": "incomplete", "reason": "max_output_tokens"} if finish_reason == "length" else None
    )
    assert realtime._audio_started
    assert closed


async def test_voice_id_shapes():
    assert voice_id("TEST_VOICE_A") == voice_id({"id": "TEST_VOICE_A"}) == "test_voice_a"
    for value in (None, {}, {"id": ""}, 1):
        with pytest.raises(ValueError):
            voice_id(value)


async def test_registered_voice_works_with_speech_api_adapter(store, model):
    store.engine_client = SimpleNamespace()
    store._diffusion_mode = False
    adapter = store._get_tts_adapter()
    await register_model_voice_presets(store, str(model))
    store._max_instructions_length = 500
    request = OpenAICreateSpeechRequest(input="你好", voice={"id": "TEST_VOICE_A"})
    assert adapter.validate(request) is None
    assert request.voice is None  # Stored voice must not turn into an IP caption.
    assert request.ref_audio.startswith("data:audio/wav;base64,")
    assert request.ref_text == "感谢你的认可。"


async def test_realtime_websocket_route(store, model):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_omni.entrypoints.openai import api_server

    await register_model_voice_presets(store, str(model))
    chat = chat_for(store)
    chat._is_model_supported = lambda _: True
    app = FastAPI()
    app.include_router(api_server.router)
    app.state.openai_serving_chat = chat
    app.state.openai_serving_speech = store
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?model=ming") as socket:
            assert socket.receive_json()["type"] == "session.created"
            socket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "audio": {"output": {"voice": {"id": "TEST_VOICE_A"}}},
                    },
                }
            )
            updated = socket.receive_json()
            assert updated["type"] == "session.updated"
            assert updated["session"]["audio"]["output"]["voice"] == "test_voice_a"


async def test_realtime_response_override_cannot_bypass_voice_lock(realtime, store, model):
    await register_model_voice_presets(store, str(model))
    await update(realtime)
    realtime._messages = [{"role": "user", "content": "hello"}]
    realtime._audio_started = True
    realtime._spoken_voice = ("reference", "test_voice_a")
    await realtime.handle_event(
        {
            "type": "response.create",
            "response": {
                "audio": {"output": {"voice": {"id": "test_voice_b"}}},
            },
        }
    )
    assert sent(realtime)[-1]["type"] == "error"
    assert realtime.generation_task is None


async def test_realtime_generation_failure_has_terminal_event(realtime, store, model):
    await register_model_voice_presets(store, str(model))
    await update(realtime)
    realtime._messages = [{"role": "user", "content": "hello"}]
    realtime.serving.create_chat_completion = AsyncMock(side_effect=RuntimeError("generation failed"))
    await realtime.handle_event({"type": "response.create"})
    await realtime.generation_task
    assert any(e["type"] == "error" for e in sent(realtime))
    assert sent(realtime)[-1]["response"]["status"] == "failed"


@pytest.mark.parametrize(
    "finish_reason,terminal_text,expected",
    [
        ("stop", "", "completed"),
        ("length", "Hello", "incomplete"),
        ("length", "", "incomplete"),
        ("length", "", "failed"),
        ("length", "", "cancelled"),
    ],
)
async def test_realtime_preserves_chat_termination(realtime, mocker, finish_reason, terminal_text, expected):
    import asyncio

    async def events():
        for content, reason in [("Hello" if not terminal_text else "", None), (terminal_text, finish_reason)]:
            yield (
                "data: " + json.dumps({"choices": [{"delta": {"content": content}, "finish_reason": reason}]}) + "\n\n"
            )
        if expected == "failed":
            raise RuntimeError("generation failed after text")
        elif expected == "cancelled":
            raise asyncio.CancelledError
        yield "data: [DONE]\n\n"

    await update(realtime, None, output_modalities=["text"], max_output_tokens=1)
    realtime._messages = [{"role": "user", "content": "hello"}]
    realtime.serving.create_chat_completion = mocker.AsyncMock(return_value=events())
    await realtime.handle_event({"type": "response.create"})
    await realtime.generation_task

    done = [event for event in sent(realtime) if event["type"] == "response.done"]
    assert len(done) == 1
    response = done[0]["response"]
    assert response["status"] == expected
    assert response.get("status_details") == (
        {"type": "incomplete", "reason": "max_output_tokens"} if expected == "incomplete" else None
    )
    assert response["output"][0]["status"] == ("completed" if expected == "completed" else "incomplete")
    assert realtime._messages[-1] == {"role": "assistant", "content": "Hello"}
    assert realtime.serving.create_chat_completion.call_args.args[0].max_completion_tokens == 1


async def test_hf_snapshot_audio_symlinks_are_supported(store, model, tmp_path):
    audio = model / "voice_presets/voice_audio/a.wav"
    blob = tmp_path / "cached-blob"
    audio.rename(blob)
    audio.symlink_to(blob)
    await register_model_voice_presets(store, str(model))
    assert "test_voice_a" in store.uploaded_speakers


async def test_hf_preset_download_uses_model_revision(store, model, monkeypatch):
    download = Mock(return_value=str(model))
    monkeypatch.setattr(
        "vllm_omni.transformers_utils.repo_utils.hf_api",
        Mock(return_value=SimpleNamespace(snapshot_download=download)),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        Mock(side_effect=AssertionError("Download bypassed the tagged Hub client")),
    )
    await register_model_voice_presets(store, "org/model", revision="commit-sha", cache_dir="/cache")
    download.assert_called_once_with(
        "org/model", revision="commit-sha", cache_dir="/cache", allow_patterns=["voice_presets/**"]
    )
    assert len(store.uploaded_speakers) == 2


async def test_realtime_pcm_format_defaults_and_invalid_format(realtime, store, model):
    await register_model_voice_presets(store, str(model))
    await realtime.handle_event(
        {
            "type": "session.update",
            "session": {
                "audio": {"output": {"voice": "TEST_VOICE_A", "format": {"type": "audio/pcm"}}},
            },
        }
    )
    assert sent(realtime)[-1]["type"] == "session.updated"
    assert realtime._session["audio"]["output"]["format"]["rate"] == 24000
    await realtime.handle_event(
        {
            "type": "session.update",
            "session": {
                "audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}},
            },
        }
    )
    assert sent(realtime)[-1]["type"] == "error"
    assert realtime._session["audio"]["output"]["format"]["rate"] == 24000


async def test_realtime_disconnect_closes_active_chat_generator(realtime, store, model):
    import asyncio

    await register_model_voice_presets(store, str(model))
    await update(realtime)
    realtime._messages = [{"role": "user", "content": "hello"}]
    started = asyncio.Event()
    closed = asyncio.Event()

    async def chunks():
        try:
            started.set()
            await asyncio.Event().wait()
            yield "data: [DONE]\n\n"
        finally:
            closed.set()

    realtime.serving.create_chat_completion = AsyncMock(return_value=chunks())
    await realtime.handle_event({"type": "response.create"})
    await asyncio.wait_for(started.wait(), timeout=2)
    realtime._is_connected = False
    await realtime.cleanup()
    assert realtime.generation_task.done()
    assert closed.is_set()


@pytest.mark.asyncio
async def test_immediate_cancel_finishes_response(realtime, store, model):
    import asyncio

    async def events():
        await asyncio.Event().wait()
        yield "data: [DONE]\n\n"

    realtime.serving.create_chat_completion = AsyncMock(return_value=events())
    realtime._is_model_validated = True
    realtime._messages = [{"role": "user", "content": "hello"}]
    await realtime.handle_event({"type": "response.create"})
    await realtime.handle_event({"type": "response.cancel"})
    await asyncio.sleep(0)
    await realtime.generation_task
    assert any(e["type"] == "response.done" for e in sent(realtime))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.create", "response": {"audio": None}},
        {"type": "conversation.item.create", "item": None},
        {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [None]}},
    ],
)
async def test_invalid_event_returns_error(realtime, event):
    realtime._messages = [{"role": "user", "content": "hello"}]
    await realtime.handle_event(event)
    assert sent(realtime)[-1]["type"] == "error"


@pytest.mark.asyncio
async def test_control_voice_validated_on_session_update(realtime):
    await realtime.handle_event(
        {"type": "session.update", "session": {"additional_information": {"voice_name": "does-not-exist"}}}
    )
    assert sent(realtime)[-1]["type"] == "error"


@pytest.mark.asyncio
async def test_reference_change_cannot_bypass_voice_lock(realtime):
    realtime._session["additional_information"] = {"ref_audio": "first"}
    realtime._audio_started = True
    realtime._spoken_voice = realtime._voice_key(realtime._session)
    await realtime.handle_event(
        {"type": "session.update", "session": {"additional_information": {"ref_audio": "different"}}}
    )
    assert sent(realtime)[-1]["type"] == "error"


@pytest.mark.parametrize("interrupt", [False, True])
async def test_cancelled_partial_text_is_kept_in_history(realtime, interrupt):
    import asyncio

    delivered = asyncio.Event()

    async def chunks():
        yield 'data: {"choices": [{"delta": {"content": "Turn left."}}]}\n\n'
        delivered.set()
        await asyncio.Event().wait()

    await realtime.handle_event({"type": "session.update", "session": {"output_modalities": ["text"]}})
    realtime._messages = [{"role": "user", "content": "Where?"}]
    realtime.serving.create_chat_completion = AsyncMock(return_value=chunks())
    await realtime.handle_event({"type": "response.create"})
    await asyncio.wait_for(delivered.wait(), 2)
    if interrupt:
        await realtime.handle_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Why left?"}]},
            }
        )
    await realtime.handle_event({"type": "response.cancel"})
    await realtime.generation_task
    assert sent(realtime)[-1]["response"]["status"] == "cancelled"
    assert realtime._messages[1] == {"role": "assistant", "content": "Turn left."}
    assert len(realtime.serving.create_chat_completion.call_args.args[0].messages) == 1
    if interrupt:
        assert realtime._messages[-1]["role"] == "user"


async def test_realtime_failed_send_marks_connection_disconnected(realtime):
    realtime.websocket.send_text.side_effect = RuntimeError("client disconnected")
    with pytest.raises(RuntimeError, match="client disconnected"):
        await realtime.send_json({"type": "response.done"})
    assert realtime._is_connected is False


@pytest.mark.parametrize("controls", [{"voice_mode": "ip"}, {"ip_name": "灵小柔"}])
async def test_legacy_ip_controls_rejected(realtime, store, controls):
    await update(realtime, {"id": "灵小柔"}, additional_information=controls)
    assert sent(realtime)[-1]["type"] == "error"
    with pytest.raises(ValueError, match="voice_mode only supports reference"):
        await chat_for(store)._attach_ming3_reference_audio({"additional_information": controls})


async def test_ip_voice_caption_preserves_case_and_overrides_registered_audio(store):
    store.uploaded_speakers["us_stella"] = {"name": "US_Stella"}
    prompt: dict = {}
    await chat_for(store)._attach_ming3_reference_audio(prompt, SimpleNamespace(audio={"voice": {"id": "us_stella"}}))
    info = build_ming_talker_additional_info("Hello", prompt)
    assert json.loads(info["instruction"])["audio_sequence"][0]["IP"] == "US_Stella"
    assert info["reference_audio"] is None


@pytest.mark.parametrize("name", ["US_Stella", "us_stella", "灵小柔"])
async def test_ip_names_are_reserved_for_uploads_and_listed(store, model, name):
    store.engine_client = SimpleNamespace()
    store._diffusion_mode = False
    store._get_tts_adapter()
    sample = UploadFile(
        io.BytesIO((model / "voice_presets/voice_audio/a.wav").read_bytes()),
        filename="voice.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )
    try:
        with pytest.raises(ValueError, match="reserved"):
            await store.upload_voice(sample, "consent", name)
    finally:
        await sample.close()
    assert not store.uploaded_speakers
    assert name.lower() in store._get_available_voices()
    assert store._adapter.capabilities.default_speaker == "灵小柔"


async def test_restored_ip_collision_uses_shared_guard_without_deleting_audio(store, model):
    sample = UploadFile(
        io.BytesIO((model / "voice_presets/voice_audio/a.wav").read_bytes()),
        filename="voice.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )
    try:
        await store.upload_voice(sample, "consent", "US_Stella")
    finally:
        await sample.close()
    saved = Path(store.uploaded_speakers["us_stella"]["file_path"])
    original = saved.read_bytes()
    await register_model_voice_presets(store, str(model))
    store.engine_client = SimpleNamespace()
    store._diffusion_mode = False
    store._get_tts_adapter()
    assert "us_stella" not in store.uploaded_speakers
    assert "test_voice_a" in store.uploaded_speakers
    assert saved.read_bytes() == original


@pytest.mark.parametrize("selected", ["us_stella", {"id": "uS_StElLa"}])
async def test_standalone_ip_matches_chat_even_with_legacy_collision(store, selected):
    # A legacy entry must never replace the IP caption in just one endpoint.
    store.uploaded_speakers["us_stella"] = {"name": "US_Stella"}
    adapter = Ming3AudioTTSAdapter(SpeechServingContext(server=store))
    request = OpenAICreateSpeechRequest(input="Hello", voice=selected)
    assert adapter.validate(request) is None
    assert request.voice == "US_Stella"
    prepared = await adapter.build(request, [], False)
    prompt: dict = {}
    await chat_for(store)._attach_ming3_reference_audio(prompt, SimpleNamespace(audio={"voice": selected}))
    chat_info = build_ming_talker_additional_info("Hello", prompt)
    speech_info = prepared.prompt["additional_information"]
    assert speech_info["instruction"] == chat_info["instruction"]
    assert speech_info["prompt_audio_mode"] == chat_info["prompt_audio_mode"] == "none"
    assert speech_info.get("reference_audio") is None


async def test_reserved_model_voice_fails_before_any_registration(store, model):
    entry = {"text": "reference", "audio": "voice_audio/a.wav"}
    (model / "voice_presets/voice_prompts.json").write_text(json.dumps({"my_voice": entry, "US_Stella": entry}))
    with pytest.raises(ValueError, match="reserved"):
        await register_model_voice_presets(store, str(model))
    assert not store.uploaded_speakers


async def test_realtime_ip_voice_locked_after_audio(realtime):
    await update(realtime, {"id": "US_Stella"})
    realtime._audio_started = True
    realtime._spoken_voice = realtime._voice_key(realtime._session)
    await update(realtime, {"id": "us_stella"})
    assert sent(realtime)[-1]["type"] == "session.updated"
    await update(realtime, {"id": "灵小柔"})
    assert sent(realtime)[-1]["type"] == "error"


async def test_realtime_implicit_default_matches_explicit_voice_after_audio(realtime):
    await realtime.handle_event({"type": "session.update", "session": {}})
    assert sent(realtime)[-1]["type"] == "session.updated"
    realtime._audio_started = True
    realtime._spoken_voice = realtime._voice_key(realtime._session)
    await update(realtime, {"id": "灵小柔"})
    assert sent(realtime)[-1]["type"] == "session.updated"
    await update(realtime, {"id": "US_Stella"})
    assert sent(realtime)[-1]["type"] == "error"

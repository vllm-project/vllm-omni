# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused contracts for the VibeVoice OpenAI Speech adapter."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from vllm import SamplingParams
from vllm.multimodal.media import MediaConnector

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.vibevoice import VibeVoiceTTSAdapter
from vllm_omni.model_executor.models.vibevoice.default_voices import (
    DEFAULT_REFERENCE_AUDIO_FILENAMES,
    get_default_reference_audio_path,
)
from vllm_omni.model_executor.models.vibevoice.pipeline import VIBEVOICE_VALID_TOKEN_IDS
from vllm_omni.model_executor.models.vibevoice.processing_vibevoice import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_TOKEN,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _adapter() -> VibeVoiceTTSAdapter:
    server = SimpleNamespace(
        _validate_ref_audio_format=lambda _: None,
        uploaded_speakers={},
        model_config=SimpleNamespace(
            allowed_local_media_path=None,
            allowed_media_domains=None,
        ),
    )
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: list(text.encode("utf-8")))
    engine_client = SimpleNamespace(
        engine=SimpleNamespace(
            input_processor=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer))
        )
    )
    return VibeVoiceTTSAdapter(SpeechServingContext(server=server, engine_client=engine_client))


def _uploaded_voice_adapter(
    *,
    voice_name: str = "alice",
    embedding_source: str = "audio",
    audio_data: str | None = "data:audio/wav;base64,dGVzdA==",
) -> VibeVoiceTTSAdapter:
    adapter = _adapter()
    server = adapter.ctx.server
    server._tts_model_type = "vibevoice"
    server.uploaded_speakers = {
        voice_name.lower(): {
            "name": voice_name,
            "embedding_source": embedding_source,
            "ref_text": "stored transcript",
        }
    }
    server._get_uploaded_audio_data = lambda _voice: audio_data
    server._apply_uploaded_speaker = lambda request: OmniOpenAIServingSpeech._apply_uploaded_speaker(server, request)
    return adapter


def test_adapter_is_registered_and_detected() -> None:
    assert resolve_adapter("vibevoice") is VibeVoiceTTSAdapter
    assert VibeVoiceTTSAdapter.output_policy.expose_finish_reason is True

    serving = object.__new__(OmniOpenAIServingSpeech)
    serving._tts_stage = SimpleNamespace(
        engine_args=SimpleNamespace(
            model_stage="vibevoice",
            model_arch="VibeVoiceForConditionalGeneration",
        )
    )
    assert serving._detect_tts_model_type() == "vibevoice"


def test_speaker_cardinality_and_format_validation() -> None:
    adapter = _adapter()
    mismatch = OpenAICreateSpeechRequest(
        input="Speaker 1: hello\nSpeaker 2: world",
        ref_audio=["file:///one.wav"],
    )
    assert adapter.validate(mismatch) == "VibeVoice found 2 speakers but received 1 reference audios"

    extra_reference = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio=["file:///one.wav", "file:///two.wav"],
    )
    assert adapter.validate(extra_reference) == "VibeVoice found 1 speakers but received 2 reference audios"

    mixed = OpenAICreateSpeechRequest(
        input="Speaker 1: hello\nthis line has no speaker",
        ref_audio=["file:///one.wav"],
    )
    assert "mixed formats" in (adapter.validate(mixed) or "")

    four = OpenAICreateSpeechRequest(
        input="\n".join(f"Speaker {index}: text" for index in range(4)),
        ref_audio=[f"file:///{index}.wav" for index in range(4)],
    )
    assert adapter.validate(four) is None

    four_defaults = OpenAICreateSpeechRequest(
        input="\n".join(f"Speaker {index}: text" for index in range(4)),
    )
    assert adapter.validate(four_defaults) is None

    five = OpenAICreateSpeechRequest(
        input="\n".join(f"Speaker {index}: text" for index in range(5)),
        ref_audio=[f"file:///{index}.wav" for index in range(5)],
    )
    assert adapter.validate(five) == "VibeVoice-1.5B supports at most 4 speakers per request"


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("speaker_embedding", [0.1, 0.2]),
        ("instructions", "speak softly"),
        ("language", "English"),
        ("ref_text", "reference transcript"),
        ("ref_audio_2", "file:///second.wav"),
        ("task_type", "Base"),
        ("ambient_sound", "rain"),
        ("duration_seconds", 1.0),
        ("x_vector_only_mode", False),
        ("initial_codec_chunk_frames", 4),
        ("non_streaming_mode", False),
        ("word_timestamps", True),
    ],
)
def test_explicit_unsupported_fields_are_rejected(field_name: str, field_value: object) -> None:
    kwargs = {
        "input": "hello",
        "ref_audio": "file:///voice.wav",
        field_name: field_value,
    }
    if field_name == "speaker_embedding":
        kwargs["ref_audio"] = None
    request = OpenAICreateSpeechRequest(**kwargs)

    assert f"does not support '{field_name}'" in (_adapter().validate(request) or "")


def test_bundled_default_references_are_packaged_and_resolvable() -> None:
    assert DEFAULT_REFERENCE_AUDIO_FILENAMES == (
        "default_0.wav",
        "default_1.wav",
        "default_2.wav",
        "default_3.wav",
    )

    expected_sources = (
        (
            "tests/assets/cosyvoice3/zero_shot_prompt.wav",
            "c7b31d6dbe7cc6a716dded00550db5b50940bf209e424e4ad207b12e657c8ff6",
        ),
        (
            "vllm_omni/model_executor/models/step_audio2/assets/default_female.wav",
            "5fc92ddcd9bc9af10437d9630642378777a98fc260f16508a9777db12c830a41",
        ),
        (
            "tests/assets/indextts2/ref_audio.wav",
            "e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8",
        ),
        (
            "tests/assets/qwen3_tts/clone_2.wav",
            "480f55f41c71c3d79c2a9acc48f0bfb3c5a46222e6e9ebf3e2888e93501a6b5c",
        ),
    )
    repository_root = Path(__file__).resolve().parents[3]
    manifest_path = get_default_reference_audio_path(0).parent.parent / "ASSET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["assets"]
    assert [record["slot"] for record in records] == list(range(4))
    assert [Path(record["packaged_path"]).name for record in records] == list(DEFAULT_REFERENCE_AUDIO_FILENAMES)
    assert manifest["excluded_candidates"][0]["path"] == "tests/assets/glm_tts/jiayan_zh.wav"

    adapter = _adapter()
    for index, (source_relative_path, expected_sha256) in enumerate(expected_sources):
        path = get_default_reference_audio_path(index)
        source_path = repository_root / source_relative_path
        assert path.read_bytes() == source_path.read_bytes()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
        assert records[index]["sha256"] == expected_sha256
        assert records[index]["canonical_vllm_omni_path"] == source_relative_path
        assert records[index]["license"] == "Apache-2.0"

        info = sf.info(path)
        assert info.channels == 1
        assert 0 < info.duration <= 60

        waveform, sample_rate = asyncio.run(adapter._resolve_default_reference(index))
        assert sample_rate == records[index]["sample_rate_hz"]
        assert waveform.ndim == 1
        assert 0 < waveform.size <= 60 * sample_rate
        assert np.isfinite(waveform).all()
        assert float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))) > 1e-5


def test_build_uses_bundled_defaults_in_first_appearance_order(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    request = OpenAICreateSpeechRequest(
        input="Speaker 8: first\nSpeaker 3: second\nSpeaker 8: third",
    )
    resolved_indices: list[int] = []

    async def resolve(index: int):
        resolved_indices.append(index)
        return np.full(3_200, index + 1, dtype=np.float32), 24_000

    monkeypatch.setattr(adapter, "_resolve_default_reference", resolve)
    prepared = asyncio.run(adapter.build(request, [], False))

    assert resolved_indices == [0, 1]
    audio_items = prepared.prompt["multi_modal_data"]["audio"]
    np.testing.assert_array_equal(audio_items[0][0], np.full(3_200, 1, dtype=np.float32))
    np.testing.assert_array_equal(audio_items[1][0], np.full(3_200, 2, dtype=np.float32))
    assert " Speaker 0: first\n" in prepared.prompt["prompt"]
    assert " Speaker 1: second\n" in prepared.prompt["prompt"]
    assert " Speaker 0: third\n" in prepared.prompt["prompt"]


def test_uploaded_voice_resolves_to_reference_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _uploaded_voice_adapter()
    request = OpenAICreateSpeechRequest(input="hello", voice="Alice")
    resolved_sources: list[str] = []

    async def resolve(source: str):
        resolved_sources.append(source)
        return np.zeros(3_200, dtype=np.float32), 24_000

    monkeypatch.setattr(adapter, "_resolve_reference", resolve)
    assert adapter.validate(request) is None
    assert adapter.validate(request) is None

    prepared = asyncio.run(adapter.build(request, [], False))

    assert resolved_sources == ["data:audio/wav;base64,dGVzdA=="]
    assert request.voice is None
    assert request.ref_audio == "data:audio/wav;base64,dGVzdA=="
    assert request.ref_text is None
    assert len(prepared.prompt["multi_modal_data"]["audio"]) == 1
    assert adapter.validate(request) is None


def test_uploaded_voice_errors_are_request_visible() -> None:
    both = OpenAICreateSpeechRequest(
        input="hello",
        voice="alice",
        ref_audio="file:///voice.wav",
    )
    assert "exactly one of 'voice' or 'ref_audio'" in (_uploaded_voice_adapter().validate(both) or "")

    unknown = OpenAICreateSpeechRequest(input="hello", voice="missing")
    assert "Unknown VibeVoice voice 'missing'" in (_uploaded_voice_adapter().validate(unknown) or "")

    embedding = OpenAICreateSpeechRequest(input="hello", voice="alice")
    assert "uses a speaker embedding" in (_uploaded_voice_adapter(embedding_source="direct").validate(embedding) or "")

    missing = OpenAICreateSpeechRequest(input="hello", voice="alice")
    adapter = _uploaded_voice_adapter(audio_data=None)
    assert adapter.validate(missing) is None
    with pytest.raises(ValueError, match="Audio file for uploaded voice 'alice' is missing"):
        asyncio.run(adapter.build(missing, [], False))


@pytest.mark.parametrize(
    ("extra_params", "seed", "message"),
    [
        ({"guidance_scale": float("nan")}, None, "guidance_scale must be finite"),
        ({"guidance_scale": True}, None, "guidance_scale must be a real number"),
        ({"guidance_scale": 20.1}, None, "guidance_scale must be between 0.0 and 20.0"),
        ({"num_diffusion_steps": True}, None, "must be a positive integer"),
        ({"num_diffusion_steps": 51}, None, "cannot exceed 50"),
        (
            {"guid" + "ence_scale": 1.3},
            None,
            "Unsupported VibeVoice extra_params: ['guid" + "ence_scale']",
        ),
        (None, 42, "request-level seed"),
    ],
)
def test_runtime_controls_are_validated(
    extra_params: dict[str, object] | None,
    seed: int | None,
    message: str,
) -> None:
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
        extra_params=extra_params,
        seed=seed,
    )
    assert message in (_adapter().validate(request) or "")


@pytest.mark.parametrize(
    "extra_params",
    [
        {"guidance_scale": 0.0, "num_diffusion_steps": 1},
        {"guidance_scale": 20.0, "num_diffusion_steps": 50},
    ],
)
def test_runtime_control_boundaries_are_accepted(extra_params: dict[str, object]) -> None:
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
        extra_params=extra_params,
    )
    assert _adapter().validate(request) is None


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "message"),
    [
        (np.zeros((2, 2, 2), dtype=np.float32), 24_000, "one- or two-dimensional"),
        (np.zeros(0, dtype=np.float32), 24_000, "reference audio is empty"),
        (np.zeros(16, dtype=np.float32), 0, "sample rate must be positive"),
        (np.zeros(60 * 24_000 + 1, dtype=np.float32), 24_000, "maximum is 60s"),
    ],
)
def test_reference_media_errors_are_request_visible(
    monkeypatch: pytest.MonkeyPatch,
    waveform: np.ndarray,
    sample_rate: int,
    message: str,
) -> None:
    adapter = _adapter()

    async def fetch_audio_async(_self, _source):
        return waveform, sample_rate

    monkeypatch.setattr(MediaConnector, "fetch_audio_async", fetch_audio_async)
    with pytest.raises(ValueError, match=message):
        asyncio.run(adapter._resolve_reference("file:///bad.wav"))


def test_build_preserves_prompt_audio_order_and_request_uuids(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    request = OpenAICreateSpeechRequest(
        input="Speaker 8: first\nSpeaker 3: second\nSpeaker 8: third",
        ref_audio=["ref-a", "ref-b"],
    )

    async def resolve(source: str):
        value = 1.0 if source == "ref-a" else 2.0
        return np.full(3_200, value, dtype=np.float32), 24_000

    monkeypatch.setattr(adapter, "_resolve_reference", resolve)
    prepared = asyncio.run(adapter.build(request, [], True))
    prepared = adapter.finalize_prepared_request(prepared, "speech-request-7")

    assert prepared.output_policy.expose_finish_reason is False
    prompt = prepared.prompt["prompt"]
    reference_segment = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
    assert prompt.count(reference_segment) == 2
    assert " Speaker 0: first\n" in prompt
    assert " Speaker 1: second\n" in prompt
    assert " Speaker 0: third\n" in prompt
    assert prompt.endswith(f" Speech output:\n{AUDIO_BOS_TOKEN}")
    audio_items = prepared.prompt["multi_modal_data"]["audio"]
    np.testing.assert_array_equal(audio_items[0][0], np.full(3_200, 1.0, dtype=np.float32))
    np.testing.assert_array_equal(audio_items[1][0], np.full(3_200, 2.0, dtype=np.float32))
    assert prepared.prompt["multi_modal_uuids"] == {"audio": ["speech-request-7:audio:0", "speech-request-7:audio:1"]}


def test_sampling_constraints_are_idempotent_without_mutating_caller() -> None:
    adapter = _adapter()
    request = OpenAICreateSpeechRequest(input="hello", ref_audio="ref")
    caller = SamplingParams(
        temperature=0.7,
        max_tokens=123,
        allowed_token_ids=[1],
        stop_token_ids=[2],
        detokenize=True,
    )
    before = copy.deepcopy(caller)

    (resolved,) = adapter.apply_sampling_overrides([caller], request)
    (repeated,) = adapter.apply_sampling_overrides([resolved], request)

    assert resolved is not caller
    assert resolved.temperature == 0.0
    assert resolved.max_tokens == 123
    assert resolved.allowed_token_ids == VIBEVOICE_VALID_TOKEN_IDS
    assert resolved.stop_token_ids == [151643]
    assert resolved.all_stop_token_ids == {151643}
    assert resolved.detokenize is False
    assert repeated == resolved
    assert caller == before


def test_dict_sampling_constraints_do_not_mutate_caller() -> None:
    adapter = _adapter()
    request = OpenAICreateSpeechRequest(input="hello", ref_audio="ref", max_new_tokens=7)
    caller = {
        "temperature": 0.7,
        "allowed_token_ids": [1],
        "stop_token_ids": [2],
        "detokenize": True,
        "max_tokens": 123,
    }

    (resolved,) = adapter.apply_sampling_overrides([caller], request)

    assert resolved is not caller
    assert resolved["temperature"] == 0.0
    assert resolved["allowed_token_ids"] == VIBEVOICE_VALID_TOKEN_IDS
    assert resolved["stop_token_ids"] == [151643]
    assert resolved["detokenize"] is False
    assert resolved["max_tokens"] == 7
    assert caller == {
        "temperature": 0.7,
        "allowed_token_ids": [1],
        "stop_token_ids": [2],
        "detokenize": True,
        "max_tokens": 123,
    }

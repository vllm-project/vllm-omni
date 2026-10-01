# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused contracts for the VibeVoice OpenAI Speech adapter."""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from vllm import SamplingParams

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
        model_config=SimpleNamespace(allowed_local_media_path=None, allowed_media_domains=None),
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


def test_speaker_cardinality_and_format_validation() -> None:
    adapter = _adapter()
    mismatch = OpenAICreateSpeechRequest(
        input="Speaker 1: hello\nSpeaker 2: world",
        ref_audio=["file:///one.wav"],
    )
    assert adapter.validate(mismatch) == "VibeVoice found 2 speakers but received 1 reference audios"

    four = OpenAICreateSpeechRequest(
        input="\n".join(f"Speaker {i}: text" for i in range(4)),
        ref_audio=[f"file:///{i}.wav" for i in range(4)],
    )
    assert adapter.validate(four) is None

    five = OpenAICreateSpeechRequest(
        input="\n".join(f"Speaker {i}: text" for i in range(5)),
        ref_audio=[f"file:///{i}.wav" for i in range(5)],
    )
    assert adapter.validate(five) == "VibeVoice-1.5B supports at most 4 speakers per request"


def test_bundled_default_references_are_packaged_and_resolvable() -> None:
    assert DEFAULT_REFERENCE_AUDIO_FILENAMES == ("default_0.wav", "default_1.wav", "default_2.wav", "default_3.wav")
    manifest_path = get_default_reference_audio_path(0).parent.parent / "ASSET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [r["slot"] for r in manifest["assets"]] == [0, 1, 2, 3]
    assert all(r["license"] == "Apache-2.0" for r in manifest["assets"])

    adapter = _adapter()
    for index in range(4):
        path = get_default_reference_audio_path(index)
        info = sf.info(path)
        assert info.channels == 1
        assert 0 < info.duration <= 60
        waveform, sample_rate = asyncio.run(adapter._resolve_default_reference(index))
        assert waveform.ndim == 1
        assert 0 < waveform.size <= 60 * sample_rate
        assert np.isfinite(waveform).all()


def test_build_uses_bundled_defaults_in_first_appearance_order(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    request = OpenAICreateSpeechRequest(input="Speaker 8: first\nSpeaker 3: second\nSpeaker 8: third")
    resolved_indices: list[int] = []

    async def resolve(index: int):
        resolved_indices.append(index)
        return np.full(3_200, index + 1, dtype=np.float32), 24_000

    monkeypatch.setattr(adapter, "_resolve_default_reference", resolve)
    prepared = asyncio.run(adapter.build(request, [], False))

    assert resolved_indices == [0, 1]
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
    prepared = asyncio.run(adapter.build(request, [], False))

    assert resolved_sources == ["data:audio/wav;base64,dGVzdA=="]
    assert request.voice is None
    assert request.ref_audio == "data:audio/wav;base64,dGVzdA=="
    assert len(prepared.prompt["multi_modal_data"]["audio"]) == 1


@pytest.mark.parametrize(
    ("extra_params", "message"),
    [
        ({"guidance_scale": float("nan")}, "guidance_scale must be finite"),
        ({"guidance_scale": 20.1}, "guidance_scale must be between 0.0 and 20.0"),
        ({"num_diffusion_steps": 51}, "cannot exceed 50"),
    ],
)
def test_runtime_controls_are_validated(extra_params, message) -> None:
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
        extra_params=extra_params,
    )
    assert message in (_adapter().validate(request) or "")


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
    assert prompt.endswith(f" Speech output:\n{AUDIO_BOS_TOKEN}")
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
    assert resolved.allowed_token_ids == VIBEVOICE_VALID_TOKEN_IDS
    assert resolved.stop_token_ids == [151643]
    assert resolved.detokenize is False
    assert repeated == resolved
    assert caller == before

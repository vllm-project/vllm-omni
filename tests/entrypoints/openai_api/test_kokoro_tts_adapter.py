# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Kokoro-82M speech adapter.

Covers detection, voice/language capabilities, the reject matrix, the
engine prompt ``build()`` hands to the single-stage chunk generator and the
synthesis-error flag.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.data_entry_keys import unflatten_payload
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import PreparedRequest, SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.kokoro import KokoroAdapter
from vllm_omni.outputs.mm_outputs import MultimodalPayload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODEL = "hexgrad/Kokoro-82M"
_VOICES = ("af_bella", "af_heart", "bm_george", "ef_dora", "zf_xiaobei")


@pytest.fixture
def adapter(tmp_path: Path, mocker: MockerFixture) -> KokoroAdapter:
    for voice in _VOICES:
        (tmp_path / f"{voice}.pt").write_bytes(b"")
    resolve = mocker.patch(
        "vllm_omni.entrypoints.openai.tts_adapters.kokoro.resolve_voices_dir", return_value=str(tmp_path)
    )
    engine = SimpleNamespace(model_config=SimpleNamespace(model=_MODEL, revision="abc123"))
    adapter = KokoroAdapter(SpeechServingContext(server=SimpleNamespace(), engine_client=engine))
    adapter.load_capabilities()
    resolve.assert_called_once_with(_MODEL, "abc123")
    return adapter


def _build(adapter: KokoroAdapter, **kwargs) -> PreparedRequest:
    request = OpenAICreateSpeechRequest(input="Hello from Kokoro.", **kwargs)
    return asyncio.run(adapter.build(request, sampling_params_list=[], has_inline_ref_audio=False))


def test_kokoro_detection_and_registration() -> None:
    assert resolve_adapter("kokoro") is KokoroAdapter
    assert KokoroAdapter.stage_keys == {"kokoro"}
    assert KokoroAdapter.model_archs == {"KokoroForConditionalGeneration"}
    assert detect_tts_model_type("kokoro", None) == "kokoro"
    assert detect_tts_model_type(None, "KokoroForConditionalGeneration") == "kokoro"
    # Speed scales predicted durations in-model; the server must not stretch again.
    assert KokoroAdapter.native_speed_control


def test_capabilities_list_checkpoint_voices_with_af_heart_default(adapter: KokoroAdapter) -> None:
    caps = adapter.capabilities
    assert caps.supported_speakers == frozenset(_VOICES)
    assert caps.default_speaker == "af_heart"
    assert {"Auto", "en-us", "en-gb", "es", "ja", "zh"} <= caps.supported_languages


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"voice": "default"},
        {"voice": "BM_George"},
        {"voice": "af_heart", "language": "Auto"},
        {"voice": "af_heart", "language": "en-gb"},
        {"voice": "ef_dora", "speed": 1.5},
        {"seed": 3},
    ],
)
def test_validate_accepts(adapter: KokoroAdapter, kwargs: dict) -> None:
    assert adapter.validate(OpenAICreateSpeechRequest(input="Hello.", **kwargs)) is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"voice": "vivian"}, "Invalid voice"),
        ({"language": "klingon"}, "Unsupported language"),
        ({"extra_params": {"temperature": 0.3}}, "extra_params"),
        ({"word_timestamps": True}, "word_timestamps"),
        ({"ref_audio": "https://example.com/ref.wav"}, "ref_audio"),
        ({"ref_text": "transcript"}, "ref_text"),
        ({"speaker_embedding": [0.1, 0.2]}, "speaker_embedding"),
        ({"x_vector_only_mode": True}, "x_vector_only_mode"),
        ({"task_type": "Base"}, "task_type"),
        ({"instructions": "speak happily"}, "instructions"),
        ({"ambient_sound": "rain"}, "ambient_sound"),
        ({"duration_seconds": 2.0}, "duration_seconds"),
        ({"max_new_tokens": 100}, "max_new_tokens"),
        ({"initial_codec_chunk_frames": 4}, "initial_codec_chunk_frames"),
        ({"non_streaming_mode": True}, "non_streaming_mode"),
    ],
)
def test_validate_rejects(adapter: KokoroAdapter, kwargs: dict, message: str) -> None:
    error = adapter.validate(OpenAICreateSpeechRequest(input="Hello.", **kwargs))
    assert error is not None and message in error


def test_validate_rejects_blank_input(adapter: KokoroAdapter) -> None:
    assert adapter.validate(OpenAICreateSpeechRequest(input="   ")) == "Input text cannot be empty"


def test_validate_rejects_language_whose_g2p_is_not_installed(adapter: KokoroAdapter, mocker: MockerFixture) -> None:
    mocker.patch(
        "vllm_omni.entrypoints.openai.tts_adapters.kokoro.missing_g2p_package",
        side_effect=lambda code: "misaki[zh]" if code == "z" else None,
    )
    error = adapter.validate(OpenAICreateSpeechRequest(input="Hello.", voice="zf_xiaobei"))
    assert error is not None and "misaki[zh]" in error
    error = adapter.validate(OpenAICreateSpeechRequest(input="Hello.", voice="af_heart", language="zh"))
    assert error is not None and "misaki[zh]" in error
    assert adapter.validate(OpenAICreateSpeechRequest(input="Hello.", voice="af_heart")) is None


@pytest.mark.parametrize(
    ("meta_error", "failed"),
    [
        (None, False),
        (torch.tensor([0, 0], dtype=torch.int32), False),
        (torch.tensor([0, 1], dtype=torch.int32), True),
        ([torch.tensor([0]), torch.tensor([1])], True),
    ],
)
def test_collect_response_metadata_flags_synthesis_error(
    adapter: KokoroAdapter, meta_error: torch.Tensor | list | None, failed: bool
) -> None:
    # Build the payload the way the output processor hands it to serving:
    # the runner's dotted "meta.error" key arrives as {"meta": {"error": ...}}.
    flat = {"model_outputs": torch.zeros(4), "sr": torch.tensor(24000)}
    if meta_error is not None:
        flat["meta.error"] = meta_error
    collect: dict = {}
    adapter.collect_response_metadata(MultimodalPayload.from_dict(unflatten_payload(flat)), collect)
    assert collect.get("audio_synthesis_error", False) is failed


def test_build_carries_request_in_additional_information(adapter: KokoroAdapter) -> None:
    prepared = _build(adapter, voice="BM_George", speed=1.25, language="en-gb", seed=11)
    assert prepared.model_type == "kokoro"
    assert prepared.output_policy.accumulate_nonstreaming
    prompt = prepared.prompt
    assert prompt["prompt_token_ids"] == [0]
    assert prompt["additional_information"] == {
        "text": ["Hello from Kokoro."],
        "voice": ["bm_george"],
        "speed": [1.25],
        "language": ["en-gb"],
        "seed": [11],
    }
    assert isinstance(prompt["cache_salt"], str) and prompt["cache_salt"]


def test_build_defaults(adapter: KokoroAdapter) -> None:
    info = _build(adapter, voice="default", language="Auto").prompt["additional_information"]
    assert info == {"text": ["Hello from Kokoro."], "voice": ["af_heart"], "speed": [1.0]}

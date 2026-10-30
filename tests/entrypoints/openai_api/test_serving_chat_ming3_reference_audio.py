# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming3 chat voice controls survive the thinker-to-talker handoff."""

import base64
import json
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.engine import AdditionalInformationPayload
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.model_executor.models.ming_flash_omni.reference_audio import (
    pack_reference_audio,
    unpack_reference_audio,
)
from vllm_omni.model_executor.stage_input_processors.ming_talker import build_ming_talker_additional_info

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _serving_chat() -> OmniOpenAIServingChat:
    chat = object.__new__(OmniOpenAIServingChat)
    chat.engine_client = SimpleNamespace(
        stage_configs=[
            {"engine_args": {"model_arch": "BailingMoeV3AudioForConditionalGeneration", "model_stage": "thinker"}},
            {"engine_args": {"model_arch": "BailingTalker2", "model_stage": "ming3_tts"}},
        ]
    )
    chat.model_config = SimpleNamespace(allowed_local_media_path="", allowed_media_domains=None)
    return chat


def test_chat_ip_name_passes_to_talker_without_reference():
    info = build_ming_talker_additional_info("你好", {"additional_information": {"voice_name": "灵小柔"}})

    assert json.loads(info["instruction"])["audio_sequence"][0]["IP"] == "灵小柔"
    assert info["prompt_audio_mode"] == "none"
    assert info["reference_audio"] is None


@pytest.mark.asyncio
async def test_chat_base64_reference_passes_decoded_samples_to_talker(mocker: MockerFixture, monkeypatch):
    monkeypatch.setenv("MING_TTS_REFERENCE_WAV", "/missing/server-reference.wav")
    monkeypatch.setenv("MING_TTS_REFERENCE_TEXT", "server transcript")
    chat = _serving_chat()
    encoded = base64.b64encode(b"RIFFfakeWAVE").decode()
    prompt = {"additional_information": {"ref_audio": encoded}}
    connector = mocker.patch("vllm_omni.entrypoints.openai.serving_chat.MediaConnector").return_value
    connector.fetch_audio_async = mocker.AsyncMock(return_value=(np.array([0.1, -0.1], dtype=np.float32), 16000))

    await chat._attach_ming3_reference_audio(prompt)
    info = build_ming_talker_additional_info("你好", prompt)

    connector.fetch_audio_async.assert_awaited_once_with(f"data:audio/wav;base64,{encoded}")
    assert "ref_audio" not in prompt["additional_information"]
    assert info["speaker_conditioning"] == "prompt"
    assert info["prompt_audio_mode"] == "speaker_only"
    reference = unpack_reference_audio(info)[0]
    assert reference["sample_rate"] == 16000
    assert reference["samples"].tolist() == pytest.approx([0.1, -0.1])


@pytest.mark.asyncio
async def test_chat_references_stay_numeric_through_engine_and_connector(mocker):
    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackDecoder, OmniMsgpackEncoder

    class NumericAudio(np.ndarray):
        def tolist(self):
            raise AssertionError("reference audio was expanded into Python floats")

    clips = [np.array([0.1, -0.2], dtype=np.float32).view(NumericAudio), np.array([0.3], dtype=np.float32)]
    rates = [16000, 24000]
    connector = mocker.patch("vllm_omni.entrypoints.openai.serving_chat.MediaConnector").return_value
    connector.fetch_audio_async = mocker.AsyncMock(side_effect=list(zip(clips, rates, strict=True)))
    prompt = {"additional_information": {"ref_audio": ["AA==", "AA=="], "prompt_text": "reference"}}
    await _serving_chat()._attach_ming3_reference_audio(prompt)
    wire = serialize_additional_information(prompt["additional_information"])
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(wire), type=AdditionalInformationPayload)
    info = build_ming_talker_additional_info("你好", deserialize_additional_information(decoded))
    payload = OmniMsgpackDecoder().decode(OmniMsgpackEncoder().encode(info))
    references = unpack_reference_audio(payload)
    assert [ref["sample_rate"] for ref in references] == rates
    assert info["prompt_audio_mode"] == "continuation"
    for reference, original in zip(references, clips, strict=True):
        torch.testing.assert_close(reference["samples"], torch.from_numpy(original))


@pytest.mark.parametrize(
    "references",
    [
        [],
        [(np.zeros(0, dtype=np.float32), 16000)],
        [(np.zeros(1), 0)],
        [(np.array([float("nan")]), 16000)],
        [(np.zeros(1), 16000)] * 17,
    ],
)
def test_reference_audio_bounds(references):
    with pytest.raises(ValueError, match="Reference audio"):
        pack_reference_audio(references)


@pytest.mark.parametrize(
    "count,length,rate", [(3, 25, 1), (2, 1_500_000, 96000)], ids=["75-seconds-total", "3000000-samples-total"]
)
def test_reference_audio_accepts_above_old_aggregate_limits(count, length, rate):
    references = [(np.full(length, index, dtype=np.float32), rate) for index in range(count)]

    restored = unpack_reference_audio(pack_reference_audio(references))

    assert len(restored) == count
    for reference, (samples, sample_rate) in zip(restored, references, strict=True):
        assert reference["sample_rate"] == sample_rate
        torch.testing.assert_close(reference["samples"], torch.from_numpy(samples))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "count,samples,fetches", [(17, [0.1], 0), (2, [float("nan")], 1)], ids=["too-many-clips", "invalid-first-clip"]
)
async def test_chat_rejects_invalid_references_before_further_io(mocker, count, samples, fetches):
    connector_class = mocker.patch("vllm_omni.entrypoints.openai.serving_chat.MediaConnector")
    connector = connector_class.return_value
    connector.fetch_audio_async = mocker.AsyncMock(return_value=(np.array(samples, dtype=np.float32), 16000))
    prompt = {"additional_information": {"ref_audio": ["AA=="] * count}}

    with pytest.raises(ValueError, match="Reference audio"):
        await _serving_chat()._attach_ming3_reference_audio(prompt)

    assert connector.fetch_audio_async.await_count == fetches
    if fetches == 0:
        connector_class.assert_not_called()


@pytest.mark.asyncio
async def test_other_models_skip_ming3_reference_validation(mocker):
    chat = _serving_chat()
    chat.engine_client.stage_configs = [
        {"engine_args": {"model_arch": "BailingMM2NativeForConditionalGeneration", "model_stage": "thinker"}}
    ]
    connector_class = mocker.patch("vllm_omni.entrypoints.openai.serving_chat.MediaConnector")
    info = {"ref_audio": ["AA=="] * 17}

    await chat._attach_ming3_reference_audio({"additional_information": info})

    connector_class.assert_not_called()
    assert info == {"ref_audio": ["AA=="] * 17}


@pytest.mark.asyncio
async def test_chat_rejects_ip_and_reference_together():
    chat = _serving_chat()
    prompt = {"additional_information": {"voice_name": "灵小柔", "ref_audio": "AA=="}}

    with pytest.raises(ValueError, match="either a stored voice"):
        await chat._attach_ming3_reference_audio(prompt)


@pytest.mark.parametrize("text", ["你好", "Hello world"])
@pytest.mark.parametrize("controls", [{}, {"voice_name": None}, {"temperature": 0.5}])
def test_default_voice_is_ip_without_reference_files(text, controls, monkeypatch):
    monkeypatch.setenv("MING_TTS_REFERENCE_WAV", "/missing/legacy.wav")
    monkeypatch.setenv("MING_TTS_REFERENCE_TEXT", "legacy reference")
    info = build_ming_talker_additional_info(text, {"additional_information": controls})
    assert info["voice_mode"] == "ip"
    assert info["ip_name"] == "灵小柔"
    assert json.loads(info["instruction"])["audio_sequence"][0]["IP"] == "灵小柔"
    assert info["prompt_audio_mode"] == "none"
    assert info["speaker_conditioning"] == "mute"
    assert all(info[key] is None for key in ("prompt_wav", "reference_audio", "prompt_text", "spk_emb"))


@pytest.mark.parametrize("server_reference", [None, "/custom/reference.wav"], ids=["unset", "set"])
def test_explicit_reference_without_source_is_rejected(monkeypatch, server_reference):
    monkeypatch.delenv("MING_TTS_REFERENCE_WAV", raising=False)
    monkeypatch.setenv("MING_TTS_REFERENCE_TEXT", "server transcript")
    if server_reference is not None:
        monkeypatch.setenv("MING_TTS_REFERENCE_WAV", server_reference)
    with pytest.raises(ValueError, match="requires reference audio"):
        build_ming_talker_additional_info("你好", {"additional_information": {"voice_mode": "reference"}})

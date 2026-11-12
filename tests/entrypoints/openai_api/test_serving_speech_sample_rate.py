# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speech serving resolves audio rates via the shared reader, with no defaults."""

import struct
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

import vllm_omni.entrypoints.openai.serving_speech as serving_speech_module
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.sample_rate import DeprecatedSampleRateKeyWarning, MissingSampleRateError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def _bare_server() -> OmniOpenAIServingSpeech:
    server = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    server._tts_model_type = "test_tts"
    server.engine_client = SimpleNamespace(model_config=SimpleNamespace(model="test-model", async_chunk=False))
    return server


@pytest.mark.asyncio
async def test_streaming_first_audio_without_rate_fails_loudly() -> None:
    # Every audio-emitting model advertises its rate; a payload without one
    # is a producer bug that must fail the stream (for every format — the
    # rate labels the PCM bytes and builds the resampler), never fall back
    # to a default.
    server = _bare_server()

    async def results():
        yield SimpleNamespace(multimodal_output={"model_outputs": torch.zeros(320)})

    chunks = server._generate_audio_chunks(results(), "req-no-sr", response_format="pcm")
    with pytest.raises(ValueError, match="sample rate metadata"):
        async for _ in chunks:
            pass


@pytest.mark.asyncio
async def test_streaming_wav_header_carries_the_model_rate() -> None:
    server = _bare_server()

    async def results():
        yield SimpleNamespace(
            multimodal_output={
                "model_outputs": torch.zeros(320),
                "sr": torch.tensor(UNUSUAL_RATE, dtype=torch.int32),
            }
        )

    chunks = [chunk async for chunk in server._generate_audio_chunks(results(), "req-sr", response_format="wav")]

    assert chunks[0][:4] == b"RIFF"
    assert struct.unpack("<I", chunks[0][24:28])[0] == UNUSUAL_RATE


@pytest.mark.asyncio
async def test_streaming_rate_change_mid_stream_raises() -> None:
    server = _bare_server()

    async def results():
        yield SimpleNamespace(
            multimodal_output={
                "model_outputs": torch.zeros(320),
                "sr": torch.tensor(24000, dtype=torch.int32),
            }
        )
        yield SimpleNamespace(
            multimodal_output={
                "model_outputs": torch.zeros(320),
                "sr": torch.tensor(44100, dtype=torch.int32),
            }
        )

    chunks = server._generate_audio_chunks(results(), "req-change", response_format="pcm")
    with pytest.raises(ValueError, match="changed during streaming"):
        async for _ in chunks:
            pass


@pytest.mark.asyncio
async def test_non_streaming_without_rate_fails_loudly(mocker: MockerFixture) -> None:
    server = _bare_server()
    server._speech_output_policies = {}

    async def results():
        yield SimpleNamespace(multimodal_output={"model_outputs": torch.zeros(320)})

    async def prepare(_request, **_kwargs):
        return "req-no-sr", results(), {}

    mocker.patch.object(server, "_prepare_speech_generation", side_effect=prepare)
    mocker.patch.object(server, "_validate_tts_generation")
    mocker.patch.object(server, "_audio_encode_speed", return_value=1.0)

    with pytest.raises(MissingSampleRateError):
        await server._generate_audio_bytes(OpenAICreateSpeechRequest(input="Hello", model="test-model"))


@pytest.mark.asyncio
async def test_diffusion_payload_legacy_key_labels_the_waveform(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The diffusion formatter surfaces the rate under its legacy
    # ``audio_sample_rate`` spelling (until the formatter PR normalizes it);
    # the serving layer must resolve it instead of silently defaulting to
    # 24000 and mislabeling the waveform.
    server = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    server._adapter = SimpleNamespace()

    async def build(_request, sampling_params_list, has_inline_ref_audio):
        return SimpleNamespace(prompt={"text": "Hello"})

    server._adapter.build = build
    monkeypatch.setattr(serving_speech_module, "validate_tts_request", lambda adapter, request: None)
    monkeypatch.setattr(server, "_get_normalized_voice", lambda voice: voice)
    monkeypatch.setattr(server, "_log_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_audio_encode_speed", lambda request: 1.0)

    async def generate(*, prompt, request_id, sampling_params_list, output_modalities, watermarking):
        yield SimpleNamespace(multimodal_output={"audio": torch.zeros(320), "audio_sample_rate": UNUSUAL_RATE})

    server._diffusion_engine = SimpleNamespace(
        default_sampling_params_list=[SimpleNamespace()],
        generate=generate,
    )

    with pytest.warns(DeprecatedSampleRateKeyWarning):
        response = await server._create_diffusion_speech(
            OpenAICreateSpeechRequest(input="Hello", model="test-diffusion")
        )

    assert response.status_code == 200
    assert struct.unpack("<I", response.body[24:28])[0] == UNUSUAL_RATE

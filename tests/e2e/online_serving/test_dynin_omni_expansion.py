# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example online tests for Dynin-Omni model.
"""

import base64
import gc
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from vllm.assets.image import ImageAsset

from tests import conftest as tests_conftest
from tests.conftest import OmniServerParams
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL = "snu-aidas/Dynin-Omni"
STAGE_CONFIG = str(Path(__file__).parent.parent / "stage_configs" / "dynin_omni_ci.yaml")
_WHISPER_SAMPLE_RATE_HZ = 16_000

T2I_PROMPT = "A high quality detailed living room interior photo."
T2S_PROMPT = "Please read this sentence naturally: Hello from Dynin-Omni online serving."
I2I_PROMPT = "Transform this outdoor nature boardwalk scene into a painting style with vivid colors."

TEST_PARAMS = [OmniServerParams(model=MODEL, stage_config_path=STAGE_CONFIG, stage_init_timeout=600)]
_STAGE_COUNT = 3
_I2I_STAGE_SAMPLING = {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0, "detokenize": False}


def _prepare_audio_waveform_for_whisper(audio_data: np.ndarray, samplerate: int) -> np.ndarray:
    """Normalize decoded audio into a mono 16 kHz float32 waveform for Whisper."""
    if samplerate <= 0:
        raise ValueError(f"Invalid audio sample rate: {samplerate}")

    waveform = np.asarray(audio_data, dtype=np.float32)
    if waveform.ndim == 0:
        raise ValueError("Audio waveform must have at least one dimension")
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if waveform.size == 0:
        raise ValueError("Empty audio waveform")

    if samplerate != _WHISPER_SAMPLE_RATE_HZ:
        target_num_samples = max(int(round(waveform.shape[0] * _WHISPER_SAMPLE_RATE_HZ / samplerate)), 1)
        source_positions = np.arange(waveform.shape[0], dtype=np.float64)
        target_positions = np.linspace(
            0.0,
            max(waveform.shape[0] - 1, 0),
            num=target_num_samples,
            dtype=np.float64,
        )
        waveform = np.interp(target_positions, source_positions, waveform).astype(np.float32)

    return np.ascontiguousarray(np.clip(waveform, -1.0, 1.0), dtype=np.float32)


def _convert_audio_bytes_to_text_without_ffmpeg(raw_bytes: bytes) -> str:
    """Dynin t2s keeps Whisper transcription local to this test module and avoids ffmpeg."""
    import whisper

    data, samplerate = sf.read(BytesIO(raw_bytes), dtype="float32", always_2d=True)
    audio_waveform = _prepare_audio_waveform_for_whisper(data, samplerate)

    model = whisper.load_model("small", device="cpu")
    try:
        transcript = model.transcribe(
            audio_waveform,
            temperature=0.0,
            word_timestamps=True,
            condition_on_previous_text=False,
        )["text"]
    finally:
        del model
        gc.collect()

    return transcript or ""


@pytest.fixture
def dynin_t2s_openai_client(openai_client, monkeypatch):
    monkeypatch.setattr(
        tests_conftest,
        "convert_audio_bytes_to_text",
        _convert_audio_bytes_to_text_without_ffmpeg,
    )
    return openai_client


def _build_t2i_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": f"<|t2i|> {prompt}"}]}]


def _build_t2s_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": f"<|t2s|> {prompt}"}]}]


def _build_i2i_messages(prompt: str) -> list[dict]:
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    buffer = BytesIO()
    input_image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<|i2i|> {prompt}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }
    ]


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_send_i2i_request_001(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": _build_i2i_messages(I2I_PROMPT),
        "modalities": ["image"],
        "extra_body": {
            "sampling_params_list": [dict(_I2I_STAGE_SAMPLING) for _ in range(_STAGE_COUNT)],
        },
    }
    openai_client.send_diffusion_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_send_t2i_request_001(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": _build_t2i_messages(T2I_PROMPT),
        "modalities": ["image"],
    }
    openai_client.send_diffusion_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_send_t2s_request_001(omni_server, dynin_t2s_openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": _build_t2s_messages(T2S_PROMPT),
        "modalities": ["audio"],
    }
    dynin_t2s_openai_client.send_omni_request(request_config)

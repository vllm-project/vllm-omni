# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kyutai AURA on POST /v1/videos/sync: the MP4 keeps the model's 16 fps clock with its speech track, and an
explicit non-native fps or an invalid ``num_chunks`` is a client error raised before any chunk is generated.

The fake engine runs the pipeline's own request check on the sampling params the endpoint builds. A pipeline's
``OmniClientError`` reaches the API process as the same error type, so raising it from ``generate`` follows
the real path from the API boundary on.
"""

import base64
import io
import json
import wave
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pytest_mock import MockerFixture

from tests.entrypoints.openai_api.test_video_server import FakeAsyncOmni, MockVideoResult
from vllm_omni.diffusion.models.kyutai_aura.config import KYUTAI_AURA_1_3B
from vllm_omni.diffusion.models.kyutai_aura.pipeline_kyutai_aura import KyutaiAuraPipeline
from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
SAMPLE_RATE = 24000
SECONDS = 2


class _AuraEngine(FakeAsyncOmni):
    """Checks each request as the pipeline does, then returns 16 fps frames and the speech, as the pipeline's
    post-process does (``fps`` and the waveform in ``multimodal_output``)."""

    def __init__(self) -> None:
        super().__init__()
        self.model_class_name = "KyutaiAuraPipeline"
        self.generated = 0

    async def generate(self, prompt, request_id, sampling_params_list):
        self.captured_sampling_params_list = sampling_params_list
        request = SimpleNamespace(prompts=[prompt], sampling_params=sampling_params_list[0])
        pipeline = cast(KyutaiAuraPipeline, SimpleNamespace(cfg=KYUTAI_AURA_1_3B))
        KyutaiAuraPipeline._request_inputs(pipeline, cast(Any, request))
        self.generated += 1
        frames = np.zeros((SECONDS * KYUTAI_AURA_1_3B.fps, 16, 16, 3), dtype=np.uint8)
        speech = np.zeros(SECONDS * SAMPLE_RATE, dtype=np.float32)
        yield MockVideoResult(
            [frames],
            audios=[speech],
            sample_rate=SAMPLE_RATE,
            multimodal_output={"video": [frames], "fps": KYUTAI_AURA_1_3B.fps},
        )


@pytest.fixture
def aura(mocker: MockerFixture):
    encode = mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    engine = _AuraEngine()
    app = FastAPI()
    app.state.api_server_count = 1
    app.include_router(router)
    app.state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=engine,
        model_name="kyutai/AURA-1.3B",
    )
    with TestClient(app) as client:
        yield client, engine, encode


def _speech_data_url() -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(np.zeros(SECONDS * SAMPLE_RATE, dtype=np.int16).tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()


def _portrait_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="gray").save(buffer, format="PNG")
    return buffer.getvalue()


def _post(client, *, fps=None, extra_params=None):
    data = {
        "prompt": "A person speaking to the camera.",
        "audio_reference": json.dumps({"audio_url": _speech_data_url()}),
    }
    if fps is not None:
        data["fps"] = str(fps)
    if extra_params is not None:
        data["extra_params"] = json.dumps(extra_params)
    files = {"input_reference": ("portrait.png", _portrait_png(), "image/png")}
    return client.post("/v1/videos/sync", data=data, files=files)


@pytest.mark.parametrize("fps", [None, 16], ids=["fps-omitted", "fps-native"])
def test_sync_keeps_the_native_clock_with_the_speech_track(aura, fps):
    client, engine, encode = aura
    response = _post(client, fps=fps)

    assert response.status_code == 200, response.text
    assert response.content == b"fake-video"
    assert engine.generated == 1
    kwargs = encode.call_args.kwargs
    assert kwargs["fps"] == KYUTAI_AURA_1_3B.fps
    assert kwargs["audio_sample_rate"] == SAMPLE_RATE
    assert len(kwargs["audio"]) == SECONDS * SAMPLE_RATE


@pytest.mark.parametrize("fps", [24, 15.5])
def test_sync_rejects_a_non_native_fps_before_generation(aura, fps):
    client, engine, encode = aura
    response = _post(client, fps=fps)

    assert response.status_code == 400
    assert "16 fps" in response.json()["detail"]
    assert engine.generated == 0
    encode.assert_not_called()


def test_sync_accepts_a_positive_chunk_limit(aura):
    client, engine, _ = aura
    response = _post(client, extra_params={"num_chunks": 2})

    assert response.status_code == 200, response.text
    assert engine.generated == 1
    assert engine.captured_sampling_params_list[0].extra_args["num_chunks"] == 2


@pytest.mark.parametrize("num_chunks", [0, -1, True, 1.5, "2"])
def test_sync_rejects_an_invalid_chunk_limit_before_generation(aura, num_chunks):
    client, engine, encode = aura
    response = _post(client, extra_params={"num_chunks": num_chunks})

    assert response.status_code == 400
    assert "num_chunks must be a positive integer" in response.json()["detail"]
    assert engine.generated == 0
    encode.assert_not_called()

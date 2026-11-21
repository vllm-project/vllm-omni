# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared env for ``http_invalid`` (real :func:`omni_server`)."""

from __future__ import annotations

import io
import os

import pytest
from PIL import Image

# Match ``tests/e2e/online_serving/*`` module-level env for subprocess serve.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_TEST_CLEAN_GPU_MEMORY", "0")


@pytest.fixture
def tiny_png_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module", autouse=True)
def _isolated_server_storage(tmp_path_factory):
    # The default /tmp/storage may belong to another user on shared hosts.
    # Set this before the server subprocess starts, not inside the probe.
    directory = tmp_path_factory.mktemp("invalid-parameter-storage")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_OMNI_SERVER_STORAGE__PATH", str(directory))
        yield directory


@pytest.fixture(scope="module", autouse=True)
def service_survives_invalid_requests(_isolated_server_storage, omni_server, request, run_level):
    """Run real inference before/after the module's rejected requests.

    One probe pair per module-scoped deployment, not per parameter row. The
    sleep-level-2 contract deliberately discards weights and is checked
    separately; it is not a rejected request from which inference can recover.
    """
    from tests.helpers.client import OnlineOmniClient

    client = OnlineOmniClient(
        host=omni_server.host,
        port=omni_server.port,
        api_key="EMPTY",
        run_level=run_level,
        log_stats=omni_server.log_stats,
    )
    _assert_valid_inference(client, omni_server, request.module)
    yield
    if request.module.__name__.endswith("test_invalid_server_control"):
        return  # The final test intentionally leaves the instance in level-2 sleep.
    _assert_valid_inference(client, omni_server, request.module)


def _assert_valid_inference(client, server, module):
    import base64
    import wave
    from io import BytesIO

    model = server.model
    # MODEL_PREFIX keeps the organization/repository suffix in local paths.
    model_id = "/".join(model.rstrip("/").split("/")[-2:])
    print(f"[invalid-parameters] verifying real inference: {model}", flush=True)
    if model_id == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
        response = client.send_chat_completions_http_request(
            {
                "json": {
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hello briefly."}],
                    "modalities": ["text", "audio"],
                    "max_tokens": 32,
                    "stream": False,
                },
                "timeout": 180,
            }
        )[0]
        assert response.status_code == 200, response.error_message
        message = response.json_body["choices"][0]["message"]
        assert message.get("content") and message.get("audio", {}).get("data"), response.error_message
    elif model_id == "Wan-AI/Wan2.2-T2V-A14B-Diffusers":
        response = client.send_video_diffusion_request(
            {
                "form_data": {
                    "model": model,
                    "prompt": "A red ball on a table",
                    "size": "832x480",
                    "num_frames": 5,
                    "num_inference_steps": 2,
                    "seed": 42,
                }
            }
        )[0]
        assert response.success, response.error_message
    elif model_id in {"Qwen/Qwen-Image", "Qwen/Qwen-Image-Edit"}:
        if model_id.endswith("-Edit"):
            image = BytesIO()
            Image.new("RGB", (512, 512), "gray").save(image, format="PNG")
            response = client.send_images_edits_http_request(
                {
                    "data": {
                        "model": model,
                        "prompt": "Make the image brighter",
                        "size": "512x512",
                        "num_inference_steps": "2",
                    },
                    "files": {"image": ("input.png", image.getvalue(), "image/png")},
                    "timeout": 180,
                }
            )[0]
        else:
            response = client.send_images_generations_http_request(
                {
                    "json": {
                        "model": model,
                        "prompt": "A red apple",
                        "size": "512x512",
                        "num_inference_steps": 2,
                        "response_format": "b64_json",
                    },
                    "timeout": 180,
                }
            )[0]
        data = response.json_body["data"]
        assert len(data) == 1
        with Image.open(BytesIO(base64.b64decode(data[0]["b64_json"], validate=True))) as decoded_image:
            decoded_image.load()
            assert decoded_image.width > 0 and decoded_image.height > 0
    else:
        body = {"model": model, "input": "Hello world.", "response_format": "wav"}
        if model_id == "stabilityai/stable-audio-open-1.0":
            body.update(input="Rain falling on a roof", audio_length=1.0, num_inference_steps=2)
            response = client.send_audio_generate_http_request({"json": body, "timeout": 180})[0]
        else:
            assert model_id == "Qwen/Qwen3-TTS-12Hz-0.6B-Base", f"Unexpected speech deployment: {model}"
            body.update(ref_audio=module.REF_AUDIO_URL, ref_text=module.REF_TEXT, voice="clone", language="English")
            response = client.send_audio_speech_http_request({"json": body, "timeout": 180})[0]
        assert response.status_code == 200, response.error_message
        with wave.open(BytesIO(response.content), "rb") as audio:
            assert audio.getnframes() > 0 and audio.getframerate() > 0


@pytest.fixture
def assert_valid_inference(online_client, omni_server, request):
    return lambda: _assert_valid_inference(online_client, omni_server, request.module)

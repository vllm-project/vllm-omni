# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end online serving test for Bagel text2img and img2img generation.

This test validates that the Bagel model can serve image generation requests
via the OpenAI-compatible chat completions API.

Equivalent to running:
    vllm-omni serve "ByteDance-Seed/BAGEL-7B-MoT" --omni --port 8091

    # text2img
    python3 examples/online_serving/bagel/openai_chat_client.py \\
        --prompt "A cute cat" --modality text2img

    # img2img
    python3 examples/online_serving/bagel/openai_chat_client.py \\
        --prompt "Let the woman wear a blue dress" --modality img2img \\
        --image-url women.jpg
"""

import base64
import os
import signal
import socket
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import requests
from PIL import Image
from vllm.assets.image import ImageAsset

from tests.utils import hardware_test

MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
STAGE_CONFIGS_PATH = str(
    Path(__file__).parent.parent / "offline_inference" / "stage_configs" / "bagel_sharedmemory_ci.yaml"
)

TEXT2IMG_PROMPT = "A cute cat"
IMG2IMG_PROMPT = "Change the grass color to red"


class BagelOmniServer:
    """Context manager to start/stop a vLLM-Omni server for Bagel model tests."""

    def __init__(
        self,
        model: str = MODEL,
        stage_configs_path: str = STAGE_CONFIGS_PATH,
        env_dict: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.stage_configs_path = stage_configs_path
        self.env_dict = env_dict
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        self.port = _find_free_port()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _start_server(self) -> None:
        env = os.environ.copy()
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--stage-configs-path",
            self.stage_configs_path,
            "--stage-init-timeout",
            "300",
        ]

        self.proc = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,
        )

        try:
            if not _wait_for_port(self.host, self.port, timeout=600, proc=self.proc):
                self.terminate()
                raise RuntimeError(f"Server failed to start within 600 seconds on {self.host}:{self.port}")
        except Exception:
            self.terminate()
            raise

    def __enter__(self):
        self._start_server()
        return self

    def terminate(self) -> None:
        if self.proc:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait()
            self.proc = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: int = 600, proc: subprocess.Popen | None = None) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            # Server process exited early
            return False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                if sock.connect_ex((host, port)) == 0:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _send_chat_request(
    server_url: str,
    prompt: str,
    *,
    modality: str = "text2img",
    image: Image.Image | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Send a chat completion request matching the openai_chat_client.py format."""
    content: list[dict[str, Any]] = [{"type": "text", "text": f"<|im_start|>{prompt}<|im_end|>"}]

    if image is not None:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
            }
        )

    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": content}],
    }

    if modality in ("text2img", "img2img"):
        payload["modalities"] = ["image"]

    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_image_from_response(data: dict[str, Any]) -> Image.Image | None:
    """Extract the generated PIL Image from a chat completion response."""
    for choice in data.get("choices", []):
        content = choice.get("message", {}).get("content")
        if isinstance(content, list) and content:
            first_item = content[0]
            if isinstance(first_item, dict) and "image_url" in first_item:
                url = first_item["image_url"].get("url", "")
                if url.startswith("data:image"):
                    _, b64 = url.split(",", 1)
                    return Image.open(BytesIO(base64.b64decode(b64)))
    return None


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
def test_bagel_text2img_online():
    """Test Bagel text2img via OpenAI-compatible chat completions API."""
    with BagelOmniServer() as server:
        response_data = _send_chat_request(
            server.base_url,
            TEXT2IMG_PROMPT,
            modality="text2img",
        )

        image = _extract_image_from_response(response_data)
        assert image is not None, f"No image in response: {response_data}"
        image.load()

        w, h = image.size
        assert w > 0 and h > 0, f"Invalid image size: {image.size}"


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
def test_bagel_img2img_online():
    """Test Bagel img2img via OpenAI-compatible chat completions API."""
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")

    with BagelOmniServer() as server:
        response_data = _send_chat_request(
            server.base_url,
            IMG2IMG_PROMPT,
            modality="img2img",
            image=input_image,
        )

        image = _extract_image_from_response(response_data)
        assert image is not None, f"No image in response: {response_data}"
        image.load()

        w, h = image.size
        assert w > 0 and h > 0, f"Invalid image size: {image.size}"

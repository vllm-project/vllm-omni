# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online serving test for Qwen-Image-Edit-2509 multi-image input.
"""

import base64
import os
import signal
import socket
import subprocess
import sys
import time
from io import BytesIO

import openai
import pytest
from PIL import Image
from vllm.assets.image import ImageAsset
from vllm.utils.network_utils import get_open_port

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Qwen/Qwen-Image-Edit-2509"]
test_params = models


class OmniServer:
    """Omniserver for vLLM-Omni tests."""

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
        self.env_dict = env_dict
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        self.port = get_open_port()

    def _start_server(self) -> None:
        """Start the vLLM-Omni server subprocess."""
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
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
        ] + self.serve_args

        print(f"Launching OmniServer with: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Set working directory to vllm-omni root
            start_new_session=True,
        )

        # Wait for server to be ready
        max_wait = 600  # 10 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((self.host, self.port))
                    if result == 0:
                        print(f"Server ready on {self.host}:{self.port}")
                        return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait()


@pytest.fixture
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights."""
    model = request.param
    with OmniServer(model, ["--num-gpus", "1"]) as server:
        yield server


@pytest.fixture
def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


@pytest.fixture(scope="session")
def base64_encoded_images() -> list[str]:
    """Base64 encoded PNG images for testing."""
    images = [
        ImageAsset("cherry_blossom").pil_image.convert("RGB"),
        ImageAsset("stop_sign").pil_image.convert("RGB"),
    ]
    encoded: list[str] = []
    for img in images:
        with BytesIO() as buffer:
            img.save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return encoded


def dummy_messages_from_image_data(
    image_data_urls: list[str],
    content_text: str = "Combine these two images into one scene.",
):
    """Create messages with image data URLs for OpenAI API."""
    content = [{"type": "text", "text": content_text}]
    for image_url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return [{"role": "user", "content": content}]


def _extract_image_data_url(message_content) -> str:
    assert isinstance(message_content, list) and len(message_content) >= 1
    content_part = message_content[0]
    if isinstance(content_part, dict):
        image_url = content_part.get("image_url", {}).get("url", "")
    else:
        image_url_obj = getattr(content_part, "image_url", None)
        if isinstance(image_url_obj, dict):
            image_url = image_url_obj.get("url", "")
        else:
            image_url = getattr(image_url_obj, "url", "")
    assert isinstance(image_url, str) and image_url
    return image_url


def _decode_data_url_to_image_bytes(data_url: str) -> bytes:
    assert data_url.startswith("data:image")
    _, b64_data = data_url.split(",", 1)
    return base64.b64decode(b64_data)


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_i2i_multi_image_input_qwen_image_edit_2509(
    client: openai.OpenAI,
    omni_server,
    base64_encoded_images: list[str],
) -> None:
    """Test multi-image input editing via OpenAI API."""
    image_data_urls = [f"data:image/png;base64,{img}" for img in base64_encoded_images]
    messages = dummy_messages_from_image_data(image_data_urls)

    height = 832
    width = 1248
    chat_completion = client.chat.completions.create(
        model=omni_server.model,
        messages=messages,
        extra_body={
            "height": height,
            "width": width,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": 42,
        },
    )

    assert len(chat_completion.choices) == 1
    choice = chat_completion.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.role == "assistant"

    image_data_url = _extract_image_data_url(choice.message.content)
    image_bytes = _decode_data_url_to_image_bytes(image_data_url)
    img = Image.open(BytesIO(image_bytes))
    img.load()
    assert img.size == (width, height)

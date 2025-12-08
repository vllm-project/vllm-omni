# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for Qwen2.5-Omni online serving API.

These tests require:
- GPU with sufficient memory (tested on H100-80G)
- Model weights downloaded from HuggingFace

Run with:
    # If model already cached:
    pytest tests/test_qwen2_5_omni_online_server.py -v -s

    # To allow model download:
    VLLM_OMNI_DOWNLOAD_MODEL=1 pytest tests/test_qwen2_5_omni_online_server.py -v -s
"""

from __future__ import annotations

import base64
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
import tempfile
from .utils import get_image_url_from_path
CI_STAGE_CONFIG_PATH = str(Path(__file__).parent / "stage_configs" / "qwen2_5_omni_ci.yaml")


import pytest

# Skip entire module if basic dependencies are missing
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

# Check GPU availability
_HAS_GPU = torch.cuda.is_available()
_GPU_COUNT = torch.cuda.device_count() if _HAS_GPU else 0

# Model configuration
_MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
_SEED = 42


def _check_model_available(model_name: str) -> bool:
    """Check if model weights are available locally or can be downloaded."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        try:
            snapshot_download(model_name, local_files_only=True)
            return True
        except LocalEntryNotFoundError:
            # Model not cached, allow download if env var is set
            return os.environ.get("VLLM_OMNI_DOWNLOAD_MODEL", "0") == "1"
    except ImportError:
        return False


_MODEL_AVAILABLE = _check_model_available(_MODEL_NAME)

# Skip markers
requires_gpu = pytest.mark.skipif(not _HAS_GPU, reason="GPU not available")
requires_model = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason=f"Model {_MODEL_NAME} not available. Set VLLM_OMNI_DOWNLOAD_MODEL=1 to download.",
)
requires_multi_gpu = pytest.mark.skipif(_GPU_COUNT < 2, reason="Multiple GPUs required for full pipeline")

# System prompt for Qwen2.5-Omni
_DEFAULT_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)

# Sampling parameters for the multi-stage pipeline (thinker/talker/code2wav)
_SAMPLING_PARAMS_LIST = [
    {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 10,
        "seed": _SEED,
        "detokenize": True,
        "repetition_penalty": 1.1,
    },
    {
        "temperature": 0.9,
        "top_p": 0.8,
        "top_k": 40,
        "max_tokens": 10,
        "seed": _SEED,
        "detokenize": True,
        "repetition_penalty": 1.05,
        "stop_token_ids": [8294],
    },
    {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 10,
        "seed": _SEED,
        "detokenize": True,
        "repetition_penalty": 1.1,
    },
]


def _create_local_128_image_path() -> str:
    """Create a temporary 128x128 PNG from built-in asset and return its path."""
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image.resize((128, 128)), "RGB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
        image.save(f, format="PNG")
        return f.name


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class _OutputStreamer:
    """Stream subprocess output in a background thread."""

    def __init__(self, process: subprocess.Popen, max_lines: int = 500):
        self.process = process
        self.lines: deque = deque(maxlen=max_lines)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._stream_output, daemon=True)
        self._thread.start()

    def _stream_output(self):
        """Read and print output lines from the process."""
        try:
            for line in iter(self.process.stdout.readline, b""):
                if self._stop_event.is_set():
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                self.lines.append(decoded)
                # Print server logs with prefix
                print(f"[SERVER] {decoded}")
        except Exception as e:
            print(f"[SERVER] Output streaming error: {e}")

    def stop(self):
        """Stop the output streaming."""
        self._stop_event.set()
        self._thread.join(timeout=2)

    def get_output(self) -> str:
        """Get all captured output."""
        return "\n".join(self.lines)


def _wait_for_server(host: str, port: int, timeout: float = 1800) -> bool:
    """Wait for the server to be ready.

    Args:
        timeout: Maximum time to wait in seconds. Default is 1800 (30 minutes)
                 because vLLM-Omni multi-stage pipeline takes a long time to initialize.
    """
    import requests

    start_time = time.time()
    url = f"http://{host}:{port}/health"

    while time.time() - start_time < timeout:
        elapsed = int(time.time() - start_time)
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"\n[TEST] Server ready after {elapsed}s")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    print(f"\n[TEST] Server failed to start after {timeout}s")
    return False


@pytest.fixture(scope="module")
def omni_server():
    """Start vLLM-Omni server as a subprocess with actual model weights.

    Uses module scope so the server starts only once for all tests.
    Multi-stage initialization can take 10-20+ minutes.
    """
    port = _find_free_port()
    host = "127.0.0.1"

    # Build the server command
    cmd = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        _MODEL_NAME,
        "--omni",
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.8",
        "--load-format",
        "dummy",
        # "--stage-configs-path",
        # CI_STAGE_CONFIG_PATH,
    ]

    print(f"\n{'=' * 60}")
    print(f"Starting vLLM-Omni server on {host}:{port}")
    print("Multi-stage init may take 10-20+ minutes...")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")

    # Start server process
    env = os.environ.copy()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,  # Create new process group for cleanup
    )

    # Start streaming server output in background
    streamer = _OutputStreamer(process)

    try:
        # Wait for server to be ready (30 min timeout for multi-stage init)
        if not _wait_for_server(host, port, timeout=1800):
            streamer.stop()
            process.terminate()
            process.wait(timeout=10)
            raise RuntimeError("Server failed to start within 30 minute timeout.\nSee server logs above for details.")

        print(f"\n{'=' * 60}")
        print(f"Server ready at http://{host}:{port}")
        print(f"{'=' * 60}\n")

        yield {
            "host": host,
            "port": port,
            "base_url": f"http://{host}:{port}",
            "process": process,
            "streamer": streamer,
        }

    finally:
        # Cleanup: terminate the process group
        print("\nShutting down server...")
        streamer.stop()
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=30)
            print("Server shut down successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            process.kill()

@requires_gpu
@requires_model
class TestQwen25OmniOnlineServingWithOpenAIClient:
    """Tests using the official OpenAI Python client."""

    def test_openai_client_text(self, omni_server):
        """Test using OpenAI Python client for text chat."""
        from openai import OpenAI

        client = OpenAI(
            api_key="EMPTY",
            base_url=f"{omni_server['base_url']}/v1",
        )

        response = client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": _DEFAULT_SYSTEM},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            temperature=0.0,
            max_tokens=32,
            seed=_SEED,
            extra_body={"sampling_params_list": _SAMPLING_PARAMS_LIST},
        )

        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    def test_openai_client_image(self, omni_server):
        """Test using OpenAI Python client with image input."""
        from openai import OpenAI

        client = OpenAI(
            api_key="EMPTY",
            base_url=f"{omni_server['base_url']}/v1",
        )

        # Create a local 128x128 PNG and encode to data URL to avoid external fetch
        image_path = _create_local_128_image_path()
        image_url = get_image_url_from_path(image_path)
        os.remove(image_path)

        response = client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": _DEFAULT_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "What is this?"},
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=64,
            seed=_SEED,
            extra_body={"sampling_params_list": _SAMPLING_PARAMS_LIST},
        )

        assert response.choices is not None
        assert len(response.choices) > 0

        content = response.choices[0].message.content
        print(f"Image response: {content}")
        assert len(content) > 5

    def test_openai_client_list_models(self, omni_server):
        """Test listing models via OpenAI Python client."""
        from openai import OpenAI

        client = OpenAI(
            api_key="EMPTY",
            base_url=f"{omni_server['base_url']}/v1",
        )

        models = client.models.list()
        model_ids = [m.id for m in models.data]

        assert len(model_ids) > 0
        print(f"Models via OpenAI client: {model_ids}")

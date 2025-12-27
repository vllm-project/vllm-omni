# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import concurrent.futures
import ctypes
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import openai
import pytest
from vllm.assets.video import VideoAsset
from vllm.utils import get_open_port

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

# CI stage config for 2*H100-80G GPUs
stage_configs = [str(Path(__file__).parent / "stage_configs" / "qwen3_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


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

        # Helper to ensure child process dies when parent dies
        libc = ctypes.CDLL("libc.so.6")

        def preexec_fn():
            # Ensure the child process receives SIGTERM when the parent (this test runner) dies.
            # This prevents orphaned processes if the test is killed unexpectedly.
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)

        print(f"Launching OmniServer with: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Set working directory to vllm-omni root
            start_new_session=True,
            preexec_fn=preexec_fn,
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
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Uses module scope so the server starts only once for all tests.
    Multi-stage initialization can take 10-20+ minutes.
    """
    model, stage_config_path = request.param
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        yield server


@pytest.fixture
def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


@pytest.fixture(scope="session")
def base64_encoded_video() -> str:
    """Base64 encoded video for testing."""
    import base64

    video = VideoAsset(name="baby_reading", num_frames=4)
    with open(video.video_path, "rb") as f:
        content = f.read()
        return base64.b64encode(content).decode("utf-8")


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def dummy_messages_from_video_data(
    video_data_url: str,
    content_text: str = "Describe the video briefly.",
):
    """Create messages with video data URL for OpenAI API."""
    return [
        get_system_prompt(),
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_data_url}},
                {"type": "text", "text": content_text},
            ],
        },
    ]


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_audio_concurrent(
    client: openai.OpenAI,
    omni_server,
    base64_encoded_video: str,
) -> None:
    """Test processing video with multiple concurrent completions, generating audio output via OpenAI API."""
    # Create data URL for the base64 encoded video
    video_data_url = f"data:video/mp4;base64,{base64_encoded_video}"

    messages = dummy_messages_from_video_data(video_data_url)

    # Test multiple concurrent completions
    num_concurrent_requests = 5

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
        # Submit multiple completion requests concurrently
        futures = [
            executor.submit(
                client.chat.completions.create,
                model=omni_server.model,
                messages=messages,
            )
            for _ in range(num_concurrent_requests)
        ]

        # Wait for all requests to complete and collect results
        chat_completions = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Verify all completions succeeded
    assert len(chat_completions) == num_concurrent_requests

    for chat_completion in chat_completions:
        assert len(chat_completion.choices) == 2  # 1 for text output, 1 for audio output

        # Verify text output
        text_choice = chat_completion.choices[0]
        assert text_choice.finish_reason == "length"

        # Verify we got a response
        text_message = text_choice.message
        assert text_message.content is not None and len(text_message.content) >= 10
        assert text_message.role == "assistant"

        # Verify audio output
        audio_choice = chat_completion.choices[1]
        assert audio_choice.finish_reason == "stop"
        audio_message = audio_choice.message

        # Check if audio was generated
        if hasattr(audio_message, "audio") and audio_message.audio:
            assert audio_message.audio.data is not None
            assert len(audio_message.audio.data) > 0

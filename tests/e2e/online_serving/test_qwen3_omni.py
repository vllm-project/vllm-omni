# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import base64
import concurrent.futures
import time
from pathlib import Path

import openai
import pytest
from vllm.assets.video import VideoAsset

from tests.conftest import (
    OmniServer,
    convert_audio_to_text,
    cosine_similarity_text,
    dummy_messages_from_mix_data,
    modify_stage_config,
)
from vllm_omni.utils import is_rocm

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

# CI stage config for 2xH100-80G GPUs or AMD GPU MI325
if is_rocm():
    # ROCm stage config optimized for MI325 GPU
    stage_configs = [str(Path(__file__).parent / "stage_configs" / "rocm" / "qwen3_omni_ci.yaml")]
else:
    stage_configs = [str(Path(__file__).parent / "stage_configs" / "qwen3_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


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


def omni_client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


@pytest.fixture(scope="session")
def base64_encoded_video() -> str:
    """Base64 encoded video for testing."""
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


def get_prompt(prompt_type="text_only"):
    prompts = {
        "text_only": "What is the capital of China?",
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def get_max_batch_size(size_type="few"):
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


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

    # Test streaming completion
    chat_completion = client.chat.completions.create(
        model=omni_server.model,
        messages=messages,
        stream=True,
    )

    # Collect text and audio data from stream
    text_content = ""
    audio_data = None

    for chunk in chat_completion:
        for choice in chunk.choices:
            if hasattr(choice, "delta"):
                content = getattr(choice.delta, "content", None)
            else:
                content = None

            modality = getattr(chunk, "modality", None)

            if modality == "audio" and content:
                # Audio chunk - decode base64 content
                if audio_data is None:
                    audio_data = base64.b64decode(content)
                else:
                    audio_data += base64.b64decode(content)
            elif modality == "text" and content:
                # Text chunk - accumulate text content
                text_content += content if content else ""

    # Verify text output
    assert text_content is not None and len(text_content) >= 2

    # Verify audio output
    assert audio_data is not None
    assert len(audio_data) > 0


@pytest.mark.skipif(is_rocm(), reason="Test skipped on AMD environment due to known output issues")
@pytest.mark.parametrize("test_config", test_params)
def test_text_to_text_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text and audio output via OpenAI API."""

    model, stage_config_path = test_config
    # TODO: Use this stage_config_path uniformly after modifying existing test cases.
    if not is_rocm():
        stage_config_path = str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")
    num_concurrent_requests = get_max_batch_size()
    stage_config_path = modify_stage_config(
        stage_config_path,
        {
            0: {"runtime.max_batch_size": num_concurrent_requests},
            1: {"runtime.max_batch_size": num_concurrent_requests},
        },
    )
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

        # Test single completion
        api_client = omni_client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(api_client.chat.completions.create, model=server.model, messages=messages)
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert "beijing" in text_choice.message.content.lower(), "The output do not contain keywords."

            # Verify text output same as audio output
            audio_content = convert_audio_to_text(audio_data)
            print(f"text content is: {text_content}")
            print(f"audio content is: {audio_content}")
            similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
            print(f"The similarity between audio and text is: {similarity}")
            assert similarity > 0.9, "The audio content is not same as the text"

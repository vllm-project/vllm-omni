# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import concurrent.futures
import time
from pathlib import Path

import openai
import pytest

from tests.conftest import (
    OmniServer,
    convert_audio_to_text,
    cosine_similarity_text,
    dummy_messages_from_mix_data,
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
)
from vllm_omni.utils import is_rocm

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

# CI stage config for 2xH100-80G GPUs or AMD GPU MI325
if is_rocm():
    # ROCm stage config optimized for MI325 GPU
    stage_configs = [str(Path(__file__).parent / "stage_configs" / "rocm" / "qwen3_omni_ci.yaml")]
else:
    stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Uses session scope so the server starts only once for the entire test session.
    Multi-stage initialization can take 10-20+ minutes.
    """
    model, stage_config_path = request.param

    print(f"Starting OmniServer with model: {model}")
    print("This may take 10-20+ minutes for initialization...")

    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "120"]) as server:
        print("OmniServer started successfully")
        yield server
        print("OmniServer stopped")


@pytest.fixture
def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


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


@pytest.mark.skipif(is_rocm(), reason="Test skipped on AMD environment due to known output issues")
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mix_to_text_audio_001(client: openai.OpenAI, omni_server) -> None:
    """
    Test multi-modal input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text + audio + video + image
    Output Modal: text + audio
    Input Setting: stream=True
    Datasets: single request
    """

    # Test single completion
    e2e_list = list()
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text=get_prompt("mix"),
    )

    # Test single completion
    start_time = time.perf_counter()
    chat_completion = client.chat.completions.create(model=omni_server.model, messages=messages, stream=True)

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
                # Audio chunk - content
                if audio_data is None:
                    audio_data = content
                else:
                    audio_data += content
            elif modality == "text" and content:
                # Text chunk - accumulate text content
                text_content += content if content else ""

    # Verify E2E
    current_e2e = time.perf_counter() - start_time
    print(f"the request e2e is: {current_e2e}")
    # TODO: Verify the E2E latency after confirmation baseline.
    e2e_list.append(current_e2e)

    print(f"the avg e2e is: {sum(e2e_list) / len(e2e_list)}")
    # Verify all completions succeeded
    assert audio_data is not None, "No audio output is generated"

    # Verify text output success
    assert text_content is not None and len(text_content) >= 2, "No text output is generated"
    assert any(
        keyword in text_content.lower() for keyword in ["square", "quadrate", "sphere", "globe", "circle", "round"]
    ), "The output does not contain any of the keywords."

    # Verify text output same as audio output
    audio_content = convert_audio_to_text(audio_data)
    print(f"text content is: {text_content}")
    print(f"audio content is: {audio_content}")
    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())
    print(f"similarity is: {similarity}")
    assert similarity > 0.9, "The audio content is not same as the text"


@pytest.mark.skipif(is_rocm(), reason="Test skipped on AMD environment due to known output issues")
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_audio_001(client: openai.OpenAI, omni_server) -> None:
    """
    Test text input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: text + audio
    Datasets: few requests
    """

    num_concurrent_requests = get_max_batch_size()
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    e2e_list = list()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
        # Submit multiple completion requests concurrently
        futures = [
            executor.submit(client.chat.completions.create, model=omni_server.model, messages=messages)
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
        print(f"similarity is: {similarity}")
        assert similarity > 0.9, "The audio content is not same as the text"

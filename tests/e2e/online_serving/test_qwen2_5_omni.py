# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import os
from pathlib import Path

import openai
import pytest
import time
import concurrent.futures

from tests.conftest import (OmniServer, dummy_messages_from_mix_data,
                            modify_stage_config, generate_synthetic_video,
                            generate_synthetic_audio, generate_synthetic_image)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Qwen/Qwen2.5-Omni-7B"]

# CI stage config optimized for 24GB GPU (L4/RTX3090) or NPU
stage_configs = [
    str(
        Path(__file__).parent.parent / "stage_configs" /
        "qwen2_5_omni_ci.yaml")
]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models
               for stage_config in stage_configs]


def get_system_prompt():
    return {
        "role":
        "system",
        "content": [{
            "type":
            "text",
            "text":
            ("You are Qwen, a virtual human developed by the Qwen Team, "
             "Alibaba Group, capable of perceiving auditory and visual inputs, "
             "as well as generating text and speech."),
        }],
    }


def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


@pytest.mark.ci
@pytest.mark.L4_2
@pytest.mark.parametrize("test_config", test_params)
def test_mixed_modalities_to_text_audio(test_config: tuple[str, str]) -> None:
    """Test processing video,audio,image,text, generating audio,text output via OpenAI API."""

    model, stage_config_path = test_config
    with OmniServer(model, [
            "--stage-configs-path", stage_config_path, "--init-sleep-seconds",
            "90"
    ]) as server:
        # Create data URL for the base64 encoded video
        video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(128,128,4)}"

        # Create data URL for the base64 encoded audio
        audio_data_url = f"data:audio/ogg;base64,{generate_synthetic_audio(3,1)}"

        # Create data URL for the base64 encoded image
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(128,128)}"

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            video_data_url=video_data_url,
            audio_data_url=audio_data_url,
            image_data_url=image_data_url)

        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=10, stop=None)

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens == 10, "The output length differs from the requested max_tokens."

        # Verify audio output success
        audio_message = chat_completion.choices[1].message
        assert audio_message.audio.data is not None, "No audio output is generated"
        assert audio_message.audio.expires_at > time.time(
        ), "The generated audio has expired."

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter()-start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.

        # Verify audio data
        # TODO: Implement similarity validation between audio content and text.


@pytest.mark.ci
@pytest.mark.L4_2
@pytest.mark.parametrize("test_config", test_params)
def test_text_audio_to_text(test_config: tuple[str, str]) -> None:
    """Test processing audio,text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    num_concurrent_requests = 5

    stage_config_path = modify_stage_config(
        stage_config_path,
        {1: {
            "runtime.max_batch_size": num_concurrent_requests
        }})
    with OmniServer(model, [
            "--stage-configs-path", stage_config_path, "--init-sleep-seconds",
            "90"
    ]) as server:
        """Test processing video, generating audio output via OpenAI API."""

        # Create data URL for the base64 encoded audio
        audio_data_url = f"data:audio/ogg;base64,{generate_synthetic_audio(3,2)}"

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            content_text="Describe the audio briefly.",
            audio_data_url=audio_data_url)

        # Test 5 completion
        api_client = client(server)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    modalities=["text"],
                    max_tokens=10,
                    stop=None,
                ) for _ in range(num_concurrent_requests)
            ]

            # Wait for all requests to complete and collect results
            chat_completions = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]

        # Verify all completions succeeded
        assert len(chat_completions
                   ) == num_concurrent_requests, "Not all requests succeeded."

        for chat_completion in chat_completions:
            # Verify only output text
            assert len(
                chat_completion.choices
            ) == 1, "The generated content includes more than just text."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, \
                "The output length differs from the requested max_tokens."

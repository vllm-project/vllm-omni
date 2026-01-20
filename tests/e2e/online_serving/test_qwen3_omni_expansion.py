# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import time
from pathlib import Path

import openai
import pytest

from tests.conftest import (
    OmniServer,
    dummy_messages_from_mix_data,
)

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

# CI stage config for 2*H100-80G GPUs
stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


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


def get_prompt(prompt_type="text_only"):
    prompts = {
        "text_only": "What is the capital of China?",
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def get_max_batch_size(size_type="few"):
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


@pytest.mark.parametrize("test_config", test_params)
def test_text_to_text_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=20, modalities=["text"]
        )
        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.

        # Verify only output text
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens <= 20, "The output length more than the requested max_tokens."
        assert "beijing" in text_choice.message.content.lower(), "The output do not contain keywords."

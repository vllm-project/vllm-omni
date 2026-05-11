"""
E2E offline tests for Raon model with text input and audio output.

Raon is a 2-stage streaming pipeline:
  Stage 0: AR text + codec generation (thinker-talker fused)
  Stage 1: Audio codec-to-waveform decode
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test

# Model identifier – placeholder for local checkpoint path.
models = ["/path/to/Raon"]

# CI stage config optimized for a single 24GB GPU (L4/RTX3090/RTX4090).
stage_config = str(Path(__file__).parent.parent / "stage_configs" / "raon_ci.yaml")

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models]


def get_question(prompt_type="tts"):
    prompts = {
        "tts": "Hello, this is a test of the Raon text-to-speech pipeline.",
    }
    return prompts.get(prompt_type, prompts["tts"])


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio(omni_runner, omni_runner_handler) -> None:
    """
    Test text input processing and audio output generation via offline inference.
    Deploy Setting: CI yaml (single GPU, async_chunk disabled)
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "prompts": get_question("tts"),
        "modalities": ["audio"],
    }

    # Test single completion
    omni_runner_handler.send_request(request_config)

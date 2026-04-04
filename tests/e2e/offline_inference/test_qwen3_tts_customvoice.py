# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E offline tests for Qwen3-TTS CustomVoice model with text input and audio output.

Async_chunk disable, cuda_graph disabled (no_async_chunk stage config).
CUDA graph is disabled by setting engine_args.enforce_eager=true via modify_stage_config().
Same structure as test_qwen3_omni (models, stage_configs, test_params, parametrize omni_runner).
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import pytest

from tests.conftest import modify_stage_config
from tests.utils import hardware_test

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def get_cuda_graph_config():
    path = modify_stage_config(
        get_stage_config(),
        updates={
            "stage_args": {
                0: {
                    "engine_args.enforce_eager": "true",
                },
                1: {"engine_args.enforce_eager": "true"},
            },
        },
    )
    return path


def get_stage_config(name: str = "qwen3_tts_no_async_chunk.yaml"):
    """Get the no_async_chunk stage config path (async_chunk disable, cuda_graph disabled)."""
    return str(Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / name)


# Same structure as test_qwen3_omni: models, stage_configs, test_params
tts_server_params = [
    pytest.param(
        (MODEL, get_cuda_graph_config()),
        id="no_cuda_graph",
    )
]


def get_prompt():
    """Text prompt for text-to-audio (same role as get_question in test_qwen3_omni)."""
    return "Hello, this is a test for text to audio."


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_runner, omni_runner_handler) -> None:
    """
    Test text input processing and audio output via offline Omni runner.
    Deploy Setting: qwen3_tts_no_async_chunk.yaml + enforce_eager=true
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: few requests
    """
    request_config = {"input": get_prompt(), "voice": "vivian"}
    omni_runner_handler.send_audio_speech_request(request_config)

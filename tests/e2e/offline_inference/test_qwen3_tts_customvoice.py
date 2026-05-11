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

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def get_cuda_graph_config():
    """Build a temp deploy yaml mirroring the deleted qwen3_tts_no_async_chunk.yaml.

    Composes the synchronous (no-async-chunk) variant on top of the bundled
    qwen3_tts.yaml prod default, with cudagraphs disabled. Replaces the deleted
    standalone variant yaml; same effective config, no checked-in file needed.
    """
    return modify_stage_config(
        get_deploy_config_path("qwen3_tts.yaml"),
        updates={
            "async_chunk": False,
            "stages": {
                0: {
                    "max_num_seqs": 1,
                    "gpu_memory_utilization": 0.2,
                    "enforce_eager": True,
                    "async_scheduling": False,
                },
                1: {
                    "gpu_memory_utilization": 0.2,
                    "enforce_eager": True,
                    "async_scheduling": False,
                },
            },
        },
    )


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
@pytest.mark.tts
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

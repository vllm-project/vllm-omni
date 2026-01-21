# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E offline tests for Omni model with video input and audio output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from pathlib import Path

import pytest
from vllm.assets.video import VideoAsset

from vllm_omni.utils import is_rocm

from .conftest import OmniRunner

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

# CI stage config for 2xH100-80G GPUs or AMD GPU MI325
if is_rocm():
    # ROCm stage config optimized for MI325 GPU
    stage_configs = [str(Path(__file__).parent / "stage_configs" / "rocm" / "qwen3_omni_ci.yaml")]
else:
    stage_configs = [str(Path(__file__).parent / "stage_configs" / "qwen3_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


@pytest.mark.parametrize("test_config", test_params)
def test_video_to_audio(omni_runner: type[OmniRunner], test_config) -> None:
    """Test processing video, generating audio output."""
    model, stage_config_path = test_config
    with omni_runner(model, seed=42, stage_configs_path=stage_config_path, stage_init_timeout=300) as runner:
        # Prepare inputs
        question = "Describe the video briefly."
        video = VideoAsset(name="baby_reading", num_frames=4).np_ndarrays

        outputs = runner.generate_multimodal(
            prompts=question,
            videos=video,
        )

        # Find and verify text output (thinker stage)
        text_output = None
        output_count = 0
        for stage_output in outputs:
            if stage_output.final_output_type == "text":
                text_output = stage_output
                output_count += 1
                break

        assert output_count > 0
        assert text_output is not None
        assert len(text_output.request_output) > 0
        text_content = text_output.request_output[0].outputs[0].text
        assert text_content is not None
        assert len(text_content.strip()) > 0

        # Find and verify audio output (code2wav stage)
        audio_output = None
        output_count = 0
        for stage_output in outputs:
            if stage_output.final_output_type == "audio":
                audio_output = stage_output
                output_count += 1
                break

        assert output_count > 0
        assert audio_output is not None
        assert len(audio_output.request_output) > 0

        # Verify audio tensor exists and has content
        audio_tensor = audio_output.request_output[0].multimodal_output["audio"]
        assert audio_tensor is not None
        assert audio_tensor.numel() > 0

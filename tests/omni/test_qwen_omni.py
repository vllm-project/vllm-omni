# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E tests for Qwen2.5-Omni model with mixed modality inputs and audio output.
"""

import os
from pathlib import Path

import pytest
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.multimodal.image import convert_image_mode
from vllm.sampling_params import SamplingParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Qwen/Qwen2.5-Omni-3B"]

# CI stage config optimized for 24GB GPU (L4/RTX3090)
CI_STAGE_CONFIG_PATH = str(Path(__file__).parent / "stage_configs" / "qwen2_5_omni_ci.yaml")


@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("max_tokens", [2048])
def test_mixed_modalities_to_audio(omni_runner, model: str, max_tokens: int) -> None:
    """Test processing audio, image, and video together, generating audio output."""
    with omni_runner(model, seed=42, stage_configs_path=CI_STAGE_CONFIG_PATH) as runner:
        # Prepare multimodal inputs
        question = "What is recited in the audio? What is in this image? Describe the video briefly."
        audio = AudioAsset("mary_had_lamb").audio_and_sample_rate
        image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
        video = VideoAsset(name="baby_reading", num_frames=16).np_ndarrays

        thinker_sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic - no randomness
            top_p=1.0,  # Disable nucleus sampling
            top_k=-1,  # Disable top-k sampling
            max_tokens=2048,
            seed=42,  # Fixed seed for sampling
            detokenize=True,
            repetition_penalty=1.1,
        )
        talker_sampling_params = SamplingParams(
            temperature=0.9,
            top_p=0.8,
            top_k=40,
            max_tokens=2048,
            seed=42,  # Fixed seed for sampling
            detokenize=True,
            repetition_penalty=1.05,
            stop_token_ids=[8294],
        )
        code2wav_sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic - no randomness
            top_p=1.0,  # Disable nucleus sampling
            top_k=-1,  # Disable top-k sampling
            max_tokens=2048,
            seed=42,  # Fixed seed for sampling
            detokenize=True,
            repetition_penalty=1.1,
        )

        sampling_params_list = [
            thinker_sampling_params,
            talker_sampling_params,
            code2wav_sampling_params,
        ]

        outputs = runner.generate_multimodal(
            prompts=question,
            audios=audio,
            images=image,
            videos=video,
            sampling_params_list=sampling_params_list,
        )

        # Verify we got outputs from multiple stages
        assert len(outputs) > 0

        # Find and verify text output (thinker stage)
        text_output = None
        for stage_output in outputs:
            if stage_output.final_output_type == "text":
                text_output = stage_output
                break

        assert text_output is not None
        assert len(text_output.request_output) > 0
        text_content = text_output.request_output[0].outputs[0].text
        assert text_content is not None
        assert len(text_content.strip()) > 0

        # Find and verify audio output (code2wav stage)
        audio_output = None
        for stage_output in outputs:
            if stage_output.final_output_type == "audio":
                audio_output = stage_output
                break

        assert audio_output is not None
        assert len(audio_output.request_output) > 0

        # Verify audio tensor exists and has content
        audio_tensor = audio_output.request_output[0].multimodal_output["audio"]
        assert audio_tensor is not None
        assert audio_tensor.numel() > 0

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E tests for Qwen2.5-Omni model with mixed modality inputs and audio output.
"""

from pathlib import Path

import pytest
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.envs import VLLM_USE_MODELSCOPE
from vllm.multimodal.image import convert_image_mode

from tests.utils import create_new_process_for_each_test, hardware_test
from vllm_omni.platforms import current_omni_platform

from .conftest import OmniRunner

models = ["Qwen/Qwen2.5-Omni-3B"]

# CI stage config optimized for 24GB GPU (L4/RTX3090) or NPU
if current_omni_platform.is_npu():
    stage_config = str(Path(__file__).parent / "stage_configs" / "npu" / "qwen2_5_omni_ci.yaml")
elif current_omni_platform.is_rocm():
    # ROCm stage config optimized for MI325 GPU
    stage_config = str(Path(__file__).parent / "stage_configs" / "rocm" / "qwen2_5_omni_ci.yaml")
else:
    stage_config = str(Path(__file__).parent / "stage_configs" / "qwen2_5_omni_ci.yaml")

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("test_config", test_params)
def test_mixed_modalities_to_audio(omni_runner: type[OmniRunner], test_config: tuple[str, str]) -> None:
    """Test processing audio, image, and video together, generating audio output."""
    model, stage_config_path = test_config
    with omni_runner(model, seed=42, stage_configs_path=stage_config_path) as runner:
        # Prepare multimodal inputs
        question = "What is recited in the audio? What is in this image? Describe the video briefly."
        audio = AudioAsset("mary_had_lamb").audio_and_sample_rate
        audio = (audio[0][: 16000 * 5], audio[1])  # Trim to first 5 seconds
        image = convert_image_mode(ImageAsset("cherry_blossom").pil_image.resize((128, 128)), "RGB")
        if not VLLM_USE_MODELSCOPE:
            video = VideoAsset(name="baby_reading", num_frames=4).np_ndarrays
        else:
            # modelscope can't access raushan-testing-hf/videos-test, skip video input temporarily
            video = None

        outputs = runner.generate_multimodal(
            prompts=question,
            audios=audio,
            images=image,
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
        audio_tensor = audio_output.request_output[0].outputs[0].multimodal_output["audio"]
        assert audio_tensor is not None
        assert audio_tensor.numel() > 0


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("test_config", test_params)
def test_mixed_modalities_to_text_only(omni_runner: type[OmniRunner], test_config: tuple[str, str]) -> None:
    """Test processing audio, image, and video together, generating audio output."""
    model, stage_config_path = test_config
    with omni_runner(model, seed=42, stage_configs_path=stage_config_path) as runner:
        # Prepare multimodal inputs
        question = "What is recited in the audio? What is in this image? Describe the video briefly."
        audio = AudioAsset("mary_had_lamb").audio_and_sample_rate
        audio = (audio[0][: 16000 * 5], audio[1])  # Trim to first 5 seconds
        image = convert_image_mode(ImageAsset("cherry_blossom").pil_image.resize((128, 128)), "RGB")
        video = VideoAsset(name="baby_reading", num_frames=4).np_ndarrays
        modalities = ["text"]

        outputs = runner.generate_multimodal(
            prompts=question,
            audios=audio,
            images=image,
            videos=video,
            modalities=modalities,
        )

        # Find and verify text output (thinker stage)
        text_output = None
        output_count = 0
        for stage_output in outputs:
            assert stage_output.final_output_type != "audio"
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

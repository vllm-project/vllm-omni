# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E tests for Step-Audio2 model with audio input and audio output.

These tests verify:
1. Processor loads correctly and handles audio input
2. Field configuration for vLLM batching (flat_from_sizes)
3. Tensor shapes are correct (3D not 4D for profiling)
4. Basic E2E inference works with audio input
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from .conftest import OmniRunner
from .utils import create_new_process_for_each_test

models = ["stepfun-ai/Step-Audio-2-mini"]

stage_configs = [str(Path(__file__).parent / "stage_configs" / "step_audio2_ci.yaml")]

test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


def create_dummy_audio(sample_rate: int = 16000, duration_sec: float = 1.0) -> tuple[np.ndarray, int]:
    """Create a dummy audio signal (sine wave) for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio, sample_rate


class StepAudio2OmniRunner(OmniRunner):
    """Test runner specialized for Step-Audio2 model."""

    def get_omni_inputs(
        self,
        prompts: list[str] | str,
        system_prompt: str | None = None,
        audios=None,
        **kwargs,
    ) -> list[dict]:
        """Construct Omni input format for Step-Audio2."""
        if system_prompt is None:
            system_prompt = "你是一个语音对话助手，能够理解音频输入并生成语音回复。"

        audio_placeholder = "<audio_patch>"

        if isinstance(prompts, str):
            prompts = [prompts]

        num_prompts = len(prompts)

        if audios is None:
            audios_list = [None] * num_prompts
        elif isinstance(audios, list):
            audios_list = audios if len(audios) == num_prompts else [audios[0]] * num_prompts
        else:
            audios_list = [audios] * num_prompts

        omni_inputs = []
        for i, prompt_text in enumerate(prompts):
            user_content = ""
            multi_modal_data = {}

            audio = audios_list[i]
            if audio is not None:
                user_content += audio_placeholder
                multi_modal_data["audio"] = audio

            user_content += prompt_text

        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        input_dict = {"prompt": full_prompt}
        if multi_modal_data:
            input_dict["multi_modal_data"] = multi_modal_data

        omni_inputs.append(input_dict)

        return omni_inputs


@pytest.fixture(scope="session")
def step_audio2_runner():
    return StepAudio2OmniRunner


@pytest.mark.core_model
def test_processor_loads_correctly():
    """Test that Step-Audio2 processor can be imported and initialized."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        StepAudio2MultiModalProcessor,
        StepAudio2ProcessingInfo,
        StepAudio2Processor,
    )

    assert StepAudio2MultiModalProcessor is not None
    assert StepAudio2ProcessingInfo is not None
    assert StepAudio2Processor is not None


@pytest.mark.core_model
def test_audio_preprocessing():
    """Test that audio preprocessing produces correct tensor shapes."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        log_mel_spectrogram,
        padding_mels,
    )

    audio, sample_rate = create_dummy_audio(sample_rate=16000, duration_sec=1.0)

    mel = log_mel_spectrogram(audio)

    assert mel.ndim == 2, f"Expected 2D tensor, got {mel.ndim}D"
    assert mel.shape[0] == 128, f"Expected 128 mel bins, got {mel.shape[0]}"

    mels = [mel, mel]
    padded_mels, mel_lens = padding_mels(mels)

    assert padded_mels.ndim == 3, f"Expected 3D tensor, got {padded_mels.ndim}D"
    assert padded_mels.shape[0] == 2, f"Expected batch size 2, got {padded_mels.shape[0]}"
    assert padded_mels.shape[1] == 128, f"Expected 128 mel bins, got {padded_mels.shape[1]}"


@pytest.mark.core_model
def test_feature_length_calculation():
    """Test that audio feature length calculation is correct."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        calculate_audio_feature_length,
    )

    assert calculate_audio_feature_length(1000) == 125

    assert calculate_audio_feature_length(100) > 0

    assert calculate_audio_feature_length(1) >= 1


@pytest.mark.core_model
def test_token_config_constants():
    """Test that token configuration constants are set correctly."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import (
        DEFAULT_TOKEN_CONFIG,
        STEP_AUDIO2_AUDIO_END,
        STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID,
        STEP_AUDIO2_AUDIO_START,
        STEP_AUDIO2_AUDIO_VOCAB_SIZE,
        STEP_AUDIO2_TEXT_MAX,
    )

    assert STEP_AUDIO2_TEXT_MAX < STEP_AUDIO2_AUDIO_START, "Text tokens should come before audio tokens"
    assert STEP_AUDIO2_AUDIO_END == STEP_AUDIO2_AUDIO_START + STEP_AUDIO2_AUDIO_VOCAB_SIZE - 1
    assert STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID > STEP_AUDIO2_TEXT_MAX
    assert STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID < STEP_AUDIO2_AUDIO_START

    assert DEFAULT_TOKEN_CONFIG.text_max == STEP_AUDIO2_TEXT_MAX
    assert DEFAULT_TOKEN_CONFIG.audio_start == STEP_AUDIO2_AUDIO_START


@pytest.mark.core_model
def test_mm_field_config_structure():
    """Test that multimodal field config is properly structured for vLLM batching."""
    from vllm.multimodal.inputs import MultiModalFieldConfig

    audio_lens = torch.tensor([100, 200, 150], dtype=torch.int32)

    field_config = MultiModalFieldConfig.flat_from_sizes("audio", audio_lens)

    assert field_config is not None


@pytest.mark.core_model
def test_audio_encoder_output_shape():
    """Test that audio encoder produces correct output shapes."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        Adaptor,
        AudioEncoder,
    )

    encoder = AudioEncoder(n_mels=128, n_ctx=1500, n_state=512, n_head=8, n_layer=6)

    batch_size = 2
    n_mels = 128
    time_steps = 400
    x = torch.randn(batch_size, n_mels, time_steps)
    x_len = torch.tensor([time_steps, time_steps // 2], dtype=torch.int32)

    encoded, encoded_lens = encoder(x, x_len)

    assert encoded.ndim == 3, f"Expected 3D tensor, got {encoded.ndim}D"
    assert encoded.shape[0] == batch_size
    assert encoded.shape[2] == 512

    adapter = Adaptor(n_state=512, n_hidden=4096, kernel_size=3, stride=2)
    adapted = adapter(encoded)

    assert adapted.ndim == 3
    assert adapted.shape[0] == batch_size
    assert adapted.shape[2] == 4096


@pytest.mark.core_model
def test_token_separation():
    """Test that token separation works correctly."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        StepAudio2ThinkerForConditionalGeneration,
    )

    token_ids = [100, 200, 151700, 300, 151800, 400]

    text_tokens, audio_tokens = StepAudio2ThinkerForConditionalGeneration.separate_tokens(token_ids)

    assert text_tokens == [100, 200, 300, 400]
    assert audio_tokens == [4, 104]


@pytest.mark.core_model
def test_has_audio_output():
    """Test detection of audio tokens in output."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        StepAudio2ThinkerForConditionalGeneration,
    )

    text_only = [100, 200, 300]
    assert not StepAudio2ThinkerForConditionalGeneration.has_audio_output(text_only)

    with_audio = [100, 200, 151700, 300]
    assert StepAudio2ThinkerForConditionalGeneration.has_audio_output(with_audio)


@pytest.mark.core_model
@pytest.mark.parametrize("test_config", test_params)
@create_new_process_for_each_test()
def test_audio_to_text_and_audio(step_audio2_runner: type[StepAudio2OmniRunner], test_config: tuple[str, str]) -> None:
    """Test processing audio input and generating text + audio output."""
    model, stage_config_path = test_config

    with step_audio2_runner(model, seed=42, stage_configs_path=stage_config_path) as runner:
        audio = create_dummy_audio(sample_rate=16000, duration_sec=2.0)

        outputs = runner.generate_multimodal(
            prompts="请复述这段音频的内容。",
            audios=audio,
        )

        assert len(outputs) > 0

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

        audio_output = None
        for stage_output in outputs:
            if stage_output.final_output_type == "audio":
                audio_output = stage_output
                break

        if audio_output is not None:
            assert len(audio_output.request_output) > 0
            audio_tensor = audio_output.request_output[0].outputs[0].multimodal_output.get("audio")
            if audio_tensor is not None:
                assert audio_tensor.numel() > 0


@pytest.mark.core_model
@pytest.mark.parametrize("test_config", test_params)
@create_new_process_for_each_test()
def test_text_only_input(step_audio2_runner: type[StepAudio2OmniRunner], test_config: tuple[str, str]) -> None:
    """Test processing text-only input (no audio)."""
    model, stage_config_path = test_config

    with step_audio2_runner(model, seed=42, stage_configs_path=stage_config_path) as runner:
        outputs = runner.generate_multimodal(
            prompts="你好，请用中文回答。",
            audios=None,
        )

        assert len(outputs) > 0

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

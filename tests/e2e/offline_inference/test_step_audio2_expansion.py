# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E expansion tests for Step-Audio2 offline inference (nightly CI).

Full-model inference with audio input and audio output.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniRunner
from vllm_omni.outputs import OmniRequestOutput

MODEL = "stepfun-ai/Step-Audio-2-mini"
STAGE_CONFIG = str(Path(__file__).parent / "stage_configs" / "step_audio2_ci.yaml")
TEST_PARAMS = [(MODEL, STAGE_CONFIG)]

pytestmark = [pytest.mark.slow, pytest.mark.tts]

DEFAULT_SYSTEM_PROMPT = "你是一个语音对话助手，能够理解音频输入并生成语音回复。"
SAMPLE_RATE = 16000


def _synthetic_audio(duration_sec: int = 2) -> tuple[np.ndarray, int]:
    audio = generate_synthetic_audio(duration_sec, 1, SAMPLE_RATE)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    return np.asarray(audio, dtype=np.float32), SAMPLE_RATE


def _build_step_audio2_input(
    user_text: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    audio: tuple[np.ndarray, int] | None = None,
) -> dict[str, Any]:
    user_content = ""
    multi_modal_data: dict[str, Any] = {}
    if audio is not None:
        user_content += "<audio_patch>"
        multi_modal_data["audio"] = audio
    user_content += user_text

    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    input_dict: dict[str, Any] = {"prompt": full_prompt}
    if multi_modal_data:
        input_dict["multi_modal_data"] = multi_modal_data
    return input_dict


def _assert_non_empty_text_output(outputs: list[OmniRequestOutput]) -> None:
    assert len(outputs) > 0

    text_output = next((o for o in outputs if o.final_output_type == "text"), None)
    assert text_output is not None
    assert text_output.request_output is not None
    text_content = text_output.request_output.outputs[0].text
    assert text_content is not None
    assert len(text_content.strip()) > 0


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", TEST_PARAMS, indirect=True)
def test_audio_to_text_and_audio(omni_runner: OmniRunner) -> None:
    """Test processing audio input and generating text + audio output."""
    audio = _synthetic_audio(duration_sec=2)
    omni_input = _build_step_audio2_input("请复述这段音频的内容。", audio=audio)
    outputs = omni_runner.generate([omni_input])

    _assert_non_empty_text_output(outputs)

    audio_output = next((o for o in outputs if o.final_output_type == "audio"), None)
    if audio_output is not None:
        assert audio_output.request_output is not None
        audio_tensor = audio_output.request_output.outputs[0].multimodal_output.get("audio")
        if audio_tensor is not None:
            assert audio_tensor.numel() > 0


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", TEST_PARAMS, indirect=True)
def test_text_only_input(omni_runner: OmniRunner) -> None:
    """Test processing text-only input (no audio)."""
    omni_input = _build_step_audio2_input("你好，请用中文回答。")
    outputs = omni_runner.generate([omni_input])
    _assert_non_empty_text_output(outputs)

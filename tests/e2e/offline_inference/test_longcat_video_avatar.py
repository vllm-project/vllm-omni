# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("LONGCAT_AVATAR_TEST_MODEL", "meituan-longcat/LongCat-Video-Avatar-1.5")
BASE_MODEL_DIR = os.environ.get("LONGCAT_AVATAR_TEST_BASE_MODEL_DIR")
PROMPT = "A person speaks calmly while facing the camera."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"


@pytest.fixture()
def synthetic_audio_path(tmp_path):
    sample_rate = 16000
    duration_s = 2.0
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    audio = (0.08 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = tmp_path / "avatar_reference.wav"
    sf.write(path, audio, sample_rate)
    return str(path)


@pytest.fixture()
def synthetic_image_path(tmp_path):
    path = tmp_path / "avatar_reference.png"
    Image.new("RGB", (768, 512), (130, 104, 88)).save(path)
    return str(path)


def _sampling(stage: str) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        num_frames=5,
        fps=25,
        num_inference_steps=8,
        guidance_scale=1.0,
        guidance_scale_2=1.0,
        seed=42,
        extra_args={
            "stage": stage,
            "resolution": "480p",
            "save_fps": 25,
            "use_distill": True,
            "use_int8": True,
        },
    )


def _omni_runner_param():
    additional_config = {
        "model_type": "avatar-v1.5",
        "resolution": "480p",
        "use_distill": True,
        "use_int8": True,
    }
    if BASE_MODEL_DIR:
        additional_config["base_model_dir"] = BASE_MODEL_DIR
    return (
        MODEL,
        None,
        {
            "model_class_name": "LongCatVideoAvatarPipeline",
            "additional_config": additional_config,
        },
    )


def _generate_avatar_frames(
    omni_runner: OmniRunner,
    *,
    stage: str,
    audio_path: str,
    image_path: str | None = None,
):
    prompt = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "multi_modal_data": {"audio": audio_path},
    }
    if image_path is not None:
        prompt["multi_modal_data"]["image"] = image_path

    outputs = omni_runner.generate([prompt], [_sampling(stage)])
    assert outputs
    output = outputs[0]
    assert output.images
    frames = output.images[0]
    if isinstance(frames, tuple) and len(frames) == 2:
        frames = frames[0]
    if isinstance(frames, dict):
        frames = frames.get("frames") or frames.get("video")
    assert isinstance(frames, list)
    return frames


def _assert_frames(frames, *, expected_size: tuple[int, int]):
    assert len(frames) == 5
    for frame in frames:
        assert isinstance(frame, Image.Image)
        assert frame.mode == "RGB"
        assert frame.size == expected_size


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize(
    "omni_runner",
    [_omni_runner_param()],
    indirect=True,
)
def test_longcat_video_avatar_at2v(omni_runner: OmniRunner, synthetic_audio_path):
    frames = _generate_avatar_frames(
        omni_runner,
        stage="at2v",
        audio_path=synthetic_audio_path,
    )
    _assert_frames(frames, expected_size=(832, 480))


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize(
    "omni_runner",
    [_omni_runner_param()],
    indirect=True,
)
def test_longcat_video_avatar_ai2v(
    omni_runner: OmniRunner,
    synthetic_audio_path,
    synthetic_image_path,
):
    frames = _generate_avatar_frames(
        omni_runner,
        stage="ai2v",
        audio_path=synthetic_audio_path,
        image_path=synthetic_image_path,
    )
    _assert_frames(frames, expected_size=(768, 512))

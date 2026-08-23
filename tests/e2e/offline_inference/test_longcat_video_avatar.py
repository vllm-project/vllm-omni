# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.models.longcat_video.pipeline_longcat_video_avatar import (
    prepare_longcat_video_avatar_model_for_omni,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "meituan-longcat/LongCat-Video-Avatar-1.5"
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
def synthetic_multi_audio_paths(tmp_path):
    sample_rate = 16000
    duration_s = 2.0
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    left = (0.08 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    right = (0.08 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    left_path = tmp_path / "avatar_left.wav"
    right_path = tmp_path / "avatar_right.wav"
    sf.write(left_path, left, sample_rate)
    sf.write(right_path, right, sample_rate)
    return {"person1": str(left_path), "person2": str(right_path)}


@pytest.fixture()
def synthetic_image_path(tmp_path):
    path = tmp_path / "avatar_reference.png"
    Image.new("RGB", (768, 512), (130, 104, 88)).save(path)
    return str(path)


def _sampling(
    stage: str,
    *,
    num_frames: int = 5,
    extra_args: dict | None = None,
) -> OmniDiffusionSamplingParams:
    sampling_extra_args = {
        "stage": stage,
        "resolution": "480p",
        "save_fps": 25,
        "use_distill": True,
        "use_int8": True,
    }
    if extra_args:
        sampling_extra_args.update(extra_args)
    return OmniDiffusionSamplingParams(
        num_frames=num_frames,
        fps=25,
        num_inference_steps=8,
        guidance_scale=1.0,
        guidance_scale_2=1.0,
        seed=42,
        extra_args=sampling_extra_args,
    )


@pytest.fixture(scope="module")
def longcat_omni_runner() -> Generator[OmniRunner, None, None]:
    model = prepare_longcat_video_avatar_model_for_omni(MODEL, use_int8=True)
    with OmniRunner(
        model,
        seed=42,
        model_class_name="LongCatVideoAvatarPipeline",
        additional_config={
            "model_type": "avatar-v1.5",
            "resolution": "480p",
            "use_distill": True,
            "use_int8": True,
        },
    ) as runner:
        yield runner


def _generate_avatar_frames(
    omni_runner: OmniRunner,
    *,
    stage: str,
    audio_path: str | dict[str, str],
    image_path: str | None = None,
    additional_information: dict | None = None,
    num_frames: int = 5,
    sampling_extra_args: dict | None = None,
):
    prompt = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "multi_modal_data": {"audio": audio_path},
        "additional_information": additional_information or {},
    }
    prompt["additional_information"].update({"stage": stage, "resolution": "480p"})
    if image_path is not None:
        prompt["multi_modal_data"]["image"] = image_path

    outputs = omni_runner.generate(
        [prompt],
        [_sampling(stage, num_frames=num_frames, extra_args=sampling_extra_args)],
    )
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


def _assert_frames(frames, *, expected_size: tuple[int, int], expected_count: int = 5):
    assert len(frames) == expected_count
    for frame in frames:
        assert isinstance(frame, Image.Image)
        assert frame.mode == "RGB"
        assert frame.size == expected_size


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
def test_longcat_video_avatar_at2v(longcat_omni_runner: OmniRunner, synthetic_audio_path):
    frames = _generate_avatar_frames(
        longcat_omni_runner,
        stage="at2v",
        audio_path=synthetic_audio_path,
    )
    _assert_frames(frames, expected_size=(832, 480))


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
def test_longcat_video_avatar_ai2v(
    longcat_omni_runner: OmniRunner,
    synthetic_audio_path,
    synthetic_image_path,
):
    frames = _generate_avatar_frames(
        longcat_omni_runner,
        stage="ai2v",
        audio_path=synthetic_audio_path,
        image_path=synthetic_image_path,
    )
    _assert_frames(frames, expected_size=(768, 512))


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
def test_longcat_video_avatar_multi_speaker_ai2v(
    longcat_omni_runner: OmniRunner,
    synthetic_multi_audio_paths,
    synthetic_image_path,
):
    frames = _generate_avatar_frames(
        longcat_omni_runner,
        stage="ai2v",
        audio_path=synthetic_multi_audio_paths,
        image_path=synthetic_image_path,
        additional_information={
            "audio_type": "para",
            "bbox": {
                "person1": [32, 48, 384, 336],
                "person2": [32, 432, 384, 720],
            },
        },
    )
    _assert_frames(frames, expected_size=(768, 512))


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
def test_longcat_video_avatar_single_speaker_ai2v_avc_continuation(
    longcat_omni_runner: OmniRunner,
    synthetic_audio_path,
    synthetic_image_path,
):
    frames = _generate_avatar_frames(
        longcat_omni_runner,
        stage="ai2v",
        audio_path=synthetic_audio_path,
        image_path=synthetic_image_path,
        num_frames=17,
        sampling_extra_args={
            "num_segments": 2,
            "num_cond_frames": 5,
            "use_kv_cache": True,
        },
    )
    _assert_frames(frames, expected_size=(768, 512), expected_count=29)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
def test_longcat_video_avatar_multi_speaker_ai2v_avc_continuation(
    longcat_omni_runner: OmniRunner,
    synthetic_multi_audio_paths,
    synthetic_image_path,
):
    frames = _generate_avatar_frames(
        longcat_omni_runner,
        stage="ai2v",
        audio_path=synthetic_multi_audio_paths,
        image_path=synthetic_image_path,
        additional_information={
            "audio_type": "para",
            "bbox": {
                "person1": [32, 48, 384, 336],
                "person2": [32, 432, 384, 720],
            },
        },
        num_frames=17,
        sampling_extra_args={
            "num_segments": 2,
            "num_cond_frames": 5,
            "use_kv_cache": True,
        },
    )
    _assert_frames(frames, expected_size=(768, 512), expected_count=29)

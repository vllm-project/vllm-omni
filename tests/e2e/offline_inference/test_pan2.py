# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline E2E test for PAN2 text-to-video and image-to-video on a tiny random-weight checkpoint.

Set ``VLLM_OMNI_PAN2_TINY_MODEL`` to run against a local copy of the tiny checkpoint.
"""

import os
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("VLLM_OMNI_PAN2_TINY_MODEL", "wuqing157/tiny-pan2-modular-pipe")
HEIGHT = 64
WIDTH = 96
NUM_FRAMES = 9
# A large scale makes the guided steps visible through the tiny random VAE.
GUIDANCE_SCALE = 30.0
CONDITION_IMAGE = Image.new("RGB", (WIDTH, HEIGHT), (200, 30, 30))

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    # The guardrails load gated weights; tests/diffusion/models/pan2/test_pan2_guardrails.py covers them.
    pytest.mark.parametrize("omni_runner", [(MODEL, None, {"model_config": {"guardrails": False}})], indirect=True),
]


def _generate(
    omni_runner: OmniRunner,
    *,
    image: Image.Image | None = None,
    num_inference_steps: int = 3,
    guidance_scale: float = GUIDANCE_SCALE,
) -> np.ndarray:
    prompt: dict[str, Any] = {"prompt": "a cat walks on the grass", "negative_prompt": "blurry"}
    if image is not None:
        prompt["multi_modal_data"] = {"image": image}
    sampling_params = OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=42,
    )
    outputs = omni_runner.omni.generate(prompt, sampling_params)
    assert len(outputs) == 1
    frames = outputs[0].images
    assert len(frames) == NUM_FRAMES
    for frame in frames:
        assert isinstance(frame, Image.Image)
        assert frame.mode == "RGB"
        assert frame.size == (WIDTH, HEIGHT)
    video = np.stack([np.asarray(frame) for frame in frames])
    assert video.dtype == np.uint8
    return video


def _max_frame_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).reshape(len(a), -1).max(axis=1)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("image", [None, CONDITION_IMAGE], ids=["t2v", "i2v"])
def test_pan2_generates_requested_video(omni_runner: OmniRunner, image: Image.Image | None):
    video = _generate(omni_runner, image=image)
    assert video.shape == (NUM_FRAMES, HEIGHT, WIDTH, 3)
    assert video.std() > 0


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("image", [None, CONDITION_IMAGE], ids=["t2v", "i2v"])
def test_pan2_guidance_applies_to_every_frame(omni_runner: OmniRunner, image: Image.Image | None):
    unguided = _generate(omni_runner, image=image, guidance_scale=1.0)
    assert _max_frame_diff(_generate(omni_runner, image=image), unguided).min() > 4

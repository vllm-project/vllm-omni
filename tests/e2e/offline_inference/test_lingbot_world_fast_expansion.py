# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L3 real-weight offline expansion for Lingbot World Fast."""

from __future__ import annotations

from typing import Any

import numpy as np
import PIL.Image
import pytest
import torch

from tests.helpers.lingbot_world_fast import (
    FPS,
    GREAT_WALL_PROMPT,
    HEIGHT,
    LONG_NUM_FRAMES,
    SEED,
    SHORT_NUM_FRAMES,
    SSIM_THRESHOLD,
    WIDTH,
    find_lingbot_world_fast_assets,
    frame_ssim,
    golden_frames_dir,
    load_camera_trajectory,
    normalize_to_uint8_rgb,
)
from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.core_model,
    pytest.mark.diffusion,
]


def _extract_frames_from_output(output: Any) -> np.ndarray:
    """Pull a ``[N, H, W, 3]`` numpy array out of an ``OmniRequestOutput``."""
    if isinstance(output, list) and output:
        output = output[0]
    if isinstance(output, OmniRequestOutput):
        if output.is_pipeline_output and output.request_output is not None:
            inner = output.request_output
            if isinstance(inner, OmniRequestOutput):
                output = inner
        if isinstance(output, OmniRequestOutput) and output.images:
            entry = output.images[0]
            if isinstance(entry, tuple) and len(entry) >= 1:
                output = entry[0]
            elif isinstance(entry, dict):
                output = entry.get("frames") or entry.get("video")
            else:
                output = entry
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if not isinstance(output, np.ndarray):
        raise AssertionError(f"Could not extract frames from output: {type(output)}")
    return normalize_to_uint8_rgb(output)


@pytest.fixture(scope="module")
def lingbot_world_fast_assets():
    assets = find_lingbot_world_fast_assets()
    if assets is None:
        pytest.skip(
            "Lingbot-World-Fast L3 assets not available. Set LINGBOT_WORLD_FAST_PATH "
            "(model dir) + LINGBOT_WORLD_FAST_CAMERA_PATH (poses.npy/intrinsics.npy) "
            "+ LINGBOT_WORLD_FAST_IMAGE (input image) to enable.",
        )
    return assets


@pytest.fixture(scope="module")
def lingbot_world_fast_omni(lingbot_world_fast_assets):
    omni = Omni(
        model=str(lingbot_world_fast_assets.weights_path),
        parallel_config=None,
        model_class_name="LingbotWorldFastPipeline",
        stage_init_timeout=6000,
        init_timeout=6000,
    )
    try:
        yield omni
    finally:
        if hasattr(omni, "close"):
            omni.close()


@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("num_frames, length", [(SHORT_NUM_FRAMES, "short"), (LONG_NUM_FRAMES, "long")])
def test_lingbot_world_offline_video(
    num_frames,
    length,
    lingbot_world_fast_assets,
    lingbot_world_fast_omni,
):
    image = (
        PIL.Image.open(lingbot_world_fast_assets.image_path)
        .convert("RGB")
        .resize((WIDTH, HEIGHT), PIL.Image.Resampling.LANCZOS)
    )
    poses, intrinsics = load_camera_trajectory(lingbot_world_fast_assets.camera_dir)
    poses = poses[:num_frames]
    intrinsics = intrinsics[:num_frames]

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED)
    sampling = OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
        num_frames=num_frames,
        frame_rate=FPS,
        extra_args={"session_id": f"SESSION_ID-{length}"},
    )

    multi_modal_data: dict = {"image": image, "camera": {"poses": poses, "intrinsics": intrinsics}}

    prompt = {
        "prompt": GREAT_WALL_PROMPT,
        "negative_prompt": "",
        "multi_modal_data": multi_modal_data,
    }

    output = lingbot_world_fast_omni.generate(prompt, sampling)

    video = _extract_frames_from_output(output)

    first_frame = video[0]
    last_frame = video[-1]

    first_path = golden_frames_dir() / f"golden_frame_{length}_first.npy"
    last_path = golden_frames_dir() / f"golden_frame_{length}_last.npy"

    first_golden = np.load(first_path)
    last_golden = np.load(last_path)

    ssim_first = frame_ssim(first_frame, first_golden)
    ssim_last = frame_ssim(last_frame, last_golden)
    print(
        f"[lingbot-world-fast L3] SSIM(first)={ssim_first:.4f} SSIM(last)={ssim_last:.4f} (threshold {SSIM_THRESHOLD})"
    )
    assert ssim_first >= SSIM_THRESHOLD, (
        f"First-frame SSIM {ssim_first:.4f} below {SSIM_THRESHOLD}: regression in first-call path."
    )
    assert ssim_last >= SSIM_THRESHOLD, (
        f"Last-frame SSIM {ssim_last:.4f} below {SSIM_THRESHOLD}: regression in last-call path."
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.diffusion,
    pytest.mark.gpu,
]


def _make_sana_wm_e2e_image() -> Any:
    from PIL import Image

    from vllm_omni.diffusion.models.sana_wm import SANA_WM_OUTPUT_HEIGHT, SANA_WM_OUTPUT_WIDTH

    return Image.new("RGB", (SANA_WM_OUTPUT_WIDTH, SANA_WM_OUTPUT_HEIGHT), (96, 128, 160))


def _sana_wm_e2e_tensor_parallel_size() -> int:
    return int(
        os.environ.get(
            "SANA_WM_E2E_TENSOR_PARALLEL_SIZE",
            os.environ.get("SANA_WM_E2E_TP", "1"),
        )
    )


def _coerce_video_array(video: Any) -> np.ndarray:
    import torch

    if isinstance(video, list):
        assert video, "SANA-WM e2e produced an empty video list."
        video = video[0]
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().float().numpy()
    video_array = np.asarray(video)
    if video_array.ndim == 5:
        assert video_array.shape[0] == 1
        video_array = video_array[0]
    assert video_array.ndim == 4
    return video_array


def _assert_sana_wm_e2e_shape(
    video: np.ndarray,
    *,
    output_type: str,
    num_frames: int,
) -> None:
    if output_type == "latent":
        # Latent output is channel-first: (C, T, H, W) after removing the
        # batch dimension. Stage-1 emits LTX-2 latents with 128 channels.
        assert video.shape[0] == 128
        assert 0 < video.shape[1] <= num_frames
        return
    assert 0 < video.shape[0] <= num_frames


def _run_sana_wm_e2e(*, output_type: str) -> np.ndarray:
    import torch

    from vllm_omni.diffusion.models.sana_wm import (
        SANA_WM_MODEL_ID,
        SANA_WM_OUTPUT_HEIGHT,
        SANA_WM_OUTPUT_WIDTH,
    )
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

    if not torch.cuda.is_available():
        pytest.skip("Sana-WM e2e requires CUDA.")

    model = os.environ.get("SANA_WM_E2E_MODEL", SANA_WM_MODEL_ID)
    model_class_name = os.environ.get("SANA_WM_E2E_MODEL_CLASS", "SanaWmPipeline")
    num_frames = int(os.environ.get("SANA_WM_E2E_NUM_FRAMES", "9"))
    stage1_steps = int(
        os.environ.get(
            "SANA_WM_E2E_STAGE1_STEPS",
            os.environ.get("SANA_WM_E2E_NUM_INFERENCE_STEPS", "1"),
        )
    )
    action = os.environ.get("SANA_WM_E2E_ACTION", f"w-{max(num_frames - 1, 1)}")
    tensor_parallel_size = _sana_wm_e2e_tensor_parallel_size()
    image = _make_sana_wm_e2e_image()
    omni = Omni(
        model=model,
        model_class_name=model_class_name,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel_size,
    )
    extra_args: dict[str, Any] = {}
    # The native Stage-1 path is latent-token capped; raise it for full-size
    # e2e runs (704x1280 exceeds the default 4096-token cap).
    native_max_tokens = os.environ.get("SANA_WM_E2E_NATIVE_MAX_TOKENS", "").strip()
    if native_max_tokens:
        extra_args["sana_wm_native_max_tokens"] = int(native_max_tokens)
    # Return the requested output type so the shape assertion matches the
    # pipeline output instead of the raw Stage-1 latent default.

    output = omni.generate(
        {
            "prompt": "A slow forward camera move through a quiet city street.",
            "multi_modal_data": {"image": image},
            "sana_wm": {
                "action": action,
                "num_frames": num_frames,
                "translation_speed": 0.055,
                "rotation_speed_deg": 1.2,
                # Avoid the optional Pi3X dependency by passing deterministic
                # camera intrinsics directly.
                "intrinsics": {
                    "fx": SANA_WM_OUTPUT_WIDTH / 2,
                    "fy": SANA_WM_OUTPUT_WIDTH / 2,
                    "cx": SANA_WM_OUTPUT_WIDTH / 2,
                    "cy": SANA_WM_OUTPUT_HEIGHT / 2,
                },
            },
        },
        OmniDiffusionSamplingParams(
            height=SANA_WM_OUTPUT_HEIGHT,
            width=SANA_WM_OUTPUT_WIDTH,
            num_frames=num_frames,
            seed=0,
            fps=16,
            num_inference_steps=stage1_steps,
            guidance_scale=1.0,
            guidance_scale_provided=True,
            output_type=output_type,
            extra_args=extra_args,
        ),
    )

    request_output = output[0] if isinstance(output, list) else output
    assert isinstance(request_output, OmniRequestOutput)
    assert request_output.error is None
    assert request_output.images

    frames = request_output.images[0]
    video = _coerce_video_array(frames)
    _assert_sana_wm_e2e_shape(video, output_type=output_type, num_frames=num_frames)
    return video


def test_sana_wm_native_generates_video() -> None:
    output_type = os.environ.get("SANA_WM_E2E_OUTPUT_TYPE", "np")
    video = _run_sana_wm_e2e(output_type=output_type)
    if output_type == "latent":
        assert video.ndim == 4
    else:
        assert video.shape[-1] in (3, 4)

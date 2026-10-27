# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real-weight L3 coverage for optional latent upscaling and second-pass refinement.

Set VLLM_TEST_MINIMAX_H3_LATENT_UPSCALER to the LBH BF16 upscaler checkpoint.
VLLM_TEST_MINIMAX_H3_MODEL may point to a local FL2VA model directory.
"""

import json
import os
from pathlib import Path

import pytest

from tests.helpers.assertions import assert_video_first_frames_differ, assert_video_valid
from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

MODEL = os.environ.get("VLLM_TEST_MINIMAX_H3_MODEL", "MiniMaxAI/MiniMax-H3")
UPSCALER = os.environ.get("VLLM_TEST_MINIMAX_H3_LATENT_UPSCALER")
pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion]


@pytest.mark.skipif(not UPSCALER, reason="set VLLM_TEST_MINIMAX_H3_LATENT_UPSCALER to the real upscaler checkpoint")
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--trust-remote-code",
                    "--task-type",
                    "fl2va",
                    "--num-gpus",
                    "2",
                    "--tensor-parallel-size",
                    "2",
                    "--usp",
                    "1",
                    "--ring",
                    "1",
                    "--text-encoder-tp-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-parallel-mode",
                    "tile",
                    "--vae-use-tiling",
                    "--enable-cpu-offload",
                    "--enforce-eager",
                    "--diffusion-attention-backend",
                    "FLASH_ATTN",
                    "--additional-config",
                    json.dumps({"latent_upscaler_path": UPSCALER, "latent_upscaler_dtype": "bf16"}),
                ],
                env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
            ),
            id="minimax_h3_latent_upscale_refine_tp2",
            marks=hardware_marks(res={"cuda": "H100"}, num_cards=2),
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("task", ["t2va", "fl2va"])
def test_latent_upscale_and_refine(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    tmp_path: Path,
    task: str,
) -> None:
    """Exercise upscale, refinement, and request-local opt-out on one live server."""
    videos = []
    frame_counts = []
    image_reference = None
    if task == "fl2va":
        image = generate_synthetic_image(512, 288, seed=42)
        image_reference = f"data:image/jpeg;base64,{image['base64']}"
    # The 1024x576 refine keyframe spans multiple 256px VAE encoder tiles:
    # both VAE ranks must encode it before broadcasting the condition latents.
    # Four base steps and the minimum four-second clip keep real-weight CI affordable.
    # Use size rather than width/height: upscale intentionally changes output dimensions.
    for name, upscale, refine, width, height in [
        ("original", False, False, 512, 288),
        ("upscaled", 2.0, False, 1024, 576),
        ("refined", 2.0, 0.5, 1024, 576),
        ("disabled_again", False, False, 512, 288),
    ]:
        responses = online_client.send_video_diffusion_request(
            {
                "model": omni_server.model,
                "image_reference": image_reference,
                "form_data": {
                    "model": omni_server.model,
                    "prompt": "A woman in a sunlit room turns toward the camera, cinematic lighting.",
                    "size": "512x288",
                    "fps": 24,
                    "num_inference_steps": 4,
                    "flow_shift": 12,
                    "seed": 1101,
                    "extra_params": json.dumps(
                        {
                            "task": task,
                            "duration": 4.0,
                            "aspect_ratio": "16:9",
                            "audio_flow_shift": 3.0,
                            "latent_upscale": upscale,
                            "latent_refine": refine,
                        }
                    ),
                },
                "expected_audio": {"sample_rate": 32000, "channels": 2},
            }
        )
        video = responses[0].videos[0]
        (tmp_path / f"{name}.mp4").write_bytes(video)
        metadata = assert_video_valid(video, width=width, height=height, fps=24)
        frame_counts.append(metadata["num_frames"])
        videos.append(video)

    assert frame_counts[0] > 0
    assert len(set(frame_counts)) == 1, f"Upscale/refinement changed the frame count: {frame_counts}"
    # Equal output resolution alone would miss a silently ignored second denoise pass.
    assert_video_first_frames_differ(videos[2], videos[1], min_mean_absolute_error=1 / 255)

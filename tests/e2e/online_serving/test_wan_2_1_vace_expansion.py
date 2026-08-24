# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Comprehensive e2e tests of diffusion features for Wan2.1-VACE in online serving mode.

Wan2.1-VACE supports: Cache-DiT, Ulysses-SP, Ring, CFG-Parallel, TP,
VAE-Patch-Parallel, HSDP, and reference-hint caching. TeaCache is NOT
supported for this model, so Cache-DiT is used in place of TeaCache for
single-card and CFG tests.

Uses the 1.3B variant for faster CI testing.

Coverage:
  Single GPU:
    - Cache-DiT + layerwise CPU offload
    - Ref-Hint forecast50
  Two GPUs:
    - Cache-DiT + Ulysses-SP = 2
    - Cache-DiT + Ring = 2
    - Cache-DiT + CFG-Parallel = 2
    - Cache-DiT + TP = 2 + VAE-Patch-Parallel = 2
    - Cache-DiT + HSDP = 2 + VAE-Patch-Parallel = 2
"""

import re

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
PROMPT = "A cat walking slowly across a sunlit garden path"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_vace_feature_cases():
    return [
        # Single GPU: CPU offload
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--enable-cpu-offload",
                    "--vae-use-tiling",
                ],
            ),
            id="single_card_cpu_offload",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # Single GPU: Cache-DiT + layerwise CPU offload
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--enable-layerwise-offload",
                    "--vae-use-tiling",
                ],
            ),
            id="single_card_001",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # 2 GPUs: Cache-DiT + Ulysses-SP = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ulysses-degree",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_001",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: Cache-DiT + Ring = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ring",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_002",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: Cache-DiT + CFG-Parallel = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--cfg-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_003",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: Cache-DiT + TP = 2 + VAE-Patch-Parallel = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_004",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: Cache-DiT + HSDP = 2 + VAE-Patch-Parallel = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--hsdp-shard-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_005",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_vace_feature_cases(),
    indirect=True,
)
def test_wan_2_1_vace(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test VACE T2V generation with all supported diffusion acceleration features."""
    openai_client.send_video_diffusion_request(
        {
            "model": MODEL,
            "form_data": {
                "prompt": PROMPT,
                "height": 480,
                "width": 320,
                "num_frames": 5,
                "fps": 8,
                "num_inference_steps": 2,
                "guidance_scale": 5.0,
                "seed": 42,
            },
        }
    )


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "ref_hint",
                    "--cache-config",
                    '{"ref_hint_refresh_interval":2,'
                    '"ref_hint_strategy":"forecast50",'
                    '"ref_hint_acknowledge_lossy":true}',
                    "--vae-use-tiling",
                ],
                capture_output=True,
            ),
            id="single_card_ref_hint",
            marks=SINGLE_CARD_FEATURE_MARKS,
        )
    ],
    indirect=True,
)
def test_wan_2_1_vace_ref_hint_cache_hits(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Exercise forecast50 past calibration and prove the model-region handler ran."""
    openai_client.send_video_diffusion_request(
        {
            "model": MODEL,
            "form_data": {
                "prompt": PROMPT,
                "height": 480,
                "width": 320,
                "num_frames": 5,
                "fps": 8,
                "num_inference_steps": 3,
                "guidance_scale": 5.0,
                "seed": 42,
            },
        }
    )

    server_log = omni_server.read_log()
    match = re.search(r"Reference-hint cache request summary:.*hits=(\d+)", server_log)
    assert match is not None, server_log[-8000:]
    assert int(match.group(1)) > 0, match.group(0)

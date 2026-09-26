# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following text-to-image models:
- Qwen-Image
- Qwen-Image-2512

One feature per test case, matching the Test Plan in PR #1682 (Qwen-Image-Edit).
Nightly covers each feature once, alternating models (5× Qwen-Image, 4× Qwen-Image-2512).
Ulysses stays on Qwen-Image. See docs/user_guide/diffusion_acceleration.md.
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient, dummy_messages_from_mix_data

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

T2I_PROMPT = "A photo of a cat sitting on a laptop keyboard, digital art style."
NEGATIVE_PROMPT = "blurry, low quality"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": ["H100", "B200"]})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=2)

MODEL_IMAGE = "Qwen/Qwen-Image"
MODEL_2512 = "Qwen/Qwen-Image-2512"


def _is_cuda_fp8_supported() -> bool:
    """Return True if the active CUDA device supports native FP8 (Compute Capability >= 8.9).

    SM80 (e.g. A100) lacks native FP8 tensor core support and Inductor fails to
    lower auto_functionalized_v2 under torch.compile.
    FP8 is supported on Ada Lovelace (SM89), Hopper (SM90), and Blackwell (SM100+).
    """
    override = os.getenv("VLLM_TEST_ENABLE_FP8")
    if override is not None:
        return override.lower() in ("1", "true", "yes")
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability() >= (8, 9)
    except Exception:
        return False


_vae_patch_parallel_server_args = [
    "--cache-backend",
    "cache_dit",
    "--tensor-parallel-size",
    "2",
    "--vae-patch-parallel-size",
    "2",
    "--vae-use-tiling",
]
if _is_cuda_fp8_supported():
    _vae_patch_parallel_server_args.extend(
        [
            "--diffusion-quantization-config",
            '{"method":"fp8"}',
        ]
    )


# One server per feature. Alternate models; keep feature pytest ids unchanged.
FEATURE_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL_IMAGE,
            server_args=["--enable-cpu-offload"],
        ),
        id="cpu_offload",
        marks=SINGLE_CARD_FEATURE_MARKS,
    ),
    pytest.param(
        OmniServerParams(model=MODEL_2512, server_args=["--step-execution"]),
        id="step_execution",
        marks=SINGLE_CARD_FEATURE_MARKS,
    ),
    pytest.param(
        OmniServerParams(model=MODEL_IMAGE, server_args=["--cache-backend", "tea_cache"]),
        id="cache_tea_cache",
        marks=SINGLE_CARD_FEATURE_MARKS,
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL_2512,
            server_args=[
                "--cache-backend",
                "cache_dit",
                "--enable-layerwise-offload",
            ],
        ),
        id="layerwise_offload",
        marks=SINGLE_CARD_FEATURE_MARKS,
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL_IMAGE,
            server_args=[
                "--cache-backend",
                "cache_dit",
                "--ulysses-degree",
                "2",
            ],
        ),
        id="ulysses_2",
        marks=PARALLEL_FEATURE_MARKS,
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL_2512,
            server_args=[
                "--cache-backend",
                "cache_dit",
                "--ring",
                "2",
            ],
        ),
        id="ring_2",
        marks=PARALLEL_FEATURE_MARKS,
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL_IMAGE,
            server_args=[
                "--cache-backend",
                "tea_cache",
                "--cfg-parallel-size",
                "2",
            ],
        ),
        id="cfg_parallel_2",
        marks=PARALLEL_FEATURE_MARKS,
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL_2512,
            server_args=_vae_patch_parallel_server_args,
        ),
        id="vae_patch_parallel_2",
        marks=PARALLEL_FEATURE_MARKS,
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL_IMAGE,
            server_args=[
                "--use-hsdp",
                "--hsdp-shard-size",
                "2",
            ],
        ),
        id="parallel_hsdp",
        marks=PARALLEL_FEATURE_MARKS,
    ),
]


@pytest.mark.parametrize("omni_server", FEATURE_CASES, indirect=True)
def test_qwen_image(omni_server: OmniServer, online_client: OnlineOmniClient):
    """One diffusion feature per case; model is chosen in FEATURE_CASES."""
    messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }
    online_client.send_diffusion_request(request_config)

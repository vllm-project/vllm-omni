"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following models:
- ByteDance-Seed/BAGEL-7B-MoT
Coverage:
- TeaCache
- Cache-DiT
- CFG-Parallel
- Ulysses-SP
- Ring-Attention
- Layerwise Offloading
- Hybrid Sharded Data Parallel
- Tensor Parallelism + VAE Patch Parallelism

assert_diffusion_response validates successful generation and the expected
512x512 resolution.
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

PROMPT = "A futuristic city skyline at twilight, cyberpunk style, ultra-detailed, high resolution."
NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, watermark"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_2_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)
HSDP_2_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

BAGEL_CI_DEPLOY = get_deploy_config_path("ci/bagel.yaml")
BAGEL_PARALLEL_2_DEPLOY = modify_stage_config(
    BAGEL_CI_DEPLOY,
    updates={"stages": {0: {"devices": "0"}, 1: {"devices": "0,1"}}},
)
BAGEL_HSDP_2_DEPLOY = modify_stage_config(
    BAGEL_CI_DEPLOY,
    updates={"stages": {0: {"devices": "0"}, 1: {"devices": "0,1"}}},
)
BAGEL_TP_VAE_PP_2_DEPLOY = modify_stage_config(
    BAGEL_CI_DEPLOY,
    updates={
        "stages": {
            0: {"devices": "0"},
            1: {
                "devices": "0,1",
                "parallel_config": {
                    "tensor_parallel_size": 2,
                    "vae_patch_parallel_size": 2,
                },
            },
        },
    },
)


def _get_diffusion_feature_cases(model: str):
    """Return L4 diffusion feature cases for Bagel.
    TeaCache, Cache-DiT, CFG-Parallel,
    Ulysses-SP, Ring-Attention, Layerwise Offloading,
    Hybrid Sharded Data Parallel, Tensor Parallelism, VAE Patch Parallelism.
    """

    return [
        # TeaCache (single-card)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                ],
            ),
            id="single_card_teacache",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # Cache-DiT (single-card)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                ],
            ),
            id="single_card_cache_dit",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # CFG-Parallel size 2 (2 GPUs, TeaCache backend)
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=BAGEL_PARALLEL_2_DEPLOY,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="parallel_cfg_2",
            marks=PARALLEL_2_FEATURE_MARKS,
        ),
        # Ulysses-SP degree=2 (2 GPUs)
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=BAGEL_PARALLEL_2_DEPLOY,
                server_args=[
                    "--usp",
                    "2",
                ],
            ),
            id="sp_ulysses_2",
            marks=PARALLEL_2_FEATURE_MARKS,
        ),
        # Ring-Attention degree=2 (2 GPUs)
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=BAGEL_PARALLEL_2_DEPLOY,
                server_args=[
                    "--ring",
                    "2",
                ],
            ),
            id="sp_ring_2",
            marks=PARALLEL_2_FEATURE_MARKS,
        ),
        # Layerwise Offloading (single-card)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--enable-layerwise-offload"],
            ),
            id="single_card_layerwise_offload",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # Hybrid Sharded Data Parallel (2 GPUs)
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=BAGEL_HSDP_2_DEPLOY,
                server_args=[
                    "--use-hsdp",
                    "--hsdp-shard-size",
                    "2",
                ],
            ),
            id="parallel_hsdp_2",
            marks=HSDP_2_FEATURE_MARKS,
        ),
        # Tensor Parallelism (TP) + VAE Patch Parallelism (size=2)
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=BAGEL_TP_VAE_PP_2_DEPLOY,
            ),
            id="tp_vae_patch_parallel_2",
            marks=PARALLEL_2_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("ByteDance-Seed/BAGEL-7B-MoT"),
    indirect=True,
)
def test_bagel(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """L4 diffusion feature coverage for Bagel on H100.

    This test exercises:
    - TeaCache
    - Cache-DiT
    - CFG-Parallel (size=2)
    - Ulysses-SP (degree=2)
    - Ring-Attention (degree=2)
    - Layerwise Offloading
    - Hybrid Sharded Data Parallel (size=2)
    - Tensor Parallelism (TP) + VAE Patch Parallelism (size=2)

    Validation is delegated to assert_diffusion_response in tests/helpers/assertions.py,
    which checks output dimensions and basic correctness.
    """

    messages = dummy_messages_from_mix_data(content_text=PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            # Enable CFG for models that use classifier-free guidance
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "cfg_img_scale": 1.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)

"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following text-to-image models:
- Qwen-Image
- Qwen-Image-2512

One feature per test case, matching the Test Plan in PR #1682 (Qwen-Image-Edit).
Supported features for Qwen-Image series: TeaCache, Cache-DiT, Ulysses-SP, Ring-Attention,
CFG-Parallel, Tensor-Parallel, VAE-Patch-Parallel, CPU offload (layerwise).
See docs/user_guide/diffusion_acceleration.md.
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)
from tests.utils import hardware_marks

T2I_PROMPT = "A photo of a cat sitting on a laptop keyboard, digital art style."
NEGATIVE_PROMPT = "blurry, low quality"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(model=model, server_args=["--step-execution"]),
            id="step_execution",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(model=model, server_args=["--cache-backend", "tea_cache"]),
            id="cache_tea_cache",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
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
                model=model,
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
                model=model,
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
                model=model,
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
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                    "--quantization-config",
                    '{"method":"fp8"}',
                ],
            ),
            id="vae_patch_parallel_2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Qwen/Qwen-Image"),
    indirect=True,
)
def test_qwen_image(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test each diffusion feature with Qwen-Image (text-to-image), one feature per case."""
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
    openai_client.send_diffusion_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Qwen/Qwen-Image-2512"),
    indirect=True,
)
def test_qwen_image_2512(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test each diffusion feature with Qwen-Image-2512 (text-to-image), one feature per case."""
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
    openai_client.send_diffusion_request(request_config)

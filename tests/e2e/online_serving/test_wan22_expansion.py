"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following models:
- Wan-AI/Wan2.2-T2V-A14B-Diffusers
- Wan-AI/Wan2.2-I2V-A14B-Diffusers
- Wan-AI/Wan2.2-TI2V-5B-Diffusers

Coverage:
- Cache-DiT
- CFG-Parallel
- Ulysses-SP
- Tensor-Parallel
- VAE-Patch-Parallel
- HSDP
- Ring-Attn

assert_diffusion_response validates successful generation
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    generate_synthetic_image,
)
from tests.utils import hardware_marks

PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
NEGATIVE_PROMPT = "low quality, blurry, distorted face, extra limbs, bad anatomy, watermark, logo, text, ugly, deformed, mutated, jpeg artifacts"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

WAN22_MODELS = [
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "t2v"),
    ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "i2v"),
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "ti2v"),
]

PARALLEL_CONFIGS = [
    ("cfg_parallel", ["--cfg-parallel-size", "2"]),
    ("ulysses_sp", ["--usp", "2"]),
    ("tp_vae_patch", ["--tensor-parallel-size", "2", "--vae-patch-parallel-size", "2"]),
    ("hsdp", ["--use-hsdp", "--hsdp-shard-size", "2"]),  # replicate_size=1 (default)
    ("ring_atten", ["--ring", "2"]),
]


def _get_wan22_feature_cases():
    """
    Generate parameterized test cases covering:
    - All 3 Wan2.2 model variants with architecture awareness
    - 1 single-card feature (Cache-DiT)
    - 6 multi-card parallelism features with CORRECT PARAMETER NAMES per spec
    """
    cases = []

    # Single-card: Cache-DiT (applies to all models)
    for model_path, model_key in WAN22_MODELS:
        cases.append(
            pytest.param(
                OmniServerParams(
                    model=model_path,
                    server_args=["--cache-backend", "cache_dit", "--enable-layerwise-offload"],
                ),
                id=f"{model_key}_cache_dit",
                marks=SINGLE_CARD_FEATURE_MARKS,
            )
        )

    # Multi-card features
    for model_path, model_key in WAN22_MODELS:
        for feat_id, server_args in PARALLEL_CONFIGS:
            cases.append(
                pytest.param(
                    OmniServerParams(model=model_path, server_args=server_args),
                    id=f"{model_key}_{feat_id}",
                    marks=PARALLEL_FEATURE_MARKS,
                )
            )

    return cases


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_wan22_feature_cases(),
    indirect=True,
)
def test_wan22_diffusion_features(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    model_path = omni_server.model
    is_i2v_or_ti2v = any(kw in model_path for kw in ["I2V", "TI2V"])
    is_moe_model = "I2V-A14B" in model_path  # Only I2V-A14B uses MoE per spec

    form_data = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": 512,
        "width": 512,
        "num_frames": 8,
        "fps": 8,
        "num_inference_steps": 2,
        "guidance_scale": 4.0,
        "seed": 42,
        # flow_shift omitted: Service uses resolution-based defaults (12.0 for 512px)
        # vae_use_slicing/tiling omitted: Service-side optimization, not request param
    }

    if is_moe_model:
        form_data.update(
            {
                "guidance_scale_2": 1.0,
                "boundary_ratio": 0.5,
            }
        )

    request_config = {
        "model": model_path,
        "form_data": form_data,
    }

    if is_i2v_or_ti2v:
        request_config["image_reference"] = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    openai_client.send_video_diffusion_request(request_config)

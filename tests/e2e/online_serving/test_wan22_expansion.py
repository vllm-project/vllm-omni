"""
Comprehensive tests of diffusion features that are available in online serving mode.

CUDA coverage (3 models × 6 features):
- Wan-AI/Wan2.2-T2V-A14B-Diffusers
- Wan-AI/Wan2.2-I2V-A14B-Diffusers
- Wan-AI/Wan2.2-TI2V-5B-Diffusers
Features: Cache-DiT, CFG-Parallel, Ulysses-SP, Tensor-Parallel + VAE-Patch-Parallel,
HSDP, Ring-Attn.

NPU coverage (Wan-AI/Wan2.2-I2V-A14B-Diffusers only): 2 cases.
- 4-card combined: cfg=2 + usp=2 + vae-patch=2 + hsdp.
- 2-card tp_layerwise: tp=2 + enable-layerwise-offload.

assert_diffusion_response validates successful generation
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
NEGATIVE_PROMPT = "low quality, blurry, distorted face, extra limbs, bad anatomy, watermark, logo, text, ugly, deformed, mutated, jpeg artifacts"

# CUDA marks (original matrix, unchanged)
CUDA_SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})
CUDA_PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

# NPU marks
NPU_TWO_CARD_MARKS = hardware_marks(res={"npu": "A2"}, num_cards=2)
NPU_FOUR_CARD_MARKS = hardware_marks(res={"npu": "A2"}, num_cards=4)

WAN22_MODELS = [
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "t2v"),
    ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "i2v"),
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "ti2v"),
]
NPU_MODELS = [("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "i2v")]

CACHE_DIT_ARGS = ["--cache-backend", "cache_dit", "--enable-layerwise-offload"]
HSDP_ARGS = ["--use-hsdp", "--hsdp-shard-size", "2"]

PARALLEL_CONFIGS = [
    ("cfg_parallel", ["--cfg-parallel-size", "2"]),
    ("ulysses_sp", ["--usp", "2"]),
    ("tp_vae_patch", ["--tensor-parallel-size", "2", "--vae-patch-parallel-size", "2"]),
    ("hsdp", ["--use-hsdp", "--hsdp-shard-size", "2"]),  # replicate_size=1 (default)
    ("ring_atten", ["--ring", "2"]),
]

# NPU: 2 cases only.
NPU_PARALLEL_CONFIGS = [
    (
        "combined",
        [
            "--cfg-parallel-size",
            "2",
            "--usp",
            "2",
            "--vae-patch-parallel-size",
            "4",
            "--use-hsdp",
            "--hsdp-shard-size",
            "4",
        ],
        NPU_FOUR_CARD_MARKS,
    ),
    (
        "tp_layerwise_offload",
        ["--tensor-parallel-size", "2", "--enable-layerwise-offload"],
        NPU_TWO_CARD_MARKS,
    ),
]


def _get_wan22_feature_cases():
    """
    Generate parameterized test cases:
    - CUDA: 3 models × (Cache-DiT + 5 parallel features), original matrix.
    - NPU: I2V-A14B only, 2 cases (4-card combined, 2-card tp_layerwise_offload).
    """
    cases = []

    # ---- CUDA cases (unchanged) ----
    for model_path, model_key in WAN22_MODELS:
        cases.append(
            pytest.param(
                OmniServerParams(model=model_path, server_args=CACHE_DIT_ARGS),
                id=f"cuda_{model_key}_cache_dit",
                marks=CUDA_SINGLE_CARD_MARKS,
            )
        )
    for model_path, model_key in WAN22_MODELS:
        for feat_id, server_args in PARALLEL_CONFIGS:
            cases.append(
                pytest.param(
                    OmniServerParams(model=model_path, server_args=server_args),
                    id=f"cuda_{model_key}_{feat_id}",
                    marks=CUDA_PARALLEL_MARKS,
                )
            )

    # ---- NPU cases (I2V-A14B only) ----
    for model_path, model_key in NPU_MODELS:
        for feat_id, server_args, marks in NPU_PARALLEL_CONFIGS:
            cases.append(
                pytest.param(
                    OmniServerParams(model=model_path, server_args=server_args),
                    id=f"npu_{model_key}_{feat_id}",
                    marks=marks,
                )
            )

    return cases


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

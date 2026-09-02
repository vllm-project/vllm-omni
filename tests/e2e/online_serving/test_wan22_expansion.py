# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Comprehensive tests of diffusion features that are available in online serving mode.

CUDA coverage uses 8 paired cases across 3 models:
- Wan-AI/Wan2.2-T2V-A14B-Diffusers
- Wan-AI/Wan2.2-I2V-A14B-Diffusers
- Wan-AI/Wan2.2-TI2V-5B-Diffusers
Features: CPU offload, Cache-DiT, layerwise offload, CFG-Parallel, Ulysses-SP,
Tensor-Parallel + VAE-Patch-Parallel, HSDP, and Ring-Attn. Each feature is
covered once without expanding to a model-by-feature Cartesian product.

NPU coverage (Wan-AI/Wan2.2-I2V-A14B-Diffusers only): 2 cases.
- 4-card combined: cfg=2 + usp=2 + vae-patch=2 + hsdp.
- 2-card tp_layerwise: tp=2 + enable-layerwise-offload.

assert_diffusion_response validates successful generation
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
NEGATIVE_PROMPT = "low quality, blurry, distorted face, extra limbs, bad anatomy, watermark, logo, text, ugly, deformed, mutated, jpeg artifacts"

# CUDA marks
CUDA_SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})
CUDA_PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

# NPU marks
NPU_TWO_CARD_MARKS = hardware_marks(res={"npu": "A2"}, num_cards=2)
NPU_FOUR_CARD_MARKS = hardware_marks(res={"npu": "A2"}, num_cards=4)

T2V_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
I2V_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
TI2V_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
NPU_MODELS = [("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "i2v")]

CUDA_FEATURE_CASES = [
    ("t2v_cpu_offload", T2V_MODEL, ["--enable-cpu-offload"], CUDA_SINGLE_CARD_MARKS),
    ("t2v_ulysses_sp", T2V_MODEL, ["--usp", "2"], CUDA_PARALLEL_MARKS),
    (
        "t2v_hsdp",
        T2V_MODEL,
        ["--use-hsdp", "--hsdp-shard-size", "2"],
        CUDA_PARALLEL_MARKS,
    ),
    ("i2v_cache_dit", I2V_MODEL, ["--cache-backend", "cache_dit"], CUDA_SINGLE_CARD_MARKS),
    ("i2v_layerwise_offload", I2V_MODEL, ["--enable-layerwise-offload"], CUDA_SINGLE_CARD_MARKS),
    ("i2v_cfg_parallel", I2V_MODEL, ["--cfg-parallel-size", "2"], CUDA_PARALLEL_MARKS),
    (
        "ti2v_tp_vae_patch",
        TI2V_MODEL,
        ["--tensor-parallel-size", "2", "--vae-patch-parallel-size", "2"],
        CUDA_PARALLEL_MARKS,
    ),
    ("ti2v_ring_atten", TI2V_MODEL, ["--ring", "2"], CUDA_PARALLEL_MARKS),
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
    - CUDA: 8 paired cases covering 3 models and 8 distinct features.
    - NPU: I2V-A14B only, 2 cases (4-card combined, 2-card tp_layerwise_offload).
    """
    cases = [
        pytest.param(
            OmniServerParams(model=model_path, server_args=server_args),
            id=f"cuda_{case_id}",
            marks=marks,
        )
        for case_id, model_path, server_args, marks in CUDA_FEATURE_CASES
    ]

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
    online_client: OnlineOmniClient,
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

    online_client.send_video_diffusion_request(request_config)

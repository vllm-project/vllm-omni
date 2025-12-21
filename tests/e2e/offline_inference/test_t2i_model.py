import os
import sys
from pathlib import Path

import pytest
import torch

from vllm_omni.utils.platform_utils import is_npu, is_rocm

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"


models = ["Tongyi-MAI/Z-Image-Turbo", "riverclouds/qwen_image_random"]

# NPU still can't run Tongyi-MAI/Z-Image-Turbo properly
# Modelscope can't find riverclouds/qwen_image_random
# TODO: When NPU support is ready, remove this branch.
if is_npu():
    models = ["Qwen/Qwen-Image"]
elif is_rocm():
    # TODO: When ROCm support is ready, remove this branch.
    # vLLM V0.11.0 has issues running riverclouds/qwen_image_random
    # on ROCm
    models = ["Tongyi-MAI/Z-Image-Turbo"]


@pytest.mark.parametrize("model_name", models)
def test_diffusion_model(model_name: str):
    m = Omni(model=model_name)
    # high resolution may cause OOM on L4
    height = 256
    width = 256
    images = m.generate(
        "a photo of a cat sitting on a laptop keyboard",
        height=height,
        width=width,
        num_inference_steps=2,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(42),
        num_outputs_per_prompt=2,
    )
    assert len(images) == 2
    # check image size
    assert images[0].width == width
    assert images[0].height == height
    images[0].save("image_output.png")

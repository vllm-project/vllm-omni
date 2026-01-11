# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from pathlib import Path

import pytest
import torch

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.platform_utils import is_npu, is_rocm

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"


def _get_zimage_model() -> str:
    # Allow overriding the model for local/offline environments.
    # Can be either a HuggingFace repo id or a local path.
    return os.environ.get("VLLM_TEST_ZIMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")


@pytest.mark.integration
def test_zimage_tensor_parallel_tp2(tmp_path: Path):
    if is_npu() or is_rocm():
        pytest.skip("Z-Image TP e2e test is only supported on CUDA for now.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Z-Image TP=2 requires >= 2 CUDA devices.")

    m = Omni(
        model=_get_zimage_model(),
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
    )

    height = 256
    width = 256
    outputs = m.generate(
        "a photo of a cat sitting on a laptop keyboard",
        height=height,
        width=width,
        num_inference_steps=2,
        guidance_scale=0.0,
        seed=42,
        num_outputs_per_prompt=1,
    )

    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output[0]
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out.images
    assert images is not None
    assert len(images) == 1
    assert images[0].width == width
    assert images[0].height == height
    images[0].save(tmp_path / "zimage_tp2.png")

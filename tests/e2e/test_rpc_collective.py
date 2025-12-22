import os
import sys
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

models = ["Tongyi-MAI/Z-Image-Turbo"]


@pytest.mark.parametrize("model_name", models)
def test_diffusion_model(model_name: str):
    m = Omni(model=model_name)
    # high resolution may cause OOM on L4
    height = 256
    width = 256
    request = OmniDiffusionRequest(
        prompt="a photo of a cat sitting on a laptop keyboard",
        height=height,
        width=width,
        num_inference_steps=2,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(42),
        num_outputs_per_prompt=2,
    )
    results = m.instance.engine.collective_rpc(
        method="generate",
        args=([request],),
        kwargs={},
        unique_reply_rank=0,
    )
    assert isinstance(results, DiffusionOutput)
    assert results.output.shape[2] == width
    assert results.output.shape[3] == height

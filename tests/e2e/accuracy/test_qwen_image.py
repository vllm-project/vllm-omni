from __future__ import annotations

import base64
import gc
import io
import os
from pathlib import Path

import pytest
import requests
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from PIL import Image

from tests.e2e.accuracy.helpers import assert_similarity, model_output_dir
from tests.helpers.env import run_post_test_cleanup, run_pre_test_cleanup
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


MODEL_ID = "Qwen/Qwen-Image"
MODEL_ENV_VAR = "QWEN_IMAGE_MODEL"
PROMPT = "A photo of a cat sitting on a laptop keyboard, digital art style."
NEGATIVE_PROMPT = "blurry, low quality"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 20
TRUE_CFG_SCALE = 4.0
SEED = 42
SSIM_THRESHOLD = 0.97
PSNR_THRESHOLD = 30.0


def _model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR, MODEL_ID)


def _local_files_only(model: str) -> bool:
    return Path(model).exists()


def _run_vllm_omni_qwen_image(*, model: str, output_path: Path) -> Image.Image:
    server_args = ["--num-gpus", "1", "--stage-init-timeout", "300", "--init-timeout", "900"]
    with OmniServer(model, server_args, use_omni=True) as omni_server:
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
            json={
                "model": omni_server.model,
                "prompt": PROMPT,
                "size": f"{WIDTH}x{HEIGHT}",
                "n": 1,
                "response_format": "b64_json",
                "negative_prompt": NEGATIVE_PROMPT,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "true_cfg_scale": TRUE_CFG_SCALE,
                "seed": SEED,
            },
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        assert len(payload["data"]) == 1
        image_bytes = base64.b64decode(payload["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
        image.save(output_path)
        return image


def _run_diffusers_qwen_image(*, model: str, output_path: Path) -> Image.Image:
    run_pre_test_cleanup()
    pipe: DiffusionPipeline | None = None
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=_local_files_only(model),
        ).to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        result = pipe(  # pyright: ignore[reportCallIssue]
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            generator=generator,
        )
        output_image = result.images[0].convert("RGB")
        output_image.save(output_path)
        return output_image
    finally:
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.accelerator.empty_cache()
        run_post_test_cleanup()


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_qwen_image_matches_diffusers(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)

    vllm_output = _run_vllm_omni_qwen_image(model=model, output_path=output_dir / "vllm_omni.png")
    diffusers_output = _run_diffusers_qwen_image(model=model, output_path=output_dir / "diffusers.png")

    assert_similarity(
        model_name=MODEL_ID,
        vllm_image=vllm_output,
        diffusers_image=diffusers_output,
        width=WIDTH,
        height=HEIGHT,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )

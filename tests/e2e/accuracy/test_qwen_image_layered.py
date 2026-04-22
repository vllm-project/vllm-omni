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

from tests.e2e.accuracy.helpers import assert_image_sequence_similarity, model_output_dir
from tests.helpers.env import run_post_test_cleanup, run_pre_test_cleanup
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


MODEL_ID = "Qwen/Qwen-Image-Layered"
MODEL_ENV_VAR = "QWEN_IMAGE_LAYERED_MODEL"
PROMPT = "decompose into layers"
NEGATIVE_PROMPT = " "
NUM_INFERENCE_STEPS = 20
TRUE_CFG_SCALE = 4.0
SEED = 777
LAYERS = 3
RESOLUTION = 640
SSIM_THRESHOLD = 0.97
PSNR_THRESHOLD = 30.0


def _model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR, MODEL_ID)


def _local_files_only(model: str) -> bool:
    return Path(model).exists()


def _normalize_layered_images(images: object) -> list[Image.Image]:
    if not isinstance(images, list) or not images:
        raise AssertionError(f"Unexpected layered output container: {type(images).__name__}")

    first_item = images[0]
    if isinstance(first_item, Image.Image):
        return [image.convert("RGBA") for image in images if isinstance(image, Image.Image)]
    if isinstance(first_item, (list, tuple)):
        return [image.convert("RGBA") for image in first_item if isinstance(image, Image.Image)]
    raise AssertionError(f"Unexpected layered image element type: {type(first_item).__name__}")


def _run_vllm_omni_qwen_image_layered(*, model: str, input_image: Image.Image, output_dir: Path) -> list[Image.Image]:
    input_image.save(output_dir / "input.png")
    server_args = ["--num-gpus", "1", "--stage-init-timeout", "300", "--init-timeout", "900"]
    with OmniServer(model, server_args, use_omni=True) as omni_server:
        buffer = io.BytesIO()
        input_image.save(buffer, format="PNG")
        buffer.seek(0)
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/images/edits",
            data={
                "model": omni_server.model,
                "prompt": PROMPT,
                "size": "auto",
                "n": 1,
                "response_format": "b64_json",
                "negative_prompt": NEGATIVE_PROMPT,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "true_cfg_scale": TRUE_CFG_SCALE,
                "seed": SEED,
                "layers": LAYERS,
                "resolution": RESOLUTION,
            },
            files=[("image", ("input.png", buffer, "image/png"))],
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        assert len(payload["data"]) == LAYERS
        output_images = []
        for item in payload["data"]:
            image_bytes = base64.b64decode(item["b64_json"])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            image.load()
            output_images.append(image)
        for index, image in enumerate(output_images, start=1):
            image.save(output_dir / f"vllm_omni_layer_{index}.png")
        return output_images


def _run_diffusers_qwen_image_layered(*, model: str, input_image: Image.Image, output_dir: Path) -> list[Image.Image]:
    run_pre_test_cleanup(enable_force=True)
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
            image=input_image,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            generator=generator,
            num_images_per_prompt=1,
            layers=LAYERS,
            resolution=RESOLUTION,
        )
        output_images = _normalize_layered_images(result.images)
        assert len(output_images) == LAYERS, f"Expected {LAYERS} diffusers layers, got {len(output_images)}"
        for index, image in enumerate(output_images, start=1):
            image.save(output_dir / f"diffusers_layer_{index}.png")
        return output_images
    finally:
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_post_test_cleanup(enable_force=True)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_qwen_image_layered_matches_diffusers(accuracy_artifact_root: Path, qwen_bear_image: Image.Image) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)
    input_image = qwen_bear_image.convert("RGBA")

    vllm_outputs = _run_vllm_omni_qwen_image_layered(model=model, input_image=input_image, output_dir=output_dir)
    diffusers_outputs = _run_diffusers_qwen_image_layered(model=model, input_image=input_image, output_dir=output_dir)

    assert_image_sequence_similarity(
        model_name=MODEL_ID,
        vllm_images=vllm_outputs,
        diffusers_images=diffusers_outputs,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
        compare_mode="RGBA",
    )

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

from tests.e2e.accuracy.helpers import assert_images_pixel_close, assert_similarity, model_output_dir
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

MODEL_2512_ID = "Qwen/Qwen-Image-2512"
MODEL_2512_ENV_VAR = "QWEN_IMAGE_2512_MODEL"
PROMPT_2512 = (
    "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and "
    "lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in "
    "twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, "
    "cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at "
    "an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no "
    "staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming "
    "with vivid, fresh, youthful charm."
)
NEGATIVE_PROMPT_2512 = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。"
    "构图混乱。文字模糊，扭曲。"
)
WIDTH_2512 = 1664
HEIGHT_2512 = 928
NUM_INFERENCE_STEPS_2512 = 50
TRUE_CFG_SCALE_2512 = 4.0
SEED_2512 = 42
MEAN_ABS_DIFF_THRESHOLD_2512 = 3e-2
P99_ABS_DIFF_THRESHOLD_2512 = 4e-1


def _model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR, MODEL_ID)


def _model_2512_name() -> str:
    return os.environ.get(MODEL_2512_ENV_VAR, MODEL_2512_ID)


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
        pipe.transformer.set_attention_backend("_flash_3_hub")
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


def _run_vllm_omni_qwen_image_2512(*, model: str, output_path: Path) -> Image.Image:
    server_args = ["--num-gpus", "1", "--stage-init-timeout", "300", "--init-timeout", "900"]
    with OmniServer(model, server_args, use_omni=True) as omni_server:
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
            json={
                "model": omni_server.model,
                "prompt": PROMPT_2512,
                "size": f"{WIDTH_2512}x{HEIGHT_2512}",
                "n": 1,
                "response_format": "b64_json",
                "negative_prompt": NEGATIVE_PROMPT_2512,
                "num_inference_steps": NUM_INFERENCE_STEPS_2512,
                "true_cfg_scale": TRUE_CFG_SCALE_2512,
                "seed": SEED_2512,
            },
            timeout=1200,
        )
        response.raise_for_status()
        payload = response.json()
        assert len(payload["data"]) == 1
        image_bytes = base64.b64decode(payload["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
        image.save(output_path)
        return image


def _run_diffusers_qwen_image_2512(*, model: str, output_path: Path) -> Image.Image:
    run_pre_test_cleanup()
    pipe: DiffusionPipeline | None = None
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=_local_files_only(model),
        ).to("cuda")
        pipe.transformer.set_attention_backend("_flash_3_hub")
        generator = torch.Generator(device="cuda").manual_seed(SEED_2512)
        result = pipe(  # pyright: ignore[reportCallIssue]
            prompt=PROMPT_2512,
            negative_prompt=NEGATIVE_PROMPT_2512,
            width=WIDTH_2512,
            height=HEIGHT_2512,
            num_inference_steps=NUM_INFERENCE_STEPS_2512,
            true_cfg_scale=TRUE_CFG_SCALE_2512,
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


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_qwen_image_2512_matches_diffusers_pixelwise(accuracy_artifact_root: Path) -> None:
    model = _model_2512_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_2512_ID)
    vllm_output_path = output_dir / "vllm_omni_2512.png"
    diffusers_output_path = output_dir / "diffusers_2512.png"

    vllm_output = _run_vllm_omni_qwen_image_2512(model=model, output_path=vllm_output_path)
    diffusers_output = _run_diffusers_qwen_image_2512(model=model, output_path=diffusers_output_path)

    print(f"{MODEL_2512_ID} generated images:")
    print(f"  vllm_omni: {vllm_output_path}")
    print(f"  diffusers: {diffusers_output_path}")

    assert_images_pixel_close(
        model_name=MODEL_2512_ID,
        vllm_image=vllm_output,
        diffusers_image=diffusers_output,
        mean_threshold=MEAN_ABS_DIFF_THRESHOLD_2512,
        p99_threshold=P99_ABS_DIFF_THRESHOLD_2512,
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import pytest
import requests
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from PIL import Image

from tests.e2e.accuracy.helpers import assert_images_pixel_close, assert_similarity, model_output_dir
from tests.helpers.clean import cleanup_test_environment
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer
from vllm_omni.platforms import current_omni_platform

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
# H100 #17798, eager TORCH_SDPA vs Diffusers native: SSIM 0.989517 / PSNR 34.677647.
# A3 #9532 and #9545: SSIM 0.955357 / PSNR 27.170992. Bounds sit just under those scores.
# B200 has no eager measurement yet and uses the H100 bounds.
_CUDA_SSIM_THRESHOLD = 0.98
_CUDA_PSNR_THRESHOLD = 34.6
_NPU_SSIM_THRESHOLD = 0.95
_NPU_PSNR_THRESHOLD = 27.0

PROMPT_PIXELWISE = (
    "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and "
    "lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in "
    "twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, "
    "cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at "
    "an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no "
    "staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming "
    "with vivid, fresh, youthful charm."
)
NEGATIVE_PROMPT_PIXELWISE = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。"
    "构图混乱。文字模糊，扭曲。"
)
# CUDA keeps the 1664x928 case. A3 #9532 OOM there (61.3 GiB, weights 52.48 GiB),
# so NPU uses 768x432. Prompt and step count stay the same.
_CUDA_WIDTH_PIXELWISE = 1664
_CUDA_HEIGHT_PIXELWISE = 928
_NPU_WIDTH_PIXELWISE = 768
_NPU_HEIGHT_PIXELWISE = 432
NUM_INFERENCE_STEPS_PIXELWISE = 50
TRUE_CFG_SCALE_PIXELWISE = 4.0
SEED_PIXELWISE = 42
# H100 #17798 at 1664x928: mean abs 4.108345e-03, p99 3.529412e-02.
# A3 #9545 at 768x432: mean abs 1.399533e-02, p99 1.882353e-01.
# Bounds sit just above those errors.
_CUDA_MEAN_ABS_DIFF_THRESHOLD_PIXELWISE = 1e-2
_CUDA_P99_ABS_DIFF_THRESHOLD_PIXELWISE = 5e-2
_NPU_MEAN_ABS_DIFF_THRESHOLD_PIXELWISE = 2e-2
_NPU_P99_ABS_DIFF_THRESHOLD_PIXELWISE = 2.5e-1


def _pixelwise_size() -> tuple[int, int]:
    if current_omni_platform.is_npu():
        return _NPU_WIDTH_PIXELWISE, _NPU_HEIGHT_PIXELWISE
    return _CUDA_WIDTH_PIXELWISE, _CUDA_HEIGHT_PIXELWISE


def _similarity_thresholds() -> tuple[float, float]:
    if current_omni_platform.is_npu():
        return _NPU_SSIM_THRESHOLD, _NPU_PSNR_THRESHOLD
    return _CUDA_SSIM_THRESHOLD, _CUDA_PSNR_THRESHOLD


def _pixel_thresholds() -> tuple[float, float]:
    if current_omni_platform.is_npu():
        return _NPU_MEAN_ABS_DIFF_THRESHOLD_PIXELWISE, _NPU_P99_ABS_DIFF_THRESHOLD_PIXELWISE
    return _CUDA_MEAN_ABS_DIFF_THRESHOLD_PIXELWISE, _CUDA_P99_ABS_DIFF_THRESHOLD_PIXELWISE


def _model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR, MODEL_ID)


def _local_files_only(model: str) -> bool:
    return Path(model).exists()


# Same Omni / Diffusers pair on CUDA (H100, B200) and NPU A3.
# FLASH_ATTN_3_HUB is Hopper-only and fails on B200/A100 (#8087). Auto-select
# can still land on that kernel. TORCH_SDPA matches Diffusers ``native`` and
# has a binding on both sides.
_OMNI_CI_ATTN_BACKEND = "TORCH_SDPA"
_DIFFUSERS_CI_ATTN_BACKEND = "native"
_OMNI_CI_ATTN_ENV = {"DIFFUSION_ATTENTION_BACKEND": _OMNI_CI_ATTN_BACKEND}


def _run_vllm_omni_qwen_image(*, model: str, output_path: Path) -> Image.Image:
    server_args = [
        "--num-gpus",
        "1",
        "--stage-init-timeout",
        "300",
        "--init-timeout",
        "900",
        "--diffusion-attention-backend",
        _OMNI_CI_ATTN_BACKEND,
        "--enforce-eager",
    ]
    with OmniServer(model, server_args, use_omni=True, env_dict=_OMNI_CI_ATTN_ENV) as omni_server:
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
    cleanup_test_environment()
    pipe: DiffusionPipeline | None = None
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=_local_files_only(model),
        ).to(current_omni_platform.device_type)
        pipe.transformer.set_attention_backend(_DIFFUSERS_CI_ATTN_BACKEND)
        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED)
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
        cleanup_test_environment()


def _run_vllm_omni_qwen_image_pixelwise(*, model: str, output_path: Path) -> Image.Image:
    server_args = [
        "--num-gpus",
        "1",
        "--stage-init-timeout",
        "300",
        "--init-timeout",
        "900",
        "--diffusion-attention-backend",
        _OMNI_CI_ATTN_BACKEND,
        "--enforce-eager",
    ]
    width, height = _pixelwise_size()
    with OmniServer(model, server_args, use_omni=True, env_dict=_OMNI_CI_ATTN_ENV) as omni_server:
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
            json={
                "model": omni_server.model,
                "prompt": PROMPT_PIXELWISE,
                "size": f"{width}x{height}",
                "n": 1,
                "response_format": "b64_json",
                "negative_prompt": NEGATIVE_PROMPT_PIXELWISE,
                "num_inference_steps": NUM_INFERENCE_STEPS_PIXELWISE,
                "true_cfg_scale": TRUE_CFG_SCALE_PIXELWISE,
                "seed": SEED_PIXELWISE,
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


def _run_diffusers_qwen_image_pixelwise(*, model: str, output_path: Path) -> Image.Image:
    cleanup_test_environment()
    pipe: DiffusionPipeline | None = None
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=_local_files_only(model),
        ).to(current_omni_platform.device_type)
        pipe.transformer.set_attention_backend(_DIFFUSERS_CI_ATTN_BACKEND)
        width, height = _pixelwise_size()
        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED_PIXELWISE)
        result = pipe(  # pyright: ignore[reportCallIssue]
            prompt=PROMPT_PIXELWISE,
            negative_prompt=NEGATIVE_PROMPT_PIXELWISE,
            width=width,
            height=height,
            num_inference_steps=NUM_INFERENCE_STEPS_PIXELWISE,
            true_cfg_scale=TRUE_CFG_SCALE_PIXELWISE,
            generator=generator,
        )
        output_image = result.images[0].convert("RGB")
        output_image.save(output_path)
        return output_image
    finally:
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        del pipe
        cleanup_test_environment()


@pytest.mark.benchmark
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=1)
def test_qwen_image_matches_diffusers(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)

    vllm_output = _run_vllm_omni_qwen_image(model=model, output_path=output_dir / "vllm_omni.png")
    diffusers_output = _run_diffusers_qwen_image(model=model, output_path=output_dir / "diffusers.png")

    ssim_threshold, psnr_threshold = _similarity_thresholds()
    assert_similarity(
        model_name=MODEL_ID,
        vllm_image=vllm_output,
        diffusers_image=diffusers_output,
        width=WIDTH,
        height=HEIGHT,
        ssim_threshold=ssim_threshold,
        psnr_threshold=psnr_threshold,
    )


@pytest.mark.benchmark
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=1)
def test_qwen_image_matches_diffusers_pixelwise(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)
    vllm_output_path = output_dir / "vllm_omni_pixelwise.png"
    diffusers_output_path = output_dir / "diffusers_pixelwise.png"

    vllm_output = _run_vllm_omni_qwen_image_pixelwise(model=model, output_path=vllm_output_path)
    diffusers_output = _run_diffusers_qwen_image_pixelwise(model=model, output_path=diffusers_output_path)

    print(f"{MODEL_ID} pixelwise generated images:")
    print(f"  vllm_omni: {vllm_output_path}")
    print(f"  diffusers: {diffusers_output_path}")

    mean_threshold, p99_threshold = _pixel_thresholds()
    assert_images_pixel_close(
        model_name=MODEL_ID,
        vllm_image=vllm_output,
        diffusers_image=diffusers_output,
        mean_threshold=mean_threshold,
        p99_threshold=p99_threshold,
    )

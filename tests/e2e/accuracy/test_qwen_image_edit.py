from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch
from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline
from PIL import Image

from benchmarks.accuracy.common import pil_to_data_url
from tests.conftest import (
    OmniServer,
    OpenAIClientHandler,
    _run_post_test_cleanup,
    _run_pre_test_cleanup,
    dummy_messages_from_mix_data,
)
from tests.e2e.accuracy.utils import assert_similarity, model_output_dir
from tests.utils import hardware_test

SINGLE_MODEL = "Qwen/Qwen-Image-Edit"
MULTIPLE_MODEL = "Qwen/Qwen-Image-Edit-2509"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 20
TRUE_CFG_SCALE = 4.0
SEED = 1234
SSIM_THRESHOLD = 0.94
PSNR_THRESHOLD = 28.0

PROMPT_SINGLE_IMAGE = "The input is a 2D cartoon bear mascot. Restyle it into a painterly oil artwork with warm colors while preserving the main structure."
PROMPT_MULTIPLE_IMAGE = "The first input is a 2D cartoon bear mascot and the second input is a furry rabbit. Blend them into one coherent scene with a cinematic style and consistent lighting."
NEGATIVE_PROMPT = "low quality, blurry, artifacts, distortion"
SERVER_ARGS = ["--num-gpus", "1", "--stage-init-timeout", "300", "--init-timeout", "900"]


def _run_vllm_omni_image_edit(
    *,
    model: str,
    openai_client: OpenAIClientHandler,
    prompt: str,
    input_image_urls: list[str],
    output_path: Path,
) -> Image.Image:
    messages = dummy_messages_from_mix_data(
        image_data_url=input_image_urls,
        content_text=prompt,
    )

    request_config = {
        "model": model,
        "messages": messages,
        "extra_body": {
            "height": HEIGHT,
            "width": WIDTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": TRUE_CFG_SCALE,
            "seed": SEED,
        },
    }

    diffusion_response = openai_client.send_diffusion_request(request_config)
    image = diffusion_response[0].images[0]
    image.save(output_path)
    return image


def _run_diffusers_image_edit(
    *,
    model: str,
    pipeline_class: type[QwenImageEditPipeline] | type[QwenImageEditPlusPipeline],
    prompt: str,
    input_images: list[Image.Image],
    output_path: Path,
) -> Image.Image:
    _run_pre_test_cleanup(enable_force=True)
    pipe: QwenImageEditPipeline | QwenImageEditPlusPipeline | None = None
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    try:
        images = input_images[0] if len(input_images) == 1 else input_images
        pipe = pipeline_class.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        pipe.set_progress_bar_config(disable=False)
        generator = torch.Generator(device=device).manual_seed(SEED)
        result = pipe(  # pyright: ignore[reportCallIssue]
            prompt=prompt,
            image=images,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            width=WIDTH,
            height=HEIGHT,
            generator=generator,
        )
        output_image = result.images[0].convert("RGB")  # pyright: ignore[reportAttributeAccessIssue]
        output_image.save(output_path)
        return output_image
    finally:
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _run_post_test_cleanup(enable_force=True)


def _vllm_omni_output_single_image(
    openai_client: OpenAIClientHandler,
    accuracy_artifact_root: Path,
    qwen_bear_image: Image.Image,
) -> Image.Image:
    output_dir = model_output_dir(accuracy_artifact_root, SINGLE_MODEL)
    output_path = output_dir / "vllm_omni_single.png"
    if output_path.exists():
        return Image.open(output_path)
    with OmniServer(model=SINGLE_MODEL, serve_args=SERVER_ARGS):
        output = _run_vllm_omni_image_edit(
            model=SINGLE_MODEL,
            openai_client=openai_client,
            prompt=PROMPT_SINGLE_IMAGE,
            input_image_urls=[pil_to_data_url(qwen_bear_image)],
            output_path=output_path,
        )
    return output


def _diffusers_output_single_image(accuracy_artifact_root: Path, qwen_bear_image: Image.Image) -> Image.Image:
    output_dir = model_output_dir(accuracy_artifact_root, SINGLE_MODEL)
    output_path = output_dir / "diffusers_single.png"
    if output_path.exists():
        return Image.open(output_path)
    return _run_diffusers_image_edit(
        model=SINGLE_MODEL,
        pipeline_class=QwenImageEditPipeline,
        prompt=PROMPT_SINGLE_IMAGE,
        input_images=[qwen_bear_image],
        output_path=output_path,
    )


def _vllm_omni_output_multiple_image(
    openai_client: OpenAIClientHandler,
    accuracy_artifact_root: Path,
    qwen_bear_image: Image.Image,
    rabbit_image: Image.Image,
) -> Image.Image:
    output_dir = model_output_dir(accuracy_artifact_root, MULTIPLE_MODEL)
    output_path = output_dir / "vllm_omni_multiple.png"
    if output_path.exists():
        return Image.open(output_path)
    with OmniServer(model=MULTIPLE_MODEL, serve_args=SERVER_ARGS):
        output = _run_vllm_omni_image_edit(
            model=MULTIPLE_MODEL,
            openai_client=openai_client,
            prompt=PROMPT_MULTIPLE_IMAGE,
            input_image_urls=[pil_to_data_url(qwen_bear_image), pil_to_data_url(rabbit_image)],
            output_path=output_path,
        )
    return output


def _diffusers_output_multiple_image(
    accuracy_artifact_root: Path, qwen_bear_image: Image.Image, rabbit_image: Image.Image
) -> Image.Image:
    output_dir = model_output_dir(accuracy_artifact_root, MULTIPLE_MODEL)
    output_path = output_dir / "diffusers_multiple.png"
    if output_path.exists():
        return Image.open(output_path)
    return _run_diffusers_image_edit(
        model=MULTIPLE_MODEL,
        pipeline_class=QwenImageEditPlusPipeline,
        prompt=PROMPT_MULTIPLE_IMAGE,
        input_images=[qwen_bear_image, rabbit_image],
        output_path=output_path,
    )


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_qwen_image_edit_single_matches_diffusers(
    openai_client: OpenAIClientHandler,
    accuracy_artifact_root: Path,
    qwen_bear_image: Image.Image,
) -> None:
    vllm_image = _vllm_omni_output_single_image(
        openai_client=openai_client,
        accuracy_artifact_root=accuracy_artifact_root,
        qwen_bear_image=qwen_bear_image,
    )
    diffusers_image = _diffusers_output_single_image(
        accuracy_artifact_root=accuracy_artifact_root,
        qwen_bear_image=qwen_bear_image,
    )
    assert_similarity(
        model_name=SINGLE_MODEL,
        vllm_image=vllm_image,
        diffusers_image=diffusers_image,
        width=WIDTH,
        height=HEIGHT,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_qwen_image_edit_multiple_matches_diffusers(
    openai_client: OpenAIClientHandler,
    accuracy_artifact_root: Path,
    qwen_bear_image: Image.Image,
    rabbit_image: Image.Image,
) -> None:
    vllm_image = _vllm_omni_output_multiple_image(
        openai_client=openai_client,
        accuracy_artifact_root=accuracy_artifact_root,
        qwen_bear_image=qwen_bear_image,
        rabbit_image=rabbit_image,
    )
    diffusers_image = _diffusers_output_multiple_image(
        accuracy_artifact_root=accuracy_artifact_root,
        qwen_bear_image=qwen_bear_image,
        rabbit_image=rabbit_image,
    )
    assert_similarity(
        model_name=MULTIPLE_MODEL,
        vllm_image=vllm_image,
        diffusers_image=diffusers_image,
        width=WIDTH,
        height=HEIGHT,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )

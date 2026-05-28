from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import requests
import torch
from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline
from PIL import Image

from benchmarks.accuracy.common import decode_base64_image, pil_to_png_bytes
from tests.e2e.accuracy.helpers import assert_images_pixel_close, assert_similarity, model_output_dir
from tests.helpers.env import run_post_test_cleanup, run_pre_test_cleanup
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


SINGLE_MODEL = "Qwen/Qwen-Image-Edit"
MULTIPLE_MODEL = "Qwen/Qwen-Image-Edit-2511"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 20
TRUE_CFG_SCALE = 4.0
SEED = 42
SSIM_THRESHOLD = 0.94
PSNR_THRESHOLD = 28.0

EDIT_2511_MODEL = "Qwen/Qwen-Image-Edit-2511"
EDIT_2511_MODEL_ENV_VAR = "QWEN_IMAGE_EDIT_2511_MODEL"
EDIT_2511_WIDTH = 1190
EDIT_2511_HEIGHT = 2032
EDIT_2511_NUM_INFERENCE_STEPS = 40
EDIT_2511_TRUE_CFG_SCALE = 4.0
EDIT_2511_GUIDANCE_SCALE = 1.0
EDIT_2511_SEED = 42
EDIT_2511_MEAN_ABS_DIFF_THRESHOLD = 3e-2
EDIT_2511_P99_ABS_DIFF_THRESHOLD = 4e-1
EDIT_2511_NEGATIVE_PROMPT = " "
EDIT_2511_PROMPT = (
    "将第二张图中人脸/猫狗脸转换为3D卡通形象，质量要求：毛发纹理自然电影级色彩校准，注意保留原图的外貌、"
    "肤色、发型特征，仅替换第一张财神爷/猫狗财神的用绿框框住的头部区域，耳朵样式和原图的3D卡通形象一致， "
    "（帽子紧贴头顶，未贴合部分可调整帽子摆放角度达到完全贴合）， 头部与身体过渡必须自然，无拼接痕迹，"
    "无额外元素出现在头部四周，纯白背景，不要出现绿色圆圈"
)
EDIT_2511_ASSET_DIR = Path(__file__).resolve().parent / "assets"
EDIT_2511_IMAGE_PATHS = [
    EDIT_2511_ASSET_DIR / "qwen_image_edit_2511_test1.png",
    EDIT_2511_ASSET_DIR / "qwen_image_edit_2511_test2.png",
]

PROMPT_SINGLE_IMAGE = "The input is a 2D cartoon bear mascot. Restyle it into a painterly oil artwork with warm colors while preserving the main structure."
PROMPT_MULTIPLE_IMAGE = "Put the cartoon bear mascot and the furry rabbit into one coherent scene with a painterly oil artwork style and consistent lighting."
NEGATIVE_PROMPT = "low quality, blurry, artifacts, distortion"
SERVER_ARGS = ["--num-gpus", "1", "--stage-init-timeout", "300", "--init-timeout", "900"]


def _edit_2511_model_name() -> str:
    return os.environ.get(EDIT_2511_MODEL_ENV_VAR, EDIT_2511_MODEL)


def _local_files_only(model: str) -> bool:
    return Path(model).exists()


def _run_vllm_omni_image_edit(
    *,
    omni_server: OmniServer,
    prompt: str,
    input_images: list[Image.Image],
    output_path: Path,
) -> Image.Image:
    response = requests.post(
        f"http://{omni_server.host}:{omni_server.port}/v1/images/edits",
        data={
            "model": omni_server.model,
            "prompt": prompt,
            "size": f"{WIDTH}x{HEIGHT}",
            "n": 1,
            "response_format": "b64_json",
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "true_cfg_scale": TRUE_CFG_SCALE,
            "seed": SEED,
        },
        files=[
            ("image", (f"image_{index}.png", pil_to_png_bytes(image), "image/png"))
            for index, image in enumerate(input_images)
        ],
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    assert len(payload["data"]) == 1
    image = decode_base64_image(payload["data"][0]["b64_json"])
    image.load()
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
    run_pre_test_cleanup()
    pipe: QwenImageEditPipeline | QwenImageEditPlusPipeline | None = None
    try:
        images = input_images[0] if len(input_images) == 1 else input_images
        pipe = pipeline_class.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")
        pipe.set_progress_bar_config(disable=False)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
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
            torch.accelerator.empty_cache()
        run_post_test_cleanup()


def _load_edit_2511_input_images() -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in EDIT_2511_IMAGE_PATHS]


def _run_vllm_omni_image_edit_2511(
    *,
    model: str,
    input_images: list[Image.Image],
    output_path: Path,
) -> Image.Image:
    with OmniServer(model=model, serve_args=SERVER_ARGS) as server:
        response = requests.post(
            f"http://{server.host}:{server.port}/v1/images/edits",
            data={
                "model": server.model,
                "prompt": EDIT_2511_PROMPT,
                "size": f"{EDIT_2511_WIDTH}x{EDIT_2511_HEIGHT}",
                "n": 1,
                "response_format": "b64_json",
                "negative_prompt": EDIT_2511_NEGATIVE_PROMPT,
                "num_inference_steps": EDIT_2511_NUM_INFERENCE_STEPS,
                "true_cfg_scale": EDIT_2511_TRUE_CFG_SCALE,
                "guidance_scale": EDIT_2511_GUIDANCE_SCALE,
                "seed": EDIT_2511_SEED,
            },
            files=[
                ("image", (f"image_{index}.png", pil_to_png_bytes(image), "image/png"))
                for index, image in enumerate(input_images)
            ],
            timeout=1200,
        )
        response.raise_for_status()
        payload = response.json()
        assert len(payload["data"]) == 1
        image = decode_base64_image(payload["data"][0]["b64_json"])
        image.load()
        image.save(output_path)
        return image


def _run_diffusers_image_edit_2511(
    *,
    model: str,
    input_images: list[Image.Image],
    output_path: Path,
) -> Image.Image:
    run_pre_test_cleanup()
    pipe: QwenImageEditPlusPipeline | None = None
    try:
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=_local_files_only(model),
        ).to("cuda")
        pipe.set_progress_bar_config(disable=False)
        generator = torch.Generator(device="cuda").manual_seed(EDIT_2511_SEED)
        result = pipe(  # pyright: ignore[reportCallIssue]
            image=input_images,
            prompt=EDIT_2511_PROMPT,
            generator=generator,
            true_cfg_scale=EDIT_2511_TRUE_CFG_SCALE,
            negative_prompt=EDIT_2511_NEGATIVE_PROMPT,
            num_inference_steps=EDIT_2511_NUM_INFERENCE_STEPS,
            guidance_scale=EDIT_2511_GUIDANCE_SCALE,
            num_images_per_prompt=1,
            width=EDIT_2511_WIDTH,
            height=EDIT_2511_HEIGHT,
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
            torch.accelerator.empty_cache()
        run_post_test_cleanup()


def _vllm_omni_output_single_image(
    accuracy_artifact_root: Path,
    qwen_bear_image: Image.Image,
) -> Image.Image:
    output_dir = model_output_dir(accuracy_artifact_root, SINGLE_MODEL)
    output_path = output_dir / "vllm_omni_single.png"
    with OmniServer(model=SINGLE_MODEL, serve_args=SERVER_ARGS) as server:
        output = _run_vllm_omni_image_edit(
            omni_server=server,
            prompt=PROMPT_SINGLE_IMAGE,
            input_images=[qwen_bear_image],
            output_path=output_path,
        )
    return output


def _diffusers_output_single_image(accuracy_artifact_root: Path, qwen_bear_image: Image.Image) -> Image.Image:
    output_dir = model_output_dir(accuracy_artifact_root, SINGLE_MODEL)
    output_path = output_dir / "diffusers_single.png"
    return _run_diffusers_image_edit(
        model=SINGLE_MODEL,
        pipeline_class=QwenImageEditPipeline,
        prompt=PROMPT_SINGLE_IMAGE,
        input_images=[qwen_bear_image],
        output_path=output_path,
    )


def _vllm_omni_output_multiple_image(
    accuracy_artifact_root: Path,
    qwen_bear_image: Image.Image,
    rabbit_image: Image.Image,
) -> Image.Image:
    output_dir = model_output_dir(accuracy_artifact_root, MULTIPLE_MODEL)
    output_path = output_dir / "vllm_omni_multiple.png"
    with OmniServer(model=MULTIPLE_MODEL, serve_args=SERVER_ARGS) as server:
        output = _run_vllm_omni_image_edit(
            omni_server=server,
            prompt=PROMPT_MULTIPLE_IMAGE,
            input_images=[qwen_bear_image, rabbit_image],
            output_path=output_path,
        )
    return output


def _diffusers_output_multiple_image(
    accuracy_artifact_root: Path, qwen_bear_image: Image.Image, rabbit_image: Image.Image
) -> Image.Image:
    output_dir = model_output_dir(accuracy_artifact_root, MULTIPLE_MODEL)
    output_path = output_dir / "diffusers_multiple.png"
    return _run_diffusers_image_edit(
        model=MULTIPLE_MODEL,
        pipeline_class=QwenImageEditPlusPipeline,
        prompt=PROMPT_MULTIPLE_IMAGE,
        input_images=[qwen_bear_image, rabbit_image],
        output_path=output_path,
    )


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_qwen_image_edit_single_matches_diffusers(
    accuracy_artifact_root: Path,
    qwen_bear_image: Image.Image,
) -> None:
    vllm_image = _vllm_omni_output_single_image(
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


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skip(
    reason="Skipping as the second image seems to be ignored by the API. Will come back to this later after #2772 is merged."
)
def test_qwen_image_edit_multiple_matches_diffusers(
    accuracy_artifact_root: Path,
    qwen_bear_image: Image.Image,
    rabbit_image: Image.Image,
) -> None:
    vllm_image = _vllm_omni_output_multiple_image(
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


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_qwen_image_edit_2511_matches_diffusers_pixelwise(accuracy_artifact_root: Path) -> None:
    model = _edit_2511_model_name()
    output_dir = model_output_dir(accuracy_artifact_root, EDIT_2511_MODEL)
    input_images = _load_edit_2511_input_images()
    input_paths: list[Path] = []
    for index, image in enumerate(input_images, start=1):
        input_path = output_dir / f"edit_2511_input_{index}.png"
        image.save(input_path)
        input_paths.append(input_path)
    vllm_output_path = output_dir / "vllm_omni_edit_2511.png"
    diffusers_output_path = output_dir / "diffusers_edit_2511.png"

    vllm_image = _run_vllm_omni_image_edit_2511(
        model=model,
        input_images=input_images,
        output_path=vllm_output_path,
    )
    diffusers_image = _run_diffusers_image_edit_2511(
        model=model,
        input_images=input_images,
        output_path=diffusers_output_path,
    )

    print(f"{EDIT_2511_MODEL} generated images:")
    for index, input_path in enumerate(input_paths, start=1):
        print(f"  input_{index}: {input_path}")
    print(f"  vllm_omni: {vllm_output_path}")
    print(f"  diffusers: {diffusers_output_path}")

    assert_images_pixel_close(
        model_name=EDIT_2511_MODEL,
        vllm_image=vllm_image,
        diffusers_image=diffusers_image,
        mean_threshold=EDIT_2511_MEAN_ABS_DIFF_THRESHOLD,
        p99_threshold=EDIT_2511_P99_ABS_DIFF_THRESHOLD,
    )

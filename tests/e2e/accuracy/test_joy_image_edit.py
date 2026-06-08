from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from PIL import Image

from benchmarks.accuracy.common import decode_base64_image, pil_to_png_bytes
from tests.e2e.accuracy.helpers import assert_similarity, model_output_dir
from tests.helpers.env import run_post_test_cleanup, run_pre_test_cleanup
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


MODEL_ID = "jdopensource/JoyAI-Image-Edit-Diffusers"
MODEL_ENV_VAR = "JOY_IMAGE_EDIT_MODEL"
INPUT_IMAGE_PATH = Path(__file__).resolve().parents[3] / "docs" / "assets" / "WeChat.jpg"
PROMPT = "Change the background to a clean studio while preserving the subject."
NEGATIVE_PROMPT = ""
WIDTH = 1024
HEIGHT = 1024
NUM_INFERENCE_STEPS = 20
TRUE_CFG_SCALE = 4.0
SEED = 42
SSIM_THRESHOLD = 0.90
PSNR_THRESHOLD = 25.0
SERVER_ARGS = [
    "--num-gpus",
    "1",
    "--enforce-eager",
    "--stage-init-timeout",
    "900",
    "--init-timeout",
    "1200",
]


def _model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR, MODEL_ID)


def _local_files_only(model: str) -> bool:
    return Path(model).exists()


def _joy_image_edit_pipeline_cls() -> type[Any]:
    diffusers = pytest.importorskip("diffusers")
    pipeline_cls = getattr(diffusers, "JoyImageEditPipeline", None)
    if pipeline_cls is None:
        pytest.skip("diffusers.JoyImageEditPipeline is required for JoyAI-Image-Edit parity testing.")
    return pipeline_cls


def _load_input_image() -> Image.Image:
    if not INPUT_IMAGE_PATH.exists():
        raise AssertionError(f"JoyAI input image does not exist: {INPUT_IMAGE_PATH}")
    image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    image.load()
    return image


def _run_vllm_omni_joy_image_edit(
    *,
    model: str,
    input_image: Image.Image,
    output_path: Path,
) -> Image.Image:
    with OmniServer(model=model, serve_args=SERVER_ARGS) as server:
        response = requests.post(
            f"http://{server.host}:{server.port}/v1/images/edits",
            data={
                "model": server.model,
                "prompt": PROMPT,
                "size": f"{WIDTH}x{HEIGHT}",
                "n": 1,
                "response_format": "b64_json",
                "negative_prompt": NEGATIVE_PROMPT,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "true_cfg_scale": TRUE_CFG_SCALE,
                "seed": SEED,
            },
            files=[("image", ("input.png", pil_to_png_bytes(input_image), "image/png"))],
            timeout=1200,
        )
        response.raise_for_status()
        payload = response.json()
        assert len(payload["data"]) == 1
        image = decode_base64_image(payload["data"][0]["b64_json"])
        image.load()
        image.save(output_path)
        return image


def _run_diffusers_joy_image_edit(
    *,
    model: str,
    input_image: Image.Image,
    output_path: Path,
) -> Image.Image:
    pipeline_cls = _joy_image_edit_pipeline_cls()
    run_pre_test_cleanup()
    pipe = None
    try:
        pipe = pipeline_cls.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=_local_files_only(model),
        ).to("cuda")
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=False)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        result = pipe(
            prompt=PROMPT,
            image=input_image,
            negative_prompt=NEGATIVE_PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=TRUE_CFG_SCALE,
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
def test_joy_image_edit_matches_diffusers(accuracy_artifact_root: Path) -> None:
    _joy_image_edit_pipeline_cls()
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)
    input_image = _load_input_image()
    input_path = output_dir / "input.png"
    vllm_output_path = output_dir / "vllm_omni.png"
    diffusers_output_path = output_dir / "diffusers.png"
    input_image.save(input_path)

    vllm_image = _run_vllm_omni_joy_image_edit(
        model=model,
        input_image=input_image,
        output_path=vllm_output_path,
    )
    diffusers_image = _run_diffusers_joy_image_edit(
        model=model,
        input_image=input_image,
        output_path=diffusers_output_path,
    )

    print(f"{MODEL_ID} generated images:")
    print(f"  input: {input_path}")
    print(f"  vllm_omni: {vllm_output_path}")
    print(f"  diffusers: {diffusers_output_path}")

    assert_similarity(
        model_name=MODEL_ID,
        vllm_image=vllm_image,
        diffusers_image=diffusers_image,
        width=WIDTH,
        height=HEIGHT,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )

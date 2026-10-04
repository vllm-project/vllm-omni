# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from PIL import Image
from vllm.config.load import LoadConfig

from benchmarks.accuracy.common import decode_base64_image, pil_to_png_bytes
from tests.e2e.accuracy.helpers import assert_similarity, model_output_dir
from tests.helpers.env import run_post_test_cleanup, run_pre_test_cleanup
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

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


def _latent_metrics(
    vllm_latents: torch.Tensor,
    diffusers_latents: torch.Tensor,
) -> dict[str, float]:
    vllm_flat = vllm_latents.detach().float().cpu().flatten()
    diffusers_flat = diffusers_latents.detach().float().cpu().flatten()
    if vllm_flat.shape != diffusers_flat.shape:
        raise AssertionError(
            "Latent outputs have different shapes: "
            f"vllm_omni={tuple(vllm_latents.shape)}, diffusers={tuple(diffusers_latents.shape)}"
        )
    delta = vllm_flat - diffusers_flat
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rmse": torch.sqrt(torch.mean(delta.square())).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            vllm_flat.unsqueeze(0),
            diffusers_flat.unsqueeze(0),
        ).item(),
    }


def _print_latent_metrics(metrics: dict[str, float]) -> None:
    print(f"{MODEL_ID} latent metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.8f}")


def _ensure_diffusion_distributed_initialized() -> None:
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29513")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
    if not model_parallel_is_initialized():
        initialize_model_parallel()


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


class _JoyImageEditRequest:
    def __init__(self, input_image: Image.Image) -> None:
        self.prompts = [
            {
                "prompt": PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "multi_modal_data": {"image": input_image},
            }
        ]
        self.sampling_params = OmniDiffusionSamplingParams(
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            seed=SEED,
        )
        self.request_id = "joy-image-edit-latent-parity"

    def is_dummy_run(self) -> bool:
        return False


def _load_vllm_omni_joy_image_edit_pipeline(model: str):
    from vllm_omni.diffusion.models.joy_image.pipeline_joy_image_edit import (
        JoyImageEditPipeline,
    )

    _ensure_diffusion_distributed_initialized()
    od_config = OmniDiffusionConfig(
        model=model,
        model_class_name="JoyImageEditPipeline",
        dtype=torch.bfloat16,
        enforce_eager=True,
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    return loader.load_model(
        load_device="cuda",
        load_format="custom_pipeline",
        custom_pipeline_name=JoyImageEditPipeline,
    )


def _extract_diffusers_latents(result: Any) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result
    images = getattr(result, "images", None)
    if isinstance(images, torch.Tensor):
        return images
    raise TypeError(f"Unexpected Diffusers latent output type: {type(result)!r}")


def _run_latent_parity_comparison(
    *,
    model: str,
    input_image: Image.Image,
) -> dict[str, float]:
    pipeline_cls = _joy_image_edit_pipeline_cls()
    run_pre_test_cleanup()
    diffusers_pipe = None
    try:
        diffusers_pipe = pipeline_cls.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=_local_files_only(model),
        ).to("cuda")
        if hasattr(diffusers_pipe, "set_progress_bar_config"):
            diffusers_pipe.set_progress_bar_config(disable=False)

        processed_image = diffusers_pipe.vae_image_processor.resize_center_crop(
            input_image,
            (HEIGHT, WIDTH),
        )
        prompt_embeds, prompt_embeds_mask = diffusers_pipe.encode_prompt_multiple_images(
            prompt=PROMPT,
            images=processed_image,
            device=torch.device("cuda"),
            num_images_per_prompt=1,
            max_sequence_length=4096,
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = diffusers_pipe.encode_prompt_multiple_images(
            prompt=NEGATIVE_PROMPT,
            images=processed_image,
            device=torch.device("cuda"),
            num_images_per_prompt=1,
            max_sequence_length=4096,
        )
        num_channels_latents = int(diffusers_pipe.transformer.config.in_channels)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        fixed_latents = torch.randn(
            1,
            1,
            num_channels_latents,
            1,
            HEIGHT // diffusers_pipe.vae_scale_factor_spatial,
            WIDTH // diffusers_pipe.vae_scale_factor_spatial,
            generator=generator,
            device="cuda",
            dtype=prompt_embeds.dtype,
        )
        diffusers_result = diffusers_pipe(
            prompt=None,
            image=input_image,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=TRUE_CFG_SCALE,
            latents=fixed_latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            output_type="latent",
        )
        diffusers_latents = _extract_diffusers_latents(diffusers_result).detach().cpu()
        prompt_embeds = prompt_embeds.detach().cpu()
        prompt_embeds_mask = prompt_embeds_mask.detach().cpu()
        negative_prompt_embeds = negative_prompt_embeds.detach().cpu()
        negative_prompt_embeds_mask = negative_prompt_embeds_mask.detach().cpu()
        fixed_latents = fixed_latents.detach().cpu()
    finally:
        if diffusers_pipe is not None and hasattr(diffusers_pipe, "maybe_free_model_hooks"):
            diffusers_pipe.maybe_free_model_hooks()
        del diffusers_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.accelerator.empty_cache()
        run_post_test_cleanup()

    run_pre_test_cleanup()
    vllm_pipe = None
    try:
        from vllm_omni.diffusion.models.joy_image.pipeline_joy_image_edit import (
            get_joy_image_edit_pre_process_func,
        )

        vllm_pipe = _load_vllm_omni_joy_image_edit_pipeline(model)
        vllm_pipe.eval()
        request = _JoyImageEditRequest(input_image)
        request = get_joy_image_edit_pre_process_func(vllm_pipe.od_config)(request)
        with torch.inference_mode():
            vllm_result = vllm_pipe(
                request,
                prompt_embeds=prompt_embeds.to(device=vllm_pipe.device, dtype=torch.bfloat16),
                prompt_embeds_mask=prompt_embeds_mask.to(device=vllm_pipe.device),
                negative_prompt_embeds=negative_prompt_embeds.to(device=vllm_pipe.device, dtype=torch.bfloat16),
                negative_prompt_embeds_mask=negative_prompt_embeds_mask.to(device=vllm_pipe.device),
                latents=fixed_latents.to(device=vllm_pipe.device, dtype=torch.bfloat16),
                output_type="latent",
            )
        vllm_latents = vllm_result.output
    finally:
        if vllm_pipe is not None and hasattr(vllm_pipe, "maybe_free_model_hooks"):
            vllm_pipe.maybe_free_model_hooks()
        del vllm_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.accelerator.empty_cache()
        run_post_test_cleanup()

    return _latent_metrics(vllm_latents, diffusers_latents)


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
    _print_latent_metrics(
        _run_latent_parity_comparison(
            model=model,
            input_image=input_image,
        )
    )

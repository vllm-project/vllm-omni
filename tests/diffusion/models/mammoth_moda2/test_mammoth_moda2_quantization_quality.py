# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""BF16 versus online-FP8 quality and memory test for MammothModa2."""

from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from vllm.sampling_params import SamplingParams

from tests.diffusion.quantization.test_quantization_quality import (
    _compute_lpips,
    _compute_psnr_and_mae,
)
from tests.helpers.mark import hardware_test
from tests.helpers.monitor import DeviceMemoryMonitor
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.repo_utils import hf_api

MODEL_PATH = "bytedance-research/MammothModa2-Preview"
DEPLOY_CONFIG = get_deploy_config_path("mammoth_moda2.yaml")

HEIGHT = 256
WIDTH = 256
NUM_INFERENCE_STEPS = 2
SEED = 42
MAX_LPIPS = 0.15
_AR_PATCH_SIZE = 16
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653


def _load_generation_config() -> dict:
    weights_dir = Path(hf_api().snapshot_download(MODEL_PATH))
    config_path = weights_dir / "t2i_generation_config.json"
    if not config_path.is_file():
        pytest.skip(f"t2i_generation_config.json not found at {config_path}")
    return json.loads(config_path.read_text())


def _format_prompt(user_prompt: str, ar_width: int, ar_height: int) -> str:
    return (
        "<|im_start|>system\nYou are a helpful image generator.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"<|image start|>{ar_width}*{ar_height}<|image token|>"
    )


def _extract_image_and_worker_peak(outputs) -> tuple[torch.Tensor, float]:
    worker_peak_memory_mb = 0.0
    for output_group in outputs:
        request_outputs = output_group if isinstance(output_group, list) else [output_group]
        for request_output in request_outputs:
            worker_peak_memory_mb = max(
                worker_peak_memory_mb,
                float(getattr(request_output, "peak_memory_mb", 0.0) or 0.0),
            )
            completions = getattr(request_output, "outputs", None)
            if not isinstance(completions, list):
                continue
            for completion in completions:
                multimodal = getattr(completion, "multimodal_output", None)
                if not isinstance(multimodal, Mapping) or "image" not in multimodal:
                    continue
                images = multimodal["image"]
                image = images[0] if isinstance(images, list) else images
                if not isinstance(image, torch.Tensor):
                    raise TypeError(f"Expected an image tensor, got {type(image)!r}")
                if image.ndim == 4:
                    image = image[0]
                if image.ndim != 3:
                    raise ValueError(f"Expected a CHW image tensor, got shape {tuple(image.shape)}")
                return image.detach().float().cpu(), worker_peak_memory_mb
    raise ValueError("MammothModa2 pipeline produced no image tensor")


def _to_pil(image: torch.Tensor) -> Image.Image:
    image = image.float()
    # Mammoth's VAE currently returns an unprocessed tensor. Support both its
    # conventional [-1, 1] output and an already-normalized [0, 1] output.
    if float(image.min()) < 0.0:
        image = image * 0.5 + 0.5
    array = image.clamp(0.0, 1.0).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _generate(quantization: str | None) -> tuple[Image.Image, float, float]:
    generation_config = _load_generation_config()
    ar_height = HEIGHT // _AR_PATCH_SIZE
    ar_width = WIDTH // _AR_PATCH_SIZE
    expected_grid_tokens = ar_height * (ar_width + 1)
    prompt = _format_prompt("A cat sitting on a laptop keyboard", ar_width, ar_height)

    ar_sampling = SamplingParams(
        temperature=0.0,
        top_k=1,
        max_tokens=expected_grid_tokens + 1,
        detokenize=False,
        seed=SEED,
    )
    dit_sampling = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False, seed=SEED)
    request = {
        "prompt": prompt,
        "additional_information": {
            "omni_task": ["t2i"],
            "ar_width": [ar_width],
            "ar_height": [ar_height],
            "eol_token_id": [int(generation_config["eol_token_id"])],
            "visual_token_start_id": [int(generation_config["visual_token_start_id"])],
            "visual_token_end_id": [int(generation_config["visual_token_end_id"])],
            "image_height": [HEIGHT],
            "image_width": [WIDTH],
            "num_inference_steps": [NUM_INFERENCE_STEPS],
            "text_guidance_scale": [1.0],
            "cfg_range": [0.0, 1.0],
            "visual_ids": [
                _IMAGE_TOKEN_ID,
                _VIDEO_TOKEN_ID,
                _VISION_START_TOKEN_ID,
                _VISION_END_TOKEN_ID,
            ],
        },
    }

    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()
    try:
        runner_kwargs = {
            "deploy_config": DEPLOY_CONFIG,
            "enforce_eager": True,
        }
        if quantization is not None:
            runner_kwargs["quantization"] = quantization
        with OmniRunner(MODEL_PATH, seed=SEED, **runner_kwargs) as runner:
            outputs = list(runner.omni.generate([request], [ar_sampling, dit_sampling]))
            image, worker_peak_memory_mb = _extract_image_and_worker_peak(outputs)
        device_peak_memory_mb = monitor.peak_used_mb
    finally:
        monitor.stop()
        gc.collect()
        current_omni_platform.empty_cache()

    assert torch.isfinite(image).all(), "Generated image contains non-finite values"
    return _to_pil(image), device_peak_memory_mb, worker_peak_memory_mb


@pytest.mark.full_model
@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
def test_mammoth_moda2_online_fp8_quality_and_memory():
    baseline, bf16_device_mem, bf16_worker_mem = _generate(quantization=None)
    quantized, fp8_device_mem, fp8_worker_mem = _generate(quantization="fp8")

    lpips_score = _compute_lpips(baseline, quantized, "t2i")
    psnr_score, mae_score = _compute_psnr_and_mae(baseline, quantized, "t2i")
    assert lpips_score <= MAX_LPIPS, f"MammothModa2 online-FP8 LPIPS {lpips_score:.4f} exceeds {MAX_LPIPS}"

    device_reduction = (bf16_device_mem - fp8_device_mem) / bf16_device_mem * 100 if bf16_device_mem > 0 else 0.0
    worker_reduction = (bf16_worker_mem - fp8_worker_mem) / bf16_worker_mem * 100 if bf16_worker_mem > 0 else 0.0
    print("\nMammothModa2 BF16 versus online FP8")
    print(f"  LPIPS:           {lpips_score:.4f} (threshold: {MAX_LPIPS})")
    print(f"  PSNR:            {psnr_score:.4f} dB")
    print(f"  MAE:             {mae_score:.6f}")
    print(f"  BF16 device:     {bf16_device_mem:.2f} MiB")
    print(f"  FP8 device:      {fp8_device_mem:.2f} MiB ({device_reduction:.1f}% reduction)")
    print(f"  BF16 DiT worker: {bf16_worker_mem:.2f} MiB")
    print(f"  FP8 DiT worker:  {fp8_worker_mem:.2f} MiB ({worker_reduction:.1f}% reduction)")

    assert np.isfinite(psnr_score) or np.isinf(psnr_score)
    assert np.isfinite(mae_score)
    assert bf16_device_mem > 0 and fp8_device_mem > 0
    assert bf16_worker_mem > 0 and fp8_worker_mem > 0

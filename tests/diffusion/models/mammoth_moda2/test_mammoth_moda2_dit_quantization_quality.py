# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""BF16 versus DiT-only online-FP8 quality, memory, and latency test for MammothModa2."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from vllm.sampling_params import SamplingParams

from tests.diffusion.quantization.test_quantization_quality import (
    _compute_lpips,
    _compute_psnr_and_mae,
)
from tests.e2e.offline_inference.test_mammoth_moda2_expansion import (
    _format_t2i_prompt,
    _load_t2i_gen_config,
)
from tests.helpers.mark import hardware_test
from tests.helpers.monitor import DeviceMemoryMonitor
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.platforms import current_omni_platform

MODEL_PATH = "bytedance-research/MammothModa2-Preview"
DEPLOY_CONFIG = get_deploy_config_path("mammoth_moda2.yaml")

HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 50
SEED = 42
MAX_LPIPS = 0.15
AR_HEIGHT = 32
AR_WIDTH = 32
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653


def _extract_image(outputs, *, height: int = HEIGHT, width: int = WIDTH) -> Image.Image:
    images = []
    for output_group in outputs:
        request_outputs = output_group if isinstance(output_group, list) else [output_group]
        for request_output in request_outputs:
            output_images = getattr(request_output, "images", None)
            if isinstance(output_images, list):
                images.extend(output_images)

    # Count across every request output and image list. Never silently discard
    # extra outputs before the quality comparison.
    if len(images) != 1:
        raise ValueError(f"Expected exactly one image, got {len(images)}")
    image = images[0]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected a PIL image, got {type(image)!r}")
    if image.mode != "RGB":
        raise ValueError(f"Expected an RGB image, got mode {image.mode!r}")
    expected_size = (width, height)
    if image.size != expected_size:
        raise ValueError(f"Expected image size {expected_size}, got {image.size}")
    return image


def _generate(deploy_config: str) -> tuple[Image.Image, float, float]:
    generation_config = _load_t2i_gen_config(MODEL_PATH)
    expected_grid_tokens = AR_HEIGHT * (AR_WIDTH + 1)
    prompt = _format_t2i_prompt("A cat sitting on a laptop keyboard", AR_WIDTH, AR_HEIGHT)

    ar_sampling = SamplingParams(
        temperature=float(generation_config["temperature"]),
        top_p=float(generation_config["top_p"]),
        top_k=int(generation_config["top_k"]),
        max_tokens=expected_grid_tokens + 1,
        detokenize=False,
        seed=SEED,
    )
    dit_sampling = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False, seed=SEED)
    request = {
        "prompt": prompt,
        "additional_information": {
            "omni_task": ["t2i"],
            "ar_width": [AR_WIDTH],
            "ar_height": [AR_HEIGHT],
            "eol_token_id": [int(generation_config["eol_token_id"])],
            "visual_token_start_id": [int(generation_config["visual_token_start_id"])],
            "visual_token_end_id": [int(generation_config["visual_token_end_id"])],
            "image_height": [HEIGHT],
            "image_width": [WIDTH],
            "num_inference_steps": [NUM_INFERENCE_STEPS],
            "text_guidance_scale": [9.0],
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
        with OmniRunner(
            MODEL_PATH,
            seed=SEED,
            deploy_config=deploy_config,
            enforce_eager=True,
        ) as runner:
            generation_start = time.perf_counter()
            outputs = list(runner.omni.generate([request], [ar_sampling, dit_sampling]))
            generation_latency_s = time.perf_counter() - generation_start
            image = _extract_image(outputs, height=HEIGHT, width=WIDTH)
        device_peak_memory_mb = monitor.peak_used_mb
    finally:
        monitor.stop()
        gc.collect()
        current_omni_platform.empty_cache()

    return image, device_peak_memory_mb, generation_latency_s


@pytest.fixture(scope="module")
def dit_fp8_deploy_config(tmp_path_factory: pytest.TempPathFactory) -> str:
    config = yaml.safe_load(Path(DEPLOY_CONFIG).read_text(encoding="utf-8"))
    stages = config["stages"]
    assert all("quantization" not in stage for stage in stages)

    dit_stage = next(stage for stage in stages if stage["stage_id"] == 1)
    dit_stage["quantization"] = "fp8"

    config_path = tmp_path_factory.mktemp("mammoth_moda2") / "mammoth_moda2_dit_fp8.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(config_path)


@pytest.mark.full_model
@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
def test_mammoth_moda2_dit_online_fp8_quality_and_memory(dit_fp8_deploy_config: str):
    baseline, bf16_device_mem, bf16_latency_s = _generate(DEPLOY_CONFIG)
    quantized, fp8_device_mem, fp8_latency_s = _generate(dit_fp8_deploy_config)

    lpips_score = _compute_lpips(baseline, quantized, "t2i")
    psnr_score, mae_score = _compute_psnr_and_mae(baseline, quantized, "t2i")
    assert lpips_score <= MAX_LPIPS, f"MammothModa2 DiT online-FP8 LPIPS {lpips_score:.4f} exceeds {MAX_LPIPS}"

    device_reduction = (bf16_device_mem - fp8_device_mem) / bf16_device_mem * 100 if bf16_device_mem > 0 else 0.0
    latency_reduction = (bf16_latency_s - fp8_latency_s) / bf16_latency_s * 100 if bf16_latency_s > 0 else 0.0
    print("\nMammothModa2 BF16 versus DiT-only online FP8")
    print(f"  LPIPS:           {lpips_score:.4f} (threshold: {MAX_LPIPS})")
    print(f"  PSNR:            {psnr_score:.4f} dB")
    print(f"  MAE:             {mae_score:.6f}")
    print(f"  BF16 device:     {bf16_device_mem:.2f} MiB")
    print(f"  FP8 device:      {fp8_device_mem:.2f} MiB ({device_reduction:.1f}% reduction)")
    print(f"  BF16 latency:    {bf16_latency_s:.4f} s")
    print(f"  FP8 latency:     {fp8_latency_s:.4f} s ({latency_reduction:.1f}% reduction)")

    assert np.isfinite(psnr_score) or np.isinf(psnr_score)
    assert np.isfinite(mae_score)
    assert bf16_device_mem > 0 and fp8_device_mem > 0
    assert bf16_latency_s > 0 and fp8_latency_s > 0

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FLUX.2-dev VAE patch-parallel correctness tests.

Compares vae_patch_parallel_size=1 vs 2 on TP=2 with tiling enabled:
- Text-to-image decode path
- Image-to-image encode + decode path (conditioning image)
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

PROMPT = "a photo of a cat sitting on a laptop keyboard"
NEGATIVE_PROMPT = "blurry, low quality"

VAE_PP_MEAN_THRESHOLD = 3e-2
VAE_PP_P99_THRESHOLD = 1e-1


def _get_flux2_dev_model() -> str:
    return os.environ.get("VLLM_TEST_FLUX2_DEV_MODEL", "black-forest-labs/FLUX.2-dev")


def _pil_to_float_rgb_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _diff_metrics(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    ta = _pil_to_float_rgb_tensor(a)
    tb = _pil_to_float_rgb_tensor(b)
    assert ta.shape == tb.shape, f"Image shapes differ: {ta.shape} vs {tb.shape}"
    abs_diff = torch.abs(ta - tb)
    p99_abs_diff = torch.quantile(abs_diff.flatten(), 0.99).item()
    return abs_diff.mean().item(), p99_abs_diff


def _extract_single_image(outputs) -> Image.Image:
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out.images
    if images is None or len(images) != 1:
        raise ValueError(f"Expected 1 image, got {0 if images is None else len(images)}")
    return images[0]


def _run_flux2_generate(
    *,
    tp_size: int,
    height: int,
    width: int,
    num_inference_steps: int,
    seed: int,
    vae_patch_parallel_size: int,
    conditioning_image: Image.Image | None = None,
) -> Image.Image:
    current_omni_platform.empty_cache()

    request: dict = {"prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT}
    if conditioning_image is not None:
        request["multi_modal_data"] = {"image": conditioning_image}

    with OmniRunner(
        _get_flux2_dev_model(),
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=tp_size,
            vae_patch_parallel_size=vae_patch_parallel_size,
        ),
        enable_cpu_offload=True,
        vae_use_tiling=True,
    ) as runner:
        outputs = list(
            runner.omni.generate(
                request,
                OmniDiffusionSamplingParams(
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=4.0,
                    seed=seed,
                    num_outputs_per_prompt=1,
                ),
            )
        )

    return _extract_single_image(outputs)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_flux2_vae_patch_parallel_decode_tp2(tmp_path: Path):
    if not current_omni_platform.is_available() or current_omni_platform.device_count() < 2:
        pytest.skip("FLUX.2-dev VAE patch parallel requires >= 2 devices.")

    height = 1152
    width = 1152
    num_inference_steps = 2
    seed = 42

    baseline_img = _run_flux2_generate(
        tp_size=2,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
        vae_patch_parallel_size=1,
    )
    pp2_img = _run_flux2_generate(
        tp_size=2,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
        vae_patch_parallel_size=2,
    )

    baseline_path = tmp_path / "flux2_tp2_vae_pp1.png"
    pp2_path = tmp_path / "flux2_tp2_vae_pp2.png"
    baseline_img.save(baseline_path)
    pp2_img.save(pp2_path)

    assert baseline_img.width == width and baseline_img.height == height
    assert pp2_img.width == width and pp2_img.height == height

    mean_abs_diff, p99_abs_diff = _diff_metrics(baseline_img, pp2_img)
    print(
        "FLUX.2-dev VAE patch parallel decode diff (TP=2, pp=1 vs pp=2): "
        f"mean_abs_diff={mean_abs_diff:.6e}, p99_abs_diff={p99_abs_diff:.6e}; "
        f"thresholds: mean<={VAE_PP_MEAN_THRESHOLD:.6e}, p99<={VAE_PP_P99_THRESHOLD:.6e}; "
        f"pp1_img={baseline_path}, pp2_img={pp2_path}"
    )
    assert mean_abs_diff <= VAE_PP_MEAN_THRESHOLD and p99_abs_diff <= VAE_PP_P99_THRESHOLD, (
        f"Image diff exceeded threshold: mean_abs_diff={mean_abs_diff:.6e}, p99_abs_diff={p99_abs_diff:.6e} "
        f"(thresholds: mean<={VAE_PP_MEAN_THRESHOLD:.6e}, p99<={VAE_PP_P99_THRESHOLD:.6e})"
    )


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_flux2_vae_patch_parallel_i2i_tp2(tmp_path: Path):
    if not current_omni_platform.is_available() or current_omni_platform.device_count() < 2:
        pytest.skip("FLUX.2-dev VAE patch parallel I2I requires >= 2 devices.")

    height = 1152
    width = 1152
    num_inference_steps = 2
    seed = 42
    conditioning_image = Image.new("RGB", (width, height), (120, 80, 40))

    baseline_img = _run_flux2_generate(
        tp_size=2,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
        vae_patch_parallel_size=1,
        conditioning_image=conditioning_image,
    )
    pp2_img = _run_flux2_generate(
        tp_size=2,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
        vae_patch_parallel_size=2,
        conditioning_image=conditioning_image,
    )

    baseline_path = tmp_path / "flux2_i2i_tp2_vae_pp1.png"
    pp2_path = tmp_path / "flux2_i2i_tp2_vae_pp2.png"
    baseline_img.save(baseline_path)
    pp2_img.save(pp2_path)

    mean_abs_diff, p99_abs_diff = _diff_metrics(baseline_img, pp2_img)
    print(
        "FLUX.2-dev VAE patch parallel I2I diff (TP=2, pp=1 vs pp=2): "
        f"mean_abs_diff={mean_abs_diff:.6e}, p99_abs_diff={p99_abs_diff:.6e}; "
        f"thresholds: mean<={VAE_PP_MEAN_THRESHOLD:.6e}, p99<={VAE_PP_P99_THRESHOLD:.6e}; "
        f"pp1_img={baseline_path}, pp2_img={pp2_path}"
    )
    assert mean_abs_diff <= VAE_PP_MEAN_THRESHOLD and p99_abs_diff <= VAE_PP_P99_THRESHOLD, (
        f"Image diff exceeded threshold: mean_abs_diff={mean_abs_diff:.6e}, p99_abs_diff={p99_abs_diff:.6e} "
        f"(thresholds: mean<={VAE_PP_MEAN_THRESHOLD:.6e}, p99<={VAE_PP_P99_THRESHOLD:.6e})"
    )

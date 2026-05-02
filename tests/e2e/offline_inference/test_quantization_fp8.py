# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for the unified quantization framework (PR #1764).

Validates FP8 quantization works correctly for all supported model types:
  - Single-stage diffusion models (FLUX.1-dev, Qwen-Image, Z-Image-Turbo)
  - Multi-stage models (BAGEL: LLM + Diffusion)

Tests verify:
  1. FP8 quantization produces valid images
  2. Memory usage is lower than BF16 baseline
  3. Multi-stage models only quantize the diffusion stage (not the LLM stage)

Usage:
    # Run all FP8 quantization tests
    pytest tests/e2e/offline_inference/test_quantization_fp8.py -v

    # Run single-stage tests only (faster, needs ~25GB VRAM)
    pytest tests/e2e/offline_inference/test_quantization_fp8.py -v -k "single_stage"

    # Run BAGEL multi-stage test (needs ~55GB VRAM, H100 recommended)
    pytest tests/e2e/offline_inference/test_quantization_fp8.py -v -k "bagel"

    # Run FLUX test only
    pytest tests/e2e/offline_inference/test_quantization_fp8.py -v -k "flux"
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ─── helpers ──────────────────────────────────────────────────────────────────


def _generate_single_stage_image(
    model: str,
    quantization: str | None = None,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 2,
    seed: int = 42,
    **extra_omni_kwargs: Any,
) -> tuple[list, float]:
    """Generate an image with a single-stage diffusion model.

    Returns (images, peak_memory_gib).
    """
    omni_kwargs: dict[str, Any] = dict(extra_omni_kwargs)
    if quantization:
        omni_kwargs["quantization"] = quantization

    with OmniRunner(model, **omni_kwargs) as runner:
        torch.cuda.reset_peak_memory_stats()

        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
        outputs = runner.omni.generate(
            "a photo of a cat sitting on a laptop keyboard",
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                generator=generator,
            ),
        )

        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

        first_output = outputs[0]
        assert first_output.final_output_type == "image"
        assert hasattr(first_output, "request_output") and first_output.request_output
        req_out = first_output.request_output[0]
        assert isinstance(req_out, OmniRequestOutput) and hasattr(req_out, "images")
        images = req_out.images
        assert len(images) >= 1
        assert images[0].width == width
        assert images[0].height == height

        return images, peak_mem


def _generate_bagel_image(
    quantization_config: str | None = None,
    num_inference_steps: int = 15,
) -> tuple[Any, float]:
    """Generate an image with BAGEL (multi-stage: LLM + Diffusion).

    Returns (generated_image, peak_memory_gib).
    """
    config_path = get_deploy_config_path("ci/bagel.yaml")
    omni_kwargs: dict[str, Any] = {
        "model": "ByteDance-Seed/BAGEL-7B-MoT",
        "stage_configs_path": config_path,
        "stage_init_timeout": 300,
    }
    if quantization_config:
        omni_kwargs["quantization_config"] = quantization_config

    model_name = omni_kwargs.pop("model")
    with OmniRunner(model_name, **omni_kwargs) as runner:
        omni = runner.omni
        torch.cuda.reset_peak_memory_stats()

        params_list = omni.default_sampling_params_list
        if len(params_list) > 1:
            params_list[1].num_inference_steps = num_inference_steps  # type: ignore
            params_list[1].extra_args = {  # type: ignore
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 1.5,
            }

        prompt = "<|im_start|>A futuristic city skyline at twilight, cyberpunk style<|im_end|>"
        omni_outputs = list(
            omni.generate(
                prompts=[{"prompt": prompt, "modalities": ["image"]}],
                sampling_params_list=params_list,
            )
        )

        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

        # Extract image
        generated_image = None
        for req_output in omni_outputs:
            if images := getattr(req_output, "images", None):
                generated_image = images[0]
                break
            if hasattr(req_output, "request_output") and req_output.request_output:
                for stage_out in req_output.request_output:
                    if hasattr(stage_out, "images") and stage_out.images:
                        generated_image = stage_out.images[0]
                        break
                if generated_image:
                    break

        assert generated_image is not None, "No images generated from BAGEL"
        assert generated_image.size == (1024, 1024), f"Expected 1024x1024, got {generated_image.size}"

        # Check LLM stage output — should have finish_reason=stop (not length)
        for req_output in omni_outputs:
            if hasattr(req_output, "request_output"):
                for stage_out in req_output.request_output:
                    if hasattr(stage_out, "outputs"):
                        for comp_out in stage_out.outputs:
                            if hasattr(comp_out, "finish_reason"):
                                assert comp_out.finish_reason == "stop", (
                                    f"LLM stage finish_reason={comp_out.finish_reason}, "
                                    f"text={comp_out.text!r}. "
                                    "FP8 may have leaked to the LLM stage."
                                )

        return generated_image, peak_mem


# ─── Single-stage diffusion model tests ──────────────────────────────────────


@hardware_test(res={"cuda": "L4"})
def test_single_stage_zimage_fp8():
    """Z-Image-Turbo with FP8 generates valid images."""
    images, _ = _generate_single_stage_image(
        model="Tongyi-MAI/Z-Image-Turbo",
        quantization="fp8",
    )
    assert len(images) >= 1
    images[0].save("test_zimage_fp8.png")


@hardware_test(res={"cuda": "L4"})
def test_single_stage_zimage_fp8_uses_less_memory():
    """FP8 should use less peak memory than BF16 for Z-Image-Turbo."""
    _, mem_bf16 = _generate_single_stage_image(
        model="Tongyi-MAI/Z-Image-Turbo",
        quantization=None,
    )
    torch.cuda.empty_cache()

    _, mem_fp8 = _generate_single_stage_image(
        model="Tongyi-MAI/Z-Image-Turbo",
        quantization="fp8",
    )

    print(f"Z-Image BF16 peak memory: {mem_bf16:.2f} GiB")
    print(f"Z-Image FP8 peak memory:  {mem_fp8:.2f} GiB")
    assert mem_fp8 < mem_bf16, f"FP8 ({mem_fp8:.2f} GiB) should use less memory than BF16 ({mem_bf16:.2f} GiB)"


@hardware_test(res={"cuda": "L4"})
def test_single_stage_qwen_image_fp8():
    """Qwen-Image (random weights) with FP8 generates valid images."""
    model = "riverclouds/qwen_image_random"
    if current_omni_platform.is_npu() or current_omni_platform.is_rocm():
        pytest.skip("qwen_image_random not available on this platform")

    images, _ = _generate_single_stage_image(
        model=model,
        quantization="fp8",
    )
    assert len(images) >= 1
    images[0].save("test_qwen_image_fp8.png")


@hardware_test(res={"cuda": "H100"})
def test_single_stage_flux_fp8():
    """FLUX.1-dev with FP8 generates valid images."""
    images, _ = _generate_single_stage_image(
        model="black-forest-labs/FLUX.1-dev",
        quantization="fp8",
        height=512,
        width=512,
        num_inference_steps=4,
    )
    assert len(images) >= 1
    images[0].save("test_flux_fp8.png")


@hardware_test(res={"cuda": "H100"})
def test_single_stage_flux_fp8_uses_less_memory():
    """FP8 should use less peak memory than BF16 for FLUX.1-dev."""
    _, mem_bf16 = _generate_single_stage_image(
        model="black-forest-labs/FLUX.1-dev",
        quantization=None,
        height=512,
        width=512,
        num_inference_steps=4,
    )
    torch.cuda.empty_cache()

    _, mem_fp8 = _generate_single_stage_image(
        model="black-forest-labs/FLUX.1-dev",
        quantization="fp8",
        height=512,
        width=512,
        num_inference_steps=4,
    )

    print(f"FLUX BF16 peak memory: {mem_bf16:.2f} GiB")
    print(f"FLUX FP8 peak memory:  {mem_fp8:.2f} GiB")
    assert mem_fp8 < mem_bf16, f"FP8 ({mem_fp8:.2f} GiB) should use less memory than BF16 ({mem_bf16:.2f} GiB)"


# ─── Multi-stage model tests (BAGEL) ─────────────────────────────────────────


@hardware_test(res={"cuda": "H100"})
def test_bagel_fp8_generates_image():
    """BAGEL with FP8 quantization_config generates a valid image.

    FP8 should only apply to the diffusion stage (Stage-1), not the
    LLM stage (Stage-0). We verify this by checking:
      1. Image is generated successfully
      2. LLM stage finish_reason is 'stop' (not 'length' from garbled output)
    """
    image, _ = _generate_bagel_image(quantization_config="fp8")
    image.save("test_bagel_fp8.png")


@hardware_test(res={"cuda": "H100"})
def test_bagel_bf16_generates_image():
    """BAGEL without quantization generates a valid image (baseline)."""
    image, _ = _generate_bagel_image(quantization_config=None)
    image.save("test_bagel_bf16.png")


# ─── Quantization config routing tests ────────────────────────────────────────


@hardware_test(res={"cuda": "L4"})
def test_quantization_key_maps_to_quantization_config():
    """The old 'quantization' kwarg should map to 'quantization_config'
    in OmniDiffusionConfig.from_kwargs for backwards compatibility."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig.from_kwargs(model="test", quantization="fp8")
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "fp8"


@hardware_test(res={"cuda": "L4"})
def test_quantization_config_key_takes_priority():
    """When both 'quantization' and 'quantization_config' are set,
    'quantization_config' takes priority."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig.from_kwargs(
        model="test",
        quantization="fp8",
        quantization_config={"method": "fp8", "activation_scheme": "static"},
    )
    assert config.quantization_config is not None
    assert config.quantization_config.activation_scheme == "static"


@hardware_test(res={"cuda": "L4"})
def test_single_stage_quantization_config_key():
    """Single-stage model using quantization_config (dict) key generates images."""
    images, _ = _generate_single_stage_image(
        model="Tongyi-MAI/Z-Image-Turbo",
        quantization=None,
        quantization_config="fp8",
    )
    assert len(images) >= 1

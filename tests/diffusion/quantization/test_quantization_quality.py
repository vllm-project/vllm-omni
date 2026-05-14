# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Quantization quality gate for diffusion models.

Developers adding a new quantization method should:
1. Add their method + model to QUALITY_CONFIGS below
2. Set a max_lpips threshold (use 0.15 for image, 0.20 for video as defaults)
3. Run: pytest tests/diffusion/quantization/test_quantization_quality.py -v -m ""
4. Paste the output table into their PR description

The test generates outputs with both BF16 and the quantized method using the
same seed, computes similarity metrics, and fails if LPIPS exceeds the threshold.

Requirements:
    pip install lpips

Example — run only FP8 tests:
    pytest tests/diffusion/quantization/test_quantization_quality.py -v -m "" -k "fp8"

Example — run a specific model:
    pytest tests/diffusion/quantization/test_quantization_quality.py -v -m "" -k "z_image"

Example — validate a local BF16 baseline against a local pre-quantized checkpoint:
    export VLLM_OMNI_QUALITY_CONFIGS=/tmp/modelopt_quality_cases.json
    pytest tests/diffusion/quantization/test_quantization_quality.py -v -m "" -k "qwen_image_2512"

Optional artifact dump:
    export VLLM_OMNI_QUALITY_OUTPUT_DIR=/tmp/modelopt_quality_outputs
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_marks

# ---------------------------------------------------------------------------
# Configuration — add new quantization methods / models here
# ---------------------------------------------------------------------------


@dataclass
class QualityTestConfig:
    """Defines a single quantization quality test case."""

    id: str  # pytest ID, e.g. "fp8_z_image"
    task: str  # "t2i" or "t2v"
    prompt: str  # generation prompt
    max_lpips: float  # fail threshold — higher = more lenient
    model: str | None = None  # HF model name
    quantization: str | None = None  # quantization method, e.g. "fp8"
    baseline_model: str | None = None  # explicit BF16/local baseline path
    quantized_model: str | None = None  # explicit quantized/local model path
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 20  # keep low for CI speed
    num_frames: int = 5  # only for t2v
    seed: int = 42
    gpu: str = "H100"  # minimum GPU requirement
    negative_prompt: str = ""
    guidance_scale: float | None = None

    def baseline_ref(self) -> str:
        return self.baseline_model or self.model or ""

    def quantized_ref(self) -> str:
        if self.quantized_model is not None:
            return self.quantized_model
        return self.model or ""

    def quantization_ref(self) -> str | None:
        if self.quantized_model is not None:
            return None
        return self.quantization

    def validate(self) -> None:
        uses_explicit_models = self.baseline_model is not None or self.quantized_model is not None
        uses_model_plus_method = self.model is not None or self.quantization is not None

        if uses_explicit_models and uses_model_plus_method:
            raise ValueError(f"{self.id}: explicit baseline/quantized paths cannot be mixed with model/quantization")

        if uses_explicit_models:
            if self.baseline_model is None or self.quantized_model is None:
                raise ValueError(f"{self.id}: baseline_model and quantized_model must be provided together")
            return

        if self.model is None or self.quantization is None:
            raise ValueError(f"{self.id}: expected either model+quantization or baseline_model+quantized_model")


# Add new quantization methods / models here.
# Developers: copy a config, change quantization + max_lpips, run the test.
QUALITY_CONFIGS = [
    QualityTestConfig(
        id="fp8_z_image",
        model="Tongyi-MAI/Z-Image-Turbo",
        quantization="fp8",
        task="t2i",
        prompt="a cup of coffee on a wooden table, morning light",
        max_lpips=0.15,
        num_inference_steps=20,
    ),
    QualityTestConfig(
        id="fp8_flux",
        model="black-forest-labs/FLUX.1-dev",
        quantization="fp8",
        task="t2i",
        prompt="a cup of coffee on a wooden table, morning light",
        max_lpips=0.20,
        num_inference_steps=10,
    ),
    QualityTestConfig(
        id="fp8_qwen_image",
        model="Qwen/Qwen-Image",
        quantization="fp8",
        task="t2i",
        prompt="a cup of coffee on a wooden table, morning light",
        max_lpips=0.35,
        seed=142,
        num_inference_steps=20,
    ),
]


def _load_extra_quality_configs() -> list[QualityTestConfig]:
    config_path = os.environ.get("VLLM_OMNI_QUALITY_CONFIGS")
    if not config_path:
        return []

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("VLLM_OMNI_QUALITY_CONFIGS must point to a JSON list")

    configs: list[QualityTestConfig] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each quality config entry must be a JSON object")
        config = QualityTestConfig(**item)
        config.validate()
        configs.append(config)
    return configs


def _all_quality_configs() -> list[QualityTestConfig]:
    configs = [*QUALITY_CONFIGS, *_load_extra_quality_configs()]
    for config in configs:
        config.validate()
    return configs


def _output_path(output_dir: Path, config: QualityTestConfig, label: str, suffix: str) -> Path:
    safe_id = config.id.replace("/", "_")
    return output_dir / f"{safe_id}_{label}{suffix}"


def _maybe_save_output(output_dir: Path | None, config: QualityTestConfig, label: str, output) -> None:
    if output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if config.task == "t2i":
        output.save(_output_path(output_dir, config, label, ".png"))
        return

    if isinstance(output, np.ndarray):
        np.save(_output_path(output_dir, config, label, ".npy"), output)
        return

    raise TypeError(f"Unsupported output type for saving: {type(output)!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_image(omni, config: QualityTestConfig):
    """Generate a single image, return (PIL.Image, peak_mem_gib)."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(
        device=current_omni_platform.device_type,
    ).manual_seed(config.seed)
    torch.accelerator.reset_peak_memory_stats()

    outputs = omni.generate(
        {"prompt": config.prompt, "negative_prompt": config.negative_prompt},
        OmniDiffusionSamplingParams(
            height=config.height,
            width=config.width,
            generator=generator,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ),
    )

    peak_mem = torch.accelerator.max_memory_allocated() / (1024**3)
    first = outputs[0]
    if hasattr(first, "images") and first.images:
        return first.images[0], peak_mem
    inner = first.request_output
    if inner is not None and hasattr(inner, "images") and inner.images:
        return inner.images[0], peak_mem
    raise ValueError("Could not extract image from output.")


def _generate_video(omni, config: QualityTestConfig):
    """Generate a video, return (np.ndarray [F,H,W,C], peak_mem_gib)."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(
        device=current_omni_platform.device_type,
    ).manual_seed(config.seed)
    torch.accelerator.reset_peak_memory_stats()

    outputs = omni.generate(
        {"prompt": config.prompt, "negative_prompt": config.negative_prompt},
        OmniDiffusionSamplingParams(
            height=config.height,
            width=config.width,
            generator=generator,
            num_inference_steps=config.num_inference_steps,
            num_frames=config.num_frames,
            guidance_scale=config.guidance_scale,
        ),
    )

    peak_mem = torch.accelerator.max_memory_allocated() / (1024**3)
    first = outputs[0]
    if hasattr(first, "request_output") and isinstance(first.request_output, list):
        inner = first.request_output[0]
        if isinstance(inner, OmniRequestOutput) and hasattr(inner, "images"):
            frames = inner.images[0] if inner.images else None
        else:
            frames = inner
    elif hasattr(first, "images") and first.images:
        frames = first.images
    else:
        raise ValueError("Could not extract video frames from output.")

    if isinstance(frames, torch.Tensor):
        video = frames.detach().cpu()
        if video.dim() == 5:
            video = video[0]
        if video.dim() == 4 and video.shape[0] in (3, 4):
            video = video.permute(1, 2, 3, 0)
        if video.is_floating_point():
            video = video.clamp(-1, 1) * 0.5 + 0.5
        return video.float().numpy(), peak_mem

    return np.asarray(frames), peak_mem


def _compute_lpips(baseline, quantized, task: str) -> float:
    """Compute LPIPS between baseline and quantized outputs."""
    from benchmarks.diffusion.quantization_quality import (
        compute_lpips_images,
        compute_lpips_video,
    )

    if task == "t2i":
        return compute_lpips_images([baseline], [quantized])[0]
    return compute_lpips_video(baseline, quantized)


def _to_float_array(output, task: str) -> np.ndarray:
    if task == "t2i":
        array = np.asarray(output.convert("RGB"), dtype=np.float32) / 255.0
    else:
        array = np.asarray(output, dtype=np.float32)
        if array.max() > 1.0 or array.min() < 0.0:
            array = np.clip(array, 0.0, 255.0) / 255.0
        else:
            array = np.clip(array, 0.0, 1.0)
    return array


def _compute_psnr_and_mae(baseline, quantized, task: str) -> tuple[float, float]:
    baseline_array = _to_float_array(baseline, task)
    quantized_array = _to_float_array(quantized, task)
    if baseline_array.shape != quantized_array.shape:
        raise ValueError(
            "Output shapes do not match for metric computation: "
            f"baseline={baseline_array.shape}, quantized={quantized_array.shape}"
        )

    diff = baseline_array - quantized_array
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(np.square(diff)))
    psnr = float("inf") if mse == 0.0 else float(20.0 * np.log10(1.0 / np.sqrt(mse)))
    return psnr, mae


def _unload(omni):
    del omni
    gc.collect()
    if torch.cuda.is_available():
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_marks = hardware_marks(res={"cuda": "H100"})
_OUTPUT_DIR = Path(os.environ["VLLM_OMNI_QUALITY_OUTPUT_DIR"]) if "VLLM_OMNI_QUALITY_OUTPUT_DIR" in os.environ else None


def _quality_param(c: QualityTestConfig):
    marks = list(_marks)
    if c.id == "fp8_z_image":
        marks.append(
            pytest.mark.skip(
                reason="Z-Image FP8 quality gate temporarily disabled: https://github.com/vllm-project/vllm-omni/issues/3531"
            )
        )
    if c.id == "fp8_qwen_image":
        marks.append(
            pytest.mark.skip(reason="Qwen-Image FP8 quality gate temporarily disabled (see CI / issue tracker).")
        )
    return pytest.param(c, id=c.id, marks=marks)


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "config",
    [_quality_param(c) for c in _all_quality_configs()],
)
def test_quantization_quality(config: QualityTestConfig):
    """Validate that quantized output stays within LPIPS threshold of BF16."""
    from vllm_omni.entrypoints.omni import Omni

    generate_fn = _generate_video if config.task == "t2v" else _generate_image

    # --- BF16 baseline ---
    omni_bl = Omni(model=config.baseline_ref())
    baseline_out, bl_mem = generate_fn(omni_bl, config)
    _unload(omni_bl)
    _maybe_save_output(_OUTPUT_DIR, config, "baseline", baseline_out)

    # --- Quantized ---
    quantization = config.quantization_ref()
    if quantization is None:
        omni_qt = Omni(model=config.quantized_ref())
    else:
        omni_qt = Omni(model=config.quantized_ref(), quantization_config=quantization)
    quant_out, qt_mem = generate_fn(omni_qt, config)
    _unload(omni_qt)
    _maybe_save_output(_OUTPUT_DIR, config, "quantized", quant_out)

    # --- Similarity metrics ---
    lpips_score = _compute_lpips(baseline_out, quant_out, config.task)
    psnr_score, mae_score = _compute_psnr_and_mae(baseline_out, quant_out, config.task)
    assert lpips_score <= config.max_lpips, (
        f"LPIPS {lpips_score:.4f} exceeds threshold {config.max_lpips} "
        f"for {config.quantization_ref() or 'pre-quantized checkpoint'} on {config.quantized_ref()}"
    )

    # --- Report ---
    mem_reduction = (bl_mem - qt_mem) / bl_mem * 100 if bl_mem > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Quantization Quality: {config.id}")
    print(f"{'=' * 60}")
    print(f"  Baseline:      {config.baseline_ref()}")
    print(f"  Quantized:     {config.quantized_ref()}")
    print(f"  Method:        {config.quantization_ref() or 'pre-quantized checkpoint'}")
    print(f"  LPIPS:         {lpips_score:.4f}  (threshold: {config.max_lpips})")
    print(f"  PSNR:          {psnr_score:.4f} dB  (higher is better)")
    print(f"  MAE:           {mae_score:.6f}  (lower is better)")
    print(f"  BF16 memory:   {bl_mem:.2f} GiB")
    print(f"  Quant memory:  {qt_mem:.2f} GiB  ({mem_reduction:.0f}% reduction)")
    print(f"  Result:        {'PASS' if lpips_score <= config.max_lpips else 'FAIL'}")
    print(f"{'=' * 60}\n")

    assert np.isfinite(psnr_score) or np.isinf(psnr_score), f"PSNR is invalid for {config.id}: {psnr_score}"
    assert np.isfinite(mae_score), f"MAE is not finite for {config.id}: {mae_score}"

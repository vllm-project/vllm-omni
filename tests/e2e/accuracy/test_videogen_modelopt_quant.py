# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end ModelOpt FP8 quality accuracy for video-gen DiTs.

Modeled on ``test_hunyuan_image3.py``'s ``test_quantized_dit_matches_bf16_accuracy``:
for each calibrated FP8 checkpoint the test generates one video with BF16 and one
with the FP8 DiT under the same seed, scores prompt-faithfulness with CLIP on the
middle frame, then asserts both ``CLIP >= absolute_floor`` and
``CLIP_drop <= drop_threshold``. The test is gated by an opt-in env var (model
paths are local) so it never runs in generic CI.

To run::

    VIDEOGEN_RUN_QUANT_ACCURACY=1 \\
    WAN22_A14B_BF16_MODEL=Wan-AI/Wan2.2-T2V-A14B-Diffusers \\
    WAN22_A14B_FP8_MODEL=/path/to/wan22-a14b-modelopt-fp8 \\
    HV15_BF16_MODEL=hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \\
    HV15_FP8_MODEL=/path/to/hv15-modelopt-fp8 \\
        pytest -s -v tests/e2e/accuracy/test_videogen_modelopt_quant.py
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from tests.e2e.accuracy.helpers import CLIPScorer, model_output_dir

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion]

# ---------------------------------------------------------------------------
# Tunables (override via env to suit local hardware / calibration budget)
# ---------------------------------------------------------------------------
SEED = 42
HEIGHT = int(os.environ.get("VIDEOGEN_QUANT_HEIGHT", "480"))
WIDTH = int(os.environ.get("VIDEOGEN_QUANT_WIDTH", "832"))
NUM_FRAMES = int(os.environ.get("VIDEOGEN_QUANT_NUM_FRAMES", "17"))
NUM_INFERENCE_STEPS = int(os.environ.get("VIDEOGEN_QUANT_STEPS", "20"))

WAN_PROMPT = "A red fox running through a snowy forest at sunrise, cinematic."
HV_PROMPT = "A red fox running through a snowy forest at sunrise, cinematic."

# CLIP gates: absolute floor + max drop from the BF16 reference.
CLIP_ABSOLUTE_FLOOR = float(os.environ.get("VIDEOGEN_QUANT_CLIP_FLOOR", "20.0"))
CLIP_DROP_FP8 = float(os.environ.get("VIDEOGEN_QUANT_CLIP_DROP_FP8", "7.0"))

RUN_ENV = "VIDEOGEN_RUN_QUANT_ACCURACY"
_TRUE = {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Spec:
    """A model under test (BF16 + its FP8 variant)."""

    family: str
    bf16_env: str
    fp8_env: str
    prompt: str
    guidance_scale: float
    guidance_scale_2: float | None = None
    boundary_ratio: float | None = None


WAN_SPEC = _Spec(
    family="wan22_a14b_t2v",
    bf16_env="WAN22_A14B_BF16_MODEL",
    fp8_env="WAN22_A14B_FP8_MODEL",
    prompt=WAN_PROMPT,
    guidance_scale=3.5,
    guidance_scale_2=4.0,
    boundary_ratio=0.875,
)

HV_SPEC = _Spec(
    family="hunyuanvideo15_t2v",
    bf16_env="HV15_BF16_MODEL",
    fp8_env="HV15_FP8_MODEL",
    prompt=HV_PROMPT,
    guidance_scale=6.0,
)


@dataclass(frozen=True)
class _QuantCase:
    name: str
    spec: _Spec
    clip_drop: float


def _cases() -> list[pytest.ParameterSet]:
    cases = [
        _QuantCase("wan_fp8", WAN_SPEC, CLIP_DROP_FP8),
        _QuantCase("hv_fp8", HV_SPEC, CLIP_DROP_FP8),
    ]
    params: list[pytest.ParameterSet] = []
    opted_in = os.environ.get(RUN_ENV, "").lower() in _TRUE
    for case in cases:
        marks: list[Any] = []
        if not opted_in:
            marks.append(pytest.mark.skip(reason=f"Set {RUN_ENV}=1 to run video-gen quant accuracy."))
        if not os.environ.get(case.spec.bf16_env):
            marks.append(pytest.mark.skip(reason=f"Set {case.spec.bf16_env} to run {case.name}."))
        if not os.environ.get(case.spec.fp8_env):
            marks.append(pytest.mark.skip(reason=f"Set {case.spec.fp8_env} to run {case.name}."))
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


# ---------------------------------------------------------------------------
def _middle_frame(frames: list[Image.Image] | list[np.ndarray] | np.ndarray) -> Image.Image:
    if isinstance(frames, np.ndarray):
        idx = frames.shape[0] // 2
        arr = frames[idx]
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if not frames:
        raise AssertionError("Empty video output")
    mid = frames[len(frames) // 2]
    if isinstance(mid, np.ndarray):
        if mid.dtype != np.uint8:
            mid = (mid * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(mid).convert("RGB")
    return mid.convert("RGB")


def _extract_frames(result) -> list[Image.Image]:
    """Pull a frame list out of an Omni text-to-video result."""
    from vllm_omni.entrypoints.omni import OmniRequestOutput

    if isinstance(result, list):
        result = result[0] if result else None
    if isinstance(result, OmniRequestOutput) and result.is_pipeline_output and result.request_output is not None:
        result = result.request_output
    images = getattr(result, "images", None)
    if not images:
        raise AssertionError("Pipeline output had no images")
    if isinstance(images, list) and images and isinstance(images[0], tuple):
        # (frames, fps)
        images = images[0][0]
    return list(images)


def _generate_video(
    *,
    model: str,
    spec: _Spec,
    quantization: str | None,
    output_path: Path,
) -> tuple[Image.Image, float]:
    """Generate one video and return its middle frame + elapsed wall time."""
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    kwargs: dict[str, Any] = dict(
        model=model,
        parallel_config=DiffusionParallelConfig(),
        enforce_eager=True,
        vae_use_tiling=True,
    )
    if quantization:
        kwargs["quantization"] = quantization
    if spec.boundary_ratio is not None:
        kwargs["boundary_ratio"] = spec.boundary_ratio
    omni = Omni(**kwargs)

    try:
        gen = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED)
        sampling = dict(
            height=HEIGHT,
            width=WIDTH,
            generator=gen,
            guidance_scale=spec.guidance_scale,
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_frames=NUM_FRAMES,
            seed=SEED,
        )
        if spec.guidance_scale_2 is not None:
            sampling["guidance_scale_2"] = spec.guidance_scale_2

        import time as _time

        t0 = _time.perf_counter()
        out = omni.generate({"prompt": spec.prompt}, OmniDiffusionSamplingParams(**sampling))
        elapsed = _time.perf_counter() - t0

        frames = _extract_frames(out)
        mid = _middle_frame(frames)
        mid.save(output_path)
        return mid, elapsed
    finally:
        del omni
        gc.collect()
        if torch.cuda.is_available():
            torch.accelerator.empty_cache()


@pytest.mark.parametrize("case", _cases())
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.accelerator.device_count() < 1,
    reason="Needs at least 1 GPU.",
)
def test_quantized_videogen_dit_matches_bf16_clip(case: _QuantCase, accuracy_artifact_root: Path) -> None:
    """FP8 DiT should preserve prompt-aligned video quality vs BF16."""
    out_dir = model_output_dir(accuracy_artifact_root, case.spec.family + "-quant")

    with tempfile.TemporaryDirectory():
        bf16_model = os.environ[case.spec.bf16_env]
        quant_model = os.environ[case.spec.fp8_env]

        bf16_frame, bf16_time = _generate_video(
            model=bf16_model,
            spec=case.spec,
            quantization=None,
            output_path=out_dir / "bf16.png",
        )
        quant_frame, quant_time = _generate_video(
            model=quant_model,
            spec=case.spec,
            quantization="fp8",
            output_path=out_dir / f"{case.name}.png",
        )

    clip = CLIPScorer()
    bf16_clip = clip.score(bf16_frame, case.spec.prompt)
    quant_clip = clip.score(quant_frame, case.spec.prompt)
    clip_drop = bf16_clip - quant_clip
    speedup = (bf16_time / quant_time) if quant_time else float("nan")

    metrics = {
        "case": case.name,
        "family": case.spec.family,
        "quantization": "fp8",
        "bf16_model": bf16_model,
        "quant_model": quant_model,
        "prompt": case.spec.prompt,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "bf16_elapsed_s": bf16_time,
        "quant_elapsed_s": quant_time,
        "speedup_vs_bf16": speedup,
        "bf16_clip_score": bf16_clip,
        "quant_clip_score": quant_clip,
        "clip_score_drop": clip_drop,
    }
    metrics_path = out_dir / f"{case.name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n[{case.name}] BF16 {bf16_time:.2f}s | FP8 {quant_time:.2f}s (x{speedup:.2f})")
    print(f"           CLIP BF16={bf16_clip:.2f}  FP8={quant_clip:.2f}  drop={clip_drop:.2f}")
    print(f"           metrics={metrics_path}")

    assert quant_clip >= CLIP_ABSOLUTE_FLOOR, (
        f"{case.name} CLIP below floor: got {quant_clip:.2f}, expected >= {CLIP_ABSOLUTE_FLOOR:.2f}"
    )
    assert clip_drop <= case.clip_drop, (
        f"{case.name} CLIP drop too large: got {clip_drop:.2f}, expected <= {case.clip_drop:.2f}"
    )

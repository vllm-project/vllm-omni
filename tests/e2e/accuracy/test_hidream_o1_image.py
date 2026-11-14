# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from tests.e2e.accuracy.helpers import (
    assert_similarity,
    measure_image_similarity,
    model_output_dir,
    write_accuracy_result,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.slow, pytest.mark.diffusion]

MODEL_ID = "HiDream-ai/HiDream-O1-Image"
PROMPT = "A cat is sitting next to a sign that says 'HiDream-O1 vLLM-Omni'"
BASELINE_PATH = Path(__file__).resolve().parents[2] / "assets" / "hidream_o1" / "hidream_o1_baseline.png"
SSIM_THRESHOLD = 0.8
PSNR_THRESHOLD = 19.0


def _assert_matches_baseline(
    parallel_config: DiffusionParallelConfig,
    label: str,
    accuracy_artifact_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    model = os.environ.get("HIDREAM_O1_IMAGE_MODEL", MODEL_ID)
    with OmniRunner(
        model,
        parallel_config=parallel_config,
        enforce_eager=True,
    ) as runner:
        outputs = runner.omni.generate(
            {"prompt": PROMPT},
            OmniDiffusionSamplingParams(
                height=2048,
                width=2048,
                num_inference_steps=50,
                guidance_scale=5.0,
                seed=42,
            ),
        )
        images = extract_images_from_outputs(outputs)
        assert images
        image = images[0].convert("RGB")

    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)
    image.save(output_dir / f"vllm_omni_{label}.png")
    assert BASELINE_PATH.exists()
    baseline = Image.open(BASELINE_PATH).convert("RGB")
    resized = image.resize(baseline.size, Image.Resampling.LANCZOS)
    ssim, psnr = measure_image_similarity(
        model_name=MODEL_ID,
        vllm_image=resized,
        diffusers_image=baseline,
    )
    write_accuracy_result(
        model_id=MODEL_ID,
        dataset="image_similarity",
        metrics={"ssim": ssim, "psnr": psnr},
        baseline={"ssim": SSIM_THRESHOLD, "psnr": PSNR_THRESHOLD},
        extra={"label": label},
    )
    assert_similarity(
        model_name=MODEL_ID,
        vllm_image=resized,
        diffusers_image=baseline,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
        ssim_score=ssim,
        psnr_score=psnr,
    )


@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_hidream_o1_image_tp2_matches_baseline(
    accuracy_artifact_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_matches_baseline(
        DiffusionParallelConfig(tensor_parallel_size=2),
        "tp2",
        accuracy_artifact_root,
        monkeypatch,
    )


@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_hidream_o1_image_cfg_parallel2_matches_baseline(
    accuracy_artifact_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_matches_baseline(
        DiffusionParallelConfig(cfg_parallel_size=2),
        "cfg_parallel2",
        accuracy_artifact_root,
        monkeypatch,
    )

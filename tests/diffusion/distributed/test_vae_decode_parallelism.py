# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""VAE patch-parallel decode parity tests for diffusion models.

Compares ``vae_patch_parallel_size=2`` against ``vae_patch_parallel_size=1`` on
TP=2 with VAE tiling enabled and verifies that the requested distributed VAE
state is active on every worker. Covers one video model (Wan 2.1 T2V 1.3B) and
the shared Qwen-Image VAE path with a lightweight transformer fixture. Official
Qwen-Image transformer loading/execution is covered by the existing official
model E2E, online-serving, and accuracy suites. CUDA/ROCm-only (>=2 devices).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

PROMPT = "A cat sitting on a table"
WORKER_EXTENSION_CLASS = "tests.diffusion.distributed.test_vae_decode_parallelism.VaeParallelStateExtension"

MODEL_CASES = [
    pytest.param(
        {
            "model_name": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "height": 512,
            "width": 512,
            "num_frames": 5,
            "needs_image": False,
            "is_moe": False,
            "mean_threshold": 3e-2,
            "p99_threshold": 4e-2,
        },
        id="wan21_t2v_1_3b",
    ),
    pytest.param(
        {
            "model_name": "riverclouds/qwen_image_random",
            "height": 512,
            "width": 512,
            "num_frames": 1,
            "needs_image": False,
            "is_moe": False,
            "mean_threshold": 5e-3,
            "p99_threshold": 1e-1,
        },
        id="qwen_image_random",
    ),
]


class VaeParallelStateExtension:
    def validate_vae_parallel_state(self, expected_parallel_size: int) -> bool:
        vae = self.model_runner.pipeline.vae
        actual_parallel_size = int(vae.distributed_executor.parallel_size)
        distributed_enabled = bool(vae.is_distributed_enabled())
        return actual_parallel_size == expected_parallel_size and distributed_enabled == (expected_parallel_size > 1)


def _to_float_array(data: Any) -> np.ndarray:
    if isinstance(data, Image.Image):
        return np.asarray(data.convert("RGB"), dtype=np.float32) / 255.0
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _extract_output_array(output: Any) -> np.ndarray:
    payload = output.images[0]
    return _to_float_array(payload)


def test_extract_output_array_accepts_flat_omni_output():
    payload = np.zeros((2, 2, 3), dtype=np.float32)

    result = _extract_output_array(SimpleNamespace(images=[payload]))

    np.testing.assert_array_equal(result, payload)


def _diff_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    assert a.shape == b.shape, f"Output shapes differ: {a.shape} vs {b.shape}"
    abs_diff = np.abs(a - b)
    return float(abs_diff.mean()), float(np.quantile(abs_diff, 0.99))


def _save_preview(arr: np.ndarray, path: Path) -> None:
    """Save a single RGB frame preview; video tensors use the first frame."""
    preview = np.asarray(arr)
    # (B, T, H, W, C) video
    if preview.ndim == 5:
        preview = preview[0, 0]
    # (T, H, W, C) video or (B, H, W, C) image batch
    elif preview.ndim == 4:
        preview = preview[0]
    if preview.ndim != 3 or preview.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HWC RGB(A) preview, got shape {arr.shape} -> {preview.shape}")
    preview = np.clip(preview[..., :3], 0.0, 1.0)
    Image.fromarray((preview * 255.0).astype(np.uint8)).save(path)


def _build_sampling_params(model_case: dict[str, Any], seed: int) -> OmniDiffusionSamplingParams:
    kwargs: dict[str, Any] = {
        "height": model_case["height"],
        "width": model_case["width"],
        "num_frames": model_case["num_frames"],
        "num_inference_steps": 2,
        "seed": seed,
    }
    if model_case["model_name"].startswith("Wan-AI/"):
        kwargs["guidance_scale"] = 4.0
    if model_case["is_moe"]:
        kwargs["guidance_scale_2"] = 1.0
        kwargs["boundary_ratio"] = 0.5
    return OmniDiffusionSamplingParams(**kwargs)


def _run_generate(
    *,
    model_case: dict[str, Any],
    vae_patch_parallel_size: int,
    seed: int,
) -> np.ndarray:
    current_omni_platform.empty_cache()
    with OmniRunner(
        model_case["model_name"],
        tensor_parallel_size=2,
        vae_patch_parallel_size=vae_patch_parallel_size,
        vae_use_tiling=True,
        enforce_eager=False,
        worker_extension_cls=WORKER_EXTENSION_CLASS,
    ) as runner:
        vae_state = runner.omni.engine.collective_rpc(
            "validate_vae_parallel_state",
            timeout=60,
            args=(vae_patch_parallel_size,),
        )
        assert vae_state == [[True]], (
            f"Expected every worker to use VAE patch parallel size {vae_patch_parallel_size}, got {vae_state}"
        )
        request: dict[str, Any] = {"prompt": model_case.get("prompt", PROMPT)}
        if model_case["needs_image"]:
            request["multi_modal_data"] = {
                "image": Image.new("RGB", (model_case["width"], model_case["height"]), (0, 0, 0))
            }
        outputs = runner.omni.generate(
            request,
            sampling_params_list=_build_sampling_params(model_case, seed),
        )
        return _extract_output_array(outputs[0])


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 4})
@pytest.mark.parametrize("model_case", MODEL_CASES)
def test_vae_patch_parallel_tp2(model_case: dict[str, Any], tmp_path: Path):
    if current_omni_platform.is_npu():
        pytest.skip("VAE patch parallel test is only supported on CUDA and ROCm for now.")
    if not current_omni_platform.is_available() or current_omni_platform.device_count() < 2:
        pytest.skip("VAE patch parallel TP=2 requires >= 2 devices.")

    seed = 42
    model_id = model_case["model_name"].split("/")[-1]

    baseline = _run_generate(model_case=model_case, vae_patch_parallel_size=1, seed=seed)
    parallel = _run_generate(model_case=model_case, vae_patch_parallel_size=2, seed=seed)

    baseline_path = tmp_path / f"{model_id}_tp2_vae_pp1.png"
    parallel_path = tmp_path / f"{model_id}_tp2_vae_pp2.png"
    _save_preview(baseline, baseline_path)
    _save_preview(parallel, parallel_path)

    mean_abs_diff, p99_abs_diff = _diff_metrics(baseline, parallel)
    mean_threshold = model_case["mean_threshold"]
    p99_threshold = model_case["p99_threshold"]
    print(
        f"{model_case['model_name']} VAE patch parallel diff (TP=2, pp=1 vs pp=2): "
        f"mean_abs_diff={mean_abs_diff:.6e}, p99_abs_diff={p99_abs_diff:.6e}; "
        f"thresholds: mean<={mean_threshold:.6e}, p99<={p99_threshold:.6e}; "
        f"pp1_preview={baseline_path}, pp2_preview={parallel_path}"
    )
    assert mean_abs_diff <= mean_threshold and p99_abs_diff <= p99_threshold, (
        f"Output diff exceeded threshold: mean_abs_diff={mean_abs_diff:.6e}, "
        f"p99_abs_diff={p99_abs_diff:.6e} "
        f"(thresholds: mean<={mean_threshold:.6e}, p99<={p99_threshold:.6e})"
    )

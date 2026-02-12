# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online serving test for /v1/images/generations with per-request LoRA.

This validates:
- The API server accepts a per-request `lora` object in the Images API payload.
- LoRA can be switched per request (adapter A -> adapter B -> no LoRA).
- Output correctness is asserted using a small image slice with tolerance.
"""

import base64
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests
import torch
from PIL import Image
from safetensors.torch import save_file

from tests.conftest import OmniServer
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Tongyi-MAI/Z-Image-Turbo"


PROMPT = "a photo of a cat sitting on a laptop keyboard"
SIZE = "256x256"
SEED = 42


@pytest.fixture(scope="module")
def omni_server():
    with OmniServer(MODEL, ["--num-gpus", "1"]) as server:
        yield server


def _write_zimage_lora(adapter_dir: Path, *, q_scale: float = 0.0, k_scale: float = 0.0, v_scale: float = 0.0):
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Z-Image transformer uses dim=3840 by default.
    dim = 3840
    module_name = "transformer.layers.0.attention.to_qkv"
    rank = 1

    lora_a = torch.zeros((rank, dim), dtype=torch.float32)
    lora_a[0, 0] = 1.0

    # QKVParallelLinear packs (Q, K, V) => out dim is 3 * dim (tp=1).
    lora_b = torch.zeros((3 * dim, rank), dtype=torch.float32)
    if q_scale:
        lora_b[:dim, 0] = q_scale
    if k_scale:
        lora_b[dim : 2 * dim, 0] = k_scale
    if v_scale:
        lora_b[2 * dim :, 0] = v_scale

    save_file(
        {
            f"base_model.model.{module_name}.lora_A.weight": lora_a,
            f"base_model.model.{module_name}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": rank,
                "lora_alpha": rank,
                "target_modules": [module_name],
            }
        ),
        encoding="utf-8",
    )


def _post_images(server: OmniServer, payload: dict) -> Image.Image:
    url = f"http://{server.host}:{server.port}/v1/images/generations"
    resp = requests.post(url, json=payload, headers={"Authorization": "Bearer EMPTY"}, timeout=900)
    resp.raise_for_status()
    data = resp.json()
    b64 = data["data"][0]["b64_json"]
    img_bytes = base64.b64decode(b64)
    img = Image.open(BytesIO(img_bytes))
    img.load()
    return img.convert("RGB")


def _image_blue_tail_slice(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    assert arr.ndim == 3 and arr.shape[-1] == 3
    tail = arr[-3:, -3:, -1].astype(np.float32)
    assert tail.shape == (3, 3)
    return tail


def _slice_diff_stats(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    diff = np.abs(actual - expected)
    return float(diff.max()), float(diff.mean())


def _assert_slice_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
    base_max: float,
    base_mean: float,
) -> None:
    assert actual.shape == (3, 3)
    assert expected.shape == (3, 3)
    max_diff, mean_diff = _slice_diff_stats(actual, expected)
    # NOTE: Different attention backends / torch.compile can introduce small
    # floating-point drift that shows up as a few LSBs in uint8 pixels. Keep
    # the reset check tolerant but bounded to avoid flaky CI.
    max_thresh = max(10.0, base_max + 4.0)
    mean_thresh = max(6.0, base_mean + 4.0)
    assert max_diff <= max_thresh and mean_diff <= mean_thresh, (
        f"{label} slice mismatch (max={max_diff:.1f} > {max_thresh:.1f} or "
        f"mean={mean_diff:.1f} > {mean_thresh:.1f}): {actual.tolist()}"
    )


def _assert_slice_diff(actual: np.ndarray, baseline: np.ndarray, *, label: str) -> None:
    assert actual.shape == (3, 3)
    assert baseline.shape == (3, 3)
    diff = np.abs(actual - baseline).mean()
    assert diff > 0.1, f"{label} slice diff too small: {diff} ({actual.tolist()} vs {baseline.tolist()})"


def _basic_payload() -> dict:
    return {
        "prompt": PROMPT,
        "n": 1,
        "size": SIZE,
        "num_inference_steps": 2,
        "guidance_scale": 0.0,
        "seed": SEED,
    }


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
def test_images_generations_per_request_lora_switching(omni_server: OmniServer, tmp_path: Path) -> None:
    # Base generation.
    base_img = _post_images(omni_server, _basic_payload())
    base_slice = _image_blue_tail_slice(base_img)
    base_ref_img = _post_images(omni_server, _basic_payload())
    base_ref_slice = _image_blue_tail_slice(base_ref_img)
    base_ref_max, base_ref_mean = _slice_diff_stats(base_ref_slice, base_slice)

    # Adapter A: apply delta to V slice only.
    lora_a_dir = tmp_path / "zimage_lora_a"
    _write_zimage_lora(lora_a_dir, v_scale=8.0)
    payload_a = _basic_payload()
    payload_a["lora"] = {"name": "a", "path": str(lora_a_dir), "scale": 64.0}
    img_a = _post_images(omni_server, payload_a)
    a_slice = _image_blue_tail_slice(img_a)
    _assert_slice_diff(a_slice, base_slice, label="lora_a_vs_base")
    a_vs_base = float(np.abs(a_slice - base_slice).mean())

    # Adapter B: apply delta to K slice only (should differ from adapter A).
    lora_b_dir = tmp_path / "zimage_lora_b"
    _write_zimage_lora(lora_b_dir, k_scale=4.0)
    payload_b = _basic_payload()
    payload_b["lora"] = {"name": "b", "path": str(lora_b_dir), "scale": 64.0}
    img_b = _post_images(omni_server, payload_b)
    b_slice = _image_blue_tail_slice(img_b)
    _assert_slice_diff(b_slice, base_slice, label="lora_b_vs_base")
    _assert_slice_diff(b_slice, a_slice, label="lora_b_vs_lora_a")
    b_vs_base = float(np.abs(b_slice - base_slice).mean())
    b_vs_a = float(np.abs(b_slice - a_slice).mean())

    # Ensure switching back to no-LoRA restores the base output.
    base_img_2 = _post_images(omni_server, _basic_payload())
    base_slice_2 = _image_blue_tail_slice(base_img_2)
    _, base_reset_mean = _slice_diff_stats(base_slice_2, base_slice)
    _assert_slice_close(
        base_slice_2,
        base_slice,
        label="base_after_reset",
        base_max=base_ref_max,
        base_mean=base_ref_mean,
    )

    # Ensure LoRA effects are clearly above the baseline drift.
    min_delta = max(base_reset_mean + 1.0, 1.5)
    assert a_vs_base > min_delta, f"lora_a_vs_base drift too small: {a_vs_base} <= {min_delta}"
    assert b_vs_base > min_delta, f"lora_b_vs_base drift too small: {b_vs_base} <= {min_delta}"
    assert b_vs_a > min_delta, f"lora_b_vs_lora_a drift too small: {b_vs_a} <= {min_delta}"

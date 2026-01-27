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
import signal
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests
import torch
from PIL import Image
from safetensors.torch import save_file
from vllm.utils.network_utils import get_open_port

from vllm_omni.utils.platform_utils import is_npu

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Tongyi-MAI/Z-Image-Turbo"


PROMPT = "a photo of a cat sitting on a laptop keyboard"
SIZE = "256x256"
SEED = 42


class OmniServer:
    """Omniserver for vLLM-Omni tests."""

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
        self.env_dict = env_dict
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        self.port = get_open_port()

    def _start_server(self) -> None:
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.serve_args

        print(f"Launching OmniServer with: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # vllm-omni root
            start_new_session=True,
        )

        # Wait for server to be ready.
        max_wait = 1200
        url = f"http://{self.host}:{self.port}/v1/models"
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                resp = requests.get(url, headers={"Authorization": "Bearer EMPTY"}, timeout=10)
                if resp.status_code == 200:
                    print(f"Server ready on {self.host}:{self.port}")
                    return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError(f"Server failed to become ready within {max_wait} seconds")

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()


@pytest.fixture(scope="module")
def omni_server():
    if is_npu():
        pytest.skip("Tongyi-MAI/Z-Image-Turbo is not supported on NPU yet.")
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


def _assert_slice_close(actual: np.ndarray, expected: np.ndarray, *, label: str) -> None:
    assert actual.shape == (3, 3)
    assert expected.shape == (3, 3)
    diff = np.abs(actual - expected)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    # NOTE: Different attention backends / torch.compile can introduce small
    # floating-point drift that shows up as a few LSBs in uint8 pixels. Keep
    # the reset check tolerant but bounded to avoid flaky CI.
    assert max_diff <= 5.0 and mean_diff <= 3.0, (
        f"{label} slice mismatch (max={max_diff:.1f}, mean={mean_diff:.1f}): {actual.tolist()}"
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


def test_images_generations_per_request_lora_switching(omni_server: OmniServer, tmp_path: Path) -> None:
    # Base generation.
    base_img = _post_images(omni_server, _basic_payload())
    base_slice = _image_blue_tail_slice(base_img)

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
    _assert_slice_close(base_slice_2, base_slice, label="base_after_reset")
    base_reset = float(np.abs(base_slice_2 - base_slice).mean())

    # Ensure LoRA effects are clearly above the baseline drift.
    min_delta = base_reset + 0.5
    assert a_vs_base > min_delta, f"lora_a_vs_base drift too small: {a_vs_base} <= {min_delta}"
    assert b_vs_base > min_delta, f"lora_b_vs_base drift too small: {b_vs_base} <= {min_delta}"
    assert b_vs_a > min_delta, f"lora_b_vs_lora_a drift too small: {b_vs_a} <= {min_delta}"

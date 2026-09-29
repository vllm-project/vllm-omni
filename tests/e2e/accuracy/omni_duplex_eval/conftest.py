# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Module-scoped judge server fixture for omni-duplex-eval CI guard."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Generator
from pathlib import Path

import pytest
import requests
import torch

_CONFIG_PATH = Path(__file__).parent / "omni_duplex_eval_ci_config.json"


def _load_judge_model() -> str:
    """Read judge model from CI config, fall back to default."""
    try:
        config = json.loads(_CONFIG_PATH.read_text())
        return config.get("judge", {}).get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
    except Exception:
        return "Qwen/Qwen2.5-VL-7B-Instruct"


def _parse_visible_devices() -> tuple[str, list[str]]:
    """Parse CUDA or Ascend NPU visible devices from environment."""
    if torch.cuda.is_available():
        env_key = "CUDA_VISIBLE_DEVICES"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        env_key = "ASCEND_RT_VISIBLE_DEVICES"
    else:
        return "", []

    raw = os.environ.get(env_key, "").strip()
    if not raw:
        # No env restriction — physical device IDs are 0, 1, 2...
        count = torch.accelerator.device_count() if hasattr(torch, "accelerator") else 0
        return env_key, [str(i) for i in range(count)]

    devices = [d.strip() for d in raw.split(",") if d.strip()]
    return env_key, devices


def _omni_server_device_env() -> dict[str, str] | None:
    """Env overrides pinning the Omni server to device[1] (split from judge).

    Returns ``None`` when fewer than two devices are allocated (nothing to
    split, e.g. on CPU-only hosts where ``_parse_visible_devices`` yields an
    empty list). Pure function: never mutates the pytest process environment.
    """
    env_key, devices = _parse_visible_devices()
    if len(devices) >= 2:
        return {env_key: devices[1]}
    return None


@pytest.fixture(scope="module")
def judge_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[str, None, None]:
    """Launch the Qwen2.5-VL-7B judge server, wait for health, yield base URL.

    If ``VLLM_DUPLEX_EVAL_JUDGE_URL`` is already set *and* the endpoint
    responds to ``/health``, reuse it (developer convenience).

    Otherwise, start the judge as a subprocess:
    - The judge is pinned to the first device of the job's allocation
      (``devices[0]``), so it never selects a device outside the job's set.
    - The omni server (managed by the ``omni_server`` fixture) is pinned to
      device[1] via ``OmniServerParams.env_dict`` (see
      ``_omni_server_device_env``), leaving the pytest process environment
      untouched.

    The fixture terminates the judge on teardown.
    """
    port = int(os.environ.get("VLLM_DUPLEX_EVAL_JUDGE_PORT", "8001"))
    base_url = f"http://127.0.0.1:{port}"

    # Reuse a running judge if URL is pre-set and healthy.
    preset_url = os.environ.get("VLLM_DUPLEX_EVAL_JUDGE_URL")
    if preset_url:
        try:
            resp = requests.get(f"{preset_url}/health", timeout=5)
            if resp.status_code == 200:
                yield preset_url
                return
        except Exception:
            pass

    # Determine judge device from the job's device allocation.
    _, devices = _parse_visible_devices()
    judge_dev = devices[0] if devices else "0"

    # Build device-isolated environment.
    env = os.environ.copy()
    if torch.cuda.is_available():
        env["CUDA_VISIBLE_DEVICES"] = judge_dev
        env["ASCEND_RT_VISIBLE_DEVICES"] = ""  # clear NPU visibility
    elif hasattr(torch, "npu") and torch.npu.is_available():
        env["ASCEND_RT_VISIBLE_DEVICES"] = judge_dev
        env["CUDA_VISIBLE_DEVICES"] = ""  # clear CUDA visibility

    # Derive media allowlist from pytest's actual temp root.
    media_path = str(tmp_path_factory.getbasetemp())
    os.makedirs(media_path, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        _load_judge_model(),
        "--port",
        str(port),
        "--max-model-len",
        "65536",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.85",
        "--allowed-local-media-path",
        media_path,
    ]

    proc = subprocess.Popen(cmd, env=env)

    # Health-check loop: up to 60 iterations × 5 s = 5 min.
    for _ in range(60):
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"Judge server failed to become healthy at {base_url} within 300 s")

    yield base_url

    # Teardown.
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E online serving test for DreamZero OpenPI websocket serving."""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import OmniServerParams
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL = "GEAR-Dreams/DreamZero-DROID"
EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "online_serving" / "dreamzero"
CLIENT_SCRIPT = EXAMPLE_DIR / "openpi_client.py"
ASSETS_DIR = EXAMPLE_DIR / "assets"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _pick_test_gpus() -> str:
    override = os.environ.get("DREAMZERO_TEST_GPUS") or os.environ.get("OPENPI_E2E_GPUS")
    if override:
        return override

    try:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return "0,1"

    gpu_rows = []
    for line in query.strip().splitlines():
        gpu_index, used_mb = [part.strip() for part in line.split(",", maxsplit=1)]
        gpu_rows.append((int(used_mb), gpu_index))
    gpu_rows.sort()
    return ",".join(gpu_index for _, gpu_index in gpu_rows[:2]) or "0,1"


test_params = [
    OmniServerParams(
        model=MODEL,
        port=8091,
        server_args=[
            "--cfg-parallel-size",
            "2",
            "--enforce-eager",
            "--disable-log-stats",
        ],
        env_dict={
            "ATTENTION_BACKEND": "torch",
            "DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "CUDA_VISIBLE_DEVICES": _pick_test_gpus(),
            "MASTER_PORT": str(_find_free_port()),
        },
    )
]


def _load_client_module():
    spec = importlib.util.spec_from_file_location("dreamzero_openpi_example_client", CLIENT_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        pytest.skip(f"DreamZero OpenPI example dependency is missing: {exc.name}")
    return module


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_dreamzero_openpi_online(omni_server) -> None:
    client_mod = _load_client_module()
    result = client_mod.run_policy_session(
        host=omni_server.host,
        port=omni_server.port,
        video_dir=ASSETS_DIR,
        session_id="dreamzero-online-e2e",
    )

    client_mod.validate_session_result(result)

    metadata = result["metadata"]
    assert metadata["needs_session_id"] is True
    assert metadata["needs_stereo_camera"] is False
    assert tuple(metadata["image_resolution"]) == (180, 320)

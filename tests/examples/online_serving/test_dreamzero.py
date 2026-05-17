# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Online serving example test: DreamZero.
See examples/online_serving/dreamzero/README.md
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.examples.helpers import run_cmd
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

pytestmark = [pytest.mark.advanced_model, pytest.mark.example]

MODEL = "GEAR-Dreams/DreamZero-DROID"
EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "online_serving" / "dreamzero"


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
        port=8092,
        server_args=[
            "--deploy-config",
            "vllm_omni/deploy/dreamzero_tp1_cfg2.yaml",
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


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_dreamzero_openpi_client_example(omni_server) -> None:
    command = [
        sys.executable,
        str(EXAMPLE_DIR / "openpi_client.py"),
        "--host",
        omni_server.host,
        "--port",
        str(omni_server.port),
    ]

    result = run_cmd(command)
    assert "Server metadata:" in result
    assert "Action 0:" in result
    assert "Action 1:" in result
    assert "Action 2:" in result
    assert "Reset status: reset successful" in result

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU tests for the SeedVR2 window-aligned SP runtime.

These tests spawn ``window_sp_worker.py`` with ``torchrun`` and assert on the
per-rank reports it writes; they are skipped when the requested number of GPUs
is unavailable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.helpers.mark import hardware_marks

# The worker cases run on the small-GPU SKU class; individual tests skip when the
# requested number of cards is unavailable.
pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.parallel,
    pytest.mark.sp,
    pytest.mark.advanced_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=2),
]

WORKER = Path(__file__).with_name("window_sp_worker.py")


def _available_gpus() -> int:
    if not torch.accelerator.is_available():
        return 0
    return torch.accelerator.device_count()


def _run_worker(case: str, world_size: int, tmp_path: Path, *extra: str) -> dict:
    report_dir = tmp_path / f"{case}-{world_size}"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4])
    env["MASTER_PORT"] = str(29500 + world_size + (0 if case == "transport" else 10))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        str(WORKER),
        "--case",
        case,
        "--seed",
        "7723",
        "--report-dir",
        str(report_dir),
        *extra,
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1800, check=False)
    report_path = report_dir / "report.json"
    assert report_path.exists(), (
        f"worker produced no report (rc={result.returncode})\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
    )
    return json.loads(report_path.read_text())


@pytest.mark.parametrize("world_size", [2, 4])
def test_transport_rows_move_bit_exactly(world_size, tmp_path):
    if _available_gpus() < world_size:
        pytest.skip(f"needs {world_size} GPUs, found {_available_gpus()}")
    report = _run_worker("transport", world_size, tmp_path, "--frames", "2", "--height", "32", "--width", "32")
    for entry in report.values():
        assert entry["status"] == "ok", entry.get("error")
        assert entry["transitions_checked"] == entry["schedule_transitions"]
        assert entry["synchronized"] is True


@pytest.mark.parametrize("world_size", [2, 4])
def test_toy_block_matches_single_rank_oracle(world_size, tmp_path):
    if _available_gpus() < world_size:
        pytest.skip(f"needs {world_size} GPUs, found {_available_gpus()}")
    report = _run_worker("toy-block", world_size, tmp_path, "--frames", "6", "--height", "40", "--width", "40")
    for entry in report.values():
        assert entry["status"] == "ok", entry.get("error")
        assert entry["video_max_abs_error"] < 1e-9
        assert entry["text_max_abs_error"] < 1e-9
        assert entry["network_transitions"] > 0, "the A->B schedule must actually exchange rows"


@pytest.mark.parametrize("world_size", [2])
def test_seedvr2_sp_degree_invariance(world_size, tmp_path):
    """Real 3B checkpoint: SP=N must match the same port at SP=1."""
    if _available_gpus() < world_size:
        pytest.skip(f"needs {world_size} GPUs, found {_available_gpus()}")
    checkpoint = os.environ.get("SEEDVR2_CHECKPOINT", "/models/seedvr2_ema_3b_fp16.safetensors")
    if not Path(checkpoint).exists():
        pytest.skip(f"SeedVR2 checkpoint not found at {checkpoint}")
    report = _run_worker(
        "seedvr2",
        world_size,
        tmp_path,
        "--ckpt",
        checkpoint,
        "--frames",
        "1",
        "--height",
        "64",
        "--width",
        "64",
        "--text-len",
        "58",
    )
    for entry in report.values():
        assert entry["status"] == "ok", entry.get("error")
        assert entry["finite"] is True
        assert entry["rel_l2"] < entry["tolerance_rtol"]
        assert entry["network_a2a_count"] > 0
        assert entry["text_all_reduce_count"] > 0

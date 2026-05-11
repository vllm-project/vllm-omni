# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline E2E test for InternVLA-A1 open-loop inference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test

REPO_ROOT = Path(__file__).resolve().parents[3]


EXAMPLE_SCRIPT = REPO_ROOT / "examples" / "offline_inference" / "internvla_a1" / "end2end.py"


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return value


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
@pytest.mark.usefixtures("clean_gpu_memory_between_tests")
def test_internvla_a1_offline_open_loop(run_level: str) -> None:
    if run_level != "advanced_model":
        pytest.skip("InternVLA-A1 offline evaluation requires real local checkpoints and dataset.")

    model_dir = _required_env("INTERNVLA_A1_MODEL_DIR")
    dataset_dir = _required_env("INTERNVLA_A1_DATASET_DIR")
    processor_dir = _required_env("INTERNVLA_A1_PROCESSOR_DIR")
    cosmos_dir = _required_env("INTERNVLA_A1_COSMOS_DIR")

    env = os.environ.copy()
    env["INTERNVLA_A1_PROCESSOR_DIR"] = processor_dir
    env["INTERNVLA_A1_COSMOS_DIR"] = cosmos_dir

    with tempfile.TemporaryDirectory(prefix="internvla_a1_e2e_") as tmpdir:
        output_dir = Path(tmpdir) / "outputs"
        cmd = [
            sys.executable,
            str(EXAMPLE_SCRIPT),
            "--model-dir",
            model_dir,
            "--dataset-dir",
            dataset_dir,
            "--output-dir",
            str(output_dir),
            "--num-samples",
            "1",
            "--num-episodes",
            "1",
            "--dtype",
            "float32",
            "--attn-implementation",
            "eager",
            "--skip-plots",
        ]
        subprocess.run(cmd, check=True, env=env)

        summary_path = output_dir / "summary.json"
        assert summary_path.exists(), f"Missing summary file: {summary_path}"

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["mode"] == "registry"
        assert summary["attn_implementation"] == "eager"
        assert summary["dtype"] == "float32"
        assert len(summary["results"]) == 1
        assert "registry" in summary["eval_summaries"]

        eval_summary = summary["eval_summaries"]["registry"]
        assert eval_summary["num_episodes"] == 1
        assert eval_summary["average_mse"] is not None
        assert eval_summary["average_mae"] is not None

        log_path = output_dir / "registry" / "log.json"
        assert log_path.exists(), f"Missing evaluation log: {log_path}"

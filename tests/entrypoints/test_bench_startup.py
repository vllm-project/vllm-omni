# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import subprocess

import pytest

from tests.helpers.mark import hardware_test
from tests.model_tests.diffusion.diff_model_builders import tiny_flux2_klein_builder


@pytest.mark.benchmark
@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
def test_bench_startup_omni(tmp_path):
    """Ensure `vllm bench startup --omni` runs end to end and reports the startup time."""
    output = tmp_path / "startup.json"
    command = [
        "vllm",
        "bench",
        "startup",
        "--omni",
        "--model",
        tiny_flux2_klein_builder(),
        "--enforce-eager",
        "--num-iters-cold",
        "1",
        "--num-iters-warmup",
        "0",
        "--num-iters-warm",
        "0",
        "--output-json",
        str(output),
    ]
    subprocess.run(command, check=True, timeout=900)

    results = json.loads(output.read_text())
    (startup_time,) = results["cold_startup_times"]
    assert startup_time > 0
    # Since we only run once with cold to keep the test lighter, we only have cold metrics here
    actual_keys = list(results.keys())
    expected_keys = ["avg_cold_startup_time", "cold_startup_times", "cold_startup_percentiles"]
    assert actual_keys == expected_keys

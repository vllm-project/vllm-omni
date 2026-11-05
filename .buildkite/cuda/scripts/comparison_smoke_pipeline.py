# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Render a one-off PR 1994 CUDA build containing only the selected SGLang smokes."""

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1] / "test-sglang-omni.yml"
# Accelerator requests correspond to each runner's actual model placement.
GPU_COUNTS = {
    "minicpm-o": 1,
    "qwen-image": 1,
    "wan22-i2v": 1,
    "hunyuan-image3": 4,
    "minimax-h3": 2,
    "cosmos3-nano": 1,
    "hunyuanvideo15": 1,
}


def smoke_pipeline(cases: list[str]) -> dict:
    if not cases or len(cases) != len(set(cases)) or set(cases) - GPU_COUNTS.keys():
        raise ValueError(f"Select distinct model keys from: {', '.join(GPU_COUNTS)}")
    # The same agent that uploads the result resolves the YAML anchors. Leave
    # environment interpolation to the final upload, after building the steps.
    base = subprocess.check_output(
        ["buildkite-agent", "pipeline", "upload", "--dry-run", "--no-interpolation", "--format", "json", str(PIPELINE)]
    )
    pipeline = json.loads(base)
    weekly_steps = {step["key"]: step for step in pipeline["steps"]}
    image = weekly_steps["image-build-sglang-omni"]
    benchmark = weekly_steps["nightly-sglang-omni-qwen3-omni"]
    pipeline["steps"] = [image]
    for case in cases:
        job = deepcopy(benchmark)
        job.update(
            key=f"smoke-sglang-{case}",
            label=f":gpu: SGLang · {case} · one request",
            env={"SMOKE_CASE": case},
            concurrency=2,
            concurrency_group="vllm-omni/pr1994/sglang-smoke",
            artifact_paths=["tests/dfx/perf/results/smoke_*.json"],
        )
        plugin = job["plugins"][0]["github.com/buildkite-plugins/kubernetes-buildkite-plugin"]
        container = plugin["podSpec"]["containers"][0]
        container["resources"]["limits"]["nvidia.com/gpu"] = GPU_COUNTS[case]
        commands = [
            'test "$$(git rev-parse HEAD)" = "$${BACKEND_COMPARISON_COMMIT}"',
            'export HF_TOKEN="$${VLLM_CI_HF_TOKEN}"',
        ]
        if case == "minicpm-o":
            # Use the same client/voice-conditioning path as the measured case.
            # Keep the single request in a distinct artifact directory.
            commands.extend(
                [
                    "export BENCHMARK_DIR=tests/dfx/perf/results/minicpm-smoke",
                    """jq '.[0].benchmark_params[0] += {num_prompts:[1],num_warmups:0,ready_check_timeout_sec:0}' """
                    "tests/dfx/perf/tests/test_minicpmo_4_5_sglang.json > /tmp/minicpm-smoke.json",
                    "pytest -s -v tests/dfx/perf/scripts/run_benchmark.py --test-config-file /tmp/minicpm-smoke.json "
                    "--run-level full_model",
                    "jq -se 'length == 1 and (.[0] | .completed == 1 and .failed == 0 "
                    "and .total_audio_duration_s > 0 and .total_audio_frames > 0 and .mean_audio_ttfp_ms > 0)' "
                    '"$${BENCHMARK_DIR}"/*.json',
                ]
            )
            job["artifact_paths"] = ["tests/dfx/perf/results/minicpm-smoke/*.json"]
        else:
            commands.append(
                '/opt/sglang/bin/python3 tests/dfx/perf/scripts/run_sglang_diffusion_smoke.py "$${SMOKE_CASE}"'
            )
        job["command"] = "\n".join(commands)
        pipeline["steps"].append(job)
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", help="Comma-separated model keys")
    args = parser.parse_args()
    try:
        print(json.dumps(smoke_pipeline(args.cases.split(","))))
    except ValueError as exc:
        parser.error(str(exc))

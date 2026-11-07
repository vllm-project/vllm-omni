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
    "minicpm-o-perf": 1,
    "qwen-image": 1,
    "wan22-i2v": 1,
    "hunyuan-image3": 4,
    "minimax-h3": 2,
    "cosmos3-nano": 1,
    "hunyuanvideo15": 1,
}


# The #18852 image ran this exact model successfully; only the checkout's
# benchmark adapter/configuration differs for this native online run.
QWEN_IMAGE_DIGEST = "sha256:125988c0f4ec5f9e9f0e19d94b92b7e3855dbf6aaf063c3025af4ca8c3b990b7"


def qwen_image_performance_job(benchmark: dict) -> dict:
    job = deepcopy(benchmark)
    job.pop("depends_on", None)
    kube = job["plugins"][0]["github.com/buildkite-plugins/kubernetes-buildkite-plugin"]
    container = kube["podSpec"]["containers"][0]
    container["resources"]["limits"]["nvidia.com/gpu"] = 1
    container["image"] = container["image"].rsplit(":", 1)[0] + "@" + QWEN_IMAGE_DIGEST
    job.update(
        key="perf-sglang-qwen-image-21",
        label=":frame_with_picture: SGLang · Qwen-Image-2.1 · 512x512 · 20 steps · 10 prompts · C1",
        env={
            "COMPARISON_CONFIG": "tests/dfx/perf/tests/test_qwen_image_21_sglang.json",
            "COMPARISON_IMAGE_DIGEST": QWEN_IMAGE_DIGEST,
        },
        command="\n".join(
            [
                'test "$$(git rev-parse HEAD)" = "$${BACKEND_COMPARISON_COMMIT}"',
                'export HF_TOKEN="$${VLLM_CI_HF_TOKEN}"',
                'export PYTHONPATH="$$PWD$${PYTHONPATH:+:$$PYTHONPATH}"',
                "export BENCHMARK_DIR=tests/dfx/perf/results",
                "python3 .buildkite/cuda/scripts/check_vllm_runtime.py",
                "python3 -m pytest -s -v tests/dfx/perf/scripts/run_benchmark.py "
                '--test-config-file "$${COMPARISON_CONFIG}" --run-level full_model',
                "jq -se 'length == 1 and (.[0] | .completed == 10 and .failed == 0 and .total_images == 10 "
                "and .average_pixels_per_image == 262144 and .request_throughput > 0 and .mean_e2el_ms > 0 "
                'and .Hardware == "H100" and .experiment_metadata.server_revision == "0.5.22")\' '
                '"$${BENCHMARK_DIR}"/result_test_sglang_qwen_image_21_single_device_*.json',
            ]
        ),
        artifact_paths=["tests/dfx/perf/results/result_test_sglang_qwen_image_21_single_device_*.json"],
    )
    return job


def smoke_pipeline(cases: list[str]) -> dict:
    qwen_performance = cases == ["qwen-image-21-perf"]
    if not qwen_performance and (not cases or len(cases) != len(set(cases)) or set(cases) - GPU_COUNTS.keys()):
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
    if qwen_performance:
        pipeline["steps"] = [qwen_image_performance_job(benchmark)]
        return pipeline
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
        if case in {"minicpm-o", "minicpm-o-perf"}:
            commands.insert(0, "set -eu")
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
            if case == "minicpm-o-perf":
                job.update(
                    key="perf-sglang-minicpm-o",
                    label=":speaker: SGLang-Omni · MiniCPM-o 4.5 · smoke then C1/32",
                )
                commands.extend(
                    [
                        "export BENCHMARK_DIR=tests/dfx/perf/results",
                        "python3 .buildkite/cuda/scripts/check_vllm_runtime.py",
                        "pytest -s -v tests/dfx/perf/scripts/run_benchmark.py "
                        "--test-config-file tests/dfx/perf/tests/test_minicpmo_4_5_sglang.json --run-level full_model",
                        "jq -se 'length == 1 and (.[0] | .completed == 32 and .failed == 0 "
                        "and .num_audio_rtf_samples == 32 and .num_audio_ttfp_samples == 32 "
                        "and .total_audio_duration_s > 0 and .mean_audio_ttfp_ms > 0)' "
                        '"$${BENCHMARK_DIR}"/result_test_sglang_minicpmo_4_5_*.json',
                    ]
                )
                job["artifact_paths"].append("tests/dfx/perf/results/result_test_sglang_minicpmo_4_5_*.json")
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Render PR 1994's weekly comparisons or explicitly selected SGLang diagnostics."""

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


def wan_performance_job(benchmark: dict) -> dict:
    # The immutable image also passed the Wan I2V smoke in #18852.
    job = qwen_image_performance_job(benchmark)
    config = "tests/dfx/perf/tests/test_wan22_i2v_sglang.json"
    job.update(
        key="perf-sglang-wan22-i2v",
        label=":movie_camera: SGLang · Wan2.2 I2V · 832x480 · 81 frames · 4 steps · C1/10",
        concurrency=2,
        concurrency_group="vllm-omni/pr1994/sglang-smoke",
        env={"COMPARISON_CONFIG": config, "COMPARISON_IMAGE_DIGEST": QWEN_IMAGE_DIGEST},
        artifact_paths=[
            "tests/dfx/perf/results/wan-smoke/*.json",
            "tests/dfx/perf/results/result_test_sglang_wan22_i2v_single_device_*.json",
            "tests/dfx/perf/results/validated_wan22_i2v_media.json",
        ],
        command="\n".join(
            [
                "set -eu",
                'test "$$(git rev-parse HEAD)" = "$${BACKEND_COMPARISON_COMMIT}"',
                'export HF_TOKEN="$${VLLM_CI_HF_TOKEN}"',
                'export PYTHONPATH="$$PWD$${PYTHONPATH:+:$$PYTHONPATH}"',
                "export BENCHMARK_DIR=tests/dfx/perf/results",
                'test ! -e "$${BENCHMARK_DIR}/wan-videos"',
                'mkdir -p "$${BENCHMARK_DIR}/wan-videos"',
                "sglang_source_root=\"$$(/opt/sglang/bin/python3 -c 'import sglang; from pathlib import Path; "
                "print(Path(sglang.__file__).resolve().parent.parent)')\"",
                'git -C /tmp apply --unsafe-paths --directory="$${sglang_source_root}" --check '
                '"$$PWD/docker/patches/sglang-video-image-reference.patch"',
                'git -C /tmp apply --unsafe-paths --directory="$${sglang_source_root}" '
                '"$$PWD/docker/patches/sglang-video-image-reference.patch"',
                "CUDA_VISIBLE_DEVICES=\"\" /opt/sglang/bin/python3 -m pytest --noconftest -o addopts='' "
                "-q tests/dfx/perf/tests/test_sglang_video_api.py",
                "export COMPARISON_VIDEO_PATCH=1",
                "python3 .buildkite/cuda/scripts/check_vllm_runtime.py",
                "export BENCHMARK_DIR=tests/dfx/perf/results/wan-smoke",
                'test ! -e "$${BENCHMARK_DIR}"',
                'jq \'.[0].server_params.serve_args["output-path"] = '
                '"tests/dfx/perf/results/wan-smoke/videos" | '
                ".[0].benchmark_params[0] += {num_prompts:1,num_warmups:0}' "
                '"$${COMPARISON_CONFIG}" > /tmp/wan-smoke.json',
                "python3 -m pytest -s -v tests/dfx/perf/scripts/run_benchmark.py "
                "--test-config-file /tmp/wan-smoke.json --run-level full_model",
                "jq -se 'length == 1 and (.[0] | .completed == 1 and .failed == 0)' "
                '"$${BENCHMARK_DIR}"/result_test_sglang_wan22_i2v_single_device_*.json',
                "python3 tests/dfx/perf/scripts/validate_sglang_videos.py "
                '"$${BENCHMARK_DIR}/videos" --config /tmp/wan-smoke.json '
                '--output "$${BENCHMARK_DIR}/validated_media.json"',
                "export BENCHMARK_DIR=tests/dfx/perf/results",
                "python3 -m pytest -s -v tests/dfx/perf/scripts/run_benchmark.py "
                '--test-config-file "$${COMPARISON_CONFIG}" --run-level full_model',
                "jq -se 'length == 1 and (.[0] | .completed == 10 and .failed == 0 "
                "and .total_video_frames == 810 and .total_video_duration_s == 50.625 "
                "and .request_throughput > 0 and .mean_e2el_ms > 0 "
                'and .Hardware == "H100" and .experiment_metadata.server_revision == "0.5.22")\' '
                '"$${BENCHMARK_DIR}"/result_test_sglang_wan22_i2v_single_device_*.json',
                "python3 tests/dfx/perf/scripts/validate_sglang_videos.py "
                '"$${BENCHMARK_DIR}/wan-videos" --config "$${COMPARISON_CONFIG}" '
                '--output "$${BENCHMARK_DIR}/validated_wan22_i2v_media.json"',
            ]
        ),
    )
    return job


def cosmos_performance_job(benchmark: dict) -> dict:
    job = qwen_image_performance_job(benchmark)
    container = next(iter(job["plugins"][0].values()))["podSpec"]["containers"][0]
    container["resources"]["limits"]["nvidia.com/gpu"] = 2
    job.update(
        key="perf-sglang-cosmos3-t2i",
        label=":frame_with_picture: SGLang · Cosmos3 T2I · TP2 H100 · 1024x1024 · 4 steps · C1/3",
        env={
            "COMPARISON_CONFIG": "tests/dfx/perf/tests/test_cosmos3_t2i_sglang.json",
            "COMPARISON_IMAGE_DIGEST": QWEN_IMAGE_DIGEST,
        },
        artifact_paths=[
            "tests/dfx/perf/results/cosmos-smoke/*.json",
            "tests/dfx/perf/results/result_test_sglang_cosmos3_t2i_tp2_*.json",
        ],
        concurrency=2,
        concurrency_group="vllm-omni/pr1994/sglang-smoke",
        command="\n".join(
            [
                "set -eu",
                'test "$$(git rev-parse HEAD)" = "$${BACKEND_COMPARISON_COMMIT}"',
                'export HF_TOKEN="$${VLLM_CI_HF_TOKEN}"',
                'export PYTHONPATH="$$PWD$${PYTHONPATH:+:$$PYTHONPATH}"',
                "python3 .buildkite/cuda/scripts/check_vllm_runtime.py",
                "export BENCHMARK_DIR=tests/dfx/perf/results/cosmos-smoke",
                "jq '.[0].benchmark_params[0] += {num_prompts:1,num_warmups:0}' "
                '"$${COMPARISON_CONFIG}" > /tmp/cosmos-smoke.json',
                "python3 -m pytest -s -v tests/dfx/perf/scripts/run_benchmark.py "
                "--test-config-file /tmp/cosmos-smoke.json --run-level full_model",
                "jq -se 'length == 1 and (.[0] | .completed == 1 and .failed == 0 and .total_images == 1 "
                "and .average_pixels_per_image == 1048576)' "
                '"$${BENCHMARK_DIR}"/result_test_sglang_cosmos3_t2i_tp2_*.json',
                "export BENCHMARK_DIR=tests/dfx/perf/results",
                "python3 -m pytest -s -v tests/dfx/perf/scripts/run_benchmark.py "
                '--test-config-file "$${COMPARISON_CONFIG}" --run-level full_model',
                "jq -se 'length == 1 and (.[0] | .completed == 3 and .failed == 0 and .total_images == 3 "
                "and .average_pixels_per_image == 1048576 and .request_throughput > 0 and .mean_e2el_ms > 0 "
                'and .Hardware == "H100" and .experiment_metadata.num_gpus == 2 '
                'and .experiment_metadata.server_revision == "0.5.22")\' '
                '"$${BENCHMARK_DIR}"/result_test_sglang_cosmos3_t2i_tp2_*.json',
            ]
        ),
    )
    return job


def smoke_pipeline(cases: list[str], *, weekly: bool = False) -> dict:
    qwen_performance = cases == ["qwen-image-21-perf"]
    wan_performance = cases == ["wan22-i2v-perf"]
    cosmos_performance = cases == ["cosmos3-t2i-perf"]
    if not (qwen_performance or wan_performance or cosmos_performance) and (
        not cases or len(cases) != len(set(cases)) or set(cases) - GPU_COUNTS.keys()
    ):
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
    if wan_performance:
        pipeline["steps"] = [wan_performance_job(benchmark)]
        return pipeline
    if cosmos_performance:
        pipeline["steps"] = [cosmos_performance_job(benchmark)]
        return pipeline
    pinned_diagnostics = not weekly and set(cases) <= {"hunyuanvideo15", "minimax-h3"}
    pipeline["steps"] = [] if pinned_diagnostics else [image]
    if weekly:
        pipeline["steps"].extend(
            [benchmark, weekly_steps["nightly-sglang-omni-qwen3-tts"], qwen_image_performance_job(benchmark)]
        )
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
        if pinned_diagnostics:
            job.pop("depends_on", None)
            job["env"]["COMPARISON_IMAGE_DIGEST"] = QWEN_IMAGE_DIGEST
            container["image"] = container["image"].rsplit(":", 1)[0] + "@" + QWEN_IMAGE_DIGEST
        commands = [
            'test "$$(git rev-parse HEAD)" = "$${BACKEND_COMPARISON_COMMIT}"',
            'export HF_TOKEN="$${VLLM_CI_HF_TOKEN}"',
        ]
        if pinned_diagnostics:
            commands.insert(0, "set -eu")
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
            if case == "hunyuanvideo15" and pinned_diagnostics:
                commands.extend(
                    [
                        "sglang_source_root=\"$$(/opt/sglang/bin/python3 -c 'import sglang; from pathlib import Path; "
                        "print(Path(sglang.__file__).resolve().parent.parent)')\"",
                        'git -C /tmp apply --unsafe-paths --directory="$${sglang_source_root}" --check '
                        '"$$PWD/docker/patches/sglang-diffusion-config-only-guider.patch"',
                        'git -C /tmp apply --unsafe-paths --directory="$${sglang_source_root}" '
                        '"$$PWD/docker/patches/sglang-diffusion-config-only-guider.patch"',
                        'CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 /opt/sglang/bin/python3 -m pytest '
                        "--noconftest -o addopts='' -q tests/dfx/perf/tests/test_sglang_diffusion_loader.py",
                    ]
                )
            commands.append(
                '/opt/sglang/bin/python3 tests/dfx/perf/scripts/run_sglang_diffusion_smoke.py "$${SMOKE_CASE}"'
            )
        job["command"] = "\n".join(commands)
        pipeline["steps"].append(job)
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("cases", nargs="?", help="Comma-separated model keys")
    selection.add_argument("--weekly", action="store_true", help="Run the four validated comparison workloads")
    args = parser.parse_args()
    try:
        cases = ["minicpm-o-perf"] if args.weekly else args.cases.split(",")
        print(json.dumps(smoke_pipeline(cases, weekly=args.weekly)))
    except ValueError as exc:
        parser.error(str(exc))

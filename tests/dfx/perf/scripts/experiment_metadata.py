# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Provenance for backend comparison artifacts; consumers verify trust separately."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def configured_num_gpus(config: dict[str, object]) -> int | None:
    marks = config.get("mark")
    if not isinstance(marks, list):
        return None
    for mark in marks:
        hardware = mark.get("hardware_marks") if isinstance(mark, dict) else None
        if isinstance(hardware, dict):
            count = hardware.get("num_cards")
            if isinstance(count, dict):
                count = count.get("cuda")
            return count if isinstance(count, int) and not isinstance(count, bool) else None
    return None


def resolved_workload(
    params: dict[str, object],
    *,
    num_prompts: int,
    num_warmups: int,
    request_rate: float | str,
    max_concurrency: int | None = None,
) -> dict[str, object]:
    return {
        **{key: value for key, value in params.items() if key != "baseline"},
        "num_prompts": num_prompts,
        "request_rate": request_rate,
        "max_concurrency": max_concurrency,
        "num_warmups": num_warmups,
        "seed": params.get("seed"),
    }


def file_fingerprint(path: str | Path) -> dict[str, str]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _REPO_ROOT / resolved
    return {
        "path": str(resolved.relative_to(_REPO_ROOT)) if resolved.is_relative_to(_REPO_ROOT) else str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def comparison_provenance(
    *, test_config: str, server_config: str | None, server_revision: str | None, server_type: str = "sglang-omni"
) -> dict[str, object]:
    runner_paths = [
        "tests/dfx/conftest.py",
        "tests/dfx/perf/scripts/run_benchmark.py",
        "tests/dfx/perf/scripts/sglang_omni_server.py",
        "tests/dfx/perf/scripts/experiment_metadata.py",
    ]
    if server_type == "sglang-diffusion":
        runner_paths.extend(["vllm_omni/benchmarks/patch/patch.py", "vllm_omni/benchmarks/metrics/metrics.py"])
        if os.environ.get("COMPARISON_VIDEO_PATCH") == "1":
            runner_paths.extend(
                [
                    "tests/dfx/perf/scripts/validate_sglang_videos.py",
                    "vllm_omni/benchmarks/data_modules/random_multi_modal_dataset.py",
                ]
            )
    server_patches = []
    if server_type == "sglang-omni":
        server_patches.append(file_fingerprint("docker/patches/sglang-omni-ignore-eos.patch"))
    elif server_type == "sglang-diffusion" and os.environ.get("COMPARISON_VIDEO_PATCH") == "1":
        server_patches.append(file_fingerprint("docker/patches/sglang-video-image-reference.patch"))
    return {
        "schema_version": 1,
        "runner_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip(),
        "server_type": server_type,
        "server_revision": server_revision,
        "runner_files": [file_fingerprint(path) for path in runner_paths],
        "test_config": file_fingerprint(test_config),
        "server_config": file_fingerprint(server_config) if server_config else None,
        "server_patches": server_patches,
        **({"image_digest": os.environ.get("COMPARISON_IMAGE_DIGEST")} if server_type == "sglang-diffusion" else {}),
        "buildkite": {
            field: os.environ.get(f"BUILDKITE_{field.upper()}")
            for field in ("build_id", "job_id", "step_key", "pipeline_slug", "branch", "source", "commit")
        },
    }

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
MiniMax-H3 T2V stability: OmniServer (diffusion) + ``diffusion_benchmark_serving.py`` / ``v1/videos``.

Configuration: ``tests/dfx/stability/tests/test_minimax_h3.json``.

``--lora-path`` is resolved like e2e FastH3 / Buildkite ``test-merge.yml``:
set ``VLLM_TEST_MINIMAX_H3_FASTH3_LORA`` to the adapter file, or populate the local
HF cache for ``FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dfx.conftest import (
    create_benchmark_indices,
    create_test_parameter_mapping,
    create_unique_server_pytest_params,
    load_configs,
)
from tests.dfx.stability.helpers import _run_one_diffusion_batch, run_stability_benchmark_loop
from tests.e2e.online_serving.minimax_h3._common import FASTH3_LORA

STABILITY_DIR = Path(__file__).resolve().parent.parent
DEPLOY_CONFIGS_DIR = STABILITY_DIR / "deploy"
CONFIG_FILE_PATH = str(STABILITY_DIR / "tests" / "test_minimax_h3.json")
DEFAULT_NUM_PROMPTS_PER_BATCH = 20
STABILITY_SERVER_TIMEOUT_ARGS = ["--stage-init-timeout", "1800", "--init-timeout", "1800"]


def _inject_fasth3_lora_path(configs: list[dict]) -> list[dict]:
    """Fill serve_args.lora-path from VLLM_TEST_MINIMAX_H3_FASTH3_LORA / HF cache."""
    if not FASTH3_LORA:
        return configs
    for config in configs:
        serve_args = config.setdefault("server_params", {}).setdefault("serve_args", {})
        serve_args["lora-path"] = FASTH3_LORA
    return configs


try:
    BENCHMARK_CONFIGS = _inject_fasth3_lora_path(load_configs(CONFIG_FILE_PATH))
except FileNotFoundError:
    BENCHMARK_CONFIGS = []

test_params = create_unique_server_pytest_params(BENCHMARK_CONFIGS, DEPLOY_CONFIGS_DIR) if BENCHMARK_CONFIGS else []
server_to_benchmark_mapping = create_test_parameter_mapping(BENCHMARK_CONFIGS) if BENCHMARK_CONFIGS else {}
benchmark_indices = create_benchmark_indices(BENCHMARK_CONFIGS, server_to_benchmark_mapping)


# Pytest marks (hardware, local_model, slow, diffusion) come from test_minimax_h3.json.
@pytest.mark.skipif(
    FASTH3_LORA is None,
    reason="set VLLM_TEST_MINIMAX_H3_FASTH3_LORA or populate the local HF cache",
)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@pytest.mark.parametrize("stability_benchmark_params", benchmark_indices, indirect=True)
def test_stability_minimax_h3(omni_server, stability_benchmark_params):
    test_name = stability_benchmark_params["test_name"]
    params = stability_benchmark_params["params"]
    duration_sec = params.get("duration_sec", 300)
    num_prompts_per_batch = params.get("num_prompts_per_batch", DEFAULT_NUM_PROMPTS_PER_BATCH)
    request_rate = params.get("request_rate")
    max_concurrency = params.get("max_concurrency")

    bench_params = {
        k: v
        for k, v in params.items()
        if k not in ("duration_sec", "request_rate", "max_concurrency", "num_prompts_per_batch")
    }

    result = run_stability_benchmark_loop(
        host=omni_server.host,
        port=omni_server.port,
        model=omni_server.model,
        duration_sec=duration_sec,
        params=bench_params,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        result_dir=str(STABILITY_DIR),
        num_prompts_per_batch=num_prompts_per_batch,
        run_one_batch=_run_one_diffusion_batch,
    )

    assert result.get("failed", 0) == 0, f"[{test_name}] Failed requests detected: {result.get('errors', [])}"
    assert result.get("completed", 0) > 0, f"[{test_name}] No requests completed"

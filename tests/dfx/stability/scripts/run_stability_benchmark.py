# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unified vLLM-Omni stability benchmark runner.

Run one model's stability config (see ``tests/dfx/stability/tests/*.json``)::

    pytest -s -v tests/dfx/stability/scripts/run_stability_benchmark.py \
        --test-config-file tests/dfx/stability/tests/test_hunyuan_image.json

A positional ``*.json`` is accepted either way::

    pytest -s -v tests/dfx/stability/scripts/run_stability_benchmark.py \
        tests/dfx/stability/tests/test_hunyuan_image.json

Or load every enabled config under ``tests/dfx/stability/tests/`` and filter with
``-m`` (backend/family markers come from each JSON ``mark`` field)::

    pytest -s -v tests/dfx/stability/scripts/run_stability_benchmark.py -m diffusion

The benchmark client is selected per ``benchmark_params`` entry:

- ``dataset: lingbot_world_realtime`` → ``benchmark_lingbot_world_realtime.py``
  (LingBot-World realtime streaming; add a new key to ``_BATCH_RUNNERS`` for a
  new protocol);
- otherwise ``dataset`` → ``diffusion_benchmark_serving.py`` (diffusion image/video);
- ``dataset_name`` → ``vllm bench serve --omni`` (omni chat / TTS).

``duration_sec``, ``num_prompts_per_batch``, ``request_rate`` / ``max_concurrency``
and the rest of the traffic shape come from the JSON config.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from tests.dfx.conftest import (
    create_paired_omni_benchmark_pytest_params,
    create_test_parameter_mapping,
    load_benchmark_configs,
)
from tests.dfx.stability.helpers import (
    RunOneBatchFn,
    _run_one_diffusion_batch,
    _run_one_lingbot_world_batch,
    _run_one_vllm_bench_batch,
    run_stability_benchmark_loop,
)

STABILITY_DIR = Path(__file__).resolve().parent.parent
DEPLOY_CONFIGS_DIR = STABILITY_DIR / "deploy"
STABILITY_TESTS_DIR = STABILITY_DIR / "tests"

DEFAULT_NUM_PROMPTS_PER_BATCH = 20
DEFAULT_DURATION_SEC = 300

_MINIMAX_H3_TEST_PREFIX = "test_minimax_h3"
_FASTH3_LORA_SKIP_REASON = "set VLLM_TEST_MINIMAX_H3_FASTH3_LORA or populate the local HF cache"

# Protocol → batch runner dispatch. ``dataset`` selects an explicit protocol; a
# new streaming protocol only needs a new key here plus a helper under
# ``tests/dfx/stability/helpers.py``.
_BATCH_RUNNERS: dict[str, RunOneBatchFn] = {
    "lingbot_world_realtime": _run_one_lingbot_world_batch,
}


def _get_config_file_from_argv() -> str | None:
    """Resolve the config path from ``--test-config-file`` or a positional ``*.json``.

    pytest does not collect ``.json`` targets, so an unhandled positional config
    would otherwise silently fall back to the bulk load of every case. Multiple
    positional JSON files (an ambiguous request) raise instead.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--test-config-file" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--test-config-file="):
            return arg.split("=", 1)[1]

    positional = [arg for arg in sys.argv[1:] if not arg.startswith("-") and arg.endswith(".json")]
    if len(positional) > 1:
        raise ValueError(
            f"Expected a single positional *.json config, got {positional}. "
            "Pass one config via --test-config-file instead."
        )
    return positional[0] if positional else None


def _is_minimax_h3_config(config: dict) -> bool:
    return str(config.get("test_name", "")).startswith(_MINIMAX_H3_TEST_PREFIX)


def _resolve_fasth3_lora(configs: list[dict]) -> str | None:
    """Resolve FastH3 LoRA only when a MiniMax-H3 case is present.

    Keeps the ``av`` / heavy MiniMax import out of unrelated stability runs.
    """
    if not any(_is_minimax_h3_config(config) for config in configs):
        return None
    try:
        from tests.e2e.online_serving.minimax_h3._common import FASTH3_LORA

        return FASTH3_LORA
    except ImportError:
        return None


def _inject_fasth3_lora_path(configs: list[dict], fasth3_lora: str | None) -> list[dict]:
    """Fill ``serve_args.lora-path`` for MiniMax-H3 cases from the resolved FastH3 LoRA."""
    if not fasth3_lora:
        return configs
    for config in configs:
        if not _is_minimax_h3_config(config):
            continue
        serve_args = config.setdefault("server_params", {}).setdefault("serve_args", {})
        if isinstance(serve_args, dict):
            serve_args["lora-path"] = fasth3_lora
    return configs


def _build_paired_server_params(configs: list[dict], stage_configs_dir: Path) -> list:
    """Paired ``(omni_server, benchmark_params)`` params, with the MiniMax-H3 skip.

    Uses the perf runner's pairing (one collected item per server × parameter
    pair) rather than a cross product, so a stability run collects exactly its
    real cases instead of skipping every mismatched combination.
    """
    paired = create_paired_omni_benchmark_pytest_params(configs, stage_configs_dir)
    if FASTH3_LORA is not None:
        return paired

    result: list = []
    for param in paired:
        server_entry, benchmark_key = param.values
        test_name = benchmark_key[0]
        if str(test_name).startswith(_MINIMAX_H3_TEST_PREFIX):
            param = pytest.param(
                server_entry,
                benchmark_key,
                marks=[*param.marks, pytest.mark.skip(reason=_FASTH3_LORA_SKIP_REASON)],
                id=param.id,
            )
        result.append(param)
    return result


def _select_batch_runner(params: dict[str, Any]) -> RunOneBatchFn:
    dataset = params.get("dataset")
    runner = _BATCH_RUNNERS.get(dataset) if dataset is not None else None
    if runner is not None:
        return runner
    if "dataset" in params and "dataset_name" not in params:
        return _run_one_diffusion_batch
    return _run_one_vllm_bench_batch


_config_file_path = _get_config_file_from_argv()
if _config_file_path is None:
    _raw_configs = load_benchmark_configs(config_dir=STABILITY_TESTS_DIR)
    print(
        f"No config file given: loaded {len(_raw_configs)} stability case(s) from "
        f"{STABILITY_TESTS_DIR}/*.json (filter with -m, e.g. -m diffusion)"
    )
else:
    _raw_configs = load_benchmark_configs(_config_file_path)
    print(f"Config file: loaded {len(_raw_configs)} stability case(s) from {_config_file_path}")

FASTH3_LORA = _resolve_fasth3_lora(_raw_configs)
BENCHMARK_CONFIGS = _inject_fasth3_lora_path(_raw_configs, FASTH3_LORA)

server_to_benchmark_mapping = create_test_parameter_mapping(BENCHMARK_CONFIGS)
paired_benchmark_params = _build_paired_server_params(BENCHMARK_CONFIGS, DEPLOY_CONFIGS_DIR)


# Pytest marks (hardware, local_model, omni/tts/diffusion) come from the JSON mark field.
@pytest.mark.parametrize("omni_server,stability_benchmark_params", paired_benchmark_params, indirect=True)
def test_stability_benchmark(omni_server, stability_benchmark_params):
    test_name = stability_benchmark_params["test_name"]
    params = stability_benchmark_params["params"]
    duration_sec = params.get("duration_sec", DEFAULT_DURATION_SEC)
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
        run_one_batch=_select_batch_runner(params),
    )

    assert result.get("failed", 0) == 0, f"[{test_name}] Failed requests detected: {result.get('errors', [])}"
    assert result.get("completed", 0) > 0, f"[{test_name}] No requests completed"

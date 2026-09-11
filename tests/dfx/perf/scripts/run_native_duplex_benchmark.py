# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native audio KV-append benchmark with explicit data-plane and EOS guards.

Uses the existing native audio scenario driver, not the text/chat fallback.
Server startup is excluded; client connect/close and artifact handling are
included in wall throughput. Real-time input pacing is recorded per case.
"""

import asyncio
import json
import os
import statistics
import time
import warnings
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from tests.dfx.conftest import (
    create_paired_omni_benchmark_pytest_params,
    create_test_parameter_mapping,
    get_runtime_resource_label,
    load_benchmark_configs,
)
from tests.dfx.perf.native_metrics import native_measurement, native_performance_gate
from tests.dfx.perf.scripts.run_benchmark import (
    _PERF_TESTS_DIR,
    DEPLOY_CONFIGS_DIR,
    _get_config_file_from_argv,
    omni_server,  # noqa: F401 -- benchmark fixture, not E2E server fixture
    omni_server_context,  # noqa: F401 -- fixture dependency
)

CONFIG_FILE_PATH = _get_config_file_from_argv()
_configs = load_benchmark_configs(CONFIG_FILE_PATH, config_dir=_PERF_TESTS_DIR)
if CONFIG_FILE_PATH and any(cfg.get("benchmark_runner") != "native-duplex" for cfg in _configs):
    raise ValueError("native duplex runner requires benchmark_runner=native-duplex in every selected case")
_native_configs = [cfg for cfg in _configs if cfg.get("benchmark_runner") == "native-duplex"]
paired_benchmark_params = create_paired_omni_benchmark_pytest_params(_native_configs, DEPLOY_CONFIGS_DIR)
_native_mapping = create_test_parameter_mapping(_native_configs)


@pytest.fixture
def benchmark_params(request):
    test_name, param_index = request.param
    return {"test_name": test_name, "params": _native_mapping[test_name]["benchmark_params"][param_index]}


def _profile_server(server, operation: str, stages: list[int]) -> None:
    """Control profiling only on the benchmark-owned local server."""
    request = Request(
        f"http://{server.host}:{server.port}/{operation}_profile",
        data=json.dumps({"stages": stages}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310 -- benchmark-owned server
        assert json.load(response).get("status") == "SUCCESS"


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "omni_server,benchmark_params", paired_benchmark_params, indirect=["omni_server", "benchmark_params"]
)
def test_native_duplex_performance(omni_server, benchmark_params):  # noqa: F811 -- imported pytest fixture
    """Benchmark native model-driven speech with unchanged scenario assertions."""
    from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
        multi_session_args,
        resolve_ref_audio,
        validated_input_wav,
    )
    from tests.e2e.online_serving.run_minicpmo_realtime_duplex_multi_session import run_multi_session

    params = benchmark_params["params"]
    assert params["workload"] == "native-duplex-audio"
    sessions, turns = int(params["sessions"]), int(params["turns"])
    warmups, repeats = int(params["warmups"]), int(params["repeats"])
    assert sessions > 0 and turns > 0 and warmups >= 1 and repeats >= 2
    name = str(params["name"])
    assert Path(name).name == name and name not in (".", "..")
    output = Path(os.environ.get("BENCHMARK_DIR", "native-duplex-results")) / benchmark_params["test_name"] / name
    output.mkdir(parents=True, exist_ok=False)
    measurements = []
    for index in range(warmups + repeats):
        args = multi_session_args(
            omni_server=omni_server,
            input_wav=validated_input_wav(),
            ref_audio=resolve_ref_audio(),
            output_dir=output / f"trial-{index}",
            response_required=True,
        )
        args.sessions, args.turns = sessions, turns
        args.turn_duration_ms = [args.first_turn_ms] * turns
        args.disconnect_session_index = args.takeover_session_index = None
        args.synchronized_start = True
        args.emit_duplex_control_results = True
        args.realtime_input = bool(params.get("realtime_input", True))
        args.timeout_s = float(params.get("timeout_s", 180))
        profile_stages = params.get("profile_stages", []) if index == warmups else []
        if profile_stages:
            _profile_server(omni_server, "start", profile_stages)
        try:
            started = time.monotonic()
            result = asyncio.run(run_multi_session(args))
            elapsed_s = time.monotonic() - started
        finally:
            if profile_stages:
                _profile_server(omni_server, "stop", profile_stages)
        measured = native_measurement(result, sessions=sessions, turns=turns, elapsed_s=elapsed_s)
        measured.update(trial=index, warmup=index < warmups)
        measured["profiled"] = bool(profile_stages)
        measurements.append(measured)
        # Persist every trial, including warmups; aggregate only measured rows.
        (output / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
        print("NATIVE_DUPLEX_MEASUREMENT " + json.dumps(measured, sort_keys=True), flush=True)
    kept = [row for row in measurements if not row["warmup"]]
    aggregate = {
        "workload": "native-duplex-audio",
        "hardware": get_runtime_resource_label(refresh=True),
        "sessions": sessions,
        "turns": turns,
        "repeats": repeats,
        "realtime_input": args.realtime_input,
        "profiling_run_not_performance_baseline": bool(params.get("profile_stages")),
        "instrumentation_run_not_performance_baseline": bool(os.environ.get("VLLM_OMNI_TEST_NATIVE_INPUT_FAULT")),
        "throughput_scope": "client workload including connect/close/artifacts, excluding server startup",
        "latency_scope": "native scenario per-response timing; raw origins retained in trial artifacts",
        "median_responses_per_s": statistics.median(row["responses_per_s"] for row in kept),
        "median_audio_seconds_per_s": statistics.median(row["audio_seconds_per_s"] for row in kept),
        "median_ttft_ms": statistics.median(row["mean_ttft_ms"] for row in kept),
        "median_ttfp_ms": statistics.median(row["mean_ttfp_ms"] for row in kept),
        "measurements": kept,
    }
    if all(row["stage0_token_metrics_available"] for row in kept):
        aggregate["median_stage0_mean_ttft_ms"] = statistics.median(row["stage0_mean_ttft_ms"] for row in kept)
        itls = [row["stage0_mean_itl_ms"] for row in kept if row["stage0_mean_itl_ms"] is not None]
        aggregate["median_stage0_mean_itl_ms"] = statistics.median(itls) if itls else None
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    aggregate["performance_gate"] = native_performance_gate(aggregate, params.get("baseline", {}))
    if aggregate["performance_gate"] == "unbaselined_hardware":
        warnings.warn(f"No signed native performance baseline for {aggregate['hardware']}; collecting evidence only.")
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print("NATIVE_DUPLEX_AGGREGATE " + json.dumps(aggregate, sort_keys=True), flush=True)

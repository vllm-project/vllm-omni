"""
Stability-specific conftest: when pytest is executed under this directory,
resource monitoring is started before each test and finalized after each test,
so each stability test case gets its own HTML report (one report per case).
No need to wrap pytest with `bash resource_monitor.sh run -- pytest ...`.

Duration-based benchmark helper functions are hosted in ``helpers.py``,
while this file focuses on pytest fixtures and setup/teardown.
"""

from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from tests.dfx.conftest import get_benchmark_params_for_server
from tests.dfx.stability.helpers import (
    finalize_resource_monitor,
    report_latest_gpu_samples,
    start_resource_monitor,
    wait_for_run_dir,
)
from tests.helpers.runtime import OmniServer

DEFAULT_STABILITY_SERVER_TIMEOUT_ARGS = ["--stage-init-timeout", "600", "--init-timeout", "900"]

_omni_server_lock = threading.Lock()


@pytest.fixture(scope="module")
def omni_server(request: pytest.FixtureRequest):
    """Start OmniServer for stability tests, with per-module timeout override."""
    timeout_args = getattr(request.module, "STABILITY_SERVER_TIMEOUT_ARGS", DEFAULT_STABILITY_SERVER_TIMEOUT_ARGS)
    with _omni_server_lock:
        # Same 5-tuple and CLI composition as ``tests/dfx/perf/scripts/run_benchmark.py`` on main;
        # ``serve_args`` from JSON are folded into ``extra_cli_args`` inside
        # ``create_unique_server_params``.
        test_name, model, deploy_path, stage_overrides, extra_cli_args = request.param

        print(f"Starting OmniServer with test: {test_name}, model: {model}")
        server_args = list(timeout_args)
        if deploy_path:
            server_args = ["--deploy-config", deploy_path] + server_args
        if stage_overrides:
            server_args = ["--stage-overrides", stage_overrides] + server_args
        if extra_cli_args:
            server_args = list(extra_cli_args) + server_args
        with OmniServer(model, server_args) as server:
            server.test_name = test_name
            print("OmniServer started successfully")
            yield server
            print("OmniServer stopping...")
        print("OmniServer stopped")


@pytest.fixture
def stability_benchmark_params(request: pytest.FixtureRequest, omni_server):
    test_name, param_index = request.param
    if test_name != omni_server.test_name:
        pytest.skip(f"Skipping parameter for {test_name} - current server is {omni_server.test_name}")

    server_to_benchmark_mapping = getattr(request.module, "server_to_benchmark_mapping", None)
    if server_to_benchmark_mapping is None:
        raise ValueError("server_to_benchmark_mapping must be defined in the test module")

    all_params = get_benchmark_params_for_server(test_name, server_to_benchmark_mapping)
    if not all_params:
        raise ValueError(f"No benchmark parameters found for test: {test_name}")
    if param_index >= len(all_params):
        raise ValueError(f"No benchmark parameters found for index {param_index} in test: {test_name}")

    current = param_index + 1
    total = len(all_params)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")
    return {"test_name": test_name, "params": all_params[param_index]}


@pytest.fixture(autouse=True)
def stability_resource_monitor_per_test(request: pytest.FixtureRequest):
    """
    For each test under this directory: start GPU monitor before the test,
    then finalize after the test so this case gets its own report.html.
    """
    proc = start_resource_monitor()
    stop_event = threading.Event()
    reporter: threading.Thread | None = None

    if proc is not None:
        reporter = threading.Thread(
            target=report_latest_gpu_samples,
            args=(stop_event,),
            name="stability-resource-monitor-reporter",
            daemon=True,
        )
        reporter.start()
        run_dir = wait_for_run_dir(timeout_sec=5)
        node_name = request.node.name
        if run_dir is not None:
            sys.stderr.write(f"[Stability] Resource monitor started for test: {node_name} | run dir: {run_dir}\n")
        else:
            sys.stderr.write(f"[Stability] Resource monitor started for test: {node_name} (run dir not ready yet)\n")

    yield

    # Teardown: stop reporter, stop monitor, finalize → one HTML per test
    if proc is not None:
        stop_event.set()
        if reporter is not None and reporter.is_alive():
            reporter.join(timeout=2)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        bundle_dir = finalize_resource_monitor()
        node_name = request.node.name
        if bundle_dir:
            sys.stderr.write(f"[Stability] Report for test «{node_name}»: {bundle_dir}/report.html\n")
        else:
            sys.stderr.write(f"[Stability] Finalize skipped or failed for test «{node_name}»\n")

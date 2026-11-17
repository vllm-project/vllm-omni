# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Process-kill fault steps. The model supplies the probe and the worker pattern."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import pytest

from tests.dfx.conftest import assert_fault_exception
from tests.dfx.reliability.helpers import (
    INFLIGHT_INJECTION_REQUEST_COUNT,
    INFLIGHT_INJECTION_REQUEST_RATE,
    FaultModel,
    FaultScenario,
    assert_api_still_listening,
    assert_no_server_tree_process_residual_and_gpu_release,
    assert_post_fault_api_unreachable,
    assert_post_fault_health_503,
    make_process_kill_fault_injector,
    make_server_root_kill_fault_injector,
    make_server_tree_kill_fault_injector,
    run_fault_injection_with_rate_load,
    sigterm_serve_root_and_assert_cleanup,
    worker_residual_timeout_after_kill_signal,
)


def _kill_worker(model: FaultModel, server: Any, signal_name: str) -> None:
    make_process_kill_fault_injector(
        grep_patterns=model.worker_pattern,
        signal_name=signal_name,
        limit=1,
        post_kill_wait_seconds=2.0,
    )(server)


def _response_status(result: Any) -> int | None:
    if result is None:
        return None
    response = result[0] if isinstance(result, (list, tuple)) else result
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _assert_concurrent_all_fail(model: FaultModel, client: Any, server: Any) -> None:
    start = time.monotonic()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=model.concurrent_workers)
    try:
        futures = [executor.submit(model.submit_concurrent, client, server) for _ in range(model.concurrent_workers)]
        done, pending = concurrent.futures.wait(
            futures,
            timeout=model.concurrent_timeout_sec,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        elapsed = time.monotonic() - start
        assert not pending, f"[process_kill concurrent] some fault-time requests hung: pending={len(pending)}"
        assert elapsed < model.concurrent_timeout_sec, (
            f"[process_kill concurrent] fault-time request convergence is too slow: {elapsed:.2f}s"
        )
        for future in done:
            try:
                result = future.result()
            except AssertionError:
                raise
            except Exception as exc:
                assert_fault_exception(exc, model.process_kill_error_keywords)
                continue
            status = _response_status(result)
            assert status is not None and status >= 400, (
                f"[process_kill concurrent] expected every request to fail after process-kill, got status={status}"
            )
    finally:
        # ``with`` calls shutdown(wait=True) and would block until each request's
        # own HTTP timeout. Fail on ``pending`` / ``elapsed`` first.
        executor.shutdown(wait=False, cancel_futures=True)


def _assert_health_503(model: FaultModel, client: Any) -> None:
    deadline = time.monotonic() + 20.0
    last_observation = ""
    final_health_status: int | None = None
    while time.monotonic() < deadline:
        try:
            response = client.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0]
            status = response.status_code
            body = response.error_message if response.error_message is not None else response.json_body
            last_observation = f"http={status}, body={str(body)[:200]!r}"
            final_health_status = status
            if status == 503:
                break
        except Exception as exc:  # noqa: BLE001
            last_observation = f"exception={exc!r}"
        time.sleep(0.5)
    if final_health_status != 503 and model.health_miss_skip:
        pytest.skip(model.health_miss_skip)
    assert final_health_status == 503, (
        "[process_kill health] expected /health 503 after fatal fault, "
        f"got status={final_health_status}, last_observation={last_observation}"
    )


def _assert_cleanup(server: Any, *, scenario: str, signal_name: str) -> None:
    assert_no_server_tree_process_residual_and_gpu_release(
        server,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


def _require_signal(signal_name: str | None) -> str:
    if signal_name is None:
        raise RuntimeError("process-kill scenario requires a signal name")
    return signal_name


class ProcessKillWorkerNoLoad(FaultScenario):
    """After a quiet worker kill, one new request returns HTTP 500 before concurrent checks.

    ``assert_fast_fail`` uses ``err_code=500``. A client timeout raises and fails the
    test. The concurrent checker runs only after that probe, then ``/health`` must end at 503.
    """

    signals_attr = "worker_signals"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        signal = _require_signal(signal_name)
        model.warmup(client, server)
        _kill_worker(model, server, signal)
        model.assert_fast_fail(client, server)
        _assert_concurrent_all_fail(model, client, server)
        _assert_health_503(model, client)


class ProcessKillWorkerWithLoad(FaultScenario):
    signals_attr = "worker_signals"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        signal = _require_signal(signal_name)
        scenario = "kill_worker_with_load"
        model.warmup(client, server)
        load_result = run_fault_injection_with_rate_load(
            submit_request=lambda: model.submit_worker_load(client, server),
            inject_fault=lambda: _kill_worker(model, server, signal),
            num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
            request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
            completion_timeout_sec=model.with_load_timeout_sec,
        )
        assert load_result["failure_observed"], (
            f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
        )
        model.assert_fast_fail(client, server)
        assert_post_fault_health_503(client, scenario=scenario)


class ProcessKillServeRootNoLoad(FaultScenario):
    signals_attr = "serve_root_signals"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        signal = _require_signal(signal_name)
        scenario = f"kill_serve_root_no_load_{signal.lower()}"
        model.warmup(client, server)
        make_server_root_kill_fault_injector(signal_name=signal, post_kill_wait_seconds=2.0)(server)
        assert_post_fault_api_unreachable(client, scenario=scenario)
        _assert_cleanup(server, scenario=scenario, signal_name=signal)


class ProcessKillServeRootWithLoad(FaultScenario):
    signals_attr = "serve_root_with_load_signals"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        signal = _require_signal(signal_name)
        scenario = f"kill_serve_root_with_load_{signal.lower()}"
        model.warmup(client, server)
        injector = make_server_root_kill_fault_injector(signal_name=signal, post_kill_wait_seconds=2.0)
        load_result = run_fault_injection_with_rate_load(
            submit_request=lambda: model.submit_serve_load(client, server),
            inject_fault=lambda: injector(server),
            num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
            request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
            completion_timeout_sec=model.with_load_timeout_sec,
        )
        assert load_result["failure_observed"], (
            f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
        )
        assert_post_fault_api_unreachable(client, scenario=scenario)
        _assert_cleanup(server, scenario=scenario, signal_name=signal)


class ProcessKillTreeNoLoad(FaultScenario):
    signals_attr = "tree_signals"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        signal = _require_signal(signal_name)
        scenario = f"kill_serve_tree_no_load_{signal.lower()}"
        model.warmup(client, server)
        make_server_tree_kill_fault_injector(
            signal_name=signal,
            post_kill_wait_seconds=2.0,
            inter_kill_wait_seconds=0.1,
        )(server)
        assert_post_fault_api_unreachable(client, scenario=scenario)
        _assert_cleanup(server, scenario=scenario, signal_name=signal)


class ProcessKillTreeWithLoad(FaultScenario):
    signals_attr = "tree_with_load_signals"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        signal = _require_signal(signal_name)
        scenario = f"kill_serve_tree_with_load_{signal.lower()}"
        model.warmup(client, server)
        injector = make_server_tree_kill_fault_injector(
            signal_name=signal,
            post_kill_wait_seconds=2.0,
            inter_kill_wait_seconds=0.1,
        )
        load_result = run_fault_injection_with_rate_load(
            submit_request=lambda: model.submit_serve_load(client, server),
            inject_fault=lambda: injector(server),
            num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
            request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
            completion_timeout_sec=model.with_load_timeout_sec,
        )
        assert load_result["failure_observed"], (
            f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
        )
        assert_post_fault_api_unreachable(client, scenario=scenario)
        _assert_cleanup(server, scenario=scenario, signal_name=signal)


class PartialStageIsolation(FaultScenario):
    """Kill one stage or one worker. Requests bound to it fail, and one SIGTERM still exits the API."""

    server_set = "lifecycle"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        scenario = "partial_stage_isolation"
        model.isolation_warmup(client, server)
        make_process_kill_fault_injector(
            grep_patterns=model.isolation_pattern,
            signal_name="SIGKILL",
            limit=1,
            post_kill_wait_seconds=2.0,
        )(server)
        model.isolation_after(client, server)
        assert_api_still_listening(client, scenario=scenario)
        sigterm_serve_root_and_assert_cleanup(server, scenario=scenario)

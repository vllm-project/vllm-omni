# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Faults that happen while the server is starting, before the fixture would mark it ready."""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any

import pytest

from tests.dfx.conftest import resolve_oom_device_spec
from tests.dfx.reliability.helpers import (
    LAUNCH_FAILURE_TIMEOUT_SEC,
    FaultModel,
    FaultScenario,
    _capture_server_fault_snapshot,
    _list_server_process_tree,
    _safe_proc_info,
    assert_launch_failure_observable,
    assert_no_server_tree_process_residual_and_gpu_release,
    assert_post_fault_api_unreachable,
    capture_exclusive_fifo_objects,
    client_for_server,
    escalate_live_pids,
    inject_gpu_oom,
    make_server_tree_kill_fault_injector,
    start_server_from_params,
    stop_gpu_oom_hogs,
)


class GpuOomAtLaunch(FaultScenario):
    """Fill GPU memory and require startup to fail observably, then remove pressure.

    Subsequent serial cases start fresh instances and exercise their workloads,
    providing suite-level restart coverage. Running this case alone does not
    verify recovery after removing pressure.
    """

    server_set = "launch"

    def run(self, model: FaultModel, server: Any, client: Any = None, signal_name: str | None = None) -> None:
        del client, signal_name
        params = server
        config = model.hard_oom_config
        device = config.get("device") or resolve_oom_device_spec(config, params.stage_config_path)
        kwargs: dict[str, Any] = {
            "device": device,
            "target_mem_ratio": config["target_mem_ratio"],
            "hold_seconds": config["hold_seconds"],
            "startup_timeout_sec": config["startup_timeout_sec"],
            "strict": config["strict"],
        }
        if "max_free_bytes" in config and config["max_free_bytes"] is not None:
            kwargs["max_free_bytes"] = config["max_free_bytes"]
        handle = inject_gpu_oom(**kwargs)
        try:
            # Sidecars intentionally own most device memory until failed-entry
            # rollback completes. Keep cleanup, but do not wait for their memory
            # to disappear during construction or failed-entry teardown.
            poisoned = start_server_from_params(
                params, startup_timeout=LAUNCH_FAILURE_TIMEOUT_SEC, wait_for_device_memory=False
            )
            try:
                poisoned.__enter__()
            except Exception as exc:
                assert_launch_failure_observable(exc)
            else:
                poisoned.__exit__(None, None, None)
                pytest.fail("server became ready while GPU memory was exhausted")
        finally:
            stop_gpu_oom_hogs(handle)


def _kill_starting_stage(server: Any, pattern: str, finished: threading.Event) -> bool:
    """Kill one matching descendant before startup completes; never target the root."""
    tree = _list_server_process_tree(server)
    for pid in tree[1:]:
        name, cmdline = _safe_proc_info(pid)
        if pattern not in f"{name} {cmdline}":
            continue
        _capture_server_fault_snapshot(server)
        assert not finished.is_set(), "startup finished before stage fault injection"
        sigkill = getattr(signal, "SIGKILL", None)
        if sigkill is None:
            raise RuntimeError("SIGKILL is required for stage fault injection")
        try:
            os.kill(pid, sigkill)
        except ProcessLookupError:
            continue
        print(f"[stage-crash-at-launch] killed stage pid={pid} pattern={pattern} name={name}", flush=True)
        return True
    return False


class PartialCrashAtLaunch(FaultScenario):
    """Kill one stage during startup, then require bounded failure and cleanup.

    The injection is scoped to descendants matching the model's worker pattern.
    Startup failure before injection or a ready server is a failure, not coverage.
    The server harness owns rollback on failed entry. Subsequent selected
    serial cases start fresh instances and check their business workloads.
    This is suite-level recovery coverage only when a later preflight succeeds;
    this case alone does not verify restart recovery.
    """

    server_set = "launch"

    def run(self, model: FaultModel, server: Any, client: Any = None, signal_name: str | None = None) -> None:
        del client, signal_name
        params = server
        starting = start_server_from_params(params, startup_timeout=LAUNCH_FAILURE_TIMEOUT_SEC)
        errors: list[BaseException] = []
        finished = threading.Event()
        ready = threading.Event()

        def _enter() -> None:
            try:
                starting.__enter__()
                ready.set()
            except BaseException as exc:  # noqa: BLE001 - propagate the startup thread's failure below
                errors.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=_enter, daemon=True)
        thread.start()
        injected = False
        try:
            deadline = time.monotonic() + LAUNCH_FAILURE_TIMEOUT_SEC
            while time.monotonic() < deadline and not finished.is_set():
                if _kill_starting_stage(starting, model.worker_pattern, finished):
                    injected = True
                    break
                time.sleep(0.05)
            assert injected, (
                f"no stage fault injected before startup completed: pattern={model.worker_pattern!r}, "
                f"ready={ready.is_set()}, errors={errors!r}"
            )
            thread.join(LAUNCH_FAILURE_TIMEOUT_SEC)
            assert not thread.is_alive(), "startup did not terminate within the post-fault deadline"
            assert not ready.is_set(), "server became ready after a stage was killed during startup"
            assert errors, "startup did not report the injected stage failure"
            assert_launch_failure_observable(errors[0])
        finally:
            # Emergency cleanup cannot turn a startup timeout into a passing case.
            if thread.is_alive() and starting.proc is not None:
                make_server_tree_kill_fault_injector(signal_name="SIGKILL")(starting)
                thread.join(30)
            if ready.is_set():
                starting.__exit__(None, None, None)
            assert not thread.is_alive(), "startup thread survived emergency process-tree cleanup"
        assert_no_server_tree_process_residual_and_gpu_release(
            starting, scenario="stage_crash_at_launch", timeout_sec=60.0
        )


class FullCrashAtLaunch(FaultScenario):
    """Kill the process tree during startup without an extra recovery launch.

    Subsequent selected serial cases start fresh instances and exercise their
    workloads, providing suite-level restart coverage. Running this case alone
    does not verify restart recovery; coverage requires a later case to run
    and successfully complete its business preflight.
    """

    server_set = "launch"

    def run(self, model: FaultModel, server: Any, client: Any = None, signal_name: str | None = None) -> None:
        del client, signal_name
        params = server
        starting = start_server_from_params(params)
        errors: list[BaseException] = []

        def _enter() -> None:
            try:
                starting.__enter__()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=_enter)
        thread.start()
        deadline = time.monotonic() + 90.0
        while starting.proc is None and not errors and time.monotonic() < deadline:
            time.sleep(0.1)
        if errors and starting.proc is None:
            thread.join(30)
            assert_launch_failure_observable(errors[0])
        else:
            assert starting.proc is not None, "server process did not appear within 90s"
            make_server_tree_kill_fault_injector(signal_name="SIGKILL", post_kill_wait_seconds=0.0)(starting)
            thread.join(120)
            if errors:
                assert_launch_failure_observable(errors[0])
            else:
                starting.__exit__(None, None, None)


class ServeRootCrashAtLaunch(FaultScenario):
    """Kill only the root after a worker appears, strictly before readiness.

    Failed-entry rollback must not kill descendants before the assertions:
    this scenario tests their own response to root death. Emergency harness
    teardown runs only after validation (including failures). Later serial
    cases provide fresh-start coverage; this case never restarts the service.
    """

    server_set = "launch"

    def run(self, model: FaultModel, server: Any, client: Any = None, signal_name: str | None = None) -> None:
        del client, signal_name
        params = server
        starting = start_server_from_params(params)
        finished = threading.Event()
        ready = threading.Event()
        errors: list[BaseException] = []

        def start() -> None:
            try:
                # __enter__ automatically kills the tree on startup failure,
                # which would hide orphaned workers after root-only SIGKILL.
                starting._start_server()
                ready.set()
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=start, daemon=True)
        thread.start()
        injected = False
        captured_pids: list[int] = []
        validated = False
        try:
            deadline = time.monotonic() + params.startup_timeout
            while time.monotonic() < deadline and not finished.is_set():
                tree = _list_server_process_tree(starting)
                workers = [pid for pid in tree[1:] if model.worker_pattern in " ".join(_safe_proc_info(pid))]
                if workers:
                    snapshot = _capture_server_fault_snapshot(starting)
                    captured_pids = list(snapshot["tree_pids"])
                    assert not finished.is_set() and not starting._owns_listening_port(), (
                        "server became ready before root startup fault injection"
                    )
                    sigkill = getattr(signal, "SIGKILL", None)
                    assert sigkill is not None, "SIGKILL is required for root crash injection"
                    root_fifos = capture_exclusive_fifo_objects(snapshot["root_pids"][:1], captured_pids)
                    os.kill(snapshot["root_pids"][0], sigkill)
                    injected = True
                    print(f"[serve-root-crash-at-launch] SIGKILL root with workers={workers}", flush=True)
                    break
                time.sleep(0.05)
            assert injected, (
                f"no root crash injected after a worker appeared: ready={ready.is_set()}, errors={errors!r}"
            )
            thread.join(LAUNCH_FAILURE_TIMEOUT_SEC)
            assert not thread.is_alive(), "startup did not fail within the post-root-crash deadline"
            assert not ready.is_set() and errors, "startup did not report failure after root SIGKILL"
            assert_launch_failure_observable(errors[0])
            assert_post_fault_api_unreachable(client_for_server(starting), scenario="serve_root_crash_at_launch")
            assert_no_server_tree_process_residual_and_gpu_release(
                starting, scenario="serve_root_crash_at_launch", timeout_sec=30, allowed_fifo_objects=root_fifos
            )
            validated = True
        finally:
            # Never use this cleanup as evidence that the service recovered its
            # children. All resource assertions above precede harness teardown.
            try:
                if not validated and captured_pids:
                    # Root death can reparent workers: rediscovering its tree
                    # in __exit__ alone cannot reclaim those orphaned PIDs.
                    escalate_live_pids(captured_pids, grace_sec=0.0)
            finally:
                starting.__exit__(None, None, None)
            thread.join(30)
            assert not thread.is_alive(), "startup thread survived emergency cleanup"

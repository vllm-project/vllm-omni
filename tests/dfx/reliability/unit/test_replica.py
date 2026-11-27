# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU reliability coverage: replica."""

from __future__ import annotations

import ast
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import psutil
import pytest

from tests.dfx.reliability.unit._loader import RELIABILITY_ROOT, exec_ast_nodes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def replica():
    # Exercise actual scenario bodies without loading model/platform packages.
    source = (RELIABILITY_ROOT / "scenarios") / "replica.py"
    nodes: list[ast.stmt] = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    module = ModuleType("replica_under_test")
    module.__dict__.update(
        Any=Any,
        FaultModel=Any,
        FaultScenario=object,
        SimpleNamespace=SimpleNamespace,
        threading=threading,
        time=time,
        psutil=psutil,
        LAUNCH_FAILURE_TIMEOUT_SEC=1,
        SIGTERM_CLEANUP_GRACE_SEC=60,
        INFLIGHT_INJECTION_REQUEST_COUNT=6,
        INFLIGHT_INJECTION_REQUEST_RATE=1,
    )
    shared = source.with_name("process_kill.py")
    nodes.extend(
        n
        for n in ast.parse(shared.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_shutdown_with_killed_targets"
    )
    exec_ast_nodes(nodes, source, module.__dict__)
    return module


@pytest.mark.parametrize("post_fault_failure", [False, True])
@pytest.mark.parametrize("failure_observed", [False, True])
def test_replica_load_uses_multimodal_path_and_requires_all_later_requests(
    replica, post_fault_failure, failure_observed
):
    events = []
    model = SimpleNamespace(
        warmup=Mock(side_effect=lambda *args: events.append("preflight")),
        submit_worker_load=Mock(side_effect=lambda *args: events.append("multimodal_load")),
        submit_serve_load=Mock(side_effect=AssertionError("text-only load must not be used")),
        with_load_timeout_sec=300,
    )
    server, client = object(), object()
    replica._kill = Mock(side_effect=lambda *args: events.append("kill"))

    def load(**kwargs):
        kwargs["submit_request"]()
        kwargs["inject_fault"]()
        return {"failure_observed": failure_observed}

    replica.run_fault_injection_with_rate_load = load
    replica._assert_survivor_serves = Mock(
        side_effect=AssertionError("later multimodal request failed") if post_fault_failure else None
    )
    if post_fault_failure:
        with pytest.raises(AssertionError, match="later multimodal"):
            replica.ReplicaCrashWithLoad().run(model, server, client)
    else:
        replica.ReplicaCrashWithLoad().run(model, server, client)
    assert events == ["preflight", "multimodal_load", "kill"]
    model.submit_serve_load.assert_not_called()
    replica._assert_survivor_serves.assert_called_once_with(model, client, server)


@pytest.mark.parametrize("fail_at", [None, 1, 2, 6])
def test_survivor_requests_must_all_succeed(replica, fail_at):
    calls = []

    def warmup(*args):
        calls.append(1)
        if len(calls) == fail_at:
            raise AssertionError("dead replica received a request")

    model = SimpleNamespace(warmup=warmup)
    client = SimpleNamespace(send_health_http_request=Mock(return_value=[SimpleNamespace(status_code=200)]))
    if fail_at is None:
        replica._assert_survivor_serves(model, client, SimpleNamespace())
        assert len(calls) == 6
    else:
        with pytest.raises(AssertionError, match="dead replica"):
            replica._assert_survivor_serves(model, client, SimpleNamespace())
        assert len(calls) == fail_at  # Never retry a failed business request.


def test_health_must_converge_before_business_requests(replica):
    clock = [0.0]
    replica.time = SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    warmup = Mock()
    statuses = iter([503, 503, 200])
    client = SimpleNamespace(
        send_health_http_request=lambda *args, **kwargs: [SimpleNamespace(status_code=next(statuses))]
    )
    replica._assert_survivor_serves(SimpleNamespace(warmup=warmup), client, SimpleNamespace())
    assert clock[0] == 1.0 and warmup.call_count == 6
    client.send_health_http_request = lambda *args, **kwargs: [SimpleNamespace(status_code=503)]
    warmup.reset_mock()
    with pytest.raises(AssertionError, match="converge"):
        replica._assert_survivor_serves(SimpleNamespace(warmup=warmup), client, SimpleNamespace())
    warmup.assert_not_called()


def test_kill_checks_only_bad_replica_resources_and_preserves_sibling(replica):
    survivor = Mock(pid=12)
    survivor.is_running.return_value = True
    survivor.status.return_value = psutil.STATUS_RUNNING
    replica.psutil = SimpleNamespace(Process=lambda pid: survivor, STATUS_ZOMBIE=psutil.STATUS_ZOMBIE)
    replica._capture_server_fault_snapshot = lambda server: {"tree_pids": [10, 11, 12, 13]}
    replica._safe_proc_info = lambda pid: (
        "worker",
        {11: "stage1_replica1", 12: "stage1_replica0", 13: "stage0_replica0"}[pid],
    )
    replica.kill_bad_replica = Mock()
    replica.assert_no_server_tree_process_residual_and_gpu_release = Mock()
    server = SimpleNamespace()
    replica._kill(SimpleNamespace(bad_replica_pattern="stage1_replica1", bad_replica_devices="2"), server)
    checked = replica.assert_no_server_tree_process_residual_and_gpu_release.call_args.args[0]
    assert checked.reliability_fault_snapshot == {"tree_pids": [11], "shm_objects": []}
    assert server.reliability_surviving_replicas == [survivor]
    survivor.is_running.return_value = False
    with pytest.raises(AssertionError, match="sibling replica exited"):
        replica._assert_survivors_alive([survivor])


@pytest.mark.parametrize("outcome", ["failed_start", "degraded_ready", "no_injection", "startup_hang"])
def test_replica_launch_fault_precedes_readiness_and_cleans_up(replica, outcome):
    release = threading.Event()
    abort = threading.Event()
    events = []

    class Server:
        proc = object()
        reliability_fault_snapshot = {"tree_pids": [10, 11]}

        def __enter__(self):
            if outcome == "no_injection":
                events.append("early_failure")
                raise RuntimeError("startup failed before injection")
            assert release.wait(3), "startup thread not released"
            if outcome == "startup_hang":
                assert abort.wait(3), "emergency cleanup did not release startup"
                events.append("rollback")
                raise RuntimeError("terminated during emergency cleanup")
            if outcome == "failed_start":
                events.append("rollback")
                raise RuntimeError("engine failed to start")
            events.append("ready")
            return self

        def __exit__(self, *args):
            events.append("exit")

    starting = Server()

    replica.start_server_from_params = Mock(return_value=starting)
    replica.make_server_tree_kill_fault_injector = lambda **kwargs: lambda server: abort.set()

    def inject(server, pattern, finished):
        assert pattern == "stage1_replica1" and not finished.is_set()
        events.append("inject")
        release.set()
        return True

    replica._kill_starting_stage = inject
    replica.assert_no_server_tree_process_residual_and_gpu_release = Mock(
        side_effect=lambda *a, **k: events.append("cleanup")
    )
    replica.client_for_server = lambda server: server
    replica._assert_survivor_serves = Mock(side_effect=lambda *a: events.append("probe"))
    expected_error = {
        "no_injection": "no startup replica fault",
        "failed_start": "failed degraded startup",
        "startup_hang": "did not finish within deadline",
    }.get(outcome)
    params = SimpleNamespace(startup_timeout=0.1 if outcome == "startup_hang" else 3)
    model = SimpleNamespace(bad_replica_pattern="stage1_replica1")
    if expected_error:
        with pytest.raises(AssertionError, match=expected_error):
            replica.ReplicaCrashAtLaunch().run(model, params)
        replica._assert_survivor_serves.assert_not_called()
        assert "cleanup" in events
        if outcome != "no_injection":
            assert events == ["inject", "rollback", "cleanup"]
    else:
        replica.ReplicaCrashAtLaunch().run(model, params)
        assert events == ["inject", "ready", "probe", "exit", "cleanup"]
    replica.start_server_from_params.assert_called_once_with(params)


@pytest.mark.parametrize("outcome", ["clean", "target_exited", "cleanup_leak", "late_cleanup"])
def test_replica_exit_kills_only_after_sigterm_and_never_recovers_on_timeout(replica, outcome):
    events: list[str] = []
    clock = [0.0]
    replica.time = SimpleNamespace(monotonic=lambda: clock[0])
    target = Mock(pid=11)
    sibling = Mock(pid=12)
    for process in (target, sibling):
        process.is_running.return_value = True
        process.status.return_value = psutil.STATUS_RUNNING

    def kill():
        assert events[-1] == "sigterm"
        events.append("kill_replica")
        clock[0] += 0.5

    target.kill.side_effect = kill
    replica.psutil = SimpleNamespace(
        Process=lambda pid: {11: target, 12: sibling}[pid],
        STATUS_ZOMBIE=psutil.STATUS_ZOMBIE,
        NoSuchProcess=psutil.NoSuchProcess,
    )
    snapshot = {"tree_pids": [10, 11, 12], "shm_objects": [("/dev/shm/test", 1, 2)]}

    def capture(server):
        events.append("snapshot")
        server.reliability_fault_snapshot = snapshot
        return snapshot

    replica._capture_server_fault_snapshot = capture
    replica._safe_proc_info = lambda pid: ("worker", {11: "stage1_replica1", 12: "stage1_replica0"}[pid])
    model = SimpleNamespace(
        bad_replica_pattern="stage1_replica1", warmup=Mock(side_effect=lambda *args: events.append("preflight"))
    )
    server, client = SimpleNamespace(), object()

    killed_fifos = [("/dev/shm/target_fifo", 1, 2)]

    def capture_fifos(target_pids, tree_pids):
        assert target_pids == [11] and tree_pids == [10, 11, 12]
        assert events[-1] == "snapshot" and "sigterm" not in events
        return killed_fifos

    replica.capture_exclusive_fifo_objects = capture_fifos

    def terminate(server):
        events.append("sigterm")
        clock[0] += 0.5
        if outcome == "target_exited":
            target.is_running.return_value = False

    root_signal = Mock(return_value=terminate)
    replica.make_server_root_kill_fault_injector = root_signal

    def cleanup(actual_server, *, scenario, timeout_sec, allowed_fifo_objects):
        assert allowed_fifo_objects == killed_fifos
        assert actual_server is server and actual_server.reliability_fault_snapshot is snapshot
        assert scenario == "replica_crash_at_exit" and timeout_sec == 59.0
        events.append("cleanup")
        if outcome == "cleanup_leak":
            raise AssertionError("resource cleanup timed out")
        if outcome == "late_cleanup":
            clock[0] = 61.0

    replica.assert_no_server_tree_process_residual_and_gpu_release = Mock(side_effect=cleanup)
    replica.assert_post_fault_api_unreachable = Mock(side_effect=lambda *a, **k: events.append("api_down"))
    replica._assert_survivor_serves = Mock(side_effect=AssertionError("must not probe serving during shutdown"))
    replica.start_server_from_params = Mock(side_effect=AssertionError("must not restart"))
    replica.make_server_tree_kill_fault_injector = Mock(side_effect=AssertionError("must not escalate to pass"))
    expected_error = {
        "target_exited": "target replica exited",
        "cleanup_leak": "resource cleanup timed out",
        "late_cleanup": "did not finish within its grace period",
    }.get(outcome)
    if expected_error:
        with pytest.raises(AssertionError, match=expected_error):
            replica.ReplicaCrashAtExit().run(model, server, client)
        replica.assert_post_fault_api_unreachable.assert_not_called()
    else:
        replica.ReplicaCrashAtExit().run(model, server, client)
        assert events == ["preflight", "snapshot", "sigterm", "kill_replica", "cleanup", "api_down"]
    root_signal.assert_called_once_with(signal_name="SIGTERM", post_kill_wait_seconds=0.0)
    sibling.kill.assert_not_called()
    replica.start_server_from_params.assert_not_called()
    replica._assert_survivor_serves.assert_not_called()
    replica.make_server_tree_kill_fault_injector.assert_not_called()

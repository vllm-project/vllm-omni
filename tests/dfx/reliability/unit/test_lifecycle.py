# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU reliability coverage: lifecycle."""

from __future__ import annotations

import ast
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.dfx.reliability.helpers import FaultModel
from tests.dfx.reliability.unit._loader import RELIABILITY_ROOT, exec_ast_nodes, load_scenario

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

launch = load_scenario("launch")
process_kill = load_scenario("process_kill")


class _LaunchModel(FaultModel):
    worker_pattern = "stage0"


@pytest.mark.parametrize("finished", [False, True])
def test_launch_stage_injection_excludes_root_and_checks_readiness(monkeypatch, finished):
    monkeypatch.setattr(launch, "signal", SimpleNamespace(SIGKILL=9))
    events: list[str | tuple[int, int]] = []
    server = object()
    done = threading.Event()
    if finished:
        done.set()
    monkeypatch.setattr(launch, "_list_server_process_tree", lambda _: [10, 11, 12])
    monkeypatch.setattr(launch, "_safe_proc_info", lambda _: ("StageEngineCoreProc_stage0", ""))
    monkeypatch.setattr(launch, "_capture_server_fault_snapshot", lambda _: events.append("snapshot"))
    monkeypatch.setattr(launch.os, "kill", lambda pid, sig: events.append((pid, sig)))
    if finished:
        with pytest.raises(AssertionError, match="startup finished"):
            launch._kill_starting_stage(server, "stage0", done)
        assert events == ["snapshot"]
    else:
        assert launch._kill_starting_stage(server, "stage0", done)
        assert events == ["snapshot", (11, 9)]


def test_launch_stage_injection_never_falls_back_to_unrelated_processes(monkeypatch):
    monkeypatch.setattr(launch, "_list_server_process_tree", lambda _: [10, 11])
    monkeypatch.setattr(launch, "_safe_proc_info", lambda _: ("unrelated", ""))

    def unexpected_kill(*args):
        pytest.fail("no matching descendant must not cause a kill")

    monkeypatch.setattr(launch.os, "kill", unexpected_kill)
    assert not launch._kill_starting_stage(object(), "stage0", threading.Event())


@pytest.mark.parametrize(
    ("outcome", "expected_error"),
    [
        ("failure", None),
        ("early_failure", "no stage fault injected"),
        ("ready_before_kill", "no stage fault injected"),
        ("ready_after_kill", "server became ready"),
        ("missing_stage", "no stage fault injected"),
        ("startup_hang", "startup did not terminate"),
    ],
)
def test_stage_crash_launch_requires_injection_failure_and_cleanup(monkeypatch, outcome, expected_error):
    events = []
    release = threading.Event()

    class Server:
        proc = object()

        def __enter__(self):
            if outcome == "ready_before_kill":
                return self
            if outcome != "early_failure":
                assert release.wait(5), "test startup thread was not released"
            if outcome == "ready_after_kill":
                return self
            # Real OmniServer.__enter__ rolls back before re-raising.
            events.append("rollback")
            raise RuntimeError("server exited during startup")

        def __exit__(self, *args):
            events.append("exit")

    starting = Server()
    monkeypatch.setattr(launch, "LAUNCH_FAILURE_TIMEOUT_SEC", 0.1)
    factory = Mock(return_value=starting)
    monkeypatch.setattr(launch, "start_server_from_params", factory)

    def inject(*args):
        if outcome in {"early_failure", "ready_before_kill"}:
            args[2].wait(1)
            return False
        if outcome == "missing_stage":
            return False
        events.append("stage_killed")
        if outcome != "startup_hang":
            release.set()
        return True

    monkeypatch.setattr(launch, "_kill_starting_stage", inject)
    monkeypatch.setattr(launch, "make_server_tree_kill_fault_injector", lambda **kwargs: lambda _: release.set())
    monkeypatch.setattr(
        launch, "assert_no_server_tree_process_residual_and_gpu_release", lambda *args, **kwargs: events.append("clean")
    )
    if expected_error is not None:
        with pytest.raises(AssertionError, match=expected_error):
            launch.PartialCrashAtLaunch().run(_LaunchModel(), object())
    else:
        launch.PartialCrashAtLaunch().run(_LaunchModel(), object())
        assert events == ["stage_killed", "rollback", "clean"]

    assert factory.call_count == 1


def test_stage_crash_launch_cleanup_failure_propagates(monkeypatch):
    release = threading.Event()

    class Server:
        proc = object()

        def __enter__(self):
            assert release.wait(5)
            raise RuntimeError("server exited")

    monkeypatch.setattr(launch, "start_server_from_params", lambda *args, **kwargs: Server())

    def inject(*args):
        release.set()
        return True

    monkeypatch.setattr(launch, "_kill_starting_stage", inject)

    def residual(*args, **kwargs):
        raise AssertionError("shared memory residual")

    monkeypatch.setattr(launch, "assert_no_server_tree_process_residual_and_gpu_release", residual)
    with pytest.raises(AssertionError, match="shared memory residual"):
        launch.PartialCrashAtLaunch().run(_LaunchModel(), object())


@pytest.mark.parametrize(
    ("outcome", "expected_error"),
    [
        ("success", None),
        ("no_worker", "no stage matched"),
        ("already_exited", "target stage exited"),
        ("zombie", "target stage exited"),
        ("disappeared", "target stage disappeared"),
        ("residual", "resource residual"),
        ("late_cleanup", "shutdown did not finish"),
        ("slow_injection", "grace period before cleanup"),
    ],
)
def test_stage_exit_requires_shutdown_before_kill_and_no_timeout_escalation(monkeypatch, outcome, expected_error):
    events = []
    clock = [0.0]

    class Model(FaultModel):
        isolation_pattern = "stage1"

        def warmup(self, client, server) -> None:
            events.append("warmup")

    model = Model()
    server = object()
    client = object()
    monkeypatch.setattr(process_kill, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(process_kill, "SIGTERM_CLEANUP_GRACE_SEC", 60.0)
    monkeypatch.setattr(process_kill, "_list_server_process_tree", lambda _: [10, 11])
    # Even if the root matches, it must never be selected as the stage target.
    monkeypatch.setattr(
        process_kill, "_safe_proc_info", lambda pid: ("other" if outcome == "no_worker" and pid == 11 else "stage1", "")
    )

    class Worker:
        pid = 11

        def is_running(self):
            return outcome != "already_exited"

        def status(self):
            return process_kill.psutil.STATUS_ZOMBIE if outcome == "zombie" else "running"

        def kill(self):
            assert events[-1] == "SIGTERM"
            if outcome == "disappeared":
                raise process_kill.psutil.NoSuchProcess(11)
            events.append("stage_SIGKILL")
            if outcome == "slow_injection":
                clock[0] = 61.0

    def process(pid):
        assert pid == 11
        return Worker()

    monkeypatch.setattr(process_kill.psutil, "Process", process)

    killed_fifos = [("/dev/shm/target_fifo", 1, 2)]

    def capture_fifos(target_pids, tree_pids):
        assert target_pids == [11] and tree_pids == [10, 11]
        assert events == ["warmup"]
        return killed_fifos

    monkeypatch.setattr(process_kill, "capture_exclusive_fifo_objects", capture_fifos)

    def root_injector(**kwargs):
        assert kwargs == {"signal_name": "SIGTERM", "post_kill_wait_seconds": 0.0}
        return lambda _: events.append("SIGTERM")

    monkeypatch.setattr(process_kill, "make_server_root_kill_fault_injector", root_injector)

    def cleanup(actual_server, *, scenario, timeout_sec, allowed_fifo_objects):
        assert allowed_fifo_objects == killed_fifos
        assert actual_server is server
        assert events[-1] == "stage_SIGKILL"
        assert timeout_sec == 60.0
        events.append("cleanup")
        if outcome == "residual":
            raise AssertionError("resource residual")
        clock[0] = 61.0 if outcome == "late_cleanup" else 5.0

    monkeypatch.setattr(process_kill, "assert_no_server_tree_process_residual_and_gpu_release", cleanup)
    monkeypatch.setattr(
        process_kill, "assert_post_fault_api_unreachable", lambda *args, **kwargs: events.append("api_gone")
    )

    def unexpected_signal(*args, **kwargs):
        pytest.fail("must not escalate or kill the tree to pass shutdown")

    monkeypatch.setattr(process_kill, "make_server_tree_kill_fault_injector", unexpected_signal)
    monkeypatch.setattr(process_kill, "make_process_kill_fault_injector", unexpected_signal)
    if expected_error is None:
        process_kill.PartialCrashAtExit().run(model, server, client)
        assert events == ["warmup", "SIGTERM", "stage_SIGKILL", "cleanup", "api_gone"]
    else:
        with pytest.raises((AssertionError, pytest.fail.Exception), match=expected_error):
            process_kill.PartialCrashAtExit().run(model, server, client)


@pytest.fixture
def root_launch():
    path = (RELIABILITY_ROOT / "scenarios") / "launch.py"
    node = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "ServeRootCrashAtLaunch"
    )
    module = ModuleType("root_launch_under_test")
    module.__dict__.update(
        Any=Any,
        FaultModel=Any,
        FaultScenario=object,
        threading=threading,
        time=time,
        LAUNCH_FAILURE_TIMEOUT_SEC=0.02,
        signal=SimpleNamespace(SIGKILL=9),
    )
    exec_ast_nodes([node], path, module.__dict__)
    return module


@pytest.mark.parametrize(
    "outcome,expected_error",
    [
        ("clean", None),
        ("no_worker", "no root crash injected"),
        ("early_failure", "no root crash injected"),
        ("ready_before", "no root crash injected"),
        ("ready_after", "did not report failure"),
        ("startup_hang", "post-root-crash deadline"),
        ("resource_leak", "orphan resource residual"),
    ],
)
def test_root_startup_crash_requires_worker_and_checks_resources_before_teardown(root_launch, outcome, expected_error):
    events: list[str] = []
    release = threading.Event()

    class Server:
        proc = object()

        def __enter__(self):
            pytest.fail("__enter__ rollback would hide orphan processes")

        def _start_server(self):
            if outcome == "early_failure":
                raise RuntimeError("startup failed without injection")
            if outcome == "ready_before":
                return
            assert release.wait(3), "startup thread was not released"
            if outcome != "ready_after":
                raise RuntimeError("server exited after root died")

        def _owns_listening_port(self):
            return False

        def __exit__(self, *args):
            events.append("teardown")
            release.set()

    server = Server()
    root_launch.start_server_from_params = Mock(return_value=server)
    root_launch._list_server_process_tree = lambda server: [10, 12] if outcome == "no_worker" else [10, 11, 12]
    root_launch._safe_proc_info = lambda pid: ("stage0_worker" if pid == 11 else "resource_tracker", "")

    def snapshot(server):
        events.append("snapshot")
        return {"root_pids": [10], "tree_pids": [10, 11, 12]}

    root_launch._capture_server_fault_snapshot = snapshot
    root_launch.capture_exclusive_fifo_objects = Mock(return_value=[("root_fifo", 1, 2)])
    root_launch.escalate_live_pids = Mock()

    def kill(pid, sig):
        assert pid == 10 and sig == 9
        assert events == ["snapshot"]
        events.append("root_killed")
        if outcome != "startup_hang":
            release.set()

    root_launch.os = SimpleNamespace(kill=kill)
    root_launch.assert_launch_failure_observable = Mock(side_effect=lambda exc: events.append("failure_seen"))
    root_launch.client_for_server = lambda server: server
    root_launch.assert_post_fault_api_unreachable = Mock(side_effect=lambda *a, **k: events.append("api_down"))

    def cleanup(server, **kwargs):
        assert events[-1] == "api_down" and "teardown" not in events
        assert kwargs == {
            "scenario": "serve_root_crash_at_launch",
            "timeout_sec": 30,
            "allowed_fifo_objects": [("root_fifo", 1, 2)],
        }
        events.append("resources_checked")
        if outcome == "resource_leak":
            raise AssertionError("orphan resource residual")

    root_launch.assert_no_server_tree_process_residual_and_gpu_release = cleanup
    params = SimpleNamespace(startup_timeout=0.08)
    if expected_error:
        with pytest.raises(AssertionError, match=expected_error):
            root_launch.ServeRootCrashAtLaunch().run(SimpleNamespace(worker_pattern="stage0_worker"), params)
    else:
        root_launch.ServeRootCrashAtLaunch().run(SimpleNamespace(worker_pattern="stage0_worker"), params)
        assert events == ["snapshot", "root_killed", "failure_seen", "api_down", "resources_checked", "teardown"]
    assert events[-1] == "teardown"
    if outcome in {"ready_after", "startup_hang", "resource_leak"}:
        root_launch.escalate_live_pids.assert_called_once_with([10, 11, 12], grace_sec=0.0)
    else:
        root_launch.escalate_live_pids.assert_not_called()
    root_launch.start_server_from_params.assert_called_once_with(params)


ROOT = RELIABILITY_ROOT


@pytest.fixture
def shutdown():
    path = ROOT / "scenarios/process_kill.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in {"GracefulShutdown", "ReplicaGracefulShutdown", "_require_signal"}
    ]
    module = ModuleType("shutdown_under_test")
    module.__dict__.update(Any=Any, FaultModel=Any, FaultScenario=object, SIGTERM_CLEANUP_GRACE_SEC=60)
    exec_ast_nodes(nodes, path, module.__dict__)
    return module


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("outcome", ["clean", "leak", "slow_signal", "late_cleanup"])
def test_normal_shutdown_has_one_sigterm_and_a_shared_cleanup_deadline(shutdown, multi, outcome):
    events: list[str] = []
    clock = [0.0]
    shutdown.time = SimpleNamespace(monotonic=lambda: clock[0])
    model = SimpleNamespace(warmup=Mock(side_effect=lambda *args: events.append("preflight")))
    server, client = object(), object()

    def terminate(actual):
        assert actual is server and events == ["preflight"]
        events.append("sigterm")
        clock[0] = 61.0 if outcome == "slow_signal" else 2.0

    shutdown.make_server_root_kill_fault_injector = Mock(return_value=terminate)

    def cleanup(actual, *, scenario, timeout_sec):
        assert actual is server and scenario == "graceful_shutdown" and timeout_sec == 58.0
        events.append("cleanup")
        if outcome == "leak":
            raise AssertionError("resource residual")
        clock[0] = 61.0 if outcome == "late_cleanup" else 5.0

    shutdown.assert_no_server_tree_process_residual_and_gpu_release = Mock(side_effect=cleanup)
    shutdown.assert_post_fault_api_unreachable = Mock(side_effect=lambda *a, **k: events.append("api_down"))
    scenario = shutdown.ReplicaGracefulShutdown() if multi else shutdown.GracefulShutdown()
    assert scenario.server_set == ("replica" if multi else "lifecycle")
    if outcome == "clean":
        scenario.run(model, server, client)
        assert events == ["preflight", "sigterm", "cleanup", "api_down"]
    else:
        with pytest.raises(AssertionError, match="resource residual|grace period"):
            scenario.run(model, server, client)
        shutdown.assert_post_fault_api_unreachable.assert_not_called()
    shutdown.make_server_root_kill_fault_injector.assert_called_once_with(
        signal_name="SIGTERM", post_kill_wait_seconds=0.0
    )


@pytest.mark.parametrize("signal", [None, "SIGTERM"])
def test_crash_rejects_non_sigkill(shutdown, signal):
    with pytest.raises((ValueError, RuntimeError)):
        shutdown._require_signal(signal)
    assert shutdown._require_signal("SIGKILL") == "SIGKILL"


def test_all_models_separate_graceful_shutdown_from_crash_signals():
    tree = ast.parse((ROOT / "helpers.py").read_text(encoding="utf-8"))
    signals = next(
        n
        for n in tree.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "CRASH_SIGNALS"
    )
    assert signals.value is not None
    assert ast.literal_eval(signals.value) == (("SIGKILL", None),)
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FaultModel")
    for statement in model.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id.endswith("_signals")
        ):
            assert isinstance(statement.value, ast.Name) and statement.value.id == "CRASH_SIGNALS"
    for filename in (
        "test_qwen3_omni.py",
        "test_minicpmo_4_5.py",
        "test_wan22.py",
        "test_hunyuan_image.py",
        "test_qwen3_omni_replica.py",
        "test_minicpmo_4_5_replica.py",
    ):
        source = (ROOT / filename).read_text(encoding="utf-8")
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        assert any(n.name == "test_graceful_shutdown" for n in functions), filename
    qwen = ast.parse((ROOT / "test_qwen3_omni.py").read_text(encoding="utf-8"))
    root_signals = next(
        n
        for n in qwen.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "QWEN_SERVE_ROOT_SIGNALS"
    )
    assert root_signals.value is not None
    assert ast.literal_eval(root_signals.value) == (("SIGKILL", "https://github.com/vllm-project/vllm/issues/43060"),)

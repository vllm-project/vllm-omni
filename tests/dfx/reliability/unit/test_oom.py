# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU reliability coverage: oom."""

from __future__ import annotations

import ast
import base64
import errno
import json
import sys
import time
from collections.abc import Callable, Iterable
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tests.dfx.reliability.unit._loader import RELIABILITY_ROOT, exec_ast_nodes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ListBackedStderrCapture:
    """Small fake of OmniServer's future-only stderr evidence capture."""

    def __init__(self, server: SimpleNamespace, markers: Iterable[str]) -> None:
        self._server = server
        self._markers = tuple(marker.lower() for marker in markers)
        self._start = len(server._server_error_lines)

    def __enter__(self) -> _ListBackedStderrCapture:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def wait(self, timeout: float = 2.0) -> str:
        del timeout
        for line in self._server._server_error_lines[self._start :]:
            if any(marker in line.lower() for marker in self._markers):
                return line
        return ""


def _add_list_backed_stderr_capture(server: SimpleNamespace) -> None:
    server.capture_stderr_evidence = lambda markers: _ListBackedStderrCapture(server, markers)


@pytest.fixture
def qwen_oom():
    # Execute the real free functions without importing GPU/platform modules or
    # evaluating model parametrization during these CPU-only control-flow tests.
    module = ModuleType("qwen_oom_under_test")
    module.__dict__.update(
        dict(
            errno=errno,
            json=json,
            sys=sys,
            time=time,
            Any=Any,
            Callable=Callable,
            base64=base64,
            FaultModel=Any,
            FaultScenario=object,
            pytest=pytest,
        )
    )
    oom_tree = ast.parse(((RELIABILITY_ROOT / "scenarios") / "oom.py").read_text(encoding="utf-8"))
    for constant_name in ("GPU_OOM_MARKERS", "EXECUTOR_SHUTDOWN_MARKERS"):
        assignment = next(
            n
            for n in oom_tree.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == constant_name for t in n.targets)
        )
        module.__dict__[constant_name] = ast.literal_eval(assignment.value)
    sources = {
        (RELIABILITY_ROOT / "test_hunyuan_image.py"): {"_run_image_oom_at_load"},
        (RELIABILITY_ROOT / "scenarios") / "oom.py": {
            "_assert_multimodal_success",
            "_assert_image_success",
            "_assert_oom_failure_evidence",
            "_looks_like_server_unreachable",
            "_require_oom_pressure",
            "_snapshot_oom_workers",
            "run_oom_under_load",
            "GpuOomUnderLoad",
        },
        (RELIABILITY_ROOT / "helpers.py"): {
            "extract_openai_error_contract_from_bytes",
            "extract_openai_error_contract_from_payload",
        },
    }
    for path, names in sources.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes: list[ast.stmt] = [
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        ]
        assert {node.name for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef))} == names
        exec_ast_nodes(nodes, path, module.__dict__)
    return module


@pytest.mark.parametrize(
    ("outcome", "expected_error"),
    [
        ("recovered", None),
        ("stage_dead", None),
        ("api_dead", None),
        ("driver_oom", None),
        ("accelerator_oom", None),
        ("hip_quoted_oom", None),
        ("allocator_oom", None),
        ("composite_oom", None),
        ("executor_shutdown_no_worker", "without explicit GPU OOM evidence"),
        ("executor_shutdown_unverified_pressure", "pressure was not verified"),
        ("worker_death_generic_error", "without explicit GPU OOM evidence"),
        ("hip_launch_resources", "without explicit GPU OOM evidence"),
        ("no_oom_evidence", "without explicit GPU OOM evidence"),
        ("stale_oom_evidence", "without explicit GPU OOM evidence"),
        ("warmup_failed", "expected successful multimodal"),
        ("request_timeout", "unexpected OOM request failure"),
        ("fault_succeeded", "must fail with 500/503"),
        ("invalid_error", "lacks an OpenAI error object"),
        ("recovery_failed", "no audio output"),
    ],
)
@pytest.mark.parametrize("entrypoint", ["qwen", "minicpm"])
def test_qwen_oom_uses_one_injection_and_stops_before_recovery(qwen_oom, outcome, expected_error, entrypoint):
    events = []
    requests = []
    payload = {"model": "qwen", "modalities": ["text", "audio"], "stream": False}
    preflight = {**payload, "messages": [{"role": "user", "content": "small preflight"}]}
    success: dict[str, Any] = {"choices": [{"message": {"content": "answer", "audio": {"data": "audio"}}}]}
    if entrypoint == "hunyuan":
        payload = {"model": "hunyuan", "size": "1024x1024"}
        preflight = {"model": "hunyuan", "size": "512x512"}
        success = {"data": [{"b64_json": "aW1hZ2U="}]}
        if outcome == "warmup_failed":
            expected_error = "expected successful image"
        elif outcome == "recovery_failed":
            expected_error = "no image output"
    qwen_oom._image_request = lambda server, *, size, steps: {"json": preflight if size == "512x512" else payload}
    qwen_oom.MODEL = SimpleNamespace(
        worker_pattern="model-specific-worker",
        hard_oom_config={
            "target_mem_ratio": 0.95,
            "hold_seconds": 0,
            "startup_timeout_sec": 120,
            "strict": False,
            "max_free_bytes": 256,
        },
    )
    qwen_oom.MODEL.oom_under_load_request = lambda _: payload
    qwen_oom.MODEL.oom_preflight_request = lambda _: preflight
    qwen_oom._mix_probe_payload = lambda _: payload
    qwen_oom._oom_preflight_request = lambda _: preflight
    qwen_oom._deploy_config_path = lambda _: None
    qwen_oom.resolve_oom_device_spec = lambda *args: "0,1"
    server = SimpleNamespace(model="qwen", host="localhost", port=8000, _server_error_lines=[])
    _add_list_backed_stderr_capture(server)
    if outcome == "stale_oom_evidence":
        server._server_error_lines.append("torch.OutOfMemoryError: CUDA out of memory from an earlier request")

    def inject(**kwargs):
        events.append("inject")
        return "handle"

    def stop(handle):
        assert handle == "handle"
        events.append("stop")

    pressure_checks = 0

    def pressure_status(handle):
        nonlocal pressure_checks
        assert handle == "handle"
        pressure_checks += 1
        if outcome == "executor_shutdown_unverified_pressure" and pressure_checks > 1:
            return False, "gpu=0,state=ready,alive=0,ceiling_reached=1"
        return True, "gpu=0,state=ready,alive=1,ceiling_reached=1"

    qwen_oom.gpu_oom_pressure_status = pressure_status
    worker_marker_calls: list[tuple[str, ...]] = []

    def snapshot_server_worker_pids(server, *, worker_markers_extra):
        worker_marker_calls.append(tuple(worker_markers_extra))
        return [101, 102]

    qwen_oom.snapshot_server_worker_pids = snapshot_server_worker_pids
    qwen_oom.wait_for_worker_pid_death = lambda pids, timeout_sec: (
        [101]
        if outcome in {"composite_oom", "executor_shutdown_unverified_pressure", "worker_death_generic_error"}
        else []
    )

    def post(host, port, path, body, *, timeout_sec):
        if entrypoint == "hunyuan":
            assert path == "/v1/images/generations"
        else:
            assert path == "/v1/chat/completions"
            assert body["modalities"] == ["text", "audio"] and body["stream"] is False
        requests.append(body)
        if len(requests) == 1:
            assert body is preflight
            assert events == []
            events.append("warmup")
            return (500 if outcome == "warmup_failed" else 200), json.dumps(success).encode()
        if len(requests) == 2:
            assert body is payload
            assert events == ["warmup", "inject"]
            events.append("fault")
            no_explicit_oom = {
                "no_oom_evidence",
                "stale_oom_evidence",
                "composite_oom",
                "executor_shutdown_no_worker",
                "executor_shutdown_unverified_pressure",
                "worker_death_generic_error",
            }
            if outcome not in no_explicit_oom:
                server._server_error_lines.append(
                    {
                        "driver_oom": "Returning 2 (CUDA_ERROR_OUT_OF_MEMORY) from cuDevicePrimaryCtxRetain",
                        "accelerator_oom": "torch.AcceleratorError: CUDA error: out of memory",
                        "hip_quoted_oom": "Hip error: 'out of memory'(2)",
                        "hip_launch_resources": "hipErrorLaunchOutOfResources",
                        "allocator_oom": (
                            "[rank0]: CUDACachingAllocator.cpp:3934 memory allocation failed with OOM on device 0 "
                            "while trying to allocate 109051904 bytes (free: 3997696, total: 84942979072)."
                        ),
                    }.get(outcome, "torch.OutOfMemoryError: CUDA out of memory")
                )
            if outcome == "request_timeout":
                raise TimeoutError("request timed out")
            if outcome == "fault_succeeded":
                return 200, json.dumps(success).encode()
            if outcome in {
                "composite_oom",
                "executor_shutdown_no_worker",
                "executor_shutdown_unverified_pressure",
            }:
                return 500, b'{"error":{"message":"Executor shut down"}}'
            return 500, b"{}" if outcome == "invalid_error" else b'{"error":{"message":"engine dead"}}'
        assert len(requests) == 3 and events[-1] == "health"
        assert "stop" in events
        events.append("recovery")
        if outcome == "recovery_failed":
            return 200, b'{"choices":[{"message":{"content":"answer"}}]}'
        return 200, json.dumps(success).encode()

    def health(*args, **kwargs):
        assert events[-1] == "stop"
        events.append("health")
        if outcome == "api_dead":
            raise ConnectionRefusedError("connection refused")
        return [SimpleNamespace(status_code=200 if outcome in {"recovered", "recovery_failed"} else 503)]

    qwen_oom.inject_gpu_oom = inject
    qwen_oom.stop_gpu_oom_hogs = stop
    qwen_oom.post_json_raw = post
    client = SimpleNamespace(send_health_http_request=health)

    def run():
        if entrypoint == "minicpm":
            qwen_oom.GpuOomUnderLoad().run(qwen_oom.MODEL, server, client)
        elif entrypoint == "hunyuan":
            qwen_oom._run_image_oom_at_load(server, client)
        else:
            qwen_oom.GpuOomUnderLoad().run(qwen_oom.MODEL, server, client)

    if expected_error is not None:
        with pytest.raises(AssertionError, match=expected_error) as error:
            run()
        if outcome in {
            "hip_launch_resources",
            "no_oom_evidence",
            "stale_oom_evidence",
            "executor_shutdown_no_worker",
            "worker_death_generic_error",
        }:
            assert "status=500" in str(error.value)
            expected_response = "Executor shut down" if outcome == "executor_shutdown_no_worker" else "engine dead"
            assert expected_response in str(error.value)
            assert "stderr_tail=" in str(error.value)
    else:
        run()
    if outcome != "warmup_failed":
        assert worker_marker_calls == [("model-specific-worker",)]
    if outcome == "warmup_failed":
        assert events == ["warmup"]
    else:
        assert events.count("inject") == events.count("stop") == 1
    if outcome in {"stage_dead", "api_dead"}:
        assert len(requests) == 2
    if outcome == "recovered":
        assert events == ["warmup", "inject", "fault", "stop", "health", "recovery"]

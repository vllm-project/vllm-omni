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
from io import BytesIO
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


def test_minicpm_fault_images_are_independent_and_preflight_stays_small(capsys):
    path = RELIABILITY_ROOT / "test_minicpmo_4_5.py"
    runtime = RELIABILITY_ROOT.parents[1] / "helpers" / "runtime.py"
    nodes = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name in {"_system_prompt", "_audio_request", "_oom_request"}
    ]
    images = []

    def image(width, height, *, seed, force_regenerate):
        assert force_regenerate  # The shared dimension cache otherwise repeats the same image.
        images.append((width, height, seed))
        return {"base64": f"independent-image-{seed}"}

    env: dict[str, Any] = {
        "Any": Any,
        "generate_synthetic_image": image,
        "generate_synthetic_audio": lambda *args: {"base64": "audio"},
    }
    message_nodes = [
        n
        for n in ast.parse(runtime.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "dummy_messages_from_mix_data"
    ]
    exec_ast_nodes(message_nodes, runtime, env)
    exec_ast_nodes(nodes, path, env)
    preflight = env["_audio_request"]("model")
    assert images == []
    fault = env["_oom_request"]("model")
    content = fault["messages"][-1]["content"]
    urls = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
    assert 1 < len(urls) <= 64
    assert len(set(urls)) == len(urls) == len(images)
    assert not any(item["type"] == "video_url" for item in content)
    assert any(item["type"] == "audio_url" for item in content)
    assert fault["modalities"] == preflight["modalities"] == ["text", "audio"]
    assert not any(item["type"] == "image_url" for item in preflight["messages"][-1]["content"])
    output = capsys.readouterr().out
    assert f"images={len(urls)}" in output
    assert "independent-image-" not in output


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
@pytest.mark.parametrize("entrypoint", ["qwen", "minicpm", "hunyuan"])
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


def test_qwen_oom_does_not_capture_live_server_output():
    tree = ast.parse((RELIABILITY_ROOT / "test_qwen3_omni.py").read_text(encoding="utf-8"))
    case = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "test_gpu_oom_at_load")
    assert [arg.arg for arg in case.args.args] == ["omni_server_function", "online_client_function"]


@pytest.fixture
def wan_oom():
    module = ModuleType("wan_oom_under_test")
    module.__dict__.update(Any=Any, errno=errno, time=time)
    root = RELIABILITY_ROOT
    for path, names in [
        (root / "test_wan22.py", {"_run_oom_at_load", "_create_and_wait_video_job"}),
        (
            root / "scenarios/oom.py",
            {
                "_assert_oom_failure_evidence",
                "_looks_like_server_unreachable",
                "_require_oom_pressure",
                "_snapshot_oom_workers",
            },
        ),
        (root / "helpers.py", {"extract_openai_error_contract_from_payload"}),
    ]:
        nodes: list[ast.stmt] = [
            n
            for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        exec_ast_nodes(nodes, path, module.__dict__)
    tree = ast.parse((root / "scenarios/oom.py").read_text(encoding="utf-8"))
    for constant_name in ("GPU_OOM_MARKERS", "EXECUTOR_SHUTDOWN_MARKERS"):
        assignment = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == constant_name for t in n.targets)
        )
        module.__dict__[constant_name] = ast.literal_eval(assignment.value)
    return module


@pytest.mark.parametrize(
    "outcome,error",
    [
        ("recovered", None),
        ("stage_dead", None),
        ("api_dead", None),
        ("composite_oom", None),
        ("executor_shutdown_no_worker", "without explicit GPU OOM"),
        ("executor_shutdown_unverified_pressure", "pressure was not verified"),
        ("worker_death_generic_error", "without explicit GPU OOM"),
        ("completed_under_pressure", "fault job must fail"),
        ("no_oom", "without explicit GPU OOM"),
        ("stale_oom", "without explicit GPU OOM"),
        ("poll_timeout", "unexpected OOM job failure"),
        ("missing_id", "missing job id"),
        ("admission_oom", None),
        ("invalid_error", "lacks an error object"),
        ("recovery_failed", "did not complete"),
    ],
)
def test_wan_oom_submits_one_fault_job_and_stops_pressure_before_recovery(wan_oom, outcome, error):
    events: list[str] = []
    server = SimpleNamespace(_server_error_lines=[])
    _add_list_backed_stderr_capture(server)
    config = dict(target_mem_ratio=0.95, hold_seconds=0, startup_timeout_sec=120, strict=False, max_free_bytes=256)

    def warmup(client, request_config):
        assert request_config == dict(
            width=256, height=256, num_frames=5, num_inference_steps=2, reference_size=(256, 256)
        )
        assert events == ["prepare"]
        events.append("preflight")
        if outcome == "stale_oom":
            server._server_error_lines.append("CUDA out of memory from earlier request")

    wan_oom.MODEL = SimpleNamespace(
        hard_oom_config=config,
        worker_pattern="wan-model-worker",
    )
    wan_oom._video_request_config = lambda **kwargs: kwargs
    wan_oom._assert_video_preflight = warmup

    def request(prompt, **kwargs):
        events.append("prepare" if not events else "prepare_recovery")
        return {"data": {"prompt": prompt}}

    wan_oom._i2v_create_request = request
    wan_oom._deploy_config_path = lambda server: None
    wan_oom.resolve_oom_device_spec = lambda *args: "0"

    def inject(**kwargs):
        assert events[-1] == "preflight"
        events.append("inject")
        return "handle"

    wan_oom.inject_gpu_oom = inject
    wan_oom.stop_gpu_oom_hogs = lambda handle: events.append("stop")
    pressure_checks = 0

    def pressure_status(handle):
        nonlocal pressure_checks
        assert handle == "handle"
        pressure_checks += 1
        if outcome == "executor_shutdown_unverified_pressure" and pressure_checks > 1:
            return False, "gpu=0,state=ready,alive=0,ceiling_reached=1"
        return True, "gpu=0,state=ready,alive=1,ceiling_reached=1"

    wan_oom.gpu_oom_pressure_status = pressure_status
    worker_marker_calls: list[tuple[str, ...]] = []

    def snapshot_server_worker_pids(server, *, worker_markers_extra):
        worker_marker_calls.append(tuple(worker_markers_extra))
        return [201]

    wan_oom.snapshot_server_worker_pids = snapshot_server_worker_pids
    wan_oom.wait_for_worker_pid_death = lambda pids, timeout_sec: (
        [201]
        if outcome in {"composite_oom", "executor_shutdown_unverified_pressure", "worker_death_generic_error"}
        else []
    )
    creates = []

    def create(request, **kwargs):
        creates.append(request)
        if len(creates) == 1:
            assert events[-1] == "inject"
            events.append("fault_create")
            no_explicit_oom = {
                "no_oom",
                "stale_oom",
                "composite_oom",
                "executor_shutdown_no_worker",
                "executor_shutdown_unverified_pressure",
                "worker_death_generic_error",
            }
            if outcome not in no_explicit_oom:
                server._server_error_lines.append("memory allocation failed with OOM on device 0")
            if outcome in {"admission_oom", "invalid_error"}:
                return [
                    SimpleNamespace(
                        status_code=500,
                        json_body={} if outcome == "invalid_error" else {"error": {"message": "CUDA out of memory"}},
                    )
                ]
            return [SimpleNamespace(status_code=200, json_body={} if outcome == "missing_id" else {"id": "fault"})]
        assert len(creates) == 2 and "stop" in events and "health" in events
        events.append("recovery_create")
        return [SimpleNamespace(status_code=200, json_body={"id": "recovery"})]

    def poll(client, video_id, **kwargs):
        if video_id == "fault":
            events.append("fault_poll")
            if outcome == "poll_timeout":
                raise TimeoutError("video job timed out")
            failed = outcome != "completed_under_pressure"
            message = (
                "Executor shut down"
                if outcome in {"composite_oom", "executor_shutdown_no_worker", "executor_shutdown_unverified_pressure"}
                else "internal error"
            )
            return SimpleNamespace(
                status_code=500 if failed else 200,
                json_body={"status": "failed" if failed else "completed", "error": {"message": message}},
            )
        assert video_id == "recovery" and events[-1] == "recovery_create"
        events.append("recovery_poll")
        return SimpleNamespace(
            status_code=200, json_body={"status": "failed" if outcome == "recovery_failed" else "completed"}
        )

    def health(*args, **kwargs):
        assert events[-1] == "stop"
        events.append("health")
        if outcome == "api_dead":
            raise ConnectionRefusedError("connection refused")
        return [SimpleNamespace(status_code=503 if outcome == "stage_dead" else 200)]

    wan_oom._await_video_job = poll

    def assert_error(response, *, err_code):
        assert response.status_code in err_code if isinstance(err_code, tuple) else response.status_code == err_code

    wan_oom.assert_http_error = assert_error
    client = SimpleNamespace(
        send_videos_create_http_request=create, send_health_http_request=health, send_video_diffusion_request=warmup
    )
    if error:
        with pytest.raises(AssertionError, match=error):
            wan_oom._run_oom_at_load(server, client)
    else:
        wan_oom._run_oom_at_load(server, client)
    assert worker_marker_calls == [("wan-model-worker",)]
    assert events.count("inject") == events.count("stop") == 1
    assert events.count("fault_create") == 1
    if outcome in {"stage_dead", "api_dead"}:
        assert len(creates) == 1
    if outcome == "recovered":
        assert events.index("stop") < events.index("recovery_create")
        assert events[-1] == "recovery_poll"


@pytest.mark.parametrize("status", ["completed", "failed", "queued", "missing_id"])
def test_preflight_requires_completed_video_job(status):
    path = RELIABILITY_ROOT / "test_wan22.py"
    nodes: list[ast.stmt] = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name in {"_assert_video_preflight", "_create_and_wait_video_job"}
    ]
    events = []

    def create(request, **kwargs):
        assert request["files"]["input_reference"][1].read() == b"image"
        assert kwargs["err_code"] == 200
        events.append("create")
        return [SimpleNamespace(status_code=200, json_body={} if status == "missing_id" else {"id": "job"})]

    def await_job(client, video_id, **kwargs):
        assert video_id == "job"
        events.append("poll")
        return SimpleNamespace(status_code=500 if status == "failed" else 200, json_body={"status": status})

    env: dict[str, Any] = {
        "Any": Any,
        "base64": base64,
        "BytesIO": BytesIO,
        "_await_video_job": await_job,
        "time": time,
    }
    exec_ast_nodes(nodes, path, env)
    client = SimpleNamespace(send_videos_create_http_request=create)
    config = {"image_reference": "data:image/jpeg;base64,aW1hZ2U=", "form_data": {"width": 512}}
    if status == "completed":
        env["_assert_video_preflight"](client, config)
        assert events == ["create", "poll"]
    else:
        with pytest.raises(AssertionError):
            env["_assert_video_preflight"](client, config)


@pytest.mark.parametrize("outcome", ["failed", "direct", "generic", "generic_direct", "completed", "invalid", "queued"])
def test_worker_probe_checks_terminal_failure_and_valid_input(outcome):
    clock = [0.0]
    path = RELIABILITY_ROOT / "test_wan22.py"
    nodes: list[ast.stmt] = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_assert_video_worker_failure", "_create_and_wait_video_job", "_await_video_job"}
    ]
    env: dict[str, Any] = {
        "Any": Any,
        "base64": base64,
        "BytesIO": BytesIO,
        "_VIDEO_FAST_FAIL_HTTP_TIMEOUT_SEC": 30,
        "time": SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + s)),
        "generate_synthetic_image": lambda *args: {"base64": base64.b64encode(b"image").decode()},
        "extract_openai_error_contract_from_payload": lambda body: body.get("error"),
    }
    error = {"message": "No image is provided" if outcome == "invalid" else "Engine is dead"}
    if outcome.startswith("generic"):
        error = {"message": "The server had an error while processing your request."}
    terminal = SimpleNamespace(status_code=500, json_body={"status": "failed", "error": error})
    calls = []

    def create(request, **kwargs):
        assert request["files"]["input_reference"][1].read() == b"image"
        calls.append("create")
        return [
            terminal
            if outcome in {"direct", "generic_direct"}
            else SimpleNamespace(status_code=200, json_body={"id": "job"})
        ]

    def retrieve(request, **kwargs):
        calls.append("poll")
        if outcome in {"queued", "completed"}:
            return [SimpleNamespace(status_code=200, json_body={"status": outcome})]
        return [terminal]

    def assert_error(response, **kwargs):
        assert response.status_code in (500, 503)

    env["assert_http_error"] = assert_error
    exec_ast_nodes(nodes, path, env)
    client = SimpleNamespace(send_videos_create_http_request=create, send_video_retrieve_http_request=retrieve)
    if outcome in {"completed", "invalid", "queued"}:
        with pytest.raises(AssertionError):
            env["_assert_video_worker_failure"](client)
    else:
        assert env["_assert_video_worker_failure"](client) is terminal
    assert calls.count("create") == 1
    assert ("poll" in calls) == (outcome not in {"direct", "generic_direct"})

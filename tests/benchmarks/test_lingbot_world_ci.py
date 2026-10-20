# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CI wiring and failure propagation, without substituting model inference."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, TypedDict
from unittest.mock import MagicMock

import pytest
import yaml

from benchmarks.lingbot_world.regression import RegressionError, check_binary_activity, check_quality
from tests.dfx.lingbot_world.runner import ENGINE_ENVIRONMENT, server_arguments, server_environment

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def smoke_entry(tmp_path, monkeypatch):
    from benchmarks.lingbot_world.workload import ChunkRecord, compute_metrics
    from tests.dfx.lingbot_world import test_smoke as smoke
    from tests.e2e.online_serving import test_lingbot_world_v2_stepwise as lifecycle
    from tests.helpers import runtime

    settings = json.loads((ROOT / "tests/dfx/lingbot_world/configs/cuda_smoke.json").read_text())
    settings.update(output=tmp_path / "output", profile_path=ROOT / "tests/dfx/lingbot_world/configs/cuda_smoke.json")
    settings["output"].mkdir()

    class Calls(TypedDict):
        arguments: list[Namespace]
        cleanup: int
        lifecycle: int
        device_cleanup: int
        corruption: str | None

    calls: Calls = {"arguments": [], "cleanup": 0, "lifecycle": 0, "device_cleanup": 0, "corruption": None}

    class Endpoint(NamedTuple):
        port: int

    @contextmanager
    def server(model, arguments, *, env_dict, startup_timeout):
        try:
            yield Endpoint(12345)
        finally:
            calls["cleanup"] += 1

    async def benchmark(args):
        # Keep CLI parsing and raw metric validation real; inference is the unit boundary.
        calls["arguments"].append(args)
        target_fps = args.target_fps if args.target_fps is not None else args.fps_override or 16
        records = [ChunkRecord(i, (i + 1) * 0.5, 0.5, 1024, 9 if i == 0 else 12) for i in range(args.num_chunks)]
        metrics = compute_metrics(records, fps=target_fps, warmup_chunks=args.warmup_chunks, ttfc_s=0.5)
        chunks = [record.to_dict() for record in records]
        change = calls["corruption"]
        if change == "missing":
            chunks.pop()
        elif change == "reordered":
            chunks[0], chunks[1] = chunks[1], chunks[0]
        elif change == "frames":
            chunks[1]["num_frames"] = 0
        elif change == "empty":
            chunks[1]["byte_length"] = 0
        elif change == "nonfinite":
            chunks[1]["arrival_s"] = float("nan")
        elif change == "policy":
            target_fps += 1
        return {
            "config": {
                "target_fps": target_fps,
                "warmup_chunks": args.warmup_chunks,
                "chunk_slo_ms": args.chunk_slo_ms,
                "playback_buffer_chunks": args.playback_buffer_chunks,
            },
            "sessions": [
                {"session_index": 0, "request_id": "smoke-unit-boundary", "chunks": chunks, "metrics": metrics}
            ],
        }

    def lifecycle_check(server):
        calls["lifecycle"] += 1

    def device_cleanup_check(dies, output):
        assert dies == settings["dies"] and output == settings["output"] / "cleanup.json"
        assert calls["cleanup"] == 1, "device cleanup must be checked after the server context exits"
        calls["device_cleanup"] += 1
        if calls["corruption"] == "device_leak":
            raise ValueError("Device Memory Timeout")

    monkeypatch.setattr(runtime, "OmniServer", server)
    monkeypatch.setattr(smoke, "hardware_contract", lambda _: {"platform": "cuda", "cards": 1})
    image = tmp_path / "image.png"
    monkeypatch.setattr(smoke, "checked_assets", lambda _: (tmp_path / "model", {settings["cases"][0]["image"]: image}))
    monkeypatch.setattr(smoke, "run_benchmark", benchmark)
    monkeypatch.setattr(smoke, "assert_device_cleanup", device_cleanup_check)
    monkeypatch.setattr(
        lifecycle, "test_lingbot_world_v2_stepwise_stop_during_paced_wait_releases_session", lifecycle_check
    )
    return smoke.test_cuda_smoke, settings, calls


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "num_chunks": 5,
            "warmup_chunks": 2,
            "mux_fps": 20,
            "target_fps": 9,
            "first_chunk_timeout_s": 11,
            "chunk_timeout_s": 7,
        },
    ],
)
def test_smoke_executes_the_recorded_profile_and_retains_raw_metrics(smoke_entry, overrides):
    run, settings, calls = smoke_entry
    settings.update(overrides)
    run(settings)
    (args,) = calls["arguments"]
    assert (
        args.num_chunks,
        args.warmup_chunks,
        args.fps_override,
        args.target_fps,
        args.first_chunk_timeout,
        args.chunk_timeout,
    ) == (
        settings["num_chunks"],
        settings["warmup_chunks"],
        settings["mux_fps"],
        settings["target_fps"],
        settings["first_chunk_timeout_s"],
        settings["chunk_timeout_s"],
    )
    assert calls["cleanup"] == calls["lifecycle"] == calls["device_cleanup"] == 1
    metrics = json.loads((settings["output"] / "metrics.json").read_text())
    assert metrics["steady_samples"] == settings["num_chunks"] - settings["warmup_chunks"] - 1


@pytest.mark.parametrize("change", ["missing", "reordered", "frames", "empty", "nonfinite", "policy"])
def test_smoke_rejects_invalid_raw_output_and_releases_server(smoke_entry, change):
    run, settings, calls = smoke_entry
    calls["corruption"] = change
    with pytest.raises(RegressionError):
        run(settings)
    assert calls["cleanup"] == 1 and calls["lifecycle"] == 0
    assert calls["device_cleanup"] == 0
    assert not (settings["output"] / "metrics.json").exists()


def test_smoke_propagates_device_cleanup_failure_after_valid_inference(smoke_entry):
    run, settings, calls = smoke_entry
    calls["corruption"] = "device_leak"
    with pytest.raises(ValueError, match="Device Memory Timeout"):
        run(settings)
    assert calls["cleanup"] == calls["lifecycle"] == calls["device_cleanup"] == 1
    assert (settings["output"] / "metrics.json").exists(), "retain successful inference evidence when cleanup fails"


@pytest.mark.parametrize(
    "platform,shell",
    [("cuda", "/bin/sh"), ("cuda", "/bin/bash"), ("amd-other", "/bin/bash"), ("amd-executor", "/bin/bash")],
)
@pytest.mark.parametrize("status", [0, 1, 2, 134])
def test_cpu_logging_preserves_pytest_exit_and_agent_artifacts(tmp_path: Path, platform: str, shell: str, status: int):
    source = ".buildkite/cuda/test-ready.yml" if platform == "cuda" else ".buildkite/amd/test-amd-ready.yml"
    pipeline = yaml.safe_load((ROOT / source).read_text())
    label = "Simple · Model Executor Test · Shard %N/%t" if platform == "amd-executor" else "Simple · Other Test"
    step = next(step for group in pipeline["steps"] for step in group.get("steps", []) if step.get("label") == label)
    application = tmp_path / "application"
    checkout = tmp_path / "agent-checkout"
    application.mkdir()
    checkout.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pytest_cli = bin_dir / "pytest"
    pytest_cli.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from pathlib import Path\n"
        "print('pytest stdout', flush=True)\n"
        "print('pytest stderr', file=sys.stderr, flush=True)\n"
        "report = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--junitxml='))\n"
        "Path(report).write_text('<testsuite/>')\n"
        f"sys.exit({status})\n"
    )
    pytest_cli.chmod(0o755)
    result = subprocess.run(
        [shell, "-e", "-c", "\n".join(step["commands"]).replace("$$", "$")],
        cwd=application,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "BUILDKITE_BUILD_CHECKOUT_PATH": str(checkout),
            "BUILDKITE_PARALLEL_JOB": "2",
            "BUILDKITE_PARALLEL_JOB_COUNT": "3",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == status, result.stdout + result.stderr
    directory = {"cuda": "cuda-other", "amd-other": "amd-other", "amd-executor": "amd-model-executor-cpu"}[platform]
    output = checkout / "artifacts" / directory
    stem = "pytest-2" if platform == "amd-executor" else "pytest"
    log = (output / (stem + ".log")).read_text()
    assert "pytest stdout" in log and "pytest stderr" in log
    assert ET.parse(output / (stem + ".xml")).getroot().tag == "testsuite"


def test_lifecycle_playhead_blocks_second_admission_before_inference() -> None:
    from benchmarks.lingbot_world.workload import pixel_frames_in_chunk
    from tests.e2e.online_serving import test_lingbot_world_v2_stepwise as lifecycle
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, _StreamingPlayback

    state = _StreamingPlayback(
        buffer_seconds=lifecycle._paced_overrides()["streaming_pacing"]["max_lead_ms"] / 1000,
        next_media_duration=pixel_frames_in_chunk(0) / 16,
    )
    engine = object.__new__(DiffusionEngine)
    engine._cv = threading.Condition()
    engine._streaming_playback = {"paced": state}
    assert not engine._playback_blocked(), "the window must fit the initial chunk"

    state.generated_seconds = state.next_media_duration
    state.next_media_duration = pixel_frames_in_chunk(1) / 16
    assert engine._playback_blocked(), "held playback must block the second chunk before its inference starts"

    state.played_seconds = state.generated_seconds
    assert not engine._playback_blocked(), "reporting playback must release admission"


class _DeviceProperties(NamedTuple):
    uuid: str


class _MountInfo(NamedTuple):
    mountpoint: str
    fstype: str
    opts: str = ""


@pytest.mark.parametrize("device_env", ["CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"])
@pytest.mark.parametrize("state", ["released", "leaked", "unavailable"])
def test_device_cleanup_is_a_blocking_single_probe(tmp_path, monkeypatch, mocker, device_env, state):
    from tests.dfx.lingbot_world import runner
    from tests.helpers import clean

    queried = []

    @contextmanager
    def device(index):
        queried.append(index)
        yield

    def memory():
        if state == "unavailable":
            raise OSError("device disappeared")
        used = 4 if state == "leaked" and queried[-1] == 1 else 1
        return ((64 - used) * 2**30, 64 * 2**30)

    monkeypatch.setenv(device_env, "4,5")
    monkeypatch.setattr(clean.current_omni_platform, "device_control_env_var", device_env)
    mocker.patch.object(clean.current_omni_platform, "device", side_effect=device)
    mocker.patch.object(clean.current_omni_platform, "mem_get_info", side_effect=memory)
    monkeypatch.setattr(clean, "_whisper_vram_allowance", lambda: (0.0, None))
    wall = iter((100.0, 0.0))
    monkeypatch.setattr(
        clean,
        "time",
        SimpleNamespace(
            time=lambda: next(wall),
            monotonic=lambda: 100.0,
            sleep=lambda _: pytest.fail("post-cleanup assertion must not add another wait"),
        ),
    )
    report = tmp_path / "cleanup.json"
    if state == "released":
        runner.assert_device_cleanup(2, report)
    else:
        expected = OSError if state == "unavailable" else ValueError
        with pytest.raises(expected, match="device disappeared|Device Memory Timeout"):
            runner.assert_device_cleanup(2, report)
    result = json.loads(report.read_text())
    assert result["passed"] is (state == "released")
    assert result["logical_devices"] == [0, 1] and result["threshold_ratio"] == 0.05
    assert queried == ([0] if state == "unavailable" else [0, 1])


@pytest.mark.parametrize("dies", [0, True, 1.0])
def test_device_cleanup_cannot_accept_empty_or_invalid_device_populations(tmp_path, dies):
    from tests.dfx.lingbot_world import runner

    report = tmp_path / "cleanup.json"
    with pytest.raises(RegressionError, match="positive integer die count"):
        runner.assert_device_cleanup(dies, report)
    assert json.loads(report.read_text())["passed"] is False


@pytest.mark.parametrize("unavailable", [None, "parent", "child"])
def test_startup_failure_records_process_state_before_cleanup(monkeypatch, unavailable):
    from tests.helpers import runtime

    events = []

    class Process:
        def __init__(self, pid):
            self.pid = pid
            if unavailable == "parent":
                raise runtime.psutil.NoSuchProcess(pid)

        def children(self, *, recursive):
            assert recursive
            return [Process(pid) for pid in range(101, 141)]

        def as_dict(self, *, attrs, ad_value):
            assert "cmdline" not in attrs and "environ" not in attrs
            assert "io_counters" not in attrs  # Not all psutil platforms expose it.
            assert {"pid", "status", "cpu_times", "memory_info"} <= set(attrs)
            if unavailable == "child" and self.pid == 101:
                raise runtime.psutil.AccessDenied(self.pid)
            return {"pid": self.pid, "status": "disk-sleep", "num_threads": 4}

    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [])
    server.proc = MagicMock(spec=runtime.subprocess.Popen)
    server.proc.pid = 100
    original_error = RuntimeError("original startup timeout")

    def start():
        raise original_error

    def wait_channel(path, *args, **kwargs):
        assert str(path).startswith("/proc/") and path.name == "wchan"
        if unavailable == "child" and path.parent.name == "101":
            raise PermissionError("wait channel unavailable")
        return "io_schedule\n"

    monkeypatch.setattr(server, "_start_server", start)
    monkeypatch.setattr(server, "_kill_process_tree", lambda pid: events.append(("cleanup", pid)))
    monkeypatch.setattr(runtime.psutil, "Process", Process)
    monkeypatch.setattr(Path, "read_text", wait_channel)
    monkeypatch.setattr(runtime.logger, "warning", lambda message, data: events.append(("snapshot", data)))
    with pytest.raises(RuntimeError, match="original startup timeout") as caught:
        with server:
            pytest.fail("failed startup cannot enter the test body")
    assert caught.value is original_error
    assert [event[0] for event in events] == ["snapshot", "cleanup"]
    snapshot = events[0][1]
    if unavailable == "parent":
        assert snapshot["processes"] == [] and "error" in snapshot
    else:
        assert len(snapshot["processes"]) == 32 and snapshot["omitted_processes"] == 9
        assert snapshot["processes"][0]["status"] == "disk-sleep"
        assert snapshot["processes"][0]["wait_channel"] == "io_schedule"
        if unavailable == "child":
            assert "error" in snapshot["processes"][1]
            assert "wait_channel_error" in snapshot["processes"][1]


def test_successful_startup_does_not_collect_process_diagnostics(monkeypatch):
    from tests.helpers import runtime

    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [])
    server.proc = MagicMock(spec=runtime.subprocess.Popen)
    server.proc.pid = 100
    monkeypatch.setattr(server, "_start_server", lambda: None)
    monkeypatch.setattr(server, "_kill_process_tree", lambda pid: None)
    monkeypatch.setattr(runtime.psutil, "Process", lambda pid: pytest.fail("successful startup must not be probed"))
    with server as ready:
        assert ready is server


@pytest.mark.parametrize(
    "membership,relative,limit_name,peak_name",
    [
        ("0::/pod", "pod", "memory.max", "memory.peak"),
        ("5:memory:/pod", "memory/pod", "memory.limit_in_bytes", "memory.max_usage_in_bytes"),
    ],
)
def test_resource_snapshot_retains_cgroup_limits_and_peaks(
    tmp_path, monkeypatch, membership, relative, limit_name, peak_name
):
    from tests.dfx.lingbot_world import resources

    proc = tmp_path / "cgroup-membership"
    proc.write_text(membership + "\n")
    root = tmp_path / "cgroups"
    group = root / relative
    group.mkdir(parents=True)
    (group / limit_name).write_text("8589934592\n")
    (group / peak_name).write_text("7516192768\n")
    (group / "memory.pressure").write_text("some avg10=12.00 total=1234\n")
    cpu = (
        {
            "cpu.max": "3200000 100000",
            "cpu.stat": "nr_periods 100\nnr_throttled 12\nthrottled_usec 456",
            "cpu.pressure": "some avg10=8.00 total=5678",
        }
        if membership.startswith("0::")
        else {}
    )
    for name, value in cpu.items():
        (group / name).write_text(value + "\n")
    monkeypatch.setattr(resources, "_PROC_CGROUP", proc)
    monkeypatch.setattr(resources, "_CGROUP_ROOT", root)

    report = resources.collect_resources()
    assert report["cgroup_membership"] == membership
    kernel = report["kernel_memory"]
    assert isinstance(kernel, dict)
    assert kernel[str(group / limit_name)] == "8589934592"
    assert kernel[str(group / peak_name)] == "7516192768"
    assert "avg10=12.00" in kernel[str(group / "memory.pressure")]
    for name, value in cpu.items():
        assert kernel[str(group / name)] == value


@pytest.mark.parametrize("relative", ["", "pod/container"])
def test_resource_snapshot_retains_v1_cpu_quota_and_throttling(tmp_path: Path, monkeypatch, relative: str):
    from tests.dfx.lingbot_world import resources

    proc = tmp_path / "cgroup"
    proc.write_text("3:cpu,cpuacct:/pod/container\n2:cpuset:/pod/container\n")
    root = tmp_path / "controllers"
    cpu = root / "cpu,cpuacct" / relative
    cpuset = root / "cpuset" / relative
    cpu.mkdir(parents=True)
    cpuset.mkdir(parents=True)
    fields = {
        cpu / "cpu.cfs_quota_us": "3200000",
        cpu / "cpu.cfs_period_us": "100000",
        cpu / "cpu.stat": "nr_periods 100\nnr_throttled 81\nthrottled_time 123456789",
        cpu / "cpuacct.usage": "987654321",
        cpuset / "cpuset.cpus": "0-63",
    }
    for path, value in fields.items():
        path.write_text(value + "\n")
    monkeypatch.setattr(resources, "_PROC_CGROUP", proc)
    monkeypatch.setattr(resources, "_CGROUP_ROOT", root)
    kernel = resources.collect_resources()["kernel_memory"]
    assert isinstance(kernel, dict)
    for path, value in fields.items():
        assert kernel[str(path)] == value


def test_resource_snapshot_resolves_weight_symlinks_and_keeps_only_mount_metadata(tmp_path, monkeypatch):
    from tests.dfx.lingbot_world import resources

    cache = tmp_path / "cache"
    cache.mkdir()
    weight = cache / "blob"
    weight.write_bytes(b"checkpoint")
    link = tmp_path / "weight.safetensors"
    link.symlink_to(weight)
    monkeypatch.setattr(resources, "_PROC_CGROUP", tmp_path / "unavailable-proc")
    monkeypatch.setattr(resources, "_CGROUP_ROOT", tmp_path / "unavailable-cgroup")
    monkeypatch.setattr(
        resources.psutil,
        "disk_partitions",
        lambda all: [
            _MountInfo(str(tmp_path), "ext4"),
            _MountInfo(str(cache), "nfs4", "test-option-must-not-be-retained"),
        ],
    )
    report = resources.collect_resources(link)
    assert report["model_storage"] == {"weight_path": str(weight), "filesystem": "nfs4", "mountpoint": str(cache)}
    assert "test-option-must-not-be-retained" not in json.dumps(report)


def test_unavailable_resource_probes_retain_errors_without_raising(tmp_path, monkeypatch):
    from tests.dfx.lingbot_world import resources

    def unavailable():
        raise PermissionError("memory probe unavailable")

    monkeypatch.setattr(resources, "_PROC_CGROUP", tmp_path / "unavailable-proc")
    monkeypatch.setattr(resources, "_CGROUP_ROOT", tmp_path / "unavailable-cgroup")
    monkeypatch.setattr(resources.psutil, "virtual_memory", unavailable)
    report = resources.collect_resources()
    assert report["host_memory"] is None and report["kernel_memory"] == {}
    assert report["model_storage"] is None
    errors = report["errors"]
    assert isinstance(errors, dict)
    assert "PermissionError" in errors["host_memory"]


def test_failed_pytest_retains_original_error_and_resource_evidence(tmp_path):
    test = tmp_path / "test_model_failure.py"
    test.write_text('def test_model_failure():\n    raise RuntimeError("original model startup failure")\n')
    output = tmp_path / "failure-artifacts"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tests.dfx.lingbot_world.conftest",
            str(test),
            f"--lingbot-output={output}",
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "BUILDKITE": "true", "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    failure = json.loads((output / "test_model_failure-call-failure.json").read_text())
    assert failure["passed"] is False and "original model startup failure" in failure["error"]
    assert "kernel_memory" in failure["resources"]
    assert "excluded from performance measurements" in failure["resources"]["scope"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "selection,expected_names",
    [
        ("core_model and cpu", ("test_selected_text_encoder_is_released_before_following_video_work",)),
        (
            "full_model and cuda and H100",
            ("test_cuda_smoke", "test_cuda_served_regression", "test_wbench_vbench_and_video_precision"),
        ),
        ("full_model and npu and A2", ("test_collect_npu_measurement_pair", "test_npu_served_regression")),
        ("core_model and xpu and B60", ()),
    ],
)
def test_model_and_dfx_packages_collect_together_on_every_platform(reverse, selection, expected_names):
    # Intel collects both trees before applying its XPU expression. DFX must
    # not shadow the existing model package, including for deselected tests.
    paths = ["tests/diffusion/models/lingbot_world/test_offload_lingbot_world.py", "tests/dfx/lingbot_world"]
    if reverse:
        paths.reverse()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *paths, "-m", selection],
        cwd=ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "BUILDKITE_PARALLEL_JOB": "0",
            "BUILDKITE_PARALLEL_JOB_COUNT": "1",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == (0 if expected_names else 5), result.stdout + result.stderr
    assert "ERROR collecting" not in result.stdout
    assert all(name in result.stdout for name in expected_names)


@pytest.mark.parametrize("timeout", [37, 1200])
def test_server_initialization_and_stage_deadlines_use_the_explicit_profile_budget(timeout):
    settings = {"deploy_config": "tests/dfx/lingbot_world/configs/npu_usp4_offload.yaml", "startup_timeout_s": timeout}
    arguments = server_arguments(settings)
    assert dict(zip(arguments[::2], arguments[1::2])) == {
        "--deploy-config": str(ROOT / settings["deploy_config"]),
        "--init-timeout": str(timeout),
        "--stage-init-timeout": str(timeout),
    }


@pytest.mark.parametrize("platform,count", [("cuda", 3), ("cuda", 6), ("npu", 6)])
def test_weekly_shards_cover_every_original_case_once_without_changing_measurements(platform, count):
    from tests.dfx.lingbot_world.prepare import profile_for_case_shard

    original = json.loads((ROOT / f"tests/dfx/lingbot_world/configs/{platform}_weekly.json").read_text())
    shards = [profile_for_case_shard(original, f"{index}/{count}") for index in range(count)]
    ids = [case["id"] for shard in shards for case in shard["cases"]]
    assert len(ids) == len(set(ids)) == len(original["cases"]) == 18
    assert set(ids) == {case["id"] for case in original["cases"]}
    assert all(len(shard["cases"]) == 18 // count for shard in shards)
    for shard in shards:
        assert {k: v for k, v in shard.items() if k not in {"cases", "case_shard"}} == {
            k: v for k, v in original.items() if k != "cases"
        }


@pytest.mark.parametrize("selection", ["", "-1/3", "3/3", "0/0", "0/19", "0/3/4"])
def test_case_shards_reject_invalid_or_empty_populations(selection):
    from tests.dfx.lingbot_world.prepare import profile_for_case_shard

    with pytest.raises(RegressionError):
        profile_for_case_shard({"cases": [{"id": str(index)} for index in range(18)]}, selection)


@pytest.mark.parametrize("mask", [None, "7,6,5,4,3,2,1,0", ",".join(f"GPU-{index}" for index in range(8))])
def test_reserved_cuda_allocation_preserves_the_original_device_mapping(mask):
    from tests.dfx.lingbot_world.prepare import select_cuda_devices

    expected = "0,1,2,3" if mask is None else ",".join(mask.split(",")[:4])
    assert select_cuda_devices(8, mask, dies=4, reserved_cards=8) == expected


@pytest.mark.parametrize(
    "available,mask,dies,reserved",
    [(4, None, 4, 8), (16, None, 4, 8), (8, None, 9, 8), (8, "0,1,2,3", 4, 8), (8, "0," * 7 + "0", 4, 8)],
)
def test_reserved_cuda_allocation_rejects_incomplete_or_ambiguous_visibility(available, mask, dies, reserved):
    from tests.dfx.lingbot_world.prepare import select_cuda_devices

    with pytest.raises(RegressionError):
        select_cuda_devices(available, mask, dies=dies, reserved_cards=reserved)


def test_exclusive_allocation_preflight_records_all_reserved_devices(tmp_path, monkeypatch):
    import torch

    from tests.dfx.lingbot_world.prepare import prepare_cuda_allocation

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,6,5,4,3,2,1,0")
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "H100")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: _DeviceProperties(uuid=f"GPU-{index}"))
    output = tmp_path / "allocation.json"
    assert (
        prepare_cuda_allocation({"platform": "cuda", "cards": 4, "dies": 4, "reserved_cards": 8}, output) == "7,6,5,4"
    )
    evidence = json.loads(output.read_text())
    assert evidence["reserved_cards"] == 8 and evidence["visible_dies"] == 4
    assert [device["uuid"] for device in evidence["devices"]] == [f"GPU-{index}" for index in range(8)]


def test_exclusive_hardware_contract_requires_matching_allocation_evidence(tmp_path, monkeypatch):
    import vllm_omni.platforms
    from tests.dfx.lingbot_world.runner import hardware_contract

    platform = vllm_omni.platforms.current_omni_platform
    monkeypatch.setattr(platform, "device_type", "cuda")
    monkeypatch.setattr(platform, "get_device_count", lambda: 4)
    monkeypatch.setattr(platform, "get_device_name", lambda index: "H100")
    monkeypatch.setattr(platform, "get_device_total_memory", lambda index: 80 * 1024**3)
    settings = {"platform": "cuda", "cards": 4, "dies": 4, "reserved_cards": 8}
    monkeypatch.delenv("LINGBOT_ALLOCATION_FILE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(RegressionError, match="allocation preflight"):
        hardware_contract(settings)
    output = tmp_path / "allocation.json"
    evidence = {
        "reserved_cards": 8,
        "visible_dies": 4,
        "visible_devices": "0,1,2,3",
        "devices": [{"uuid": f"GPU-{index}"} for index in range(8)],
    }
    output.write_text(json.dumps(evidence))
    monkeypatch.setenv("LINGBOT_ALLOCATION_FILE", str(output))
    assert hardware_contract(settings)["allocation"] == evidence
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    with pytest.raises(RegressionError, match="allocation evidence"):
        hardware_contract(settings)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    evidence["devices"] = [{"uuid": "GPU-duplicate"}] * 8
    output.write_text(json.dumps(evidence))
    with pytest.raises(RegressionError, match="allocation evidence"):
        hardware_contract(settings)


@pytest.mark.parametrize("mask", [None, "7,6,5,4,3,2,1,0", "8,9,10,11,12,13,14,15"])
def test_npu_allocation_preserves_cann_runtime_mapping(mask):
    from tests.dfx.lingbot_world.prepare import select_npu_devices

    expected = "0,1,2,3" if mask is None else ",".join(mask.split(",")[:4])
    assert select_npu_devices(8, mask, dies=4, reserved_cards=8) == expected


@pytest.mark.parametrize(
    "available,mask", [(4, None), (16, None), (8, "0,1,2,3"), (8, "0," * 7 + "0"), (8, "-1,1,2,3,4,5,6,7")]
)
def test_npu_allocation_rejects_partial_or_ambiguous_reservations(available, mask):
    from tests.dfx.lingbot_world.prepare import select_npu_devices

    with pytest.raises(RegressionError):
        select_npu_devices(available, mask, dies=4, reserved_cards=8)


def test_npu_preflight_and_contract_verify_mask_and_actual_device_properties(tmp_path, monkeypatch):
    import vllm_omni.platforms
    from tests.dfx.lingbot_world import runner
    from tests.dfx.lingbot_world.prepare import prepare_npu_allocation

    platform = vllm_omni.platforms.current_omni_platform
    count = [8]
    monkeypatch.setattr(platform, "device_type", "npu")
    monkeypatch.setattr(platform, "get_device_count", lambda: count[0])
    monkeypatch.setattr(platform, "get_device_name", lambda index: "Ascend910B3")
    monkeypatch.setattr(platform, "get_device_total_memory", lambda index: 64 * 1024**3)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda package: "test-version")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7,6,5,4,3,2,1,0")
    settings = {"platform": "npu", "cards": 4, "dies": 4, "reserved_cards": 8}
    output = tmp_path / "allocation.json"
    assert prepare_npu_allocation(settings, output) == "7,6,5,4"
    evidence = json.loads(output.read_text())
    assert [device["runtime_id"] for device in evidence["devices"]] == list("76543210")
    count[0] = 4
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7,6,5,4")
    monkeypatch.delenv("LINGBOT_ALLOCATION_FILE", raising=False)
    with pytest.raises(RegressionError, match="allocation preflight"):
        runner.hardware_contract(settings)
    monkeypatch.setenv("LINGBOT_ALLOCATION_FILE", str(output))
    assert runner.hardware_contract(settings)["allocation"] == evidence
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(RegressionError, match="allocation evidence"):
        runner.hardware_contract(settings)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7,6,5,4")
    evidence["devices"][0]["total_bytes"] -= 1
    output.write_text(json.dumps(evidence))
    with pytest.raises(RegressionError, match="visible hardware"):
        runner.hardware_contract(settings)


def test_reference_server_keeps_backend_import_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = tmp_path / "cann"
    backend.mkdir()
    (backend / "compiler_backend.py").write_text('origin = "backend"\n')
    for name in ("reference", "candidate"):
        package = tmp_path / name / "vllm_omni"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f'origin = "{name}"\n')
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(tmp_path / "candidate"), str(backend))))
    result = subprocess.run(
        [sys.executable, "-c", "import vllm_omni, compiler_backend; print(vllm_omni.origin, compiler_backend.origin)"],
        cwd=tmp_path,
        env={**os.environ, **server_environment(tmp_path / "reference")},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "reference backend"


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_engine_switches_override_inherited_environment_before_import(tmp_path, monkeypatch, role):
    monkeypatch.setenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER", "1")
    monkeypatch.setenv("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH", "999")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,runpy,sys; config=runpy.run_path(sys.argv[1]); "
            "print(config['contiguous_kv_gather_enabled']()); "
            "print(os.environ['AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH'])",
            str(ROOT / "vllm_omni/experimental/ar_diffusion/kv_cache/config.py"),
        ],
        env={**os.environ, **server_environment(tmp_path / role)},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.splitlines() == ["False", "0"]


def test_generation_records_fixed_engine_environment_before_capture_failure(tmp_path, monkeypatch):
    from tests.dfx.lingbot_world import runner

    output = tmp_path / "output"
    profile = tmp_path / "profile.json"
    config = json.loads((ROOT / "tests/dfx/lingbot_world/configs/npu_nightly.json").read_text())
    profile.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "checked_assets", lambda settings: (tmp_path / "model", {}))
    monkeypatch.setattr(runner, "validate_reference", lambda settings: tmp_path / "reference")
    monkeypatch.setattr(runner, "hardware_contract", lambda settings: {"platform": "npu"})

    def fail_capture(*args):
        checkpoint = json.loads((output / "generation.json").read_text())
        assert checkpoint["candidate_revision"] == runner.git_revision(ROOT)
        assert not checkpoint["passed"] and checkpoint["status"] == "running"
        raise RuntimeError("precision provisioning failed")

    monkeypatch.setattr(runner, "_capture", fail_capture)
    with pytest.raises(RuntimeError, match="precision provisioning failed"):
        runner.generate({"output": output, "profile_path": profile, "deploy_config": config["deploy_config"]})
    assert json.loads((output / "contract.json").read_text())["engine_environment"] == {
        "VLLM_OMNI_AR_DIFFUSION_KV_GATHER": "0",
        "AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH": "0",
    }


def test_fixed_worker_threads_apply_before_torch_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "19")
    monkeypatch.setenv("MKL_NUM_THREADS", "23")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "29")
    env = server_environment(tmp_path / "reference", cpu_threads=1)
    assert all(env[name] == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.get_num_threads())"],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "1"


@pytest.mark.parametrize("threads", [0, True])
def test_invalid_worker_thread_count_fails_before_launch(tmp_path: Path, threads):
    with pytest.raises(RegressionError, match="positive integer"):
        server_environment(tmp_path, cpu_threads=threads)


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_fixed_worker_threads_also_reach_precision_capture(tmp_path: Path, monkeypatch, role: str):
    from tests.dfx.lingbot_world import runner

    calls = []
    monkeypatch.setenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER", "1")
    monkeypatch.setenv("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH", "999")
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    source = tmp_path / role
    runner._capture(
        source,
        tmp_path / "model",
        tmp_path / "precision.npz",
        {
            "height": 480,
            "width": 832,
            "precision_timeout_s": 600,
            "cpu_threads_per_worker": 1,
        },
    )
    assert len(calls) == 1
    options = calls[0][1]
    assert options["cwd"] == source
    assert all(options["env"][name] == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    assert options["env"]["VLLM_OMNI_AR_DIFFUSION_KV_GATHER"] == "0"
    assert options["env"]["AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH"] == "0"


def evaluator_repository(root: Path):
    directory = root / "tests/dfx/lingbot_world"
    directory.mkdir(parents=True)
    (directory / "evaluators.lock.json").write_text('{"repositories": {}, "files": {}}')
    (directory / "opencv_provider.py").write_text("# canonical provider policy\n")
    (root / "requirements").mkdir()
    (root / "requirements/lingbot-quality.txt").write_text("example==1\n")


@pytest.mark.parametrize("version", ["0.32.0", "0.32.0+ascend", "0.31.0", "0.33.0", "0.32.0rc1", "0.32.0.dev1"])
def test_inference_runtime_requires_v032_release(version, monkeypatch):
    from tests.dfx.lingbot_world import prepare

    monkeypatch.setattr(prepare.importlib.metadata, "version", lambda name: version)
    if version in {"0.32.0", "0.32.0+ascend"}:
        prepare.verify_inference_runtime()
    else:
        with pytest.raises(RegressionError, match="requires vLLM 0.32.0"):
            prepare.verify_inference_runtime()


@pytest.mark.parametrize("broken_role", [None, "reference", "candidate"])
def test_source_import_preflight_uses_each_checkout(tmp_path, monkeypatch, broken_role):
    from tests.dfx.lingbot_world import prepare

    for role in ("reference", "candidate"):
        package = tmp_path / role / "vllm_omni"
        cli = package / "entrypoints/cli"
        cli.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (cli / "main.py").write_text("raise ImportError('removed runtime API')\n" if role == broken_role else "")
    monkeypatch.setattr(prepare, "ROOT", tmp_path / "candidate")
    # A candidate inherited from the job must not shadow the reference.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "candidate"))
    if broken_role:
        with pytest.raises(RegressionError, match=f"{broken_role} serving import preflight failed"):
            prepare.verify_source_imports(tmp_path / "reference")
    else:
        prepare.verify_source_imports(tmp_path / "reference")


@pytest.mark.parametrize("failure", ["runtime", "imports"])
def test_reference_preparation_checks_compatibility_before_model_download(tmp_path, monkeypatch, failure):
    from tests.dfx.lingbot_world import prepare

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"reference_revision": "a" * 40}))
    monkeypatch.setattr(
        sys, "argv", ["prepare", "reference", "--profile", str(profile), "--root", str(tmp_path / "ref")]
    )
    monkeypatch.delenv("LINGBOT_MODEL_PATH", raising=False)
    calls = []

    def check(stage):
        calls.append(stage)
        if stage == failure:
            raise RegressionError("incompatible source/runtime")

    monkeypatch.setattr(prepare, "verify_inference_runtime", lambda: check("runtime"))
    monkeypatch.setattr(prepare, "checkout", lambda *args: check("checkout"))
    monkeypatch.setattr(prepare, "verify_source_imports", lambda *args: check("imports"))
    monkeypatch.setattr(prepare.subprocess, "check_output", lambda *args, **kwargs: pytest.fail("download started"))
    with pytest.raises(RegressionError, match="incompatible source/runtime"):
        prepare.main()
    assert calls == (["runtime"] if failure == "runtime" else ["runtime", "checkout", "imports"])


def test_explicit_model_provisioning_preserves_offline_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.dfx.lingbot_world import prepare

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"reference_revision": "a" * 40}))
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "model.json"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("LINGBOT_MODEL_PATH", raising=False)
    monkeypatch.setattr(prepare, "verify_inference_runtime", lambda: None)
    monkeypatch.setattr(prepare, "verify_source_imports", lambda *args: None)
    monkeypatch.setattr(prepare, "checkout", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "reference",
            "--profile",
            str(profile),
            "--root",
            str(tmp_path / "ref"),
            "--env-output",
            str(output),
        ],
    )
    calls = []

    def download(command, **kwargs):
        calls.append((command, kwargs))
        return str(model) + "\n"

    monkeypatch.setattr(prepare.subprocess, "check_output", download)
    prepare.main()
    command, options = calls[0]
    lock = json.loads((ROOT / "tests/dfx/lingbot_world/assets.lock.json").read_text())
    assert command == ["hf", "download", lock["model_id"], "--revision", lock["model_revision"], "--quiet"]
    assert options["env"]["HF_HUB_OFFLINE"] == "0"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert json.loads(output.read_text())["model_path"] == str(model)


def test_evaluator_cache_isolated_by_code_weights_and_dependencies(tmp_path: Path):
    from tests.dfx.lingbot_world.prepare import evaluator_cache_key

    evaluator_repository(tmp_path)
    initial = evaluator_cache_key(tmp_path)
    (tmp_path / "requirements/lingbot-quality.txt").write_text("example==2\n")
    assert evaluator_cache_key(tmp_path) != initial
    (tmp_path / "requirements/lingbot-quality.txt").write_text("example==1\n")
    (tmp_path / "tests/dfx/lingbot_world/evaluators.lock.json").write_text('{"revision": "different"}')
    assert evaluator_cache_key(tmp_path) != initial
    (tmp_path / "tests/dfx/lingbot_world/evaluators.lock.json").write_text('{"repositories": {}, "files": {}}')
    (tmp_path / "tests/dfx/lingbot_world/opencv_provider.py").write_text("# changed canonical provider policy\n")
    assert evaluator_cache_key(tmp_path) != initial


def test_ready_evaluator_is_never_resynced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.dfx.lingbot_world import prepare

    repository = tmp_path / "repository"
    evaluator_repository(repository)
    monkeypatch.setattr(prepare, "ROOT", repository)
    monkeypatch.setattr(
        prepare,
        "verify_environment",
        lambda *args: {"opencv-python": "4.11.0.86", "opencv-python-headless": "4.11.0.86"},
    )
    calls = []

    def provision(command, **kwargs):
        calls.append(command)
        if command[:2] == ["uv", "venv"]:
            python = Path(command[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        if len(command) > 2 and command[1] == "-I":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "schema_version": 1,
                        "provider": "opencv-python-headless",
                        "provider_version": "4.11.0.86",
                        "opencv_version": "4.11.0",
                        "opencv_build_sha256": "a" * 64,
                        "native_module_sha256": "b" * 64,
                        "module_path": str(tmp_path / "cv2/__init__.py"),
                        "python": command[0],
                        "verified_files": 2,
                    }
                ),
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare.subprocess, "run", provision)
    root = tmp_path / "cache"
    prepare._prepare_evaluators(root)
    prepare._prepare_evaluators(root)
    assert sum(command[:3] == ["uv", "pip", "sync"] for command in calls) == 1
    assert sum(command[:3] == ["uv", "pip", "install"] for command in calls) == 1
    (repository / "requirements/lingbot-quality.txt").write_text("example==2\n")
    with pytest.raises(RegressionError, match="fresh evaluator root"):
        prepare._prepare_evaluators(root)
    assert sum(command[:3] == ["uv", "pip", "sync"] for command in calls) == 1


def leaves(document):
    for step in document.get("steps", []):
        if "steps" in step:
            yield from leaves(step)
        else:
            yield step


@pytest.mark.parametrize(
    "generation_passed,mutation",
    [
        (True, None),
        (False, None),
        (False, "unlisted"),
        (False, "corrupt"),
        (True, "population"),
        (False, "population"),
        (True, "population_reference"),
        (True, "population_candidate"),
        (True, "population_pixels"),
        (True, "population_long"),
        (False, "population_long"),
        (True, "population_long_reference"),
        (True, "population_long_candidate"),
        (True, "population_long_pixels"),
        (True, "population_long_spread"),
        (True, "population_spread"),
        (True, "population_spread_candidate"),
        (True, "population_case_spread"),
        (True, "population_case_spread_tight"),
        (True, "population_missing_spreads"),
        (True, "population_unrecorded"),
        (False, "population_unrecorded"),
        (True, "population_long_unrecorded"),
        (True, "population_general_unrecorded"),
        (True, "opencv_provider_missing"),
        (True, "opencv_worker_build"),
        (True, "quality_case_0"),
        (True, "quality_case_1"),
        (True, "quality_case_2"),
        (False, "quality_case_2"),
        (True, "quality_case_invalid"),
        (True, "quality_case_corrupt"),
    ],
)
def test_quality_retains_scores_after_generation_failure_without_passing(
    tmp_path, monkeypatch, generation_passed, mutation
):
    import cv2
    import numpy as np

    from benchmarks.lingbot_world.regression import sha256, write_json
    from tests.dfx.lingbot_world import test_quality as gate

    source = tmp_path / "source"
    policy = source / "tests/dfx/lingbot_world"
    policy.mkdir(parents=True)
    limits = json.loads((ROOT / "tests/dfx/lingbot_world/quality.json").read_text())
    write_json(policy / "quality.json", limits)
    write_json(
        policy / "evaluators.lock.json",
        {"files": {}, "repositories": {suite: {"revision": "snapshot"} for suite in ("vbench", "wbench")}},
    )
    generation = tmp_path / "generation"
    artifacts = {}
    y, x = np.indices((24, 32))
    texture = ((x * 3 + y * 5) % 80 + 40).astype(np.uint8)
    frame = np.stack([texture, texture + 5, texture + 10], axis=-1)
    for role in ("reference", "candidate"):
        for repeat in range(3):
            path = generation / role / str(repeat) / "scene/rollout.mp4"
            path.parent.mkdir(parents=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16, (32, 24))
            assert writer.isOpened(), "CPU quality coordinator test requires an MP4 encoder"
            try:
                for _ in range(9):
                    writer.write(frame)
            finally:
                writer.release()
            artifacts[str(path.relative_to(generation))] = sha256(path)
    if mutation == "unlisted":
        artifacts.pop("candidate/2/scene/rollout.mp4")
    elif mutation == "corrupt":
        (generation / "candidate/2/scene/rollout.mp4").write_bytes(b"corrupt output")
    profile = tmp_path / "profile.json"
    write_json(profile, {"platform": "npu", "num_chunks": 1})
    write_json(
        generation / "generation.json",
        {
            "passed": generation_passed,
            "contract": {"profile": json.loads(profile.read_text())},
            "candidate_revision": "snapshot",
            "artifacts": artifacts,
        },
    )
    monkeypatch.setattr(gate, "ROOT", source)
    monkeypatch.setattr(gate, "git_revision", lambda path: "snapshot")
    packages = {
        "numpy": "1.26.4",
        "opencv-python": "4.11.0.86",
        "opencv-python-headless": "4.11.0.86",
        "scikit-image": "0.25.2",
    }
    pixel_packages = {name: packages[name] for name in ("numpy", "opencv-python", "scikit-image")}
    monkeypatch.setattr(gate, "verify_environment", lambda *args: dict(packages))
    monkeypatch.setattr(gate, "hardware_contract", lambda settings: {"unit_test": True})
    monkeypatch.setattr(gate.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    monkeypatch.setenv("LINGBOT_EVALUATOR_ROOT", str(tmp_path / "evaluators"))
    monkeypatch.setenv("LINGBOT_EVALUATOR_PYTHON", sys.executable)
    provider_record = {
        "schema_version": 1,
        "provider": "opencv-python-headless",
        "provider_version": "4.11.0.86",
        "opencv_version": "4.11.0",
        "opencv_build_sha256": "a" * 64,
        "native_module_sha256": "b" * 64,
        "module_path": str(tmp_path / "cv2/__init__.py"),
        "python": sys.executable,
        "verified_files": 2,
    }
    if mutation != "opencv_provider_missing":
        write_json(tmp_path / "evaluators/opencv-provider.json", provider_record)
    for name in ("VBENCH_CACHE_DIR", "WBENCH_WEIGHTS_DIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    calls = []

    def score(videos, output, *, suite, dimensions, **kwargs):
        calls.append(suite)
        if mutation and mutation.startswith("quality_case_"):
            assert len(videos) == 6
            assert {path.parent.name for path in videos.values()} == {f"scene-{settings['quality_case_index']}"}
        # A job-level kill can prevent the coordinator's finally block from running.
        checkpoint = json.loads((output.parent / "quality.json").read_text())
        assert checkpoint["evaluator_packages"] == packages
        assert checkpoint["candidate_revision"] == "snapshot"
        assert checkpoint["evaluator_opencv_provider"] == provider_record
        assert checkpoint["evaluator_hardware"] == {"unit_test": True}
        assert checkpoint["generation_manifest_sha256"] == sha256(generation / "generation.json")
        assert checkpoint["generation_passed"] is generation_passed
        assert not checkpoint["passed"] and not checkpoint["quality_passed"]
        frames = (settings["num_chunks"] * 3 - 1) * 4 + 1
        scores = {label: {name: 1.0 for name in dimensions} for label in videos}
        if population_mode and suite == "wbench":
            for label, row in scores.items():
                row["hpsv3_quality"] = 0.81 if "reference" in label else 0.79
                if mutation.endswith("_reference") and "reference" in label:
                    row["hpsv3_quality"] = 0.7
                if (
                    mutation.endswith("_candidate")
                    and mutation != "population_spread_candidate"
                    and "candidate" in label
                ):
                    row["hpsv3_quality"] = 0.7
                if mutation in (
                    "population_long_spread",
                    "population_spread",
                    "population_spread_candidate",
                    "population_case_spread",
                    "population_case_spread_tight",
                ):
                    row["aesthetic_quality"] = (
                        0.78
                        if mutation == "population_spread_candidate" and "candidate" in label
                        else 0.83
                        if label.endswith("reference1")
                        else 0.8
                    )
        return {
            "status": "passed",
            "protocol": {"dynamic_degree_backend": "raft"},
            "scores": scores,
            "native_provenance": {
                label: {
                    "background_clip_inference": {
                        "batch_size": 32,
                        "encoded_images": frames,
                        "forward_batches": (frames + 31) // 32,
                        "max_forward_batch": min(frames, 32),
                        "gradients": False,
                    }
                }
                for label in videos
            },
        }

    monkeypatch.setattr(gate, "evaluate_video_quality", score)
    output = tmp_path / "quality"

    def compare_in_environment(first, second, destination, *, python, expected_packages, **kwargs):
        from benchmarks.lingbot_world.video_quality import compare_video_pair

        # The coordinator test mocks the interpreter boundary; real isolated
        # decoding/version/timeout checks live in test_lingbot_world_video_quality.
        assert python == Path(sys.executable) and expected_packages == packages
        assert destination == output / "pixels"
        kwargs.pop("timeout_s")
        row = compare_video_pair(first, second, **kwargs)
        row["pixel_environment"] = {
            "packages": dict(pixel_packages),
            "synthetic_boundary": True,
            "opencv_build_sha256": ("c" if mutation == "opencv_worker_build" else "a") * 64,
        }
        return row

    monkeypatch.setattr(gate, "compare_video_pair_in_environment", compare_in_environment)
    settings = {
        "output": output,
        "generation": generation,
        "profile_path": profile,
        "platform": "npu",
        "num_chunks": 1,
        "width": 32,
        "height": 24,
        "mux_fps": 16,
        "cases": [{"id": "scene"}],
    }
    if mutation and mutation.startswith("quality_case_"):
        settings["cases"] = [{"id": f"scene-{index}"} for index in range(3)]
        settings["quality_case_index"] = (
            3
            if mutation == "quality_case_invalid"
            else (1 if mutation == "quality_case_corrupt" else int(mutation.rsplit("_", 1)[1]))
        )
        manifest = json.loads((generation / "generation.json").read_text())
        manifest["artifacts"] = {}
        for relative in artifacts:
            for case in settings["cases"]:
                target = generation / relative.replace("scene/", case["id"] + "/")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((generation / relative).read_bytes())
                manifest["artifacts"][str(target.relative_to(generation))] = sha256(target)
        write_json(profile, {"platform": "npu", "num_chunks": 1, "cases": settings["cases"]})
        manifest["contract"]["profile"] = json.loads(profile.read_text())
        write_json(generation / "generation.json", manifest)
        if mutation == "quality_case_corrupt":
            # Even an unselected case must retain the producer's manifest integrity.
            (generation / "candidate/2/scene-0/rollout.mp4").write_bytes(b"corrupt")
        if mutation in ("quality_case_invalid", "quality_case_corrupt"):
            with pytest.raises(RegressionError, match="Quality case index|checksum"):
                gate.test_wbench_vbench_and_video_precision(settings)
            assert calls == []
            assert not json.loads((output / "quality.json").read_text())["passed"]
            return
    population_mode = mutation is not None and mutation.startswith("population")
    if population_mode:
        # Synthetic coordinator inputs exercise baseline selection and the
        # final AND gate. Real decoding is covered by the NPU cases above.
        chunks = 600 if mutation.startswith("population_long") else 40
        frames = (chunks * 3 - 1) * 4 + 1
        settings.update(platform="cuda", num_chunks=chunks)
        write_json(profile, {"platform": "cuda", "num_chunks": chunks, "cases": settings["cases"]})
        contract = {
            "profile": json.loads(profile.read_text()),
            "deploy_sha256": "deployment",
            "hardware": {"platform": "cuda", "cards": 4, "dies": 4, "devices": [], "packages": {}},
            "engine_environment": dict(ENGINE_ENVIRONMENT),
        }
        manifest = json.loads((generation / "generation.json").read_text())
        manifest["contract"] = contract
        write_json(generation / "generation.json", manifest)
        write_json(policy / "assets.lock.json", {})
        calibration = policy / "calibration"
        calibration.mkdir()
        write_json(calibration / "cuda_nightly_reference.json", {})
        monkeypatch.setattr(gate, "verify_reference_calibration_contract", lambda *args: None)
        if mutation == "population_general_unrecorded":

            def reject_unrecorded(*args):
                raise RegressionError("Reference calibration engine environment is missing or changed")

            monkeypatch.setattr(gate, "verify_reference_calibration_contract", reject_unrecorded)
        pixel = {
            "pixel_environment": {"packages": dict(pixel_packages), "opencv_build_sha256": "a" * 64},
            "ssim": {"mean": 0.4},
            "total": {"psnr_db": 30.0, "psnr_is_infinite": False},
            "windows": [
                {"index": 0, "start_frame": 0, "frame_count": frames, "psnr_db": 30.0, "psnr_is_infinite": False}
            ],
            "decoded_frames": frames,
            "uniform_frame_indices": {"reference": [], "candidate": []},
        }
        suites = {}
        for suite in ("vbench", "wbench"):
            row = {name: 1.0 for name in limits["candidate_tolerances"]["cuda"][suite] if name != "dynamic_degree"}
            if suite == "wbench":
                row["hpsv3_quality"] = 0.8
                if mutation in (
                    "population_long_spread",
                    "population_spread",
                    "population_spread_candidate",
                    "population_case_spread",
                    "population_case_spread_tight",
                ):
                    row["aesthetic_quality"] = 0.8
            suites[suite] = [row] * 6
        reference_spreads = json.loads(json.dumps(limits["max_reference_spreads"]["cuda"]))
        if mutation in ("population_long_spread", "population_spread", "population_spread_candidate"):
            reference_spreads["wbench"]["aesthetic_quality"] = 0.05 if mutation == "population_long_spread" else 0.035
        write_json(
            calibration
            / ("cuda_long_quality_population.json" if chunks == 600 else "cuda_nightly_quality_population.json"),
            {
                "schema_version": 1,
                "contract": contract,
                "evaluator_packages_sha256": hashlib.sha256(
                    json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "evaluator_lock_sha256": sha256(policy / "evaluators.lock.json"),
                "asset_lock_sha256": sha256(policy / "assets.lock.json"),
                "max_reference_spreads": reference_spreads,
                "cases": {"scene": {"input": settings["cases"][0], "pixels": [pixel] * 6, "suites": suites}},
            },
        )
        if mutation in ("population_case_spread", "population_case_spread_tight"):
            path = calibration / "cuda_nightly_quality_population.json"
            population = json.loads(path.read_text())
            population["schema_version"] = 3
            population["reference_pixel_calibration"] = {"protocol": "all_unordered_pairs_of_six_references"}
            case = population["cases"]["scene"]
            case["pixels"] = [pixel] * 15
            case["pixel_pairs"] = [[first, second] for first in range(6) for second in range(first + 1, 6)]
            case["reference_videos"] = [
                {
                    "generation_job_id": f"cohort-{cohort}",
                    "path": f"reference/{repeat}/scene/rollout.mp4",
                    "sha256": "d" * 64,
                }
                for cohort in range(2)
                for repeat in range(3)
            ]
            case["max_reference_spreads"] = json.loads(json.dumps(reference_spreads))
            case["max_reference_spreads"]["wbench"]["aesthetic_quality"] = (
                0.005 if mutation.endswith("_tight") else 0.035
            )
            # Opposite global limits prove the evaluator uses this case's cap.
            population["max_reference_spreads"]["wbench"]["aesthetic_quality"] = (
                0.5 if mutation.endswith("_tight") else 0.005
            )
            write_json(path, population)
        if mutation == "population_missing_spreads":
            population_path = calibration / "cuda_nightly_quality_population.json"
            invalid = json.loads(population_path.read_text())
            del invalid["max_reference_spreads"]
            write_json(population_path, invalid)
        if mutation.endswith("_unrecorded") and mutation != "population_general_unrecorded":
            path = calibration / (
                "cuda_long_quality_population.json" if chunks == 600 else "cuda_nightly_quality_population.json"
            )
            unrecorded = json.loads(path.read_text())
            del unrecorded["contract"]["engine_environment"]
            write_json(path, unrecorded)

        def compare(first, second, destination, *, python, expected_packages, **kwargs):
            assert python == Path(sys.executable) and expected_packages == packages
            assert destination == output / "pixels"
            row = json.loads(json.dumps(pixel))
            candidate = "candidate" in second.parts
            row["ssim"]["mean"] = (0.2 if mutation.endswith("_pixels") else 0.37) if candidate else 0.6
            row["reference"] = {"frame_count": frames, "width": 32, "height": 24, "fps": 16}
            return row

        monkeypatch.setattr(gate, "compare_video_pair_in_environment", compare)
    if mutation in ("unlisted", "corrupt"):
        with pytest.raises(RegressionError, match="checksum|Unhashed"):
            gate.test_wbench_vbench_and_video_precision(settings)
        assert calls == []
        assert json.loads((output / "quality.json").read_text())["candidate_revision"] == "snapshot"
        return
    if mutation == "opencv_provider_missing":
        with pytest.raises(FileNotFoundError, match="opencv-provider"):
            gate.test_wbench_vbench_and_video_precision(settings)
        assert calls == []
        return
    if mutation == "opencv_worker_build":
        with pytest.raises(RegressionError, match="Pixel worker OpenCV build"):
            gate.test_wbench_vbench_and_video_precision(settings)
        assert calls == ["vbench", "wbench"]
        assert not json.loads((output / "quality.json").read_text())["passed"]
        return
    uncalibrated = mutation is not None and (
        mutation.endswith("_unrecorded") or mutation == "population_missing_spreads"
    )
    quality_passed = (
        (not population_mode or not mutation.endswith(("_reference", "_candidate", "_pixels")))
        and not uncalibrated
        and mutation != "population_case_spread_tight"
    )
    if generation_passed and quality_passed:
        gate.test_wbench_vbench_and_video_precision(settings)
    else:
        with pytest.raises(AssertionError, match="Quality regression"):
            gate.test_wbench_vbench_and_video_precision(settings)
    report = json.loads((output / "quality.json").read_text())
    assert calls == ["vbench", "wbench"]
    assert report["quality_passed"] is quality_passed
    assert report["generation_passed"] is generation_passed
    assert report["passed"] is (generation_passed and quality_passed)
    if mutation and mutation.startswith("quality_case_"):
        selected = f"scene-{settings['quality_case_index']}"
        assert report["evaluated_cases"] == [selected]
        assert report["quality_case_index"] == settings["quality_case_index"]
        assert set(report["pixels"]) == {selected}
        assert all(set(rows) == {selected} for rows in report["suites"].values())
        assert len(json.loads((output / "profile.json").read_text())["cases"]) == 3
    if mutation in ("population_spread", "population_spread_candidate"):
        checks = report["suites"]["wbench"]["scene"]
        assert all(row["passed"] for row in checks if "reference_repeat" in row)
        if mutation == "population_spread_candidate":
            for gate_check in checks:
                if "candidate_repeat" in gate_check:
                    dimension_result = gate_check["dimensions"]["aesthetic_quality"]
                    assert not gate_check["passed"] and not dimension_result["passed"]
                    assert dimension_result["minimum"] == pytest.approx(0.79)
                    assert dimension_result["candidate_tolerance"] == 0.01
    if population_mode:
        assert (output / "quality-reference-population.json").is_file()
        if not uncalibrated:
            assert len(report["pixels"]["scene"]["reference_checks"]) == 3
    if uncalibrated:
        expected_error = "reference spread" if mutation == "population_missing_spreads" else "engine environment"
        assert report["calibration_errors"] and all(
            expected_error in row["error"] for row in report["calibration_errors"]
        )
        assert len(report["pixels"]["scene"]["reference_repeat"]) == len(report["pixels"]["scene"]["candidate"]) == 3
        assert all(row["status"] == "unjudged" and not row["passed"] for row in report["pixels"]["scene"]["checks"])
    assert (output / "vbench-raw.json").is_file() and (output / "wbench-raw.json").is_file()


@pytest.mark.parametrize(
    "level,key",
    [
        ("nightly", "lingbot-cuda-nightly-generation"),
        ("weekly", "lingbot-cuda-weekly-generation"),
        ("weekly", "lingbot-cuda-weekly-long"),
    ],
)
def test_cuda_generation_retries_only_one_lost_agent(level, key):
    steps = leaves(yaml.safe_load((ROOT / f".buildkite/cuda/test-{level}.yml").read_text()))
    step = next(item for item in steps if item.get("key") == key)
    assert step["retry"] == {"automatic": [{"exit_status": -1, "limit": 1}]}
    assert step["mirror_hardwares"] == "h100_4_exclusive" and not step.get("soft_fail")
    rules = step["retry"]["automatic"]
    # Numerical/quality assertions, OOM, cancellation and timeout stay failed.
    for status in (1, 2, 124, 128, 130, 137, 143):
        assert not any(rule["exit_status"] in ("*", status) for rule in rules)


@pytest.mark.parametrize("platform", ["cuda", "npu"])
@pytest.mark.parametrize("level", ["nightly", "weekly"])
def test_generation_and_gpu_evaluation_are_blocking_and_connected(platform, level):
    name = f"test-{level}.yml" if platform == "cuda" else f"test-npu-{level}.yml"
    steps = list(leaves(yaml.safe_load((ROOT / f".buildkite/{platform}/{name}").read_text())))
    keyed = {step.get("key"): step for step in steps}
    prefix = f"lingbot-{platform}-{level}"
    generate, evaluate = keyed[prefix + "-generation"], keyed[prefix + "-quality"]
    assert len(evaluate["commands"][0].split()) == 6
    if platform == "cuda":
        assert generate["mirror_hardwares"] == "h100_4_exclusive"
    else:
        assert generate["mirror_hardwares"] == "a2b3_npu_8"
    producer = prefix + ("-gate" if platform == "npu" else "-generation")
    assert {"step": producer, "allow_failure": True} in evaluate["depends_on"]
    assert evaluate["mirror_hardwares"] == "h100_1"
    for step in (generate, evaluate):
        assert not step.get("soft_fail")
        assert step["artifact_paths"]
        assert step["source_file_dependencies"] == "lingbot_world_regression"
    if platform == "npu":
        assert prefix + "-evaluator-image" in evaluate["depends_on"]
        if level == "nightly":
            assert generate["matrix"] == {"setup": {"repeat": ["0", "1", "2"]}}
            assert generate["env"]["LINGBOT_MEASUREMENT_REPEAT"] == "{{matrix.repeat}}"
            assert " collect " in generate["commands"][0] and generate["timeout_in_minutes"] == 240
            gate = keyed[producer]
            assert prefix + "-generation" in gate["depends_on"]
            assert prefix + "-evaluator-image" in gate["depends_on"]
            assert gate["mirror_hardwares"] == "l4_1" and not gate.get("soft_fail")
            assert gate["artifact_paths"] and gate["source_file_dependencies"] == "lingbot_world_regression"
            assert " aggregate " in gate["commands"][0] and len(gate["commands"][0].split()) == 6
            assert evaluate["commands"][0].endswith("lingbot-artifacts/npu-nightly")
    if level == "weekly":
        assert prefix + "-long" in keyed
        count = 6
        shard_matrix = {"setup": {"shard": [str(i) for i in range(count)]}}
        assert evaluate["matrix"] == {"setup": {**shard_matrix["setup"], "case": ["0", "1", "2"]}}
        assert evaluate["env"]["LINGBOT_QUALITY_CASE_INDEX"] == "{{matrix.case}}"
        assert "quality/shard-{{matrix.shard}}/case-{{matrix.case}} " in evaluate["commands"][0]
        assert (
            generate["env"]["LINGBOT_CASE_SHARD"]
            == evaluate["env"]["LINGBOT_CASE_SHARD"]
            == ("{{matrix.shard}}/" + str(count))
        )
        if platform == "cuda":
            assert generate["matrix"] == shard_matrix
            # Six videos per evaluator; all 18 cases remain required matrix jobs.
            assert evaluate["timeout_in_minutes"] == 240
            assert keyed[prefix + "-long"]["mirror_hardwares"] == "h100_4_exclusive"
        else:
            assert generate["matrix"] == {"setup": {"shard": [str(i) for i in range(count)], "repeat": ["0", "1", "2"]}}
            assert generate["env"]["LINGBOT_MEASUREMENT_REPEAT"] == "{{matrix.repeat}}"
            assert generate["timeout_in_minutes"] == 240
            assert " collect " in generate["commands"][0]
            assert generate["commands"][0].endswith(
                "lingbot-artifacts/npu-weekly-measurements/shard-{{matrix.shard}}/pair-{{matrix.repeat}}"
            )
            weekly_gate = keyed[prefix + "-gate"]
            assert weekly_gate["matrix"] == shard_matrix
            assert weekly_gate["env"]["LINGBOT_CASE_SHARD"] == "{{matrix.shard}}/6"
            assert {"step": prefix + "-generation", "allow_failure": True} in weekly_gate["depends_on"]
            assert prefix + "-evaluator-image" in weekly_gate["depends_on"]
            assert weekly_gate["mirror_hardwares"] == "l4_1" and not weekly_gate.get("soft_fail")
            assert weekly_gate["timeout_in_minutes"] == 60
            assert (
                weekly_gate["artifact_paths"] and weekly_gate["source_file_dependencies"] == "lingbot_world_regression"
            )
            assert " aggregate " in weekly_gate["commands"][0]
            assert weekly_gate["commands"][0].endswith(
                "lingbot-artifacts/npu-weekly/shard-{{matrix.shard}} "
                "lingbot-artifacts/npu-weekly-measurements/shard-{{matrix.shard}}"
            )
            collection = keyed[prefix + "-long"]
            assert collection["mirror_hardwares"] == "a2b3_npu_8"
            gate = keyed[prefix + "-long-gate"]
            quality = keyed[prefix + "-long-quality"]
            assert collection["matrix"] == {"setup": {"repeat": ["0", "1", "2"]}}
            assert " collect " in collection["commands"][0]
            assert prefix + "-long" in gate["depends_on"]
            assert gate["mirror_hardwares"] == "l4_1" and not gate.get("soft_fail")
            assert {"step": prefix + "-long-gate", "allow_failure": True} in quality["depends_on"]
            assert quality["commands"][0].endswith("lingbot-artifacts/npu-long")
            assert len(gate["commands"][0].split()) == len(quality["commands"][0].split()) == 6
        assert {"step": prefix + "-long" + ("-gate" if platform == "npu" else ""), "allow_failure": True} in keyed[
            prefix + "-long-quality"
        ]["depends_on"]


def test_selected_ci_skip_is_a_failed_pytest_job(tmp_path: Path):
    test = tmp_path / "test_missing_device.py"
    test.write_text('import pytest\ndef test_missing_device():\n    pytest.skip("missing device")\n')
    output = tmp_path / "artifacts"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tests.dfx.lingbot_world.conftest",
            str(test),
            "--rootdir",
            str(tmp_path),
            "--confcutdir",
            str(tmp_path),
            "--lingbot-output",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "BUILDKITE": "true", "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    reports = list(output.glob("*-failure.json"))
    assert reports and "missing device" in reports[0].read_text()


def test_activity_is_not_a_higher_is_better_quality_score():
    result = check_quality(
        [{"dynamic_degree": 0.1}] * 3,
        {"dynamic_degree": 0.9},
        tolerances={"dynamic_degree": 0.1},
        activity_dimensions=("dynamic_degree",),
    )
    assert not result["passed"]


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_every_configured_quality_dimension_rejects_a_collapse(platform):
    limits = json.loads((ROOT / "tests/dfx/lingbot_world/quality.json").read_text())
    # Both pinned evaluators return these dimensions on a [0, 1] scale.
    # A tolerance accidentally expressed as percentage points must not make
    # a dimension incapable of blocking a regression.
    for suite in ("vbench", "wbench"):
        binary_dimensions = set(limits["binary_activity_dimensions"])
        assert binary_dimensions == {"dynamic_degree"}
        tolerances = limits["candidate_tolerances"][platform][suite]
        assert set(tolerances) - binary_dimensions == set(limits["max_reference_spreads"][platform][suite])
        for dimension, tolerance in tolerances.items():
            if dimension in binary_dimensions:
                assert not check_binary_activity([1, 1, 1], [0, 0, 0], tolerance=tolerance)["passed"]
                assert not check_binary_activity([0, 0, 0], [1, 1, 1], tolerance=tolerance)["passed"]
                continue
            result = check_quality(
                [{dimension: 0.9}] * 3,
                {dimension: 0.0},
                tolerances={dimension: tolerance},
                activity_dimensions=("dynamic_degree",),
                max_reference_spreads={dimension: limits["max_reference_spreads"][platform][suite][dimension]},
            )
            assert not result["passed"], (suite, dimension)


@pytest.mark.parametrize("stage", ["aggregate", "evaluate"])
def test_matrix_artifact_handoff_uses_unique_build_prefix(tmp_path, stage):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    generation = "generation/pairs" if stage == "aggregate" else "generation/shard-1"
    expected = (
        [f"{generation}/pair-{i}.json" for i in range(3)] if stage == "aggregate" else [f"{generation}/manifest.json"]
    )
    artifacts = (
        [[path] for path in expected] if stage == "aggregate" else [["generation/shard-0/manifest.json"], expected]
    )
    agent = bin_dir / "buildkite-agent"
    trace = tmp_path / "download.json"
    agent.write_text(
        f"#!{sys.executable}\nimport fnmatch,json,sys\nfrom pathlib import Path\n"
        f"jobs={artifacts!r}\n"
        "paths=jobs[0] if '--step' in sys.argv else [p for job in jobs for p in job]\n"
        "matches=[p for p in paths if fnmatch.fnmatch(p,sys.argv[3])]\n"
        f"Path({str(trace)!r}).write_text(json.dumps({{'args':sys.argv[1:],'paths':matches}}))\n"
        "if not matches: sys.exit(23)\n"
        "for value in matches:\n    p=Path(value); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}')\n"
    )
    consumed = tmp_path / "consumed"
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\nimport os,sys\nfrom pathlib import Path\n"
        "if sys.argv[1:3]==['-m','pytest'] or "
        "sys.argv[1:4]==['-m','tests.dfx.lingbot_world.prepare','evaluators']:\n"
        f"    if not all(Path(p).exists() for p in {expected!r}): sys.exit(29)\n"
        f"    Path({str(consumed)!r}).touch()\n    sys.exit(17)\n"
        "else:\n    os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n"
    )
    for path in (agent, python):
        path.chmod(0o755)
    env = {
        **os.environ,
        "BUILDKITE": "true",
        "PYTHONPATH": str(ROOT),
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
    }
    env.pop("LINGBOT_CASE_SHARD", None)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / ".buildkite/common/scripts/run-lingbot-ci.sh"),
            stage,
            str(ROOT / "tests/dfx/lingbot_world/configs/npu_long.json"),
            str(output),
            generation,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    assert consumed.exists()
    assert json.loads(trace.read_text())["paths"] == expected
    assert ET.parse(output / "pytest.xml").find("testcase/failure") is not None


@pytest.mark.parametrize("case_index", [None, "0", "2"])
def test_quality_launcher_forwards_case_selection(tmp_path: Path, case_index):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    arguments = tmp_path / "pytest-arguments.json"
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\nimport os,sys\n"
        "if sys.argv[1:4] == ['-m','tests.dfx.lingbot_world.prepare','evaluators']: sys.exit(0)\n"
        "if sys.argv[1:2] == ['-c']: print('test-cache'); sys.exit(0)\n"
        "os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n"
    )
    pytest_command = bin_dir / "pytest"
    pytest_command.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
        f"Path({str(arguments)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(17)\n"
    )
    for executable in (python, pytest_command):
        executable.chmod(0o755)
    environment = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
    for name in ("BUILDKITE", "LINGBOT_CASE_SHARD", "LINGBOT_QUALITY_CASE_INDEX"):
        environment.pop(name, None)
    if case_index is not None:
        environment["LINGBOT_QUALITY_CASE_INDEX"] = case_index
    output = tmp_path / "quality"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / ".buildkite/common/scripts/run-lingbot-ci.sh"),
            "evaluate",
            "profile.json",
            str(output),
            "generation",
        ],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    forwarded = [arg for arg in json.loads(arguments.read_text()) if arg.startswith("--lingbot-quality-case-index")]
    assert forwarded == ([] if case_index is None else [f"--lingbot-quality-case-index={case_index}"])
    assert ET.parse(output / "pytest.xml").find("testcase/failure") is not None


@pytest.mark.parametrize("stage", ["generate", "unit"])
def test_launcher_setup_failure_retains_nonzero_exit_and_junit(tmp_path: Path, stage):
    output = tmp_path / "artifacts"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / ".buildkite/common/scripts/run-lingbot-ci.sh"),
            stage,
            str(tmp_path / "missing-profile.json"),
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert ET.parse(output / "pytest.xml").find("testcase/failure") is not None
    assert not json.loads((output / "setup-failure.json").read_text())["passed"]


@pytest.mark.parametrize("stage", ["generate", "collect"])
@pytest.mark.parametrize("preflight_status", [0, 19])
def test_npu_launcher_limits_visibility_before_model_preparation(tmp_path, stage, preflight_status):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    reached = tmp_path / "reference-environment.json"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "if sys.argv[1:4] == ['-m', 'tests.dfx.lingbot_world.prepare', 'devices']:\n"
        f"    if {preflight_status}: sys.exit({preflight_status})\n"
        "    root = Path(sys.argv[sys.argv.index('--root')+1])\n"
        "    (root/'allocation.json').write_text(json.dumps({'visible_devices': '7,6,5,4'}))\n"
        "    print('INFO backend initialized')\n"
        "    print('7,6,5,4')\n"
        "elif sys.argv[1:4] == ['-m', 'tests.dfx.lingbot_world.prepare', 'reference']:\n"
        f"    Path({str(reached)!r}).write_text(json.dumps({{key: os.environ.get(key) for key in "
        "['ASCEND_RT_VISIBLE_DEVICES', 'LINGBOT_ALLOCATION_FILE']}))\n"
        "    sys.exit(17)\n"
        "else:\n    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    output = tmp_path / "artifacts"
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
    env.pop("LINGBOT_CASE_SHARD", None)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / ".buildkite/common/scripts/run-lingbot-ci.sh"),
            stage,
            str(ROOT / "tests/dfx/lingbot_world/configs/npu_nightly.json"),
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == (preflight_status or 17), result.stdout + result.stderr
    assert reached.exists() is (preflight_status == 0)
    if reached.exists():
        environment = json.loads(reached.read_text())
        assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "7,6,5,4"
        assert Path(environment["LINGBOT_ALLOCATION_FILE"]) == output / "allocation.json"
    assert ET.parse(output / "pytest.xml").find("testcase/failure") is not None


@pytest.mark.parametrize("cancel_signal", [signal.SIGTERM, signal.SIGKILL])
def test_launcher_cancellation_retains_failure_junit(tmp_path: Path, cancel_signal):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "if sys.argv[1:4] == ['-m', 'tests.dfx.lingbot_world.prepare', 'devices']:\n"
        "    root = Path(sys.argv[sys.argv.index('--root')+1])\n"
        "    (root/'allocation.json').write_text(json.dumps({'visible_devices':'0,1,2,3'}))\n"
        "    print('0,1,2,3')\n"
        "elif sys.argv[1:4] == ['-m', 'tests.dfx.lingbot_world.prepare', 'reference']:\n"
        "    Path(sys.argv[sys.argv.index('--env-output')+1]).write_text(json.dumps({'model_path':'unused'}))\n"
        "else:\n    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    pytest_cli = bin_dir / "pytest"
    ready = tmp_path / "ready"
    pytest_cli.write_text(
        f"#!{sys.executable}\nimport os, time\nfrom pathlib import Path\n"
        "assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '0,1,2,3'\n"
        "assert Path(os.environ['LINGBOT_ALLOCATION_FILE']).is_file()\n"
        f"Path({str(ready)!r}).touch()\ntime.sleep(120)\n"
    )
    python.chmod(0o755)
    pytest_cli.chmod(0o755)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"platform": "npu", "dies": 4}))
    output = tmp_path / "artifacts"
    process = subprocess.Popen(
        ["bash", str(ROOT / ".buildkite/common/scripts/run-lingbot-ci.sh"), "generate", str(profile), str(output)],
        cwd=ROOT,
        env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "Fake generation did not reach the interruption point"
        os.killpg(process.pid, cancel_signal)
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    assert process.returncode != 0
    assert ET.parse(output / "pytest.xml").find("testcase/failure") is not None
    assert not json.loads((output / "setup-failure.json").read_text())["passed"]


def test_profiles_pin_inputs_and_use_distinct_reference_starts():
    directory = ROOT / "tests/dfx/lingbot_world"
    lock = json.loads((directory / "assets.lock.json").read_text())
    for path in (directory / "configs").glob("*.json"):
        profile = json.loads(path.read_text())
        assert len(profile["reference_revision"]) == 40
        assert profile["model_revision"] == lock["model_revision"]
        assert profile["num_chunks"] > profile["warmup_chunks"] + 1
        assert len({case["id"] for case in profile["cases"]}) == len(profile["cases"])
        assert all(case["image"] in lock["images"] for case in profile["cases"])
        assert (ROOT / profile["deploy_config"]).is_file()
        if profile["cards"] == 4:
            assert profile["reserved_cards"] == 8 and profile["dies"] == 4
    assert len(lock["model_files"]) >= 20


def test_npu_profiles_offload_only_text_encoder_without_reducing_workload():
    from vllm_omni.config.pipeline_registry import resolve_pipeline_config
    from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
    from vllm_omni.diffusion.offloader.config import OffloadMode, parse_diffusion_offload_config

    for name in ("npu_nightly", "npu_weekly", "npu_long"):
        profile = json.loads((ROOT / f"tests/dfx/lingbot_world/configs/{name}.json").read_text())
        deployment = yaml.safe_load((ROOT / profile["deploy_config"]).read_text())
        stage = deployment["stages"][0]
        assert stage["enable_multithread_weight_load"] is False
        resolved = merge_pipeline_deploy(
            resolve_pipeline_config("lingbot_world"), load_deploy_config(ROOT / profile["deploy_config"])
        )
        assert resolved[0].yaml_engine_args["enable_multithread_weight_load"] is False
        offload = parse_diffusion_offload_config(stage["diffusion_offload_config"])
        assert offload is not None and offload.mode == OffloadMode.MODULE
        assert offload.components == {"text_encoder"}
        assert stage["parallel_config"]["sequence_parallel_size"] == 4
        assert (profile["height"], profile["width"]) == (480, 832)
        assert not {"window_chunks", "sink_chunks"} & stage["model_config"]["ar_diffusion_kv_config"].keys()


def opencv_provider_installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import binascii
    from importlib.metadata import PathDistribution
    from types import ModuleType

    from tests.dfx.lingbot_world import opencv_provider

    site = tmp_path / "site"
    module = site / "cv2"
    module.mkdir(parents=True)
    expected = {"__init__.py": b"# headless loader\n", "cv2.abi3.so": b"headless native binary\n"}
    for name, value in expected.items():
        (module / name).write_bytes(value)
    info = site / "opencv_python_headless-4.11.0.86.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: opencv-python-headless\nVersion: 4.11.0.86\n")
    records = []
    for name, value in expected.items():
        digest = binascii.b2a_base64(hashlib.sha256(value).digest(), newline=False).decode().rstrip("=")
        digest = digest.replace("+", "-").replace("/", "_")
        records.append(f"cv2/{name},sha256={digest},{len(value)}")
    (info / "RECORD").write_text("\n".join(records) + "\n")
    distribution = PathDistribution(info)
    lookup = opencv_provider.metadata.distribution
    monkeypatch.setattr(
        opencv_provider.metadata,
        "distribution",
        lambda name: distribution if name == opencv_provider.PROVIDER else lookup(name),
    )
    cv2 = ModuleType("cv2")
    cv2.__dict__.update(
        __file__=str(module / "__init__.py"),
        __version__="4.11.0",
        getBuildInformation=lambda: "canonical headless build",
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return opencv_provider, module, info, cv2


def test_opencv_provider_verifies_imported_files_not_only_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    provider, module, _, _ = opencv_provider_installation(tmp_path, monkeypatch)
    result = provider.verify_opencv_provider("4.11.0.86")
    assert result["provider"] == "opencv-python-headless"
    assert result["provider_version"] == "4.11.0.86"
    assert result["module_path"] == str((module / "__init__.py").resolve())
    assert result["native_module_sha256"] == hashlib.sha256((module / "cv2.abi3.so").read_bytes()).hexdigest()
    assert result["verified_files"] == 2


@pytest.mark.parametrize("overwritten", ["__init__.py", "cv2.abi3.so"])
def test_opencv_provider_rejects_shared_namespace_overwrite(tmp_path, monkeypatch, overwritten):
    provider, module, _, _ = opencv_provider_installation(tmp_path, monkeypatch)
    (module / overwritten).write_bytes(b"GUI distribution overwrote this file; package versions stayed the same")
    with pytest.raises(ValueError, match="provider file mismatch"):
        provider.verify_opencv_provider("4.11.0.86")


def test_opencv_provider_rejects_import_from_another_location(tmp_path, monkeypatch):
    provider, _, _, cv2 = opencv_provider_installation(tmp_path, monkeypatch)
    cv2.__file__ = str(tmp_path / "other/cv2/__init__.py")
    with pytest.raises(ValueError, match="imported module"):
        provider.verify_opencv_provider("4.11.0.86")


def test_opencv_provider_requires_recorded_native_binary(tmp_path, monkeypatch):
    provider, _, info, _ = opencv_provider_installation(tmp_path, monkeypatch)
    record = info / "RECORD"
    record.write_text("\n".join(row for row in record.read_text().splitlines() if "cv2.abi3.so" not in row) + "\n")
    with pytest.raises(ValueError, match="native module"):
        provider.verify_opencv_provider("4.11.0.86")


def test_opencv_provider_rejects_wrong_pinned_version(tmp_path, monkeypatch):
    provider, _, _, _ = opencv_provider_installation(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="provider version"):
        provider.verify_opencv_provider("4.12.0.88")


@pytest.mark.parametrize("field,value", [("provider", "opencv-python"), ("verified_files", 0), ("python", "other")])
def test_opencv_provider_record_rejects_incompatible_interpreter_or_owner(tmp_path, monkeypatch, field, value):
    provider, _, _, _ = opencv_provider_installation(tmp_path, monkeypatch)
    record = provider.verify_opencv_provider("4.11.0.86")
    record[field] = value
    with pytest.raises(ValueError, match="OpenCV provider"):
        provider.validate_opencv_provider_record(record, expected_version="4.11.0.86", python=sys.executable)


@pytest.mark.parametrize("mutation", ["binary", "record"])
def test_ready_evaluator_rejects_changed_opencv_without_reinstall(tmp_path, monkeypatch, mutation):
    from tests.dfx.lingbot_world import prepare

    repository = tmp_path / "repository"
    evaluator_repository(repository)
    monkeypatch.setattr(prepare, "ROOT", repository)
    monkeypatch.setattr(
        prepare,
        "verify_environment",
        lambda *args: {"opencv-python": "4.11.0.86", "opencv-python-headless": "4.11.0.86"},
    )
    root = tmp_path / "ready"
    python = root / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    (root / "environment-ready").write_text(prepare.evaluator_cache_key(repository) + "\n")
    provider = {
        "schema_version": 1,
        "provider": "opencv-python-headless",
        "provider_version": "4.11.0.86",
        "opencv_version": "4.11.0",
        "opencv_build_sha256": "a" * 64,
        "native_module_sha256": "b" * 64,
        "module_path": str(tmp_path / "cv2/__init__.py"),
        "python": str(python),
        "verified_files": 2,
    }
    stored = {**provider, "opencv_build_sha256": "c" * 64} if mutation == "record" else provider
    (root / "opencv-provider.json").write_text(json.dumps(stored))
    original = {p: p.read_bytes() for p in (root / "environment-ready", root / "opencv-provider.json")}
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        if len(command) > 2 and command[1] == "-I":
            if mutation == "binary":
                raise subprocess.CalledProcessError(1, command, stderr="OpenCV provider file mismatch: cv2/cv2.abi3.so")
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(provider))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare.subprocess, "run", probe)
    with pytest.raises(RegressionError, match="OpenCV provider"):
        prepare._prepare_evaluators(root)
    assert not any(command[:2] == ["uv", "pip"] for command in calls)
    assert all(p.read_bytes() == value for p, value in original.items())

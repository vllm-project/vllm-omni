# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.config.omni_config import OmniStageRuntimeConfig
from vllm_omni.engine import cuda_mps
from vllm_omni.engine.stage_runtime import StageRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def controls(monkeypatch):
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.setattr(cuda_mps.shutil, "which", lambda name: "/bin/mps-control")
    run = Mock(return_value=SimpleNamespace(stdout=""))
    monkeypatch.setattr(cuda_mps.subprocess, "run", run)
    return run


def test_private_server_has_scoped_environment_and_idempotent_close(controls):
    original = dict(os.environ)
    server = cuda_mps.CudaMPSServer("GPU-example")
    pipe = Path(server.env["CUDA_MPS_PIPE_DIRECTORY"])
    assert pipe.is_dir()
    assert controls.call_args_list[0].args[0] == ["/bin/mps-control", "-d"]
    assert controls.call_args_list[0].kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-example"
    assert dict(os.environ) == original
    server.close()
    count = controls.call_count
    server.close()
    assert controls.call_count == count
    assert controls.call_args.kwargs["input"] == "quit\n"
    assert not pipe.exists()


def test_operator_daemon_is_checked_but_never_stopped(controls, monkeypatch):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/operator/mps")
    server = cuda_mps.CudaMPSServer("GPU-example")
    server.close()
    assert controls.call_count == 1
    assert controls.call_args.kwargs["input"] == "get_server_list\n"


def test_explicit_pipe_overrides_parent_without_taking_ownership(controls, monkeypatch):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/operator/parent")
    server = cuda_mps.CudaMPSServer("GPU-example", pipe_directory="/operator/stage")

    assert controls.call_args.kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"] == "/operator/stage"
    assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == "/operator/parent"
    server.close()
    assert controls.call_count == 1


def test_explicit_empty_pipe_selects_private_daemon(controls, monkeypatch):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/operator/parent")
    server = cuda_mps.CudaMPSServer("GPU-example", pipe_directory="")
    pipe = server.env["CUDA_MPS_PIPE_DIRECTORY"]
    try:
        assert pipe != "/operator/parent"
        assert controls.call_args_list[0].args[0] == ["/bin/mps-control", "-d"]
        assert controls.call_args_list[0].kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"] == pipe
        assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == "/operator/parent"
    finally:
        server.close()
    assert controls.call_args.kwargs["input"] == "quit\n"
    assert controls.call_args.kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"] == pipe
    assert not Path(pipe).exists()


@pytest.fixture
def mps_runtime(monkeypatch):
    from vllm_omni.engine import stage_runtime

    runtime = StageRuntime.__new__(StageRuntime)
    runtime._mps_servers = {}
    runtime._mps_unavailable_gpus = {}
    monkeypatch.setattr(stage_runtime, "physical_gpu_uuid", lambda device: "GPU-example")
    monkeypatch.setattr(stage_runtime.current_omni_platform, "is_cuda", lambda: True)
    yield runtime
    runtime._close_mps_servers()


@pytest.mark.parametrize("pipe_directory", ["/operator/stage", ""])
def test_runtime_shares_first_stage_pipe_policy(controls, monkeypatch, mps_runtime, pipe_directory):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/operator/parent")
    config = OmniStageRuntimeConfig(cuda_mps=True, env={"CUDA_MPS_PIPE_DIRECTORY": pipe_directory})
    first_env = mps_runtime._mps_environment("0", config)
    selected_pipe = first_env["CUDA_MPS_PIPE_DIRECTORY"]
    if pipe_directory:
        assert selected_pipe == pipe_directory
    else:
        assert selected_pipe != "/operator/parent"
        assert Path(selected_pipe).is_dir()
    call_count = controls.call_count

    assert mps_runtime._mps_environment("0", {"cuda_mps": True}) is first_env
    assert mps_runtime._mps_environment("0", config) is first_env
    assert controls.call_count == call_count
    assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == "/operator/parent"


@pytest.mark.parametrize(
    "first_pipe,conflicting_pipe",
    [("/operator/first", "/operator/other"), ("/operator/first", ""), ("", "/operator/other")],
)
def test_runtime_rejects_conflicting_explicit_stage_pipes(controls, mps_runtime, first_pipe, conflicting_pipe):
    mps_runtime._mps_environment("0", {"cuda_mps": True, "env": {"CUDA_MPS_PIPE_DIRECTORY": first_pipe}})
    call_count = controls.call_count

    with pytest.raises(ValueError, match="Conflicting CUDA_MPS_PIPE_DIRECTORY"):
        mps_runtime._mps_environment("0", {"cuda_mps": True, "env": {"CUDA_MPS_PIPE_DIRECTORY": conflicting_pipe}})

    assert controls.call_count == call_count


def test_failed_startup_cleanup_is_not_reported_as_unavailable(controls):
    launch_error = subprocess.TimeoutExpired(["mps-control", "-d"], 15)
    controls.side_effect = [launch_error, subprocess.TimeoutExpired(["mps-control"], 15)]
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        cuda_mps.CudaMPSServer("GPU-example")
    assert raised.value is launch_error
    assert not isinstance(raised.value, cuda_mps.CudaMPSUnavailableError)
    pipe = Path(controls.call_args.kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"])
    assert pipe.is_dir()
    shutil.rmtree(pipe.parent)


def test_unavailable_gpu_is_not_retried_by_later_stages(controls, mps_runtime):
    controls.side_effect = [subprocess.CalledProcessError(1, ["mps-control", "-d"]), SimpleNamespace(stdout="")]
    first, second = {"cuda_mps": True}, {"cuda_mps": True}
    assert mps_runtime._mps_environment("0", first) == {}
    call_count = controls.call_count
    assert mps_runtime._mps_environment("0", second) == {}
    assert controls.call_count == call_count
    assert first["cuda_mps"] is False and second["cuda_mps"] is False


@pytest.mark.parametrize(
    "first_pipe,conflicting_pipe",
    [("/operator/first", "/operator/other"), ("/operator/first", ""), ("", "/operator/other")],
)
def test_unavailable_gpu_still_rejects_conflicting_stage_pipes(controls, mps_runtime, first_pipe, conflicting_pipe):
    def fail_startup(args, *, input=None, **kwargs):
        if input == "quit\n":
            return SimpleNamespace(stdout="")
        raise subprocess.CalledProcessError(1, args)

    controls.side_effect = fail_startup
    first = {"cuda_mps": True, "env": {"CUDA_MPS_PIPE_DIRECTORY": first_pipe}}
    assert mps_runtime._mps_environment("0", first) == {}
    call_count = controls.call_count

    with pytest.raises(ValueError, match="Conflicting CUDA_MPS_PIPE_DIRECTORY"):
        mps_runtime._mps_environment("0", {"cuda_mps": True, "env": {"CUDA_MPS_PIPE_DIRECTORY": conflicting_pipe}})
    inherited = {"cuda_mps": True}
    assert mps_runtime._mps_environment("0", inherited) == {}
    assert inherited["cuda_mps"] is False
    assert controls.call_count == call_count


def test_startup_failure_closes_only_the_private_daemon(controls):
    controls.side_effect = [SimpleNamespace(stdout=""), RuntimeError("health failed"), SimpleNamespace(stdout="")]
    with pytest.raises(RuntimeError, match="health failed"):
        cuda_mps.CudaMPSServer("GPU-example")
    assert controls.call_args.kwargs["input"] == "quit\n"


def test_runtime_opt_in_and_shared_gpu_ownership(monkeypatch):
    from vllm_omni.engine import stage_runtime

    runtime = StageRuntime.__new__(StageRuntime)
    runtime._mps_servers = {}
    runtime._mps_unavailable_gpus = {}
    create = Mock(
        return_value=SimpleNamespace(
            env={"CUDA_VISIBLE_DEVICES": "GPU-example"}, close=Mock(), operator_pipe_directory=None
        )
    )
    monkeypatch.setattr(stage_runtime, "CudaMPSServer", create)
    monkeypatch.setattr(stage_runtime, "physical_gpu_uuid", lambda device: "GPU-example")
    monkeypatch.setattr(stage_runtime, "current_omni_platform", SimpleNamespace(is_cuda=lambda: True))
    assert runtime._mps_environment("0", {}) == {}
    assert create.call_count == 0
    for _ in range(2):
        assert runtime._mps_environment("0", {"cuda_mps": True}) == {"CUDA_VISIBLE_DEVICES": "GPU-example"}
    assert create.call_count == 1
    runtime._close_mps_servers()
    runtime._close_mps_servers()
    create.return_value.close.assert_called_once()


@pytest.mark.parametrize("devices", [None, "0,1"])
def test_runtime_skips_ambiguous_placement(devices, monkeypatch):
    from vllm_omni.engine import stage_runtime

    runtime = StageRuntime.__new__(StageRuntime)
    monkeypatch.setattr(stage_runtime, "current_omni_platform", SimpleNamespace(is_cuda=lambda: True))
    config = {"cuda_mps": True}
    assert runtime._mps_environment(devices, config) == {}
    assert config["cuda_mps"] is False


def test_runtime_skips_when_mps_control_is_missing(monkeypatch, mps_runtime):
    monkeypatch.setattr(cuda_mps.shutil, "which", lambda name: None)
    config = {"cuda_mps": True}
    assert mps_runtime._mps_environment("0", config) == {}
    assert config["cuda_mps"] is False
    assert mps_runtime._mps_servers == {}


def test_failed_shutdown_preserves_socket_for_retry(controls):
    server = cuda_mps.CudaMPSServer("GPU-example")
    pipe = Path(server.env["CUDA_MPS_PIPE_DIRECTORY"])
    controls.side_effect = RuntimeError("quit failed")
    with pytest.raises(RuntimeError, match="quit failed"):
        server.close()
    assert pipe.is_dir()
    controls.side_effect = None
    server.close()
    assert not pipe.exists()


@pytest.mark.parametrize("launch_mode,stage_type", [("remote", "llm"), ("local", "diffusion")])
def test_mps_skips_unsupported_stages(launch_mode, stage_type):
    runtime = StageRuntime.__new__(StageRuntime)
    metadata = SimpleNamespace(runtime_cfg={"cuda_mps": True}, stage_type=stage_type)
    stage_cfg = SimpleNamespace(runtime=metadata.runtime_cfg)
    plans = [
        SimpleNamespace(replicas=[SimpleNamespace(metadata=metadata, stage_cfg=stage_cfg, launch_mode=launch_mode)])
    ]
    runtime._validate_mps_topology(plans)
    assert metadata.runtime_cfg["cuda_mps"] is False


def test_mps_skips_parallel_initialization():
    runtime = StageRuntime.__new__(StageRuntime)
    runtime._parallel_stage_init = True
    stage_runtime_cfg = {"cuda_mps": True}
    # Legacy Mooncake producers launch from a deepcopy stored on the metadata.
    launch_runtime_cfg = {"cuda_mps": True, "env": {"VLLM_MOONCAKE_BOOTSTRAP_PORT": "8998"}}
    replica = SimpleNamespace(
        metadata=SimpleNamespace(stage_type="llm", runtime_cfg=launch_runtime_cfg),
        stage_cfg=SimpleNamespace(runtime=stage_runtime_cfg),
        launch_mode="local",
    )
    runtime._validate_mps_topology([SimpleNamespace(replicas=[replica])])
    assert stage_runtime_cfg["cuda_mps"] is False
    assert launch_runtime_cfg["cuda_mps"] is False


def test_failed_launch_command_still_checks_private_daemon_cleanup(controls):
    controls.side_effect = [RuntimeError("launch failed after fork"), SimpleNamespace(stdout="")]
    with pytest.raises(RuntimeError, match="launch failed after fork"):
        cuda_mps.CudaMPSServer("GPU-example")
    assert controls.call_args.kwargs["input"] == "quit\n"


@pytest.mark.parametrize(
    "launch_error",
    [subprocess.CalledProcessError(1, ["mps-control", "-d"]), subprocess.TimeoutExpired(["mps-control", "-d"], 15)],
)
def test_runtime_skips_when_private_daemon_cannot_start(controls, mps_runtime, launch_error):
    controls.side_effect = [launch_error, SimpleNamespace(stdout="")]
    config = {"cuda_mps": True}
    assert mps_runtime._mps_environment("0", config) == {}
    assert config["cuda_mps"] is False
    assert mps_runtime._mps_servers == {}
    pipe = Path(controls.call_args.kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"])
    assert controls.call_args.kwargs["input"] == "quit\n"
    assert not pipe.parent.exists()


def test_runtime_skips_unreachable_operator_daemon(controls, mps_runtime):
    controls.side_effect = subprocess.CalledProcessError(1, ["mps-control"])
    config = {"cuda_mps": True, "env": {"CUDA_MPS_PIPE_DIRECTORY": "/operator/stage"}}
    assert mps_runtime._mps_environment("0", config) == {}
    assert config["cuda_mps"] is False
    assert controls.call_count == 1

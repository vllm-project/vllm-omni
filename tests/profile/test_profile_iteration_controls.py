# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Profile controls survive stage dispatch and advance on worker executions."""

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import WorkerProfiler

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.worker.base import OmniGPUWorkerBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class RecordingProfiler(WorkerProfiler):
    """Use the real vLLM scheduling state machine without a device profiler."""

    def __init__(self):
        super().__init__(ProfilerConfig())
        self.events = []

    def _start(self):
        self.events.append("start")

    def _stop(self):
        self.events.append("stop")


@pytest.mark.parametrize("profiler_overrides", [None, {"record_shapes": False}])
@pytest.mark.parametrize("route", ["inline", "subprocess"])
@pytest.mark.parametrize(
    "controls",
    [
        {},
        {"delay_iterations": 2},
        {"max_iterations": 3},
        {"delay_iterations": 2, "max_iterations": 3},
        {"delay_iterations": 0, "max_iterations": 0},
    ],
)
async def test_frontend_profile_controls_reach_diffusion_profiler(route, controls, profiler_overrides, mocker):
    profiler = RecordingProfiler()
    profiler.start = mocker.Mock(wraps=profiler.start)
    profiler.set_profiler_kwargs = mocker.Mock(wraps=profiler.set_profiler_kwargs)
    worker = SimpleNamespace(_get_profiler=lambda: profiler)

    def worker_rpc(*, method, args, kwargs=None):
        assert method == "profile"
        return DiffusionWorker.profile(worker, *args, **(kwargs or {}))

    async def async_worker_rpc(**rpc_kwargs):
        return worker_rpc(**rpc_kwargs)

    engine = SimpleNamespace(collective_rpc=worker_rpc, async_collective_rpc=async_worker_rpc)
    for name in ("_profile_args", "_profile_failed", "profile", "async_profile"):
        setattr(engine, name, partial(getattr(DiffusionEngine, name), engine))
    with ThreadPoolExecutor(max_workers=1) as executor:
        client = SimpleNamespace(_engine=engine, _executor=executor, stage_id=1, replica_id=0)

        async def dispatch(*, method, args, kwargs=None, stage_ids=None):
            assert stage_ids == [1]
            if route == "inline":
                return await InlineStageDiffusionClient.collective_rpc_async(client, method, args=args, kwargs=kwargs)
            return await StageDiffusionProc._handle_collective_rpc(client, method, None, args, kwargs or {})

        frontend = SimpleNamespace(collective_rpc=dispatch)
        await AsyncOmni.start_profile(
            frontend, "test-profile", stages=[1], profiler_kwargs=profiler_overrides, **controls
        )
        profiler.set_profiler_kwargs.assert_called_once_with(profiler_overrides)
        profiler.start.assert_called_once_with(**controls)
        assert profiler._delay_iters == controls.get("delay_iterations", 0)
        assert profiler._max_iters == controls.get("max_iterations", 0)
        await AsyncOmni.stop_profile(frontend, stages=[1])
        assert not profiler.is_running


@pytest.mark.parametrize("controls", [{"delay_iterations": -1}, {"max_iterations": -1}])
async def test_invalid_controls_fail_before_stage_dispatch(controls, mocker):
    frontend = SimpleNamespace(collective_rpc=mocker.AsyncMock())
    with pytest.raises(ValueError):
        await AsyncOmni.start_profile(frontend, **controls)
    frontend.collective_rpc.assert_not_called()


def _npu_worker_class(monkeypatch):
    # Only the external Ascend base is unavailable on CPU; load the real Omni
    # worker method without importing torch_npu or instantiating a device worker.
    ascend_worker = ModuleType("vllm_ascend.worker.worker")
    monkeypatch.setattr(ascend_worker, "NPUWorker", object, raising=False)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", ascend_worker)
    path = Path(__file__).resolve().parents[2] / "vllm_omni/platforms/npu/worker/base.py"
    spec = importlib.util.spec_from_file_location("_profile_test_npu_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OmniNPUWorkerBase


@pytest.mark.parametrize("profiler_overrides", [None, {"record_shapes": False}])
@pytest.mark.parametrize("kind", ["ar", "diffusion", "npu"])
@pytest.mark.parametrize("controls", [{}, {"delay_iterations": 2, "max_iterations": 3}])
def test_worker_profile_forwards_controls_and_preserves_default_call(
    monkeypatch, mocker, kind, controls, profiler_overrides
):
    profiler = RecordingProfiler()
    profiler.start = mocker.Mock(wraps=profiler.start)
    profiler.stop = mocker.Mock(wraps=profiler.stop)
    profiler.set_profiler_kwargs = mocker.Mock(wraps=profiler.set_profiler_kwargs)
    worker = SimpleNamespace(profiler=profiler, _get_profiler=lambda: profiler)
    cls = {"ar": OmniGPUWorkerBase, "diffusion": DiffusionWorker}.get(kind)
    if cls is None:
        cls = _npu_worker_class(monkeypatch)
    cls.profile(worker, True, "test-profile", profiler_kwargs=profiler_overrides, **controls)
    profiler.start.assert_called_once_with(**controls)
    profiler.set_profiler_kwargs.assert_called_once_with(profiler_overrides)
    cls.profile(worker, False)
    profiler.stop.assert_called_once_with()


@pytest.mark.parametrize("execution", ["execute_model", "execute_model_batch", "execute_stepwise"])
def test_diffusion_execution_advances_native_iteration_schedule(execution, mocker):
    profiler = RecordingProfiler()
    worker = object.__new__(DiffusionWorker)
    worker.profiler = profiler
    worker.lora_manager = None
    worker._step_lora_state = {}
    worker.model_runner = SimpleNamespace(**{execution: mocker.Mock(return_value=object())})
    scheduler_output = SimpleNamespace(scheduled_new_reqs=[], scheduled_request_ids=[], finished_req_ids=[])
    if execution == "execute_stepwise":
        execute = partial(worker.execute_stepwise, scheduler_output)
    elif execution == "execute_model_batch":
        execute = partial(worker.execute_model_batch, scheduler_output, None)
    else:
        execute = partial(worker.execute_model, SimpleNamespace(), None)

    worker.profile(delay_iterations=2, max_iterations=1)
    execute()
    assert profiler.events == []
    execute()
    assert profiler.events == ["start"]
    execute()
    assert profiler.events == ["start", "stop"]
    assert not profiler.is_running

    # Auto-stop must leave the profiler reusable, and omitted overrides restore
    # the configured defaults rather than retaining the preceding session caps.
    worker.profile()
    assert profiler.events == ["start", "stop", "start"]
    for _ in range(4):
        execute()
    assert profiler.is_running
    worker.profile(False)
    assert profiler.events == ["start", "stop", "start", "stop"]


def test_npu_profiler_constructor_overrides_and_defaults(monkeypatch, mocker, tmp_path):
    from vllm_omni.platforms.npu.profiler import NPUTorchProfilerWrapper

    torch_npu = ModuleType("torch_npu")
    backend = mocker.Mock()
    monkeypatch.setattr(torch_npu, "profiler", backend, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_record_shapes=True,
        torch_profiler_with_stack=True,
    )
    profiler = NPUTorchProfilerWrapper(config, "stage1_rank0", 0)
    defaults = backend.profile.call_args.kwargs
    assert defaults["record_shapes"] is True
    assert defaults["with_modules"] is True
    assert defaults["activities"] == [backend.ProfilerActivity.CPU, backend.ProfilerActivity.NPU]
    handler = mocker.Mock()
    profiler.set_profiler_kwargs({"record_shapes": False, "with_modules": False, "on_trace_ready": handler})
    overrides = backend.profile.call_args.kwargs
    assert overrides["record_shapes"] is False
    assert overrides["with_modules"] is False
    assert overrides["on_trace_ready"] is handler
    assert overrides["experimental_config"] is not None
    profiler.set_profiler_kwargs(None)
    restored = backend.profile.call_args.kwargs
    assert restored["record_shapes"] is True
    assert restored["with_modules"] is True
    assert restored["on_trace_ready"] is defaults["on_trace_ready"]

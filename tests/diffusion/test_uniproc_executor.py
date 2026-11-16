# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Single-GPU (uniproc) diffusion executor: selection and RPC behaviour."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeODConfig:
    def __init__(self, num_gpus: int = 1, backend: str | None = None) -> None:
        self.num_gpus = num_gpus
        self.distributed_executor_backend = backend
        self.worker_extension_cls = None
        self.custom_pipeline_args = None


def _od_config(num_gpus: int = 1, backend: str | None = None) -> _FakeODConfig:
    return _FakeODConfig(num_gpus=num_gpus, backend=backend)


def test_single_gpu_defaults_to_uniproc():
    assert DiffusionExecutor.get_class(_od_config(num_gpus=1)) is UniProcDiffusionExecutor


def test_explicit_mp_is_honored_on_single_gpu():
    assert DiffusionExecutor.get_class(_od_config(num_gpus=1, backend="mp")) is MultiprocDiffusionExecutor


def test_multi_gpu_defaults_to_multiproc():
    assert DiffusionExecutor.get_class(_od_config(num_gpus=2)) is MultiprocDiffusionExecutor


def test_explicit_uni_backend_is_honored():
    assert DiffusionExecutor.get_class(_od_config(num_gpus=1, backend="uni")) is UniProcDiffusionExecutor


@pytest.fixture
def executor(monkeypatch: pytest.MonkeyPatch):
    """A ``UniProcDiffusionExecutor`` with a mocked worker (no model load)."""
    worker = MagicMock()
    monkeypatch.setattr(
        "vllm_omni.diffusion.executor.uniproc_executor.current_omni_platform.get_diffusion_worker_cls",
        MagicMock(return_value="vllm_omni.diffusion.worker.diffusion_worker.DiffusionWorker"),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.executor.uniproc_executor.resolve_obj_by_qualname",
        MagicMock(return_value=object),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.WorkerWrapperBase",
        MagicMock(return_value=worker),
    )
    return UniProcDiffusionExecutor(_od_config(num_gpus=1)), worker


def test_rejects_multi_gpu_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.executor.uniproc_executor.current_omni_platform.get_diffusion_worker_cls",
        MagicMock(return_value="unused"),
    )
    with pytest.raises(ValueError, match="single GPU only"):
        UniProcDiffusionExecutor(_od_config(num_gpus=4))


def test_shutdown_is_safe_on_partially_constructed_executor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.executor.uniproc_executor.current_omni_platform.get_diffusion_worker_cls",
        MagicMock(return_value="unused"),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.executor.uniproc_executor.resolve_obj_by_qualname",
        MagicMock(return_value=object),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.WorkerWrapperBase",
        MagicMock(side_effect=RuntimeError("model load blew up")),
    )
    constructed = UniProcDiffusionExecutor.__new__(UniProcDiffusionExecutor)
    with pytest.raises(RuntimeError, match="model load blew up"):
        constructed.__init__(_od_config(num_gpus=1))

    constructed.shutdown()


def test_collective_rpc_calls_worker_directly(executor):
    exec_, worker = executor
    worker.execute_method.return_value = "result"

    out = exec_.collective_rpc("some_method", args=(1, 2), kwargs={"k": "v"}, unique_reply_rank=0)

    assert out == "result"
    worker.execute_method.assert_called_once_with("some_method", 1, 2, k="v")


def test_collective_rpc_returns_list_when_no_reply_rank(executor):
    exec_, worker = executor
    worker.execute_method.return_value = "result"

    assert exec_.collective_rpc("some_method") == ["result"]


def test_collective_rpc_propagates_worker_exceptions(executor):
    exec_, worker = executor
    worker.execute_method.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        exec_.collective_rpc("some_method", unique_reply_rank=0)


def test_recoverable_worker_failure_keeps_the_executor_alive(executor, monkeypatch):
    exec_, worker = executor
    worker.execute_method.side_effect = RuntimeError("CUDA out of memory")
    monkeypatch.setattr(exec_, "_device_is_usable", lambda: True)
    died = MagicMock()
    exec_.register_failure_callback(died)

    with pytest.raises(RuntimeError):
        exec_.collective_rpc("some_method", unique_reply_rank=0)

    assert exec_.is_dead is False
    died.assert_not_called()
    exec_.check_health()


def test_poisoned_cuda_context_marks_the_executor_dead(executor, monkeypatch):
    exec_, worker = executor
    worker.execute_method.side_effect = RuntimeError("an illegal memory access was encountered")
    monkeypatch.setattr(exec_, "_device_is_usable", lambda: False)
    died = MagicMock()
    exec_.register_failure_callback(died)

    with pytest.raises(RuntimeError):
        exec_.collective_rpc("some_method", unique_reply_rank=0)

    assert exec_.is_dead is True
    died.assert_called_once_with()
    with pytest.raises(EngineDeadError):
        exec_.check_health()


def test_failure_is_latched_and_callbacks_fire_once(executor, monkeypatch):
    exec_, worker = executor
    worker.execute_method.side_effect = RuntimeError("boom")
    monkeypatch.setattr(exec_, "_device_is_usable", lambda: False)
    died = MagicMock()
    exec_.register_failure_callback(died)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            exec_.collective_rpc("some_method", unique_reply_rank=0)

    died.assert_called_once_with()


def test_device_probe_is_skipped_when_cuda_was_never_initialized(executor):
    exec_, worker = executor
    worker.execute_method.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        exec_.collective_rpc("some_method", unique_reply_rank=0)

    if not torch.cuda.is_initialized():
        assert exec_.is_dead is False


def test_device_probe_skipped_when_accelerator_unavailable(executor, monkeypatch):
    exec_, worker = executor
    worker.execute_method.side_effect = RuntimeError("boom")
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    sync = MagicMock()
    monkeypatch.setattr(torch.accelerator, "synchronize", sync)

    with pytest.raises(RuntimeError, match="boom"):
        exec_.collective_rpc("some_method", unique_reply_rank=0)

    sync.assert_not_called()
    assert exec_.is_dead is False


def test_device_probe_skipped_when_device_not_initialized(executor, monkeypatch):
    exec_, worker = executor
    worker.execute_method.side_effect = RuntimeError("boom")
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: SimpleNamespace(type="npu"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_initialized=lambda: False), raising=False)
    sync = MagicMock()
    monkeypatch.setattr(torch.accelerator, "synchronize", sync)

    with pytest.raises(RuntimeError, match="boom"):
        exec_.collective_rpc("some_method", unique_reply_rank=0)

    sync.assert_not_called()
    assert exec_.is_dead is False


def test_device_probe_latches_sticky_fault_on_npu(executor, monkeypatch):
    """NPU must not short-circuit the probe the way a CUDA-only guard did."""
    exec_, worker = executor
    worker.execute_method.side_effect = RuntimeError("boom")
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: SimpleNamespace(type="npu"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_initialized=lambda: True), raising=False)
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        MagicMock(side_effect=RuntimeError("NPU context poisoned")),
    )
    died = MagicMock()
    exec_.register_failure_callback(died)

    with pytest.raises(RuntimeError, match="boom"):
        exec_.collective_rpc("some_method", unique_reply_rank=0)

    assert exec_.is_dead is True
    died.assert_called_once_with()


def test_check_health_ok_then_dead(executor):
    exec_, _ = executor
    exec_.check_health()

    exec_._is_failed = True
    with pytest.raises(EngineDeadError):
        exec_.check_health()


def test_shutdown_is_idempotent_and_closes_executor(executor, monkeypatch):
    exec_, worker = executor
    collect = MagicMock()
    empty_cache = MagicMock()
    monkeypatch.setattr("vllm_omni.diffusion.executor.uniproc_executor.gc.collect", collect)
    monkeypatch.setattr(
        "vllm_omni.diffusion.executor.uniproc_executor.current_omni_platform.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.executor.uniproc_executor.current_omni_platform.empty_cache",
        empty_cache,
    )
    exec_.register_failure_callback(MagicMock())

    exec_.shutdown()
    exec_.shutdown()

    worker.shutdown.assert_called_once()
    assert exec_.driver_worker is None
    assert exec_._failure_callbacks == []
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        exec_.collective_rpc("some_method", unique_reply_rank=0)


def _single_request_wave(request_id: str = "req-1"):
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                request_id=request_id,
                req=SimpleNamespace(request_id=request_id),
                diffusion_kv_metadata=None,
            )
        ],
        kv_prefetch_job=None,
    )


def test_execute_request_keeps_client_error_status(executor, monkeypatch):
    """A pipeline's OmniClientError must keep its 4xx status on a single GPU.

    The worker runs inline, so the exception reaches execute_request directly.
    Flattening it to DiffusionOutput(error=str(exc)) made the engine rebuild a
    RuntimeError and the API answer HTTP 500 for a request validation error.
    """
    from vllm_omni.errors import OmniClientError

    exec_, worker = executor
    worker.execute_method.side_effect = OmniClientError("clip exceeds the frame budget", status_code=422)
    monkeypatch.setattr(exec_, "_device_is_usable", lambda: True)

    batch = exec_.execute_request(_single_request_wave())

    (runner_output,) = batch.runner_outputs
    assert runner_output.finished is True
    assert runner_output.result.error == "clip exceeds the frame budget"
    assert runner_output.result.error_status_code == 422
    assert runner_output.result.error_type == "BadRequestError"
    assert exec_.is_dead is False


def test_execute_request_server_error_has_no_client_status(executor, monkeypatch):
    exec_, worker = executor
    worker.execute_method.side_effect = RuntimeError("CUDA out of memory")
    monkeypatch.setattr(exec_, "_device_is_usable", lambda: True)

    batch = exec_.execute_request(_single_request_wave())

    (runner_output,) = batch.runner_outputs
    assert runner_output.result.error == "CUDA out of memory"
    assert runner_output.result.error_status_code is None
    assert runner_output.result.error_type is None


@pytest.mark.parametrize(
    "error,expected_status,expected_type",
    [
        (OmniClientError("invalid schedule"), 400, "BadRequestError"),
        (OmniClientError("rejected", status_code=422, error_type="ScheduleError"), 422, "ScheduleError"),
        (RuntimeError("worker failed"), None, None),
        (ValueError("ordinary value error"), None, None),
    ],
)
def test_execute_request_keeps_error_metadata_and_next_request(
    executor, monkeypatch, error, expected_status, expected_type
):
    exec_, worker = executor
    monkeypatch.setattr(exec_, "_device_is_usable", lambda: True)
    good = DiffusionOutput(output="ok")
    worker.execute_method.side_effect = [error, good]
    scheduler_output = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[
            NewRequestData(
                request_id=name,
                req=OmniDiffusionRequest(
                    request_id=name,
                    prompt={"prompt": "a test"},
                    sampling_params=OmniDiffusionSamplingParams(num_inference_steps=4),
                ),
            )
            for name in ("bad", "good")
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=2,
        num_waiting_reqs=0,
    )

    result = exec_.execute_request(scheduler_output)

    bad, healthy = result.runner_outputs
    assert bad.request_id == "bad"
    assert bad.finished is True
    assert bad.result.error == str(error)
    assert bad.result.error_status_code == expected_status
    assert bad.result.error_type == expected_type
    assert healthy.request_id == "good"
    assert healthy.result is good
    assert exec_.is_dead is False

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-GPU (uniproc) diffusion executor: selection and RPC behaviour."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor

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

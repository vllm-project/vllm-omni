# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.metadata import (
    DiffusionKVMetadata,
    DiffusionKVSequenceMetadata,
)
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerWrapperBase

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def make_metadata(request_id: str = "req-0") -> DiffusionKVMetadata:
    return DiffusionKVMetadata(
        request_id=request_id,
        allocation_generation=1,
        sequences=(
            DiffusionKVSequenceMetadata(
                sequence_id=0,
                prefix_len=4,
                target_len=2,
                seq_len=8,
                block_ids=([1, 2],),
            ),
        ),
    )


def make_runner(mode: DiffusionKVCacheMode) -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = SimpleNamespace(diffusion_kv_mode=mode)
    return runner


def make_scheduler_output(*new_reqs: NewRequestData) -> DiffusionSchedulerOutput:
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=len(new_reqs),
        num_waiting_reqs=0,
    )


def make_executor() -> tuple[MultiprocDiffusionExecutor, list[tuple]]:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(
        enable_distributed_layerwise_offload=False,
        dlo_use_allgather=True,
    )
    executor._ensure_open = lambda: None
    calls: list[tuple] = []

    def collective_rpc(method, *, args, unique_reply_rank, exec_all_ranks):
        calls.append((method, args, unique_reply_rank, exec_all_ranks))
        return DiffusionOutput(output=None)

    executor.collective_rpc = collective_rpc
    return executor, calls


def test_new_request_data_carries_scheduler_allocation_atomically() -> None:
    req = SimpleNamespace(request_id="req-0")
    state = SimpleNamespace(request_id="req-0", req=req)
    metadata = make_metadata()

    new_req = NewRequestData.from_state(state, diffusion_kv_metadata=metadata)

    assert new_req.req is req
    assert new_req.diffusion_kv_metadata is metadata


def test_paged_request_without_metadata_fails_before_request_forward() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner._execute_request_list = Mock()
    req = SimpleNamespace(request_id="req-0")

    with pytest.raises(ValueError, match="requires Diffusion KV metadata"):
        runner.execute_model(req)

    runner._execute_request_list.assert_not_called()


def test_dense_request_rejects_metadata_before_request_forward() -> None:
    runner = make_runner(DiffusionKVCacheMode.DENSE_LEGACY)
    runner._execute_request_list = Mock()

    with pytest.raises(ValueError, match="dense_legacy.*must not carry"):
        runner.execute_model(
            SimpleNamespace(request_id="req-0"),
            diffusion_kv_metadata=make_metadata(),
        )

    runner._execute_request_list.assert_not_called()


def test_paged_request_rejects_mismatched_metadata_identity() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner._execute_request_list = Mock()

    with pytest.raises(ValueError, match="request mismatch"):
        runner.execute_model(
            SimpleNamespace(request_id="req-0"),
            diffusion_kv_metadata=make_metadata("stale"),
        )

    runner._execute_request_list.assert_not_called()


def test_request_rpc_rejects_envelope_request_identity_mismatch() -> None:
    executor, calls = make_executor()
    req = SimpleNamespace(request_id="actual-request-id")
    new_req = NewRequestData(
        request_id="envelope-id",
        req=req,
        diffusion_kv_metadata=make_metadata("envelope-id"),
    )

    with pytest.raises(ValueError, match="request identity mismatch"):
        executor.execute_request(make_scheduler_output(new_req))

    assert calls == []


def test_batch_path_rejects_missing_metadata_before_forward() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner._execute_request_list = Mock()
    new_req = NewRequestData(
        request_id="req-0",
        req=SimpleNamespace(request_id="req-0"),
    )

    with pytest.raises(ValueError, match="requires Diffusion KV metadata"):
        runner.execute_model_batch(make_scheduler_output(new_req), runner.od_config)

    runner._execute_request_list.assert_not_called()


def test_batch_path_rejects_envelope_request_identity_mismatch() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner._execute_request_list = Mock()
    new_req = NewRequestData(
        request_id="envelope-id",
        req=SimpleNamespace(request_id="actual-request-id"),
        diffusion_kv_metadata=make_metadata("envelope-id"),
    )

    with pytest.raises(ValueError, match="request identity mismatch"):
        runner.execute_model_batch(make_scheduler_output(new_req), runner.od_config)

    runner._execute_request_list.assert_not_called()


def test_step_path_rejects_missing_metadata_before_step_support_check() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.pipeline = object()
    runner._supports_step_mode = Mock(return_value=False)
    new_req = NewRequestData(
        request_id="req-0",
        req=SimpleNamespace(request_id="req-0"),
    )

    with pytest.raises(ValueError, match="requires Diffusion KV metadata"):
        runner.execute_stepwise(make_scheduler_output(new_req))

    runner._supports_step_mode.assert_not_called()


def test_step_path_rejects_envelope_request_identity_mismatch() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.pipeline = object()
    runner._supports_step_mode = Mock()
    new_req = NewRequestData(
        request_id="envelope-id",
        req=SimpleNamespace(request_id="actual-request-id"),
        diffusion_kv_metadata=make_metadata("envelope-id"),
    )

    with pytest.raises(ValueError, match="request identity mismatch"):
        runner.execute_stepwise(make_scheduler_output(new_req))

    runner._supports_step_mode.assert_not_called()


def test_dense_request_rpc_keeps_legacy_positional_shape() -> None:
    executor, calls = make_executor()
    prepared_layout = object()
    req = SimpleNamespace(request_id="req-0", prepared_layout=prepared_layout)
    new_req = NewRequestData(request_id="req-0", req=req)

    result = executor.execute_request(make_scheduler_output(new_req))

    assert result.request_ids == ["req-0"]
    assert calls == [
        (
            "execute_model",
            (req, executor.od_config, None),
            0,
            True,
        )
    ]
    assert calls[0][1][0].prepared_layout is prepared_layout


def test_request_rpc_sends_attached_metadata_for_only_that_request() -> None:
    executor, calls = make_executor()
    req_0 = SimpleNamespace(request_id="req-0")
    req_1 = SimpleNamespace(request_id="req-1")
    metadata_0 = make_metadata("req-0")
    metadata_1 = make_metadata("req-1")
    new_reqs = (
        NewRequestData(request_id="req-0", req=req_0, diffusion_kv_metadata=metadata_0),
        NewRequestData(request_id="req-1", req=req_1, diffusion_kv_metadata=metadata_1),
    )

    result = executor.execute_request(make_scheduler_output(*new_reqs))

    assert result.request_ids == ["req-0", "req-1"]
    assert [call[1] for call in calls] == [
        (req_0, executor.od_config, None, metadata_0),
        (req_1, executor.od_config, None, metadata_1),
    ]


def test_worker_wrapper_only_extends_paged_request_rpc() -> None:
    calls: list[tuple] = []

    class Worker:
        def execute_model(self, req, od_config, **kwargs):
            calls.append((req, od_config, kwargs))
            return DiffusionOutput(output=None)

    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.worker = Worker()
    req = SimpleNamespace(request_id="req-0")
    od_config = SimpleNamespace()
    metadata = make_metadata()

    wrapper.execute_model(req, od_config)
    wrapper.execute_model(req, od_config, diffusion_kv_metadata=metadata)

    assert calls == [
        (req, od_config, {"kv_prefetch_job": None}),
        (
            req,
            od_config,
            {
                "kv_prefetch_job": None,
                "diffusion_kv_metadata": metadata,
            },
        ),
    ]


def test_dense_worker_call_does_not_extend_model_runner_signature() -> None:
    calls: list[tuple] = []

    class DenseModelRunner:
        def execute_model(self, req, kv_prefetch_job=None):
            calls.append((req, kv_prefetch_job))
            return DiffusionOutput(output=None)

    worker = object.__new__(DiffusionWorker)
    worker.model_runner = DenseModelRunner()
    worker.lora_manager = None
    worker._get_profiler = lambda: None
    req = SimpleNamespace(request_id="req-0")

    worker.execute_model(req, SimpleNamespace())

    assert calls == [(req, None)]


def test_dlo_worker_selects_request_and_metadata_from_same_envelope(monkeypatch) -> None:
    calls: list[tuple] = []

    class ModelRunner:
        def execute_model(self, req, **kwargs):
            calls.append((req, kwargs))
            return DiffusionOutput(output=None)

    worker = object.__new__(DiffusionWorker)
    worker.model_runner = ModelRunner()
    worker.lora_manager = None
    worker._get_profiler = lambda: None
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_data_parallel_rank",
        lambda: 1,
    )
    req_0 = SimpleNamespace(request_id="req-0")
    req_1 = SimpleNamespace(request_id="req-1")
    metadata_0 = make_metadata("req-0")
    metadata_1 = make_metadata("req-1")
    envelopes = [
        NewRequestData(request_id="req-0", req=req_0, diffusion_kv_metadata=metadata_0),
        NewRequestData(request_id="req-1", req=req_1, diffusion_kv_metadata=metadata_1),
    ]

    result = worker.execute_model(envelopes, SimpleNamespace())

    assert calls == [
        (
            req_1,
            {
                "kv_prefetch_job": None,
                "diffusion_kv_metadata": metadata_1,
            },
        )
    ]
    assert result["dp_rank"] == 1

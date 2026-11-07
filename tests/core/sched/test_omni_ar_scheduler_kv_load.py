# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 receive completion and failure propagation into the Omni scheduler.

Exercise the real connector poll and scheduler update methods with simulated
transfer results. These CPU tests do not read external caches or run a model.
"""

from types import MethodType, SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata, KVConnectorTransferResults
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.gpu import kv_connector as runner_connector

from tests.core.sched.test_omni_ar_scheduler_logprobs import _make_scheduler_stub
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def pending_load(mocker, monkeypatch):
    request = Request("remote-load", list(range(65)), SamplingParams(max_tokens=1), pooling_params=None)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_computed_tokens = 64
    scheduler = _make_scheduler_stub([request])
    scheduler.running = []
    scheduler.kv_holding_waiting.append(request)
    scheduler.recompute_kv_load_failures = True
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.prefix_replay_tokens = 0
    scheduler.prefix_replay_group_ids = []
    scheduler._kv_fetch_stages = None
    scheduler.vllm_config.model_config.use_v2_model_runner = True
    scheduler.kv_cache_manager.cache_blocks = mocker.Mock()
    scheduler.kv_cache_manager.free = mocker.Mock()
    scheduler.connector = mocker.Mock()
    scheduler.connector.get_kv_connector_stats.return_value = None
    scheduler.connector.take_events.return_value = None
    for name in (
        "_handle_failed_recving",
        "_handle_blocked_waiting_request",
        "_update_waiting_for_remote_kv",
        "_update_from_kv_xfer_finished",
        "_set_kv_fetch_stage",
        "_mark_prefix_replay",
        "_load_restores_replay_window",
    ):
        setattr(scheduler, name, MethodType(getattr(OmniARScheduler, name), scheduler))

    transfer = mocker.Mock()
    transfer.get_transfer_results.return_value = KVConnectorTransferResults()
    transfer.get_block_ids_with_load_errors.return_value = set()
    transfer.get_kv_connector_stats.return_value = None
    transfer.get_kv_connector_kv_cache_events.return_value = None
    transfer.build_connector_worker_meta.return_value = None
    worker = runner_connector.ActiveKVConnector.__new__(runner_connector.ActiveKVConnector)
    worker.kv_connector = transfer
    worker._disabled = False
    worker._pending_load_kwargs = None
    # Avoid creating a CUDA forward context; the poll and result propagation
    # below remain the real MRv2 methods.
    monkeypatch.setattr(runner_connector, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(runner_connector, "get_forward_context", lambda: None)
    step = SchedulerOutput.make_empty()
    step.kv_connector_metadata = KVConnectorMetadata()
    return SimpleNamespace(request=request, scheduler=scheduler, worker=worker, transfer=transfer, step=step)


def _poll(load, result):
    load.transfer.get_transfer_results.return_value = result
    worker_output = load.worker.no_forward(load.step)
    assert worker_output.sampled_token_ids == []
    outputs = OmniARScheduler.update_from_output(load.scheduler, load.step, worker_output)
    assert outputs == {}
    assert list(load.request.output_token_ids) == []
    return worker_output.kv_connector_output


@pytest.mark.parametrize("loaded_tokens", [32, 64, 65])
def test_mrv2_waits_for_receive_completion(pending_load, loaded_tokens):
    """No request is readmitted until the connector reports receive completion."""
    load = pending_load
    load.request.num_computed_tokens = loaded_tokens
    for _ in range(3):
        _poll(load, KVConnectorTransferResults())
        assert not load.scheduler._handle_blocked_waiting_request(load.request)
        assert load.request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        load.scheduler.kv_cache_manager.cache_blocks.assert_not_called()

    _poll(load, KVConnectorTransferResults(finished_recving={"remote-load"}))
    assert load.scheduler._handle_blocked_waiting_request(load.request)
    assert load.request.status == RequestStatus.WAITING
    assert load.request.num_computed_tokens == min(loaded_tokens, 64)
    load.scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(load.request, loaded_tokens)
    load.scheduler.kv_cache_manager.free.assert_not_called()
    assert load.scheduler.finished_recving_kv_req_ids == set()


def test_failed_receive_resets_prefix_before_readmission(pending_load):
    """A request-level failure invalidates the reused prefix before retry."""
    load = pending_load
    result = _poll(load, KVConnectorTransferResults(failed_recving={"remote-load"}, finished_recving={"remote-load"}))
    assert result.failed_recving == {"remote-load"}
    assert load.request.num_computed_tokens == 0
    assert load.request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert load.scheduler.failed_recving_kv_req_ids == {"remote-load"}
    load.scheduler.kv_cache_manager.free.assert_not_called()

    assert load.scheduler._handle_blocked_waiting_request(load.request)
    assert load.request.status == RequestStatus.WAITING
    assert load.request.num_tokens - load.request.num_computed_tokens == 65
    load.scheduler.kv_cache_manager.free.assert_called_once_with(load.request)
    load.scheduler.kv_cache_manager.cache_blocks.assert_not_called()
    assert load.scheduler.failed_recving_kv_req_ids == set()
    assert load.scheduler.finished_recving_kv_req_ids == set()


def test_failure_without_completion_does_not_release_buffers(pending_load):
    """Failure alone must not release blocks while an asynchronous copy runs."""
    load = pending_load
    _poll(load, KVConnectorTransferResults(failed_recving={"remote-load"}))
    assert load.request.num_computed_tokens == 0
    assert not load.scheduler._handle_blocked_waiting_request(load.request)
    load.scheduler.kv_cache_manager.free.assert_not_called()

    _poll(load, KVConnectorTransferResults(finished_recving={"remote-load"}))
    assert load.scheduler._handle_blocked_waiting_request(load.request)
    load.scheduler.kv_cache_manager.free.assert_called_once_with(load.request)

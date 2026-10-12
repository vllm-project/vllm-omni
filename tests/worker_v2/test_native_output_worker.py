# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_omni.worker_v2.omni_ar_model_runner as ar_runner
from tests.worker_v2.test_lychee_history_recovery import _history, _request, _runner
from tests.worker_v2.test_lychee_model_state import _session_request, _session_state
from tests.worker_v2.test_omni_ar_model_runner import _async_output
from vllm_omni.worker_v2.native_output_worker import NativeOutputWorker, OwnerAsyncOutput
from vllm_omni.worker_v2.output_snapshot import RequestOutputSnapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("transport", ["native", "direct"])
@pytest.mark.parametrize("failure_phase", ["aux", "snapshot_finalizer", "output_finalizer", "snapshot_partition"])
def test_host_materialization_failure_recovers_original_batch_on_owner_preserving_newer_batch(
    monkeypatch, failure_phase, transport
):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    state = _session_state(monkeypatch)
    state.add_request(0, _request(_history()))
    state.add_request(1, _session_request(req_id="newer"))
    runner = _runner(state)
    newer_state = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["newer"]))
    runner.execute_model_state = newer_state
    runner._last_aux_output = object()
    runner._last_multimodal_outputs = object()
    runner._last_multimodal_snapshot_slot = 2
    runner._async_mm_snapshot_pending = [False, False, True, False]
    saved_aux, saved_mm = runner._last_aux_output, runner._last_multimodal_outputs
    owner_thread = get_ident()
    recovered = []

    def recover(req_ids, exception):
        assert get_ident() == owner_thread
        recovered.append((req_ids, exception))
        return runner._abort_failed_materialization(req_ids, exception)

    output = _async_output(on_materialization_error=recover)
    failure = ValueError(f"injected {failure_phase} failure")

    def fail(*args, **kwargs):
        raise failure

    if failure_phase == "aux":
        output.pending_aux_output = SimpleNamespace(process_output=fail)
    elif failure_phase == "output_finalizer":
        output._finalize_output = fail
    else:
        output._need_pooler = output._async_chunk = True
        output._num_reqs = 1
        output._mm_snapshot = None
        if failure_phase == "snapshot_finalizer":
            output._finalize_multimodal = fail
        else:
            output._mm_snapshot = RequestOutputSnapshot(inter_stage=[], client=None)
    plane = Mock()
    plane.get_omni_connector_output.return_value = None
    worker = NativeOutputWorker(2)
    try:
        owned = worker.submit(output, plane) if transport == "native" else OwnerAsyncOutput(output)
        result = owned.get_output()
        assert owned.get_output() is result
    finally:
        worker.close()
    assert len(recovered) == 1
    assert recovered[0][0] == ["req-0"]
    assert result.req_ids == ["req-0"]
    assert result.sampled_token_ids == [[]]
    assert "output materialization" in result.request_errors["req-0"]
    assert failure_phase in result.request_errors["req-0"] or failure_phase == "snapshot_partition"
    assert state._poisoned_rows == {0}
    assert not bool(state._poisoned[1])
    assert runner.execute_model_state is newer_state
    assert runner._last_aux_output is saved_aux
    assert runner._last_multimodal_outputs is saved_mm
    assert runner._last_multimodal_snapshot_slot == 2
    assert runner._async_mm_snapshot_pending == [False, False, True, False]
    runner.main_stream.synchronize.assert_called_once()
    runner.output_copy_stream.synchronize.assert_called_once()
    plane.enqueue_outputs.assert_not_called()


@pytest.mark.parametrize("failure_phase", ["event", "ep", "transport"])
def test_device_and_started_transport_faults_remain_fatal_and_never_invoke_request_recovery(monkeypatch, failure_phase):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    recover = Mock()
    output = _async_output(on_materialization_error=recover)
    plane = Mock()
    plane.get_omni_connector_output.return_value = None
    if failure_phase == "event":
        output.copy_event.synchronize = Mock(side_effect=RuntimeError("CUDA event failed"))
        expected = "CUDA event failed"
    elif failure_phase == "ep":
        output._has_fault = torch.tensor(True)
        monkeypatch.setattr(
            ar_runner,
            "get_ep_all2all_manager",
            lambda: SimpleNamespace(query_active_mask=lambda: torch.tensor([False])),
        )
        expected = "EP all2all"
    else:
        plane.enqueue_outputs.side_effect = RuntimeError("transport failed after publication began")
        expected = "publication began"
    worker = NativeOutputWorker(1)
    try:
        native = worker.submit(output, plane)
        with pytest.raises(RuntimeError, match=expected):
            native.get_output()
        with pytest.raises(RuntimeError, match=expected):
            native.get_output()
    finally:
        worker.close()
    recover.assert_not_called()
    if failure_phase != "transport":
        plane.enqueue_outputs.assert_not_called()
    else:
        plane.enqueue_outputs.assert_called_once()


def test_materialization_failure_without_model_recovery_capability_propagates(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    output = _async_output()
    output.pending_aux_output = SimpleNamespace(process_output=Mock(side_effect=ValueError("unsupported model fault")))
    plane = Mock()
    worker = NativeOutputWorker(1)
    try:
        native = worker.submit(output, plane)
        with pytest.raises(ValueError, match="unsupported model fault"):
            native.get_output()
    finally:
        worker.close()
    plane.enqueue_outputs.assert_not_called()


@pytest.mark.parametrize("failure_phase", ["event", "ep"])
def test_direct_owner_device_and_ep_faults_remain_fatal_and_cached(monkeypatch, failure_phase):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    recover = Mock()
    output = _async_output(on_materialization_error=recover)
    if failure_phase == "event":
        failure = RuntimeError("CUDA event failed")
        output.copy_event.synchronize = Mock(side_effect=failure)
    else:
        output._has_fault = torch.tensor(True)
        monkeypatch.setattr(
            ar_runner,
            "get_ep_all2all_manager",
            lambda: SimpleNamespace(query_active_mask=lambda: torch.tensor([False])),
        )
        failure = None
    owned = OwnerAsyncOutput(output)
    errors = []
    for _ in range(2):
        with pytest.raises(RuntimeError) as info:
            owned.get_output()
        errors.append(info.value)
    assert errors[0] is errors[1]
    if failure is not None:
        assert errors[0] is failure
    recover.assert_not_called()


@pytest.mark.parametrize("transport", ["native", "direct"])
def test_deferred_failure_after_slot_reuse_does_not_poison_or_mutate_new_owner(monkeypatch, transport):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    state = _session_state(monkeypatch)
    state.add_request(0, _request(_history()))
    runner = _runner(state)
    output = _async_output(on_materialization_error=runner._abort_failed_materialization)
    output._finalize_output = Mock(side_effect=ValueError("late original finalizer failure"))
    state.on_requests_finished({"req-0"})
    state.remove_request("req-0")
    state.add_request(0, _session_request(req_id="reused-owner"))
    state._committed_ticks[0] = 42
    state._control_modes[0] = 1
    state._last_text_tokens[0] = 123
    new_epoch = state._execution_epochs[0].item()
    newer = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["reused-owner"]))
    runner.execute_model_state = newer
    worker = NativeOutputWorker(1)
    plane = Mock()
    plane.get_omni_connector_output.return_value = None
    try:
        owned = worker.submit(output, plane) if transport == "native" else OwnerAsyncOutput(output)
        result = owned.get_output()
        assert owned.get_output() is result
    finally:
        worker.close()
    assert list(result.request_errors) == ["req-0"]
    assert state._poisoned_rows == set()
    assert state._execution_epochs[0].item() == new_epoch
    assert state._committed_ticks[0].item() == 42
    assert state._control_modes[0].item() == 1
    assert state._last_text_tokens[0].item() == 123
    assert runner.execute_model_state is newer
    plane.enqueue_outputs.assert_not_called()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import UtilityOutput
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.request import RequestStatus

from tests.distributed.omni_connectors.test_chunk_transfer_adapter import build_adapter as _build_adapter_fixture
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc
from vllm_omni.model_executor.stage_input_processors.personaplex import talker2code2wav_async_chunk
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
build_adapter = _build_adapter_fixture


def _request(external_id: str = "epoch-one") -> OmniRequest:
    request = OmniRequest(
        request_id="internal-" + external_id,
        external_req_id=external_id,
        resumable=True,
        additional_information=None,
        prompt_token_ids=[0],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    # A resumable segment has completed, not the enclosing duplex stream.
    request.status = RequestStatus.FINISHED_STOPPED
    request.num_computed_tokens = 1
    return request


@dataclass
class _CodecConnector:
    config: dict[str, object]


@dataclass
class _CodecTransferState:
    """Processor-only state, without starting a connector sender thread."""

    connector: _CodecConnector
    request_payload: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class _CodecScheduler:
    """Only the scheduler-owned lookup and adapter used by this utility."""

    requests: dict[str, OmniRequest]
    chunk_transfer_adapter: OmniChunkTransferAdapter | None = None


def _row(index):
    return torch.arange(8, dtype=torch.long).reshape(1, 8) + index * 100


def _manager(chunk_frames: int = 5) -> _CodecTransferState:
    return _CodecTransferState(
        connector=_CodecConnector(
            config={"extra": {"codec_chunk_frames": chunk_frames, "initial_codec_chunk_frames": 1}}
        )
    )


def _append_rows(manager, request, count):
    for index in range(count):
        request.additional_information = {"codes": {"audio": _row(index)}}
        talker2code2wav_async_chunk(manager, None, request, is_finished=True)


def _adapter(build_adapter, count=4):
    adapter, connector = build_adapter(
        stage_id=0, connector_extra={"codec_chunk_frames": 5, "initial_codec_chunk_frames": 1}
    )
    adapter.custom_process_next_stage_input_func = talker2code2wav_async_chunk
    request = _request()
    for index in range(count):
        adapter.save_async({"codes": {"audio": _row(index)}}, request, is_segment_finished=True)
    return adapter, connector, request


def _send_queue(adapter):
    while adapter._pending_save_reqs:
        adapter._send_single_request(adapter._pending_save_reqs.popleft())


def test_flush_preserves_one_successor_and_never_appends_stale_request_codes():
    manager, request = _manager(), _request()
    _append_rows(manager, request, 4)
    result = talker2code2wav_async_chunk(manager, None, request, flush_prefix=4)
    expected = torch.cat([torch.cat([_row(i)[:, :1], _row(i + 1)[:, 1:]], dim=1) for i in (1, 2)])
    assert torch.equal(result.codes.audio, expected.T.contiguous().reshape(-1))
    assert not result.meta.finished.item()
    assert [row.tolist() for row in manager.request_payload[request.external_req_id]["personaplex_frames"]] == [
        _row(3).reshape(-1).tolist()
    ]
    assert talker2code2wav_async_chunk(manager, None, request, flush_prefix=4) is None


def test_flush_frozen_prefix_does_not_consume_later_buffered_frames():
    manager, request = _manager(), _request()
    _append_rows(manager, request, 5)
    early = talker2code2wav_async_chunk(manager, None, request, flush_prefix=3)
    assert early.codes.audio.tolist() == torch.cat([_row(1)[:, :1], _row(2)[:, 1:]], dim=1).reshape(-1).tolist()
    assert len(manager.request_payload[request.external_req_id]["personaplex_frames"]) == 3
    later = talker2code2wav_async_chunk(manager, None, request, flush_prefix=5)
    assert later.codes.audio.numel() == 2 * 8
    assert len(manager.request_payload[request.external_req_id]["personaplex_frames"]) == 1


@pytest.mark.parametrize("count", [1, 2, 7])
def test_flush_already_emitted_prefix_never_fabricates_a_successor(count):
    manager, request = _manager(), _request()
    _append_rows(manager, request, count)
    assert talker2code2wav_async_chunk(manager, None, request, flush_prefix=count) is None
    assert len(manager.request_payload[request.external_req_id]["personaplex_frames"]) == 1


@pytest.mark.parametrize("prefix", [0, -1, True, 1.5, 5])
def test_invalid_or_not_generated_prefix_is_rejected_without_changing_tail(prefix):
    manager, request = _manager(), _request()
    _append_rows(manager, request, 4)
    before = [row.clone() for row in manager.request_payload[request.external_req_id]["personaplex_frames"]]
    with pytest.raises(ValueError):
        talker2code2wav_async_chunk(manager, None, request, flush_prefix=prefix)
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, manager.request_payload[request.external_req_id]["personaplex_frames"])
    )


def test_core_utility_waits_for_fifo_flush_without_blocking_engine(build_adapter):
    adapter, connector, request = _adapter(build_adapter)
    core = StageEngineCoreProc.__new__(StageEngineCoreProc)
    core.scheduler = _CodecScheduler(requests={request.request_id: request}, chunk_transfer_adapter=adapter)
    future = core.omni_flush_codec_prefix(request.external_req_id, 4)
    assert not future.done()
    queued: list[UtilityOutput] = []
    output = UtilityOutput(call_id=17)
    EngineCoreProc._invoke_utility_method("omni_flush_codec_prefix", lambda: future, output, queued.append)
    assert queued == []
    _send_queue(adapter)
    assert future.result(timeout=0) == 4
    assert queued == [output]
    assert output.failure_message is None
    payload = connector.put.call_args.kwargs["data"]
    assert payload.codes.audio.numel() == 2 * 8
    assert not payload.meta.finished.item()
    assert payload.meta.is_segment_finished.item()


def test_repeated_flush_has_no_empty_transport_marker(build_adapter):
    adapter, connector, request = _adapter(build_adapter)
    first = adapter.flush_prefix_async(request, 4)
    assert adapter.flush_prefix_async(request, 4) is first
    _send_queue(adapter)
    assert first.result(timeout=0) == 4
    previous_puts = connector.put.call_count
    second = adapter.flush_prefix_async(request, 4)
    _send_queue(adapter)
    assert second.result(timeout=0) == 4
    assert connector.put.call_count == previous_puts


def test_flush_bounds_pending_control_work_per_request(build_adapter):
    adapter, _, request = _adapter(build_adapter)
    adapter.flush_prefix_async(request, 4)
    with pytest.raises(RuntimeError, match="pending"):
        adapter.flush_prefix_async(request, 3)
    assert len(adapter._pending_save_reqs) == 5


@pytest.mark.parametrize("failure", [False, OSError("connector failed")])
def test_failed_put_cannot_acknowledge_or_retry_consumed_prefix(build_adapter, failure):
    adapter, connector, request = _adapter(build_adapter)
    _send_queue(adapter)
    if failure is False:
        connector.put.return_value = (False, 0, {})
    else:
        connector.put.side_effect = failure
    future = adapter.flush_prefix_async(request, 4)
    try:
        _send_queue(adapter)
    except OSError:
        pass
    with pytest.raises((RuntimeError, OSError)):
        future.result(timeout=0)
    with pytest.raises(RuntimeError):
        adapter.flush_prefix_async(request, 4)


def test_prior_regular_put_failure_prevents_false_noop_flush_success(build_adapter):
    adapter, connector, request = _adapter(build_adapter, count=2)
    connector.put.return_value = (False, 0, {})
    future = adapter.flush_prefix_async(request, 2)
    _send_queue(adapter)
    with pytest.raises(RuntimeError):
        future.result(timeout=0)


def test_cleanup_fails_queued_flush_and_keeps_successor_generation_isolated(build_adapter):
    adapter, connector, request = _adapter(build_adapter)
    future = adapter.flush_prefix_async(request, 4)
    adapter.cleanup_sender(request.external_req_id)
    with pytest.raises(RuntimeError):
        future.result(timeout=0)
    adapter.save_async({"codes": {"audio": _row(50)}}, request, is_segment_finished=True)
    _send_queue(adapter)
    assert connector.put.call_count == 1
    assert len(adapter.request_payload[request.external_req_id]["personaplex_frames"]) == 1
    assert adapter.request_payload[request.external_req_id]["personaplex_frames"][0].equal(_row(50).reshape(-1))


def test_cleanup_does_not_wait_for_blocked_flush_put(build_adapter):
    adapter, connector, request = _adapter(build_adapter)
    _send_queue(adapter)
    entered, release = threading.Event(), threading.Event()

    def blocked_put(**kwargs):
        entered.set()
        assert release.wait(5), "test must release its own connector operation"
        return True, 1, {}

    connector.put.side_effect = blocked_put
    future = adapter.flush_prefix_async(request, 4)
    thread = threading.Thread(target=_send_queue, args=(adapter,), daemon=True)
    thread.start()
    try:
        assert entered.wait(5)
        assert not future.done()
        adapter.cleanup_sender(request.external_req_id)
        with pytest.raises(RuntimeError):
            future.result(timeout=0)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert request.external_req_id not in adapter.request_payload


def test_cancelled_queued_flush_does_not_consume_codec_tail(build_adapter):
    adapter, connector, request = _adapter(build_adapter)
    _send_queue(adapter)
    before = connector.put.call_count
    future = adapter.flush_prefix_async(request, 4)
    assert future.cancel()
    _send_queue(adapter)
    assert connector.put.call_count == before
    assert len(adapter.request_payload[request.external_req_id]["personaplex_frames"]) == 3


def test_shutdown_fails_pending_flush_without_draining_queue(build_adapter):
    adapter, _, request = _adapter(build_adapter)
    future = adapter.flush_prefix_async(request, 4)
    adapter.shutdown()
    with pytest.raises(RuntimeError):
        future.result(timeout=0)
    with pytest.raises(RuntimeError):
        adapter.flush_prefix_async(request, 4)


def test_processor_failure_is_reported_instead_of_publishing_an_empty_marker(build_adapter):
    adapter, connector, request = _adapter(build_adapter)
    _send_queue(adapter)
    original = adapter.custom_process_next_stage_input_func

    def raising_processor(*args, flush_prefix=None, **kwargs):
        raise ValueError("invalid codec prefix")

    adapter.custom_process_next_stage_input_func = raising_processor
    previous_puts = connector.put.call_count
    future = adapter.flush_prefix_async(request, 4)
    with pytest.raises(ValueError):
        _send_queue(adapter)
    with pytest.raises(ValueError):
        future.result(timeout=0)
    assert connector.put.call_count == previous_puts
    adapter.custom_process_next_stage_input_func = original


def test_unsupported_processor_cannot_claim_prefix_flushing(build_adapter):
    adapter, _, request = _adapter(build_adapter)
    adapter.custom_process_next_stage_input_func = lambda **kwargs: None
    with pytest.raises(RuntimeError, match="support"):
        adapter.flush_prefix_async(request, 4)


@pytest.mark.parametrize("external_id", ["missing", "internal-epoch-one"])
def test_core_does_not_confuse_external_and_internal_request_ids(build_adapter, external_id):
    adapter, _, request = _adapter(build_adapter)
    core = StageEngineCoreProc.__new__(StageEngineCoreProc)
    core.scheduler = _CodecScheduler(requests={request.request_id: request}, chunk_transfer_adapter=adapter)
    with pytest.raises(RuntimeError):
        core.omni_flush_codec_prefix(external_id, 4)


def test_flush_round_trips_real_shared_memory_without_finishing_stream(build_adapter):
    adapter, _, _ = _adapter(build_adapter, count=0)
    connector = SharedMemoryConnector(
        {"stage_id": 0, "extra": {"codec_chunk_frames": 5, "initial_codec_chunk_frames": 1}}
    )
    adapter.connector = connector
    request = _request("prefix-flush-" + uuid.uuid4().hex)
    try:
        for index in range(4):
            adapter.save_async({"codes": {"audio": _row(index)}}, request, is_segment_finished=True)
        future = adapter.flush_prefix_async(request, 4)
        _send_queue(adapter)
        assert future.result(timeout=0) == 4
        result = connector.get("0", "1", f"{request.external_req_id}_0_4")
        assert result is not None
        payload, _ = result
        assert payload["codes"]["audio"].numel() == 2 * 8
        assert not payload["meta"]["finished"].item()
        assert payload["meta"]["is_segment_finished"].item()
        assert len(adapter.request_payload[request.external_req_id]["personaplex_frames"]) == 1
    finally:
        adapter.cleanup_sender(request.external_req_id)
        _send_queue(adapter)
        connector.close()


def test_future_callback_can_cleanup_without_sender_lock_deadlock(build_adapter):
    adapter, _, request = _adapter(build_adapter)
    future = adapter.flush_prefix_async(request, 4)
    future.add_done_callback(lambda _: adapter.cleanup_sender(request.external_req_id))
    thread = threading.Thread(target=_send_queue, args=(adapter,), daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert future.result(timeout=0) == 4
    assert request.external_req_id not in adapter.request_payload


def test_invalid_codebooks_cannot_produce_a_flush_acknowledgement():
    manager, request = _manager(), _request()
    _append_rows(manager, request, 3)
    invalid = _row(3)
    invalid[0, 2] = -1
    request.additional_information = {"codes": {"audio": invalid}}
    talker2code2wav_async_chunk(manager, None, request, is_finished=True)
    before = len(manager.request_payload[request.external_req_id]["personaplex_frames"])
    with pytest.raises(ValueError, match="incomplete"):
        talker2code2wav_async_chunk(manager, None, request, flush_prefix=4)
    assert len(manager.request_payload[request.external_req_id]["personaplex_frames"]) == before


def test_core_without_v1_adapter_fails_closed():
    core = StageEngineCoreProc.__new__(StageEngineCoreProc)
    core.scheduler = _CodecScheduler(requests={})
    with pytest.raises(RuntimeError, match="queue"):
        core.omni_flush_codec_prefix("epoch-one", 4)

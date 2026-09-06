import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker_v2.delivery import DeliveryState, OmniDeliveryManager
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _new_request(req_id: str = "internal", external_req_id: str = "external") -> SimpleNamespace:
    sampling_params = SimpleNamespace(stop_token_ids=[2150])
    return SimpleNamespace(
        req_id=req_id,
        external_req_id=external_req_id,
        prompt_token_ids=[10, 11],
        additional_information=SimpleNamespace(entries={}),
        sampling_params=sampling_params,
        num_computed_tokens=2,
        resumable=True,
    )


def _bare_plane(
    *,
    delivery_timeout_s: float = 1.0,
    shutdown_timeout_s: float = 1.0,
) -> OmniRunnerDataPlane:
    plane = object.__new__(OmniRunnerDataPlane)
    plane._native_requests = {}
    plane._native_output_lock = threading.RLock()
    plane._native_send_lock = threading.Lock()
    plane._native_output_error_lock = threading.Lock()
    plane._native_output_error = None
    plane._native_output_closed = False
    plane._native_outputs_in_flight = defaultdict(int)
    plane._native_terminal_pending = set()
    plane._put_req_chunk = defaultdict(int)
    plane._ramp_chunk_count = defaultdict(int)
    plane._delivery_manager = OmniDeliveryManager(
        delivery_timeout_s=delivery_timeout_s,
        shutdown_timeout_s=shutdown_timeout_s,
    )
    return plane


class _AlwaysFailConnector:
    def __init__(self) -> None:
        self.put_calls = 0

    def put(self, **_kwargs):
        self.put_calls += 1
        return False, 0, None

    def close(self) -> None:
        pass


class _BlockingConnector:
    def __init__(self) -> None:
        self.put_started = threading.Event()
        self.close_called = threading.Event()
        self.release_put = threading.Event()

    def put(self, **_kwargs):
        self.put_started.set()
        self.release_put.wait()
        return True, 1, None

    def close(self) -> None:
        self.close_called.set()


class _BlockFirstConnector:
    def __init__(self) -> None:
        self.put_keys: list[str] = []
        self.first_put_started = threading.Event()
        self.release_first_put = threading.Event()

    def put(self, **kwargs):
        self.put_keys.append(kwargs["put_key"])
        if len(self.put_keys) == 1:
            self.first_put_started.set()
            assert self.release_first_put.wait(timeout=2)
        return True, 1, None

    def close(self) -> None:
        pass


def test_data_plane_initializes_connector_with_current_mixin_contract(monkeypatch) -> None:
    calls = []

    def fake_init_connectors(self, **kwargs):
        calls.append(kwargs)
        self._custom_process_func = None

    monkeypatch.setattr(OmniRunnerDataPlane, "init_omni_connectors", fake_init_connectors)
    monkeypatch.setattr(OmniRunnerDataPlane, "_connector_delivery_timeout", lambda _self: 1.0)
    monkeypatch.setattr(OmniRunnerDataPlane, "_start_output_worker", lambda _self, **_kwargs: None)
    model_config = SimpleNamespace(stage_id=0)

    OmniRunnerDataPlane(SimpleNamespace(), model_config)

    assert calls == [{"model_config": model_config}]


def _start_real_save_thread(plane: OmniRunnerDataPlane, connector: _AlwaysFailConnector) -> None:
    plane._omni_connector = connector
    plane._stage_id = 0
    plane._next_stage_id = 1
    plane._lock = threading.Lock()
    plane._pending_save_reqs = {}
    plane._pending_save_counts = defaultdict(int)
    plane._deferred_send_cleanup = set()
    plane._request_ids_mapping = {}
    plane._send_side_request_payload = {}
    plane._code_prompt_token_ids = defaultdict(list)
    plane._cached_ic = {}
    plane._work_available = threading.Event()
    plane._stop_event = threading.Event()
    plane._custom_process_batch_func = lambda **kwargs: kwargs["pooling_outputs"]
    plane._can_send = True
    plane._MAX_SEND_RETRIES = 0
    plane.is_data_transfer_rank = lambda: True
    plane._connector_send_error_sink = plane._record_output_error
    plane._recv_thread = None
    plane._save_thread = threading.Thread(target=plane._save_loop, daemon=True)
    plane._save_thread.start()


def _stop_real_save_thread(plane: OmniRunnerDataPlane) -> None:
    plane._stop_event.set()
    plane._work_available.set()
    plane._save_thread.join(timeout=2)
    assert not plane._save_thread.is_alive()


def test_runner_data_plane_owns_cumulative_request_state() -> None:
    plane = _bare_plane()
    sent = []
    plane.send_chunks = lambda entries, **_kwargs: sent.extend(entries) or len(entries)
    plane.cleanup_finished_request = lambda _req_id: None

    plane.register_request(_new_request())
    plane.emit_chunks(
        req_ids=["internal"],
        inter_stage_outputs=[{"codes.audio": "chunk-0"}],
        sampled_token_ids=[[21]],
        terminal_req_ids=set(),
    )
    plane.emit_chunks(
        req_ids=["internal"],
        inter_stage_outputs=[{"codes.audio": "chunk-1"}],
        sampled_token_ids=[[22]],
        terminal_req_ids={"internal"},
    )

    first, second = sent
    assert first[1] == {"codes": {"audio": "chunk-0"}}
    assert second[1] == {"codes": {"audio": "chunk-1"}}
    assert first[0].external_req_id == "external"
    assert first[0].prompt_token_ids == [10, 11]
    assert first[0].output_token_ids == [21]
    assert first[0].all_token_ids == [10, 11, 21]
    assert first[0].sampling_params.stop_token_ids == [2150]
    assert first[0].num_computed_tokens == 2
    assert first[0].resumable is True
    assert first[0].is_finished() is False
    assert second[0].output_token_ids == [21, 22]
    assert second[0].all_token_ids == [10, 11, 21, 22]
    assert second[0].is_finished() is True


def test_runner_data_plane_emits_terminal_without_new_payload() -> None:
    plane = _bare_plane()
    sent = []
    cleaned = []
    plane.send_chunks = lambda entries, **_kwargs: sent.extend(entries) or len(entries)
    plane.cleanup_finished_request = cleaned.append

    plane.register_request(_new_request())
    plane.emit_chunks(
        req_ids=[],
        inter_stage_outputs=None,
        sampled_token_ids=None,
        terminal_req_ids={"internal"},
    )

    assert len(sent) == 1
    assert sent[0][1] is None
    assert sent[0][0].is_finished() is True
    assert cleaned == ["internal"]


def test_runner_data_plane_cleans_terminal_request_when_stage_cannot_send() -> None:
    plane = _bare_plane()
    plane._can_send = False
    cleaned = []
    plane.cleanup_finished_request = cleaned.append
    plane.register_request(_new_request())

    assert plane.request_terminal({"internal"}) == 0

    assert "internal" not in plane._native_requests
    assert "internal" not in plane._native_terminal_pending
    assert cleaned == ["internal"]


def test_runner_data_plane_retries_terminal_after_enqueue_failure() -> None:
    plane = _bare_plane()
    plane.cleanup_finished_request = lambda _req_id: None
    plane.register_request(_new_request())

    plane.send_chunks = lambda _entries, **_kwargs: (_ for _ in ()).throw(RuntimeError("enqueue failed"))
    with pytest.raises(RuntimeError, match="enqueue failed"):
        plane.request_terminal({"internal"})

    assert "internal" in plane._native_requests
    assert "internal" in plane._native_terminal_pending

    batches = []
    plane.send_chunks = lambda entries, **_kwargs: batches.append(entries) or len(entries)

    assert plane.request_terminal(set()) == 1
    assert len(batches) == 1
    assert batches[0][0][0].is_finished() is True
    assert "internal" not in plane._native_requests
    assert "internal" not in plane._native_terminal_pending


def test_runner_data_plane_emits_ready_terminals_as_one_cohort() -> None:
    plane = _bare_plane()
    cleaned = []
    batches = []
    plane.cleanup_finished_request = cleaned.append
    plane.register_request(_new_request("internal-0", "external-0"))
    plane.register_request(_new_request("internal-1", "external-1"))

    plane.send_chunks = lambda entries, **_kwargs: batches.append(entries) or len(entries)

    assert plane.request_terminal({"internal-0", "internal-1"}) == 2

    assert len(batches) == 1
    assert {request.external_req_id for request, _payload in batches[0]} == {
        "external-0",
        "external-1",
    }
    assert all(request.is_finished() for request, _payload in batches[0])
    assert cleaned == ["internal-0", "internal-1"]
    assert plane._native_terminal_pending == set()


def test_runner_data_plane_defers_terminal_until_reserved_output_completes() -> None:
    plane = _bare_plane()
    batches = []
    cleaned = []
    plane.send_chunks = lambda entries, **_kwargs: batches.append(entries) or len(entries)
    plane.cleanup_finished_request = cleaned.append

    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])

    assert plane.request_terminal({"internal"}) == 0
    assert batches == []
    assert cleaned == []

    emitted = plane.complete_outputs(
        req_ids=["internal"],
        inter_stage_outputs=[{"codes.audio": "late-chunk"}],
        sampled_token_ids=[[21]],
    )

    assert emitted == 2
    assert len(batches) == 2
    chunk_request, chunk_payload = batches[0][0]
    terminal_request, terminal_payload = batches[1][0]
    assert chunk_payload == {"codes": {"audio": "late-chunk"}}
    assert chunk_request.output_token_ids == [21]
    assert chunk_request.is_finished() is False
    assert terminal_payload is None
    assert terminal_request.output_token_ids == [21]
    assert terminal_request.is_finished() is True
    assert cleaned == ["internal"]


def test_runner_data_plane_waits_for_all_reserved_outputs_before_terminal() -> None:
    plane = _bare_plane()
    batches = []
    plane.send_chunks = lambda entries, **_kwargs: batches.append(entries) or len(entries)
    plane.cleanup_finished_request = lambda _req_id: None

    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    plane.reserve_outputs(["internal"])
    plane.request_terminal({"internal"})

    plane.complete_outputs(
        req_ids=["internal"],
        inter_stage_outputs=[{"codes.audio": "chunk-0"}],
        sampled_token_ids=[[21]],
    )
    assert len(batches) == 1
    assert batches[0][0][0].is_finished() is False

    plane.complete_outputs(
        req_ids=["internal"],
        inter_stage_outputs=[{"codes.audio": "chunk-1"}],
        sampled_token_ids=[[22]],
    )

    assert len(batches) == 3
    assert batches[1][0][1] == {"codes": {"audio": "chunk-1"}}
    assert batches[1][0][0].output_token_ids == [21, 22]
    assert batches[1][0][0].is_finished() is False
    assert batches[2][0][1] is None
    assert batches[2][0][0].output_token_ids == [21, 22]
    assert batches[2][0][0].is_finished() is True


def test_runner_data_plane_output_worker_does_not_block_scheduler_or_reorder_terminal() -> None:
    plane = _bare_plane()
    batches = []
    send_started = threading.Event()
    release_send = threading.Event()

    def send_chunks(entries, **_kwargs):
        if entries and not entries[0][0].is_finished():
            send_started.set()
            assert release_send.wait(timeout=2)
        batches.append(entries)
        return len(entries)

    plane.send_chunks = send_chunks
    plane.cleanup_finished_request = lambda _req_id: None
    plane._start_output_worker(max_pending_batches=2)
    try:
        plane.register_request(_new_request())
        plane.reserve_outputs(["internal"])

        plane.enqueue_outputs(
            req_ids=["internal"],
            inter_stage_outputs=[{"codes.audio": "chunk-0"}],
            sampled_token_ids=[[21]],
        )
        assert send_started.wait(timeout=2)

        terminal_results = []
        terminal_thread = threading.Thread(target=lambda: terminal_results.append(plane.request_terminal({"internal"})))
        terminal_thread.start()
        terminal_thread.join(timeout=0.2)

        assert not terminal_thread.is_alive()
        assert terminal_results == [0]
        assert batches == []

        release_send.set()
        plane.drain_outputs()

        assert len(batches) == 2
        assert batches[0][0][1] == {"codes": {"audio": "chunk-0"}}
        assert batches[0][0][0].is_finished() is False
        assert batches[1][0][1] is None
        assert batches[1][0][0].is_finished() is True
    finally:
        release_send.set()
        plane._stop_output_worker()


def test_runner_data_plane_output_worker_surfaces_connector_failure() -> None:
    plane = _bare_plane()
    plane.send_chunks = lambda _entries, **_kwargs: (_ for _ in ()).throw(RuntimeError("enqueue failed"))
    plane.cleanup_finished_request = lambda _req_id: None
    plane._start_output_worker(max_pending_batches=2)
    try:
        plane.register_request(_new_request())
        plane.reserve_outputs(["internal"])
        plane.enqueue_outputs(
            req_ids=["internal"],
            inter_stage_outputs=[{"codes.audio": "chunk-0"}],
            sampled_token_ids=[[21]],
        )

        with pytest.raises(RuntimeError, match="enqueue failed"):
            plane.drain_outputs()
        with pytest.raises(RuntimeError, match="enqueue failed"):
            plane.enqueue_outputs(
                req_ids=["internal"],
                inter_stage_outputs=[{"codes.audio": "chunk-1"}],
                sampled_token_ids=[[22]],
            )
    finally:
        plane._stop_output_worker()


def test_runner_data_plane_close_drains_outputs_before_connector_shutdown() -> None:
    plane = _bare_plane()
    events = []
    plane.send_chunks = lambda entries, **_kwargs: events.append(("send", entries)) or len(entries)
    plane.cleanup_finished_request = lambda _req_id: None
    plane.shutdown_omni_connectors = lambda: events.append(("shutdown", None))
    plane._start_output_worker(max_pending_batches=2)

    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    plane.enqueue_outputs(
        req_ids=["internal"],
        inter_stage_outputs=[{"codes.audio": "chunk-0"}],
        sampled_token_ids=[[21]],
    )

    plane.close()

    assert [event for event, _payload in events] == ["send", "shutdown"]


def test_runner_data_plane_emits_duplicate_terminal_request_once() -> None:
    plane = _bare_plane()
    batches = []
    plane.send_chunks = lambda entries, **_kwargs: batches.append(entries) or len(entries)
    plane.cleanup_finished_request = lambda _req_id: None

    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])

    assert plane.request_terminal({"internal"}) == 0
    assert plane.request_terminal({"internal"}) == 0
    assert (
        plane.complete_outputs(
            req_ids=["internal"],
            inter_stage_outputs=[{"codes.audio": "chunk-0"}],
            sampled_token_ids=[[21]],
        )
        == 2
    )

    assert len(batches) == 2
    assert batches[0][0][0].is_finished() is False
    assert batches[1][0][0].is_finished() is True


def test_runner_data_plane_keeps_lifecycle_hold_when_output_enqueue_fails() -> None:
    plane = _bare_plane()
    plane.send_chunks = lambda _entries, **_kwargs: (_ for _ in ()).throw(RuntimeError("enqueue failed"))
    plane.cleanup_finished_request = lambda _req_id: pytest.fail("failed enqueue must not clean request state")

    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    plane.request_terminal({"internal"})

    with pytest.raises(RuntimeError, match="enqueue failed"):
        plane.complete_outputs(
            req_ids=["internal"],
            inter_stage_outputs=[{"codes.audio": "late-chunk"}],
            sampled_token_ids=[[21]],
        )

    assert "internal" in plane._native_requests
    assert plane._native_outputs_in_flight["internal"] == 1
    assert "internal" in plane._native_terminal_pending


def test_runner_data_plane_surfaces_permanent_background_put_failure() -> None:
    plane = _bare_plane()
    connector = _AlwaysFailConnector()
    _start_real_save_thread(plane, connector)
    plane.cleanup_finished_request = lambda _req_id: None
    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    plane.request_terminal({"internal"})

    try:
        with pytest.raises(RuntimeError, match="connector send failed"):
            plane.complete_outputs(
                req_ids=["internal"],
                inter_stage_outputs=[{"codes.audio": "late-chunk"}],
                sampled_token_ids=[[21]],
            )

        assert connector.put_calls == 1
        assert plane._native_output_error is not None
        assert "internal" in plane._native_requests
        assert plane._native_outputs_in_flight["internal"] == 1
        assert "internal" in plane._native_terminal_pending
    finally:
        _stop_real_save_thread(plane)


def test_runner_data_plane_close_is_bounded_when_connector_put_never_returns() -> None:
    plane = _bare_plane(delivery_timeout_s=0.05, shutdown_timeout_s=0.1)
    connector = _BlockingConnector()
    _start_real_save_thread(plane, connector)
    plane.cleanup_finished_request = lambda _req_id: None
    plane._start_output_worker(max_pending_batches=2)
    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    plane.enqueue_outputs(
        req_ids=["internal"],
        inter_stage_outputs=[{"codes.audio": "chunk-0"}],
        sampled_token_ids=[[21]],
    )
    assert connector.put_started.wait(timeout=1)

    close_errors: list[BaseException] = []

    def close_plane() -> None:
        try:
            plane.close()
        except BaseException as error:
            close_errors.append(error)

    close_thread = threading.Thread(target=close_plane)
    close_thread.start()
    close_thread.join(timeout=1)

    try:
        assert not close_thread.is_alive()
        assert connector.close_called.is_set()
        assert len(close_errors) == 1
        assert "delivery timed out" in str(close_errors[0])
        assert plane._delivery_manager.is_quarantined
    finally:
        connector.release_put.set()
        _stop_real_save_thread(plane)


def test_runner_data_plane_does_not_send_queued_ticket_after_delivery_timeout() -> None:
    plane = _bare_plane(delivery_timeout_s=0.05)
    connector = _BlockFirstConnector()
    _start_real_save_thread(plane, connector)
    request = SimpleNamespace(request_id="external")
    first_enqueued, first_ticket = plane._enqueue_chunk_payload(
        request,
        {"chunk": "first"},
        wait_for_delivery=True,
    )
    second_enqueued, second_ticket = plane._enqueue_chunk_payload(
        request,
        {"chunk": "second"},
        wait_for_delivery=True,
    )

    try:
        assert first_enqueued and second_enqueued
        assert connector.first_put_started.wait(timeout=1)
        with pytest.raises(TimeoutError, match="delivery timed out"):
            first_ticket.wait()
        assert plane._delivery_manager.is_quarantined

        connector.release_first_put.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with plane._lock:
                if "external" not in plane._pending_save_counts:
                    break
            time.sleep(0.001)

        assert connector.put_keys == [first_ticket.put_key]
        assert second_ticket.state is DeliveryState.FAILED
        with plane._lock:
            assert "external" not in plane._pending_save_counts
            assert "external" not in plane._pending_save_reqs
    finally:
        connector.release_first_put.set()
        _stop_real_save_thread(plane)


def test_runner_data_plane_abort_invalidates_deferred_output_once() -> None:
    plane = _bare_plane()
    batches = []
    cleaned = []
    plane.send_chunks = lambda entries, **_kwargs: batches.append(entries) or len(entries)
    plane.cleanup_finished_request = cleaned.append

    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    plane.request_terminal({"internal"})

    assert plane.abort_requests({"internal"}) == 1
    assert len(batches) == 1
    terminal_request, terminal_payload = batches[0][0]
    assert terminal_request.is_finished() is True
    assert terminal_payload is None
    assert cleaned == ["internal"]
    assert "internal" not in plane._native_requests
    assert "internal" not in plane._native_outputs_in_flight
    assert "internal" not in plane._native_terminal_pending

    assert plane.abort_requests({"internal"}) == 0
    assert (
        plane.complete_outputs(
            req_ids=["internal"],
            inter_stage_outputs=[{"codes.audio": "stale-chunk"}],
            sampled_token_ids=[[21]],
        )
        == 0
    )
    assert len(batches) == 1
    assert cleaned == ["internal"]


def test_runner_data_plane_abort_cannot_overtake_built_deferred_output() -> None:
    plane = _bare_plane()
    send_order = []
    data_ready_to_send = threading.Event()
    allow_data_send = threading.Event()
    terminal_sent = threading.Event()

    plane.send_chunks = lambda entries, **_kwargs: len(entries)
    original_send_entries = plane._send_chunk_entries

    def gated_send_entries(entries, **kwargs):
        is_terminal = entries[0][0].is_finished()
        if not is_terminal:
            data_ready_to_send.set()
            assert allow_data_send.wait(timeout=2)
        result = original_send_entries(entries, **kwargs)
        if is_terminal:
            send_order.append("terminal")
            terminal_sent.set()
        else:
            send_order.append("data")
        return result

    plane._send_chunk_entries = gated_send_entries
    plane.cleanup_finished_request = lambda _req_id: None
    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])

    complete_thread = threading.Thread(
        target=plane.complete_outputs,
        kwargs={
            "req_ids": ["internal"],
            "inter_stage_outputs": [{"codes.audio": "chunk-0"}],
            "sampled_token_ids": [[21]],
        },
    )
    abort_thread = threading.Thread(
        target=plane.abort_requests,
        args=({"internal"},),
    )

    complete_thread.start()
    assert data_ready_to_send.wait(timeout=2)
    abort_thread.start()
    # A correct commit protocol keeps abort behind the already-built data
    # enqueue. The old implementation publishes terminal while data is paused.
    terminal_overtook_data = terminal_sent.wait(timeout=0.1)
    allow_data_send.set()
    complete_thread.join(timeout=2)
    abort_thread.join(timeout=2)

    assert not complete_thread.is_alive()
    assert not abort_thread.is_alive()
    assert not terminal_overtook_data
    assert send_order == ["data", "terminal"]


def test_runner_data_plane_batches_all_outputs_from_one_model_step() -> None:
    plane = _bare_plane()
    batches = []
    plane.send_chunks = lambda entries, **_kwargs: batches.append(entries) or len(entries)
    plane.cleanup_finished_request = lambda _req_id: None
    plane.send_chunk = lambda *_args, **_kwargs: pytest.fail(
        "native output finalization must not fall back to per-request send_chunk"
    )

    plane.register_request(_new_request("internal-0", "external-0"))
    plane.register_request(_new_request("internal-1", "external-1"))
    emitted = plane.emit_chunks(
        req_ids=["internal-0", "internal-1"],
        inter_stage_outputs=[
            {"codes.audio": "chunk-0"},
            {"codes.audio": "chunk-1"},
        ],
        sampled_token_ids=[[21], [22]],
        terminal_req_ids={"internal-1"},
    )

    assert emitted == 2
    assert len(batches) == 1
    assert len(batches[0]) == 2
    first_request, first_payload = batches[0][0]
    second_request, second_payload = batches[0][1]
    assert first_request.external_req_id == "external-0"
    assert first_request.output_token_ids == [21]
    assert first_payload == {"codes": {"audio": "chunk-0"}}
    assert second_request.external_req_id == "external-1"
    assert second_request.output_token_ids == [22]
    assert second_request.is_finished() is True
    assert second_payload == {"codes": {"audio": "chunk-1"}}


def test_runner_data_plane_omits_cumulative_token_lists_after_first_chunk() -> None:
    plane = _bare_plane()
    sent = []
    plane.send_chunks = lambda entries, **_kwargs: sent.extend(entries) or len(entries)
    plane.cleanup_finished_request = lambda _req_id: None

    plane.register_request(_new_request())
    state = plane._native_requests["internal"]
    state.resumable = False
    state.output_token_ids.extend(range(1024))
    plane._put_req_chunk["external"] = 1

    plane.emit_chunks(
        req_ids=["internal"],
        inter_stage_outputs=[{"hidden_states.layer_0": "delta"}],
        sampled_token_ids=[[1024]],
        terminal_req_ids=set(),
    )

    request, _payload = sent[0]
    assert request.output_token_count == 1025
    assert request.prompt_token_ids == []
    assert request.output_token_ids == []
    assert request.all_token_ids == []


def test_runner_data_plane_consumes_decode_delta_without_dropping_prefill() -> None:
    plane = object.__new__(OmniRunnerDataPlane)
    plane._lock = threading.Lock()
    plane._request_ids_mapping = {"internal": "external"}
    decode = object()
    prefill = object()
    hidden = object()
    accumulated = {
        "embed": {"prefill": prefill, "decode": decode},
        "hidden_states": {"output": hidden},
        "ids": {"prompt": [1, 2], "output": [3, 4]},
        "meta": {"finished": False},
    }
    plane._send_side_request_payload = {"external": accumulated}
    plane._local_stage_payload_cache = {"internal": accumulated}

    payload = plane.pop_local_stage_payload("internal")

    assert payload["embed"] == {"prefill": prefill, "decode": decode}
    assert payload["hidden_states"] == {"output": hidden}
    assert plane._send_side_request_payload["external"]["embed"] == {"prefill": prefill}
    assert "output" not in plane._send_side_request_payload["external"]["hidden_states"]
    assert plane._send_side_request_payload["external"]["ids"] == {"prompt": [1, 2]}


def test_runner_data_plane_recv_and_consume_decode_delta_are_atomic() -> None:
    plane = object.__new__(OmniRunnerDataPlane)
    plane._lock = threading.Lock()
    plane._omni_connector = object()
    plane._async_chunk = True
    plane._model_mode = "ar"
    plane._stage_id = 1
    plane._get_req_chunk = defaultdict(int)
    plane._request_ids_mapping = {"internal": "external"}
    plane._finished_load_reqs = set()
    plane._chunk_finished_req_ids = set()
    plane._chunk_stream_completed = set()
    plane._async_chunk_updated_req_ids = set()
    plane._local_request_metadata = {}
    plane._pending_load_reqs = {"internal": object()}

    initial = {
        "embed": {
            "decode": torch.tensor([[1.0], [2.0]]),
            "decode_token_start": 1,
            "decode_token_end": 3,
        },
        "meta": {"finished": False},
    }
    plane._send_side_request_payload = {"external": initial}
    plane._local_stage_payload_cache = {"internal": initial}
    incoming = {
        "embed": {
            "decode": torch.tensor([[3.0]]),
            "decode_token_start": 3,
            "decode_token_end": 4,
        },
        "meta": {"finished": False},
    }
    plane._recv_async_chunk_result = lambda *_args: (incoming, 0)

    accumulated = threading.Event()
    release_recv = threading.Event()
    original_accumulate = plane._accumulate_payload

    def paused_accumulate(req_id, payload):
        result = original_accumulate(req_id, payload)
        accumulated.set()
        assert release_recv.wait(timeout=2)
        return result

    plane._accumulate_payload = paused_accumulate
    recv_thread = threading.Thread(target=plane._poll_single_request, args=("internal",))
    recv_thread.start()
    assert accumulated.wait(timeout=2)

    consumed = []
    consume_thread = threading.Thread(target=lambda: consumed.append(plane.pop_local_stage_payload("internal")))
    consume_thread.start()
    time.sleep(0.05)
    release_recv.set()
    recv_thread.join(timeout=2)
    consume_thread.join(timeout=2)

    assert not recv_thread.is_alive()
    assert not consume_thread.is_alive()
    assert len(consumed) == 1
    assert consumed[0]["embed"]["decode_token_start"] == 1
    assert consumed[0]["embed"]["decode_token_end"] == 4
    assert torch.equal(
        consumed[0]["embed"]["decode"],
        torch.tensor([[1.0], [2.0], [3.0]]),
    )


def test_runner_data_plane_deduplicates_absolute_decode_span() -> None:
    plane = object.__new__(OmniRunnerDataPlane)
    plane._send_side_request_payload = {}
    decode = torch.tensor([[1.0], [2.0]])
    payload = {
        "embed": {
            "decode": decode,
            "decode_token_start": 3,
            "decode_token_end": 5,
        }
    }

    plane._accumulate_payload("external", payload)
    accumulated = plane._accumulate_payload("external", payload)

    assert accumulated["embed"]["decode_token_start"] == 3
    assert accumulated["embed"]["decode_token_end"] == 5
    assert torch.equal(accumulated["embed"]["decode"], decode)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lifecycle contracts for OmniRunnerDataPlane/NativeOutputWorker."""

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.worker_v2.delivery import DeliveryCancelledError, DeliveryState, OmniDeliveryManager
from vllm_omni.worker_v2.native_output_worker import NativeOutputWorker
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _complete(plane, payloads, token=21, req="internal"):
    return plane.complete_outputs(
        req_ids=[req], inter_stage_outputs=payloads, sampled_token_ids=[[token]] * len(payloads)
    )


def _new_request(req_id="internal", external_req_id="external"):
    return SimpleNamespace(
        req_id=req_id,
        external_req_id=external_req_id,
        prompt_token_ids=[10, 11],
        num_computed_tokens=2,
        resumable=True,
        additional_information=SimpleNamespace(entries={}),
        sampling_params=SimpleNamespace(stop_token_ids=[2150]),
    )


def _bare_plane(*, delivery_timeout_s=1.0, shutdown_timeout_s=1.0):
    plane = object.__new__(OmniRunnerDataPlane)
    plane.__dict__.update(
        _native_requests={},
        _native_output_lock=threading.RLock(),
        _native_send_lock=threading.Lock(),
        _native_output_error=None,
        _native_output_error_lock=threading.Lock(),
        _native_output_closed=False,
        _native_outputs_in_flight=defaultdict(int),
        _native_terminal_pending=set(),
        _async_chunk=True,
        _pending_full_payload_send={},
        _put_req_chunk=defaultdict(int),
        _ramp_chunk_count=defaultdict(int),
        _delivery_manager=OmniDeliveryManager(
            delivery_timeout_s=delivery_timeout_s, shutdown_timeout_s=shutdown_timeout_s
        ),
    )
    return plane


def _make_plane(replace_send):
    p = _bare_plane()
    p.record = SimpleNamespace(batches=[], cleaned=[])
    if replace_send:
        p.send_chunks = lambda entries, **_kw: p.record.batches.append(entries) or len(entries)
    p.get_omni_connector_output, p.cleanup_finished_request = (lambda: None), p.record.cleaned.append
    return p


@pytest.fixture
def raw_plane():  # keeps the real connector send path
    p = _make_plane(replace_send=False)
    yield p
    if getattr(p, "_native_output_worker", None) is not None:
        p._stop_output_worker()


@pytest.fixture
def plane():
    p = _make_plane(replace_send=True)
    yield p
    if getattr(p, "_native_output_worker", None) is not None:
        p._stop_output_worker()


def test_full_payload_waits_for_terminal_and_last_deferred_frame(plane):
    plane._async_chunk = False
    plane._full_payload_replace_keys_cached = frozenset({"codes.ref"})
    request = _new_request()
    request.resumable = False
    plane.register_request(request)
    first = torch.tensor([[1, 2]])
    last = torch.tensor([[3, 4]])
    ref = torch.tensor([[5, 6]])
    plane.reserve_outputs(["internal"])
    assert _complete(plane, [{"codes.audio": first, "codes.ref": ref}], token=21) == 0
    assert not plane.record.batches

    plane.reserve_outputs(["internal"])
    assert plane.request_terminal({"internal"}) == 0
    assert not plane.record.cleaned
    assert _complete(plane, [{"codes.audio": last, "codes.ref": ref}], token=2150) == 1

    [(snapshot, payload)] = plane.record.batches[0]
    assert snapshot.is_finished()
    assert snapshot.output_token_ids == [21, 2150]
    torch.testing.assert_close(payload["codes.audio"], torch.cat([first, last]))
    torch.testing.assert_close(payload["codes.ref"], ref)
    assert plane.record.cleaned == ["internal"]
    assert not plane._pending_full_payload_send
    assert plane.request_terminal({"internal"}) == 0


def test_full_payload_abort_discards_partial_and_late_outputs(plane):
    plane._async_chunk = False
    plane._full_payload_replace_keys_cached = frozenset()
    plane.register_request(_new_request())
    assert _complete(plane, [{"codes.audio": torch.tensor([[1, 2]])}]) == 0
    plane.reserve_outputs(["internal"])
    assert plane.abort_requests({"internal"}) == 1
    [(snapshot, payload)] = plane.record.batches[0]
    assert snapshot.is_finished() and payload is None
    assert _complete(plane, [{"codes.audio": torch.tensor([[3, 4]])}]) == 0
    assert len(plane.record.batches) == 1
    assert not plane._pending_full_payload_send


def test_full_payload_qwen_builder_receives_complete_codec(raw_plane, monkeypatch):
    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import talker2code2wav_full_payload

    raw_plane._async_chunk = False
    raw_plane._omni_connector = object()
    raw_plane._request_ids_mapping = {}
    raw_plane._custom_process_func = talker2code2wav_full_payload
    raw_plane._custom_process_batch_func = None
    monkeypatch.setattr(raw_plane, "is_data_transfer_rank", lambda: True)
    sent = []

    def publish(entries, **kwargs):
        sent.extend(entries)
        return len(entries)

    monkeypatch.setattr(raw_plane, "_publish_chunk_cohort", publish)
    raw_plane.register_request(_new_request())
    for token in (21, 22, 2150):
        _complete(raw_plane, [{"codes.audio": torch.full((1, 16), token)}], token=token)
    assert not sent
    assert raw_plane.request_terminal({"internal"}) == 1
    [(request, payload)] = sent
    assert request.output_token_ids == [21, 22, 2150]
    assert payload["meta"]["finished"].item()
    # EOS is filtered, and the two remaining frames are codebook-major.
    assert payload["codes"]["audio"].tolist() == [21, 22] * 16


def _gated_connector():
    # put() blocks its first call until released (or forever without release).
    conn = SimpleNamespace(
        put_keys=[], put_started=threading.Event(), release=threading.Event(), close_called=threading.Event()
    )

    def put(**kwargs):
        conn.put_keys.append(kwargs["put_key"])
        if len(conn.put_keys) == 1:
            conn.put_started.set()
            assert conn.release.wait(timeout=2)
        return True, 1, None

    conn.put, conn.close = put, conn.close_called.set
    return conn


def _start_save_thread(plane, connector):
    plane.__dict__.update(
        _omni_connector=connector,
        _stage_id=0,
        _next_stage_id=1,
        _lock=threading.Lock(),
        _pending_save_reqs={},
        _pending_save_counts=defaultdict(int),
        _deferred_send_cleanup=set(),
        _request_ids_mapping={},
        _cached_ic={},
        _send_side_request_payload={},
        _code_prompt_token_ids=defaultdict(list),
        _work_available=threading.Event(),
        _stop_event=threading.Event(),
        _MAX_SEND_RETRIES=0,
        _can_send=True,
        _recv_thread=None,
        _custom_process_batch_func=lambda **kwargs: kwargs["pooling_outputs"],
        is_data_transfer_rank=lambda: True,
        _connector_send_error_sink=plane._record_output_error,
    )
    plane._save_thread = threading.Thread(target=plane._save_loop, daemon=True)
    plane._save_thread.start()


def _stop_save_thread(plane):
    plane._stop_event.set()
    plane._work_available.set()
    plane._save_thread.join(timeout=2)
    assert not plane._save_thread.is_alive()


class _Output:
    copy_event = None

    def __init__(self, rid="internal", gate=None, started=None, error=None, payload=None, token=1):
        self.rid, self.gate, self.started = rid, gate, started
        self.error, self.payload, self.token = error, payload, token

    def get_output(self):
        if self.started is not None:
            self.started.set()
        if self.gate is not None:
            assert self.gate.wait(timeout=3)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            req_ids=[self.rid],
            inter_stage_outputs=[self.payload] if self.payload is not None else None,
            sampled_token_ids=[[self.token]],
        )


def test_emit_cohort_one_batch_and_trims_non_resumable_history(plane):
    plane.send_chunk = lambda *_a, **_kw: pytest.fail("must not fall back to per-request send_chunk")
    plane.register_request(_new_request("r0", "ext-0"))
    plane.register_request(_new_request("r1", "ext-1"))
    emitted = plane.emit_chunks(
        req_ids=["r0", "r1"],
        inter_stage_outputs=[{"codes.audio": "c0"}, {"codes.audio": "c1"}],
        sampled_token_ids=[[21]],
        terminal_req_ids={"r1"},
    )
    assert emitted == 2 and len(plane.record.batches) == 1  # the whole step cohort goes out in one batch
    (req0, _), (req1, _) = plane.record.batches[0]
    assert (
        req0.all_token_ids == [10, 11, 21]
        and not req0.is_finished()
        and req1.is_finished()
        and plane.record.cleaned == ["r1"]
    )
    # A non-resumable request omits token history after its first chunk.
    plane.register_request(_new_request("r2", "ext-2"))
    state = plane._native_requests["r2"]
    state.resumable, state.output_token_ids, plane._put_req_chunk["ext-2"] = False, list(range(1024)), 1
    plane.emit_chunks(
        req_ids=["r2"], inter_stage_outputs=[{"h": "d"}], sampled_token_ids=[[1024]], terminal_req_ids={"r2"}
    )
    request, _ = plane.record.batches[1][0]
    assert request.output_token_count == 1025 and request.all_token_ids == []


def test_terminal_reserved_abort_and_cleanup_lifecycle(plane):
    batches, cleaned = plane.record.batches, plane.record.cleaned
    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    plane.reserve_outputs(["internal"])
    # The terminal defers while reservations are in flight, and is idempotent.
    assert plane.request_terminal({"internal"}) == 0
    assert plane.request_terminal({"internal"}) == 0 and batches == []
    assert _complete(plane, [{"codes.audio": "c0"}]) == 1 and not batches[0][0][0].is_finished()
    assert _complete(plane, [{"codes.audio": "c1"}], token=22) == 2  # data chunk, then the released terminal
    chunk, terminal = batches[1][0], batches[2][0]
    assert chunk[1] == {"codes": {"audio": "c1"}} and chunk[0].output_token_ids == [21, 22]
    assert terminal[1] is None and terminal[0].is_finished() and cleaned == ["internal"]
    # Stale terminal/complete after cleanup are no-ops; cleanup ran exactly once.
    assert plane.request_terminal({"internal"}) == 0 and _complete(plane, [{"codes.audio": "stale"}]) == 0
    # Abort emits exactly one terminal and silently drops the stale deferred output.
    plane.register_request(_new_request("ra"))
    plane.reserve_outputs(["ra"])
    plane.request_terminal({"ra"})
    assert plane.abort_requests({"ra"}) == 1
    request, payload = batches[-1][0]
    assert request.is_finished() and payload is None and cleaned == ["internal", "ra"]
    assert (
        plane.abort_requests({"ra"}) == 0 and _complete(plane, [{"s": 1}], req="ra") == 0
    )  # duplicate abort, stale output
    assert sum(b[0][0].request_id == "ra" for b in batches) == 1  # terminal exactly once


def test_abort_cannot_overtake_committed_output(plane):
    in_lock, release, sent = threading.Event(), threading.Event(), threading.Event()

    def gated(entries, _orig=plane._send_chunk_entries, **kw):
        terminal = entries[0][0].is_finished()
        if not terminal:
            in_lock.set()
            assert release.wait(timeout=2)  # hold the committed data send inside _native_send_lock
        result = _orig(entries, **kw)
        if terminal:
            sent.set()
        return result

    plane._send_chunk_entries = gated
    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    threads = [
        threading.Thread(target=lambda: _complete(plane, [{"c": 0}])),
        threading.Thread(target=plane.abort_requests, args=({"internal"},)),
    ]
    threads[0].start()
    try:
        assert in_lock.wait(timeout=2)
        threads[1].start()
        overtook = sent.wait(timeout=0.1)  # the terminal must not pass the committed data send
    finally:
        release.set()
        for t in threads:
            if t.ident is not None:
                t.join(timeout=2)
                assert not t.is_alive()
    assert not overtook


def test_output_worker_fifo_and_scheduler_not_blocked(plane):
    batches, send_started, release_send = [], threading.Event(), threading.Event()

    def send_chunks(entries, **_kw):
        if not entries[0][0].is_finished():
            send_started.set()
            assert release_send.wait(timeout=2)
        batches.append(entries)
        return len(entries)

    plane.send_chunks = send_chunks
    plane._start_output_worker(max_pending_batches=2)
    plane.register_request(_new_request())
    plane.reserve_outputs(["internal"])
    plane.enqueue_outputs(req_ids=["internal"], inter_stage_outputs=[{"c": 0}], sampled_token_ids=[[21]])
    assert send_started.wait(timeout=2)
    # A stuck deferred send must not block the scheduler path.
    assert plane.request_terminal({"internal"}) == 0 and batches == []
    release_send.set()
    plane.drain_outputs()
    assert [e[0].is_finished() for batch in batches for e in batch] == [False, True]


def test_worker_failure_surfaces_and_close_drains_in_order(plane):
    events = []
    plane.send_chunks = lambda _e, **_kw: (_ for _ in ()).throw(RuntimeError("enqueue failed"))
    plane.shutdown_omni_connectors = lambda: events.append("shutdown")
    plane._start_output_worker(max_pending_batches=2)
    plane.register_request(_new_request())
    plane.enqueue_outputs(req_ids=["internal"], inter_stage_outputs=[{"c": 0}], sampled_token_ids=[[21]])
    with pytest.raises(RuntimeError, match="enqueue failed"):
        plane.drain_outputs()
    with pytest.raises(RuntimeError, match="enqueue failed"):  # an errored worker rejects new submissions
        plane.enqueue_outputs(req_ids=["internal"], inter_stage_outputs=[{"c": 1}], sampled_token_ids=[[22]])
    # close() drains pending batches before shutting connectors down.
    plane._native_output_error = None

    def send_chunks(entries, **_kw):
        events.append("send")
        return len(entries)

    plane.send_chunks = send_chunks
    plane.enqueue_outputs(req_ids=["internal"], inter_stage_outputs=[{"c": 1}], sampled_token_ids=[[22]])
    plane.close()
    assert events == ["send", "shutdown"]


def test_permanent_put_failure_quarantines_and_holds_state(raw_plane):
    puts = []

    def put(**_kw):
        puts.append(1)
        return False, 0, None

    connector = SimpleNamespace(put=put, close=lambda: None)
    _start_save_thread(raw_plane, connector)
    raw_plane.register_request(_new_request())
    raw_plane.reserve_outputs(["internal"])
    raw_plane.request_terminal({"internal"})
    with pytest.raises(RuntimeError, match="connector send failed"):
        raw_plane.complete_outputs(req_ids=["internal"], inter_stage_outputs=[{"c": 0}], sampled_token_ids=[[21]])
    assert len(puts) == 1 and raw_plane._native_output_error is not None  # fail fast, no retry
    assert raw_plane._native_outputs_in_flight["internal"] == 1 and "internal" in raw_plane._native_terminal_pending
    with pytest.raises(RuntimeError, match="quarantined"):  # the quarantine rejects new tickets
        raw_plane._delivery_manager.create_ticket(request_id="other", put_key="other_0_0")
    _stop_save_thread(raw_plane)


def test_delivery_timeout_drops_queued_ticket(raw_plane):
    raw_plane._delivery_manager = OmniDeliveryManager(delivery_timeout_s=0.05, shutdown_timeout_s=0.5)
    connector = _gated_connector()
    _start_save_thread(raw_plane, connector)
    req = SimpleNamespace(request_id="external")
    _, first = raw_plane._enqueue_chunk_payload(req, {"c": 1}, wait_for_delivery=True)
    _, second = raw_plane._enqueue_chunk_payload(req, {"c": 2}, wait_for_delivery=True)
    assert connector.put_started.wait(timeout=1)
    with pytest.raises(TimeoutError, match="delivery timed out"):
        first.wait()
    assert raw_plane._delivery_manager.is_quarantined
    connector.release.set()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and "external" in raw_plane._pending_save_counts:  # the save loop drops it
        time.sleep(0.001)
    # After quarantine the queued ticket is never put and is marked FAILED.
    assert connector.put_keys == [first.put_key] and second.state is DeliveryState.FAILED
    assert "external" not in raw_plane._pending_save_counts and "external" not in raw_plane._pending_save_reqs
    _stop_save_thread(raw_plane)


def test_close_cancels_waiting_delivery_once_and_shutdown_stays_bounded():
    # A queued delivery waiter is cancelled exactly once on shutdown.
    manager = OmniDeliveryManager(delivery_timeout_s=30.0, shutdown_timeout_s=0.5)
    ticket = manager.create_ticket(request_id="r", put_key="k")
    manager.shutdown(RuntimeError("close"))
    manager.shutdown(RuntimeError("again"))  # idempotent: cancelled at most once
    with pytest.raises(DeliveryCancelledError, match="cancelled"):
        ticket.wait()
    # A connector put() that cannot be cancelled must not hang shutdown.
    plane = _bare_plane(delivery_timeout_s=0.05, shutdown_timeout_s=0.2)
    connector = _gated_connector()
    _start_save_thread(plane, connector)
    assert plane._enqueue_chunk_payload(SimpleNamespace(request_id="external"), {"c": 0}, wait_for_delivery=True)[0]
    assert connector.put_started.wait(timeout=1)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="shutdown exceeded"):
        plane.shutdown_omni_connectors()
    assert time.monotonic() - start < 1.5 and connector.close_called.is_set()
    connector.release.set()
    _stop_save_thread(plane)


@pytest.mark.parametrize("abort", [False, True])
def test_materialization_fences_terminal_and_abort(plane, abort):
    batches, cleaned, gate, started = plane.record.batches, plane.record.cleaned, threading.Event(), threading.Event()
    plane._start_output_worker(max_pending_batches=2)
    materializer = NativeOutputWorker(2)
    try:
        plane.register_request(_new_request())
        plane.reserve_outputs(["internal"])
        output = materializer.submit(
            _Output(gate=gate, started=started, payload={"codes.audio": "c0"}, token=21), plane
        )
        assert started.wait(timeout=3)
        assert plane.request_terminal({"internal"}) == 0  # fenced behind the in-flight materialization
        assert not abort or plane.abort_requests({"internal"}) == 1
        gate.set()
        output.get_output()
        materializer.close()
        plane.drain_outputs()
        assert cleaned == ["internal"] and len(batches) == (1 if abort else 2)
        assert abort or (batches[0][0][1] == {"codes": {"audio": "c0"}} and not batches[0][0][0].is_finished())
        assert batches[-1][0][0].is_finished()
    finally:
        gate.set()
        materializer.close()


def test_native_worker_fifo_thread_affinity_and_signals():
    owner, gate = threading.get_ident(), threading.Event()
    calls: list[tuple[Any, ...]] = []

    def get_signals():
        calls.append(("signals", threading.get_ident()))
        return "ready"

    plane = SimpleNamespace(
        enqueue_outputs=lambda **kw: calls.append(("publish", kw["req_ids"], threading.get_ident())),
        get_omni_connector_output=get_signals,
    )
    worker = NativeOutputWorker(2)
    try:
        a = worker.submit(_Output("a", gate=gate, payload={"v": 1}), plane)
        b = worker.submit(_Output("b", payload={"v": 1}), plane)
        gate.set()
        b.get_output()  # publication of b implies a already published (FIFO, single worker thread)
        publishes = [c for c in calls if c[0] == "publish"]
        assert [c[1] for c in publishes] == [["a"], ["b"]]
        assert all(c[2] != owner for c in publishes)  # published on the worker thread
        result = a.get_output()
        assert a.get_output() is result and result.inter_stage_outputs is None  # resolution is cached
        assert result.omni_connector_output == "ready"
        assert calls[-2:] == [("signals", owner)] * 2  # signals resolve lazily on the caller thread
    finally:
        gate.set()
        worker.close()


def test_native_worker_capacity_error_cache_and_tp_gates():
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    plane = SimpleNamespace(enqueue_outputs=lambda **kw: None, get_omni_connector_output=lambda: None)
    worker, gate = NativeOutputWorker(1), threading.Event()
    try:
        first = worker.submit(_Output("a", gate=gate), plane)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(worker.submit, _Output("b"), plane)
            time.sleep(0.05)
            assert not future.done()  # capacity 1: pending materialization is bounded
            gate.set()
            second = future.result(3)
        assert first.get_output().req_ids == ["a"] and second.get_output().req_ids == ["b"]
        bad = worker.submit(_Output("x", error=ValueError("crash")), plane)
    finally:
        gate.set()
        worker.close()
    # A failed materialization is cached: repeated reads raise the same error.
    for _ in range(2):
        with pytest.raises(ValueError, match="crash"):
            bad.get_output()
    runner = SimpleNamespace(
        _omni_data_plane=object(), vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2))
    )
    assert OmniGPUModelRunner._uses_native_output_materializer(runner) is False  # TP>1 keeps TP consumers
    runner.vllm_config.parallel_config.tensor_parallel_size = 1
    assert OmniGPUModelRunner._uses_native_output_materializer(runner) is True


def test_abort_before_first_chunk_cleans_receiver_state():
    plane = _bare_plane()
    plane._stage_id, plane._lock, plane._work_available = 1, threading.Lock(), threading.Event()
    for name in (
        "_pending_full_payload_send",
        "_request_ids_mapping",
        "_pending_save_counts",
        "_send_side_request_payload",
        "_code_prompt_token_ids",
        "_cached_ic",
        "_adaptive_states",
        "_kv_pending_transfers",
        "_get_req_chunk",
        "_pending_load_reqs",
        "_local_stage_payload_cache",
        "_local_request_metadata",
    ):
        setattr(plane, name, {})
    for name in (
        "_deferred_send_cleanup",
        "_kv_active_transfers",
        "_kv_completed_transfers",
        "_kv_triggered_requests",
        "_finished_load_reqs",
        "_chunk_ready_req_ids",
        "_chunk_finished_req_ids",
        "_chunk_stream_completed",
        "_stage_recv_req_ids",
        "_full_payload_pending_broadcast_req_ids",
        "_async_chunk_updated_req_ids",
    ):
        setattr(plane, name, set())
    plane.register_receivers([SimpleNamespace(request_id="r", external_req_id="external")])
    plane._local_stage_payload_cache["r"] = {"codes": torch.ones(1)}
    assert "r" in plane._pending_load_reqs and plane._request_ids_mapping["r"] == "external"
    assert plane.abort_requests({"r"}) == 0
    assert plane.abort_requests({"r"}) == 0
    assert not plane._pending_load_reqs and not plane._request_ids_mapping and not plane._local_stage_payload_cache

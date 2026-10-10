# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The first-chunk fast path emits audio out of band; the main path resumes the stream."""

import queue
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from tests.model_executor.models.moss_tts.test_streaming_terminal_batch import session as cpu_session
from vllm_omni.model_executor.models.moss_tts.first_chunk_fast_path import MossFirstChunkFastPath, _SlotHandoff
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import MossTTSCodecDecoder
from vllm_omni.worker_v2.first_audio_sender import engine_output_queue_sink

pytestmark = pytest.mark.core_model

N_VQ, SAMPLES, CHANNELS = 2, 4, 2


class _Session:
    def __init__(self, capacity):
        self.free = list(range(capacity))
        self.reset_waits = []

    def acquire(self, *, oldest=False):
        return self.free.pop(0 if oldest else -1) if self.free else None

    def order_after_reset(self, slot, stream):
        self.reset_waits.append(slot)

    def release(self, slot, **kwargs):
        self.free.append(slot)


class _Wrapper:
    batch_sizes = [1, 2, 4, 8]

    def __init__(self, device):
        self.state = torch.zeros(16, device=device)
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()

    def decode(self, codes, slots):
        self.gate.wait()
        self.calls.append(int(codes.shape[1]))
        base = codes.sum(0)[:, 0].float() + 100 * slots.float()
        channel = torch.arange(CHANNELS, device=codes.device).float()
        audio = base[:, None, None] + channel[None, :, None] + torch.zeros(1, 1, SAMPLES, device=codes.device)
        self.state[slots] = 1
        return audio.to(torch.bfloat16), None, int(codes.shape[1])


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def _fast(device, capacity=4):
    session, wrapper = _Session(capacity), _Wrapper(device)
    fast = MossFirstChunkFastPath(
        session,
        wrapper,
        n_vq=N_VQ,
        frames=1,
        codebook_size=1024,
        samples_per_frame=SAMPLES,
        n_channels=CHANNELS,
        sample_rate=torch.tensor(24000, dtype=torch.int32),
        device=device,
    )
    return fast, session, wrapper


class _CallbackSink:
    def __init__(self, callback):
        self.callback = callback

    def prepare(self, request_ids):
        callback = self.callback

        class Delivery:
            routes = {rid: (3 if rid == "r0" else 0) for rid in request_ids}

            def __call__(self, ids, rows, sr):
                for rid, row in zip(ids, rows, strict=True):
                    callback(self.routes[rid], rid, {"model_outputs": row, "sr": sr})

            def fail(self, ids):
                pass

        return Delivery()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_emits_each_first_chunk_and_hands_slot_to_main_path(cuda):
    fast, session, wrapper = _fast(cuda)
    emitted = {}
    done = threading.Event()

    def sink(client, req_id, payload):
        emitted[req_id] = (client, payload["model_outputs"], int(payload["sr"]))
        if len(emitted) == 2:
            done.set()

    slots: dict[str, int] = {}
    assert not fast.submit("r0", "k0", [1, 2], slots)  # unbound: main path owns it
    fast.bind(_CallbackSink(sink))
    wrapper.gate.clear()
    assert fast.submit("r0", "k0", torch.tensor([1, 2]), slots)
    assert fast.submit("r1", "k1", [5, 7], slots)
    assert not fast.submit("r0", "k0", [1, 2], slots)  # already has a slot
    assert not fast.submit("r2", "k2", [1, 2, 3, 4], slots)  # not a 1-frame chunk
    assert slots == {"k0": 0, "k1": 1} and session.free == [2, 3]
    wrapper.gate.set()
    assert done.wait(10)
    fast.close()

    assert wrapper.calls in ([1, 1], [2])
    for req_id, client, code_sum, slot in (("r0", 3, 3, 0), ("r1", 0, 12, 1)):
        got_client, wav, sr = emitted[req_id]
        assert got_client == client and sr == 24000 and wav.device.type == "cpu"
        expected = (code_sum + 100 * slot + torch.arange(CHANNELS).float())[:, None].expand(CHANNELS, SAMPLES)
        torch.testing.assert_close(wav, expected)
    assert sorted(session.reset_waits) == [0, 1]

    # The main path consumes the marker once and is ordered after the decode.
    assert fast.take_decoded("k0") and not fast.take_decoded("k0")
    fast.order_after(0)
    fast.order_after(0)  # no pending handoff: no-op
    assert wrapper.state[:2].tolist() == [1, 1]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_order_after_blocks_until_decode_finishes(cuda):
    fast, _, wrapper = _fast(cuda)
    fast.bind(_CallbackSink(lambda *args: None))
    wrapper.gate.clear()
    assert fast.submit("r0", "k0", [1, 2], {})
    finished = threading.Event()

    def wait_for_slot():
        fast.order_after(0)
        finished.set()

    waiter = threading.Thread(target=wait_for_slot)
    waiter.start()
    assert not finished.wait(0.2)
    wrapper.gate.set()
    assert finished.wait(10)
    waiter.join()
    fast.close()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_no_free_slot_leaves_chunk_to_main_path(cuda):
    fast, _, _ = _fast(cuda, capacity=0)
    fast.bind(_CallbackSink(lambda *args: None))
    slots: dict[str, int] = {}
    assert not fast.submit("r0", "k0", [1, 2], slots)
    assert slots == {} and not fast.take_decoded("k0")
    fast.close()


class _FakeFast:
    frames = 1

    def __init__(self, decoded, unsent=None):
        self.decoded = set(decoded)
        self.unsent = dict(unsent or {})
        self.ordered = []
        self.forgotten = []

    def take_decoded(self, key):
        if key in self.decoded:
            self.decoded.remove(key)
            return True
        return False

    def get_request_slot(self, key, slots):
        return slots.get(key)

    def order_after(self, slot):
        self.ordered.append(slot)

    def take_unsent_audio(self, key):
        return self.unsent.pop(key, None)

    def forget(self, key):
        self.forgotten.append(key)

    def gate(self):
        self.gated = getattr(self, "gated", 0) + 1


def _decoder(s, fast):
    decoder = object.__new__(MossTTSCodecDecoder)
    nn.Module.__init__(decoder)
    decoder._stream_max_step_frames = 15
    decoder._stream_state_capacity = 8
    decoder._stream_req_slots = {}
    decoder._ensure_stream_session = lambda: s
    decoder._first_chunk_fast_path = fast
    decoder._stream_session = s
    decoder._codec_stream = None
    return decoder


@pytest.mark.cpu
def test_main_path_skips_fast_decoded_chunk_and_resumes_state():
    s = cpu_session()
    reference = cpu_session()
    fast = _FakeFast({"a"})
    decoder = _decoder(s, fast)
    # The fast path leased slot 0 and decoded frame 0 of "a" into it.
    slot = s.acquire(oldest=True)
    decoder._stream_req_slots["a"] = slot
    s.step({slot: torch.full((2, 1), 3.0)})
    ref_slot = reference.acquire()
    reference.step({ref_slot: torch.full((2, 1), 3.0)})

    out = decoder._decode_streaming_batch([(0, "a", torch.full((2, 1), 3.0), False), (1, "b", torch.ones(2, 1), False)])
    assert 0 not in out and out[1].tolist() == [[2.0]]
    assert fast.ordered == [slot]

    out = decoder._decode_streaming_batch([(0, "a", torch.ones(2, 3), True)])
    expected = reference.step({ref_slot: torch.ones(2, 3)})[ref_slot]
    torch.testing.assert_close(out[0], expected)
    assert "a" not in decoder._stream_req_slots and fast.ordered == [slot, slot]


@pytest.mark.cpu
def test_abort_orders_release_after_fast_decode():
    s = cpu_session()
    fast = _FakeFast({"a"})
    decoder = _decoder(s, fast)
    slot = decoder._stream_req_slots["a"] = s.acquire(oldest=True)
    decoder.on_requests_finished({"a"})
    assert fast.ordered == [slot] and fast.forgotten == ["a"] and decoder._stream_req_slots == {}
    assert slot in s._free_stream_slots


@pytest.mark.cpu
def test_main_path_returns_unsent_first_chunk_without_decoding_twice():
    s = cpu_session()
    first = torch.full((1, 1), 7.0)
    fast = _FakeFast({"a"}, {"a": first})
    decoder = _decoder(s, fast)
    slot = s.acquire(oldest=True)
    decoder._stream_req_slots["a"] = slot
    s.step({slot: torch.full((2, 1), 3.0)})

    out = decoder._decode_streaming_batch([(0, "a", torch.full((2, 1), 3.0), False)])
    torch.testing.assert_close(out[0], first)
    assert fast.unsent == {} and fast.ordered == [slot]

    # A payload containing a later frame must concatenate it after the
    # preserved first samples, while advancing state only for the later frame.
    first = torch.full((1, 1), 7.0)
    fast.decoded.add("b")
    fast.unsent["b"] = first
    slot_b = s.acquire(oldest=True)
    decoder._stream_req_slots["b"] = slot_b
    s.step({slot_b: torch.full((2, 1), 3.0)})
    out = decoder._decode_streaming_batch([(0, "b", torch.full((2, 2), 3.0), False)])
    torch.testing.assert_close(out[0][..., :1], first)
    assert out[0].shape[-1] == 2


@pytest.mark.cpu
def test_oldest_acquire_reuses_least_recently_released_slot():
    s = cpu_session()
    a, b = s.acquire(), s.acquire()
    s.release(a)
    s.release(b)
    assert s.acquire() == b
    assert s.acquire(oldest=True) == 7  # never-used slots were freed before any release
    assert s._free_stream_slots[-1] == a


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_failed_decode_reports_error_without_replaying_state(cuda):
    fast, _, wrapper = _fast(cuda)

    def fail(codes, slots):
        raise RuntimeError("boom")

    wrapper.decode = fail
    fast.bind(_CallbackSink(lambda *args: pytest.fail("no output expected")))
    slots: dict[str, int] = {}
    assert fast.submit("r0", "k0", [1, 2], slots)
    with pytest.raises(RuntimeError, match="first-chunk decode failed"):
        fast.order_after(slots["k0"])
    assert fast.take_decoded("k0")
    fast.close()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_failed_output_handoff_retains_decoded_samples(cuda):
    fast, _, _ = _fast(cuda)

    def reject(*args):
        raise RuntimeError("output queue closed")

    fast.bind(_CallbackSink(reject))
    slots: dict[str, int] = {}
    assert fast.submit("r0", "k0", [1, 2], slots)
    fast.order_after(slots["k0"])
    assert fast.take_decoded("k0")
    wav = fast.take_unsent_audio("k0")
    assert wav is not None and wav.shape == (CHANNELS, SAMPLES)
    assert fast.take_unsent_audio("k0") is None
    fast.close()


@pytest.mark.cpu
@pytest.mark.parametrize("enabled,complete", [(False, False), (True, False), (True, True), (True, None)])
def test_gate_waits_only_for_enabled_inflight_decode(mocker, enabled, complete):
    # A fixed GPU sleep cannot prove that an event is still pending when the
    # host reaches gate(), particularly across CUDA and ROCm implementations.
    fast = object.__new__(MossFirstChunkFastPath)
    fast._device = torch.device("cuda")
    fast._gate_main = enabled
    event = None if complete is None else mocker.Mock()
    if event is not None:
        event.query.return_value = complete
    fast._inflight = event
    stream = mocker.Mock()
    current_stream = mocker.patch("torch.cuda.current_stream", return_value=stream)
    fast.gate()
    if enabled and complete is False:
        current_stream.assert_called_once_with(fast._device)
        stream.wait_event.assert_called_once_with(event)
    else:
        current_stream.assert_not_called()
        stream.wait_event.assert_not_called()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_gate_hands_inflight_gpu_output_to_main_stream(cuda):
    fast, _, _ = _fast(cuda)
    side = torch.cuda.Stream()
    output = torch.zeros(16, device=cuda)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        output.fill_(7)
        pending = torch.cuda.Event()
        pending.record(side)
    fast._inflight = pending
    fast._gate_main = True
    fast.gate()
    received = output.clone()
    torch.testing.assert_close(received, torch.full_like(received, 7))
    fast.close()


def _cpu_fast(monkeypatch, *, scheduler=None):
    # Replace only CUDA allocations, retaining the real lifecycle and worker.
    import vllm_omni.model_executor.models.moss_tts.first_chunk_fast_path as module

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: None)
    monkeypatch.setattr(torch.cuda, "Event", lambda: None)
    monkeypatch.setattr(module, "current_omni_platform", SimpleNamespace(set_device=lambda device: None))
    session = _Session(2)
    fast = MossFirstChunkFastPath(
        session,
        SimpleNamespace(batch_sizes=[1, 2]),
        n_vq=2,
        frames=1,
        codebook_size=1024,
        samples_per_frame=4,
        n_channels=1,
        sample_rate=torch.tensor(24000),
        device=torch.device("cpu:0"),
        handoff_timeout_s=0.01,
    )
    outputs: queue.Queue = queue.Queue()
    scheduler = scheduler or SimpleNamespace(requests={"r": SimpleNamespace(client_index=3)})
    sink = engine_output_queue_sink(outputs, scheduler, upstream_first_audio=False)
    return fast, session, sink, outputs


def _cancel_during_admission(fast, session, scheduler, sink):
    entered, release = threading.Event(), threading.Event()

    class PausedSink:
        def prepare(self, ids):
            delivery = sink.prepare(ids)
            if delivery.routes:
                entered.set()
                assert release.wait(5)
            return delivery

    fast.bind(PausedSink())
    decoder = _decoder(session, fast)
    errors = []
    finished = threading.Event()
    finishing = threading.Event()

    def submit():
        try:
            assert fast.submit("r", "r", [1, 2], decoder._stream_req_slots)
        except BaseException as error:
            errors.append(error)

    def cancel():
        try:
            finishing.set()
            decoder.on_requests_finished({"r"})
            finished.set()
        except BaseException as error:
            errors.append(error)

    producer = threading.Thread(target=submit)
    cleaner = threading.Thread(target=cancel)
    producer.start()
    try:
        assert entered.wait(5)
        assert decoder._stream_req_slots == {}
        # Scheduler cancellation retires the route before the runner hook.
        scheduler.requests.pop("r")
        cleaner.start()
        assert finishing.wait(5)
        assert not finished.wait(0.1)
        release.set()
        producer.join(5)
        cleaner.join(5)
        assert not producer.is_alive() and not cleaner.is_alive()
        assert not errors, errors
        assert finished.is_set()
        assert decoder._stream_req_slots == {}
        assert fast._decoded == {} and fast._handoffs == {}
        assert sorted(session.free) == [0, 1]
        assert not fast.submit("r", "r", [1, 2], decoder._stream_req_slots)

        # A fresh request can use and return the same capacity immediately.
        scheduler.requests["reuse"] = SimpleNamespace(client_index=3)
        release.set()
        assert fast.submit("reuse", "reuse", [1, 2], decoder._stream_req_slots)
        scheduler.requests.pop("reuse")
        decoder.on_requests_finished({"reuse"})
        assert decoder._stream_req_slots == {} and sorted(session.free) == [0, 1]
    finally:
        release.set()
        producer.join(5)
        if cleaner.ident is not None:
            cleaner.join(5)
        fast.close()


@pytest.mark.cpu
def test_cancel_waits_for_first_chunk_admission_and_reclaims_slot(monkeypatch):
    scheduler = SimpleNamespace(requests={"r": SimpleNamespace(client_index=3)})
    fast, session, sink, _ = _cpu_fast(monkeypatch, scheduler=scheduler)

    # Keep the actual queue/worker/handoff lifecycle, omitting CUDA decode.
    def decode(jobs):
        for job in jobs:
            job.handoff.done.set()

    monkeypatch.setattr(fast, "_decode", decode)
    fast._handoff_timeout_s = 5.0
    _cancel_during_admission(fast, session, scheduler, sink)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_cancel_during_first_chunk_admission_orders_cuda_handoff_and_reuse(cuda):
    fast, session, _ = _fast(cuda, capacity=2)
    scheduler = SimpleNamespace(requests={"r": SimpleNamespace(client_index=3)})
    sink = engine_output_queue_sink(queue.Queue(), scheduler, upstream_first_audio=False)
    _cancel_during_admission(fast, session, scheduler, sink)


@pytest.mark.cpu
def test_slot_timeout_keeps_handoff_and_prevents_slot_reuse(monkeypatch):
    fast, _, _, _ = _cpu_fast(monkeypatch)
    handoff = _SlotHandoff()
    fast._handoffs[0] = handoff
    for _ in range(2):
        with pytest.raises(TimeoutError, match="slot 0"):
            fast.order_after(0)
        assert fast._handoffs[0] is handoff
    handoff.done.set()
    fast.order_after(0)
    assert fast._handoffs == {}


@pytest.mark.cpu
def test_closed_decoder_rejects_jobs_and_cannot_restart(monkeypatch):
    fast, session, sink, _ = _cpu_fast(monkeypatch)
    fast.bind(sink)
    fast.close()
    slots: dict[str, int] = {}
    assert not fast.submit("r", "external", [1, 2], slots)
    assert slots == {} and session.free == [0, 1]
    with pytest.raises(RuntimeError, match="closed"):
        fast.bind(sink)


@pytest.mark.cpu
def test_missing_route_does_not_claim_slot(monkeypatch):
    fast, session, sink, outputs = _cpu_fast(monkeypatch, scheduler=SimpleNamespace(requests={}))
    fast.bind(sink)
    try:
        assert not fast.submit("gone", "external", [1, 2], {})
        assert session.free == [0, 1] and outputs.empty()
    finally:
        fast.close()


@pytest.mark.cpu
def test_decode_failure_wakes_current_and_queued_handoffs(monkeypatch):
    from vllm.v1.engine import FinishReason

    fast, _, sink, outputs = _cpu_fast(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def fail(jobs):
        entered.set()
        assert release.wait(2)
        raise RuntimeError("partial replay failed")

    fast._decode = fail
    fast.bind(sink)
    slots: dict[str, int] = {}
    try:
        assert fast.submit("r", "a", [1, 2], slots)
        assert entered.wait(2)
        assert fast.submit("r", "b", [3, 4], slots)
        release.set()
        fast._thread.join(2)
        assert not fast._thread.is_alive()
        for slot in slots.values():
            with pytest.raises(RuntimeError, match="decode failed"):
                fast.order_after(slot)
        assert all(fast.take_decoded(key) for key in slots)
        assert not fast.submit("r", "c", [1, 2], slots)
        assert outputs.qsize() == 2
        while not outputs.empty():
            assert outputs.get_nowait()[1].outputs[0].finish_reason == FinishReason.ERROR
    finally:
        release.set()
        fast.close()


@pytest.mark.cpu
def test_platform_failure_wakes_pending_handoff(monkeypatch):
    import vllm_omni.model_executor.models.moss_tts.first_chunk_fast_path as module

    fast, _, _, _ = _cpu_fast(monkeypatch)
    fast._handoffs[0] = _SlotHandoff()

    def fail(device):
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(module, "current_omni_platform", SimpleNamespace(set_device=fail))
    fast._run()
    assert fast._closed
    with pytest.raises(RuntimeError, match="decode failed"):
        fast.order_after(0)


@pytest.mark.cpu
@pytest.mark.tts
@pytest.mark.parametrize("limit", [1, 2])
def test_congested_admission_preserves_routes_and_slots_then_recovers(monkeypatch, limit):
    import sys
    from functools import partial

    helpers = sys.modules[__name__]
    from vllm_omni.model_executor.models.moss_tts.first_chunk_fast_path import MossFirstChunkFastPath

    monkeypatch.setattr(helpers, "MossFirstChunkFastPath", partial(MossFirstChunkFastPath, max_active_streams=limit))
    fast, session, sink, _ = helpers._cpu_fast(monkeypatch)
    calls = []

    class Sink:
        def prepare(self, ids):
            calls.append(ids)
            return sink.prepare(ids)

    fast._sink = Sink()
    thread = threading.Thread()
    monkeypatch.setattr(thread, "is_alive", lambda: True)
    fast._thread = thread
    slots = {f"busy{i}": session.acquire() for i in range(limit)}
    before_slots, before_free = slots.copy(), session.free.copy()
    codes = torch.tensor([1, 2])
    assert not fast.submit("r", "r", codes, slots)
    assert slots == before_slots and session.free == before_free
    assert not calls and not fast._decoded and not fast._handoffs and fast._jobs.empty()
    released = slots.pop("busy0")
    session.release(released)
    assert fast.submit("r", "r", codes, slots)
    codes.fill_(99)
    job = fast._jobs.get_nowait()
    assert job.slot == slots["r"] and calls == [["r"]]
    assert job.codes.tolist() == [[1], [2]]

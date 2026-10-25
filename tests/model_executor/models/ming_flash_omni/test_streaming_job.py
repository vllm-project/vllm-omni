# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage of the Ming latent queue and stateful VAE handoff."""

from queue import Empty, Queue
from threading import Event, Thread
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.ming_flash_omni import streaming
from vllm_omni.model_executor.models.ming_flash_omni.streaming import MingStreamingJob, _CancelledError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def job():
    # Constructor owns CUDA-stream setup; exercise the real consumer on CPU.
    value = object.__new__(MingStreamingJob)
    value.device = torch.device("cpu")
    value.poll_seconds = 0.05
    value.queue = Queue(maxsize=32)
    value.cancelled = Event()
    value.done = Event()
    value.error = None
    value.past_key_values = None
    value.stream_state = (None, None, None)
    value.finished = False
    value._has_audio_output = False
    value.low_latency_streaming = True
    value.vae = Mock()
    value.vae.decode.side_effect = lambda latent, **kwargs: (latent, (1, 2, 3), "cache")
    return value


def test_legacy_collection_waits_before_draining_ready_latents(job, monkeypatch):
    job.low_latency_streaming = False
    job.queue.put((torch.ones(1, 1, 1), False))

    def polling_wait(timeout):
        assert timeout == job.poll_seconds
        job.queue.put((torch.full((1, 1, 1), 2.0), True))
        job.done.set()
        return False

    wait = Mock(side_effect=polling_wait)
    monkeypatch.setattr(job.cancelled, "wait", wait)
    audio, last = job.next_audio()
    wait.assert_called_once_with(job.poll_seconds)
    assert last
    torch.testing.assert_close(audio, torch.tensor([[[1.0], [2.0]]]))


def test_legacy_collection_drains_once_without_waiting_for_later_arrivals(job, monkeypatch):
    job.low_latency_streaming = False
    job._has_audio_output = True
    job.queue.put((torch.ones(1, 1, 1), False))
    monkeypatch.setattr(job.cancelled, "wait", Mock(return_value=False))
    get = job.queue.get

    def no_blocking_get(*args, **kwargs):
        assert kwargs.get("timeout") is None, "legacy collection must only drain after its polling wait"
        return get(*args, **kwargs)

    monkeypatch.setattr(job.queue, "get", no_blocking_get)
    audio, last = job.next_audio()
    assert not last and audio.numel() == 1


@pytest.mark.parametrize("has_output", [False, True])
def test_ready_latents_do_not_wait_and_flush_final_once(job, monkeypatch, has_output):
    job._has_audio_output = has_output
    get = job.queue.get

    def no_blocking_get(*args, **kwargs):
        assert kwargs.get("timeout") is None
        return get(*args, **kwargs)

    monkeypatch.setattr(job.queue, "get", no_blocking_get)
    monkeypatch.setattr(job.cancelled, "wait", Mock(side_effect=AssertionError("ready queue must not wait")))
    job.queue.put((torch.ones(1, 2, 1), False))
    job.queue.put((torch.full((1, 1, 1), 2.0), True))
    job.done.set()
    audio, last = job.next_audio()
    assert last and job.finished
    torch.testing.assert_close(audio, torch.tensor([[[1.0], [1.0], [2.0]]]))
    assert job.vae.decode.call_count == 1
    assert job.vae.decode.call_args.kwargs["last_chunk"]


def test_empty_queue_is_woken_by_producer_and_retains_vae_state(job, monkeypatch):
    entered = Event()
    get = job.queue.get

    def blocking_get(*args, **kwargs):
        if kwargs.get("timeout") is not None:
            entered.set()
        return get(*args, **kwargs)

    monkeypatch.setattr(job.queue, "get", blocking_get)
    result = []
    worker = Thread(target=lambda: result.append(job.next_audio()))
    worker.start()
    try:
        assert entered.wait(2)
        job.queue.put((torch.ones(1, 1, 1), False))
        worker.join(2)
        assert not worker.is_alive()
        assert not result[0][1]
        job.queue.put((torch.ones(1, 1, 1), True))
        _, last = job.next_audio()
        assert last
        assert job.vae.decode.call_args.kwargs == {
            "past_key_values": "cache",
            "stream_state": (1, 2, 3),
            "use_cache": True,
            "last_chunk": True,
        }
    finally:
        worker.join(2)


def test_empty_queue_returns_empty_until_producer_finishes(job):
    audio, last = job.next_audio()
    assert audio.shape == (1, 1, 0) and not last
    job.vae.decode.assert_not_called()
    job.done.set()
    with pytest.raises(RuntimeError, match="without a final latent"):
        job.next_audio()


def test_producer_error_is_propagated_without_decoding(job):
    job.error = ValueError("producer failed")
    job.done.set()
    with pytest.raises(ValueError, match="producer failed"):
        job.next_audio()
    job.vae.decode.assert_not_called()


def test_cancelled_consumer_does_not_decode_queued_latents(job):
    job.queue.put((torch.ones(1, 1, 1), True))
    job.cancelled.set()
    with pytest.raises(_CancelledError):
        job.next_audio()
    job.vae.decode.assert_not_called()


@pytest.mark.parametrize(("fragments", "interrupt"), [(1, None), (2, None), (0, "cancel"), (0, "error")])
def test_completion_after_empty_read_keeps_final_latent(job, monkeypatch, fragments, interrupt):
    missed = Event()
    published = Event()
    get_nowait = job.queue.get_nowait
    misses = 0

    def consumer_get():
        nonlocal misses
        try:
            return get_nowait()
        except Empty:
            misses += 1
            if misses == 2:  # Initial miss, timeout, then the drain's miss.
                missed.set()
                assert published.wait(2)
            raise

    def produce():
        if missed.wait(2):
            for index in range(fragments):
                job.queue.put((torch.ones(1, 1, 1), index == fragments - 1))
            if interrupt == "cancel":
                job.cancelled.set()
            elif interrupt == "error":
                job.error = ValueError("producer failed")
            job.done.set()
            published.set()

    monkeypatch.setattr(job.queue, "get_nowait", consumer_get)
    producer = Thread(target=produce)
    producer.start()
    try:
        if interrupt:
            with pytest.raises(_CancelledError if interrupt == "cancel" else ValueError):
                job.next_audio()
            job.vae.decode.assert_not_called()
        else:
            audio, last = job.next_audio()
            assert last and audio.numel() == fragments
            assert job.vae.decode.call_count == 1
    finally:
        producer.join(2)


def test_first_nonempty_audio_is_emitted_without_a_coalescing_wait(job, monkeypatch):
    get = job.queue.get
    allow_coalescing = False
    timeouts: list[float] = []

    def no_blocking_get(*args, **kwargs):
        if kwargs.get("timeout") is not None:
            assert allow_coalescing
            timeouts.append(kwargs["timeout"])
            job.queue.put((torch.ones(1, 1, 1), True))
        return get(*args, **kwargs)

    monkeypatch.setattr(job.queue, "get", no_blocking_get)
    outputs = [torch.empty(1, 1, 0), torch.ones(1, 1, 1), torch.ones(1, 2, 1)]
    job.vae.decode.side_effect = lambda *args, **kwargs: (outputs.pop(0), (1, 2, 3), "cache")
    for frames in (0, 1):
        job.queue.put((torch.ones(1, 1, 1), False))
        audio, last = job.next_audio()
        assert audio.numel() == frames and not last
    allow_coalescing = True
    job.queue.put((torch.ones(1, 1, 1), False))
    audio, last = job.next_audio()
    assert last and audio.numel() == 2
    assert len(timeouts) == 1


def test_later_audio_coalesces_arrivals_and_flushes_final_without_full_wait(job, monkeypatch):
    job._has_audio_output = True
    job.poll_seconds = 5  # Final arrival, rather than this timeout, must finish.
    first_wait, second_wait, finished = Event(), Event(), Event()
    waits = [first_wait, second_wait]
    get = job.queue.get

    def blocking_get(*args, **kwargs):
        if kwargs.get("timeout") is not None:
            waits.pop(0).set()
        return get(*args, **kwargs)

    monkeypatch.setattr(job.queue, "get", blocking_get)
    results, failures = [], []

    def consume():
        try:
            results.append(job.next_audio())
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    job.queue.put((torch.ones(1, 1, 1), False))
    worker = Thread(target=consume)
    worker.start()
    try:
        assert first_wait.wait(2)
        job.queue.put((torch.full((1, 1, 1), 2.0), False))
        assert second_wait.wait(2)
        job.queue.put((torch.full((1, 1, 1), 3.0), True))
        job.done.set()
        assert finished.wait(2)
        assert not failures
        audio, last = results[0]
        assert last
        torch.testing.assert_close(audio, torch.tensor([[[1.0], [2.0], [3.0]]]))
        assert job.vae.decode.call_count == 1
    finally:
        job.cancelled.set()
        job.queue.put((torch.zeros(1, 1, 1), True))
        worker.join(2)


def test_initial_queue_wait_and_coalescing_share_one_deadline(job, monkeypatch):
    job._has_audio_output = True
    job.poll_seconds = 1
    clock = [100.0]
    monkeypatch.setattr(streaming, "monotonic", lambda: clock[0], raising=False)
    get, get_nowait = job.queue.get, job.queue.get_nowait
    timeouts: list[tuple[float, float]] = []

    def initial_empty():
        try:
            return get_nowait()
        except Empty:
            if not timeouts:
                clock[0] = 100.75
            raise

    def blocking_get(*args, **kwargs):
        timeout = kwargs.get("timeout")
        if timeout is not None:
            timeouts.append((clock[0], timeout))
            last = len(timeouts) == 2
            job.queue.put((torch.ones(1, 1, 1), last))
            clock[0] = 101.0 if last else 100.9
        return get(*args, **kwargs)

    monkeypatch.setattr(job.queue, "get_nowait", initial_empty)
    monkeypatch.setattr(job.queue, "get", blocking_get)
    _, last = job.next_audio()
    assert last
    assert timeouts and all(start + timeout <= 101.0 + 1e-9 for start, timeout in timeouts)
    assert job.vae.decode.call_count == 1


@pytest.mark.parametrize("interrupt", ["cancel", "error"])
def test_interrupted_coalescing_does_not_decode_pending_latents(job, monkeypatch, interrupt):
    job._has_audio_output = True
    job.poll_seconds = 5
    waiting = Event()
    get = job.queue.get

    def blocking_get(*args, **kwargs):
        if kwargs.get("timeout") is not None:
            waiting.set()
        return get(*args, **kwargs)

    monkeypatch.setattr(job.queue, "get", blocking_get)
    failures = []

    def consume():
        try:
            job.next_audio()
        except BaseException as exc:
            failures.append(exc)

    job.queue.put((torch.ones(1, 1, 1), False))
    worker = Thread(target=consume)
    worker.start()
    try:
        assert waiting.wait(2)
        if interrupt == "cancel":
            job.cancelled.set()
        else:
            job.error = ValueError("producer failed during collection")
        job.queue.put((torch.ones(1, 1, 1), False))
        worker.join(2)
        assert not worker.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], _CancelledError if interrupt == "cancel" else ValueError)
        job.vae.decode.assert_not_called()
    finally:
        job.cancelled.set()
        job.queue.put((torch.zeros(1, 1, 1), True))
        worker.join(2)

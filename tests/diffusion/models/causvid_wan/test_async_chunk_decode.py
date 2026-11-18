# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for CausVid async stream-decode side path."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.causvid_wan import async_chunk_decode as asd_mod
from vllm_omni.diffusion.models.causvid_wan.async_chunk_decode import (
    _async_host_copy,
    _start_local,
)
from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import CausVidWanPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.request_streaming import bind_emit_chunk, reset_emit_chunk
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeDecoder:
    def __init__(self) -> None:
        self.chunks = 0

    def new_decode_state(self, session_id: str):
        return SimpleNamespace(session_id=session_id)

    def decode_chunk(self, latent: torch.Tensor, state):
        del state
        self.chunks += 1
        # Return a tiny BCTHW stand-in.
        return latent[:, :3].expand(-1, 3, -1, -1, -1).contiguous()

    def release(self, state) -> None:
        del state


class _FakeVae(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_tiling = False
        self.dtype = torch.float32
        self.config = SimpleNamespace(latents_mean=[0.0] * 4, latents_std=[1.0] * 4)
        self._p = torch.nn.Parameter(torch.zeros(1))

    def is_distributed_enabled(self) -> bool:
        return False

    def parameters(self, recurse: bool = True):
        yield self._p


class _Work:
    """One simulated point-to-point transfer."""

    def __init__(self, buffer=None):
        self.buffer = buffer
        self.done = threading.Event()

    def wait(self, timeout=None):
        seconds = timeout.total_seconds() if timeout is not None else 3.0
        if not self.done.wait(seconds):
            raise TimeoutError("unmatched receive")
        return True


class _Transport:
    """FIFO-per-tag stand-in for the stream group's ``irecv``/``isend`` pair.

    Each instance owns its own pending queues, send log and condition, so tests
    that run concurrently cannot observe each other's transfers.
    """

    def __init__(self) -> None:
        self.pending: defaultdict[int, deque[_Work]] = defaultdict(deque)
        self.sent: list[int] = []
        self.cv = threading.Condition()

    def irecv(self, buffer, *, src, group, tag=0):
        del src, group
        work = _Work(buffer)
        with self.cv:
            self.pending[tag].append(work)
        return work

    def isend(self, host, *, dst, group, tag=0):
        del dst, group
        with self.cv:
            recv = self.pending[tag].popleft()
            recv.buffer.copy_(host)
            recv.done.set()
            self.sent.append(tag)
            self.cv.notify_all()
        work = _Work()
        work.done.set()
        return work

    def start_jobs(self, tag: int, shape: tuple[int, ...], *, chunks: int = 2):
        """Start one owner (rank 0) and one source (rank 1) job on ``tag``."""
        kwargs = dict(
            session_id=str(tag),
            n_chunks=chunks,
            latent_shape=shape,
            dtype=torch.float32,
            source_rank=1,
            pp_group=SimpleNamespace(ranks=[0, 1]),
            stream_group=self.cv,
            decode_tag=tag,
        )
        owner = asd_mod.start_async_chunk_decode(vae=_FakeVae(), decoder=_FakeDecoder(), pp_rank=0, **kwargs)
        source = asd_mod.start_async_chunk_decode(vae=None, decoder=None, pp_rank=1, **kwargs)
        return owner, source

    def emit(self, source, tag: int, chunk: int, value: float, shape: tuple[int, ...]) -> None:
        """Publish one chunk and assert it arrived under ``tag``."""
        with self.cv:
            before = len(self.sent)
        source.on_finished(chunk, torch.full(shape, value))
        with self.cv:
            assert self.cv.wait_for(lambda: len(self.sent) > before, timeout=2)
            assert self.sent[-1] == tag


def _capture_thread(*args, real_thread, started, **kwargs):
    """Thread factory that records every thread it creates."""
    thread = real_thread(*args, **kwargs)
    started.append(thread)
    return thread


def _raise_wancell(*_args, **_kwargs):
    """Stand-in ``WanCell`` whose construction fails."""
    raise RuntimeError("wancell boom")


def _make_cancelling_job(cancelled: list[bool]):
    """Stream job whose only observable is that ``cancel`` was called."""
    return SimpleNamespace(on_finished=None, cancel=partial(cancelled.append, True))


def _unpinned_empty(*args, **kwargs):
    """``torch.empty`` without ``pin_memory``, which the CPU test device rejects."""
    return _REAL_EMPTY(*args, **{k: v for k, v in kwargs.items() if k != "pin_memory"})


_REAL_EMPTY = torch.empty


def test_async_host_copy_cpu_path():
    latent = torch.randn(1, 4, 2, 8, 8)
    host, event = _async_host_copy(latent, shape=None, dtype=None)
    assert event is None
    assert host.device.type == "cpu"
    assert torch.equal(host, latent)


@pytest.mark.parametrize(
    "session_ids,expected_seen",
    [
        ([0, 1], [0, 1]),
        ([5, 6], [0, 1]),  # session-absolute ids remap to request-relative
    ],
)
def test_local_on_chunk_emits_each_chunk_and_join_returns_none(session_ids, expected_seen):
    vae = _FakeVae()
    decoder = _FakeDecoder()
    seen: list[tuple[int, tuple[int, ...]]] = []

    def on_chunk(idx: int, pixels: torch.Tensor) -> None:
        seen.append((idx, tuple(pixels.shape)))
        assert pixels.shape[1] == 3

    job = _start_local(vae=vae, decoder=decoder, session_id="req-chunk", on_chunk=on_chunk)
    assert job.streams_to_client is True
    assert job.on_finished is not None
    for session_id in session_ids:
        job.on_finished(session_id, torch.randn(1, 4, 2, 4, 4))
    assert job.join() is None
    assert [idx for idx, _ in seen] == expected_seen
    assert all(shape == (1, 3, 2, 4, 4) for _, shape in seen)
    assert decoder.chunks == 2
    assert job.emitted_chunks == 2


def test_local_sidepath_does_not_decode_inside_on_finished():
    vae = _FakeVae()
    decoder = _FakeDecoder()
    job = _start_local(vae=vae, decoder=decoder, session_id="req-1")
    assert job.on_finished is not None
    # on_finished must return without waiting on decode.
    job.on_finished(0, torch.randn(1, 4, 2, 4, 4))
    job.on_finished(1, torch.randn(1, 4, 2, 4, 4))
    assert decoder.chunks == 0
    video = job.join()
    assert video is not None
    assert decoder.chunks == 2
    assert video.shape[1] == 3


def test_local_cancel_stops_consumer_without_join(monkeypatch):
    """cancel() must exit the consumer via the stop sentinel (not only bound chunks)."""
    started: list[threading.Thread] = []

    monkeypatch.setattr(
        asd_mod.threading,
        "Thread",
        partial(_capture_thread, real_thread=threading.Thread, started=started),
    )

    vae = _FakeVae()
    decoder = _FakeDecoder()
    job = _start_local(vae=vae, decoder=decoder, session_id="req-cancel")
    assert job.on_finished is not None
    consumers = [t for t in started if t.name == "causvid-local-stream-decode"]
    assert len(consumers) == 1 and consumers[0].is_alive()
    job.on_finished(0, torch.randn(1, 4, 2, 4, 4))
    job.cancel()
    # Second cancel is a no-op.
    job.cancel()
    # A missed stop sentinel would leave the consumer blocked past cancel's 5s join.
    assert not consumers[0].is_alive()
    assert decoder.chunks <= 1


def test_pipeline_forward_cancels_stream_job_when_cell_compute_fails(monkeypatch):
    """Creating a stream job then failing in the cell loop must cancel the job."""
    pipeline = CausVidWanPipeline.__new__(CausVidWanPipeline)
    pipeline._chunk_ctx = SimpleNamespace()
    cancelled: list[bool] = []
    job = SimpleNamespace(cancel=lambda: cancelled.append(True))
    from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell

    adapter = WanCell.__new__(WanCell)

    monkeypatch.setattr(
        pipeline,
        "admit_cell_request",
        lambda _req, **_kwargs: (adapter, "kv-1", False, job),
    )

    def _boom_cell(**_kwargs):
        raise RuntimeError("compute boom")

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.run_cell_pipeline",
        _boom_cell,
    )

    req = OmniDiffusionRequest(
        prompt="x",
        request_id="stream-fail",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"async_chunk_decode": True}),
    )
    with pytest.raises(RuntimeError, match="compute boom"):
        pipeline.forward(req)
    assert cancelled == [True]


def test_chunk_decode_cancelled_if_wancell_fails(monkeypatch):
    """Narrow seam: stream start + WanCell build; construction failure cancels the job."""
    cancelled: list[bool] = []
    fake_job = _make_cancelling_job(cancelled)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.start_async_chunk_decode",
        lambda **_kwargs: fake_job,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.WanCell",
        _raise_wancell,
    )

    pipeline = CausVidWanPipeline.__new__(CausVidWanPipeline)
    pipeline.vae = None
    pipeline.transformer = SimpleNamespace()
    with pytest.raises(RuntimeError, match="wancell boom"):
        pipeline._start_chunk_decode_and_wan_cell(
            kv_id="kv-admit",
            n_chunks=1,
            latent_shape=(1, 4, 1, 1, 1),
            dtype=torch.float32,
            device=torch.device("cpu"),
            seed=0,
            prompt_embeds=torch.zeros(1, 1, 1),
            sampler=SimpleNamespace(),
            async_chunk_decode=True,
            finish_rank=0,
            source_rank=None,
            pp_rank=0,
            pp_group=None,
            timeline_file=None,
        )
    assert cancelled == [True]


@pytest.mark.parametrize("cancel_a", [False, True])
def test_remote_decode_isolates_interleaved_members_and_reuses_drained_tags(monkeypatch, cancel_a):
    """Real job threads over a FIFO-per-tag transport preserve request ownership."""
    transport = _Transport()
    monkeypatch.setattr(asd_mod.torch, "empty", _unpinned_empty)
    monkeypatch.setattr(asd_mod.dist, "irecv", transport.irecv)
    monkeypatch.setattr(asd_mod.dist, "isend", transport.isend)
    shape = (1, 4, 1, 2, 2)

    owner_a, source_a = transport.start_jobs(0, shape)
    owner_b, source_b = transport.start_jobs(1, shape)
    try:
        transport.emit(source_a, 0, 0, 0.1, shape)
        transport.emit(source_b, 1, 0, 0.3, shape)
        if cancel_a:
            # Stop production first; its filler completes A's outstanding recv.
            source_a.cancel()
            owner_a.cancel()
        else:
            transport.emit(source_a, 0, 1, 0.2, shape)
            source_a.join()
            torch.testing.assert_close(owner_a.join()[0, 0, :, 0, 0], torch.tensor([0.1, 0.2]))
        transport.emit(source_b, 1, 1, 0.4, shape)
        source_b.join()
        torch.testing.assert_close(owner_b.join()[0, 0, :, 0, 0], torch.tensor([0.3, 0.4]))
        assert all(not receives for receives in transport.pending.values())
        # Same tags can serve the next wave only after all old receives drain.
        owner_next, source_next = transport.start_jobs(0, shape, chunks=1)
        try:
            transport.emit(source_next, 0, 0, 0.7, shape)
            source_next.join()
            torch.testing.assert_close(owner_next.join(), torch.full((1, 3, 1, 2, 2), 0.7))
        finally:
            source_next.cancel()
            owner_next.cancel()
    finally:
        source_a.cancel()
        source_b.cancel()
        owner_a.cancel()
        owner_b.cancel()


def test_chunk_decode_on_chunk_when_request_streaming_bound(monkeypatch):
    captured: dict[str, object] = {}
    emitted: list[DiffusionOutput] = []

    def _fake_start(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(on_finished=None, cancel=lambda: None)

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.start_async_chunk_decode",
        _fake_start,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.WanCell",
        lambda *_a, **_k: SimpleNamespace(),
    )
    pipeline = CausVidWanPipeline.__new__(CausVidWanPipeline)
    pipeline.vae = None
    pipeline.transformer = SimpleNamespace()
    pipeline.od_config = SimpleNamespace(streaming_output=True)
    token = bind_emit_chunk(emitted.append)
    try:
        pipeline._start_chunk_decode_and_wan_cell(
            kv_id="kv-stream",
            n_chunks=3,
            latent_shape=(1, 4, 1, 1, 1),
            dtype=torch.float32,
            device=torch.device("cpu"),
            seed=0,
            prompt_embeds=torch.zeros(1, 1, 1),
            sampler=SimpleNamespace(),
            async_chunk_decode=True,
            finish_rank=0,
            source_rank=None,
            pp_rank=0,
            pp_group=None,
            timeline_file=None,
        )
        on_chunk = captured["on_chunk"]
        assert callable(on_chunk)
        on_chunk(1, torch.zeros(1, 3, 2, 2, 2))
    finally:
        reset_emit_chunk(token)

    assert len(emitted) == 1
    assert emitted[0].finished is False
    assert emitted[0].chunk_index == 1
    assert emitted[0].total_chunks == 3


def test_remote_cancel_drains_posted_irecvs(monkeypatch):
    """Mid-request cancel must finish every posted irecv (FIFO mock)."""
    from vllm_omni.diffusion.models.causvid_wan.async_chunk_decode import start_async_chunk_decode

    monkeypatch.setattr(asd_mod.torch, "empty", _unpinned_empty)

    class _ReceiveWork:
        def __init__(self, payload):
            self.payload = payload
            self._done = threading.Event()

        def wait(self, timeout=None):
            seconds = timeout.total_seconds() if timeout is not None else 5.0
            assert self._done.wait(timeout=seconds)

        def complete(self):
            self._done.set()

    pending: list[_ReceiveWork] = []
    completed: list[_ReceiveWork] = []

    def fake_irecv(buf, src, group, tag=0):
        del src, group, tag
        work = _ReceiveWork(buf)
        pending.append(work)
        return work

    def fake_isend(host, dst, group, tag=0):
        del dst, group, tag
        if not pending:
            raise AssertionError("isend with no matching irecv")
        work = pending.pop(0)
        work.payload.copy_(host)
        work.complete()
        completed.append(work)
        return work

    monkeypatch.setattr(asd_mod.dist, "irecv", fake_irecv)
    monkeypatch.setattr(asd_mod.dist, "isend", fake_isend)

    shape = (1, 4, 2, 4, 4)
    n_chunks = 3
    pp_group = SimpleNamespace(ranks=[0, 1])
    owner = start_async_chunk_decode(
        vae=_FakeVae(),
        decoder=_FakeDecoder(),
        session_id="drain",
        n_chunks=n_chunks,
        latent_shape=shape,
        dtype=torch.float32,
        pp_rank=0,
        source_rank=1,
        pp_group=pp_group,
        stream_group=object(),
        on_chunk=lambda *_: None,
    )
    source = start_async_chunk_decode(
        vae=None,
        decoder=None,
        session_id="drain",
        n_chunks=n_chunks,
        latent_shape=shape,
        dtype=torch.float32,
        pp_rank=1,
        source_rank=1,
        pp_group=pp_group,
        stream_group=object(),
    )
    assert len(pending) == n_chunks
    assert source.on_finished is not None
    source.on_finished(0, torch.ones(shape))
    # Owner cancel waits on remaining irecvs; run it concurrently so the
    # source cancel can fill the gap without deadlocking this thread.
    owner_done = threading.Event()

    def _cancel_owner():
        owner.cancel()
        owner_done.set()

    threading.Thread(target=_cancel_owner, daemon=True).start()
    source.cancel()
    assert owner_done.wait(timeout=5.0)
    assert not pending
    assert len(completed) == n_chunks


def test_remote_cancel_refuses_fill_while_sender_stuck_on_copy(monkeypatch):
    """Stuck D2H event must not filler-then-late-isend (would pollute the group)."""
    import time

    from vllm_omni.diffusion.models.causvid_wan.async_chunk_decode import start_async_chunk_decode

    monkeypatch.setattr(asd_mod, "_SENDER_CANCEL_JOIN_TIMEOUT_S", 0.05)
    sends: list[torch.Tensor] = []

    class _Work:
        def wait(self):
            return None

    def fake_isend(host, dst, group, tag=0):
        del dst, group, tag
        sends.append(host.detach().clone())
        return _Work()

    monkeypatch.setattr(asd_mod.dist, "isend", fake_isend)

    hold = threading.Event()
    copy_entered = threading.Event()

    class _DelayedEvent:
        def synchronize(self):
            copy_entered.set()
            assert hold.wait(timeout=5.0)

    monkeypatch.setattr(
        asd_mod,
        "_async_host_copy",
        lambda latent, *, shape, dtype: (torch.zeros(shape, dtype=dtype), _DelayedEvent()),
    )

    shape = (1, 4, 2, 4, 4)
    source = start_async_chunk_decode(
        vae=None,
        decoder=None,
        session_id="stuck-sender",
        n_chunks=1,
        latent_shape=shape,
        dtype=torch.float32,
        pp_rank=1,
        source_rank=1,
        pp_group=SimpleNamespace(ranks=[0, 1]),
        stream_group=object(),
    )
    assert source.on_finished is not None
    source.on_finished(0, torch.ones(shape))
    assert copy_entered.wait(timeout=2.0)
    with pytest.raises(SystemExit, match="latent-send did not stop"):
        source.cancel()
    assert sends == []
    hold.set()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not sends:
        time.sleep(0.01)
    # Cancelled sender must skip the late real isend; no fillers either.
    assert sends == []


@pytest.mark.parametrize("stalled", [False, True])
def test_local_cancel_quiesces_decode_or_retires_worker(monkeypatch, stalled):
    entered, release, joined, cache_released, done = (threading.Event() for _ in range(5))
    seen, failures = [], []
    original = RuntimeError("original forward failure")
    real_join = asd_mod._join_on_cancel

    class Decoder(_FakeDecoder):
        def decode_chunk(self, latent, state):
            entered.set()
            assert release.wait(timeout=3.0)
            return super().decode_chunk(latent, state)

        def release(self, state):
            super().release(state)
            cache_released.set()

    def join_thread(thread, **kwargs):
        joined.set()
        real_join(thread, **kwargs)

    monkeypatch.setattr(asd_mod, "_join_on_cancel", join_thread)
    monkeypatch.setattr(asd_mod, "_SENDER_CANCEL_JOIN_TIMEOUT_S", 0.02 if stalled else 1.0)
    decoder = Decoder()
    job = _start_local(vae=_FakeVae(), decoder=decoder, session_id="cancel", on_chunk=lambda *args: seen.append(args))
    assert job.on_finished is not None
    job.on_finished(0, torch.ones(1, 4, 1, 2, 2))
    assert entered.wait(timeout=2.0)
    job.on_finished(1, torch.ones(1, 4, 1, 2, 2))

    def cancel():
        try:
            raise original
        except RuntimeError:
            try:
                job.cancel()
            except BaseException as exc:
                failures.append(exc)
            finally:
                done.set()

    thread = threading.Thread(target=cancel)
    thread.start()
    try:
        assert joined.wait(timeout=2.0)
        if stalled:
            assert done.wait(timeout=1.0)
            assert len(failures) == 1 and isinstance(failures[0], SystemExit)
            assert failures[0].__context__ is original
            assert not cache_released.is_set()
        else:
            assert not done.is_set(), "cancel returned while decoder still owns the VAE"
            release.set()
            assert done.wait(timeout=2.0)
            assert failures == []
            assert cache_released.is_set()
    finally:
        release.set()
        thread.join(timeout=2.0)
        assert cache_released.wait(timeout=2.0)
    assert decoder.chunks == 1, "queued work was decoded after cancellation"
    assert seen == [], "cancelled decode emitted a late chunk"


def test_remote_owner_cancel_retires_worker_if_receive_cannot_drain(monkeypatch):
    monkeypatch.setattr(asd_mod.torch, "empty", _unpinned_empty)
    monkeypatch.setattr(asd_mod, "_CANCEL_DRAIN_TIMEOUT_S", 0.02)
    transport = _Transport()
    monkeypatch.setattr(asd_mod.dist, "irecv", transport.irecv)
    released = threading.Event()
    decoder = _FakeDecoder()
    monkeypatch.setattr(decoder, "release", lambda state: released.set())
    owner = asd_mod.start_async_chunk_decode(
        vae=_FakeVae(),
        decoder=decoder,
        session_id="missing-source",
        n_chunks=1,
        latent_shape=(1, 4, 1, 2, 2),
        dtype=torch.float32,
        pp_rank=0,
        source_rank=1,
        pp_group=SimpleNamespace(ranks=[0, 1]),
        stream_group=object(),
    )
    try:
        with pytest.raises(SystemExit, match="retiring Worker"):
            owner.cancel()
    finally:
        for work in transport.pending[0]:
            work.done.set()
        assert released.wait(timeout=2.0)


@pytest.mark.parametrize("failed_chunk", [1, 2])
def test_remote_decode_error_remains_recoverable_after_cancel(monkeypatch, failed_chunk):
    transport = _Transport()
    monkeypatch.setattr(asd_mod.torch, "empty", _unpinned_empty)
    monkeypatch.setattr(asd_mod.dist, "irecv", transport.irecv)
    monkeypatch.setattr(asd_mod.dist, "isend", transport.isend)
    original_decode = _FakeDecoder.decode_chunk

    def decode(self, latent, state):
        if self.chunks + 1 == failed_chunk:
            raise ValueError("decode failed")
        return original_decode(self, latent, state)

    monkeypatch.setattr(_FakeDecoder, "decode_chunk", decode)
    shape = (1, 4, 1, 2, 2)
    owner, source = transport.start_jobs(0, shape)
    try:
        for chunk in range(2):
            transport.emit(source, 0, chunk, 0.1, shape)
        source.join()
        with pytest.raises(ValueError, match="decode failed"):
            owner.join()
        owner.cancel()
        owner.cancel()  # Cleanup is idempotent after draining.
        assert not transport.pending[0]
        with pytest.raises(ValueError, match="decode failed"):
            owner.join()
    finally:
        source.cancel()
        owner.cancel()

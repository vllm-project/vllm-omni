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

from vllm_omni.diffusion.models.causvid_wan import async_stream_decode as asd_mod
from vllm_omni.diffusion.models.causvid_wan.async_stream_decode import (
    _async_host_copy,
    _start_local,
)
from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import CausVidWanPipeline

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
        owner = asd_mod.start_async_stream_decode(vae=_FakeVae(), decoder=_FakeDecoder(), pp_rank=0, **kwargs)
        source = asd_mod.start_async_stream_decode(vae=None, decoder=None, pp_rank=1, **kwargs)
        return owner, source

    def emit(self, source, tag: int, chunk: int, value: float, shape: tuple[int, ...]) -> None:
        """Publish one chunk and assert it arrived under ``tag``."""
        with self.cv:
            before = len(self.sent)
        source.on_finished(chunk, torch.full(shape, value))
        with self.cv:
            assert self.cv.wait_for(lambda: len(self.sent) > before, timeout=2)
            assert self.sent[-1] == tag


class _FailedTransfer:
    """Transfer whose peer never drains."""

    def wait(self, timeout):
        del timeout
        raise TimeoutError("peer did not drain")


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


def test_stream_decode_cancelled_if_wancell_fails(monkeypatch):
    """Narrow seam: stream start + WanCell build; construction failure cancels the job."""
    cancelled: list[bool] = []
    fake_job = _make_cancelling_job(cancelled)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.start_async_stream_decode",
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
        pipeline._start_stream_decode_and_wan_cell(
            kv_id="kv-admit",
            n_chunks=1,
            latent_shape=(1, 4, 1, 1, 1),
            dtype=torch.float32,
            device=torch.device("cpu"),
            seed=0,
            prompt_embeds=torch.zeros(1, 1, 1),
            sampler=SimpleNamespace(),
            stream_decode=True,
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


def test_cancelled_latent_drain_failure_stops_worker():
    with pytest.raises(SystemExit, match="stop the Worker"):
        asd_mod._drain_latent_work(_FailedTransfer())

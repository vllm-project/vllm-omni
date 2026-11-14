# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for CausVid async stream-decode side path."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.causvid_wan import async_stream_decode as asd_mod
from vllm_omni.diffusion.models.causvid_wan.async_stream_decode import (
    _async_host_copy,
    _start_local,
)
from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import CausVidWanPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
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
    real_thread = threading.Thread

    def _capture_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        started.append(thread)
        return thread

    monkeypatch.setattr(asd_mod.threading, "Thread", _capture_thread)

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
    from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell

    pipeline = CausVidWanPipeline.__new__(CausVidWanPipeline)
    pipeline._chunk_ctx = SimpleNamespace()
    cancelled: list[bool] = []
    job = SimpleNamespace(cancel=lambda: cancelled.append(True))
    # forward() requires a WanCell adapter; fields are unused when compute is mocked.
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
        sampling_params=OmniDiffusionSamplingParams(extra_args={"stream_decode": True}),
    )
    with pytest.raises(RuntimeError, match="compute boom"):
        pipeline.forward(req)
    assert cancelled == [True]


def test_stream_decode_cancelled_if_wancell_fails(monkeypatch):
    """Narrow seam: stream start + WanCell build; construction failure cancels the job."""
    cancelled: list[bool] = []
    fake_job = SimpleNamespace(on_finished=None, cancel=lambda: cancelled.append(True))
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.start_async_stream_decode",
        lambda **_kwargs: fake_job,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.WanCell",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("wancell boom")),
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
    from collections import defaultdict, deque

    pending: defaultdict[int, deque[Work]] = defaultdict(deque)
    sent = []
    cv = threading.Condition()
    real_empty = torch.empty
    monkeypatch.setattr(
        asd_mod.torch, "empty", lambda *a, **kw: real_empty(*a, **{k: v for k, v in kw.items() if k != "pin_memory"})
    )

    class Work:
        def __init__(self, buffer=None):
            self.buffer = buffer
            self.done = threading.Event()

        def wait(self, timeout=None):
            seconds = timeout.total_seconds() if timeout is not None else 3.0
            if not self.done.wait(seconds):
                raise TimeoutError("unmatched receive")
            return True

    def irecv(buffer, *, src, group, tag=0):
        del src, group
        work = Work(buffer)
        with cv:
            pending[tag].append(work)
        return work

    def isend(host, *, dst, group, tag=0):
        del dst, group
        with cv:
            recv = pending[tag].popleft()
            recv.buffer.copy_(host)
            recv.done.set()
            sent.append(tag)
            cv.notify_all()
        work = Work()
        work.done.set()
        return work

    monkeypatch.setattr(asd_mod.dist, "irecv", irecv)
    monkeypatch.setattr(asd_mod.dist, "isend", isend)
    shape = (1, 4, 1, 2, 2)

    def jobs(tag, chunks=2):
        kwargs = dict(
            session_id=str(tag),
            n_chunks=chunks,
            latent_shape=shape,
            dtype=torch.float32,
            source_rank=1,
            pp_group=SimpleNamespace(ranks=[0, 1]),
            stream_group=None,
            decode_tag=tag,
        )
        kwargs["stream_group"] = cv
        owner = asd_mod.start_async_stream_decode(vae=_FakeVae(), decoder=_FakeDecoder(), pp_rank=0, **kwargs)
        source = asd_mod.start_async_stream_decode(vae=None, decoder=None, pp_rank=1, **kwargs)
        return owner, source

    def emit(source, tag, chunk, value):
        with cv:
            before = len(sent)
        source.on_finished(chunk, torch.full(shape, value))
        with cv:
            assert cv.wait_for(lambda: len(sent) > before, timeout=2)
            assert sent[-1] == tag

    owner_a, source_a = jobs(0)
    owner_b, source_b = jobs(1)
    try:
        emit(source_a, 0, 0, 0.1)
        emit(source_b, 1, 0, 0.3)
        if cancel_a:
            # Stop production first; its filler completes A's outstanding recv.
            source_a.cancel()
            owner_a.cancel()
        else:
            emit(source_a, 0, 1, 0.2)
            source_a.join()
            torch.testing.assert_close(owner_a.join()[0, 0, :, 0, 0], torch.tensor([0.1, 0.2]))
        emit(source_b, 1, 1, 0.4)
        source_b.join()
        torch.testing.assert_close(owner_b.join()[0, 0, :, 0, 0], torch.tensor([0.3, 0.4]))
        assert all(not receives for receives in pending.values())
        # Same tags can serve the next wave only after all old receives drain.
        owner_next, source_next = jobs(0, chunks=1)
        try:
            emit(source_next, 0, 0, 0.7)
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
    class FailedTransfer:
        def wait(self, timeout):
            raise TimeoutError("peer did not drain")

    with pytest.raises(SystemExit, match="stop the Worker"):
        asd_mod._drain_latent_work(FailedTransfer())

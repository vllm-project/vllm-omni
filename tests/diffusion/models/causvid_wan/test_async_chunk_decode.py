# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for CausVid async stream-decode side path."""

from __future__ import annotations

import threading
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
    pipeline = CausVidWanPipeline.__new__(CausVidWanPipeline)
    pipeline._chunk_ctx = SimpleNamespace()
    cancelled: list[bool] = []
    job = SimpleNamespace(cancel=lambda: cancelled.append(True))

    monkeypatch.setattr(
        pipeline,
        "admit_cell_request",
        lambda _req: (SimpleNamespace(), "kv-1", False, job),
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
    fake_job = SimpleNamespace(on_finished=None, cancel=lambda: cancelled.append(True))
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.start_async_chunk_decode",
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

    pending: list[object] = []
    completed: list[object] = []

    class _Work:
        def __init__(self, payload):
            self.payload = payload
            self._done = threading.Event()

        def wait(self):
            assert self._done.wait(timeout=5.0)

        def complete(self):
            self._done.set()

    def fake_irecv(buf, src, group):
        del src, group
        work = _Work(buf)
        pending.append(work)
        return work

    def fake_isend(host, dst, group):
        del dst, group
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

    def fake_isend(host, dst, group):
        del dst, group
        sends.append(host.detach().clone())
        return _Work()

    monkeypatch.setattr(asd_mod.dist, "isend", fake_isend)

    hold = threading.Event()

    class _DelayedEvent:
        def synchronize(self):
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
    time.sleep(0.05)  # let sender dequeue and block on synchronize
    with pytest.raises(RuntimeError, match="latent sender did not stop"):
        source.cancel()
    assert sends == []
    hold.set()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not sends:
        time.sleep(0.01)
    # Cancelled sender must skip the late real isend; no fillers either.
    assert sends == []

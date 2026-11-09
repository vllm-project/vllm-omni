# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for CausVid async stream-decode side path."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.causvid_wan import async_stream_decode as asd_mod
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.causvid_wan.async_stream_decode import (
    _async_host_copy,
    _start_local,
)
from vllm_omni.diffusion.worker.request_chunk_emit import bind_emit_chunk, reset_emit_chunk
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


def test_local_on_chunk_emits_each_chunk_and_join_returns_none():
    vae = _FakeVae()
    decoder = _FakeDecoder()
    seen: list[int] = []

    def on_chunk(idx: int, pixels: torch.Tensor) -> None:
        seen.append(idx)
        assert pixels.shape[1] == 3

    job = _start_local(vae=vae, decoder=decoder, session_id="req-chunk", on_chunk=on_chunk)
    assert job.on_finished is not None
    job.on_finished(0, torch.randn(1, 4, 2, 4, 4))
    job.on_finished(1, torch.randn(1, 4, 2, 4, 4))
    assert job.join() is None
    assert seen == [0, 1]
    assert decoder.chunks == 2
    assert job.streams_to_client is True
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


def test_stream_decode_on_chunk_when_request_streaming_bound(monkeypatch):
    captured: dict[str, object] = {}
    emitted: list[DiffusionOutput] = []

    def _fake_start(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(on_finished=None, cancel=lambda: None)

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.start_async_stream_decode",
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
        pipeline._start_stream_decode_and_wan_cell(
            kv_id="kv-stream",
            n_chunks=3,
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
        on_chunk = captured["on_chunk"]
        assert callable(on_chunk)
        on_chunk(1, torch.zeros(1, 3, 2, 2, 2))
    finally:
        reset_emit_chunk(token)

    assert len(emitted) == 1
    assert emitted[0].finished is False
    assert emitted[0].chunk_index == 1
    assert emitted[0].total_chunks == 3
    assert CausVidWanPipeline.supports_request_streaming_output is True


def test_local_on_chunk_does_not_accumulate():
    vae = _FakeVae()
    decoder = _FakeDecoder()
    seen: list[tuple[int, tuple[int, ...]]] = []

    def on_chunk(idx: int, pixels: torch.Tensor) -> None:
        seen.append((idx, tuple(pixels.shape)))

    job = _start_local(vae=vae, decoder=decoder, session_id="req-stream", on_chunk=on_chunk)
    assert job.streams_to_client is True
    assert job.on_finished is not None
    job.on_finished(0, torch.randn(1, 4, 2, 4, 4))
    job.on_finished(1, torch.randn(1, 4, 2, 4, 4))
    video = job.join()
    assert video is None
    assert decoder.chunks == 2
    assert seen == [(0, (1, 3, 2, 4, 4)), (1, (1, 3, 2, 4, 4))]
    assert job.emitted_chunks == 2


def test_local_on_chunk_remaps_session_absolute_ids_to_request_relative():
    """Session continuation uses chunk_base>0; client protocol still starts at 0."""
    vae = _FakeVae()
    decoder = _FakeDecoder()
    seen: list[int] = []

    def on_chunk(idx: int, pixels: torch.Tensor) -> None:
        del pixels
        seen.append(idx)

    job = _start_local(vae=vae, decoder=decoder, session_id="world-e2e", on_chunk=on_chunk)
    assert job.on_finished is not None
    # WanCell reports session-absolute ids (chunk_base=5 continuation).
    job.on_finished(5, torch.randn(1, 4, 2, 4, 4))
    job.on_finished(6, torch.randn(1, 4, 2, 4, 4))
    assert job.join() is None
    assert seen == [0, 1]
    assert job.emitted_chunks == 2

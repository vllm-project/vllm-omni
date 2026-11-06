# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for CausVid async stream-decode side path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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


def test_local_cancel_stops_consumer_without_join():
    vae = _FakeVae()
    decoder = _FakeDecoder()
    job = _start_local(vae=vae, decoder=decoder, session_id="req-cancel")
    assert job.on_finished is not None
    job.on_finished(0, torch.randn(1, 4, 2, 4, 4))
    job.cancel()
    # Second cancel is a no-op.
    job.cancel()
    # Thread should exit via stop sentinel; do not call join() after cancel.
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

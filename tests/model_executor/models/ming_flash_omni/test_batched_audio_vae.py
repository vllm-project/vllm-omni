# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VAE chunk budgets, per-request failures, finalization and row reuse."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.common.ming.audio_dsp import ISTFTHead
from vllm_omni.model_executor.models.common.ming.audio_vae import StreamingLinearUpsample
from vllm_omni.model_executor.models.ming_flash_omni import talker_batch
from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
    MingFlashOmniTalkerForConditionalGeneration as Talker,
)
from vllm_omni.model_executor.models.ming_flash_omni.streaming import MingBatchedStreamJob

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def decoder(monkeypatch):
    class IdentityBackbone:
        BLOCK = 256

        def __init__(self, model, *, max_len, **kwargs):
            self.max_len = max_len

        def run(self, xs, rows, starts):
            return torch.cat(xs)

    monkeypatch.setattr(talker_batch, "PackedQwen2", IdentityBackbone)
    backbone = torch.nn.Linear(8, 8)
    backbone.config = SimpleNamespace(sliding_window=128)
    core = SimpleNamespace(
        fc1=torch.nn.Linear(4, 8),
        patch_size=4,
        upsampling=StreamingLinearUpsample(4),
        decoder=backbone,
        head=ISTFTHead(8, n_fft=16, hop_length=4),
    )
    core.upsampling.frame_holdback = True
    return talker_batch.BatchedAudioVAEDecoder(SimpleNamespace(decoder=core), rows=2)


@torch.inference_mode()
@pytest.mark.parametrize("holdback", [False, True])
def test_backlog_is_bounded_and_matches_unsplit_audio(decoder, holdback, mocker):
    decoder.decoder.upsampling.frame_holdback = holdback
    latents = torch.randn(1, 701, 4)
    projected = decoder.decoder.fc1(latents)
    frames = decoder.decoder.upsampling.upsampler(projected.transpose(1, 2)).transpose(1, 2)
    expected, *_ = decoder.head(frames, streaming=True, last_chunk=True)
    spectrum = mocker.spy(decoder.head, "spectrum")
    states: list[dict] = [{}, {}]
    # Exercise retained upsampler state, several bounded passes and final flush.
    first = decoder.decode([(states[0], latents[:, :31], False), (states[1], latents[:, :2], True)])[0]
    final = decoder.decode([(states[0], latents[:, 31:], True)])[0]
    torch.testing.assert_close(torch.cat([first, final], dim=-1), expected, atol=1e-6, rtol=1e-5)
    assert max(call.args[0].shape[0] for call in spectrum.call_args_list) <= decoder.PASS_FRAMES
    assert sorted(decoder.free_rows) == [0, 1]
    assert all("row" not in state for state in states)
    # A fresh request can reuse the released row without previous ISTFT tails.
    torch.testing.assert_close(decoder.decode([({}, latents, True)])[0], expected, atol=1e-6, rtol=1e-5)


def test_frame_limit_is_per_request_and_bad_stream_does_not_advance_peers(decoder):
    decoder.max_frames = 24
    states = [{"row": 0, "position": 20, "upsample": None, "istft": (None, None)}, {}]
    decoder.free_rows = [1]
    result = decoder.decode([(states[0], torch.ones(1, 1, 4), True), (states[1], torch.ones(1, 2, 4), True)])
    assert all(isinstance(x, torch.Tensor) for x in result)
    assert sorted(decoder.free_rows) == [0, 1]
    bad = {"row": 0, "position": 23, "upsample": None, "istft": (None, None)}
    decoder.free_rows = [1]
    result = decoder.decode([(bad, torch.ones(1, 1, 4), False), ({}, torch.ones(1, 2, 4), True)])
    # The first latent is held back, so it has not exceeded the frame limit yet.
    assert isinstance(result[0], torch.Tensor)
    result = decoder.decode([(bad, torch.ones(1, 1, 4), True), ({}, torch.ones(1, 2, 4), True)])
    assert isinstance(result[0], ValueError)
    assert isinstance(result[1], torch.Tensor) and result[1].numel()
    assert bad["position"] == 23 and "row" not in bad
    assert sorted(decoder.free_rows) == [0, 1]


@torch.inference_mode()
def test_invalid_istft_envelope_fails_only_its_request(decoder):
    latents = torch.randn(1, 701, 4)
    projected = decoder.decoder.fc1(latents)
    frames = decoder.decoder.upsampling.upsampler(projected.transpose(1, 2)).transpose(1, 2)
    expected, *_ = decoder.head(frames, streaming=True, last_chunk=True)
    bad: dict = {}
    good: dict = {}
    first = decoder.decode([(bad, latents[:, :31], False), (good, latents[:, :31], False)])[1]
    # Corrupt only one stream's overlap state; the shared window stays valid.
    bad["istft"][1].fill_(-10)
    rejected, completed = decoder.decode([(bad, latents[:, 31:], False), (good, latents[:, 31:], True)])
    assert isinstance(rejected, ValueError) and "window envelope underflowed" in str(rejected)
    torch.testing.assert_close(torch.cat([first, completed], dim=-1), expected, atol=1e-6, rtol=1e-5)
    assert "row" not in bad and "row" not in good
    assert sorted(decoder.free_rows) == [0, 1]
    torch.testing.assert_close(decoder.decode([({}, latents, True)])[0], expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("fatal", [False, True])
def test_forward_cleans_terminal_jobs_and_propagates_device_failures(decoder, fatal, mocker):
    jobs = {key: MingBatchedStreamJob(None, torch.device("cpu")) for key in ("bad", "good", "done")}
    for job in jobs.values():
        job.request = object()
    jobs["bad"].vae_state = {"row": 0, "position": decoder.max_frames, "upsample": None, "istft": (None, None)}
    decoder.free_rows = [1]
    jobs["bad"].publish(torch.ones(1, 2, 4), False)
    jobs["good"].publish(torch.ones(1, 2, 4), True)
    jobs["done"].final_published = True
    engine = SimpleNamespace(error=None, wait_for_output=mocker.Mock(), cancel=mocker.Mock())
    model = SimpleNamespace(
        audio_vae=SimpleNamespace(config=SimpleNamespace(sample_rate=24000)),
        device=torch.device("cpu"),
        config=SimpleNamespace(low_latency_streaming=True),
        _stream_jobs=jobs,
        _batch_engine=engine,
        _batch_vae=decoder,
        _ensure_batch_engine=lambda: engine,
    )
    model.on_requests_finished = lambda ids: Talker.on_requests_finished(model, ids)
    if fatal:
        decoder.llm.run = mocker.Mock(side_effect=RuntimeError("device failure"))
        with pytest.raises(RuntimeError, match="device failure"):
            Talker._batched_forward(model, list(jobs), None, None, None, None)
    else:
        output = Talker._batched_forward(model, list(jobs), None, None, None, None)
        assert output.multimodal_outputs["audio"][0].numel() == 0
        assert output.multimodal_outputs["audio"][1].numel() > 0
    assert not jobs
    assert sorted(decoder.free_rows) == [0, 1]
    assert engine.cancel.called

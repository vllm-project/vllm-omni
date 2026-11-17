# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, field
from threading import Event, get_ident

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE
from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing
from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _waveform(rate: int, seconds: int = 3) -> torch.Tensor:
    time = torch.arange(rate * seconds, dtype=torch.float32) / rate
    return torch.sin(time * 880 * torch.pi)[None]


def _media() -> MiniMaxH3EncoderMediaInput:
    return MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=32,
        width=32,
        num_frames=124,
        latent_t=32,
        audio_t=208,
        videos=(torch.zeros((1, 32, 32, 3), dtype=torch.uint8),),
        video_audios=((_waveform(44100), 44100),),
        audios=((_waveform(16000, 6), 16000),),
    )


@pytest.mark.parametrize("rate", [16000, 32000, 44100])
@pytest.mark.parametrize("channels", [1, 2, 3])
def test_waveform_preparation_preserves_existing_order_and_samples(rate: int, channels: int) -> None:
    torchaudio = pytest.importorskip("torchaudio")
    value = _waveform(rate).repeat(channels, 1).double()
    if channels == 1:
        value = value[0]
    expected = value.float()
    if expected.ndim == 1:
        expected = expected[None]
    if rate != 32000:
        expected = torchaudio.transforms.Resample(rate, 32000)(expected)
    if expected.shape[0] < 2:
        expected = expected.repeat(2, 1)
    expected = expected[:2]
    actual, output_rate = MiniMaxH3AudioVAE._prepare_waveform(value, rate)
    assert output_rate == 32000
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@dataclass
class _AudioVAE:
    started: Event
    release: Event
    owner: int
    sample_rate: int = 32000
    received: list[tuple[torch.Tensor, int]] = field(default_factory=list)

    def _prepare_waveform(self, waveform: torch.Tensor, rate: int) -> tuple[torch.Tensor, int]:
        assert get_ident() != self.owner
        self.started.set()
        assert self.release.wait(10), "visual encoder did not release preparation"
        return MiniMaxH3AudioVAE._prepare_waveform(waveform, rate)

    def encode_waveform(self, waveform: torch.Tensor, rate: int) -> tuple[torch.Tensor, int]:
        assert get_ident() == self.owner
        self.received.append((waveform, rate))
        return torch.zeros((80, 96)), 80


@dataclass
class _VideoVAE:
    started: Event | None = None
    release: Event | None = None
    fail: bool = False

    def encode_video(self, frames: np.ndarray) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if self.started is not None:
            assert self.started.wait(10), "CPU preparation did not overlap visual encoding"
        if self.release is not None:
            self.release.set()
        if self.fail:
            raise RuntimeError("visual encode failed")
        return torch.zeros((1, 96)), (1, 2, 2)


def test_cpu_preparation_overlaps_visual_and_preserves_crop_and_order() -> None:
    started, release = Event(), Event()
    audio = _AudioVAE(started, release, get_ident())
    media = _media()
    try:
        result = processing.encode_media(
            media, video_vae=_VideoVAE(started, release), audio_vae=audio, emit_conditioning=True
        )
    finally:
        release.set()
    assert result is not None
    assert len(audio.received) == 2
    embedded = media.video_audios[0]
    assert embedded is not None
    inputs = [embedded, (media.audios[0][0][..., : round(124 / 24 * 16000)], 16000)]
    for source, actual in zip(inputs, audio.received, strict=True):
        expected = MiniMaxH3AudioVAE._prepare_waveform(*source)
        assert actual[1] == expected[1]
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)


def test_visual_failure_cancels_queued_work_and_joins_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = processing.ThreadPoolExecutor(max_workers=1)
    started, release = Event(), Event()
    futures = []
    submit, shutdown = pool.submit, pool.shutdown

    def block() -> None:
        started.set()
        assert release.wait(10)

    blocker = submit(block)
    assert started.wait(10)

    def record_submit(*args, **kwargs):
        future = submit(*args, **kwargs)
        futures.append(future)
        return future

    def close(*, wait: bool = True, cancel_futures: bool = False) -> None:
        shutdown(wait=False, cancel_futures=cancel_futures)
        release.set()
        shutdown(wait=wait)

    monkeypatch.setattr(pool, "submit", record_submit)
    monkeypatch.setattr(pool, "shutdown", close)
    monkeypatch.setattr(processing, "ThreadPoolExecutor", lambda **_kwargs: pool)
    audio = _AudioVAE(Event(), Event(), get_ident())
    try:
        with pytest.raises(RuntimeError, match="visual encode failed"):
            processing.encode_media(_media(), video_vae=_VideoVAE(fail=True), audio_vae=audio, emit_conditioning=True)
        assert len(futures) == 2
        assert all(future.cancelled() for future in futures)
        assert blocker.done()
    finally:
        release.set()
        shutdown(wait=True, cancel_futures=True)


def test_nonleader_keeps_collectives_without_audio_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_pool(**_kwargs):
        pytest.fail("nonleader must not create an audio worker")

    monkeypatch.setattr(processing, "ThreadPoolExecutor", unexpected_pool)
    assert processing.encode_media(_media(), video_vae=_VideoVAE(), audio_vae=None, emit_conditioning=False) is None


def test_native_rate_references_do_not_start_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_pool(**_kwargs):
        pytest.fail("native-rate references need no preparation worker")

    monkeypatch.setattr(processing, "ThreadPoolExecutor", unexpected_pool)
    media = _media()
    media = MiniMaxH3EncoderMediaInput(
        task=media.task,
        height=media.height,
        width=media.width,
        num_frames=media.num_frames,
        latent_t=media.latent_t,
        audio_t=media.audio_t,
        videos=media.videos,
        video_audios=((_waveform(32000), 32000),),
    )
    audio = _AudioVAE(Event(), Event(), get_ident())
    assert processing.encode_media(media, video_vae=_VideoVAE(), audio_vae=audio, emit_conditioning=True) is not None
    assert audio.received[0][1] == 32000


def test_audio_only_keeps_synchronous_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_pool(**_kwargs):
        pytest.fail("audio-only input has no visual overlap window")

    monkeypatch.setattr(processing, "ThreadPoolExecutor", unexpected_pool)
    source = _media()
    media = MiniMaxH3EncoderMediaInput(
        task=source.task,
        height=source.height,
        width=source.width,
        num_frames=source.num_frames,
        latent_t=source.latent_t,
        audio_t=source.audio_t,
        audios=source.audios,
    )
    audio = _AudioVAE(Event(), Event(), get_ident())
    assert processing.encode_media(media, video_vae=None, audio_vae=audio, emit_conditioning=True) is not None
    assert audio.received[0][1] == 16000


def test_cpu_autocast_keeps_preparation_on_calling_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_pool(**_kwargs):
        pytest.fail("CPU autocast is thread-local and must retain synchronous preparation")

    monkeypatch.setattr(processing, "ThreadPoolExecutor", unexpected_pool)
    audio = _AudioVAE(Event(), Event(), get_ident())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert (
            processing.encode_media(_media(), video_vae=_VideoVAE(), audio_vae=audio, emit_conditioning=True)
            is not None
        )
    assert audio.received[0][1] == 44100


def test_preparation_failure_propagates_after_join(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingAudioVAE(_AudioVAE):
        def _prepare_waveform(self, waveform: torch.Tensor, rate: int) -> tuple[torch.Tensor, int]:
            raise ValueError("waveform preparation failed")

    pool = processing.ThreadPoolExecutor(max_workers=1)
    shutdown = pool.shutdown
    joined = []

    def close(*, wait: bool = True, cancel_futures: bool = False) -> None:
        shutdown(wait=wait, cancel_futures=cancel_futures)
        joined.append((wait, cancel_futures))

    monkeypatch.setattr(pool, "shutdown", close)
    monkeypatch.setattr(processing, "ThreadPoolExecutor", lambda **_kwargs: pool)
    try:
        with pytest.raises(ValueError, match="waveform preparation failed"):
            processing.encode_media(
                _media(),
                video_vae=_VideoVAE(),
                audio_vae=FailingAudioVAE(Event(), Event(), get_ident()),
                emit_conditioning=True,
            )
        assert joined == [(True, True)]
    finally:
        shutdown(wait=True, cancel_futures=True)

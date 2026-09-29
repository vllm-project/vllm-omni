# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import hashlib

import av
import numpy as np
import pytest

from vllm_omni.diffusion.utils.media_utils import ChunkedMP4Encoder, mux_av_video_audio_bytes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _frames() -> np.ndarray:
    rng = np.random.default_rng(6872)
    return rng.integers(0, 256, size=(8, 16, 24, 3), dtype=np.uint8)


def test_chunked_mp4_is_byte_identical_to_whole_mux() -> None:
    frames = _frames()
    baseline = mux_av_video_audio_bytes(
        (av.VideoFrame.from_ndarray(frame, format="rgb24") for frame in frames),
        width=24,
        height=16,
        fps=24,
        video_codec_options={"threads": "1"},
    )
    encoder = ChunkedMP4Encoder(
        width=24,
        height=16,
        fps=24,
        max_pending=2,
        video_codec_options={"threads": "1"},
    )
    for start in (0, 1, 5):
        stop = {0: 1, 1: 5, 5: len(frames)}[start]
        encoder.push(np.ascontiguousarray(frames[start:stop]))
    chunked = encoder.finish()
    assert chunked == baseline
    assert hashlib.sha256(chunked).digest() == hashlib.sha256(baseline).digest()

    audio = np.zeros((2, 320), dtype=np.float32)
    baseline_audio = mux_av_video_audio_bytes(
        (av.VideoFrame.from_ndarray(frame, format="rgb24") for frame in frames),
        width=24,
        height=16,
        fps=24,
        audio_waveform=audio,
        audio_sample_rate=32000,
        video_codec_options={"threads": "1"},
    )
    encoder_audio = ChunkedMP4Encoder(
        width=24,
        height=16,
        fps=24,
        audio_waveform=audio,
        audio_sample_rate=32000,
        video_codec_options={"threads": "1"},
    )
    encoder_audio.push(frames)
    assert encoder_audio.finish() == baseline_audio


def test_chunked_mp4_abort_joins_worker() -> None:
    encoder = ChunkedMP4Encoder(width=24, height=16, fps=24)
    encoder.push(_frames()[:1])
    encoder.abort()
    assert not encoder._thread.is_alive()
    with pytest.raises(RuntimeError, match="already closed"):
        encoder.push(_frames()[:1])


def test_chunked_mp4_validates_shape_and_dtype() -> None:
    encoder = ChunkedMP4Encoder(width=24, height=16, fps=24)
    with pytest.raises(ValueError, match="shape"):
        encoder.push(np.zeros((16, 24, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="dtype"):
        encoder.push(np.zeros((1, 16, 24, 3), dtype=np.float32))
    encoder.abort()


def test_chunked_mp4_close_returns_encoded_bytes() -> None:
    import io

    encoder = ChunkedMP4Encoder(width=24, height=16, fps=24)
    encoder.push(_frames())
    result = encoder.close()
    assert result == encoder.finish() == encoder.close()
    with av.open(io.BytesIO(result)) as container:
        assert len(list(container.decode(video=0))) == len(_frames())


def test_chunked_mp4_failure_during_push(monkeypatch) -> None:
    from threading import Event

    from vllm_omni.diffusion.utils import media_utils

    fail = Event()
    draining = Event()
    error = RuntimeError("encoding failed")

    def mux(*args, **kwargs):
        assert fail.wait(5)
        raise error

    monkeypatch.setattr(media_utils, "mux_av_video_audio_bytes", mux)
    drain = ChunkedMP4Encoder._drain_until_done

    def notified_drain(self):
        draining.set()
        drain(self)

    monkeypatch.setattr(ChunkedMP4Encoder, "_drain_until_done", notified_drain)
    encoder = ChunkedMP4Encoder(width=24, height=16, fps=24)
    put = encoder._queue.put

    def fail_before_put(item):
        fail.set()
        assert draining.wait(5)
        put(item)

    monkeypatch.setattr(encoder._queue, "put", fail_before_put)
    try:
        with pytest.raises(RuntimeError, match="encoding failed") as exc:
            encoder.push(_frames())
        assert exc.value is error
        with pytest.raises(RuntimeError, match="encoding failed"):
            encoder.finish()
    finally:
        fail.set()
        encoder.abort()
    assert not encoder._thread.is_alive()


def test_chunked_mp4_flush_failure_does_not_hang(monkeypatch) -> None:
    from threading import Thread

    from vllm_omni.diffusion.utils import media_utils

    error = RuntimeError("flush failed")

    def mux(frames, **kwargs):
        list(frames)
        raise error

    monkeypatch.setattr(media_utils, "mux_av_video_audio_bytes", mux)
    encoder = ChunkedMP4Encoder(width=24, height=16, fps=24)
    encoder.push(_frames())
    errors = []

    def finish():
        try:
            encoder.finish()
        except RuntimeError as exc:
            errors.append(exc)

    closer = Thread(target=finish, daemon=True)
    closer.start()
    closer.join(5)
    try:
        assert not closer.is_alive(), "finish hung after the worker consumed its sentinel"
        assert errors == [error]
    finally:
        if closer.is_alive():
            encoder._send_done()
            closer.join(5)


def test_chunked_mp4_abort_discards_pending_chunks(monkeypatch) -> None:
    from threading import Event, Thread

    from vllm_omni.diffusion.utils import media_utils

    entered = Event()
    release = Event()
    done_sent = Event()
    encoded = []

    def mux(frames, **kwargs):
        encoded.append(next(frames))
        entered.set()
        assert release.wait(5)
        encoded.extend(frames)
        return b"unused"

    monkeypatch.setattr(media_utils, "mux_av_video_audio_bytes", mux)
    encoder = ChunkedMP4Encoder(width=24, height=16, fps=24, max_pending=1)
    send_done = encoder._send_done

    def notified_done():
        send_done()
        done_sent.set()

    monkeypatch.setattr(encoder, "_send_done", notified_done)
    encoder.push(_frames()[:1])
    assert entered.wait(5)
    encoder.push(_frames()[1:])
    closer = Thread(target=encoder.abort, daemon=True)
    closer.start()
    try:
        assert done_sent.wait(5), "abort waited for encoding before discarding the pending queue"
    finally:
        release.set()
        closer.join(5)
    assert not closer.is_alive()
    assert len(encoded) == 1
    with pytest.raises(RuntimeError, match="aborted"):
        encoder.finish()

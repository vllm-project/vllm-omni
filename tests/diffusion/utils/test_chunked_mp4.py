# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import hashlib
import io
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import av
import numpy as np
import pytest

from vllm_omni.diffusion.utils.chunked_video import ChunkedVideoMP4Session
from vllm_omni.diffusion.utils.media_utils import ChunkedMP4Encoder, mux_av_video_audio_bytes, mux_video_audio_bytes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _frames() -> np.ndarray:
    rng = np.random.default_rng(6872)
    return rng.integers(0, 256, size=(8, 16, 24, 3), dtype=np.uint8)


@pytest.fixture
def audio_done(monkeypatch):
    audio_done = Event()
    open_container = av.open

    class AudioStream:
        def __init__(self, stream):
            self.stream = stream

        @property
        def layout(self):
            return self.stream.layout

        @layout.setter
        def layout(self, value):
            self.stream.layout = value

        def encode(self, frame=None):
            packets = self.stream.encode(frame)
            if frame is None:
                audio_done.set()
            return packets

    @contextmanager
    def recorded_container(*args, **kwargs):
        with open_container(*args, **kwargs) as container:

            def add_stream(*args, **kwargs):
                stream = container.add_stream(*args, **kwargs)
                return AudioStream(stream) if stream.type == "audio" else stream

            yield SimpleNamespace(add_stream=add_stream, mux=container.mux)

    monkeypatch.setattr(av, "open", recorded_container)
    return audio_done


@pytest.mark.parametrize("encode_audio_first", [None, False, True])
def test_audio_priority_is_opt_in(audio_done, encode_audio_first) -> None:
    def video():
        assert audio_done.is_set() == (encode_audio_first is True)
        yield from (av.VideoFrame.from_ndarray(frame, format="rgb24") for frame in _frames())

    options = {} if encode_audio_first is None else {"encode_audio_first": encode_audio_first}
    assert mux_av_video_audio_bytes(
        video(),
        width=24,
        height=16,
        fps=24,
        audio_waveform=np.zeros((2, 8000), dtype=np.float32),
        audio_sample_rate=24000,
        **options,
    )
    assert audio_done.is_set()


@pytest.mark.parametrize("eager_session", [False, True])
def test_ready_audio_is_encoded_before_waiting_for_video(audio_done, eager_session) -> None:
    """Audio work should overlap a producer that has not delivered video yet."""

    if eager_session:
        import torch

        session = ChunkedVideoMP4Session(
            value_range=(0.0, 1.0),
            fps=24,
            audio_waveforms=[np.zeros((2, 8000), dtype=np.float32)],
            audio_sample_rate=24000,
            transfer_slots=0,
        )
        try:
            session.start(width=24, height=16, batch_size=1)
            assert audio_done.wait(timeout=5), "AAC waited for the first video chunk"
            session.push(torch.zeros(1, 3, 8, 16, 24))
            assert session.finish()
        finally:
            session.abort()
        return

    def delayed_video():
        assert audio_done.is_set(), "audio encoding was deferred until video completion"
        yield from (av.VideoFrame.from_ndarray(frame, format="rgb24") for frame in _frames())

    result = mux_av_video_audio_bytes(
        delayed_video(),
        width=24,
        height=16,
        fps=24,
        audio_waveform=np.zeros((2, 8000), dtype=np.float32),
        audio_sample_rate=24000,
        encode_audio_first=True,
    )
    assert result


@pytest.mark.parametrize("encode_audio_first", [False, True])
def test_early_audio_preserves_decoded_media_and_timestamps(encode_audio_first) -> None:
    """Moving AAC work must preserve samples, video pixels and stream timing."""
    frames = _frames()
    time = np.arange(8000, dtype=np.float32) / 24000
    audio = np.stack([0.1 * np.sin(2 * np.pi * frequency * time) for frequency in (440, 660)])
    options = dict(fps=24, audio_sample_rate=24000, video_codec_options={"threads": "1"})
    baseline = mux_video_audio_bytes(frames, audio, **options)
    result = mux_av_video_audio_bytes(
        (av.VideoFrame.from_ndarray(frame, format="rgb24") for frame in frames),
        width=24,
        height=16,
        audio_waveform=audio,
        encode_audio_first=encode_audio_first,
        **options,
    )

    def decoded(data):
        with av.open(io.BytesIO(data)) as container:
            timing = [
                (stream.type, stream.start_time, stream.duration, stream.time_base) for stream in container.streams
            ]
            video = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
        with av.open(io.BytesIO(data)) as container:
            audio_frames = list(container.decode(audio=0))
            samples = np.concatenate([frame.to_ndarray() for frame in audio_frames], axis=1)
            audio_timing = [(frame.pts, frame.time_base, frame.sample_rate) for frame in audio_frames]
        return video, samples, timing, audio_timing

    old_video, old_audio, old_timing, old_audio_timing = decoded(baseline)
    new_video, new_audio, new_timing, new_audio_timing = decoded(result)
    np.testing.assert_array_equal(new_video, old_video)
    np.testing.assert_array_equal(new_audio, old_audio)
    assert new_timing == old_timing
    assert new_audio_timing == old_audio_timing


def test_chunked_mp4_is_byte_identical_to_whole_mux() -> None:
    # mbtree lookahead is not bit-stable across the pauses between chunks.
    # threads=1 leaves mbtree enabled; AMD CI (Lavf 62) then mismatches.
    video_codec_options = {"threads": "1", "mbtree": "0"}
    frames = _frames()
    baseline = mux_av_video_audio_bytes(
        (av.VideoFrame.from_ndarray(frame, format="rgb24") for frame in frames),
        width=24,
        height=16,
        fps=24,
        video_codec_options=video_codec_options,
    )
    encoder = ChunkedMP4Encoder(
        width=24,
        height=16,
        fps=24,
        max_pending=2,
        video_codec_options=video_codec_options,
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
        video_codec_options=video_codec_options,
        encode_audio_first=True,
    )
    encoder_audio = ChunkedMP4Encoder(
        width=24,
        height=16,
        fps=24,
        audio_waveform=audio,
        audio_sample_rate=32000,
        video_codec_options=video_codec_options,
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for incremental frame accounting in the generic client."""

import asyncio
import contextlib
import importlib
import io
import json
import threading
from pathlib import Path

import av
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.example]


@pytest.fixture
def client(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "examples/online_serving/streaming_video_generation"
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("streaming_video_client")


@pytest.fixture
def fragments():
    """Three media batches (9 + 12 + 12 frames) and a muxer trailer."""
    output = io.BytesIO()
    parts = []
    with av.open(
        output,
        "w",
        format="mp4",
        options={"movflags": "+frag_every_frame+empty_moov+default_base_moof"},
    ) as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.max_b_frames = 0
        stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
        offset = 0
        for count in (9, 12, 12):
            for _ in range(count):
                frame = av.VideoFrame(64, 48, "yuv420p")
                for plane in frame.planes:
                    plane.update(bytes([128]) * plane.buffer_size)
                for packet in stream.encode(frame):
                    container.mux(packet)
            parts.append(output.getvalue()[offset:])
            offset = output.tell()
        for packet in stream.encode():
            container.mux(packet)
    parts.append(output.getvalue()[offset:])
    return parts


def test_camera_send_proceeds_while_decoder_is_busy(client, fragments, monkeypatch, tmp_path):
    """A decoder waiting for a control send must not block that send's loop."""
    entered = threading.Event()
    released = threading.Event()
    decode_saw_control = []
    real_open = av.open
    saved = []

    def gated_open(*args, **kwargs):
        entered.set()
        decode_saw_control.append(released.wait(3))
        return real_open(*args, **kwargs)

    class Connection:
        def __init__(self):
            self.messages = iter([json.dumps({"type": "video.start", "format": "m4s"}), b"".join(fragments)])
            self.controls = []

        async def send(self, message):
            payload = json.loads(message)
            if payload["type"] == "session.interaction":
                assert await asyncio.to_thread(entered.wait, 3)
                self.controls.append(payload)
                released.set()

        async def recv(self):
            message = next(self.messages, None)
            if message is not None:
                return message
            assert await asyncio.to_thread(released.wait, 5)
            return json.dumps({"type": "session.done"})

    connection = Connection()

    @contextlib.asynccontextmanager
    async def connect(*args, **kwargs):
        yield connection

    monkeypatch.setattr(client, "connect", connect)
    monkeypatch.setattr(client.av, "open", gated_open)
    monkeypatch.setattr(client, "save_video", lambda chunks, *args, **kwargs: saved.append(b"".join(chunks)))
    monkeypatch.setattr("sys.argv", ["client", "--output", str(tmp_path / "out.mp4")])
    args = client.parse_args()
    args.camera_updates = [client.ScheduledCameraUpdate(at=0, camera={"mode": "velocity"}, transition_chunks=None)]
    try:
        asyncio.run(client.stream_video(args))
    finally:
        released.set()
    assert decode_saw_control == [True]
    assert len(connection.controls) == 1
    assert saved == [b"".join(fragments)]


def test_incremental_count_includes_trailer(client, fragments):
    async def run():
        counter = client._StreamingFrameCounter(stream_format="m4s")
        try:
            for part in fragments:
                counter.feed(part)
            assert await counter.finish() == 33
        finally:
            await counter.finish()

    asyncio.run(run())


def test_partial_stream_counts_only_decodable_frames(client, fragments):
    async def run():
        counter = client._StreamingFrameCounter(stream_format="m4s")
        counter.feed(b"".join(fragments[:-1]))
        assert await counter.finish() == 32

    asyncio.run(run())


def test_invalid_media_reports_unknown(client):
    async def run():
        counter = client._StreamingFrameCounter(stream_format="m4s")
        counter.feed(b"not a video")
        assert await counter.finish() is None
        assert counter.error == "decode_failed"

    asyncio.run(run())


def test_empty_stream_has_zero_frames(client):
    async def run():
        counter = client._StreamingFrameCounter(stream_format="m4s")
        counter.feed(b"")
        assert await counter.finish() == 0

    asyncio.run(run())


def test_live_progress_and_single_pass_decode(client, fragments, monkeypatch):
    real_open = av.open
    decoded_frames = []
    opened = []

    @contextlib.contextmanager
    def observed_open(*args, **kwargs):
        with real_open(*args, **kwargs) as container:
            opened.append(container)

            class ObservedContainer:
                streams = container.streams

                def decode(self, stream):
                    for frame in container.decode(stream):
                        decoded_frames.append(frame.pts)
                        yield frame

            yield ObservedContainer()

    monkeypatch.setattr(client.av, "open", observed_open)

    async def run():
        counter = client._StreamingFrameCounter(stream_format="m4s")
        try:
            counter.feed(fragments[0])

            async def wait_for_progress():
                while counter.count == 0:
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(wait_for_progress(), 5)
            assert 0 < counter.count < 33
            for part in fragments[1:]:
                counter.feed(part)
            assert await counter.finish() == 33
        finally:
            await counter.finish()

    asyncio.run(run())
    assert len(opened) == 1
    assert len(decoded_frames) == 33
    assert len(set(decoded_frames)) == 33


def test_buffer_overflow_disables_count_without_blocking(client, monkeypatch):
    release = threading.Event()
    real_open = av.open

    def blocked_open(*args, **kwargs):
        assert release.wait(5)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(client.av, "open", blocked_open)
    monkeypatch.setattr(client._StreamingFrameCounter, "_MAX_PENDING_BYTES", 8)

    async def run():
        counter = client._StreamingFrameCounter(stream_format="m4s")
        try:
            counter.feed(b"12345678")
            counter.feed(b"9")
            assert counter.count is None
            assert counter.error == "buffer_limit_exceeded"
        finally:
            release.set()
            assert await counter.finish() is None

    asyncio.run(run())


def test_shutdown_timeout_reports_unknown(client, fragments, monkeypatch):
    release = threading.Event()
    real_open = av.open

    def blocked_open(*args, **kwargs):
        assert release.wait(5)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(client.av, "open", blocked_open)
    monkeypatch.setattr(client._StreamingFrameCounter, "_JOIN_TIMEOUT", 0.01)

    async def run():
        counter = client._StreamingFrameCounter(stream_format="m4s")
        try:
            counter.feed(b"".join(fragments))
            assert await counter.finish() is None
            assert counter.error == "shutdown_timeout"
        finally:
            release.set()
            monkeypatch.setattr(client._StreamingFrameCounter, "_JOIN_TIMEOUT", 5)
            await counter.finish()
        assert not counter._thread.is_alive()

    asyncio.run(run())


@pytest.mark.parametrize("ending", ["done", "disconnect", "cancel", "scheduler_error"])
def test_session_cleanup_drains_counter_and_preserves_media(client, fragments, monkeypatch, tmp_path, capsys, ending):
    counters = []
    original_counter = client._StreamingFrameCounter
    saved = []

    def create_counter(**kwargs):
        counter = original_counter(**kwargs)
        counters.append(counter)
        return counter

    class Connection:
        def __init__(self):
            self.messages = iter([json.dumps({"type": "video.start", "format": "m4s"}), *fragments[:-1]])
            self.control_attempted = asyncio.Event()

        async def send(self, message):
            if ending == "scheduler_error" and json.loads(message)["type"] == "session.interaction":
                self.control_attempted.set()
                raise RuntimeError("control send failed")

        async def recv(self):
            message = next(self.messages, None)
            if message is not None:
                return message
            if ending == "disconnect":
                raise OSError("connection lost")
            if ending == "cancel":
                raise asyncio.CancelledError
            if ending == "scheduler_error":
                await self.control_attempted.wait()
            return json.dumps({"type": "session.done"})

        async def close(self, **kwargs):
            pass

    @contextlib.asynccontextmanager
    async def connect(*args, **kwargs):
        yield Connection()

    monkeypatch.setattr(client, "connect", connect)
    monkeypatch.setattr(client, "_StreamingFrameCounter", create_counter)
    monkeypatch.setattr(client, "save_video", lambda chunks, *args, **kwargs: saved.append(b"".join(chunks)))
    monkeypatch.setattr("sys.argv", ["client", "--output", str(tmp_path / "out.mp4")])
    args = client.parse_args()
    if ending == "scheduler_error":
        args.camera_updates = [client.ScheduledCameraUpdate(at=0, camera={"mode": "velocity"}, transition_chunks=None)]
    error = {"disconnect": OSError, "cancel": asyncio.CancelledError, "scheduler_error": RuntimeError}.get(ending)
    if error is None:
        asyncio.run(client.stream_video(args))
    else:
        with pytest.raises(error):
            asyncio.run(client.stream_video(args))
    assert len(counters) == 1
    assert counters[0].count == 32
    assert not counters[0]._thread.is_alive()
    assert saved == [b"".join(fragments[:-1])]
    assert "[decode] total_frames=32 status=complete" in capsys.readouterr().out


def test_unexpected_decoder_failure_does_not_report_success(client, monkeypatch):
    def fail_open(*args, **kwargs):
        raise RuntimeError("decoder initialization failed")

    monkeypatch.setattr(client.av, "open", fail_open)

    async def run():
        counter = client._StreamingFrameCounter(stream_format="m4s")
        counter.feed(b"media")
        assert await counter.finish() is None
        assert counter.error == "decode_failed"

    asyncio.run(run())

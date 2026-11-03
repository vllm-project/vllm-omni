# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import wave
from urllib.parse import parse_qs, urlsplit

import pytest

from vllm_omni.experimental.fullduplex.client import (
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    read_pcm16_wav,
)
from vllm_omni.experimental.fullduplex.video_stacking import (
    concat_frames,
    unit_subframe_offsets,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import (
    MiniCPMO45DuplexPolicy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_realtime_client_removes_obsolete_duplex_url_parameters():
    url = build_realtime_url(
        "ws://localhost:8099/v1/realtime?custom=1&duplex=0&model=stale&native_duplex=0&session_id=stale",
        "openbmb/MiniCPM-o-4_5",
    )

    query = parse_qs(urlsplit(url).query)
    assert query == {
        "custom": ["1"],
        "duplex": ["1"],
        "model": ["openbmb/MiniCPM-o-4_5"],
    }


def test_seed_tts_initial_text_is_part_of_native_duplex_context():
    prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
        "Speak exactly.",
        True,
        "The quick brown fox.",
    )

    assert prefix == "<|im_start|>system\nSpeak exactly.\n<|audio_start|>"
    assert suffix == (
        "<|audio_end|><|im_end|>\n<|im_start|>user\nThe quick brown fox.<|im_end|>\n<|im_start|>assistant\n"
    )


@pytest.mark.parametrize("scheme, expected_scheme", [("http", "ws"), ("https", "wss"), ("ws", "ws"), ("wss", "wss")])
def test_realtime_client_preserves_resume_url(scheme, expected_scheme):
    url = build_realtime_url(
        f"{scheme}://localhost:8099/v1/realtime?model=existing&resume=1&custom=&autostart=1#fragment",
        None,
        autostart=False,
    )

    parts = urlsplit(url)
    assert (parts.scheme, parts.netloc, parts.path, parts.fragment) == (
        expected_scheme,
        "localhost:8099",
        "/v1/realtime",
        "fragment",
    )
    assert parse_qs(parts.query, keep_blank_values=True) == {
        "model": ["existing"],
        "resume": ["1"],
        "custom": [""],
        "duplex": ["1"],
        "autostart": ["0"],
    }


@pytest.mark.asyncio
async def test_realtime_client_configure_omits_ref_audio_by_default():
    class Client(RealtimeDuplexClient):
        def __init__(self):
            super().__init__("ws://unused")
            self.sent = []

        async def send(self, event):
            self.sent.append(event)
            self.events.add({"type": "session.created"})

    client = Client()

    await client.configure(
        "openbmb/MiniCPM-o-4_5",
        idle_timeout_s=900,
        timeout_s=1,
    )

    session = client.sent[0]["session"]
    assert "ref_audio" not in session
    assert session["idle_timeout_s"] == 900
    assert "session_id" not in session
    assert session["extra_body"] == {"auto_response": True, "force_listen_count": 0}


@pytest.mark.asyncio
async def test_realtime_client_configure_sends_explicit_ref_audio():
    class Client(RealtimeDuplexClient):
        def __init__(self):
            super().__init__("ws://unused")
            self.sent = []

        async def send(self, event):
            self.sent.append(event)
            self.events.add({"type": "session.created"})

    client = Client()

    await client.configure(
        "openbmb/MiniCPM-o-4_5",
        ref_audio="data:audio/wav;base64,AAAA",
        timeout_s=1,
    )

    session = client.sent[0]["session"]
    assert session["ref_audio"] == "data:audio/wav;base64,AAAA"
    assert "idle_timeout_s" not in session


@pytest.mark.asyncio
async def test_realtime_client_reports_reader_exit_to_event_waiters():
    client = RealtimeDuplexClient("ws://unused")

    async def stopped():
        return None

    client._reader_task = asyncio.create_task(stopped())
    await client._reader_task

    with pytest.raises(ConnectionError, match="closed before"):
        client.raise_if_reader_stopped()


@pytest.mark.asyncio
async def test_realtime_client_configure_reports_reader_failure_immediately():
    class Client(RealtimeDuplexClient):
        async def send(self, event):
            async def fail():
                raise RuntimeError("reader failed")

            self._reader_task = asyncio.create_task(fail())
            await asyncio.sleep(0)

    client = Client("ws://unused")
    with pytest.raises(ConnectionError, match="reader failed"):
        await client.configure("model", timeout_s=10.0)


@pytest.mark.asyncio
async def test_realtime_client_close_reports_reader_cancellation_immediately():
    class Client(RealtimeDuplexClient):
        async def send(self, event):
            async def wait_forever():
                await asyncio.Future()

            self._reader_task = asyncio.create_task(wait_forever())
            self._reader_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await self._reader_task

    client = Client("ws://unused")
    with pytest.raises(ConnectionError, match="reader was cancelled"):
        await client.close_session(timeout_s=10.0)


@pytest.mark.asyncio
async def test_realtime_client_close_waits_for_a_new_session_closed_event():
    class Client(RealtimeDuplexClient):
        async def send(self, event):
            assert event["type"] == "session.close"

            async def acknowledge_close():
                await asyncio.sleep(0.02)
                self.events.add({"type": "session.closed"})

            asyncio.create_task(acknowledge_close())

    client = Client("ws://unused")
    client.events.add({"type": "session.closed", "reason": "old"})
    started_at = asyncio.get_running_loop().time()

    await client.close_session(timeout_s=0.2)

    assert asyncio.get_running_loop().time() - started_at >= 0.01


@pytest.mark.asyncio
async def test_realtime_client_configure_sends_seed_tts_text_condition():
    class Client(RealtimeDuplexClient):
        def __init__(self):
            super().__init__("ws://unused")
            self.sent = []

        async def send(self, event):
            self.sent.append(event)
            self.events.add({"type": "session.created"})

    client = Client()

    await client.configure(
        "openbmb/MiniCPM-o-4_5",
        instructions="Speak the requested text exactly.",
        initial_user_text="The quick brown fox.",
        timeout_s=1,
    )

    session = client.sent[0]["session"]
    assert session["instructions"] == "Speak the requested text exactly."
    assert session["extra_body"]["duplex_initial_user_text"] == "The quick brown fox."


@pytest.mark.asyncio
async def test_realtime_client_configure_disables_auto_response_without_obsolete_opt_in():
    class Client(RealtimeDuplexClient):
        def __init__(self):
            super().__init__("ws://unused")
            self.sent = []

        async def send(self, event):
            self.sent.append(event)
            self.events.add({"type": "session.created"})

    client = Client()
    extra_body = {"ref_audio": "data:audio/wav;base64,AAAA", "native_duplex": False}

    await client.configure(
        "openbmb/MiniCPM-o-4_5",
        auto_response=False,
        extra_body=extra_body,
        timeout_s=1,
    )

    session_extra_body = client.sent[0]["session"]["extra_body"]
    assert session_extra_body == {
        "ref_audio": "data:audio/wav;base64,AAAA",
        "auto_response": False,
        "force_listen_count": 0,
    }
    assert extra_body == {"ref_audio": "data:audio/wav;base64,AAAA", "native_duplex": False}


@pytest.mark.asyncio
async def test_realtime_client_stamps_append_after_send():
    class _FakeWs:
        async def send(self, raw: str) -> None:
            del raw

    client = RealtimeDuplexClient("ws://unused")
    client._ws = _FakeWs()
    await client.send(
        {
            "type": "input_audio_buffer.append",
            "audio": "xx",
            "video_frames": ["frame"],
        }
    )
    await client.send({"type": "input_audio_buffer.commit"})
    assert len(client.events.client_appends) == 1
    assert client.events.client_appends[0].keys() == {"sent_at_s", "event_id"}
    assert client.events.client_appends[0]["event_id"] == ""


def test_realtime_event_collector_partitions_audio_by_response():
    collector = RealtimeEventCollector()
    collector.add({"type": "response.created", "response": {"id": "resp-a"}})
    collector.add(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(b"audio-a").decode("ascii"),
            "sample_rate_hz": 16_000,
        }
    )

    assert collector.response_ids == ["resp-a"]
    assert collector.audio_bytes("resp-a") == b"audio-a"
    assert collector.output_sample_rate_hz == 16_000
    assert collector.first_received_at("response.created") is not None
    assert collector.last_received_at("response.output_audio.delta") is not None


def test_realtime_client_reads_pcm16_wav(tmp_path):
    path = tmp_path / "audio.wav"
    pcm16 = b"\x01\x00\x02\x00"

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(pcm16)

    assert read_pcm16_wav(path) == pcm16


def _stack_test_image(color, size=(40, 30)):
    from PIL import Image

    return Image.new("RGB", size, color)


def test_video_stacking_tiles_a_units_subframes_into_one_image():
    frames = [_stack_test_image((255, 0, 0)), _stack_test_image((0, 255, 0))]

    composite = concat_frames(frames)

    # Landscape cells stack into a column because that canvas is squarer, and
    # the interior seam carries the separator band that lets the model tell the
    # sub-frames apart.
    assert composite.size == (40, 30 * 2 + 6)
    assert composite.getpixel((20, 30 + 3)) == (0, 0, 0)


def test_video_stacking_picks_the_squarest_grid():
    portrait = [_stack_test_image((255, 0, 0), size=(20, 60)) for _ in range(2)]
    four = [_stack_test_image((0, 0, 255)) for _ in range(4)]

    # A portrait clip tiles side by side, four frames always tile 2x2.
    assert concat_frames(portrait).size == (20 * 2 + 6, 60)
    assert concat_frames(four).size == (40 * 2 + 6, 30 * 2 + 6)


def test_video_stacking_skips_the_subframe_that_duplicates_the_base_frame():
    # Official samples stack_frames=5 as 0.2/0.4/0.6/0.8 s: offset 0 is the base
    # frame, already sent as frame_list[0].
    assert unit_subframe_offsets(5) == pytest.approx([0.2, 0.4, 0.6, 0.8])
    assert unit_subframe_offsets(1) == []


def test_realtime_client_sends_each_units_composite_beside_its_base_frame():
    sent: list[dict] = []

    class _Client(RealtimeDuplexClient):
        def __init__(self):
            pass

        async def send(self, event):
            sent.append(event)

    asyncio.run(
        _Client().stream_pcm16(
            b"\x01\x00" * (16_000 * 3),
            chunk_ms=200,
            realtime=False,
            video_frames=["f0", "f1"],
            stacked_video_frames=["s0", None],
        )
    )

    # A composite belongs to the unit it was captured in, so it rides the same
    # append as that unit's base frame; a unit without one sends the base alone.
    assert [event["video_frames"] for event in sent if "video_frames" in event] == [["f0", "s0"], ["f1"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("unit_ms", [32, 1000])
@pytest.mark.parametrize("send_cost_s", [0.001, 0.040])
async def test_stream_av_units_pacing_absorbs_send_frame_cost_and_sleep_overshoot(monkeypatch, unit_ms, send_cost_s):
    from vllm_omni.experimental.fullduplex import client as client_module

    class Clock:
        now = 7.0

        def monotonic(self):
            return self.now

        async def sleep(self, delay):
            assert delay >= 0
            # Model scheduling overshoot without relying on host load.
            self.now += delay + (0.002 if delay else 0.0)

    clock = Clock()
    sent, frames = [], []

    class Client(RealtimeDuplexClient):
        async def send(self, event):
            clock.now += send_cost_s
            sent.append(event)

    async def frame_sink(items):
        clock.now += 0.005
        frames.extend(items)

    monkeypatch.setattr(client_module, "time", clock)
    monkeypatch.setattr(client_module, "asyncio", clock)
    client = Client("ws://unused")
    chunk = b"\x01\x00" * (16 * unit_ms)
    count = 60_000 // unit_ms
    started = clock.now

    frame_count = await client.stream_av_units(
        ((chunk, [f"frame-{index}"]) for index in range(count)), frame_sink=frame_sink
    )

    # Slow sends may overrun the media clock, but must not add another full
    # chunk's sleep. Ordinary send/frame work and overshoot must not accumulate.
    processing_s = count * (send_cost_s + 0.005)
    expected_s = max(60.0, processing_s) + (0.002 if processing_s < 60.0 else 0.0)
    assert clock.now - started == pytest.approx(expected_s)
    assert frame_count == count and frames == [f"frame-{index}" for index in range(count)]
    assert b"".join(base64.b64decode(event["audio"]) for event in sent) == chunk * count
    assert sent[-1]["audio_end_ms"] == client._media_clock_ms == 60_000
    assert not any("video_frames" in event for event in sent)

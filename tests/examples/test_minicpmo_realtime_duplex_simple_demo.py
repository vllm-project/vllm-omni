# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import importlib.util
import sys
import wave
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEMO_PATH = Path(__file__).resolve().parents[2] / "examples/online_serving/minicpmo/realtime_duplex_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location(
        "minicpmo_realtime_duplex_simple_demo_test",
        DEMO_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_post_commit_decision_ignores_streaming_listens_before_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "session.created"},
        {"type": "response.listen"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert commit_index == 3
    assert demo._post_commit_model_decision(events, commit_index) is None


def test_post_commit_decision_accepts_listen_after_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
        {"type": "response.listen"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) == "listen"


def test_post_commit_decision_accepts_drained_speak_after_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.created", "response": {"id": "resp-a"}},
        {"type": "response.output_audio.delta", "response_id": "resp-a", "delta": "AAAA"},
        {"type": "response.done", "response_id": "resp-a"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) == "speak"


def test_post_commit_decision_ignores_cancelled_response():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.done", "response": {"status": "cancelled"}},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) is None


def test_commit_ack_search_starts_after_stream_cursor():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    assert demo._input_committed_index(events, after_index=1) == 2


def test_exact_model_unit_boundary_does_not_require_an_extra_decision():
    demo = _load_demo_module()
    unit_bytes = demo.PCM16_SAMPLE_RATE * demo.PCM16_BYTES_PER_SAMPLE

    assert not demo._has_residual_model_unit(b"\0" * (unit_bytes * 6), chunk_period_ms=1000)


def test_partial_model_unit_requires_a_post_commit_decision():
    demo = _load_demo_module()
    unit_bytes = demo.PCM16_SAMPLE_RATE * demo.PCM16_BYTES_PER_SAMPLE

    assert demo._has_residual_model_unit(b"\0" * (unit_bytes * 6 + 512), chunk_period_ms=1000)


def test_latest_streaming_decision_is_used_at_exact_boundary():
    demo = _load_demo_module()
    events = [
        {"type": "response.listen"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    assert demo._latest_model_decision(events, after_index=0) == "listen"


def test_ref_audio_data_url_encodes_explicit_wav(tmp_path):
    demo = _load_demo_module()
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFfake")

    assert demo._ref_audio_data_url(str(ref)) == "data:audio/wav;base64,UklGRmZha2U="
    assert demo._ref_audio_data_url(None) is None


def test_realtime_duplex_demo_allows_the_server_default_ref_audio(monkeypatch):
    demo = _load_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "realtime_duplex_demo.py",
            "--input-wav",
            "input.wav",
        ],
    )

    args = demo.parse_args()

    assert args.ref_audio is None


def test_realtime_duplex_demo_accepts_explicit_ref_audio(monkeypatch):
    demo = _load_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "realtime_duplex_demo.py",
            "--input-wav",
            "input.wav",
            "--ref-audio",
            "ref.wav",
        ],
    )

    args = demo.parse_args()

    assert args.ref_audio == "ref.wav"


def test_open_streaming_response_requires_post_commit_drain():
    demo = _load_demo_module()

    assert demo._response_in_progress([{"type": "response.created"}])
    assert not demo._response_in_progress(
        [
            {"type": "response.created"},
            {"type": "response.done"},
        ]
    )


def test_final_input_commit_checkpoints_playback_first():
    demo = _load_demo_module()
    collector = demo.EventCollector()
    collector.add(
        {
            "type": "response.created",
            "response": {"id": "resp-zero-audio"},
        }
    )
    calls: list[dict[str, object]] = []

    class _Client:
        async def ack_playback(self, played_ms, *, response_id=None, item_id=None, committed_ms=None):
            calls.append({"type": "playback.ack", "response_id": response_id, "played_ms": int(played_ms)})

        async def commit(self):
            calls.append({"type": "input_audio_buffer.commit"})

    commit_sent_at_s = asyncio.run(demo._commit_input_after_playback_checkpoint(_Client(), collector))

    # The zero-audio response is still acked (0 ms) before the commit: the
    # ack checkpoints its history position on the server.
    assert [event["type"] for event in calls] == ["playback.ack", "input_audio_buffer.commit"]
    assert calls[0]["response_id"] == "resp-zero-audio"
    assert calls[0]["played_ms"] == 0
    assert isinstance(commit_sent_at_s, float)


def test_streaming_collector_acks_each_finished_reply_like_the_web_client():
    demo = _load_demo_module()
    audio = base64.b64encode(bytes(2 * 2400)).decode("ascii")
    raw_events = [
        {"type": "response.created", "response": {"id": "resp-audio"}},
        {"type": "response.output_audio.delta", "response_id": "resp-audio", "delta": audio},
        {"type": "response.done", "response": {"id": "resp-audio"}},
        {"type": "response.created", "response": {"id": "resp-silent"}},
        {"type": "response.done", "response": {"id": "resp-silent"}},
    ]
    acks: list[dict[str, object]] = []

    class _Event:
        def __init__(self, raw):
            self.raw = raw

    class _Client:
        async def events(self):
            for raw in raw_events:
                yield _Event(raw)

        async def ack_playback(self, played_ms, *, response_id=None, item_id=None, committed_ms=None):
            acks.append({"response_id": response_id, "item_id": item_id, "played_ms": int(played_ms)})

    class _Writer:
        def handle(self, event):
            pass

    asyncio.run(demo._StreamingEventCollector(_Writer()).consume(_Client()))

    assert acks == [{"response_id": "resp-audio", "item_id": "item_resp-audio", "played_ms": 100}]


def _asymmetric_image():
    from PIL import Image

    image = Image.new("RGB", (4, 2), (0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0))
    return image


@pytest.mark.parametrize(
    "rotation,expected_corner",
    [
        (0, (0, 0)),
        (90, (0, 3)),
        (180, (3, 1)),
        (-90, (1, 0)),
        (270, (1, 0)),
        (-270, (0, 3)),
    ],
)
def test_upright_applies_the_display_matrix_quarter_turn(rotation, expected_corner):
    demo = _load_demo_module()

    rotated = demo._upright(_asymmetric_image(), rotation)

    assert rotated.getpixel(expected_corner) == (255, 0, 0)


def test_decoded_frames_apply_the_container_display_matrix(monkeypatch, tmp_path):
    """A -90 phone clip must reach the model upright, not on its side."""
    demo = _load_demo_module()
    source = _asymmetric_image()

    class _Frame:
        rotation = -90

        def to_image(self):
            return source

    class _Container:
        streams = type("_Streams", (), {"video": [type("_Stream", (), {})()]})()

        def decode(self, stream):
            yield _Frame()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, "av", type("_Av", (), {"open": staticmethod(lambda path: _Container())}))

    (image,) = demo._decode_video_frames_rgb(tmp_path / "clip.mp4", [0])

    assert image.size == (2, 4)
    assert image.getpixel((1, 0)) == (255, 0, 0)


def test_streaming_output_writer_persists_audio_deltas_as_they_arrive(tmp_path, capsys):
    demo = _load_demo_module()
    writer = demo._StreamingOutputWriter(tmp_path)
    first_pcm = b"\x01\x00\x02\x00"
    second_pcm = b"\x03\x00"

    writer.handle(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(first_pcm).decode("ascii"),
            "sample_rate_hz": 24_000,
        }
    )
    writer.handle(
        {
            "type": "response.output_audio_transcript.delta",
            "response_id": "resp-a",
            "delta": "你好",
        }
    )
    writer.handle(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(second_pcm).decode("ascii"),
            "sample_rate_hz": 24_000,
        }
    )

    assert (tmp_path / "output.pcm").read_bytes() == first_pcm + second_pcm
    chunk_paths = sorted((tmp_path / "audio_chunks").glob("*.wav"))
    assert [path.name for path in chunk_paths] == ["chunk_0001.wav", "chunk_0002.wav"]
    with wave.open(str(chunk_paths[0]), "rb") as chunk_wav:
        assert chunk_wav.getframerate() == 24_000
        assert chunk_wav.readframes(chunk_wav.getnframes()) == first_pcm
    assert writer.audio_chunk_paths == chunk_paths
    stderr = capsys.readouterr().err
    assert "audio chunk 1" in stderr
    assert "你好" in stderr


@pytest.mark.parametrize("audio_samples", [3 * 16000 + 512, 2 * 16000])
def test_video_soundtrack_aligns_to_video_units(monkeypatch, tmp_path, audio_samples):
    """AAC padding must not create an extra frame-less model unit at EOF."""
    demo = _load_demo_module()
    audio_path = tmp_path / "input.wav"
    pcm = b"\x01\x00" * audio_samples
    demo.write_pcm16_wav(audio_path, pcm, sample_rate_hz=16000)
    monkeypatch.setattr(demo, "_probe_video_frames_and_fps", lambda path: (90, 30.0))
    monkeypatch.setattr(demo, "_extract_video_soundtrack", lambda *args: audio_path)

    def extract(path, work_dir, **kwargs):
        assert kwargs["duration_s"] == 3.0
        return ["f0", "f1", "f2"], [None] * 3

    monkeypatch.setattr(demo, "_extract_video_frames", extract)
    wav, frames, _ = demo._extract_video_input(tmp_path / "clip.mp4", tmp_path, fps=1.0, max_side=0, stack_frames=1)
    aligned = demo.read_pcm16_wav(wav)
    assert aligned == pcm[:96000].ljust(96000, b"\0")
    assert len(frames) == 3
    assert not demo._has_residual_model_unit(aligned, chunk_period_ms=1000)


def test_video_without_soundtrack_uses_video_duration(monkeypatch, tmp_path):
    demo = _load_demo_module()
    monkeypatch.setattr(demo, "_video_has_audio", lambda path: False)
    monkeypatch.setattr(demo, "_probe_video_frames_and_fps", lambda path: (90, 30.0))

    def extract(path, work_dir, **kwargs):
        assert kwargs["duration_s"] == 3.0
        return ["f0", "f1", "f2"], [None] * 3

    monkeypatch.setattr(demo, "_extract_video_frames", extract)
    wav, frames, _ = demo._resolve_duplex_av_inputs(
        input_wav=None,
        input_video="silent.mp4",
        work_dir=tmp_path,
        fps=1.0,
        max_side=0,
    )
    assert wav is None
    assert len(frames) == 3

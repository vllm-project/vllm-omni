# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise actual client output files without the model-serving dependencies."""

import importlib
import io
import math
import subprocess
import sys
import textwrap
from array import array
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import av
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.example]

_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples/online_serving/streaming_video_generation"
_PIXELS = [32, 64, 96, 128, 160, 192]
_SAVE_SCRIPT = textwrap.dedent("""
    import importlib.abc
    import sys
    from pathlib import Path

    class RejectInferenceImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'torch', 'vllm', 'vllm_omni', 'numpy'}:
                raise ImportError(f'Lightweight client imported {fullname}')

    sys.meta_path.insert(0, RejectInferenceImports())
    sys.path.insert(0, sys.argv[1])
    from streaming_video_client import save_video

    data = Path(sys.argv[2]).read_bytes()
    chunks = [data[i:i + 137] for i in range(0, len(data), 137)]
    save_video(chunks, Path(sys.argv[3]), fps=float(sys.argv[4]), stream_format='m4s')
""")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(_EXAMPLE_DIR))
    return importlib.import_module("streaming_video_client")


def _make_stream(*, audio: bool = False) -> bytes:
    buffer = io.BytesIO()
    with av.open(
        buffer, mode="w", format="mp4", options={"movflags": "frag_keyframe+empty_moov+default_base_moof"}
    ) as container:
        video = container.add_stream("h264", rate=24)
        video.width, video.height = 32, 24
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "0", "g": "2", "bf": "0", "sc_threshold": "0"}
        sound = container.add_stream("aac", rate=48000) if audio else None
        if sound is not None:
            sound.layout = "mono"
        packets = []
        for index, value in enumerate(_PIXELS):
            frame = av.VideoFrame(32, 24, "rgb24")
            frame.planes[0].update(bytes([value]) * frame.planes[0].buffer_size)
            frame.pts = index
            frame.time_base = Fraction(1, 24)
            packets.extend(video.encode(frame))
        packets.extend(video.encode())
        if sound is not None:
            for index in range(24):
                frame = av.AudioFrame(format="fltp", layout="mono", samples=1024)
                frame.sample_rate = 48000
                frame.pts = index * 1024
                frame.time_base = Fraction(1, 48000)
                samples = array(
                    "f", (0.1 * math.sin(2 * math.pi * 440 * (index * 1024 + n) / 48000) for n in range(1024))
                )
                frame.planes[0].update(samples.tobytes())
                packets.extend(sound.encode(frame))
            packets.extend(sound.encode())
        packets.sort(key=lambda packet: float(packet.dts or 0) * float(packet.time_base or 1))
        for packet in packets:
            container.mux(packet)
    return buffer.getvalue()


def _boxes(data: bytes) -> list[tuple[bytes, int]]:
    boxes = []
    offset = 0
    while offset < len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        assert size >= 8
        offset += size
        boxes.append((data[offset - size + 4 : offset - size + 8], offset))
    assert offset == len(data)
    return boxes


def _save_in_subprocess(data: bytes, output: Path, fps: float) -> None:
    source = output.parent / "input.m4s"
    source.write_bytes(data)
    try:
        subprocess.run(
            [sys.executable, "-I", "-c", _SAVE_SCRIPT, str(_EXAMPLE_DIR), str(source), str(output), str(fps)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"Real client save failed:\n{exc.stderr}")


@pytest.mark.parametrize("fps", [Fraction(12), Fraction(30000, 1001)], ids=["integer-fps", "fractional-fps"])
@pytest.mark.parametrize("interrupted", [False, True], ids=["complete", "interrupted"])
def test_lightweight_save_produces_playable_mp4(tmp_path: Path, fps: Fraction, interrupted: bool) -> None:
    data = _make_stream()
    if interrupted:
        # Stop at a complete fragment, before subsequent frames or the trailer.
        end = next(end for kind, end in _boxes(data) if kind == b"mdat")
        data = data[:end]
    expected = _PIXELS[:2] if interrupted else _PIXELS
    with av.open(io.BytesIO(data)) as source:
        assert len(list(source.decode(video=0))) == len(expected)

    output = tmp_path / "saved.mp4"
    _save_in_subprocess(data, output, float(fps))

    box_types = [kind for kind, _ in _boxes(output.read_bytes())]
    assert b"moov" in box_types and b"moof" not in box_types
    with av.open(str(output)) as saved:
        stream = saved.streams.video[0]
        assert float(stream.average_rate) == pytest.approx(float(fps))
        assert float(stream.duration * stream.time_base) == pytest.approx(len(expected) / float(fps), abs=0.0001)
        frames = list(saved.decode(video=0))
        assert len(frames) == len(expected)
        assert [(frame.width, frame.height) for frame in frames] == [(32, 24)] * len(expected)
        assert [bytes(frame.reformat(format="rgb24").planes[0])[0] for frame in frames] == pytest.approx(
            expected, abs=3
        )
        assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx(
            [index / float(fps) for index in range(len(expected))], abs=0.0001
        )


def _media_packets(data: bytes) -> dict[str, list[tuple[bytes, Fraction, Fraction]]]:
    result: dict[str, list[tuple[bytes, Fraction, Fraction]]] = {"video": [], "audio": []}
    with av.open(io.BytesIO(data)) as container:
        for packet in container.demux():
            if packet.dts is not None:
                assert packet.pts is not None and packet.time_base is not None
                result[packet.stream.type].append(
                    (bytes(packet), packet.pts * packet.time_base, packet.dts * packet.time_base)
                )
    return result


def test_lightweight_save_preserves_audio_and_video_packets(tmp_path: Path) -> None:
    data = _make_stream(audio=True)
    output = tmp_path / "saved.mp4"
    _save_in_subprocess(data, output, 12.0)
    expected = _media_packets(data)
    assert expected["video"] and expected["audio"]
    assert _media_packets(output.read_bytes()) == expected
    assert b"moof" not in [kind for kind, _ in _boxes(output.read_bytes())]
    with av.open(str(output)) as saved:
        assert len(list(saved.decode(video=0))) == len(_PIXELS)
    with av.open(str(output)) as saved:
        assert sum(frame.samples for frame in saved.decode(audio=0)) > 0


@pytest.mark.parametrize("width", [32, 34], ids=["aligned", "padded-rows"])
def test_save_preserves_bt709_colors(tmp_path: Path, width: int) -> None:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4", options={"movflags": "frag_keyframe+empty_moov"}) as container:
        video = container.add_stream("h264", rate=24)
        video.width, video.height = width, 24
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "0", "bf": "0"}
        video.codec_context.colorspace = 1  # BT.709
        video.codec_context.color_primaries = 1
        video.codec_context.color_trc = 1
        video.codec_context.color_range = 1  # Limited range
        for index in range(6):
            frame = av.VideoFrame(width, 24, "yuv420p")
            frame.colorspace = 1
            frame.color_range = 1
            for plane, value in zip(frame.planes, [105, 111, 214]):
                plane.update(bytes([value]) * plane.buffer_size)
            frame.pts = index
            frame.time_base = Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

    def center_pixels(data: bytes) -> list[bytes]:
        with av.open(io.BytesIO(data)) as container:
            values = []
            for frame in container.decode(video=0):
                rgb = frame.reformat(format="rgb24").planes[0]
                offset = 12 * rgb.line_size + (width // 2) * 3
                values.append(bytes(rgb)[offset : offset + 3])
            return values

    data = buffer.getvalue()
    expected = center_pixels(data)
    output = tmp_path / "saved.mp4"
    _save_in_subprocess(data, output, 12.0)
    actual = center_pixels(output.read_bytes())
    assert len(actual) == len(expected) == 6
    for received, saved in zip(expected, actual):
        assert list(saved) == pytest.approx(list(received), abs=3)


def test_empty_input_does_not_create_output(client: ModuleType, tmp_path: Path) -> None:
    output = tmp_path / "missing" / "saved.mp4"
    client.save_video([], output, fps=12.0, stream_format="m4s")
    assert not output.parent.exists()


@pytest.mark.parametrize("data", [b"", b"not a media container"])
def test_undecodable_input_is_retained(client: ModuleType, tmp_path: Path, data: bytes) -> None:
    output = tmp_path / "saved.mp4"
    client.save_video([data], output, fps=12.0, stream_format="m4s")
    assert output.read_bytes() == data


def test_save_creates_parent_and_replaces_destination(client: ModuleType, tmp_path: Path) -> None:
    output = tmp_path / "nested" / "saved.mp4"
    client.save_video([_make_stream()], output, fps=12.0, stream_format="m4s")
    output.write_bytes(b"old file")
    # An unrelated predictable .tmp path must not be overwritten.
    sibling = output.with_suffix(".mp4.tmp")
    sibling.write_bytes(b"unrelated")
    client.save_video([_make_stream()], output, fps=12.0, stream_format="m4s")
    assert sibling.read_bytes() == b"unrelated"
    with av.open(str(output)) as saved:
        assert len(list(saved.decode(video=0))) == len(_PIXELS)
    assert set(output.parent.iterdir()) == {output, sibling}


def test_failed_replace_preserves_destination_and_cleans_temporary_file(
    client: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "saved.mp4"
    output.write_bytes(b"old file")

    def fail_replace(source: Path, destination: Path) -> None:
        assert source.is_file() and source.parent == destination.parent
        raise OSError("publication failed")

    monkeypatch.setattr(client.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failed"):
        client.save_video([_make_stream()], output, fps=12.0, stream_format="m4s")
    assert output.read_bytes() == b"old file"
    assert list(tmp_path.iterdir()) == [output]


def test_encoding_failure_preserves_destination(
    client: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _make_stream()
    output = tmp_path / "saved.mp4"
    output.write_bytes(b"old file")
    real_open = av.open

    def fail_output_open(file, mode="r", **kwargs):
        if mode == "w":
            raise RuntimeError("encoder unavailable")
        return real_open(file, mode=mode, **kwargs)

    monkeypatch.setattr(client.av, "open", fail_output_open)
    with pytest.raises(RuntimeError, match="encoder unavailable"):
        client.save_video([data], output, fps=12.0, stream_format="m4s")
    assert output.read_bytes() == b"old file"
    assert list(tmp_path.iterdir()) == [output]

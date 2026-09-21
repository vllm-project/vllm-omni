# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Do not import package __init__ files, model registries, torch, or the GPU pipeline.
_PATH = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/minimax_h3/timeline_guides.py"
_SPEC = importlib.util.spec_from_file_location("_test_h3_timeline_guides_cpu", _PATH)
guides = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = guides
_SPEC.loader.exec_module(guides)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def limits():
    return guides.TimelineGuideLimits()


@pytest.mark.parametrize(
    "source,expected", [(1, 1), (2, 1), (4, 1), (5, 5), (6, 5), (21, 5), (22, 22), (23, 22), (38, 22), (39, 39)]
)
def test_visual_normalization(source, expected):
    assert guides.normalize_visual_frame_count(source) == expected


@pytest.mark.parametrize("source", [0, -1, True, 1.0, "22", None])
def test_visual_normalization_rejects_invalid_counts(source):
    with pytest.raises(ValueError):
        guides.normalize_visual_frame_count(source)


@pytest.mark.parametrize(
    "index,length,expected", [(36, 1, 36), (-1, 1, 123), (-22, 22, 102), (-124, 22, 0), (102, 22, 102), (0, 124, 0)]
)
def test_resolve_exact_pixel_starts(index, length, expected):
    assert guides.resolve_guide_start(index, 124, length) == expected


@pytest.mark.parametrize("index,length", [(-1, 22), (-125, 1), (124, 1), (103, 22), (True, 1), (1.0, 1), (0, 0)])
def test_start_rejects_overflow_without_trimming(index, length):
    with pytest.raises(ValueError):
        guides.resolve_guide_start(index, 124, length)


@pytest.mark.parametrize("start,expected", [(0, 207), (1, 205), (36, 147), (102, 37), (123, 2)])
def test_audio_limit_uses_fractional_origin(start, expected):
    assert guides.guide_audio_limit(124, start) == expected


@pytest.mark.parametrize("start", [-1, 124, True, 1.0])
def test_audio_limit_requires_resolved_valid_start(start):
    with pytest.raises(ValueError):
        guides.guide_audio_limit(124, start)


def test_audio_only_final_frame_for_all_native_output_lengths():
    for k in range(100):
        output_frames = 5 + 17 * k
        start = guides.resolve_guide_start(-1, output_frames)
        assert guides.guide_audio_limit(output_frames, start) >= 1


def test_limits_defaults_and_config(limits):
    assert guides.GUIDES_EXTRA_KEY == "_minimax_h3_timeline_guides"
    assert [limits.max_entries, limits.max_unique_files, limits.max_outstanding_requests] == [8, 16, 4]
    assert [limits.max_image_bytes, limits.max_video_bytes, limits.max_audio_bytes] == [30 << 20, 50 << 20, 15 << 20]
    assert limits.max_total_upload_bytes == 128 << 20
    assert [limits.max_guide_rows, limits.max_packed_rows] == [65536, 262144]
    assert limits.max_source_pixels == 16777216
    assert limits.max_decoded_visual_pixels == 268435456
    assert limits.max_decoded_audio_samples == 8388608
    assert limits.subprocess_timeout_seconds == 60
    assert guides.TimelineGuideLimits.from_config(None) == limits
    assert guides.TimelineGuideLimits.from_config({"unrelated": 0}) == limits
    configured = guides.TimelineGuideLimits.from_config({"minimax_h3_timeline_guides": {"max_entries": 2}})
    assert configured == replace(limits, max_entries=2)
    with pytest.raises(FrozenInstanceError):
        configured.max_entries = 3


@pytest.mark.parametrize("field", [field.name for field in fields(guides.TimelineGuideLimits)])
@pytest.mark.parametrize("value", [0, -1, True, "4", None, float("inf"), float("nan")])
def test_limits_reject_nonfinite_unlimited_or_non_numeric(field, value):
    with pytest.raises(ValueError):
        guides.TimelineGuideLimits.from_config({"minimax_h3_timeline_guides": {field: value}})


@pytest.mark.parametrize(
    "config", [[], {"minimax_h3_timeline_guides": None}, {"minimax_h3_timeline_guides": {"unknown_limit": 8}}]
)
def test_limits_reject_malformed_config(config):
    with pytest.raises(ValueError):
        guides.TimelineGuideLimits.from_config(config)


def test_descriptors_preserve_order_reuse_overlap_and_count_bytes_once(tmp_path, limits):
    source = tmp_path / "still.png"
    source.write_bytes(b"placeholder")
    items = [
        {"frame_index": 36, "image": str(source)},
        {"frame_index": 0, "image": str(source)},
        {"frame_index": 36, "image": str(source)},
    ]
    limits = replace(limits, max_unique_files=1, max_total_upload_bytes=source.stat().st_size)
    result = guides.validate_guide_descriptors(items, limits)
    assert result == items
    assert result is not items and result[0] is not items[0]
    assert guides.validate_guide_descriptors([], limits) == []


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"frame_index": True, "image": "x"},
        {"frame_index": 0.0, "image": "x"},
        {"frame_index": 0},
        {"frame_index": 0, "image": "x", "video": "y"},
        {"frame_index": 0, "mask": "x"},
        {"frame_index": 0, "audio": None},
        {"frame_index": 0, "audio": {"upload_index": 0}},
        {"frame_index": 0, "image": "https://example.com/a"},
    ],
)
def test_invalid_descriptors(item, limits):
    with pytest.raises(ValueError):
        guides.validate_guide_descriptors([item], limits)


def test_descriptor_file_and_entry_budgets(tmp_path, limits):
    image = tmp_path / "image"
    audio = tmp_path / "audio"
    image.write_bytes(b"12")
    audio.write_bytes(b"345")
    items = [{"frame_index": 0, "image": str(image), "audio": str(audio)}]
    assert guides.validate_guide_descriptors(items, limits) == items
    for options in (
        {"max_unique_files": 1},
        {"max_image_bytes": 1},
        {"max_audio_bytes": 2},
        {"max_total_upload_bytes": 4},
    ):
        with pytest.raises(ValueError):
            guides.validate_guide_descriptors(items, replace(limits, **options))
    with pytest.raises(ValueError, match="entries"):
        guides.validate_guide_descriptors(items * 2, replace(limits, max_entries=1))
    for path in (tmp_path, tmp_path / "missing"):
        with pytest.raises(ValueError):
            guides.validate_guide_descriptors([{"frame_index": 0, "image": str(path)}], limits)
    image.write_bytes(b"")
    with pytest.raises(ValueError, match="nonempty"):
        guides.validate_guide_descriptors([{"frame_index": 0, "image": str(image)}], limits)


def test_aggregate_budget_checks_are_atomic(limits):
    limits = replace(
        limits, max_decoded_visual_pixels=10, max_decoded_audio_samples=10, max_guide_rows=10, max_packed_rows=10
    )
    budget = guides.TimelineGuideBudget(limits)
    for method, attr in (
        (budget.add_visual, "decoded_visual_pixels"),
        (budget.add_audio, "decoded_audio_samples"),
        (budget.add_guide_rows, "guide_rows"),
    ):
        method(5)
        method(5)
        with pytest.raises(ValueError):
            method(1)
        assert getattr(budget, attr) == 10
    budget.check_packed_rows(10)
    with pytest.raises(ValueError):
        budget.check_packed_rows(11)


def test_still_rgb_center_crop_and_reused_decode_budget(tmp_path, limits):
    pixels = np.zeros((12, 36, 3), dtype=np.uint8)
    pixels[:, :12] = [255, 0, 0]
    pixels[:, 12:24] = [0, 255, 0]
    pixels[:, 24:] = [0, 0, 255]
    source = tmp_path / "wide.png"
    Image.fromarray(pixels).save(source)
    limits = replace(limits, max_decoded_visual_pixels=12 * 36)
    budget = guides.TimelineGuideBudget(limits)
    decoded = guides.decode_guide_image(str(source), 12, 12, limits, budget=budget)
    assert decoded.mode == "RGB" and decoded.size == (12, 12)
    np.testing.assert_array_equal(np.asarray(decoded)[6, 6], [0, 255, 0])
    with pytest.raises(ValueError, match="aggregate"):
        guides.decode_guide_image(str(source), 12, 12, limits, budget=budget)


def test_still_rejects_pixels_before_loading(tmp_path, monkeypatch, limits):
    source = tmp_path / "large.png"
    Image.new("RGB", (10, 10)).save(source)
    monkeypatch.setattr(Image.Image, "load", lambda *_: pytest.fail("oversized image was loaded"))
    with pytest.raises(ValueError, match="pixel limit"):
        guides.decode_guide_image(str(source), 2, 2, replace(limits, max_source_pixels=99))


def test_still_rejects_animated_and_invalid_files(tmp_path, limits):
    source = tmp_path / "animated.gif"
    Image.new("RGB", (4, 4), "red").save(source, save_all=True, append_images=[Image.new("RGB", (4, 4), "blue")])
    with pytest.raises(ValueError, match="still image"):
        guides.decode_guide_image(str(source), 4, 4, limits)
    source.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="invalid"):
        guides.decode_guide_image(str(source), 4, 4, limits)


def test_still_rejects_external_delegate_formats(tmp_path, monkeypatch, limits):
    from PIL import EpsImagePlugin

    source = tmp_path / "delegate.eps"
    Image.new("RGB", (4, 4)).save(source, "EPS")
    monkeypatch.setattr(
        EpsImagePlugin, "Ghostscript", lambda *_args, **_kwargs: pytest.fail("external delegate invoked")
    )
    with pytest.raises(ValueError, match="invalid"):
        guides.decode_guide_image(str(source), 4, 4, limits)


def test_still_animation_detection_does_not_scan_all_frames(tmp_path, monkeypatch, limits):
    from PIL import GifImagePlugin

    source = tmp_path / "animated.gif"
    Image.new("RGB", (4, 4), "red").save(source, save_all=True, append_images=[Image.new("RGB", (4, 4), "blue")])
    monkeypatch.setattr(GifImagePlugin.GifImageFile, "n_frames", property(lambda _: pytest.fail("all-frame scan")))
    with pytest.raises(ValueError, match="still image"):
        guides.decode_guide_image(str(source), 4, 4, limits)


def test_bounded_subprocess_success_and_exit_error(limits):
    assert guides._run_bounded([sys.executable, "-c", "print('ok')"], limits, 3) == b"ok\n"
    with pytest.raises(ValueError, match="failed"):
        guides._run_bounded([sys.executable, "-c", "raise SystemExit(2)"], limits, 10)
    with pytest.raises(ValueError, match="cannot start"):
        guides._run_bounded(["/no-such-guide-media-tool"], limits, 10)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_bounded_subprocess_caps_both_pipes(stream, limits):
    with pytest.raises(ValueError, match="exceeds budget"):
        guides._run_bounded([sys.executable, "-c", f"import sys; sys.{stream}.write('x' * 100000)"], limits, 10)


def test_bounded_subprocess_timeout_kills_and_reaps(monkeypatch, limits):
    original = subprocess.Popen
    children = []

    def track(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(guides.subprocess, "Popen", track)
    with pytest.raises(ValueError, match="timed out"):
        guides._run_bounded(
            [sys.executable, "-c", "import time; time.sleep(20)"], replace(limits, subprocess_timeout_seconds=0.1), 10
        )
    assert children[0].poll() is not None
    assert children[0].stdout.closed and children[0].stderr.closed


@pytest.mark.parametrize(
    "script,error",
    [
        (
            "import sys; sys.stdout.write('x' * 100000); sys.stdout.flush(); import time; time.sleep(20)",
            "exceeds budget",
        ),
        (
            "import sys; sys.stderr.write('x' * 100000); sys.stderr.flush(); import time; time.sleep(20)",
            "exceeds budget",
        ),
        ("import sys; sys.stdout.write('partial'); raise SystemExit(2)", "failed"),
        ("import os,time; os.close(1); os.close(2); time.sleep(20)", "timed out"),
    ],
)
def test_bounded_subprocess_failures_close_pipes_and_reap(monkeypatch, limits, script, error):
    original = subprocess.Popen
    children = []

    def track(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(guides.subprocess, "Popen", track)
    with pytest.raises(ValueError, match=error):
        guides._run_bounded([sys.executable, "-c", script], replace(limits, subprocess_timeout_seconds=0.5), 10)
    assert children[0].poll() is not None
    assert children[0].stdout.closed and children[0].stderr.closed


@pytest.mark.parametrize("failure", [OSError("pipe failure"), KeyboardInterrupt()])
def test_bounded_subprocess_io_error_and_interruption_cleanup(monkeypatch, limits, failure):
    original = subprocess.Popen
    children = []

    def fail_read(*_args):
        raise failure

    def track(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        # Install after Popen has read its own internal exec-error pipe.
        monkeypatch.setattr(guides.os, "read", fail_read)
        return child

    monkeypatch.setattr(guides.subprocess, "Popen", track)
    expected = ValueError if isinstance(failure, OSError) else KeyboardInterrupt
    with pytest.raises(expected):
        guides._run_bounded(
            [sys.executable, "-c", "print('ready', flush=True); import time; time.sleep(20)"], limits, 10
        )
    assert children[0].poll() is not None
    assert children[0].stdout.closed and children[0].stderr.closed


def test_probe_is_bounded_and_never_scans_all_frames(tmp_path, monkeypatch, limits):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"placeholder")

    def run(command, actual_limits, cap):
        assert actual_limits is limits and cap == 65536
        assert "-count_frames" not in command and "-show_frames" not in command
        assert "-nofind_stream_info" in command
        assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
        assert command[command.index("-select_streams") + 1] == "v:0"
        assert command[command.index("-max_pixels") + 1] == str(min(128 * (limits.max_source_pixels + 127), 2**31 - 1))
        assert command[command.index("-max_samples") + 1] == str(limits.max_decoded_audio_samples)
        assert int(command[command.index("-max_alloc") + 1]) > 0
        assert command[command.index("-threads") + 1] == "1"
        return json.dumps({"streams": [{"codec_type": "video", "width": 8, "height": 8}]}).encode()

    monkeypatch.setattr(guides, "_run_bounded", run)
    assert guides._probe(str(source), "video", limits)["width"] == 8


def test_probe_visible_pixel_admission_is_separate_from_padding(tmp_path, monkeypatch, limits):
    source = tmp_path / "130.mp4"
    source.write_bytes(b"placeholder")
    limits = replace(limits, max_source_pixels=130 * 130)

    def run(command, *_args):
        assert int(command[command.index("-max_pixels") + 1]) >= 192 * 130
        return json.dumps({"streams": [{"codec_type": "video", "width": 130, "height": 130}]}).encode()

    monkeypatch.setattr(guides, "_run_bounded", run)
    assert guides._probe(str(source), "video", limits)["width"] == 130
    with pytest.raises(ValueError, match="pixel limit"):
        guides._probe(str(source), "video", replace(limits, max_source_pixels=16899))


def test_audio_probe_omits_video_pixel_option_but_keeps_sample_cap(tmp_path, monkeypatch, limits):
    source = tmp_path / "short.wav"
    source.write_bytes(b"placeholder")

    def run(command, *_args):
        assert "-max_pixels" not in command
        assert command[command.index("-max_samples") + 1] == str(limits.max_decoded_audio_samples)
        assert "-max_alloc" in command
        return json.dumps({"streams": [{"codec_type": "audio"}]}).encode()

    monkeypatch.setattr(guides, "_run_bounded", run)
    assert guides._probe(str(source), "audio", limits)["codec_type"] == "audio"


@pytest.mark.parametrize("payload", [b"not json", b"{}", b'{"streams": []}', b'{"streams": [{"codec_type": "audio"}]}'])
def test_probe_rejects_invalid_or_mismatched_media(tmp_path, monkeypatch, limits, payload):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(guides, "_run_bounded", lambda *args: payload)
    with pytest.raises(ValueError):
        guides._probe(str(source), "video", limits)


def test_video_normalizes_without_trimming_to_placement(monkeypatch, limits):
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 2, "height": 2})
    calls = []

    def run(command, actual_limits, cap, **kwargs):
        calls.append(command)
        kwargs["on_stderr"](b"frame:0 pts:0\n" * 23)
        return bytes(23 * 2 * 2 * 3)

    monkeypatch.setattr(guides, "_run_bounded", run)
    budget = guides.TimelineGuideBudget(limits)
    images = guides.decode_guide_video("trusted.mp4", 2, 2, 124, limits, budget=budget)
    assert len(images) == 22
    assert budget.decoded_visual_pixels == 23 * 4
    assert all(image.mode == "RGB" for image in images)
    assert "fps=24" in calls[0][calls[0].index("-vf") + 1]
    assert "-nofind_stream_info" in calls[0]
    assert calls[0][calls[0].index("-max_pixels") + 1] == str(128 * 128)
    assert "-an" in calls[0]
    assert guides.resolve_guide_start(-22, 124, len(images)) == 102
    with pytest.raises(ValueError):
        guides.resolve_guide_start(-1, 124, len(images))


def test_video_source_telemetry_counts_fragmented_lines_before_fps(monkeypatch, limits):
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 2, "height": 2})
    limits = replace(limits, max_decoded_visual_pixels=12)

    def run(command, _limits, _cap, **kwargs):
        filters = command[command.index("-vf") + 1]
        assert "trim=end_frame=4" in filters
        assert filters.index("trim=") < filters.index("metadata=mode=print") < filters.index("fps=24")
        for chunk in (b"fra", b"me:0 pts:0\nframe:", b"1 pts:1\nframe:2 pts:2\n"):
            kwargs["on_stderr"](chunk)
        return bytes(12)  # Only one output frame, but all three source frames count.

    monkeypatch.setattr(guides, "_run_bounded", run)
    budget = guides.TimelineGuideBudget(limits)
    assert len(guides.decode_guide_video("trusted.mp4", 2, 2, 1, limits, budget=budget)) == 1
    assert budget.decoded_visual_pixels == 12


def test_video_rejects_missing_source_telemetry(monkeypatch, limits):
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 2, "height": 2})
    monkeypatch.setattr(guides, "_run_bounded", lambda *args, **kwargs: bytes(12))
    with pytest.raises(ValueError, match="did not report source frames"):
        guides.decode_guide_video("trusted.mp4", 2, 2, 1, limits)


def test_source_frame_budget_failure_kills_and_reaps_decoder(monkeypatch, limits):
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 2, "height": 2})
    original = subprocess.Popen
    children = []

    def spawn(_command, **kwargs):
        child = original(
            [
                sys.executable,
                "-c",
                "import sys,time; sys.stderr.write('frame:0 pts:0\\n' * 3); sys.stderr.flush(); time.sleep(20)",
            ],
            **kwargs,
        )
        children.append(child)
        return child

    monkeypatch.setattr(guides.subprocess, "Popen", spawn)
    with pytest.raises(ValueError, match="source-frame pixel budget"):
        guides.decode_guide_video("trusted.mp4", 2, 2, 1, replace(limits, max_decoded_visual_pixels=8))
    assert children[0].poll() is not None
    assert children[0].stdout.closed and children[0].stderr.closed


@pytest.mark.parametrize("duration", ["100000", "nan", "inf", "0", "broken"])
def test_video_rejects_oversize_or_invalid_duration_before_decode(monkeypatch, limits, duration):
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 2, "height": 2, "duration": duration})
    monkeypatch.setattr(guides, "_run_bounded", lambda *_: pytest.fail("oversized video was decoded"))
    with pytest.raises(ValueError):
        guides.decode_guide_video("trusted.mp4", 2, 2, 124, limits)


@pytest.mark.parametrize("raw", [b"", bytes(3), bytes(5 * 12)])
def test_video_bounded_fallback_rejects_empty_partial_or_sentinel(monkeypatch, limits, raw):
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 2, "height": 2})
    monkeypatch.setattr(guides, "_run_bounded", lambda *_, **kwargs: raw)
    with pytest.raises(ValueError):
        guides.decode_guide_video("trusted.mp4", 2, 2, 1, limits)


@pytest.mark.parametrize(
    "raw", [b"", b"x", np.zeros((11, 2), dtype="<f4").tobytes(), np.full((1, 2), np.nan, dtype="<f4").tobytes()]
)
def test_audio_rejects_empty_partial_oversized_nonfinite(monkeypatch, limits, raw):
    monkeypatch.setattr(guides, "_probe", lambda *_: {})
    monkeypatch.setattr(guides, "_run_bounded", lambda *_: raw)
    with pytest.raises(ValueError):
        guides.decode_guide_audio("trusted.flac", replace(limits, max_decoded_audio_samples=20))


def test_audio_channel_major_short_decode_and_aggregate_limit(monkeypatch, limits):
    monkeypatch.setattr(guides, "_probe", lambda *_: {})
    raw = np.array([[0.1, 0.2], [0.3, 0.4]], dtype="<f4").tobytes()

    def run(command, _limits, _cap):
        assert command[command.index("-t") + 1] == "0.00009375"
        assert "-max_pixels" not in command
        assert command[command.index("-max_samples") + 1] == "65536"
        assert _cap == 24
        return raw

    monkeypatch.setattr(guides, "_run_bounded", run)
    limits = replace(limits, max_decoded_audio_samples=4)
    budget = guides.TimelineGuideBudget(limits)
    audio, rate = guides.decode_guide_audio("trusted.flac", limits, budget=budget)
    assert rate == 32000 and audio.dtype == np.float32 and audio.shape == (2, 2)
    np.testing.assert_allclose(audio, [[0.1, 0.3], [0.2, 0.4]])
    with pytest.raises(ValueError):
        guides.decode_guide_audio("trusted.flac", limits, budget=budget)


@pytest.mark.parametrize("duration", ["100000", "nan", "inf", "0", "broken", "0.144"])
def test_audio_admission_uses_pcm_not_unreliable_duration(monkeypatch, limits, duration):
    monkeypatch.setattr(guides, "_probe", lambda *_: {"duration": duration})
    monkeypatch.setattr(guides, "_run_bounded", lambda *_: np.zeros((2, 2), dtype="<f4").tobytes())
    audio, rate = guides.decode_guide_audio("trusted.mp3", replace(limits, max_decoded_audio_samples=4))
    assert rate == 32000 and audio.shape == (2, 2)


def test_audio_probe_rounding_does_not_reject_exact_sample_budget(monkeypatch, limits):
    monkeypatch.setattr(guides, "_probe", lambda *_: {"duration": "0.000063"})
    monkeypatch.setattr(guides, "_run_bounded", lambda *_: np.zeros((2, 2), dtype="<f4").tobytes())
    audio, rate = guides.decode_guide_audio("trusted.flac", replace(limits, max_decoded_audio_samples=4))
    assert rate == 32000 and audio.shape == (2, 2)


@pytest.fixture
def real_ffmpeg(monkeypatch):
    binary = shutil.which("ffmpeg")
    if binary is None:
        try:
            import imageio_ffmpeg

            binary = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            pytest.skip("ffmpeg not installed")
    original = guides._run_bounded

    def run(command, *args, **kwargs):
        if command[0] == "ffmpeg":
            command = [binary, *command[1:]]
        return original(command, *args, **kwargs)

    monkeypatch.setattr(guides, "_run_bounded", run)
    return binary


@pytest.fixture
def h264_encoder(real_ffmpeg):
    encoders = subprocess.run(
        [real_ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True, timeout=20
    ).stdout
    names = {line.split()[1] for line in encoders.splitlines() if len(line.split()) >= 2}
    for name in ("libx264", "libopenh264"):
        if name in names:
            return name
    pytest.skip("ffmpeg has no software H.264 encoder for test fixtures")


@pytest.fixture
def real_probe():
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not installed; probe contract is covered with bounded mock output")


@pytest.mark.parametrize("fps", [12, 30, 60])
def test_real_video_decode_uses_elapsed_time_at_24fps(tmp_path, monkeypatch, limits, real_ffmpeg, h264_encoder, fps):
    source = tmp_path / "rate.mp4"
    subprocess.run(
        [
            real_ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=32x16:rate={fps}:duration=1",
            "-c:v",
            h264_encoder,
            "-threads",
            "1",
            str(source),
        ],
        check=True,
        timeout=20,
    )
    # Real decode with controlled metadata also runs where only bundled ffmpeg exists.
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 32, "height": 16, "duration": "1"})
    budget = guides.TimelineGuideBudget(limits)
    images = guides.decode_guide_video(str(source), 16, 16, 124, limits, budget=budget)
    assert len(images) == 22
    assert images[0].size == (16, 16)
    assert budget.decoded_visual_pixels == max(24, fps) * 32 * 16


@pytest.mark.parametrize("extension", ["wav", "mp3", "flac"])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("sample_budget", [2, 4, 100])
def test_real_short_audio_resamples_and_decodes_formats(
    tmp_path, monkeypatch, limits, real_ffmpeg, extension, channels, sample_budget
):
    source = tmp_path / f"audio.{extension}"
    subprocess.run(
        [
            real_ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=0.05",
            "-ac",
            str(channels),
            str(source),
        ],
        check=True,
        timeout=20,
    )
    monkeypatch.setattr(guides, "_probe", lambda *_: {"channels": channels})
    audio, rate = guides.decode_guide_audio(str(source), limits)
    assert rate == 32000 and audio.shape == (2, 1600)
    assert audio.dtype == np.float32 and np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.01
    np.testing.assert_allclose(audio[0], audio[1])
    with pytest.raises(ValueError, match="sample budget"):
        guides.decode_guide_audio(str(source), replace(limits, max_decoded_audio_samples=sample_budget))


@pytest.mark.parametrize("extension", ["wav", "mp3", "flac"])
@pytest.mark.parametrize("channels", [1, 2])
def test_real_probe_and_decode_end_to_end(tmp_path, limits, real_ffmpeg, real_probe, extension, channels):
    source = tmp_path / f"short.{extension}"
    subprocess.run(
        [
            real_ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=16000:duration=0.05",
            "-ac",
            str(channels),
            str(source),
        ],
        check=True,
        timeout=20,
    )
    exact = replace(limits, max_decoded_audio_samples=3200)
    budget = guides.TimelineGuideBudget(exact)
    audio, rate = guides.decode_guide_audio(str(source), exact, budget=budget)
    assert rate == 32000 and audio.shape == (2, 1600)
    assert budget.decoded_audio_samples == 3200 and np.max(np.abs(audio)) > 0.01
    np.testing.assert_allclose(audio[0], audio[1])
    with pytest.raises(ValueError, match="sample budget"):
        guides.decode_guide_audio(str(source), replace(exact, max_decoded_audio_samples=3198))


@pytest.mark.parametrize(
    "fps,frames,width,height,expected",
    [
        (12, 12, 32, 16, 22),
        (30, 30, 32, 16, 22),
        (60, 60, 32, 16, 22),
        (600, 600, 16, 16, 22),
        (24, 1, 130, 130, 1),
        (24, 22, 32, 16, 22),
    ],
)
def test_real_video_probe_and_decode_end_to_end(
    tmp_path, limits, real_ffmpeg, real_probe, h264_encoder, fps, frames, width, height, expected
):
    source = tmp_path / "guide.mp4"
    subprocess.run(
        [
            real_ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}",
            "-frames:v",
            str(frames),
            "-c:v",
            h264_encoder,
            "-threads",
            "1",
            str(source),
        ],
        check=True,
        timeout=20,
    )
    exact = replace(limits, max_source_pixels=width * height)
    budget = guides.TimelineGuideBudget(exact)
    images = guides.decode_guide_video(str(source), 16, 16, 124, exact, budget=budget)
    assert len(images) == expected and images[0].size == (16, 16)
    assert guides.resolve_guide_start(-expected, 124, len(images)) == 124 - expected
    assert budget.decoded_visual_pixels == max(frames, round(frames * 24 / fps)) * width * height
    if fps == 600:
        with pytest.raises(ValueError, match="source-frame pixel budget"):
            guides.decode_guide_video(str(source), 16, 16, 124, replace(exact, max_decoded_visual_pixels=24 * 256))


def test_real_high_fps_video_charges_source_frames(tmp_path, monkeypatch, limits, real_ffmpeg, h264_encoder):
    source = tmp_path / "high_fps.mp4"
    subprocess.run(
        [
            real_ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=16x16:rate=600:duration=1",
            "-c:v",
            h264_encoder,
            "-threads",
            "1",
            str(source),
        ],
        check=True,
        timeout=20,
    )
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 16, "height": 16, "duration": "1"})
    with pytest.raises(ValueError, match="source-frame pixel budget"):
        guides.decode_guide_video(str(source), 16, 16, 124, replace(limits, max_decoded_visual_pixels=24 * 256))
    exact = replace(limits, max_decoded_visual_pixels=600 * 256)
    budget = guides.TimelineGuideBudget(exact)
    assert len(guides.decode_guide_video(str(source), 16, 16, 124, exact, budget=budget)) == 22
    assert budget.decoded_visual_pixels == 600 * 256
    with pytest.raises(ValueError, match="pixel budget"):
        guides.decode_guide_video(str(source), 16, 16, 124, exact, budget=budget)


def test_real_video_visible_pixel_boundary_allows_codec_padding(
    tmp_path, monkeypatch, limits, real_ffmpeg, h264_encoder
):
    source = tmp_path / "padding.mp4"
    subprocess.run(
        [
            real_ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=130x130:rate=24",
            "-frames:v",
            "1",
            "-c:v",
            h264_encoder,
            "-threads",
            "1",
            str(source),
        ],
        check=True,
        timeout=20,
    )
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 130, "height": 130})
    exact = replace(limits, max_source_pixels=16900, max_decoded_visual_pixels=16900)
    budget = guides.TimelineGuideBudget(exact)
    images = guides.decode_guide_video(str(source), 32, 32, 124, exact, budget=budget)
    assert len(images) == 1 and images[0].size == (32, 32)
    assert budget.decoded_visual_pixels == 16900
    with pytest.raises(ValueError, match="pixel limit"):
        guides.decode_guide_video(str(source), 32, 32, 124, replace(exact, max_source_pixels=16899))


def test_real_mp3_padding_does_not_reject_exact_pcm_budget(tmp_path, monkeypatch, limits, real_ffmpeg):
    source = tmp_path / "padded.mp3"
    subprocess.run(
        [
            real_ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=0.05",
            str(source),
        ],
        check=True,
        timeout=20,
    )
    # This stream duration includes MP3 delay/padding; decoded PCM is only 50 ms.
    monkeypatch.setattr(guides, "_probe", lambda *_: {"channels": 1, "duration": "0.144"})
    exact = replace(limits, max_decoded_audio_samples=3200)
    audio, rate = guides.decode_guide_audio(str(source), exact)
    assert rate == 32000 and audio.shape == (2, 1600)
    with pytest.raises(ValueError, match="sample budget"):
        guides.decode_guide_audio(str(source), replace(exact, max_decoded_audio_samples=3198))


def test_real_video_center_crop_and_oversized_unknown_duration(
    tmp_path, monkeypatch, limits, real_ffmpeg, h264_encoder
):
    source = tmp_path / "crop.mp4"
    subprocess.run(
        [
            real_ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=48x16:rate=24:duration=1,drawbox=x=16:y=0:w=16:h=16:color=green:t=fill",
            "-c:v",
            h264_encoder,
            "-threads",
            "1",
            str(source),
        ],
        check=True,
        timeout=20,
    )
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 48, "height": 16})
    images = guides.decode_guide_video(str(source), 16, 16, 124, limits)
    assert len(images) == 22
    # Edges too must be green, not red letterboxing or a stretched wide source.
    pixels = np.asarray(images[0])
    assert np.all(pixels[:, :, 1] > 100)
    assert np.all(pixels[:, :, 0] < 20)
    with pytest.raises(ValueError, match="budget"):
        guides.decode_guide_video(str(source), 16, 16, 1, limits)


@pytest.mark.parametrize("fps,frames", [(24, 1), (24, 2), (24, 4), (60, 1)])
def test_real_short_video_keeps_first_frame(tmp_path, monkeypatch, limits, real_ffmpeg, h264_encoder, fps, frames):
    source = tmp_path / "short.mp4"
    subprocess.run(
        [
            real_ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=16x16:rate={fps}",
            "-frames:v",
            str(frames),
            "-c:v",
            h264_encoder,
            "-threads",
            "1",
            str(source),
        ],
        check=True,
        timeout=20,
    )
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 16, "height": 16})
    images = guides.decode_guide_video(str(source), 16, 16, 1, limits)
    assert len(images) == 1
    assert np.asarray(images[0]).max() > 0


def test_real_video_rejects_midstream_resolution_change(tmp_path, monkeypatch, limits, real_ffmpeg, h264_encoder):
    clips = []
    for width in (16, 32):
        result = subprocess.run(
            [
                real_ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size={width}x16:rate=24",
                "-frames:v",
                "3",
                "-c:v",
                h264_encoder,
                "-threads",
                "1",
                "-f",
                "h264",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
            timeout=20,
        )
        clips.append(result.stdout)
    source = tmp_path / "changed.mp4"
    subprocess.run(
        [real_ffmpeg, "-v", "error", "-r", "24", "-f", "h264", "-i", "pipe:0", "-c:v", "copy", str(source)],
        input=b"".join(clips),
        check=True,
        timeout=20,
    )
    monkeypatch.setattr(guides, "_probe", lambda *_: {"width": 16, "height": 16})
    with pytest.raises(ValueError, match="decode failed"):
        guides.decode_guide_video(str(source), 16, 16, 124, limits)

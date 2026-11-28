# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import contextlib
import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np
import psutil
import pytest
import torch
from PIL import Image

from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing
from vllm_omni.model_executor.models.minimax_h3 import reference_video

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FRAMES = 124
WIDTH, HEIGHT = 160, 96
ENV = reference_video.MINIMAX_H3_REFERENCE_DECODE_WORKERS_ENV
SINE = ["-f", "lavfi", "-i", "sine=sample_rate=44100"]
META = {
    "width": 1280,
    "height": 720,
    "fps": 24.0,
    "frame_count": 240,
    "duration": 5.0,
    "format_names": ("mp4",),
    "video_codec": "h264",
    "audio_codecs": (),
    "file_size": 1024,
    "stream_count": 2,
}


def _ffmpeg(*args) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *map(str, args)], check=True, timeout=60)


def _testsrc(size: str, fps: int) -> list[str]:
    return ["-f", "lavfi", "-i", f"testsrc2=s={size}:r={fps}"]


def _x264(pix_fmt: str = "yuv420p", preset: str = "ultrafast") -> list[str]:
    return ["-c:v", "libx264", "-preset", preset, "-pix_fmt", pix_fmt]


@pytest.fixture(scope="module")
def encoders() -> str:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "libx264rgb" not in encoders or not reference_video.direct_reference_decode_supported():
        pytest.skip("requires ffmpeg with libx264rgb on a release with direct reference decoding")
    return encoders


@pytest.fixture(scope="module")
def sources(tmp_path_factory, encoders) -> dict[str, str]:
    root = tmp_path_factory.mktemp("minimax_h3_reference_frames")
    names = ["s30_audio", "s25_short", "s60_rotate", "full_range", "late_video", "late_audio", "bframes_no_editlist"]
    path = {name: str(root / f"{name}.mp4") for name in names}
    path["timecode_mov"] = str(root / "timecode.mov")
    video, audio, bframes = root / "video.mp4", root / "audio.m4a", root / "bframes.mp4"
    _ffmpeg(*_testsrc("384x256", 30), *SINE, "-t", 6, *_x264(), "-c:a", "aac", path["s30_audio"])
    _ffmpeg(*_testsrc("288x256", 25), "-t", 3.1, *_x264(), path["s25_short"])
    _ffmpeg(*_testsrc("320x256", 60), "-t", 5, *_x264(), "-metadata:s:v:0", "rotate=90", path["s60_rotate"])
    _ffmpeg(*_testsrc("320x256", 24), "-t", 4, *_x264("yuvj420p"), path["full_range"])
    _ffmpeg(*_testsrc("320x256", 25), "-t", 5, *_x264(), "-timecode", "01:00:00:00", path["timecode_mov"])
    _ffmpeg(*_testsrc("320x256", 30), "-t", 6, *_x264(), video)
    _ffmpeg(*SINE, "-t", 6.2, "-c:a", "aac", audio)
    remux = ["-map", "0:v", "-map", "1:a", "-c", "copy"]
    _ffmpeg("-itsoffset", 0.1, "-i", video, "-i", audio, *remux, path["late_video"])
    _ffmpeg("-i", video, "-itsoffset", 0.5, "-i", audio, *remux, path["late_audio"])
    _ffmpeg(*_testsrc("320x256", 30), *SINE, "-t", 6, *_x264(preset="superfast"), "-bf", 2, "-c:a", "aac", bframes)
    _ffmpeg("-i", bframes, "-c", "copy", "-use_editlist", 0, path["bframes_no_editlist"])
    segments = {}
    annexb = ["-bsf:v", "h264_mp4toannexb", "-f", "h264"]
    for name, size, pix_fmt, seconds in (
        ("a", "320x256", "yuv420p", 3.1),
        ("b", "384x288", "yuv420p", 3),
        ("c", "320x256", "yuv444p", 3),
    ):
        segments[name] = root / f"segment_{name}.h264"
        _ffmpeg(*_testsrc(size, 30), "-t", seconds, *_x264(pix_fmt), *annexb, segments[name])
    with_audio = [*SINE, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-t", 6.1]
    for name, second, extra in (
        ("size_change", "b", []),
        ("size_change_audio", "b", with_audio),
        ("format_change_audio", "c", with_audio),
    ):
        stream = root / f"{name}.h264"
        stream.write_bytes(segments["a"].read_bytes() + segments[second].read_bytes())
        path[name] = str(root / f"{name}.mp4")
        _ffmpeg("-framerate", 30, "-i", stream, *extra, "-c:v", "copy", path[name])
    return path


@pytest.fixture(autouse=True)
def frame_jobs(monkeypatch):
    jobs: list[reference_video.ReferenceVideoFrames] = []
    init = reference_video.ReferenceVideoFrames.__init__

    def tracked_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        jobs.append(self)

    monkeypatch.setattr(reference_video.ReferenceVideoFrames, "__init__", tracked_init)
    yield jobs
    for job in jobs:
        job.cancel()


def _ffmpeg_children() -> list[int]:
    pids = []
    for child in psutil.Process().children(recursive=True):
        with contextlib.suppress(psutil.NoSuchProcess):
            if child.name() in ("ffmpeg", "ffprobe"):
                pids.append(child.pid)
    return pids


def _assert_reaped(*jobs: reference_video.ReferenceVideoFrames) -> None:
    for job in jobs:
        assert job._processes and all(process.returncode is not None for process in job._processes)
        assert all(pipe is None or pipe.closed for p in job._processes for pipe in (p.stdout, p.stderr))
        assert not any(thread.is_alive() for thread in job._threads)
    assert _ffmpeg_children() == []


@pytest.fixture(params=["ffmpeg", "decord"])
def legacy_decoder(request, monkeypatch) -> str:
    if request.param == "decord":
        pytest.importorskip("decord")
    else:
        monkeypatch.setitem(sys.modules, "decord", None)
    return request.param


def _transcode(source, workdir, *, frame_count=FRAMES, start=0.0, duration=None) -> str:
    return reference_video._transcode_reference_video(
        source,
        target_width=WIDTH,
        target_height=HEIGHT,
        target_frame_count=frame_count,
        workdir=str(workdir),
        start_time_seconds=start,
        duration_seconds=duration,
    )


def _read_frames(path) -> torch.Tensor:
    return processing._frames_to_tensor(reference_video.load_video_frames(path))


def _job(source, *, frame_count=FRAMES, start=0.0, duration=5.0, workers=3, stream_count=2):
    return reference_video.ReferenceVideoFrames(
        source,
        width=WIDTH,
        height=HEIGHT,
        frame_count=frame_count,
        start_time_seconds=start,
        duration_seconds=duration,
        workers=workers,
        stream_count=stream_count,
    )


def _probe(source, start=0.0, duration=None) -> tuple[float, int]:
    meta = reference_video._probe_video(source)
    return duration or float(meta["duration"]) - start, meta["stream_count"]


@pytest.mark.parametrize(
    ("name", "start", "duration", "frame_count", "expected"),
    [
        ("s30_audio", 0.0, None, FRAMES, FRAMES),
        ("s30_audio", 0.37, None, FRAMES, FRAMES),
        ("s30_audio", 0.0, 4.52, FRAMES, 108),
        ("s30_audio", 0.0, None, 5, 5),
        ("s30_audio", 0.37, 4.52, FRAMES, 109),
        ("late_video", 0.0, None, FRAMES, FRAMES),
        ("late_video", 0.0, 3.03, FRAMES, 73),
        ("late_audio", 0.37, 4.52, FRAMES, 109),
        ("bframes_no_editlist", 0.0, None, FRAMES, FRAMES),
        ("bframes_no_editlist", 0.37, 4.52, FRAMES, 109),
        ("timecode_mov", 0.0, None, FRAMES, 120),
        ("timecode_mov", 0.0, 3.03, FRAMES, 73),
        ("s25_short", 0.37, None, FRAMES, None),
        ("s60_rotate", 0.0, None, FRAMES, None),
        ("s60_rotate", 0.0, 4.52, FRAMES, 108),
    ],
)
def test_direct_frames_match_prepared_video(
    sources, tmp_path, legacy_decoder, frame_jobs, name, start, duration, frame_count, expected
):
    duration, stream_count = _probe(sources[name], start, duration)
    options = {"frame_count": frame_count, "start": start, "duration": duration}
    legacy = _read_frames(_transcode(sources[name], tmp_path, **options))
    assert legacy.shape[0] < frame_count if expected is None else legacy.shape[0] == expected
    for workers in (1, 3, 8):
        job = _job(sources[name], workers=workers, stream_count=stream_count, **options)
        frames = job.result()
        assert job.workers == min(workers, frame_count)
        assert frames.dtype == torch.uint8 and frames.is_contiguous() and torch.equal(frames, legacy), workers
        _assert_reaped(job)
    assert len(frame_jobs) == 3


@pytest.mark.parametrize(
    ("name", "start", "duration"),
    [
        ("size_change", 0.0, None),
        ("size_change", 0.0, 4.52),
        ("size_change_audio", 0.37, 4.52),
        ("format_change_audio", 0.0, None),
    ],
)
def test_mid_stream_frame_property_change_falls_back_to_one_process(
    sources, tmp_path, frame_jobs, name, start, duration
):
    duration, stream_count = _probe(sources[name], start, duration)
    legacy = _read_frames(_transcode(sources[name], tmp_path, start=start, duration=duration))
    for workers in (1, 2, 3, 8):
        frame_jobs.clear()
        frames = _job(
            sources[name], start=start, duration=duration, workers=workers, stream_count=stream_count
        ).result()
        assert torch.equal(frames, legacy), workers
        assert [job.workers for job in frame_jobs] == ([1] if workers == 1 else [workers, 1])
        _assert_reaped(*frame_jobs)


def test_mid_stream_display_matrix_change_matches_prepared_video(tmp_path, encoders, frame_jobs):
    plain, rotated, source = tmp_path / "plain.h264", tmp_path / "rotated.h264", str(tmp_path / "rotated.mp4")
    _ffmpeg(*_testsrc("320x256", 30), "-t", 2.2, *_x264(), "-f", "h264", plain)
    try:
        bsf = "h264_metadata=display_orientation=insert:rotate=90"
        _ffmpeg(*_testsrc("320x256", 30), "-t", 2, *_x264(), "-bsf:v", bsf, "-f", "h264", rotated)
    except subprocess.CalledProcessError:
        pytest.skip("ffmpeg cannot insert an H.264 display orientation SEI")
    rotated.write_bytes(plain.read_bytes() + rotated.read_bytes())
    _ffmpeg("-framerate", 30, "-i", rotated, "-c:v", "copy", source)
    duration, stream_count = _probe(source)
    legacy = _read_frames(_transcode(source, tmp_path, duration=duration))
    for workers in (1, 2, 4):
        frames = _job(source, duration=duration, workers=workers, stream_count=stream_count).result()
        assert torch.equal(frames, legacy), workers
    _assert_reaped(*frame_jobs)


def test_prepare_reference_videos_splits_workers_and_matches_transcode(sources, tmp_path):
    videos = [sources["s30_audio"], sources["s25_short"], sources["s60_rotate"]]
    options = {"target_frame_count": FRAMES, "start_time_seconds": 0.37}
    legacy = reference_video.prepare_reference_videos(videos, workdir=str(tmp_path / "legacy"), **options)
    direct = reference_video.prepare_reference_videos(
        videos, workdir=str(tmp_path / "direct"), frame_workers=8, **options
    )
    assert not (tmp_path / "direct").exists()
    for old, new in zip(legacy, direct, strict=True):
        job = new.pop("frames")
        assert job.workers == 2 and new == {**old, "prepared_path": None}
        assert torch.equal(job.result(), _read_frames(old["prepared_path"]))
        with pytest.raises(RuntimeError, match="already consumed"):
            job.result()
    with pytest.raises(ValueError, match="no prepared video"):
        reference_video.serialize_prepared_reference_videos(direct, str(tmp_path))
    assert _ffmpeg_children() == []


def test_prepare_reference_videos_validates_every_video_before_decoding(monkeypatch, tmp_path, frame_jobs):
    durations = {"a.mp4": 10.0, "b.mp4": 5.02}
    monkeypatch.setattr(reference_video, "_probe_video", lambda path: {**META, "duration": durations[path]})
    monkeypatch.setattr(reference_video, "direct_reference_decode_supported", lambda: True)
    with pytest.raises(OmniClientError, match="15 seconds"):
        reference_video.prepare_reference_videos(
            ["a.mp4", "b.mp4"], target_frame_count=FRAMES, workdir=str(tmp_path), frame_workers=8
        )
    assert frame_jobs == []


def test_unsupported_ffmpeg_keeps_the_prepared_video_path(monkeypatch, tmp_path, frame_jobs):
    transcodes = []

    def transcode(source, **kwargs):
        transcodes.append(source)
        return kwargs["workdir"]

    monkeypatch.setattr(reference_video, "_probe_video", lambda path: META)
    monkeypatch.setattr(reference_video, "direct_reference_decode_supported", lambda: False)
    monkeypatch.setattr(reference_video, "_transcode_reference_video", transcode)
    prepared = reference_video.prepare_reference_videos(
        ["a.mp4", "b.mp4"], target_frame_count=FRAMES, workdir=str(tmp_path), frame_workers=8
    )
    assert transcodes == ["a.mp4", "b.mp4"] and frame_jobs == []
    assert [item["prepared_path"] for item in prepared] == [str(tmp_path / f"video_{index}") for index in range(2)]


@pytest.mark.parametrize(
    ("banner", "supported"),
    [
        ("ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers", True),
        ("ffmpeg version n7.0.2 Copyright (c) 2000-2024 the FFmpeg developers", True),
        ("ffmpeg version 7.1.3-0+deb13u1 Copyright (c) 2000-2025 the FFmpeg developers", False),
        ("ffmpeg version 5.0.3 Copyright (c) 2000-2022 the FFmpeg developers", False),
        ("ffmpeg version N-113432-g0123456789 Copyright (c) 2000-2024 the FFmpeg developers", False),
        (None, False),
    ],
)
def test_direct_decode_is_limited_to_checked_ffmpeg_releases(monkeypatch, banner, supported):
    def run(command, **kwargs):
        assert command == ["ffmpeg", "-version"]
        if banner is None:
            raise FileNotFoundError("ffmpeg")
        return subprocess.CompletedProcess(command, 0, stdout=f"{banner}\nbuilt with gcc\n", stderr="")

    monkeypatch.setattr(reference_video.subprocess, "run", run)
    reference_video.direct_reference_decode_supported.cache_clear()
    try:
        assert reference_video.direct_reference_decode_supported() is supported
    finally:
        reference_video.direct_reference_decode_supported.cache_clear()


@pytest.mark.parametrize("workers", [1, 3])
def test_failed_decode_raises_called_process_error(tmp_path, capfd, encoders, workers):
    source = tmp_path / "bad.mp4"
    source.write_bytes(b"not a video\n" * 512)
    job = _job(str(source), duration=4.0, workers=workers)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        job.result()
    _assert_reaped(job)
    printed = [line for line in capfd.readouterr().err.splitlines() if str(source) in line]
    assert len(printed) == 1 and "[error]" not in printed[0]
    assert failure.value.stderr is None if workers == 1 else str(source).encode() in failure.value.stderr


def test_split_decode_prints_no_ffmpeg_warnings(sources, tmp_path, capfd):
    source, warning = sources["full_range"], "deprecated pixel format used"
    probe = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "warning", "-i", source, "-vf", f"scale={WIDTH}:{HEIGHT}", "-frames:v", "1"]
        + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
        timeout=60,
    )
    assert warning.encode() in probe.stderr
    duration, _ = _probe(source)
    legacy = _read_frames(_transcode(source, tmp_path, duration=duration))
    capfd.readouterr()
    job = _job(source, duration=duration, stream_count=1)
    assert torch.equal(job.result(), legacy)
    assert warning not in capfd.readouterr().err
    _assert_reaped(job)


def _patch_command(monkeypatch, edit) -> None:
    command = reference_video.ReferenceVideoFrames._command

    def patched(self, worker):
        argv = command(self, worker)
        edit(self, worker, argv)
        return argv

    monkeypatch.setattr(reference_video.ReferenceVideoFrames, "_command", patched)


def test_frame_protocol_error_is_the_cause_of_the_ffmpeg_failure(monkeypatch, sources, frame_jobs):
    def drop_frame_limit(self, worker, argv):
        index = argv.index("-frames:v")
        del argv[index : index + 2]

    _patch_command(monkeypatch, drop_frame_limit)
    for workers in (1, 3):
        frame_jobs.clear()
        with pytest.raises((subprocess.CalledProcessError, RuntimeError)) as failure:
            _job(sources["s30_audio"], frame_count=5, workers=workers).result()
        error = failure.value
        if isinstance(error, subprocess.CalledProcessError):
            error = error.__cause__
        assert isinstance(error, RuntimeError) and not isinstance(error, OmniClientError)
        assert "extra frames" in str(error)
        assert [job.workers for job in frame_jobs] == ([1] if workers == 1 else [3, 1])
        _assert_reaped(*frame_jobs)


def test_split_frame_counts_that_do_not_interleave_fall_back_to_one_process(monkeypatch, sources, tmp_path, frame_jobs):
    def stop_second_process_early(self, worker, argv):
        if self.workers > 1 and worker == 1:
            index = argv.index("-frames:v") + 1
            argv[index] = str(int(argv[index]) - 2)

    duration, _ = _probe(sources["s30_audio"])
    legacy = _read_frames(_transcode(sources["s30_audio"], tmp_path, duration=duration))
    _patch_command(monkeypatch, stop_second_process_early)
    assert torch.equal(_job(sources["s30_audio"], duration=duration).result(), legacy)
    assert [job.workers for job in frame_jobs] == [3, 1]
    _assert_reaped(*frame_jobs)


def test_single_process_command_is_the_transcode_with_raw_output(monkeypatch, tmp_path):
    transcodes, direct = [], []
    monkeypatch.setattr(reference_video.subprocess, "run", lambda command, **kwargs: transcodes.append(command))

    def popen(command, **kwargs):
        direct.append((command, kwargs["stderr"]))
        raise OSError("not started")

    monkeypatch.setattr(reference_video.subprocess, "Popen", popen)
    _transcode("in.mp4", tmp_path, start=0.37, duration=4.52)
    for workers in (1, 3):
        with pytest.raises(OSError, match="not started"):
            _job("in.mp4", start=0.37, duration=4.52, workers=workers)
    expected = [arg for arg in transcodes[0][: transcodes[0].index("-metadata:s:v:0")] if arg != "-y"]
    assert direct[0] == ([*expected, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], None)
    split, split_stderr = direct[1]
    assert " ".join(split[:7]) == "ffmpeg -nostdin -loglevel level+verbose -filter_threads 1 -ss"
    assert split_stderr is subprocess.PIPE


def test_start_failure_and_cancel_stop_every_process(monkeypatch, sources):
    job = _job(sources["s30_audio"])
    job.cancel()
    _assert_reaped(job)
    real_popen, started = subprocess.Popen, list[subprocess.Popen]()

    def popen(*args, **kwargs):
        if len(started) == 2:
            raise OSError("cannot start ffmpeg")
        started.append(real_popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(reference_video.subprocess, "Popen", popen)
    with pytest.raises(OSError, match="cannot start ffmpeg"):
        _job(sources["s30_audio"], workers=8)
    assert len(started) == 2
    assert all(process.returncode is not None and process.stdout and process.stdout.closed for process in started)
    assert _ffmpeg_children() == []


def test_reference_frame_workers_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    for cpus, expected in ((1, 1), (15, 1), (16, 2), (31, 3), (32, 4), (120, 4)):
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid, cpus=cpus: set(range(cpus)), raising=False)
        assert reference_video.reference_frame_workers() == expected, cpus
    for raw, expected in (("0", 0), ("1", 1), (" 12 ", 12)):
        monkeypatch.setenv(ENV, raw)
        assert reference_video.reference_frame_workers() == expected
    for raw in ("-1", "eight"):
        monkeypatch.setenv(ENV, raw)
        with pytest.raises(ValueError, match=ENV):
            reference_video.reference_frame_workers()


@pytest.fixture(scope="module")
def reference_mp3(tmp_path_factory, encoders) -> dict[str, str]:
    if "libmp3lame" not in encoders:
        pytest.skip("ffmpeg with libmp3lame is required")
    root = tmp_path_factory.mktemp("minimax_h3_reference_audio")
    paths = {name: str(root / f"{name}.mp3") for name in ("valid", "too_short", "corrupt")}
    for name, seconds in (("valid", 3), ("too_short", 1)):
        _ffmpeg(*SINE, "-t", seconds, "-c:a", "libmp3lame", paths[name])
    (root / "corrupt.mp3").write_bytes(b"not audio\n" * 64)
    return paths


def _sig(value) -> tuple:
    array = np.ascontiguousarray(value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
    return array.shape, array.dtype.str, hashlib.sha256(array.tobytes()).hexdigest()


def _digest(prepared) -> dict:
    media = prepared.media
    return {
        "shape": (media.height, media.width, media.num_frames, media.latent_t, media.audio_t),
        "images": [_sig(image) for image in media.images],
        "videos": [_sig(video) for video in media.videos],
        "video_audios": [item and (_sig(item[0]), item[1]) for item in media.video_audios],
        "audios": [(_sig(waveform), rate) for waveform, rate in media.audios],
        "qwen_videos": [(_sig(frames), meta) for frames, meta in prepared.qwen_videos],
        "pil_images": [(image.mode, _sig(image)) for image in prepared.images],
        "video_timestamps": prepared.video_timestamps,
        "condition_labels": prepared.condition_labels,
    }


def _prepare(videos, audio, *, images=True, start=None, audio_mode="native"):
    data = {"video": videos, "audio": audio}
    if images:
        rng = np.random.default_rng(0)
        data["image"] = [
            Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for w, h in ((300, 280), (520, 260))
        ]
    extra_args = {"task": "ref2va", "duration": 4.0, "start_time_seconds": start, "audio_mode": audio_mode}
    return processing.prepare_encoder_inputs(
        {"prompt": "reference", "multi_modal_data": data},
        OmniDiffusionSamplingParams(height=256, width=384, extra_args=extra_args),
    )


def _spy(monkeypatch, owner, name) -> list:
    calls, original = [], getattr(owner, name)

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, spy)
    return calls


@pytest.mark.parametrize(
    ("videos", "video_labels"),
    [
        (["s30_audio"], [("audio", 1), ("video", 1)]),
        (["s30_audio", "s25_short", "s60_rotate"], [("audio", 1), ("video", 1), ("video", 2), ("video", 3)]),
    ],
    ids=["one_video", "three_videos"],
)
def test_encoder_inputs_are_identical_to_the_legacy_path(
    monkeypatch, sources, reference_mp3, legacy_decoder, videos, video_labels
):
    spies = {
        "legacy_loads": _spy(monkeypatch, processing, "load_video_frames"),
        "prefetches": _spy(monkeypatch, processing._ReferenceAudioPrefetch, "start"),
        "resizes": _spy(monkeypatch, processing, "_prepare_encoder_images"),
    }
    digests = {}
    for raw in ("0", "1", "3", "8", None):
        if raw is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, raw)
        for calls in spies.values():
            calls.clear()
        digests[raw] = _digest(_prepare([sources[video] for video in videos], reference_mp3["valid"], start=0.37))
        assert len(spies["legacy_loads"]) == (len(videos) if raw == "0" else 0), raw
        assert [args[1] for args, _ in spies["prefetches"]] == ([] if raw == "0" else [[reference_mp3["valid"]]]), raw
        assert [kwargs["parallel"] for _, kwargs in spies["resizes"]] == [raw != "0"], raw
        assert _ffmpeg_children() == []
    assert digests["0"]["video_audios"][0] is not None
    assert digests["0"]["condition_labels"] == [("image", 1), ("image", 2), *video_labels, ("audio", 2)]
    assert all(digest == digests["0"] for digest in digests.values())


@pytest.mark.parametrize("workers", ["0", "8"])
@pytest.mark.parametrize(
    ("videos", "audios", "options", "error", "match", "work"),
    [
        (["s30_audio"], ["valid"], {"audio_mode": "lock_source"}, None, None, (0, 1)),
        (["s25_short"], ["too_short"], {"images": False}, OmniClientError, r"audio 0 duration .*got 1\.0", (1, 0)),
        ([], ["valid"] * 4, {}, OmniClientError, "at most 3 standalone audio references", (0, 0)),
        ([], ["corrupt"], {}, subprocess.CalledProcessError, "ffprobe", (1, 0)),
    ],
    ids=["lock_source", "invalid", "rejected", "corrupt"],
)
def test_reference_audio_is_probed_and_decoded_only_where_the_serial_path_does(
    monkeypatch, sources, reference_mp3, workers, videos, audios, options, error, match, work
):
    monkeypatch.setenv(ENV, workers)
    probes, loads = _spy(monkeypatch, reference_video, "_probe_audio"), _spy(monkeypatch, processing, "load_audio_file")
    with pytest.raises(error, match=match) if error else contextlib.nullcontext():
        _prepare([sources[video] for video in videos], [reference_mp3[audio] for audio in audios], **options)
    assert (len(probes), len(loads)) == work
    assert _ffmpeg_children() == []


def test_unconsumed_frame_jobs_are_cancelled_on_error(monkeypatch, sources, reference_mp3, frame_jobs):
    def fail(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setenv(ENV, "2")
    monkeypatch.setattr(processing, "sample_reference_video_frames", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        _prepare([sources["s30_audio"], sources["s60_rotate"]], reference_mp3["valid"], images=False)
    assert [job.workers for job in frame_jobs] == [1, 1]
    _assert_reaped(*frame_jobs)

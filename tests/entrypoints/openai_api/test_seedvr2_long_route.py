# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Window, overlap and job-lifecycle logic of the SeedVR2 long-video route.

The model is mocked out, so every case here is pure CPU bookkeeping: which
source frame reaches which output frame, how the seam between windows is
blended, and how a job settles.
"""

from __future__ import annotations

import json
import os
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi import HTTPException

from vllm_omni.entrypoints.openai.video.seedvr2_long import (
    FPS,
    OVERLAP,
    JobError,
    _background,
    _job_dir,
    _run,
    _status,
    _sweep_expired_jobs,
    _window,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
SIZE = 32
# Small windows keep several seams inside short clips.
WINDOW = 13


@pytest.fixture(autouse=True)
def _small_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW", str(WINDOW))


# Flat frames survive 4:2:0 and crf18, but not exactly; keep levels far apart.
TOLERANCE = 4


def _source_values(count: int) -> list[int]:
    """Distinct grey levels, spread across the range so a frame names its index."""
    step = max(1, min(8, 235 // max(count - 1, 1)))
    return [20 + step * index for index in range(count)]


def _write_source(path: Path, values: list[int]) -> None:
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = SIZE, SIZE, "yuv420p"
        stream.options = {"crf": "0", "preset": "veryfast"}
        for index, value in enumerate(values):
            frame = av.VideoFrame.from_ndarray(np.full((SIZE, SIZE, 3), value, np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _output_levels(path: Path) -> list[float]:
    with av.open(str(path)) as container:
        return [float(frame.to_ndarray(format="rgb24").mean()) for frame in container.decode(video=0)]


def _job(tmp_path: Path, frames: int) -> Path:
    job = tmp_path / "job"
    job.mkdir()
    _write_source(job / "input.mp4", _source_values(frames))
    _status(job, "queued", 0)
    return job


def _echo_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return each window unchanged, so output levels still name their source."""

    def restore(frames, width, height, seed, method, port, authorization):
        return [frame.to_ndarray(format="rgb24") for frame in frames]

    monkeypatch.setattr("vllm_omni.entrypoints.openai.video.seedvr2_long._restore", restore)


def _constant_restore(monkeypatch: pytest.MonkeyPatch, levels: list[int]) -> None:
    """Give each window its own flat level, which makes the seam blend visible."""
    calls = iter(levels)

    def restore(frames, width, height, seed, method, port, authorization):
        level = next(calls)
        return [np.full((SIZE, SIZE, 3), level, np.uint8) for _ in frames]

    monkeypatch.setattr("vllm_omni.entrypoints.openai.video.seedvr2_long._restore", restore)


def _restore_once_then_fail(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def restore(frames, width, height, seed, method, port, authorization):
        raise error

    monkeypatch.setattr("vllm_omni.entrypoints.openai.video.seedvr2_long._restore", restore)


def run(job: Path, target: int, loop: bool = False) -> None:
    _run(job, SIZE, SIZE, target, loop, 7723, "lab", 0, "")


# Below, at and above the window, plus counts that do not land on the stride.
@pytest.mark.parametrize("target", [1, OVERLAP, WINDOW - 1, WINDOW, WINDOW + 1, 20, 27, 3 * WINDOW])
def test_every_requested_frame_is_written(target: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(tmp_path, target)
    _echo_restore(monkeypatch)
    run(job, target)
    assert json.loads((job / "result.json").read_text())["frames"] == target
    assert len(_output_levels(job / "output.mp4")) == target


@pytest.mark.parametrize("target", [WINDOW, 20, 27])
def test_output_frames_keep_their_source_order(target: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(tmp_path, target)
    _echo_restore(monkeypatch)
    run(job, target)
    # An echoing model blends each overlap frame with itself, so levels survive.
    for index, (actual, expected) in enumerate(zip(_output_levels(job / "output.mp4"), _source_values(target))):
        assert abs(actual - expected) <= TOLERANCE, f"frame {index} came from the wrong source"


def test_overlap_blends_linearly_across_the_seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = 2 * WINDOW - OVERLAP
    job = _job(tmp_path, target)
    _constant_restore(monkeypatch, [40, 160])
    run(job, target)
    levels = _output_levels(job / "output.mp4")

    # Two windows cover the clip: the first contributes its frames before the
    # seam alone, the seam blends OVERLAP frames, and the second the rest.
    seam = [40 + (160 - 40) * (offset + 1) / (OVERLAP + 1) for offset in range(OVERLAP)]
    expected = [40] * (WINDOW - OVERLAP) + seam + [160] * (WINDOW - OVERLAP)
    assert len(expected) == target
    for index, (actual, want) in enumerate(zip(levels, expected)):
        assert abs(actual - want) <= TOLERANCE, f"frame {index} blended to {actual}, expected {want}"


def test_loop_input_repeats_a_short_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(tmp_path, WINDOW)
    _echo_restore(monkeypatch)
    run(job, 20, loop=True)
    levels = _output_levels(job / "output.mp4")
    # The source plays once, then frames past its end wrap back to the start.
    source = _source_values(WINDOW)
    expected = source + source[: 20 - WINDOW]
    assert len(levels) == 20
    for index, (actual, want) in enumerate(zip(levels, expected)):
        assert abs(actual - want) <= TOLERANCE, f"frame {index} is {actual}, expected the wrap to give {want}"
    assert json.loads((job / "result.json").read_text())["frames"] == 20


def test_short_source_without_loop_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(tmp_path, WINDOW)
    _echo_restore(monkeypatch)
    with pytest.raises(JobError, match="fewer frames than requested"):
        run(job, 20)


def test_status_moves_from_queued_to_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(tmp_path, WINDOW)
    assert json.loads((job / "status.json").read_text())["status"] == "queued"
    _echo_restore(monkeypatch)
    _background(job, SIZE, SIZE, WINDOW, False, 7723, "lab", 0, "")
    record = json.loads((job / "status.json").read_text())
    assert (record["status"], record["frames"], record["error"]) == ("completed", WINDOW, "")


def test_failure_is_reported_without_internal_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(tmp_path, WINDOW)
    secret = "http://127.0.0.1:8098/v1/videos/sync exploded at /tmp//job/input.mp4"
    _restore_once_then_fail(monkeypatch, RuntimeError(secret))
    _background(job, SIZE, SIZE, WINDOW, False, 7723, "lab", 0, "")
    record = json.loads((job / "status.json").read_text())
    assert record["status"] == "failed"
    assert record["error"] == "SeedVR2 long-video restoration failed"
    assert "127.0.0.1" not in record["error"] and secret not in record["error"]


def test_client_safe_failures_keep_their_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(tmp_path, WINDOW)
    _echo_restore(monkeypatch)
    _background(job, SIZE, SIZE, 20, False, 7723, "lab", 0, "")
    record = json.loads((job / "status.json").read_text())
    assert record["status"] == "failed"
    assert "set loop_input=true" in record["error"]


def test_cancel_settles_the_job_at_a_window_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(tmp_path, 3 * WINDOW)
    _echo_restore(monkeypatch)
    (job / "cancel").touch()
    _background(job, SIZE, SIZE, 3 * WINDOW, False, 7723, "lab", 0, "")
    record = json.loads((job / "status.json").read_text())
    assert record["status"] == "cancelled"
    assert not (job / "result.json").exists()


def test_intermediate_video_is_removed_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(tmp_path, WINDOW)
    _echo_restore(monkeypatch)
    run(job, WINDOW)
    assert (job / "output.mp4").exists()
    assert not (job / "video.mp4").exists()


def test_windows_carry_source_pixels_and_come_back_at_the_output_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path, WINDOW)
    shapes: list[tuple[int, int]] = []

    def restore(frames, width, height, seed, method, port, authorization):
        shapes.extend((frame.width, frame.height) for frame in frames)
        return [np.full((height, width, 3), 128, np.uint8) for _ in frames]

    monkeypatch.setattr("vllm_omni.entrypoints.openai.video.seedvr2_long._restore", restore)
    _run(job, 2 * SIZE, 2 * SIZE, WINDOW, False, 7723, "lab", 0, "")
    # The model upsamples on the device, so the host never ships output-size input.
    assert set(shapes) == {(SIZE, SIZE)}
    with av.open(str(job / "output.mp4")) as container:
        assert (container.streams.video[0].width, container.streams.video[0].height) == (2 * SIZE, 2 * SIZE)


def test_window_is_the_longest_4n_plus_1_clip_the_budget_admits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW", "1000")
    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS", str(100 * 100))
    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS", str(100 * 100 * 40))
    assert _window(100, 100) == 37
    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_MAX_FRAMES", "30")
    assert _window(100, 100) == 29
    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW", "12")
    assert _window(100, 100) == 9
    # A frame over the per-frame budget fits no window at all.
    assert _window(100, 101) == 0


@pytest.mark.parametrize("job_id", ["", "short", "../../etc/passwd", "g" * 32, "A" * 32, "0" * 31, "0" * 33])
def test_job_ids_that_are_not_plain_hex_are_rejected(job_id: str) -> None:
    with pytest.raises(HTTPException) as error:
        _job_dir(job_id)
    assert error.value.status_code == 404


def test_valid_job_id_resolves_under_the_jobs_root() -> None:
    job_id = "0123456789abcdef" * 2
    assert _job_dir(job_id).name == job_id


def _settled_job(root: Path, job_id: str, status: str, pid: int, age: float) -> Path:
    job = root / "seedvr2-long" / job_id
    job.mkdir(parents=True)
    record = {"status": status, "frames": 0, "error": "", "pid": pid}
    (job / "status.json").write_text(json.dumps(record) + "\n")
    os.utime(job / "status.json", (time.time() - age, time.time() - age))
    return job


def test_the_sweep_drops_only_jobs_that_are_settled_and_expired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS", "60")
    old = _settled_job(tmp_path, "a" * 32, "completed", os.getpid(), 600)
    fresh = _settled_job(tmp_path, "b" * 32, "completed", os.getpid(), 1)
    running = _settled_job(tmp_path, "c" * 32, "running", os.getpid(), 600)
    # A restart leaves a "running" job nobody owns; the sweep has to reclaim it.
    abandoned = _settled_job(tmp_path, "d" * 32, "running", os.getpid() + 1, 600)
    _sweep_expired_jobs()
    assert not old.exists()
    assert not abandoned.exists()
    assert fresh.exists()
    assert running.exists()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
from concurrent.futures import CancelledError

import numpy as np
import pytest

from vllm_omni.diffusion.utils.video_encoding import (
    VideoEncodingScheduler,
    calculate_encoding_allocation,
    encode_video_segment,
    remux_video_segments,
    run_ordered_encoding_jobs,
    write_imageio_video,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("jobs", "cpus", "workers", "threads"),
    [
        (0, 14, 0, 0),
        (1, 1, 1, 1),
        (1, 7, 1, 4),
        (7, 14, 7, 2),
        (14, 14, 7, 2),
        (7, 5, 2, 2),
    ],
)
def test_encoding_allocation(jobs, cpus, workers, threads):
    allocation = calculate_encoding_allocation(jobs, cpus)
    assert (allocation.workers, allocation.encoder_threads) == (workers, threads)


def test_ordered_jobs_keep_input_order_after_out_of_order_completion():
    release_first = threading.Event()
    second_finished = threading.Event()

    def encode(job: int, _threads: int) -> int:
        if job == 0:
            assert release_first.wait(2)
        else:
            second_finished.set()
            release_first.set()
        return job

    results, allocation = run_ordered_encoding_jobs([0, 1], encode, cpu_count=4)
    assert second_finished.is_set()
    assert results == [0, 1]
    assert allocation.workers == 2


def test_scheduler_bounds_tokens_and_round_robins_requests():
    scheduler = VideoEncodingScheduler(4)
    release = threading.Event()
    both_started = threading.Barrier(3)
    active = 0
    maximum = 0
    order = []
    lock = threading.Lock()

    def blocked(name: str):
        def run(cancel_event):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                order.append(name)
            both_started.wait(2)
            assert release.wait(2) or cancel_event.is_set()
            with lock:
                active -= 1
            return name

        return run

    try:
        futures = [
            scheduler.submit("one", blocked("one-1"), tokens=2),
            scheduler.submit("one", lambda _: "one-2", tokens=2),
            scheduler.submit("two", blocked("two-1"), tokens=2),
        ]
        both_started.wait(2)
        assert scheduler.reserved_tokens == 4
        assert order == ["one-1", "two-1"]
        release.set()
        assert [future.result(2).value for future in futures] == ["one-1", "one-2", "two-1"]
        assert maximum == 2
    finally:
        release.set()
        scheduler.shutdown()


def test_all_cpu_job_is_not_starved_by_later_small_jobs():
    scheduler = VideoEncodingScheduler(4)
    first_started = threading.Event()
    release_first = threading.Event()
    legacy_started = threading.Event()
    later_started = threading.Event()

    def first(_):
        first_started.set()
        assert release_first.wait(2)

    try:
        first_future = scheduler.submit("first", first, tokens=2)
        assert first_started.wait(2)
        legacy = scheduler.submit("legacy", lambda _: legacy_started.set(), tokens=4)
        later = scheduler.submit("later", lambda _: later_started.set(), tokens=2)
        assert not legacy_started.is_set()
        assert not later_started.is_set()
        release_first.set()
        first_future.result(2)
        legacy.result(2)
        assert legacy_started.is_set()
        later.result(2)
        assert later_started.is_set()
    finally:
        release_first.set()
        scheduler.shutdown()


def test_scheduler_cancels_before_dispatch_and_during_work():
    scheduler = VideoEncodingScheduler(2)
    active_started = threading.Event()
    active_stopped = threading.Event()

    def active(cancel_event):
        active_started.set()
        assert cancel_event.wait(2)
        active_stopped.set()

    try:
        running = scheduler.submit("running", active, tokens=2)
        assert active_started.wait(2)
        queued = scheduler.submit("queued", lambda _: pytest.fail("queued job ran"), tokens=2)
        scheduler.cancel_request("queued")
        with pytest.raises(CancelledError):
            queued.result()
        scheduler.cancel_request("running")
        scheduler.wait_request("running")
        assert active_stopped.is_set()
        running.result(2)
        assert scheduler.reserved_tokens == 0
    finally:
        scheduler.shutdown()


def test_worker_exception_cancels_siblings_and_shutdown_drains():
    scheduler = VideoEncodingScheduler(2)
    failed = threading.Event()
    sibling_stopped = threading.Event()

    def fail(_):
        failed.set()
        raise RuntimeError("encode failed")

    def sibling(cancel_event):
        assert cancel_event.wait(2)
        sibling_stopped.set()

    failure = scheduler.submit("request", fail, tokens=1)
    sibling_future = scheduler.submit("request", sibling, tokens=1)
    assert failed.wait(2)
    with pytest.raises(RuntimeError, match="encode failed"):
        failure.result(2)
    scheduler.shutdown()
    assert sibling_stopped.is_set() or sibling_future.cancelled()
    assert scheduler.reserved_tokens == 0


def test_streaming_writer_preserves_uint8_and_truncates_floats(monkeypatch, tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    appended = []
    writer_kwargs = {}

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def append_data(self, frame):
            appended.append(frame.copy())

    def get_writer(_path, **kwargs):
        writer_kwargs.update(kwargs)
        return Writer()

    monkeypatch.setattr(imageio, "get_writer", get_writer)
    uint8_frame = np.array([[[0, 127, 255]]], dtype=np.uint8)
    float_frame = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    write_imageio_video(
        [uint8_frame, float_frame],
        tmp_path / "video.mp4",
        fps=24,
        encoder_threads=3,
    )

    assert np.array_equal(appended[0], uint8_frame)
    assert appended[1].tolist() == [[[0, 127, 255]]]
    assert writer_kwargs["quality"] == 5.0
    assert writer_kwargs["macro_block_size"] == 16
    assert writer_kwargs["output_params"] == ["-threads", "3"]


def test_segment_signed_normalization_uses_http_rounding(monkeypatch):
    pytest.importorskip("av")
    from vllm_omni.diffusion.utils import video_encoding

    captured = []
    original = video_encoding._build_planar_frame

    def capture(frame, common_dtype, scratch):
        captured.append(frame.copy())
        return original(frame, common_dtype, scratch)

    monkeypatch.setattr(video_encoding, "_build_planar_frame", capture)
    row = [[-1.0, 0.0, 1.0], [-0.5, 0.5, 0.25]]
    source = np.array([[row, row]], dtype=np.float32)
    segment = encode_video_segment(
        source,
        fps=12,
        encoder_threads=1,
        video_codec_options={"preset": "ultrafast"},
        cancel_event=threading.Event(),
        normalization="signed",
    )
    try:
        assert captured[0].tolist() == [
            [[0, 128, 255], [64, 191, 159]],
            [[0, 128, 255], [64, 191, 159]],
        ]
    finally:
        segment.close()


@pytest.mark.parametrize("fps", [12.0, 30000 / 1001])
@pytest.mark.parametrize("preset", ["ultrafast", "medium"])
def test_segment_remux_matches_decoded_camera_concatenation(fps, preset):
    av = pytest.importorskip("av")
    cancel_event = threading.Event()
    cameras = []
    for camera in range(2):
        frames = np.zeros((5, 24, 32, 3), dtype=np.uint8)
        frames[..., camera] = np.arange(5, dtype=np.uint8)[:, None, None] * 30 + 20
        cameras.append(frames)
    segments = [
        encode_video_segment(
            list(frames),
            fps=fps,
            encoder_threads=1,
            video_codec_options={"preset": preset, "crf": "18"},
            cancel_event=cancel_event,
        )
        for frames in cameras
    ]
    try:
        expected = []
        for segment in segments:
            segment.buffer.seek(0)
            with av.open(segment.buffer, "r", format="mp4") as container:
                expected.extend(frame.to_ndarray(format="rgb24") for frame in container.decode(video=0))
        payload = remux_video_segments(segments, fps=fps, cancel_event=cancel_event)
        with av.open(__import__("io").BytesIO(payload), "r", format="mp4") as container:
            decoded = list(container.decode(video=0))
            actual = [frame.to_ndarray(format="rgb24") for frame in decoded]
        assert len(actual) == 10
        assert all(np.array_equal(left, right) for left, right in zip(actual, expected, strict=True))
        timestamps = [float(frame.pts * frame.time_base) for frame in decoded]
        assert timestamps == sorted(timestamps)
        assert timestamps[0] == 0
        assert timestamps[-1] + 1 / fps == pytest.approx(10 / fps, abs=1e-6)
    finally:
        for segment in segments:
            segment.close()

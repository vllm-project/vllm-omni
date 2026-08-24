# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for planar video frame conversion helpers."""

import threading
from collections import deque

import av
import numpy as np
import pytest

from vllm_omni.entrypoints.openai import video_api_utils, video_frame_conversion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_serial_and_parallel_planar_frames_have_identical_pixels(dtype):
    if dtype == np.float32:
        planar = np.linspace(-0.25, 1.25, num=3 * 9 * 2 * 2, dtype=np.float32).reshape(3, 9, 2, 2)
        frames = list(planar.transpose(1, 2, 3, 0))
    else:
        frames = [np.full((2, 2, 3), index, dtype=dtype) for index in range(9)]

    serial = list(video_frame_conversion._iter_planar_video_frames(frames, np.dtype(dtype), worker_count=1))
    parallel = list(video_frame_conversion._iter_planar_video_frames(frames, np.dtype(dtype), worker_count=8))

    serial_pixels = np.stack([frame.to_ndarray(format="rgb24") for frame in serial])
    parallel_pixels = np.stack([frame.to_ndarray(format="rgb24") for frame in parallel])
    np.testing.assert_array_equal(serial_pixels, parallel_pixels)
    if dtype == np.float32:
        expected_pixels = np.rint(np.clip(np.stack(frames), 0.0, 1.0) * 255.0).astype(np.uint8)
        np.testing.assert_array_equal(serial_pixels, expected_pixels)


def test_planar_worker_pool_preserves_fifo_order_and_bounds_pending(monkeypatch):
    class TrackingDeque(deque):
        max_size = 0

        def append(self, value):
            super().append(value)
            type(self).max_size = max(type(self).max_size, len(self))

    def fake_build(frame, common_dtype):
        del common_dtype
        return int(frame[0, 0, 0])

    monkeypatch.setattr(video_frame_conversion, "deque", TrackingDeque)
    monkeypatch.setattr(video_frame_conversion, "_build_planar_video_frame", fake_build)
    frames = [np.full((1, 1, 3), index, dtype=np.uint8) for index in range(13)]

    output = list(video_frame_conversion._iter_planar_video_frames(frames, np.dtype(np.uint8), worker_count=3))

    assert output == list(range(13))
    assert TrackingDeque.max_size <= 6


def test_planar_worker_exception_propagates_and_leaves_no_pool_threads(monkeypatch):
    def failing_build(frame, common_dtype):
        del common_dtype
        marker = int(frame[0, 0, 0])
        if marker == 2:
            raise RuntimeError("frame conversion failed")
        return marker

    monkeypatch.setattr(video_frame_conversion, "_build_planar_video_frame", failing_build)
    frames = [np.full((1, 1, 3), index, dtype=np.uint8) for index in range(6)]

    with pytest.raises(RuntimeError, match="frame conversion failed"):
        list(video_frame_conversion._iter_planar_video_frames(frames, np.dtype(np.uint8), worker_count=3))

    assert not any(thread.name.startswith("video-planar") for thread in threading.enumerate())


def test_workers_one_and_single_frame_do_not_create_a_pool(monkeypatch):
    def fail_pool(*args, **kwargs):
        raise AssertionError("serial conversion must not create a thread pool")

    monkeypatch.setattr(video_frame_conversion, "ThreadPoolExecutor", fail_pool)
    frames = [
        np.full((1, 2, 3), 0.25, dtype=np.float32),
        np.full((1, 2, 3), 0.75, dtype=np.float32),
    ]

    output = list(video_frame_conversion._iter_planar_video_frames(frames, np.dtype(np.float32), worker_count=1))
    assert len(output) == len(frames)
    for actual, source in zip(output, frames):
        expected = np.rint(np.clip(source, 0.0, 1.0) * 255.0).astype(np.uint8)
        np.testing.assert_array_equal(actual.to_ndarray(format="rgb24"), expected)

    single_frame_output = list(
        video_frame_conversion._iter_planar_video_frames(frames[:1], np.dtype(np.float32), worker_count=8)
    )
    assert len(single_frame_output) == 1


def test_planar_scratch_is_reused_per_thread_and_not_shared():
    barrier = threading.Barrier(2)

    def observe_scratch():
        first = video_frame_conversion._planar_channel_scratch(3, 4, np.dtype(np.float32))
        barrier.wait(timeout=2)
        second = video_frame_conversion._planar_channel_scratch(3, 4, np.dtype(np.float32))
        return first, second

    with video_frame_conversion.ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(lambda _: observe_scratch(), range(2)))

    assert first[0] is first[1]
    assert second[0] is second[1]
    assert first[0] is not second[0]


def test_planar_generator_close_cancels_pending_and_shuts_down(monkeypatch):
    cancelled = []
    shutdown_calls = []

    class RecordingFuture(video_frame_conversion.Future):
        def __init__(self, marker):
            super().__init__()
            self.marker = marker

        def cancel(self):
            cancelled.append(self.marker)
            return super().cancel()

    class RecordingExecutor:
        def __init__(self, max_workers, thread_name_prefix):
            del max_workers, thread_name_prefix

        def submit(self, function, frame, common_dtype):
            del function, common_dtype
            future = RecordingFuture(int(frame[0, 0, 0]))
            if future.marker == 0:
                future.set_result(future.marker)
            return future

        def shutdown(self, *, wait, cancel_futures):
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(video_frame_conversion, "ThreadPoolExecutor", RecordingExecutor)
    frames = [np.full((1, 1, 3), index, dtype=np.uint8) for index in range(5)]
    generator = video_frame_conversion._iter_planar_video_frames(frames, np.dtype(np.uint8), worker_count=2)

    assert next(generator) == 0
    generator.close()

    assert cancelled == [1, 2, 3, 4]
    assert shutdown_calls == [(True, True)]


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("channels", [3, 4])
def test_planar_frames_decode_with_bounded_rgb_quantization(dtype, channels):
    channel_values = np.array(
        [
            [[-0.1, 0.0, 0.1, 0.5, 0.9, 1.0, 1.1]],
            [[1.0, 0.9, 0.5, 0.1, 0.0, -0.1, 1.1]],
            [[0.25, 0.75, 0.501, 0.499, 0.125, 0.875, 0.5]],
            [[0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.5]],
        ],
        dtype=dtype,
    )[:channels]
    fhwc = np.transpose(channel_values[:, None, :, :], (1, 2, 3, 0))
    frames = list(video_frame_conversion._iter_planar_video_frames(list(fhwc), np.dtype(dtype)))

    decoded = frames[0].to_ndarray(format="rgb24")
    expected = np.rint(np.clip(fhwc[0, ..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(decoded, expected)


def test_planar_frame_writes_gbr_planes_and_clears_padding():
    width = 17
    planar = np.stack(
        [
            np.full((2, width), 11, dtype=np.uint8),
            np.full((2, width), 22, dtype=np.uint8),
            np.full((2, width), 33, dtype=np.uint8),
        ]
    )
    fhwc = np.transpose(planar[:, None, :, :], (1, 2, 3, 0))

    frame = next(video_frame_conversion._iter_planar_video_frames(list(fhwc), np.dtype(np.uint8)))

    for plane, expected in zip(frame.planes, (22, 33, 11)):
        plane_data = np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)
        np.testing.assert_array_equal(plane_data[:2, :width], expected)
        np.testing.assert_array_equal(plane_data[:2, width:], 0)
        np.testing.assert_array_equal(plane_data[2:], 0)


def test_planar_bool_frames_match_bounded_compatible_output():
    planar = np.array(
        [
            [[[False, True], [True, False]]],
            [[[True, False], [True, False]]],
            [[[False, False], [True, True]]],
        ],
        dtype=np.bool_,
    )
    direct_video = np.transpose(planar, (1, 2, 3, 0))
    compatible_video = np.ascontiguousarray(direct_video)

    prepared_frames, _, common_dtype = video_api_utils._prepare_video_frames(direct_video)
    direct_frames = list(video_frame_conversion._iter_planar_video_frames(prepared_frames, common_dtype))
    decoded_direct = np.stack([frame.to_ndarray(format="rgb24") for frame in direct_frames])
    bounded_compatible = video_api_utils._coerce_video_to_uint8_frames(compatible_video)

    np.testing.assert_array_equal(decoded_direct, bounded_compatible)


@pytest.mark.parametrize(
    ("plane_height", "line_size"),
    [(1, 2), (2, 1)],
)
def test_planar_frame_rejects_undersized_plane(monkeypatch, plane_height, line_size):
    class FakePlane:
        pass

    plane = FakePlane()
    plane.height = plane_height
    plane.line_size = line_size

    class FakeFrame:
        planes = [plane]

    monkeypatch.setattr(av, "VideoFrame", lambda *args, **kwargs: FakeFrame())
    planar = np.zeros((3, 1, 2, 2), dtype=np.uint8)
    fhwc = np.transpose(planar, (1, 2, 3, 0))

    with pytest.raises(ValueError, match="smaller than"):
        next(video_frame_conversion._iter_planar_video_frames(list(fhwc), np.dtype(np.uint8)))

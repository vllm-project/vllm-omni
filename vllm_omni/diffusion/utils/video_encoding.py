# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-budgeted helpers for independent video encoding jobs.

This module deliberately keeps admission separate from frame conversion. Jobs
remain cheap references while queued and only allocate per-frame scratch space
after the scheduler admits them.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import numpy as np

T = TypeVar("T")
R = TypeVar("R")

_MIN_ENCODER_THREADS = 2
_MAX_ENCODER_THREADS = 4
_SEGMENT_SPOOL_LIMIT = 16 * 1024 * 1024


def available_cpu_count() -> int:
    """Return CPUs available to this process and child codec threads."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class EncodingAllocation:
    cpu_count: int
    workers: int
    encoder_threads: int


def calculate_encoding_allocation(job_count: int, cpu_count: int | None = None) -> EncodingAllocation:
    """Apply the shared multiview encoder allocation policy."""
    cpus = available_cpu_count() if cpu_count is None else max(1, int(cpu_count))
    if job_count <= 0:
        return EncodingAllocation(cpus, 0, 0)
    minimum_threads = min(_MIN_ENCODER_THREADS, cpus)
    workers = min(job_count, max(1, cpus // minimum_threads))
    encoder_threads = min(_MAX_ENCODER_THREADS, max(minimum_threads, cpus // workers))
    return EncodingAllocation(cpus, workers, encoder_threads)


def run_ordered_encoding_jobs(
    jobs: Sequence[T],
    encode: Callable[[T, int], R],
    *,
    parallel: bool = True,
    cpu_count: int | None = None,
) -> tuple[list[R], EncodingAllocation]:
    """Run bounded jobs and return results in input order."""
    allocation = calculate_encoding_allocation(len(jobs) if parallel else min(1, len(jobs)), cpu_count)
    if not jobs:
        return [], allocation
    worker_count = allocation.workers if parallel else 1
    if worker_count == 1:
        return [encode(job, allocation.encoder_threads) for job in jobs], allocation

    results: list[R | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="multiview-save") as executor:
        futures = {executor.submit(encode, job, allocation.encoder_threads): index for index, job in enumerate(jobs)}
        try:
            for future, index in futures.items():
                results[index] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [cast(R, result) for result in results], allocation


@dataclass(frozen=True)
class ScheduledResult(Generic[R]):
    value: R
    queue_wait_seconds: float
    run_seconds: float


@dataclass
class _ScheduledJob(Generic[R]):
    request_id: str
    tokens: int
    function: Callable[[threading.Event], R]
    future: Future[ScheduledResult[R]]
    cancel_event: threading.Event = field(default_factory=threading.Event)
    queued_at: float = field(default_factory=time.perf_counter)
    started_at: float | None = None


class VideoEncodingScheduler:
    """Fair, CPU-token admission for encoding work in one API process."""

    def __init__(self, cpu_count: int | None = None) -> None:
        self.cpu_count = available_cpu_count() if cpu_count is None else max(1, int(cpu_count))
        self._executor = ThreadPoolExecutor(max_workers=self.cpu_count, thread_name_prefix="video-encode")
        self._condition = threading.Condition()
        self._queues: OrderedDict[str, deque[_ScheduledJob[Any]]] = OrderedDict()
        self._round_robin: deque[str] = deque()
        self._active: dict[str, set[int]] = {}
        self._active_jobs: dict[int, _ScheduledJob[Any]] = {}
        self._reserved_tokens = 0
        self._closing = False
        self._dispatcher: threading.Thread | None = None

    @property
    def reserved_tokens(self) -> int:
        with self._condition:
            return self._reserved_tokens

    def submit(
        self,
        request_id: str,
        function: Callable[[threading.Event], R],
        *,
        tokens: int,
    ) -> Future[ScheduledResult[R]]:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if tokens <= 0 or tokens > self.cpu_count:
            raise ValueError(f"encoding job tokens must be between 1 and {self.cpu_count}")
        future: Future[ScheduledResult[R]] = Future()
        job = _ScheduledJob(request_id=request_id, tokens=tokens, function=function, future=future)
        with self._condition:
            if self._closing:
                raise RuntimeError("video encoding scheduler is shut down")
            if self._dispatcher is None:
                self._dispatcher = threading.Thread(target=self._dispatch, name="video-encode-dispatch", daemon=True)
                self._dispatcher.start()
            queue = self._queues.get(request_id)
            if queue is None:
                queue = deque()
                self._queues[request_id] = queue
                self._round_robin.append(request_id)
            queue.append(job)
            self._condition.notify_all()

        def cancel_if_needed(completed: Future[ScheduledResult[R]]) -> None:
            if completed.cancelled():
                self._cancel_job(job)

        future.add_done_callback(cancel_if_needed)
        return future

    def _cancel_job(self, job: _ScheduledJob[Any]) -> None:
        with self._condition:
            job.cancel_event.set()
            queue = self._queues.get(job.request_id)
            if queue is not None:
                try:
                    queue.remove(job)
                except ValueError:
                    pass
                self._remove_empty_queue(job.request_id)
            self._condition.notify_all()

    def cancel_request(self, request_id: str) -> None:
        """Cancel queued jobs and ask active jobs to stop at a safe boundary."""
        with self._condition:
            queue = self._queues.pop(request_id, deque())
            self._round_robin = deque(item for item in self._round_robin if item != request_id)
            for job in queue:
                job.cancel_event.set()
                job.future.cancel()
            for job_id in tuple(self._active.get(request_id, ())):
                self._active_jobs[job_id].cancel_event.set()
            self._condition.notify_all()

    def wait_request(self, request_id: str) -> None:
        with self._condition:
            self._condition.wait_for(lambda: request_id not in self._queues and request_id not in self._active)

    def _remove_empty_queue(self, request_id: str) -> None:
        queue = self._queues.get(request_id)
        if queue is not None and not queue:
            self._queues.pop(request_id, None)
            self._round_robin = deque(item for item in self._round_robin if item != request_id)

    def _next_runnable(self) -> _ScheduledJob[Any] | None:
        free_tokens = self.cpu_count - self._reserved_tokens
        while self._round_robin:
            request_id = self._round_robin.popleft()
            queue = self._queues.get(request_id)
            if not queue:
                self._queues.pop(request_id, None)
                continue
            while queue and queue[0].future.cancelled():
                queue.popleft()
            if not queue:
                self._queues.pop(request_id, None)
                continue
            job = queue[0]
            if job.tokens > free_tokens:
                # Keep this oldest round-robin request at the front. Allowing
                # smaller later jobs through indefinitely could starve an
                # all-CPU legacy encode.
                self._round_robin.appendleft(request_id)
                return None
            queue.popleft()
            if queue:
                self._round_robin.append(request_id)
            else:
                self._queues.pop(request_id, None)
            if not job.future.set_running_or_notify_cancel():
                continue
            return job
        return None

    def _dispatch(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closing or (self._round_robin and self._reserved_tokens < self.cpu_count)
                )
                if self._closing:
                    return
                job = self._next_runnable()
                if job is None:
                    self._condition.wait()
                    continue
                job.started_at = time.perf_counter()
                job_id = id(job)
                self._reserved_tokens += job.tokens
                self._active_jobs[job_id] = job
                self._active.setdefault(job.request_id, set()).add(job_id)
            worker = self._executor.submit(job.function, job.cancel_event)
            worker.add_done_callback(lambda done, scheduled=job: self._finish(scheduled, done))

    def _finish(self, job: _ScheduledJob[Any], worker: Future[Any]) -> None:
        finished_at = time.perf_counter()
        failed = False
        try:
            value = worker.result()
        except BaseException as exc:
            failed = True
            if not job.future.done():
                job.future.set_exception(exc)
        else:
            if not job.future.done():
                assert job.started_at is not None
                job.future.set_result(
                    ScheduledResult(
                        value=value,
                        queue_wait_seconds=job.started_at - job.queued_at,
                        run_seconds=finished_at - job.started_at,
                    )
                )
        finally:
            with self._condition:
                job_id = id(job)
                self._reserved_tokens -= job.tokens
                self._active_jobs.pop(job_id, None)
                request_jobs = self._active.get(job.request_id)
                if request_jobs is not None:
                    request_jobs.discard(job_id)
                    if not request_jobs:
                        self._active.pop(job.request_id, None)
                self._condition.notify_all()
        if failed:
            self.cancel_request(job.request_id)

    def shutdown(self) -> None:
        """Reject new jobs, cancel queued work, drain workers, then stop."""
        with self._condition:
            if self._closing:
                return
            self._closing = True
            request_ids = set(self._queues) | set(self._active)
            self._condition.notify_all()
        for request_id in request_ids:
            self.cancel_request(request_id)
        if self._dispatcher is not None:
            self._dispatcher.join()
        self._executor.shutdown(wait=True, cancel_futures=True)


class EncodingCancelledError(RuntimeError):
    pass


@dataclass
class EncodedSegment:
    buffer: Any
    frame_count: int
    width: int
    height: int

    def close(self) -> None:
        self.buffer.close()


def _build_planar_frame(frame: np.ndarray, common_dtype: np.dtype, scratch: np.ndarray | None) -> Any:
    import av

    height, width = frame.shape[:2]
    av_frame = av.VideoFrame(width, height, format="gbrp")
    for plane, channel in zip(av_frame.planes, (1, 2, 0), strict=True):
        plane_view = np.frombuffer(memoryview(plane), dtype=np.uint8, count=plane.height * plane.line_size).reshape(
            plane.height, plane.line_size
        )
        plane_view.fill(0)
        if frame.dtype == np.uint8:
            plane_view[:height, :width] = frame[..., channel]
        else:
            assert scratch is not None
            np.copyto(scratch, frame[..., channel], casting="unsafe")
            np.clip(scratch, 0.0, 1.0, out=scratch)
            scratch *= 255.0
            np.rint(scratch, out=scratch)
            plane_view[:height, :width] = scratch
    return av_frame


def encode_video_segment(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    encoder_threads: int,
    video_codec_options: Mapping[str, Any] | None,
    cancel_event: threading.Event,
    common_dtype: np.dtype | None = None,
    normalization: str = "identity",
) -> EncodedSegment:
    """Encode one camera segment using frame-sized conversion scratch."""
    import av

    if len(frames) == 0:
        raise ValueError("video segment contains no frames")
    frame_shape = frames[0].shape
    if len(frame_shape) != 3 or frame_shape[-1] not in (3, 4):
        raise ValueError(f"video frames must have shape (H, W, 3|4), got {frame_shape}")
    if any(frame.shape != frame_shape for frame in frames):
        raise ValueError("All video frames must have the same shape.")
    common_dtype = (
        np.dtype(common_dtype) if common_dtype is not None else np.result_type(*(frame.dtype for frame in frames))
    )
    if normalization not in {"identity", "signed", "integer"}:
        raise ValueError(f"unsupported frame normalization mode: {normalization}")
    if not (
        common_dtype == np.dtype(np.uint8)
        or np.issubdtype(common_dtype, np.bool_)
        or np.issubdtype(common_dtype, np.floating)
        or (normalization == "integer" and np.issubdtype(common_dtype, np.integer))
    ):
        raise ValueError(f"unsupported video frame dtype: {common_dtype}")

    height, width = frame_shape[:2]
    rate = Fraction(fps).limit_denominator(10000)
    time_base = Fraction(rate.denominator, rate.numerator)
    buffer = tempfile.SpooledTemporaryFile(max_size=_SEGMENT_SPOOL_LIMIT, mode="w+b", suffix=".mp4")
    scratch_dtype = common_dtype if np.issubdtype(common_dtype, np.floating) else np.dtype(np.float64)
    scratch = (
        None
        if common_dtype == np.dtype(np.uint8) and normalization == "identity"
        else np.empty((height, width), dtype=scratch_dtype)
    )
    try:
        with cast(Any, av.open(buffer, mode="w", format="mp4")) as container:
            stream = container.add_stream("h264", rate=rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            options: dict[str, str] = {"crf": "18"}
            if video_codec_options:
                options.update(
                    {str(key): str(value) for key, value in video_codec_options.items() if str(key) != "threads"}
                )
            options["threads"] = str(encoder_threads)
            stream.options = options
            for frame_index, source in enumerate(frames):
                if cancel_event.is_set():
                    raise EncodingCancelledError("video segment encoding was cancelled")
                source = source[..., :3]
                if normalization == "identity":
                    frame = _build_planar_frame(source, common_dtype, scratch)
                else:
                    assert scratch is not None
                    normalized = np.empty_like(source, dtype=np.uint8)
                    for channel in range(3):
                        np.copyto(scratch, source[..., channel], casting="unsafe")
                        if normalization == "signed":
                            np.clip(scratch, -1.0, 1.0, out=scratch)
                            scratch *= 0.5
                            scratch += 0.5
                        else:
                            scratch /= 255.0
                        np.clip(scratch, 0.0, 1.0, out=scratch)
                        scratch *= 255.0
                        np.rint(scratch, out=scratch)
                        normalized[..., channel] = scratch
                    frame = _build_planar_frame(normalized, np.dtype(np.uint8), None)
                frame.pts = frame_index
                frame.time_base = time_base
                for packet in stream.encode(frame):
                    if cancel_event.is_set():
                        raise EncodingCancelledError("video segment encoding was cancelled")
                    container.mux(packet)
            for packet in stream.encode():
                if cancel_event.is_set():
                    raise EncodingCancelledError("video segment encoding was cancelled")
                container.mux(packet)
        buffer.seek(0)
        return EncodedSegment(buffer, len(frames), width, height)
    except BaseException:
        buffer.close()
        raise


def _stream_signature(stream: Any) -> tuple[Any, ...]:
    context = stream.codec_context
    pixel_format = None if context.format is None else context.format.name
    return (
        context.name,
        getattr(context, "profile", None),
        getattr(context, "level", None),
        getattr(context, "codec_tag", None),
        context.width,
        context.height,
        pixel_format,
        bytes(context.extradata or b""),
        stream.time_base,
    )


def remux_video_segments(
    segments: Sequence[EncodedSegment],
    *,
    fps: float,
    cancel_event: threading.Event,
) -> bytes:
    """Concatenate camera-major H.264 MP4 segments by stream-copy remuxing."""
    import av

    if not segments:
        raise ValueError("cannot remux an empty segment list")
    output_buffer = BytesIO()
    output = av.open(output_buffer, mode="w", format="mp4")
    output_stream = None
    expected_signature = None
    cumulative_frames = 0
    rate = Fraction(fps).limit_denominator(10000)
    try:
        for segment in segments:
            if cancel_event.is_set():
                raise EncodingCancelledError("video segment remux was cancelled")
            segment.buffer.seek(0)
            with cast(Any, av.open(segment.buffer, mode="r", format="mp4")) as source:
                if len(source.streams.video) != 1 or source.streams.audio:
                    raise ValueError("parallel multiview segments must contain exactly one video stream and no audio")
                input_stream = source.streams.video[0]
                signature = _stream_signature(input_stream)
                if expected_signature is None:
                    expected_signature = signature
                    add_from_template = getattr(output, "add_stream_from_template", None)
                    output_stream = (
                        add_from_template(input_stream)
                        if add_from_template is not None
                        else output.add_stream(template=input_stream)
                    )
                elif signature != expected_signature:
                    raise ValueError("parallel multiview camera segments have incompatible codec parameters")
                assert output_stream is not None
                packet_time_base = input_stream.time_base
                if packet_time_base is None:
                    raise ValueError("parallel multiview segment is missing its video time base")
                offset_units = Fraction(cumulative_frames, 1) / rate / packet_time_base
                if offset_units.denominator != 1:
                    raise ValueError("camera segment timestamps cannot be represented in the common stream time base")
                offset = offset_units.numerator
                packet_count = 0
                for packet in source.demux(input_stream):
                    if cancel_event.is_set():
                        raise EncodingCancelledError("video segment remux was cancelled")
                    if packet.size == 0:
                        continue
                    if packet.pts is None or packet.dts is None:
                        raise ValueError("nonempty camera segment packet has invalid timestamps")
                    packet.pts += offset
                    packet.dts += offset
                    packet.stream = output_stream
                    output.mux(packet)
                    packet_count += 1
                if packet_count == 0:
                    raise ValueError("parallel multiview segment contains no video packets")
            cumulative_frames += segment.frame_count
            segment.close()
        output.close()
        return output_buffer.getvalue()
    except BaseException:
        output.close()
        raise


def write_imageio_video(
    frames: Sequence[Any],
    path: Path,
    *,
    fps: float,
    encoder_threads: int,
    macro_block_size: int | None = 16,
    normalize_negative: bool = False,
) -> None:
    """Stream Diffusers-compatible frames to ImageIO one at a time."""
    import imageio.v2 as imageio

    with imageio.get_writer(
        str(path),
        fps=fps,
        quality=5.0,
        bitrate=None,
        macro_block_size=macro_block_size,
        output_params=["-threads", str(encoder_threads)],
    ) as writer:
        for source in frames:
            frame = np.asarray(source)
            if frame.dtype == np.uint8:
                converted = frame
            elif np.issubdtype(frame.dtype, np.floating):
                normalized = np.clip(frame, -1.0, 1.0) * 0.5 + 0.5 if normalize_negative else frame
                converted = (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                converted = frame.astype(np.uint8)
            writer.append_data(converted[..., :3])

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Multi-process FFmpeg video encoder backend for chunked video.

This module extends :class:`~vllm_omni.diffusion.utils.chunked_video.ChunkedVideoMP4Session`
by adding an optional ``SegmentedFFmpegEncoder`` that partitions a single video
output into contiguous frame ranges and hands each range to an isolated FFmpeg
process for concurrent encoding. The existing PyAV backend remains the default.

Design follows RFC #8887 on vLLM-Omni.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Segment planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentRange:
    """Half-open frame range ``[start, end)`` for one segment."""

    index: int
    start: int
    end: int

    @property
    def frame_count(self) -> int:
        return self.end - self.start

    @property
    def is_empty(self) -> bool:
        return self.frame_count == 0


@dataclass(frozen=True)
class SegmentPlan:
    """Immutable frame-range plan for multi-process FFmpeg encoding.

    ``segments[i]`` covers frame range ``[start_i, end_i)`` with
    ``end_N == total_frames`` and no gaps or overlaps.
    """

    total_frames: int
    process_count: int
    segments: tuple[SegmentRange, ...]

    def __post_init__(self) -> None:
        if self.total_frames <= 0:
            raise ValueError(f"total_frames must be positive, got {self.total_frames}")
        if self.process_count <= 0:
            raise ValueError(f"process_count must be positive, got {self.process_count}")
        if self.process_count > self.total_frames:
            raise ValueError(f"process_count ({self.process_count}) cannot exceed total_frames ({self.total_frames})")
        if len(self.segments) != self.process_count:
            raise ValueError("segments count must equal process_count")
        # Verify no gaps or overlaps.
        expected_start = 0
        seen = set()
        for seg in self.segments:
            if seg.index in seen:
                raise ValueError(f"duplicate segment index {seg.index}")
            seen.add(seg.index)
            if seg.start != expected_start:
                raise ValueError(f"segment {seg.index}: start={seg.start}, expected {expected_start}")
            if seg.end < seg.start:
                raise ValueError(f"segment {seg.index}: end={seg.end} before start={seg.start}")
            if seg.is_empty:
                raise ValueError(f"segment {seg.index} is empty")
            expected_start = seg.end
        if expected_start != self.total_frames:
            raise ValueError(f"segments cover up to frame {expected_start}, expected {self.total_frames}")


def plan_segments(total_frames: int, process_count: int, weights: list[float] | None = None) -> SegmentPlan:
    """Build an immutable :class:`SegmentPlan` from total frames and weights.

    ``weights`` normalizes to process allocation ratios. ``None`` or all-ones
    means equal partitioning; any decreasing sequence lets later-started
    processes handle fewer frames, reducing tail latency from staggered startup.

    Algorithm:
    1. Compute ``q[i] = total_frames * w[i] / sum(w)`` for each process.
    2. Allocate ``floor(q[i])`` frames to each segment.
    3. Distribute remaining frames by decreasing fractional remainder; on ties,
       stable-sort by segment index so the earlier process keeps its share.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if process_count <= 0:
        raise ValueError(f"process_count must be positive, got {process_count}")
    if process_count > total_frames:
        raise ValueError(f"process_count ({process_count}) cannot exceed total_frames ({total_frames})")

    if weights is None:
        weights = [1.0] * process_count
    if len(weights) != process_count:
        raise ValueError(f"weights length ({len(weights)}) != process_count ({process_count})")
    if any(w <= 0 for w in weights):
        raise ValueError("all weights must be positive")

    total_w = sum(weights)
    quotients = [total_frames * w / total_w for w in weights]
    floors = [int(np.floor(q)) for q in quotients]
    remainders = [q - f for q, f in zip(quotients, floors)]
    remaining = total_frames - sum(floors)

    # Decreasing fractional remainder; ties broken by earlier index.
    order = sorted(range(process_count), key=lambda i: (-remainders[i], i))
    sizes = floors[:]
    for idx in order[:remaining]:
        sizes[idx] += 1

    # Build contiguous segments.
    segments: list[SegmentRange] = []
    cursor = 0
    for i, size in enumerate(sizes):
        if size <= 0:
            raise ValueError(f"segment {i} has zero frames after planning")
        segments.append(SegmentRange(index=i, start=cursor, end=cursor + size))
        cursor += size

    return SegmentPlan(total_frames=total_frames, process_count=process_count, segments=tuple(segments))


# ---------------------------------------------------------------------------
# FFmpeg subprocess management
# ---------------------------------------------------------------------------


def _default_ffmpeg_path() -> str:
    """Return the absolute path to the ffmpeg binary, or raise FileNotFoundError."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise FileNotFoundError("ffmpeg binary not found on PATH; segmented encoding requires ffmpeg")
    return path


class FFmpegProcessError(RuntimeError):
    """Raised when an FFmpeg subprocess fails to start, times out, or exits
    with a non-zero code."""

    def __init__(self, segment_index: int, message: str, *, stderr_tail: str | None = None) -> None:
        full = f"segment {segment_index}: {message}"
        if stderr_tail:
            tail_lines = stderr_tail.strip().splitlines()[-10:]
            full += "\n  stderr tail:\n" + "\n".join("    " + ln for ln in tail_lines)
        super().__init__(full)
        self.segment_index = segment_index


# ---------------------------------------------------------------------------
# Encoder constants and preflight
# ---------------------------------------------------------------------------


# The PyAV encoder's contract: uint8 RGB frames (T, H, W, 3), h264 video codec,
# no audio in per-segment output (audio is muxed once at the end).
#
# Each segment FFmpeg reads raw RGB24 video from stdin and writes a silent
# MP4 segment. Segments are later stream-copy concatenated and audio-remuxed.


def build_ffmpeg_command(
    *,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    video_codec: str = "h264",
    crf: str = "18",
    threads: int = 1,
    video_codec_options: dict[str, str] | None = None,
    output_path: str,
) -> list[str]:
    """Build a validated FFmpeg command line for one segment (no shell).

    Reads raw RGB24 ``width x height`` video from stdin for exactly
    ``frame_count`` frames, encodes with ``video_codec``, writes an MP4 to
    ``output_path``. No audio track is produced.
    """
    ffmpeg = _default_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        # Input: raw RGB24 from stdin.
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "pipe:0",
        # Duration in frames -- consume exactly the planned count.
        "-frames:v",
        str(frame_count),
        # No audio.
        "-an",
        # Output codec selection.
        "-c:v",
        video_codec,
        "-crf",
        str(crf),
        "-threads",
        str(threads),
        # Constrain output to what the final stream-copy concat accepts later:
        # constant frame rate, yuv420p, no b-frames for simpler stream copy.
        "-pix_fmt",
        "yuv420p",
        "-bf",
        "0",
        "-g",
        str(max(1, int(fps))),  # keyframe at ~1s cadence
        "-movflags",
        "+faststart",
    ]
    if video_codec_options:
        for k, v in video_codec_options.items():
            cmd.extend(["-preset" if k == "preset" else f"-{k}", str(v)])
    cmd.append(output_path)
    return cmd


def preflight_check(
    *,
    width: int,
    height: int,
    process_count: int,
    total_frames: int = 0,
    ffmpeg_path: str | None = None,
) -> None:
    """Validate inputs before any GPU work begins. Fails fast on bad params.

    ``total_frames`` defaults to 0 meaning the final frame count is not yet
    known. In that case we skip the ``process_count <= total_frames`` bound
    (which only matters when the plan is actually built in :meth:`finish`).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    if process_count not in (1, 2, 4):
        raise ValueError(f"segmented encoding currently supports 1, 2, or 4 processes, got {process_count}")
    if total_frames < 0:
        raise ValueError("total_frames must be non-negative at preflight time")
    if total_frames > 0 and process_count > total_frames:
        raise ValueError(f"process_count ({process_count}) cannot exceed total_frames ({total_frames})")
    # Confirm FFmpeg is reachable early.
    path = ffmpeg_path or _default_ffmpeg_path()
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"ffmpeg binary at {path} is not executable")
    # Quick version probe.
    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg -version failed: {result.stderr.decode(errors='replace')[:200]}")
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg at {path} vanished between probe and use: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg -version timed out during preflight") from exc


# ---------------------------------------------------------------------------
# Segmented FFmpeg Encoder -- N independent per-segment queues
# ---------------------------------------------------------------------------

_SEGMENT_EOS = object()  # All frames for this segment are in.


@dataclass
class _SegmentState:
    """Runtime bookkeeping for one FFmpeg subprocess."""

    index: int
    frame_count: int
    received: int = 0
    output_path: str = ""
    queue: queue.Queue[object] = None  # type: ignore[assignment]
    process: subprocess.Popen[bytes] | None = None
    stderr_buffer: list[str] = None  # type: ignore[assignment]
    error: BaseException | None = None
    done_event: threading.Event = None  # type: ignore[assignment]


class SegmentedFFmpegEncoder:
    """Encode committed video chunks using N parallel FFmpeg subprocesses.

    Each segment owns exactly one bounded queue and one FFmpeg subprocess.
    Frame routing is done once at dispatch time -- each frame goes directly
    to the queue of its destination segment, so no cross-segment requeueing
    is needed.

    Follows the same ``push()`` / ``finish()`` / ``abort()`` contract as
    :class:`ChunkedMP4Encoder`.

    Pass ``total_frames`` up front to enable streaming mode: the segment plan
    is built and workers start immediately in ``__init__``, so ``push()``
    routes frames directly to the correct queue without accumulating the
    whole video in RAM.  When ``total_frames`` is omitted the encoder falls
    back to a buffering path that still routes at ``finish()``.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float,
        audio_waveform: np.ndarray | None = None,
        audio_sample_rate: int | None = None,
        process_count: int = 4,
        weights: list[float] | None = None,
        max_pending: int = 4,
        video_codec: str = "h264",
        audio_codec: str = "aac",
        crf: str = "18",
        video_codec_options: dict[str, str] | None = None,
        per_process_threads: int = 1,
        total_frames: int | None = None,
    ) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        if process_count not in (1, 2, 4):
            raise ValueError(f"segmented encoding only supports 1, 2, or 4 processes, got {process_count}")
        if per_process_threads <= 0:
            raise ValueError("per_process_threads must be positive")
        if total_frames is not None and total_frames <= 0:
            raise ValueError("total_frames must be positive when provided")

        preflight_check(
            width=width,
            height=height,
            process_count=process_count,
            total_frames=total_frames or 0,
        )

        self.width = int(width)
        self.height = int(height)
        self._fps = float(fps)
        self._audio_waveform = audio_waveform
        self._audio_sample_rate = audio_sample_rate
        self._process_count = process_count
        self._weights = weights
        self._max_pending = max_pending
        self._video_codec = video_codec
        self._audio_codec = audio_codec
        self._crf = crf
        self._video_codec_options = video_codec_options
        self._per_process_threads = per_process_threads

        self._plan: SegmentPlan | None = None
        self._segments: dict[int, _SegmentState] = {}
        self._segment_threads: dict[int, threading.Thread] = {}
        self._dispatch_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._closed = False
        self._aborted = False
        self._failure: BaseException | None = None

        self._pending_frames: list[np.ndarray] = []
        self._total_frames_seen = 0

        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

        # Streaming path (preferred): caller promised a frame count up front,
        # so we can build the plan and start workers immediately.  push()
        # then routes directly to per-segment queues without caching.
        if total_frames is not None:
            self._build_plan_and_workers(total_frames)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, chunk: np.ndarray, *, on_consumed=None) -> None:  # noqa: ANN001
        self._raise_if_closed_or_failed()
        self._validate_chunk(chunk)

        chunk_frames = int(chunk.shape[0])

        if self._plan is not None:
            # Streaming path -- verify we won't overrun the promised total.
            if self._total_frames_seen + chunk_frames > self._plan.total_frames:
                raise ValueError(
                    f"push() would exceed total_frames={self._plan.total_frames}: "
                    f"already sent {self._total_frames_seen}, this chunk adds {chunk_frames}"
                )
            # Route directly, keep no frame copies.
            self._route_chunk_to_segments(chunk)
        else:
            # Fallback -- we don't yet know total_frames, so buffer until
            # finish() builds the plan.
            self._pending_frames.append(chunk.copy())

        self._total_frames_seen += chunk_frames
        if on_consumed is not None:
            on_consumed()

    def finish(self) -> bytes:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("SegmentedFFmpegEncoder is already closed")
            self._closed = True

        if self._plan is not None:
            # Streaming path -- verify frame count matches, then send EOS.
            if self._total_frames_seen != self._plan.total_frames:
                self._failure = RuntimeError(
                    f"promised total_frames={self._plan.total_frames} but received {self._total_frames_seen}"
                )
                self._send_eos_all()  # unblock workers even on mismatch
            else:
                self._send_eos_all()
        else:
            # Fallback path -- build the plan and start workers now.
            if self._total_frames_seen == 0:
                raise RuntimeError("SegmentedFFmpegEncoder received zero frames")
            self._build_and_dispatch()

        self._wait_for_segments()
        result = self._finalize()
        self._cleanup_temp()
        self._raise_if_failed()
        return result

    def abort(self) -> None:
        with self._state_lock:
            self._closed = True
            self._aborted = True

        # Drain each per-segment queue and inject EOS so workers exit.
        for seg in self._segments.values():
            try:
                while True:
                    seg.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                seg.queue.put_nowait(_SEGMENT_EOS)
            except queue.Full:
                pass

        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=5.0)

        # Kill subprocesses.
        for seg in self._segments.values():
            if seg.process is not None and seg.process.poll() is None:
                try:
                    seg.process.terminate()
                except Exception:
                    pass
                try:
                    seg.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        seg.process.kill()
                    except Exception:
                        pass
                    try:
                        seg.process.wait(timeout=1.0)
                    except Exception:
                        pass

        for t in self._segment_threads.values():
            t.join(timeout=3.0)

        self._cleanup_temp()

    close = finish

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_chunk(self, chunk: np.ndarray) -> None:
        if chunk.ndim != 4 or chunk.shape[-1] != 3:
            raise ValueError("video chunk must have shape (T, H, W, 3)")
        if chunk.dtype != np.uint8:
            raise ValueError("video chunk must have dtype uint8")
        if chunk.shape[1] != self.height or chunk.shape[2] != self.width:
            raise ValueError(
                f"video chunk has wrong frame size: expected {self.width}x{self.height}, "
                f"got {chunk.shape[2]}x{chunk.shape[1]}"
            )

    def _raise_if_closed_or_failed(self) -> None:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("SegmentedFFmpegEncoder is already closed")

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _build_plan_and_workers(self, total_frames: int) -> None:
        """Plan segments, create per-segment queues, and start FFmpeg workers."""
        self._plan = plan_segments(total_frames, self._process_count, self._weights)

        self._temp_dir = tempfile.TemporaryDirectory(prefix="vllm-omni-ffmpeg-")

        # Per-segment bounded queues -- each segment owns exactly one.
        # maxsize * process_count total entries across all queues bounds memory.
        per_queue_max = max(1, self._max_pending)

        self._segments = {}
        for seg_range in self._plan.segments:
            self._segments[seg_range.index] = _SegmentState(
                index=seg_range.index,
                frame_count=seg_range.frame_count,
                output_path=os.path.join(self._temp_dir.name, f"segment_{seg_range.index:02d}.mp4"),
                queue=queue.Queue(maxsize=per_queue_max),
                stderr_buffer=[],
                done_event=threading.Event(),
            )

        # One subprocess worker per segment.
        for idx, seg in self._segments.items():
            self._segment_threads[idx] = threading.Thread(
                target=self._segment_worker,
                args=(idx,),
                name=f"ffmpeg-seg-{idx}",
                daemon=True,
            )
            self._segment_threads[idx].start()

    def _build_and_dispatch(self) -> None:
        """Fallback path: build plan from buffered frames and start dispatch."""
        assert self._plan is None
        self._build_plan_and_workers(self._total_frames_seen)
        self._dispatch_buffered_frames()

    def _dispatch_buffered_frames(self) -> None:
        """Walks buffered frames and routes each to its segment's queue.

        Used only by the fallback path when total_frames was not known up front.
        """
        self._dispatch_thread = threading.Thread(target=self._dispatch_worker, name="ffmpeg-dispatch", daemon=True)
        self._dispatch_thread.start()

    def _dispatch_worker(self) -> None:
        try:
            frame_cursor = 0
            for chunk in self._pending_frames:
                for local_offset in range(chunk.shape[0]):
                    if self._aborted or self._failure is not None:
                        return
                    seg_idx = self._segment_for_frame(frame_cursor)
                    if seg_idx is None:
                        return
                    frame = chunk[local_offset : local_offset + 1].copy()
                    try:
                        self._segments[seg_idx].queue.put(frame, timeout=15.0)
                    except queue.Full:
                        if self._failure is None:
                            self._failure = RuntimeError(f"segment {seg_idx} queue timed out waiting for slot")
                        return
                    except Exception:
                        if self._failure is None:
                            self._failure = RuntimeError(f"dispatch to segment {seg_idx} failed")
                        return
                    frame_cursor += 1

            self._send_eos_all()
        except BaseException as exc:  # noqa: BLE001
            if self._failure is None:
                self._failure = exc

        # Release buffered frame copies so memory can be reclaimed.
        self._pending_frames.clear()

    def _route_chunk_to_segments(self, chunk: np.ndarray) -> None:
        """Stream one chunk's frames directly into per-segment queues.

        Called by :meth:`push` in streaming mode.  No frame copies are kept in
        the encoder after this returns.
        """
        assert self._plan is not None
        t = int(chunk.shape[0])
        for local_offset in range(t):
            if self._aborted or self._failure is not None:
                return
            seg_idx = self._segment_for_frame(self._total_frames_seen + local_offset)
            if seg_idx is None:
                return
            frame = chunk[local_offset : local_offset + 1].copy()
            try:
                self._segments[seg_idx].queue.put(frame, timeout=15.0)
            except queue.Full:
                if self._failure is None:
                    self._failure = RuntimeError(f"segment {seg_idx} queue timed out waiting for slot")
                return
            except Exception:
                if self._failure is None:
                    self._failure = RuntimeError(f"dispatch to segment {seg_idx} failed")
                return

    def _send_eos_all(self) -> None:
        """Send the EOS sentinel to every segment queue."""
        for seg in self._segments.values():
            if self._aborted or self._failure is not None:
                return
            try:
                seg.queue.put(_SEGMENT_EOS, timeout=30.0)
            except queue.Full:
                if self._failure is None:
                    self._failure = RuntimeError(f"cannot put EOS into segment {seg.index} queue")
                return

    def _segment_for_frame(self, frame_index: int) -> int | None:
        if self._plan is None or frame_index < 0 or frame_index >= self._plan.total_frames:
            return None
        # Linear scan -- process_count is 1/2/4 so this is fine.
        for seg in self._plan.segments:
            if seg.start <= frame_index < seg.end:
                return seg.index
        return None

    def _segment_worker(self, segment_index: int) -> None:
        """Reader thread for one FFmpeg subprocess -- reads only its own queue."""
        seg = self._segments[segment_index]
        seg.start_time = time.perf_counter()

        try:
            cmd = build_ffmpeg_command(
                width=self.width,
                height=self.height,
                fps=self._fps,
                frame_count=seg.frame_count,
                video_codec=self._video_codec,
                crf=self._crf,
                threads=self._per_process_threads,
                video_codec_options=self._video_codec_options,
                output_path=seg.output_path,
            )
            seg.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            seg.error = FFmpegProcessError(segment_index, f"ffmpeg binary missing: {exc}")
            seg.done_event.set()
            return
        except OSError as exc:
            seg.error = FFmpegProcessError(segment_index, f"cannot start subprocess: {exc}")
            seg.done_event.set()
            return

        stderr_thread = threading.Thread(
            target=_drain_pipe,
            args=(seg.process.stderr, seg.stderr_buffer),
            daemon=True,
        )
        stderr_thread.start()

        try:
            while True:
                if self._aborted:
                    break
                entry = seg.queue.get()
                if entry is _SEGMENT_EOS:
                    break
                assert isinstance(entry, np.ndarray)
                try:
                    assert seg.process is not None and seg.process.stdin is not None
                    seg.process.stdin.write(entry.tobytes())
                    seg.received += entry.shape[0]
                except (BrokenPipeError, OSError) as exc:
                    seg.error = FFmpegProcessError(
                        segment_index,
                        f"write to ffmpeg stdin failed: {exc}",
                        stderr_tail=_stderr_tail(seg),
                    )
                    break
        finally:
            try:
                assert seg.process is not None and seg.process.stdin is not None
                seg.process.stdin.close()
            except Exception:
                pass

            assert seg.process is not None
            try:
                seg.process.wait(timeout=120.0)
            except subprocess.TimeoutExpired:
                seg.process.kill()
                try:
                    seg.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
                if seg.error is None:
                    seg.error = FFmpegProcessError(
                        segment_index,
                        "ffmpeg did not exit within 120s; killed",
                        stderr_tail=_stderr_tail(seg),
                    )

            stderr_thread.join(timeout=5.0)

            if seg.process.returncode != 0 and seg.error is None:
                seg.error = FFmpegProcessError(
                    segment_index,
                    f"ffmpeg exited with code {seg.process.returncode}",
                    stderr_tail=_stderr_tail(seg),
                )

            if seg.error is None and seg.received != seg.frame_count:
                seg.error = RuntimeError(
                    f"segment {segment_index}: received {seg.received} frames, expected {seg.frame_count}"
                )

        seg.done_event.set()

    def _wait_for_segments(self) -> None:
        for idx, t in self._segment_threads.items():
            t.join(timeout=130.0)
        if self._plan is None:
            return
        for seg in self._plan.segments:
            state = self._segments[seg.index]
            if state.error is not None:
                self._failure = state.error
                return

    def _finalize(self) -> bytes:
        assert self._plan is not None
        segment_paths = [self._segments[seg.index].output_path for seg in self._plan.segments]
        try:
            concatenated = _stream_copy_concat(segment_paths)
            return _mux_audio_once(
                concatenated,
                waveform=self._audio_waveform,
                sample_rate=self._audio_sample_rate,
            )
        except BaseException as exc:  # noqa: BLE001
            if self._failure is None:
                self._failure = exc
            raise

    def _cleanup_temp(self) -> None:
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                logger.exception("Failed to clean up segmented FFmpeg temp directory")
            self._temp_dir = None


# ---------------------------------------------------------------------------
# Pipe & concat helpers
# ---------------------------------------------------------------------------


def _drain_pipe(stream, buffer: list[str], *, max_lines: int = 200) -> None:
    if stream is None:
        return
    try:
        for raw in stream:
            text = raw.decode(errors="replace").rstrip("\n")
            buffer.append(text)
            if len(buffer) > max_lines:
                del buffer[: len(buffer) // 2]
    except (BrokenPipeError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _stderr_tail(seg: _SegmentState, *, n: int = 15) -> str | None:
    if not seg.stderr_buffer:
        return None
    return "\n".join(seg.stderr_buffer[-n:])


def _stream_copy_concat(segment_paths: list[str]) -> bytes:
    """Stream-copy concatenate MP4 segments using ffmpeg concat demuxer."""
    if not segment_paths:
        raise ValueError("no segments to concatenate")
    if len(segment_paths) == 1:
        with open(segment_paths[0], "rb") as f:
            return f.read()

    # Build concat list file with properly escaped paths.
    def _escape(path: str) -> str:
        return path.replace("\\", "/").replace("'", "\\'")

    lines = [f"file '{_escape(os.path.abspath(p))}'" for p in segment_paths]
    list_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    list_file.write("\n".join(lines) + "\n")
    list_file.close()

    out = tempfile.NamedTemporaryFile(mode="rb", suffix=".mp4", delete=False)
    out.close()

    try:
        cmd = [
            _default_ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file.name,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            out.name,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed (code {result.returncode}): {result.stderr.decode(errors='replace')[-1000:]}"
            )
        with open(out.name, "rb") as f:
            return f.read()
    finally:
        for p in [list_file.name, out.name]:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def _mux_audio_once(video_bytes: bytes, *, waveform: np.ndarray | None, sample_rate: int | None) -> bytes:
    """Attach an audio track to an already-encoded MP4 without re-encoding the video.

    Uses an ffmpeg subprocess to stream-copy the video track and encode / mux
    the audio from a raw PCM dump of ``waveform``. This preserves the
    previously-encoded video bitstream (no decode--re-encode round-trip) and is
    significantly faster than the previous PyAV round-trip.
    """
    if waveform is None:
        return video_bytes

    if sample_rate is None:
        raise ValueError("audio_sample_rate is required when audio_waveform is provided")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")
        audio_path = os.path.join(tmpdir, "audio.pcm")
        out_path = os.path.join(tmpdir, "output.mp4")

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # Normalise waveform layout to (channels, samples) float32 in [-1, 1].
        samples = waveform.astype(np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        elif samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
            samples = np.ascontiguousarray(samples.T)
        num_channels = samples.shape[0]

        # Convert float32 -> int16 raw PCM for ffmpeg's s16le input format.
        pcm = np.clip(samples, -1.0, 1.0)
        pcm_int16 = (pcm * 32767.0).astype(np.int16)
        pcm_int16.tofile(audio_path)

        cmd = [
            _default_ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_path,
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(num_channels),
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            out_path,
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio mux failed (code {result.returncode}): {result.stderr.decode(errors='replace')[-1000:]}"
            )

        with open(out_path, "rb") as f:
            return f.read()

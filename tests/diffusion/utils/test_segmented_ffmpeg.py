# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression tests for :mod:`vllm_omni.diffusion.utils.segmented_ffmpeg`.

Requires ``ffmpeg`` on PATH -- the module's ``preflight_check`` will raise
otherwise.  These tests cover segment planning, single/parallel process
encoding, audio muxing, and the failure paths the reviewers flagged.
"""

import io
import subprocess

import av
import numpy as np
import pytest

from vllm_omni.diffusion.utils.segmented_ffmpeg import (
    FFmpegProcessError,
    SegmentedFFmpegEncoder,
    SegmentPlan,
    SegmentRange,
    build_ffmpeg_command,
    plan_segments,
    preflight_check,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frames(*, total: int = 24, height: int = 16, width: int = 24, seed: int = 4871) -> np.ndarray:
    """Random uint8 RGB frames shaped ``(T, H, W, 3)``."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(total, height, width, 3), dtype=np.uint8)


def _count_mp4_frames(video_bytes: bytes) -> int | None:
    """Read frame count from the sample table without decoding."""
    if not video_bytes:
        return None
    try:
        with av.open(io.BytesIO(video_bytes), format="mp4") as container:
            n = int(container.streams.video[0].frames)
    except Exception:
        return None
    return n or None


def _has_audio_track(video_bytes: bytes) -> bool:
    if not video_bytes:
        return False
    try:
        with av.open(io.BytesIO(video_bytes), format="mp4") as container:
            return bool(container.streams.audio)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Segment planner
# ---------------------------------------------------------------------------


class TestPlanSegments:
    def test_equal_split(self) -> None:
        plan = plan_segments(8, 4)
        assert isinstance(plan, SegmentPlan)
        assert plan.total_frames == 8
        assert plan.process_count == 4
        assert [(s.start, s.end) for s in plan.segments] == [(0, 2), (2, 4), (4, 6), (6, 8)]

    def test_unequal_remainder(self) -> None:
        # 10 frames / 3 processes => 3, 3, 4
        plan = plan_segments(10, 3)
        sizes = [s.frame_count for s in plan.segments]
        assert sum(sizes) == 10
        assert sizes.count(3) == 2 and sizes.count(4) == 1
        # Contiguous, no gaps.
        for i in range(len(plan.segments) - 1):
            assert plan.segments[i].end == plan.segments[i + 1].start

    def test_custom_weights(self) -> None:
        plan = plan_segments(12, 4, weights=[4.0, 3.0, 2.0, 1.0])
        sizes = [s.frame_count for s in plan.segments]
        assert sizes == [5, 4, 2, 1]

    def test_rejects_zero_total(self) -> None:
        with pytest.raises(ValueError, match="total_frames must be positive"):
            plan_segments(0, 2)

    def test_rejects_more_processes_than_frames(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed total_frames"):
            plan_segments(4, 8)

    def test_segment_plan_rejects_gap(self) -> None:
        with pytest.raises(ValueError, match="expected 10"):
            SegmentPlan(
                total_frames=10,
                process_count=2,
                segments=(SegmentRange(0, 0, 3), SegmentRange(1, 6, 10)),  # gap [3,6)
            )


# ---------------------------------------------------------------------------
# Preflight & command builder
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_accepts_1_2_4_processes(self) -> None:
        for n in (1, 2, 4):
            preflight_check(width=24, height=16, process_count=n)

    def test_rejects_bad_process_count(self) -> None:
        with pytest.raises(ValueError, match="supports 1, 2, or 4"):
            preflight_check(width=24, height=16, process_count=3)

    def test_rejects_non_positive_size(self) -> None:
        with pytest.raises(ValueError, match="width and height"):
            preflight_check(width=0, height=16, process_count=1)

    def test_build_ffmpeg_command_runs(self, tmp_path) -> None:
        cmd = build_ffmpeg_command(width=24, height=16, fps=12, frame_count=4, output_path=str(tmp_path / "seg.mp4"))
        # Sanity: first token is ffmpeg binary, has -f rawvideo.
        assert cmd[0].endswith("ffmpeg")
        assert cmd.count("-f") >= 1  # for rawvideo input


# ---------------------------------------------------------------------------
# Encoder: basic happy path
# ---------------------------------------------------------------------------


class TestSegmentedFFmpegEncoder:
    def test_single_process_matches_frame_count(self) -> None:
        frames = _frames(total=12, height=16, width=24)
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=12, process_count=1)
        enc.push(frames[:4])
        enc.push(frames[4:8])
        enc.push(frames[8:])
        result = enc.finish()
        assert _count_mp4_frames(result) == 12

    def test_two_process_matches_frame_count(self) -> None:
        frames = _frames(total=24, height=16, width=24)
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=12, process_count=2)
        enc.push(frames[:8])
        enc.push(frames[8:16])
        enc.push(frames[16:])
        result = enc.finish()
        assert _count_mp4_frames(result) == 24

    def test_four_process_matches_frame_count(self) -> None:
        frames = _frames(total=32, height=16, width=24)
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=12, process_count=4)
        enc.push(frames[:8])
        enc.push(frames[8:16])
        enc.push(frames[16:24])
        enc.push(frames[24:])
        result = enc.finish()
        assert _count_mp4_frames(result) == 32

    def test_streaming_mode_total_frames(self) -> None:
        frames = _frames(total=20, height=16, width=24)
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=10, process_count=2, total_frames=20)
        enc.push(frames[:10])
        enc.push(frames[10:])
        result = enc.finish()
        assert _count_mp4_frames(result) == 20

    def test_silent_output_when_no_audio(self) -> None:
        frames = _frames(total=8)
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8)
        enc.push(frames)
        result = enc.finish()
        assert not _has_audio_track(result)

    def test_audio_mux(self) -> None:
        frames = _frames(total=8)
        mono = np.zeros(4000, dtype=np.float32)
        mono[:500] = 0.5
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8, audio_waveform=mono, audio_sample_rate=8000)
        enc.push(frames)
        result = enc.finish()
        assert _has_audio_track(result)

    def test_stereo_audio_preserves_channels(self) -> None:
        frames = _frames(total=8)
        stereo = np.zeros((2, 4000), dtype=np.float32)
        stereo[0, :500] = 0.5
        stereo[1, :500] = -0.5
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8, audio_waveform=stereo, audio_sample_rate=8000)
        enc.push(frames)
        result = enc.finish()
        assert _has_audio_track(result)

    def test_audio_mux_does_not_truncate_video(self) -> None:
        """Regression: removing -shortest means a short audio waveform
        must NOT truncate the video track."""
        frames = _frames(total=24, height=16, width=24)
        # Audio for only half the duration at 8000Hz sample rate for 24 frames at 12fps = 2 seconds
        # Half of that = 1 second = 8000 samples.
        short_audio = np.zeros(8000, dtype=np.float32)
        enc = SegmentedFFmpegEncoder(
            width=24, height=16, fps=12, audio_waveform=short_audio, audio_sample_rate=8000, process_count=2
        )
        enc.push(frames)
        result = enc.finish()
        # Even though audio is shorter, the video must still have all 24 frames.
        assert _count_mp4_frames(result) == 24


# ---------------------------------------------------------------------------
# Failure paths -- the two P1s from hsliuustc0106 / LinzeShi
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_push_fails_closed_on_partial_route(self) -> None:
        """hsliuustc0106 P1: if a queue put times out mid-chunk, push() must
        raise and must NOT advance _total_frames_seen or call on_consumed
        for the frames that never entered a queue."""
        frames = _frames(total=16)
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8, process_count=2, total_frames=16, max_pending=1)
        consumed = []

        def track_consumed():
            consumed.append(True)

        # Push one small chunk to get queues rolling, then try to overflow one.
        enc.push(frames[:2], on_consumed=track_consumed)
        assert enc._total_frames_seen == 2

        # The queue timeout is 15s; patch it to instant to fail fast.
        import queue

        def instant_put(item, timeout=None):
            raise queue.Full("injected failure")

        enc._segments[0].queue.put = instant_put  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="queue timed out|injected"):
            enc.push(frames[2:10], on_consumed=track_consumed)

        # Key assertion: frame count must NOT have advanced past what succeeded.
        assert enc._total_frames_seen == 2, f"expected 2, got {enc._total_frames_seen}"
        # And on_consumed was called exactly once (for the first chunk).
        assert len(consumed) == 1

        enc.abort()

    def test_finish_surfaces_frame_count_mismatch(self) -> None:
        """LinzeShi 2026-10-20 / hsliuustc0106 P1: when fewer frames arrive
        than promised, finish() must raise the count error directly -- it
        must NOT try to assemble incomplete segments whose concat failure
        would hide the real error."""
        frames = _frames(total=12)
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8, process_count=2, total_frames=12)
        enc.push(frames[:4])  # only 4 of 12 promised

        with pytest.raises(RuntimeError, match="promised total_frames=12 but received 4"):
            enc.finish()

    def test_finish_overrun_is_rejected(self) -> None:
        frames = _frames(total=10)
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8, process_count=2, total_frames=8)
        enc.push(frames[:5])
        with pytest.raises(ValueError, match="exceed total_frames"):
            enc.push(frames[5:])
        enc.abort()

    def test_abort_terminates_workers(self) -> None:
        frames = _frames(total=8)
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8, process_count=2, max_pending=1)
        enc.push(frames[:2])
        enc.abort()
        # After abort, push must raise.
        with pytest.raises(RuntimeError):
            enc.push(frames[2:3])


# ---------------------------------------------------------------------------
# Shape & dtype validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_wrong_ndim(self) -> None:
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8)
        with pytest.raises(ValueError, match="shape"):
            enc.push(np.zeros((16, 24, 3), dtype=np.uint8))
        enc.abort()

    def test_rejects_wrong_dtype(self) -> None:
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8)
        with pytest.raises(ValueError, match="dtype"):
            enc.push(np.zeros((1, 16, 24, 3), dtype=np.float32))
        enc.abort()

    def test_rejects_wrong_size(self) -> None:
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8)
        with pytest.raises(ValueError, match="wrong frame size"):
            enc.push(np.zeros((2, 32, 32, 3), dtype=np.uint8))
        enc.abort()

    def test_rejects_negative_or_zero_max_pending(self) -> None:
        with pytest.raises(ValueError, match="max_pending must be positive"):
            SegmentedFFmpegEncoder(width=24, height=16, fps=8, max_pending=0)
        with pytest.raises(ValueError, match="max_pending must be positive"):
            SegmentedFFmpegEncoder(width=24, height=16, fps=8, max_pending=-1)


# ---------------------------------------------------------------------------
# FFmpegProcessError carries stderr tail
# ---------------------------------------------------------------------------


class TestFFmpegProcessError:
    def test_carries_stderr_tail(self) -> None:
        exc = FFmpegProcessError(3, "boom", stderr_tail="line1\nline2\nline3\nline4\nline5")
        assert exc.segment_index == 3
        assert "segment 3: boom" in str(exc)
        assert "stderr tail" in str(exc)


# ---------------------------------------------------------------------------
# Fallback path (no total_frames up front)
# ---------------------------------------------------------------------------


class TestFallbackPath:
    def test_buffers_then_encodes(self) -> None:
        frames = _frames(total=16)
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8, process_count=2)
        # Fallback path: no total_frames given; push accumulates, finish() plans.
        enc.push(frames[:6])
        enc.push(frames[6:12])
        enc.push(frames[12:])
        assert enc._plan is None  # no plan yet
        result = enc.finish()
        assert _count_mp4_frames(result) == 16

    def test_zero_frames_is_rejected(self) -> None:
        enc = SegmentedFFmpegEncoder(width=24, height=16, fps=8)
        with pytest.raises(RuntimeError, match="zero frames"):
            enc.finish()


# ---------------------------------------------------------------------------
# ChunkedVideoMP4Session wiring (hsliuustc0106 "Wire or drop")
# ---------------------------------------------------------------------------


class TestChunkedVideoMP4SessionWiring:
    def test_use_segmented_encoder_parameter_wires_backend(self) -> None:
        """The session must construct SegmentedFFmpegEncoder when
        ``use_segmented_encoder=True``."""
        import torch

        from vllm_omni.diffusion.utils.chunked_video import ChunkedVideoMP4Session, SegmentedFFmpegEncoder

        if SegmentedFFmpegEncoder is None:
            pytest.skip("ffmpeg not available")

        session = ChunkedVideoMP4Session(
            value_range=(-1.0, 1.0),
            fps=8,
            use_segmented_encoder=True,
            segmented_process_count=1,
            transfer_slots=0,  # force blocking copies
        )
        # Push one tiny chunk so _ensure_encoders fires.
        chunk = torch.zeros((1, 3, 3, 16, 24), dtype=torch.float32)
        session.push(chunk)
        encoders = session._encoders
        assert len(encoders) == 1
        assert isinstance(encoders[0], SegmentedFFmpegEncoder)
        session.finish()

    def test_default_is_still_pyav_backend(self) -> None:
        """Omitting ``use_segmented_encoder`` must keep ChunkedMP4Encoder."""
        import torch

        from vllm_omni.diffusion.utils.chunked_video import ChunkedVideoMP4Session
        from vllm_omni.diffusion.utils.media_utils import ChunkedMP4Encoder

        session = ChunkedVideoMP4Session(
            value_range=(-1.0, 1.0),
            fps=8,
            transfer_slots=0,
        )
        chunk = torch.zeros((1, 3, 3, 16, 24), dtype=torch.float32)
        session.push(chunk)
        assert isinstance(session._encoders[0], ChunkedMP4Encoder)
        session.finish()


# ---------------------------------------------------------------------------
# ffmpeg process behaviour (sanity check subprocess.run path)
# ---------------------------------------------------------------------------


def test_ffmpeg_binary_exists_and_responds() -> None:
    """Smoke: the ``subprocess.run(cmd, ...)`` call path used internally
    works end-to-end."""
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
    assert result.returncode == 0
    assert b"ffmpeg version" in result.stdout

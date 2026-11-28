# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the ffmpeg-backed chunked video encoder backends."""

from __future__ import annotations

import io

import av
import numpy as np
import pytest

from vllm_omni.diffusion.utils.ffmpeg_encoder import (
    _ffmpeg_available,
    _plan_segment_ranges,
    make_chunked_video_encoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture()
def frames() -> np.ndarray:
    rng = np.random.default_rng(6872)
    return rng.integers(0, 256, size=(12, 16, 24, 3), dtype=np.uint8)


@pytest.fixture()
def audio_waveform() -> np.ndarray:
    return np.zeros((2, 48000), dtype=np.float32)


pytest_skip_no_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="ffmpeg binary not available on PATH",
)


# ---------------------------------------------------------------------------
# Planning helpers
# ---------------------------------------------------------------------------


class TestPlanSegmentRanges:
    def test_equal_partitioning(self) -> None:
        assert _plan_segment_ranges(12, 4, None) == [
            (0, 3),
            (3, 6),
            (6, 9),
            (9, 12),
        ]

    def test_weighted_partitioning(self) -> None:
        result = _plan_segment_ranges(12, 4, "4,3,2,1")
        assert sum(e - s for s, e in result) == 12
        # First segment gets the largest share.
        assert result[0][1] - result[0][0] > result[-1][1] - result[-1][0]

    def test_single_segment(self) -> None:
        assert _plan_segment_ranges(12, 1, None) == [(0, 12)]

    def test_one_frame(self) -> None:
        assert _plan_segment_ranges(1, 8, None) == [(0, 1)]

    def test_coverage_and_disjoint(self) -> None:
        for total in (1, 7, 13, 24, 100):
            for num in (1, 2, 4, 8):
                ranges = _plan_segment_ranges(total, num, None)
                flat = [f for s, e in ranges for f in range(s, e)]
                assert flat == list(range(total))

    def test_weight_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="segment_ratio has"):
            _plan_segment_ranges(12, 4, "1,2")

    def test_non_positive_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _plan_segment_ranges(12, 4, "1,2,0,1")


# ---------------------------------------------------------------------------
# Shared contract tests across all three backends
# ---------------------------------------------------------------------------


def _probe_frame_count(video_bytes: bytes) -> int | None:
    """Decoded frame count via ffmpeg / PyAV."""
    try:
        with av.open(io.BytesIO(video_bytes), format="mp4") as container:
            video_stream = next((s for s in container.streams if s.type == "video"), None)
            if video_stream is None:
                return 0
            return len(list(container.decode(video_stream)))
    except Exception:
        return None


@pytest.mark.parametrize("backend", ["pyav", "ffmpeg", "segmented_ffmpeg"])
@pytest_skip_no_ffmpeg
class TestEncoderContract:
    def test_push_finish_roundtrip(self, frames: np.ndarray, backend: str) -> None:
        enc = make_chunked_video_encoder(
            backend,
            width=24,
            height=16,
            fps=24,
            max_pending=4,
        )
        enc.push(frames[:4])
        enc.push(frames[4:8])
        enc.push(frames[8:])
        result = enc.finish()
        assert result
        assert _probe_frame_count(result) == 12

    def test_single_push(self, frames: np.ndarray, backend: str) -> None:
        enc = make_chunked_video_encoder(backend, width=24, height=16, fps=24)
        enc.push(frames)
        result = enc.finish()
        assert _probe_frame_count(result) == 12

    def test_close_equals_finish(self, frames: np.ndarray, backend: str) -> None:
        enc = make_chunked_video_encoder(backend, width=24, height=16, fps=24)
        enc.push(frames[:1])
        first = enc.close()
        enc2 = make_chunked_video_encoder(backend, width=24, height=16, fps=24)
        enc2.push(frames[:1])
        second = enc2.finish()
        assert _probe_frame_count(first) == _probe_frame_count(second) == 1

    def test_abort_rejects_late_push(self, frames: np.ndarray, backend: str) -> None:
        enc = make_chunked_video_encoder(backend, width=24, height=16, fps=24)
        enc.push(frames[:1])
        enc.abort()
        with pytest.raises(RuntimeError, match="already closed|closed"):
            enc.push(frames[:1])

    def test_validate_chunk_shape(self, backend: str) -> None:
        enc = make_chunked_video_encoder(backend, width=24, height=16, fps=24)
        with pytest.raises(ValueError, match="shape"):
            enc.push(np.zeros((16, 24, 3), dtype=np.uint8))
        enc.abort()

    def test_validate_chunk_dtype(self, backend: str) -> None:
        enc = make_chunked_video_encoder(backend, width=24, height=16, fps=24)
        with pytest.raises(ValueError, match="dtype"):
            enc.push(np.zeros((1, 16, 24, 3), dtype=np.float32))
        enc.abort()


# ---------------------------------------------------------------------------
# Audio overlay (ffmpeg backends)
# ---------------------------------------------------------------------------


@pytest_skip_no_ffmpeg
@pytest.mark.parametrize("backend", ["ffmpeg", "segmented_ffmpeg"])
class TestFFmpegAudio:
    def test_with_audio_waveform(self, frames: np.ndarray, audio_waveform: np.ndarray, backend: str) -> None:
        kwargs = dict(
            width=24,
            height=16,
            fps=24,
            audio_waveform=audio_waveform,
            audio_sample_rate=48000,
        )
        if backend == "segmented_ffmpeg":
            kwargs["num_segments"] = 3
        enc = make_chunked_video_encoder(backend, **kwargs)
        enc.push(frames)
        result = enc.finish()
        # Verify both streams are present.
        with av.open(io.BytesIO(result), format="mp4") as container:
            assert any(s.type == "video" for s in container.streams)
            assert any(s.type == "audio" for s in container.streams)


# ---------------------------------------------------------------------------
# FFmpeg-specific behavior
# ---------------------------------------------------------------------------


@pytest_skip_no_ffmpeg
class TestFFmpegBinaryEncoder:
    def test_encoder_defaults_to_libx264(self, frames: np.ndarray) -> None:
        enc = make_chunked_video_encoder("ffmpeg", width=24, height=16, fps=24)
        enc.push(frames)
        result = enc.finish()
        with av.open(io.BytesIO(result), format="mp4") as container:
            stream = next(s for s in container.streams if s.type == "video")
            assert stream.codec.name in {"h264", "libx264"}

    def test_frames_contiguous_after_encode(self, frames: np.ndarray) -> None:
        """ffmpeg rawvideo input requires contiguous bytes."""
        enc = make_chunked_video_encoder("ffmpeg", width=24, height=16, fps=24)
        # Supply slices that are not naturally contiguous.
        non_contig = frames[::2]
        assert not non_contig.flags.c_contiguous
        enc.push(non_contig)
        enc.push(frames[1::2])
        result = enc.finish()
        assert _probe_frame_count(result) == 12 - 6 + 6


# ---------------------------------------------------------------------------
# Segmented-specific behavior
# ---------------------------------------------------------------------------


@pytest_skip_no_ffmpeg
class TestSegmentedFFmpegEncoder:
    def test_auto_segment_count(self, frames: np.ndarray) -> None:
        """num_segments=0 triggers auto partitioning based on host CPU count."""
        enc = make_chunked_video_encoder(
            "segmented_ffmpeg",
            width=24,
            height=16,
            fps=24,
            num_segments=0,
        )
        enc.push(frames)
        result = enc.finish()
        assert _probe_frame_count(result) == 12

    def test_custom_segment_ratio(self, frames: np.ndarray) -> None:
        """Falling ratios put more work on earlier segments."""
        enc = make_chunked_video_encoder(
            "segmented_ffmpeg",
            width=24,
            height=16,
            fps=24,
            num_segments=4,
            segment_ratio="5,3,2,2",
        )
        enc.push(frames)
        result = enc.finish()
        assert _probe_frame_count(result) == 12

    def test_single_segment_is_passthrough(self, frames: np.ndarray) -> None:
        """num_segments=1 should behave like FFmpegBinaryEncoder (one encode, one concat path)."""
        enc = make_chunked_video_encoder(
            "segmented_ffmpeg",
            width=24,
            height=16,
            fps=24,
            num_segments=1,
        )
        enc.push(frames)
        result = enc.finish()
        assert _probe_frame_count(result) == 12

    def test_empty_push_rejected(self) -> None:
        enc = make_chunked_video_encoder(
            "segmented_ffmpeg",
            width=24,
            height=16,
            fps=24,
            num_segments=2,
        )
        with pytest.raises(ValueError, match="shape"):
            enc.push(np.zeros((0, 16, 24, 3), dtype=np.uint8))
        enc.abort()


# ---------------------------------------------------------------------------
# ChunkedVideoMP4Session wiring
# ---------------------------------------------------------------------------


@pytest_skip_no_ffmpeg
class TestSessionBackendWiring:
    def test_session_switch_backend(self, frames: np.ndarray) -> None:
        import torch

        from vllm_omni.diffusion.utils.chunked_video import ChunkedVideoMP4Session

        # Build fake chunks in BCTHW layout (B=1, C=3, T=slice_len, H=16, W=24),
        # which is what ChunkedVideoMP4Session.push expects.  The fixture
        # provides (T, H, W, 3) = THWC numpy chunks; swap axes and add the
        # leading batch dim.
        # THWC (T, H, W, C) -> permute(3, 0, 1, 2) => B missing -> unsqueeze(0)
        chunks = [
            torch.from_numpy(f.copy()).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
            for f in [frames[:4], frames[4:8], frames[8:]]
        ]

        for backend in ("pyav", "ffmpeg", "segmented_ffmpeg"):
            session = ChunkedVideoMP4Session(
                value_range=(0, 255),
                fps=24,
                encoder_backend=backend,
                num_segments=3 if backend == "segmented_ffmpeg" else 0,
                transfer_slots=0,  # disable pinned ring for CPU-only test
            )
            for chunk in chunks:
                session.push(chunk)
            videos = session.finish()
            assert len(videos) == 1
            assert _probe_frame_count(videos[0]) == 12, f"backend={backend}"

    def test_session_invalid_backend_raises(self) -> None:
        import torch

        from vllm_omni.diffusion.utils.chunked_video import ChunkedVideoMP4Session

        # BCTHW shape: B=1, C=3, T=4, H=16, W=24.
        session = ChunkedVideoMP4Session(
            value_range=(0, 255),
            fps=24,
            encoder_backend="not_a_backend",
            transfer_slots=0,
        )
        with pytest.raises(ValueError, match="unknown encoder backend"):
            session.push(torch.zeros(1, 3, 4, 16, 24, dtype=torch.uint8))


# ---------------------------------------------------------------------------
# Errors propagate correctly
# ---------------------------------------------------------------------------


@pytest_skip_no_ffmpeg
class TestErrorPropagation:
    def test_invalid_ratio_instantiation_fails_early(self) -> None:
        enc = make_chunked_video_encoder(
            "segmented_ffmpeg",
            width=24,
            height=16,
            fps=24,
            num_segments=4,
            segment_ratio="1,2",
        )
        enc.push(np.zeros((4, 16, 24, 3), dtype=np.uint8))
        # The bad ratio is caught at finish-time planning.
        with pytest.raises(ValueError):
            enc.finish()


# ---------------------------------------------------------------------------
# Factory rejects unknowns
# ---------------------------------------------------------------------------


class TestFactory:
    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown encoder backend"):
            make_chunked_video_encoder("does_not_exist", width=24, height=16, fps=24)

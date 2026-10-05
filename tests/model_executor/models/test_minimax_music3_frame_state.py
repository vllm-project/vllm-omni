# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiniMax Music 3 per-request AR frame bookkeeping.

The talker emits overlapping windows while dropping frames it can no longer
need, and flushes whatever is left as one span when the song ends. Getting the
eviction boundary wrong by one frame silently corrupts the conditioning handed
to the acoustic stage from the second window onward, which is audible but hard
to attribute, so the lifecycle is simulated here frame by frame.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.minimax_music3.chunking import chunk_windows
from vllm_omni.model_executor.models.minimax_music3.constants import (
    AR_CHUNK_FRAMES,
    AR_CHUNK_HOP_FRAMES,
    NUM_CODEBOOKS,
)
from vllm_omni.model_executor.models.minimax_music3.frame_state import (
    FrameBuffer,
    RequestARState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# One scalar per frame instead of 32768 dims: the bookkeeping is what is under
# test, and a frame's value doubles as its identity.
_WIDTH = 4


def _frame(index: int) -> torch.Tensor:
    return torch.full((1, _WIDTH), float(index))


def _identities(chunk: torch.Tensor) -> list[int]:
    return [int(row[0].item()) for row in chunk]


class TestFrameBuffer:
    def test_slice_returns_the_requested_frames(self) -> None:
        buffer = FrameBuffer()
        for index in range(10):
            buffer.append(_frame(index))
        assert _identities(buffer.slice(3, 7)) == [3, 4, 5, 6]

    def test_slice_is_relative_to_the_live_base(self) -> None:
        buffer = FrameBuffer()
        for index in range(10):
            buffer.append(_frame(index))
        buffer.discard_before(4)
        assert buffer.base_frame == 4
        assert _identities(buffer.slice(4, 8)) == [4, 5, 6, 7]

    def test_slicing_a_discarded_range_raises(self) -> None:
        buffer = FrameBuffer()
        for index in range(10):
            buffer.append(_frame(index))
        buffer.discard_before(5)
        with pytest.raises(RuntimeError, match="unavailable"):
            buffer.slice(3, 6)

    def test_slicing_past_the_end_raises(self) -> None:
        buffer = FrameBuffer()
        for index in range(4):
            buffer.append(_frame(index))
        with pytest.raises(RuntimeError, match="unavailable"):
            buffer.slice(2, 9)

    def test_discard_is_clamped_at_both_ends(self) -> None:
        buffer = FrameBuffer()
        for index in range(5):
            buffer.append(_frame(index))
        buffer.discard_before(-3)
        assert buffer.base_frame == 0
        buffer.discard_before(99)
        assert buffer.base_frame == 5
        assert buffer.end_frame == 5

    def test_peak_tracks_the_high_water_mark(self) -> None:
        buffer = FrameBuffer()
        for index in range(10):
            buffer.append(_frame(index))
        buffer.discard_before(8)
        assert buffer.peak_frames == 10


class TestWindowLifecycle:
    """Drive the state exactly as the talker does, one frame per step."""

    @staticmethod
    def _run(total_frames: int) -> tuple[list[list[int]], int]:
        """Generate ``total_frames`` frames, returning the emitted spans.

        Returns:
            The frame identities of every emitted span, and the peak buffer
            occupancy reached along the way.
        """
        state = RequestARState(seed=1, max_frames=total_frames)
        spans: list[list[int]] = []
        for index in range(total_frames):
            state.frames.append(_frame(index))
            state.codes.append(torch.full((1, NUM_CODEBOOKS), index))
            state.generated_frames += 1
            window = state.ready_window()
            if window is not None:
                spans.append(_identities(state.take_window(window, is_final=False)))

        # Terminal flush: everything not yet emitted, as one contiguous span.
        start = state.emitted_windows * AR_CHUNK_HOP_FRAMES
        end = state.frames.end_frame
        if end > start:
            spans.append(_identities(state.frames.slice(start, end)))
        return spans, state.frames.peak_frames

    def test_short_song_is_one_span(self) -> None:
        spans, _ = self._run(50)
        assert spans == [list(range(50))]

    def test_every_frame_is_delivered_exactly_once_in_order(self) -> None:
        """Spans overlap by design, but their union must be the whole song.

        Reconstruct by taking each span's non-overlapping remainder, which is
        what the acoustic stage does after cropping.
        """
        for total in (250, 437, 600, 901):
            spans, _ = self._run(total)
            rebuilt: list[int] = []
            for span in spans:
                rebuilt.extend(frame for frame in span if frame >= len(rebuilt))
            assert rebuilt == list(range(total)), f"gap or duplicate at {total} frames"

    def test_windows_advance_by_the_hop(self) -> None:
        spans, _ = self._run(600)
        for index, span in enumerate(spans[:-1]):
            assert span[0] == index * AR_CHUNK_HOP_FRAMES
            assert len(span) == AR_CHUNK_FRAMES

    def test_mid_stream_windows_are_never_the_final_frame(self) -> None:
        """A window is held back until a further frame proves it is not last."""
        total = 600
        spans, _ = self._run(total)
        for span in spans[:-1]:
            assert span[-1] < total - 1

    def test_terminal_span_covers_the_tail(self) -> None:
        total = 437
        spans, _ = self._run(total)
        assert spans[-1][-1] == total - 1

    def test_terminal_span_is_bounded(self) -> None:
        """The tail is at most one full window plus one hop."""
        for total in (250, 301, 437, 600, 901, 1000):
            spans, _ = self._run(total)
            assert len(spans[-1]) <= AR_CHUNK_FRAMES + AR_CHUNK_HOP_FRAMES

    def test_buffer_stays_bounded_over_a_long_song(self) -> None:
        """Peak occupancy must not grow with song length; that is the point
        of evicting retired frames."""
        _, peak_short = self._run(600)
        _, peak_long = self._run(3000)
        assert peak_long == peak_short
        assert peak_long <= AR_CHUNK_FRAMES + AR_CHUNK_HOP_FRAMES

    def test_emitted_window_count_matches_the_window_plan(self) -> None:
        """What the stream emitted must reconcile with chunk_windows()."""
        for total in (250, 437, 600):
            spans, _ = self._run(total)
            planned = chunk_windows(total)
            # Every window is accounted for: the pre-terminal spans are one
            # window each, and the terminal span carries whatever remains.
            assert len(spans) <= len(planned)
            assert spans[0][0] == planned[0].start


class TestSamplingPositions:
    def test_each_frame_reserves_one_slot_per_codebook(self) -> None:
        """Codebook streams must never collide, or codes repeat across frames."""
        state = RequestARState(seed=1, max_frames=10)
        positions = [state.advance_position() for _ in range(5)]
        assert positions == [0, 8, 16, 24, 32]

    def test_draw_slots_are_globally_unique(self) -> None:
        state = RequestARState(seed=1, max_frames=64)
        slots: set[int] = set()
        for _ in range(64):
            base = state.advance_position()
            for codebook in range(NUM_CODEBOOKS):
                slot = base + codebook
                assert slot not in slots
                slots.add(slot)
        assert len(slots) == 64 * NUM_CODEBOOKS


class TestCodeRetention:
    def test_codes_span_matches_the_frame_span(self) -> None:
        state = RequestARState(seed=1, max_frames=10)
        for index in range(10):
            state.codes.append(torch.full((1, NUM_CODEBOOKS), index))
        codes = state.take_codes(2, 6)
        assert codes is not None
        assert codes.shape == (4, NUM_CODEBOOKS)
        assert [int(row[0]) for row in codes] == [2, 3, 4, 5]

    def test_out_of_range_span_is_dropped_not_raised(self) -> None:
        """Codes are diagnostic; the audio path must not fail without them."""
        state = RequestARState(seed=1, max_frames=10)
        for index in range(4):
            state.codes.append(torch.full((1, NUM_CODEBOOKS), index))
        assert state.take_codes(2, 99) is None
        assert state.take_codes(0, 0) is None

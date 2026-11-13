# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request AR state for MiniMax Music 3.

The talker accumulates one 32768-dimensional conditioning frame per decode
step and hands the acoustic stage overlapping 200-frame windows. Windows are
emitted one frame late on purpose: a window can only be declared non-final
once a further frame proves the song did not end at its boundary, and the
tail is flushed when the request finishes.

State is keyed by request id rather than batch row, because a row's identity
is not stable across engine steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .chunking import ChunkWindow
from .constants import AR_CHUNK_FRAMES, AR_CHUNK_HOP_FRAMES, NUM_CODEBOOKS


class FrameBuffer:
    """Rolling buffer of conditioning frames addressed by absolute frame index.

    Frames before an emitted window's hop point are dropped, so peak memory is
    bounded by the window size rather than by the length of the song.
    """

    def __init__(self) -> None:
        self.base_frame = 0
        self._frames: list[torch.Tensor] = []
        self.peak_frames = 0

    @property
    def end_frame(self) -> int:
        return self.base_frame + len(self._frames)

    def append(self, frame: torch.Tensor) -> None:
        """Append one ``[1, hidden]`` frame."""
        self._frames.append(frame)
        self.peak_frames = max(self.peak_frames, len(self._frames))

    def slice(self, start: int, end: int) -> torch.Tensor:
        """Return frames ``[start, end)`` stacked as ``[end - start, hidden]``.

        Raises:
            RuntimeError: If the range has already been discarded or has not
                been produced yet.
        """
        rel_start = start - self.base_frame
        rel_end = end - self.base_frame
        if rel_start < 0 or rel_end > len(self._frames):
            raise RuntimeError(
                f"MiniMax Music 3 frame range [{start}, {end}) is unavailable; "
                f"the buffer holds [{self.base_frame}, {self.end_frame})"
            )
        return torch.cat(self._frames[rel_start:rel_end], dim=0)

    def discard_before(self, frame: int) -> None:
        drop = min(max(frame - self.base_frame, 0), len(self._frames))
        if drop:
            del self._frames[:drop]
            self.base_frame += drop

    def clear(self) -> None:
        self._frames.clear()
        self.base_frame = 0


@dataclass
class RequestARState:
    """Everything the talker tracks for one conditioned request."""

    seed: int
    max_frames: int
    # Prompt length in tokens, used to tell a decode row apart from a row that
    # is still working through a chunked prefill.
    prompt_tokens: int = 0
    sampling_position: int = 0
    generated_frames: int = 0
    emitted_windows: int = 0
    finished: bool = False
    # The engine is told to stop one step after the song ends. Payloads are
    # built from the buffer that ``postprocess`` filled on the previous step,
    # so without that extra step the terminal span would be written and then
    # discarded with the request.
    pending_stop: bool = False
    finish_reason: str = "length"
    frames: FrameBuffer = field(default_factory=FrameBuffer)
    codes: list[torch.Tensor] = field(default_factory=list)
    codes_base_frame: int = 0
    # The previous frame's eight codes, fed back as the next step's input
    # embedding. Held on the request rather than on a batch row, because the
    # row a request occupies is not stable between steps.
    last_codes: torch.Tensor | None = None

    def advance_position(self) -> int:
        """Return this frame's base draw position and reserve the frame's slot.

        Each frame consumes ``NUM_CODEBOOKS`` positions so that the ``c0`` draw
        and the seven depth draws never share an RNG stream.
        """
        position = self.sampling_position
        self.sampling_position += NUM_CODEBOOKS
        return position

    def ready_window(self) -> ChunkWindow | None:
        """Return the next window that a lookahead frame has proven non-final."""
        start = self.emitted_windows * AR_CHUNK_HOP_FRAMES
        if self.frames.end_frame <= start + AR_CHUNK_FRAMES:
            return None
        return ChunkWindow(
            index=self.emitted_windows,
            start=start,
            end=start + AR_CHUNK_FRAMES,
            is_first=self.emitted_windows == 0,
            is_last=False,
        )

    def take_window(self, window: ChunkWindow, *, is_final: bool) -> torch.Tensor:
        """Materialize ``window`` and release the frames it retires."""
        chunk = self.frames.slice(window.start, window.end)
        self.emitted_windows = max(self.emitted_windows, window.index + 1)
        if not is_final:
            self.frames.discard_before(window.start + AR_CHUNK_HOP_FRAMES)
        return chunk

    def take_codes(self, start: int, end: int) -> torch.Tensor | None:
        """Return the sampled codes for frames ``[start, end)``, if retained.

        Codes are carried alongside the conditioning frames so a downstream
        stage or a debugging harness can recover the exact RVQ trajectory.
        They are dropped rather than raising when a span predates what is
        still buffered, since nothing in the audio path depends on them.
        """
        if not self.codes or end <= start:
            return None
        offset = self.codes_base_frame
        lo, hi = start - offset, end - offset
        if lo < 0 or hi > len(self.codes):
            return None
        return torch.cat(self.codes[lo:hi], dim=0)

    def discard_codes_before(self, frame: int) -> None:
        drop = min(max(frame - self.codes_base_frame, 0), len(self.codes))
        if drop:
            del self.codes[:drop]
            self.codes_base_frame += drop

    def release(self) -> None:
        self.frames.clear()
        self.codes.clear()


__all__ = ["FrameBuffer", "RequestARState"]

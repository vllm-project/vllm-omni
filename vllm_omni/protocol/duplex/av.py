# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Integer media clocks for the optional synchronized AV audio-event extension."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AVChunkClock:
    chunk_index: int
    frame_start: int
    frame_stop: int
    audio_start_sample: int
    audio_stop_sample: int
    fps: int = 24
    sample_rate_hz: int = 32000

    def __post_init__(self):
        values = (
            self.chunk_index,
            self.frame_start,
            self.frame_stop,
            self.audio_start_sample,
            self.audio_stop_sample,
            self.fps,
            self.sample_rate_hz,
        )
        if any(type(x) is not int for x in values) or min(values) < 0 or min(self.fps, self.sample_rate_hz) == 0:
            raise ValueError("Invalid integer AV clock")
        if self.frame_stop <= self.frame_start:
            raise ValueError("AV chunk has no video frames")

        def samples(frame):
            return (frame * self.sample_rate_hz + self.fps - 1) // self.fps

        if (self.audio_start_sample, self.audio_stop_sample) != (samples(self.frame_start), samples(self.frame_stop)):
            raise ValueError("Audio bounds do not match the video clock")

    def to_wire(self):
        return dict(
            chunk_index=self.chunk_index,
            frame_start=self.frame_start,
            frame_stop=self.frame_stop,
            audio_start_sample=self.audio_start_sample,
            audio_stop_sample=self.audio_stop_sample,
            video_time_base=[1, self.fps],
            audio_time_base=[1, self.sample_rate_hz],
        )

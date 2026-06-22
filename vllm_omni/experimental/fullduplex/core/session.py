# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class DuplexState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    RESPONDING = "responding"
    CLOSED = "closed"


@dataclass
class DuplexSessionConfig:
    input_modalities: tuple[str, ...] = ("audio",)
    output_modalities: tuple[str, ...] = ("audio",)
    sample_rate: int = 24000

    proactive: bool = False


@dataclass
class DuplexSession:
    session_id: str
    config: DuplexSessionConfig = field(default_factory=DuplexSessionConfig)
    state: DuplexState = DuplexState.IDLE

    epoch: int = 0
    response_index: int = 0

    playback_cursor: int = 0

    def begin_response(self) -> tuple[int, int]:
        self.response_index += 1
        self.state = DuplexState.RESPONDING
        return self.response_index, self.epoch

    def end_response(self) -> None:
        if self.state is DuplexState.RESPONDING:
            self.state = DuplexState.LISTENING

    def barge_in(self) -> int:
        self.epoch += 1
        self.state = DuplexState.LISTENING
        return self.epoch

    def is_stale(self, epoch: int) -> bool:
        return epoch != self.epoch

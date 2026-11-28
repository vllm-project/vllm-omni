# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    eos: bool

    @property
    def recent_start(self) -> int:
        return max(1, self.start - 12)

    @property
    def rope_shift_blocks(self) -> int:
        if self.eos and self.stop <= 47:
            return 0
        return max(0, self.start // 5 - 8)


class WindowPlanner:
    def __init__(self) -> None:
        self.available = 0
        self.position = 0
        self.next_stop = 9
        self.ended = False

    def append(self, *, eos: bool = False) -> tuple[Window, ...]:
        if self.ended or (eos and self.available < 10):
            raise ValueError("Invoker-Render requires two full proxy packets before EOS and no packets after EOS")
        self.available += 2 if eos else 5
        windows = []
        while self.position < self.available:
            stop = next(
                (boundary for boundary in (45, 70) if self.position < boundary < self.next_stop), self.next_stop
            )
            if stop > self.available:
                if not eos:
                    break
                stop = self.available
            windows.append(Window(self.position, stop, eos and stop == self.available))
            self.position = stop
            if stop == self.next_stop or stop == 45:
                self.next_stop = stop + 5
        self.ended = eos
        return tuple(windows)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fail-closed compatibility data plane for Qwen3-Omni."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm_omni.engine.duplex.plugin import DuplexDataPlane

EncodeAudio = Callable[[object, int, str, float | None], str | None]
_MAX_RETAINED_TERMINAL_IDS = 1024


@dataclass(frozen=True, slots=True)
class Qwen3OmniDataPlaneContext:
    """Context shape retained for the generic duplex plugin contract."""

    epoch: int = 0
    turn_id: int = 0
    active_response_turn_id: int | None = None
    active_response_id: str | None = None
    auto_responds: bool = False
    response_format: str = "wav"
    speed: float | None = None
    modalities: tuple[str, ...] = ()


class Qwen3OmniDataPlaneSession(DuplexDataPlane):
    """Track request terminality while rejecting unsupported native projection."""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._terminal: set[str] = set()
        self._terminal_order: deque[str] = deque()

    def begin_request(self, request_id: str) -> None:
        self._terminal.discard(request_id)
        try:
            self._terminal_order.remove(request_id)
        except ValueError:
            pass

    def is_terminal(self, request_id: str | None) -> bool:
        return request_id is None or request_id in self._terminal

    def mark_terminal(self, request_id: str) -> None:
        if request_id in self._terminal:
            return
        if len(self._terminal_order) >= _MAX_RETAINED_TERMINAL_IDS:
            self._terminal.remove(self._terminal_order.popleft())
        self._terminal_order.append(request_id)
        self._terminal.add(request_id)

    def close_stream(self, request_id: str) -> None:
        self.mark_terminal(request_id)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        del session_id
        if active_request_id is not None:
            self.mark_terminal(active_request_id)

    def project(self, result: object, *, context: object | None = None) -> Iterable[dict[str, object]]:
        del result, context
        raise RuntimeError("Qwen3-Omni uses the chat fallback; native data-plane projection is disabled")


__all__ = ["Qwen3OmniDataPlaneContext", "Qwen3OmniDataPlaneSession"]

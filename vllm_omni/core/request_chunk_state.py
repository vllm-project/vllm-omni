"""Per-request chunk state management."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkState:
    """Per-request chunk tracking state."""

    sent_chunks_ar: int = 0
    sent_chunks_generation: int = 0
    received_chunks: int = 0
    first_chunk_after_prefill: bool = True
    first_chunk_for_generation: bool = True
    upstream_finished: bool = False
    accumulated_tokens: list[int] = field(default_factory=list)
    accumulated_outputs: list[Any] = field(default_factory=list)


class RequestChunkStateManager:
    """Manages chunk state for all requests."""

    def __init__(self):
        self._states: dict[str, ChunkState] = defaultdict(ChunkState)

    def get_state(self, request_id: str) -> ChunkState:
        return self._states[request_id]

    def is_upstream_finished(self, request_id: str) -> bool:
        return self._states[request_id].upstream_finished

    def mark_upstream_finished(self, request_id: str) -> None:
        self._states[request_id].upstream_finished = True

    def increment_chunk_received(self, request_id: str) -> None:
        self._states[request_id].received_chunks += 1

    def increment_chunk_sent(self, request_id: str, generation: bool = False) -> None:
        state = self._states[request_id]
        if generation:
            state.sent_chunks_generation += 1
        else:
            state.sent_chunks_ar += 1

    def cleanup_request(self, request_id: str) -> None:
        if request_id in self._states:
            del self._states[request_id]

    def normalize_request_id(self, request_id: str) -> str:
        """Remove stage suffix from request ID if present.

        Example: "scheduler-req-123-0" -> "scheduler-req-123" if needed,
        but typically helps if IDs are modified by previous stages.
        For now, returns as is or strips suffix if standard format used.
        """
        # Implementation depends on ID format. Assuming standard for now.
        return request_id

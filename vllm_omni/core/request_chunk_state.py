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
    accumulated_data: list[Any] = field(default_factory=list)
    first_batch_skipped: bool = False


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

    def get_global_request_id(self, request_id: str) -> str:
        """Remove stage suffix from request ID if present.

        Example: Each stage gets appended with a stage specific UUID
        at the end of the request ID. This is used for intra stage operations.
        But for inter-stage comm, this has to be stripped off so that all the stages
        within same request share the same global request id
        Ex: stage-0 - chatcmpl-bk45hkj478-jhb45, stage-1 - chatcmpl-bk45hkj478-84g45
        stage-2 - chatcmpl-bk45hkj478-9gjh43
        Strip off the last part.
        """
        # If request_id has 3 parts (e.g., "chatcmpl-xxx-stageuuid"), strip the stage suffix
        if len(request_id.split("-")) == 3:
            return "-".join(request_id.split("-")[:-1])
        return request_id

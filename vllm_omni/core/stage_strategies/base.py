"""Abstract base for stage strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm_omni.core.chunk_processor import BaseChunkProcessor
    from vllm_omni.core.request_chunk_state import RequestChunkStateManager


@dataclass
class ReceiveOutput:
    """Result of receiving chunks."""

    stopped_running: set[Request]
    stopped_preempted: set[Request]
    received_count: int = 0


class StageStrategy(ABC):
    """Abstract strategy for stage-specific chunk behavior."""

    @abstractmethod
    def should_send_chunk(
        self,
        request: Request,
        pooler_output: Any,
        token_ids: list[int],
        processor: "BaseChunkProcessor",
        state_manager: "RequestChunkStateManager",
    ) -> bool:
        """Determine if chunk should be sent."""
        ...

    @abstractmethod
    def receive_chunk(
        self,
        active_requests: list[Request],
        connector: Any,
        stage_id: int,
        processor: "BaseChunkProcessor",
        state_manager: "RequestChunkStateManager",
    ) -> ReceiveOutput:
        """Handle receiving chunks for active requests."""
        ...

    def prepare_connector_key(self, chunk_id: int, stage_id: int, req_id: str) -> str:
        return f"{req_id}_{stage_id}_{chunk_id}"

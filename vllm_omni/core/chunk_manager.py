"""
This module contains the AsyncChunkManager class, which is responsible for managing
the asynchronous transmission of chunks between different stages in the vLLM-Omni
distributed pipeline.
"""

from typing import Any

from vllm.logger import init_logger

from vllm_omni.core.chunk_processor import BaseChunkProcessor
from vllm_omni.core.request_chunk_state import RequestChunkStateManager
from vllm_omni.core.stage_strategies import ReceiveOutput, StageStrategy

logger = init_logger(__name__)


class ChunkManager:
    """Chunk manager for all stage types.

    Delegates stage-specific logic to StageStrategy and model-specific
    logic to ChunkProcessorProtocol.
    """

    def __init__(
        self,
        connector: Any,
        stage_id: int,
        stage_strategy: StageStrategy,
        chunk_processor: BaseChunkProcessor,
    ):
        self.connector = connector
        self.stage_id = stage_id
        self.next_stage_id = stage_id + 1
        self.stage_strategy = stage_strategy
        self.chunk_processor = chunk_processor
        self.state_manager = RequestChunkStateManager()

    def send_chunk(
        self,
        request: Any,
        pooler_output: Any = None,
        token_ids: list[int] | None = None,
    ) -> bool:
        """Send chunk to next stage if ready."""
        if not self.stage_strategy.should_send_chunk(
            request, pooler_output, token_ids or [], self.chunk_processor, self.state_manager
        ):
            return False

        req_id = self.state_manager.get_global_request_id(request.request_id)
        state = self.state_manager.get_state(req_id)

        if self.chunk_processor.chunk_batch_size is not None:
            chunk = self.chunk_processor.accumulate_and_prepare_batch(request, state, pooler_output, token_ids)
        else:
            chunk = self.chunk_processor.prepare_outgoing_chunk(request, pooler_output, token_ids)

        if not chunk:
            return False

        chunk["last_chunk"] = request.is_finished()

        if self.chunk_processor.chunk_batch_size is not None:
            key = self.stage_strategy.prepare_connector_key(state.sent_chunks_generation, self.stage_id, req_id)
            success, _, _ = self.connector.put_chunk(str(self.stage_id), str(self.next_stage_id), key, chunk)
            if success:
                self.state_manager.increment_chunk_sent(req_id, generation=True)
        else:
            key = self.stage_strategy.prepare_connector_key(state.sent_chunks_ar, self.stage_id, req_id)
            success, _, _ = self.connector.put_chunk(str(self.stage_id), str(self.next_stage_id), key, chunk)
            if success:
                self.state_manager.increment_chunk_sent(req_id, generation=False)

        return success

    def receive_chunk(self, active_requests: list) -> ReceiveOutput:
        """Receive chunks for active requests."""
        if self.stage_id == 0:
            return ReceiveOutput(set(), set(), 0)
        return self.stage_strategy.receive_chunk(
            active_requests, self.connector, self.stage_id, self.chunk_processor, self.state_manager
        )

    def cleanup_request(self, request_id: str) -> None:
        req_id = self.state_manager.get_global_request_id(request_id)
        self.state_manager.cleanup_request(req_id)
        self.chunk_processor.on_request_complete(req_id)


def create_chunk_manager(
    connector: Any,
    stage_id: int,
    model_arch: str | None = None,
    model_stage: str | None = None,
    stage_strategy_type: str = "ar",
    chunk_processor: str | None = None,
    processor_config: dict | None = None,
) -> ChunkManager:
    """Factory to create ChunkManager from config.

    Args:
        connector: Stage connector instance
        stage_id: Current stage ID
        model_arch: Model architecture name for processor lookup
        model_stage: Model stage name (e.g., "thinker", "talker", "code2wav")
        stage_strategy_type: "ar" or "generation"
        chunk_processor: Explicit processor class path (overrides registry lookup)
        processor_config: Config dict passed to processor constructor

    Returns:
        Configured ChunkManager instance
    """
    from vllm_omni.core.chunk_registry import ChunkProcessorRegistry
    from vllm_omni.core.stage_strategies import ARStrategy, GenerationStrategy

    # Get processor from registry or explicit path
    processor = ChunkProcessorRegistry.get_processor(model_arch, model_stage, chunk_processor, processor_config)

    # Select strategy based on type
    if stage_strategy_type == "generation":
        strategy = GenerationStrategy()
    else:
        strategy = ARStrategy()

    return ChunkManager(connector, stage_id, strategy, processor)

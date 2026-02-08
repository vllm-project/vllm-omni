"""
BaseChunkProcessor

New models can inherit this base class to add async chunk streaming support
WITHOUT modifying scheduler, chunk_manager, or workers.
"""

from typing import TYPE_CHECKING, Any

from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm_omni.core.request_chunk_state import ChunkState


class BaseChunkProcessor:
    stage_type: str = "ar"
    chunk_batch_size: int | None = None
    should_skip_first_chunk: bool = True

    def is_prefill(self, pooler_output: Any, request: Request) -> bool:
        """Default: check hidden state dimension."""
        if not pooler_output or "hidden" not in pooler_output:
            return True
        return pooler_output["hidden"].shape[0] > 1

    def prepare_outgoing_chunk(
        self, request: Request, pooler_output: Any = None, new_token_ids: list[int] | None = None
    ) -> dict | None:
        """Prepare a single chunk from current output.

        For non-batched stages, this returns the chunk to send.
        For batched stages, override accumulate_and_prepare_batch instead.
        """
        return pooler_output

    def accumulate_and_prepare_batch(
        self, request: Request, state: "ChunkState", pooler_output: Any = None, new_token_ids: list[int] | None = None
    ) -> dict | None:
        """Accumulate data and return batch when ready.

        Override this for model-specific batching logic.
        Returns None if batch not ready, or the batch dict to send.
        """
        # Default implementation: no batching, just prepare and return
        return self.prepare_outgoing_chunk(request, pooler_output, new_token_ids)

    def apply_incoming_chunk(self, chunk: dict, request: Request, request_state: Any) -> None:
        """Default: store chunk on request.pending_chunk."""
        request.pending_chunk = chunk

    def map_chunk_to_worker(self, chunk: dict) -> dict:
        """Default: pass through unchanged."""
        return chunk

    def on_upstream_finished(self, request: Request) -> bool:
        """Return True if request should stop when upstream finishes.

        - AR stages: Continue running (False)
        - Generation stages: Stop request (True)
        """
        return self.stage_type == "generation"

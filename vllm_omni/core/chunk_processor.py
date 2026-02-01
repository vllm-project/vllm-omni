"""
BaseChunkProcessor

New models can inherit this base class to add async chunk streaming support
WITHOUT modifying scheduler, chunk_manager, or workers.
"""

from dataclasses import dataclass
from typing import Any

from vllm.v1.request import Request


@dataclass
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
        """Default: pass through pooler output as-is."""
        return pooler_output

    def apply_incoming_chunk(self, chunk: dict, request: Request, request_state: Any) -> None:
        """Default: store chunk on request.pending_chunk."""
        request.pending_chunk = chunk

    def map_chunk_to_worker(self, chunk: dict) -> dict:
        """Default: pass through unchanged."""
        return chunk

    def on_request_complete(self, request_id: str) -> None:
        """Override if you have per-request state to clean."""
        pass

    def on_upstream_finished(self, request: Request) -> bool:
        """Return True if request should stop when upstream finishes.

        - AR stages: Continue running (False)
        - Generation stages: Stop request (True)
        """
        return self.stage_type == "generation"

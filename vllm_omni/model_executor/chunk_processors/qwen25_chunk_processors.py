"""
Chunk processors for Qwen2.5-Omni model.

Model teams: This is the ONLY file you need to modify to add chunk support.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.request import Request

from vllm_omni.core.chunk_processor import BaseChunkProcessor

if TYPE_CHECKING:
    from vllm_omni.core.request_chunk_state import ChunkState

logger = init_logger(__name__)


def _ensure_list(x):
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    elif not isinstance(x, list):
        return x
    return list(x)


@dataclass
class Qwen25ThinkerChunkProcessor(BaseChunkProcessor):
    """Chunk processor for Qwen2.5-Omni Thinker stage.

    Sends hidden states to Talker stage incrementally during decode.
    """

    stage_type: str = "ar"
    chunk_batch_size: int | None = None
    should_skip_first_chunk: bool = True  # Orchestrator handles first

    def is_prefill(self, pooler_output: Any, request: Request) -> bool:
        if not pooler_output or "hidden" not in pooler_output:
            return True
        # if hidden dim 0 is greater than 1, then it is in prefill
        # Because decode stage generates one hidden state at a time
        return pooler_output["hidden"].shape[0] > 1

    def prepare_outgoing_chunk(
        self, request: Request, pooler_output: Any = None, new_token_ids: list[int] | None = None
    ) -> dict | None:
        """Prepare hidden states for Talker.

        For prefill: includes prompt_embeds, prompt_token_ids, thinker_output_token_ids
        For decode: includes only thinker_result
        """
        if pooler_output is None or "hidden" not in pooler_output:
            return None

        thinker_output = pooler_output["hidden"]

        # Decode mode: just send the hidden states
        return {"thinker_result": thinker_output}

    def map_chunk_to_worker(self, chunk: dict) -> dict:
        """Process chunk for worker consumption (key mapping/filtering)."""
        info = {}
        if "thinker_result" in chunk:
            info["thinker_reply_part"] = chunk["thinker_result"]
        if "last_chunk" in chunk:
            info["last_chunk"] = chunk["last_chunk"]
        return info


@dataclass
class Qwen25TalkerChunkProcessor(BaseChunkProcessor):
    """Chunk processor for Qwen2.5-Omni Talker stage.

    Batches codec tokens for Code2Wav stage.

    IMPORTANT: Qwen2.5 accumulates token IDs directly (already transformed).
    This is different from Qwen3 which accumulates raw tensors.
    """

    stage_type: str = "ar"
    chunk_batch_size: int = 36  # Code2Wav frame size
    should_skip_first_chunk: bool = True

    def is_prefill(self, pooler_output: Any, request: Request) -> bool:
        """Never in prefill for generation output."""
        return False

    def prepare_outgoing_chunk(
        self, request: Request, pooler_output: Any = None, new_token_ids: list[int] | None = None
    ) -> dict | None:
        """Not used for batched sends - use accumulate_and_prepare_batch instead."""
        return None

    def accumulate_and_prepare_batch(
        self, request: Request, state: "ChunkState", pooler_output: Any = None, new_token_ids: list[int] | None = None
    ) -> dict | None:
        """Qwen2.5-specific: Accumulate token IDs and batch.

        This matches the old _process_chunk_for_generation logic:
        1. Extend accumulated_data with new output_token_ids
        2. When batch_size reached OR request finished:
           - Send accumulated tokens
           - Clear accumulator
        3. Skip first batch if should_skip_first_chunk is True
        """
        if not new_token_ids or len(new_token_ids) == 0:
            return None

        # Accumulate token IDs in state
        state.accumulated_data.extend(new_token_ids)

        is_finished = request.is_finished()
        accumulated_count = len(state.accumulated_data)

        # Check if batch is ready
        if accumulated_count >= self.chunk_batch_size or is_finished:
            # Skip first batch if configured
            if self.should_skip_first_chunk and not state.first_batch_skipped:
                if not is_finished:  # Don't skip if it's the last chunk
                    state.first_batch_skipped = True
                    state.accumulated_data = []
                    return None

            # Get accumulated tokens
            tokens = list(state.accumulated_data)
            if not tokens:
                return None

            # Clear accumulator
            state.accumulated_data = []

            return {"tokens": tokens}

        return None

    def apply_incoming_chunk(self, chunk: dict, request: Request, request_state: Any) -> None:
        """Apply incoming chunk to request."""
        request.pending_chunk = chunk

        if request_state is not None:
            info = getattr(request_state, "additional_information_cpu", {})
            if not isinstance(info, dict):
                info = {}

            mapped_info = self.map_chunk_to_worker(chunk)
            info.update(mapped_info)

            request_state.additional_information_cpu = info

    def map_chunk_to_worker(self, chunk: dict) -> dict:
        """Process incoming chunk keys for worker."""
        info = {}
        # Talker needs thinker_result mapped to thinker_reply_part
        if "thinker_result" in chunk:
            info["thinker_reply_part"] = chunk["thinker_result"]

        # Pass through keys needed by Talker model
        for key in ["last_chunk", "prompt_embeds", "prompt_token_ids", "thinker_output_token_ids"]:
            if key in chunk:
                info[key] = chunk[key]
        return info


@dataclass
class Qwen25Code2WavChunkProcessor(BaseChunkProcessor):
    """Chunk processor for Code2Wav (receiving) stage.

    Receives batched codec tokens from Talker.
    """

    stage_type: str = "generation"
    chunk_batch_size: int | None = None
    should_skip_first_chunk: bool = False

    def is_prefill(self, pooler_output: Any, request: Request) -> bool:
        # There is not prefill or decode phase for generation
        # Generation models generate everything in single step
        return False

    def prepare_outgoing_chunk(
        self, request: Request, pooler_output: Any = None, new_token_ids: list[int] | None = None
    ) -> dict | None:
        # code2wav is the last stage in qwen2.5-omni
        return None

    def apply_incoming_chunk(self, chunk: dict, request: Request, request_state: Any) -> None:
        """Apply codec tokens to request."""
        if "tokens" in chunk:
            request.pending_chunk = chunk["tokens"]

    def map_chunk_to_worker(self, chunk: dict) -> dict:
        return {}

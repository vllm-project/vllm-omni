"""
Chunk processors for Qwen3-Omni model.

Model teams: This is the ONLY file you need to modify to add chunk support.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.request import Request

from vllm_omni.core.chunk_processor import BaseChunkProcessor

if TYPE_CHECKING:
    from vllm_omni.core.request_chunk_state import ChunkState


@dataclass
class Qwen3ThinkerChunkProcessor(BaseChunkProcessor):
    """Chunk processor for Qwen3-Omni Thinker stage.

    Qwen3 requires at least 3 output tokens before leaving prefill.
    """

    stage_type: str = "ar"
    chunk_batch_size: int | None = None
    should_skip_first_chunk: bool = True
    min_tokens_for_decode: int = 2  # Qwen3-specific requirement

    def is_prefill(self, pooler_output: Any, request: Request) -> bool:
        """Qwen3 needs at least 2 tokens to leave prefill."""
        if not hasattr(request, "output_token_ids"):
            return True
        return len(request.output_token_ids) < self.min_tokens_for_decode

    def prepare_outgoing_chunk(
        self, request: Request, pooler_output: Any = None, new_token_ids: list[int] | None = None
    ) -> dict | None:
        """Prepare embeddings and hidden states for Talker."""
        if pooler_output is None:
            return None
        return {
            "thinker_embeddings": pooler_output["0"],
            "thinker_hidden_states": pooler_output["hidden"],
            "tts_bos_embed": pooler_output["tts_bos_embed"],
            "tts_eos_embed": pooler_output["tts_eos_embed"],
            "tts_pad_embed": pooler_output["tts_pad_embed"],
        }

    def map_chunk_to_worker(self, chunk: dict) -> dict:
        """Process chunk for worker consumption (key mapping/filtering)."""
        info = {}
        keys_to_copy = [
            "thinker_embeddings",
            "thinker_hidden_states",
            "tts_bos_embed",
            "tts_eos_embed",
            "tts_pad_embed",
            "last_chunk",
        ]
        for key in keys_to_copy:
            if key in chunk:
                info[key] = chunk[key]
        return info


@dataclass
class Qwen3TalkerChunkProcessor(BaseChunkProcessor):
    """Chunk processor for Qwen3-Omni Talker stage.

    Receives embeddings from Thinker.
    Sends codec codes to Code2Wav.
    """

    stage_type: str = "ar"
    # The chunk batch size is 50 (25 for left context and 25 for current chunks)
    # We want to preserve previous 25 tokens as context for code2wav (chunked_decode_streaming)
    chunk_batch_size: int = 50
    should_skip_first_chunk: bool = False

    def is_prefill(self, pooler_output: Any, request: Request) -> bool:
        """Always false for generation output streaming."""
        return False

    def prepare_outgoing_chunk(
        self, request: Request, pooler_output: Any = None, new_token_ids: list[int] | None = None
    ) -> dict | None:
        """Not used for Qwen3 Talker - use accumulate_and_prepare_batch instead."""
        return None

    def accumulate_and_prepare_batch(
        self, request: Request, state: "ChunkState", pooler_output: Any = None, new_token_ids: list[int] | None = None
    ) -> dict | None:
        """Qwen3-specific: Accumulate raw tensors, batch, then transform."""
        if pooler_output is None:
            return None

        if "code_predictor_codes" not in pooler_output:
            return None

        raw_codes = pooler_output["code_predictor_codes"]

        if raw_codes is None or (hasattr(raw_codes, "shape") and raw_codes.shape[0] <= 0):
            return None

        # Skip prefill output - prefill produces batch with shape[0] > 1
        if hasattr(raw_codes, "shape") and raw_codes.shape[0] > 1:
            return None

        state.accumulated_data.append(raw_codes)

        is_finished = request.is_finished()
        accumulated_count = len(state.accumulated_data)

        if accumulated_count >= self.chunk_batch_size or is_finished:
            raw_codes = state.accumulated_data
            if not raw_codes:
                return None

            code_tensor = torch.cat(raw_codes, dim=0)

            codec_codes = code_tensor.to(torch.long).transpose(0, 1).cpu().reshape(-1).tolist()

            # Retain the latter half of accumulated raw_codes to serve as left context
            # for the next chunk during streaming decoding
            state.accumulated_data = state.accumulated_data[self.chunk_batch_size // 2 :]

            return {"tokens": codec_codes}

        return None

    def apply_incoming_chunk(self, chunk: dict, request: Request, request_state: Any) -> None:
        """Apply incoming chunk from Thinker."""
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
        keys_to_copy = [
            "thinker_embeddings",
            "thinker_hidden_states",
            "tts_bos_embed",
            "tts_eos_embed",
            "tts_pad_embed",
            "last_chunk",
        ]
        for key in keys_to_copy:
            if key in chunk:
                info[key] = chunk[key]
        return info


@dataclass
class Qwen3Code2WavChunkProcessor(BaseChunkProcessor):
    """Chunk processor for Qwen3-Omni Code2Wav stage.

    Receives tokens from Talker.
    """

    stage_type: str = "generation"
    chunk_batch_size: int | None = None
    should_skip_first_chunk: bool = False

    def is_prefill(self, pooler_output: Any, request: Request) -> bool:
        return False

    def prepare_outgoing_chunk(
        self, request: Request, pooler_output: Any = None, new_token_ids: list[int] | None = None
    ) -> dict | None:
        return None

    def apply_incoming_chunk(self, chunk: dict, request: Request, request_state: Any) -> None:
        """Apply codec tokens to request."""
        if "tokens" in chunk:
            request.pending_chunk = chunk["tokens"]

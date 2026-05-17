from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
)
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import TALKER_CODEC_PAD_TOKEN_ID

logger = init_logger(__name__)

# Maximum tokens supported by the code2wav stage. The flattened talker codec
# sequence fed to stage-1 must not exceed this, otherwise gpu_input_batch
# add_request will fail with a broadcast error when copying prompt_token_ids
# into token_ids_cpu. Keep in sync with the stage-1 ``max_model_len`` in
# ``vllm_omni/model_executor/stage_configs/mimo_audio.yaml`` and the offline
# example ``examples/offline_inference/mimo_audio/end2end.py``.
MAX_CODE2WAV_TOKENS = 18192


def prepend_and_flatten_colmajor(x: torch.Tensor, pad_vec: torch.Tensor) -> torch.Tensor:
    """
    Prepend a padding vector to the input tensor and flatten in column-major order.

    This function expands the padding vector to match the batch dimensions of the input
    tensor, prepends it to the row dimension, and then flattens the result in column-major
    order (transposing before flattening).

    Args:
        x: Input tensor with shape (..., R, C) where R is the row dimension and C is
            the column dimension. Example: (B, 1, 8, 4) where B is batch size.
        pad_vec: Padding vector with shape (C,) to be prepended to x. The vector will
            be expanded to match the batch dimensions of x.

    Returns:
        A flattened 1D tensor in column-major order with shape (-1,). The result
        contains the padded row followed by all rows of x, flattened column by column.
    """
    pad_row = pad_vec.view(1, -1)

    # Expand pad_row to the front of x, keeping other batch dimensions consistent
    # Example: x shape = (B,1,R,C) → pad shape = (B,1,1,C)
    pad_expand = pad_row.view(*([1] * (x.dim() - 2)), 1, x.size(-1)).expand(*x.shape[:-2], 1, x.size(-1))

    # Prepend to the row dimension
    y = torch.cat([pad_expand, x], dim=-2)  # (..., R+1, C)

    # Flatten in column-major order:
    # First transpose (..., R+1, C) → (..., C, R+1)
    # Then flatten
    y_col_major = y.permute(*range(y.dim() - 2), -1, -2).reshape(-1)

    return y_col_major


def _make_finished_sentinel() -> OmniPayloadStruct:
    """Return a minimal payload with finished=True so Stage-1 can end the request."""
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def _flush_remaining_codes(
    transfer_manager: Any,
    request_id: str,
    chunk_size: int,
    left_context_size: int,
) -> OmniPayloadStruct:
    """Flush any accumulated but unsent codes when the request finishes."""
    accumulated = transfer_manager.code_prompt_token_ids.get(request_id, [])
    if not accumulated:
        return _make_finished_sentinel()

    length = len(accumulated)
    chunk_length = length % chunk_size
    context_length = chunk_length if chunk_length != 0 else chunk_size
    end_index = min(length, left_context_size + context_length)

    # Align with qwen3_omni talker2code2wav_async_chunk: decoder strip uses explicit frame count.
    left_ctx_frames = max(0, min(length - context_length, left_context_size))
    flat_codes = torch.tensor(accumulated[-end_index:]).reshape(-1)

    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat_codes),
        meta=MetaStruct(
            left_context_size=left_ctx_frames,
            codec_chunk_frames=chunk_size,
            codec_left_context_frames=left_context_size,
            code_flat_numel=int(flat_codes.numel()),
            finished=torch.tensor(True, dtype=torch.bool),
        ),
    )


def _is_codes_empty(codes: Any) -> bool:
    """Check whether code_predictor_codes should be treated as empty / invalid."""
    if codes is None:
        return True
    if isinstance(codes, torch.Tensor):
        return codes.numel() == 0 or not codes.any()
    if hasattr(codes, "__len__") and len(codes) == 0:
        return True
    t = torch.tensor(codes, dtype=torch.long) if not isinstance(codes, torch.Tensor) else codes
    return not t.any()


def _to_code_tensor(codes: Any) -> torch.Tensor | None:
    """Convert codes to a (B, 1, 8, 4) long tensor, or return None if shape is invalid."""
    code_tensor = codes.to(torch.long) if isinstance(codes, torch.Tensor) else torch.tensor(codes, dtype=torch.long)
    if code_tensor.ndim == 3:
        code_tensor = code_tensor.unsqueeze(0)
    if code_tensor.ndim != 4 or code_tensor.shape[-2:] != (8, 4):
        return None
    return code_tensor


def llm2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """
    Async chunk version: convert stage-0 pooling_output to code2wav payload (pooling / connector accumulation).

    Accumulates codes in connector per request_id,
    returns payload only when chunk_size is full or request is finished; returns None when waiting.
    """
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 3))
    left_context_size = int(cfg.get("codec_left_context_frames", 3))

    request_id = getattr(request, "external_req_id", None)

    po_codes = pooling_output.get("codes", {})
    if "audio" not in po_codes:
        if is_finished:
            return _flush_remaining_codes(transfer_manager, request_id, chunk_size, left_context_size)
        return None

    code_predictor_codes = po_codes["audio"]
    code_tensor = _to_code_tensor(code_predictor_codes)
    if code_tensor is None:
        if is_finished:
            return _flush_remaining_codes(transfer_manager, request_id, chunk_size, left_context_size)
        return None

    pad_vec = torch.tensor([TALKER_CODEC_PAD_TOKEN_ID] * 4, device=code_tensor.device, dtype=code_tensor.dtype)
    code_list = prepend_and_flatten_colmajor(code_tensor, pad_vec).tolist()

    if sum(code_list) == 0:
        if is_finished:
            return _flush_remaining_codes(transfer_manager, request_id, chunk_size, left_context_size)
        return None

    if request_id is None:
        return None

    transfer_manager.code_prompt_token_ids[request_id].append(code_list)
    length = len(transfer_manager.code_prompt_token_ids[request_id])
    chunk_length = length % chunk_size
    if chunk_length != 0 and not is_finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size
    end_index = min(length, left_context_size + context_length)
    left_ctx_frames = max(0, min(length - context_length, left_context_size))
    flat_codes = torch.tensor(transfer_manager.code_prompt_token_ids[request_id][-end_index:]).reshape(-1).tolist()

    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(flat_codes)),
        meta=MetaStruct(
            left_context_size=left_ctx_frames,
            codec_chunk_frames=chunk_size,
            codec_left_context_frames=left_context_size,
            code_flat_numel=len(flat_codes),
            finished=torch.tensor(is_finished, dtype=torch.bool),
        ),
    )


def llm2code2wav(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.

    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage

    Args:
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    talker_outputs = source_outputs
    code2wav_inputs = []

    # Process each talker output
    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]

        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        mm = output.multimodal_output
        mm_codes = mm.get("codes", {})
        mm_hs = mm.get("hidden_states", {})
        if "audio" in mm_codes:
            codec_codes = mm_codes["audio"].to(torch.long)  # [seq_batch_size, 1, 8, 4]
            is_all_zero = (codec_codes == 0).all(dim=(1, 2, 3))
            non_zero_indices = (~is_all_zero).nonzero(as_tuple=True)[0]
            if len(non_zero_indices) == 0:
                # All codec codes are zero - skip this request with a warning
                request_id = getattr(talker_output, "request_id", f"unknown_{i}")
                logger.warning(
                    "Skipping request %s: all codec codes are zero (empty output from Stage-0). "
                    "This may indicate the model failed to generate valid audio codes.",
                    request_id,
                )
            else:
                if len(non_zero_indices) < codec_codes.shape[0]:
                    codec_codes = codec_codes[non_zero_indices]
        elif "output" in mm_hs and "audio" not in mm_codes:
            codec_codes = torch.zeros(1, 1, 8, 4, dtype=torch.long)
        else:
            raise ValueError(f"Invalid multimodal_output: {output.multimodal_output}")

        pad_vec = torch.tensor([TALKER_CODEC_PAD_TOKEN_ID] * 4)

        code_final = prepend_and_flatten_colmajor(codec_codes, pad_vec)
        code_final = code_final.tolist()

        # Guard against flattened sequences longer than code2wav's max_model_len.
        # Without this, add_request raises ``could not broadcast input array
        # from shape (N,) into shape (max_model_len,)`` and kills the engine
        # core (see issue #2683). Mirrors the offline end2end.py safeguard.
        if len(code_final) > MAX_CODE2WAV_TOKENS:
            request_id = getattr(talker_output, "request_id", f"unknown_{i}")
            logger.warning(
                "Request %s: code_final len=%d > MAX_CODE2WAV_TOKENS=%d, truncating.",
                request_id,
                len(code_final),
                MAX_CODE2WAV_TOKENS,
            )
            code_final = code_final[:MAX_CODE2WAV_TOKENS]

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=code_final,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs

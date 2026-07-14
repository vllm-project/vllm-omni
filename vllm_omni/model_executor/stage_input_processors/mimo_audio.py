from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
)
from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import TALKER_CODEC_PAD_TOKEN_ID

logger = init_logger(__name__)

# Maximum tokens supported by the code2wav stage. The flattened talker codec
# sequence fed to stage-1 must not exceed this, otherwise gpu_input_batch
# add_request will fail with a broadcast error when copying prompt_token_ids
# into token_ids_cpu. Keep in sync with the stage-1 ``max_model_len`` in
# ``vllm_omni/model_executor/stage_configs/mimo_audio.yaml`` and the offline
# example ``examples/offline_inference/mimo_audio/end2end.py``.
MAX_CODE2WAV_TOKENS = 18192

# Minimum safe values for codec streaming parameters.
# codec_left_context_frames must cover the vocoder attention window
# (vocoder_attn_window_size defaults to [40, 10]).  Values below the minimum
# cause acoustic-state resets at chunk boundaries, producing voice instability
# (multiple speakers / timbre shifts in the output audio).
_MIN_CODEC_CHUNK_FRAMES = 3
_MIN_CODEC_LEFT_CONTEXT_FRAMES = 40
_DEFAULT_CODEC_CHUNK_FRAMES = 10
_DEFAULT_CODEC_LEFT_CONTEXT_FRAMES = 40


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
    # When the accumulated length aligns with chunk_size boundary (remainder == 0),
    # we still need to flush the final chunk with full context to give the vocoder
    # enough attention window — otherwise the tail audio cuts off and produces
    # voice instability. Fall back to chunk_size as the context length.
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
    multimodal_output: OmniPayload | dict[str, Any],
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """
    Async chunk version: convert stage-0 multimodal_output to code2wav payload (pooling / connector accumulation).

    Accumulates codes in connector per request_id,
    returns payload only when chunk_size is full or request is finished; returns None when waiting.
    """
    # Null guard: chunk_transfer_adapter calls this every emit step
    # including no-output steps where multimodal_output is None.
    if multimodal_output is None or not isinstance(multimodal_output, dict):
        if is_finished:
            connector = getattr(transfer_manager, "connector", None)
            raw_cfg = getattr(connector, "config", {}) or {}
            cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
            chunk_size = int(cfg.get("codec_chunk_frames", 3))
            left_context_size = int(cfg.get("codec_left_context_frames", 3))
            request_id = getattr(request, "external_req_id", None)
            return _flush_remaining_codes(transfer_manager, request_id, chunk_size, left_context_size)
        return None
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", _DEFAULT_CODEC_CHUNK_FRAMES))
    if chunk_size < _MIN_CODEC_CHUNK_FRAMES:
        logger.warning(
            "codec_chunk_frames=%d is below minimum %d; falling back to %d.",
            chunk_size,
            _MIN_CODEC_CHUNK_FRAMES,
            _DEFAULT_CODEC_CHUNK_FRAMES,
        )
        chunk_size = _DEFAULT_CODEC_CHUNK_FRAMES

    left_context_size = int(cfg.get("codec_left_context_frames", _DEFAULT_CODEC_LEFT_CONTEXT_FRAMES))
    if left_context_size < _MIN_CODEC_LEFT_CONTEXT_FRAMES:
        logger.warning(
            "codec_left_context_frames=%d is below minimum %d (must cover vocoder attention window); "
            "falling back to %d to prevent voice instability.",
            left_context_size,
            _MIN_CODEC_LEFT_CONTEXT_FRAMES,
            _DEFAULT_CODEC_LEFT_CONTEXT_FRAMES,
        )
        left_context_size = _DEFAULT_CODEC_LEFT_CONTEXT_FRAMES

    request_id = getattr(request, "external_req_id", None)

    # Text-only paths (e.g. modalities=["text"]) yield no codec pooling output;
    # stage-0 still drives the chunk transfer adapter, so treat None as "no codes
    # this step" rather than letting `.get()` raise AttributeError.
    po_codes = multimodal_output.get("codes", {}) if multimodal_output is not None else {}
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


# ============================================================================
# Worker-connector data plane (non-async-chunk path).
# AR runner's `flatten_payload` converts the model emit
# `multimodal_outputs={"codes": {"audio": ...}}` to flat
# `pooling_output["codes.audio"]` before the full-payload accumulator runs, so default
# CONCAT semantics build the full codec tensor across all decode steps.
# ============================================================================

# Per-model REPLACE-keys for the full-payload accumulator.  mimo_audio's
# producer side emits per-step codec frames that should be CONCAT'd across
# steps (not REPLACE'd), so this stays empty.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset()


def _filter_zero_codec_rows(codec_codes: torch.Tensor) -> torch.Tensor:
    """Drop zero-padded codec rows from a 4-D `[N, 1, 8, 4]` tensor.

    Shared by the sync placeholder builder (``llm2code2wav_token_only``) and
    the full-payload producer (``llm2code2wav_full_payload``) so both size the
    downstream codec sequence off the same non-zero frames.
    """
    if codec_codes.ndim != 4 or codec_codes.numel() == 0:
        return codec_codes
    is_all_zero = (codec_codes == 0).all(dim=(1, 2, 3))
    nonzero_idx = (~is_all_zero).nonzero(as_tuple=True)[0]
    if len(nonzero_idx) == 0:
        # All rows are zero-padded; return an empty tensor so the caller
        # can detect this via numel()==0 and skip the request.
        return codec_codes[:0]
    if len(nonzero_idx) < codec_codes.shape[0]:
        return codec_codes[nonzero_idx]
    return codec_codes


def llm2code2wav_token_only(
    source_outputs: list,
    _prompt=None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Sync-side placeholder for the non-async-chunk Stage-1 (code2wav) input.

    Returns an ``OmniTokensPrompt`` sized to the orchestrator-shape codec
    length so the consumer runtime allocates the right number of slots.
    The actual codec ids are delivered via the worker connector payload
    built by ``llm2code2wav_full_payload``.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    code2wav_inputs: list = []
    for output_wrapper in source_outputs:
        out = output_wrapper.outputs[0]
        mm = out.multimodal_output if hasattr(out, "multimodal_output") else None
        mm = mm if isinstance(mm, dict) else {}
        mm_codes = mm.get("codes", {}) if isinstance(mm, dict) else {}
        prompt_len = 0
        if isinstance(mm_codes, dict) and "audio" in mm_codes:
            audio = mm_codes["audio"]
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                audio = audio.to(torch.long)
                audio = _filter_zero_codec_rows(audio)
                # +B*4 per batch row for the prepended pad_vec (see prepend_and_flatten_colmajor)
                batch_size = int(audio.shape[0]) if audio.ndim >= 1 else 1
                prompt_len = int(audio.numel()) + batch_size * 4
        if prompt_len > MAX_CODE2WAV_TOKENS:
            prompt_len = MAX_CODE2WAV_TOKENS
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def llm2code2wav_full_payload(
    transfer_manager,
    pooling_output: dict,
    request,
) -> dict | None:
    """Producer-side payload builder for the worker connector data plane.

    AR runner's ``flatten_payload`` converts the per-step model emit
    ``{"codes": {"audio": ...}}`` to ``pooling_output["codes.audio"]``.
    The accumulator CONCATs per-step tensors along dim 0, so by flush
    time this holds the full ``[total_steps, 1, 8, 4]`` codec tensor.

    A back-compat fallback to nested ``pooling_output["codes"]["audio"]``
    is kept in case a future runtime path bypasses `flatten_payload`.
    """
    del transfer_manager
    rid = getattr(request, "request_id", "?")
    if not isinstance(pooling_output, dict):
        logger.warning(
            "mimo_audio.llm2code2wav_full_payload: pooling_output not a dict "
            "(type=%s) for req=%s; consumer wait gate may hang.",
            type(pooling_output).__name__,
            rid,
        )
        return None
    codec_codes = pooling_output.get("codes.audio")
    if codec_codes is None:
        # Back-compat fallback for un-flattened pooler emits.
        codes = pooling_output.get("codes")
        if isinstance(codes, dict):
            codec_codes = codes.get("audio")
    if not isinstance(codec_codes, torch.Tensor) or codec_codes.numel() == 0:
        logger.warning(
            "mimo_audio.llm2code2wav_full_payload: missing/empty codes.audio "
            "(keys=%s) for req=%s; consumer wait gate may hang.",
            list(pooling_output.keys()),
            rid,
        )
        return None
    codec_codes = codec_codes.to(torch.long)
    codec_codes = _filter_zero_codec_rows(codec_codes)
    if codec_codes.numel() == 0:
        logger.warning(
            "mimo_audio.llm2code2wav_full_payload: codec_codes empty after _filter_zero_codec_rows for req=%s.",
            rid,
        )
        return None

    pad_vec = torch.tensor([TALKER_CODEC_PAD_TOKEN_ID] * 4)
    code_final = prepend_and_flatten_colmajor(codec_codes, pad_vec).tolist()
    if len(code_final) > MAX_CODE2WAV_TOKENS:
        code_final = code_final[:MAX_CODE2WAV_TOKENS]

    return {
        "codes": {"audio": code_final},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }

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

logger = init_logger(__name__)


def generator2tokenizer(
    source_outputs: list[Any],
    _prompt: OmniTokensPrompt | TextPrompt = None,
    _requires_multimodal_data: bool = False,
):
    tokenizer_inputs = []
    for generator_output in source_outputs:
        output = generator_output.outputs[0]
        audio_tokens = torch.cat(output.multimodal_output["audio"], dim=-1).flatten().detach().cpu().tolist()
        tokenizer_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_tokens,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return tokenizer_inputs


def _extract_last_frame(pooling_output: OmniPayload) -> torch.Tensor | None:
    audio = pooling_output.get("audio")
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        return None
    return audio.flatten()


def generator2tokenizer_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            codec_codes = frame.cpu().tolist()
            transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
    elif not finished:
        # Some steps may not produce pooling_output. Only flush on finish.
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    chunk_size_at_begin = int(cfg.get("codec_chunk_frames_at_begin", 5))
    left_context_size = int(cfg.get("codec_left_context_frames", 25))
    if chunk_size <= 0 or left_context_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size}"
        )
    length = len(transfer_manager.code_prompt_token_ids[request_id])

    # Avoid emitting empty chunks during normal streaming. If the request is
    # finished and nothing was produced, emit an EOF marker.
    if length <= 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    # Use a small chunk size at begin
    if length <= chunk_size:
        chunk_size = chunk_size_at_begin

    chunk_length = length % chunk_size

    if chunk_length != 0 and not finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size
    end_index = min(length, left_context_size + context_length)
    ctx_frames = max(0, int(end_index - context_length))
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Pack context + chunk into codebook-major flat codes for adapter.
    code_predictor_codes = torch.tensor(window_frames).reshape(-1).tolist()

    return OmniPayloadStruct(
        codes=CodesStruct(
            audio=torch.tensor(
                [int(ctx_frames), int(context_length)] + code_predictor_codes,
                dtype=torch.long,
            ),
        ),
        meta=MetaStruct(finished=torch.tensor(finished, dtype=torch.bool)),
    )

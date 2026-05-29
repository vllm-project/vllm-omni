# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for Kimi-Audio thinker -> code2wav handoff."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
    _validate_stage_inputs,
)

logger = init_logger(__name__)

# Mirrors KimiAudioCode2Wav.KIMIA_TOKEN_OFFSET.
KIMIA_TOKEN_OFFSET = 152064


def _extract_audio_tokens(
    output_token_ids: list[int] | torch.Tensor,
    multimodal_output: dict[str, Any] | None,
) -> list[int]:
    """Prefer the explicit audio_tokens payload; fall back to filtering
    the token-id stream for IDs >= KIMIA_TOKEN_OFFSET. The MIMO head
    samples full-vocab (so msg_end/media_end can signal EOD) and emits
    ``kimia_text_blank`` during the audio-delay lag — filter those out
    before handing tokens to code2wav, matching upstream's
    ``t >= kimia_token_offset`` filter in kimia.py."""
    if multimodal_output and "audio_tokens" in multimodal_output:
        codes = multimodal_output["audio_tokens"]
        if isinstance(codes, torch.Tensor):
            flat = codes.reshape(-1).to(torch.long).tolist()
        else:
            flat = list(codes)
        return [int(t) for t in flat if int(t) >= KIMIA_TOKEN_OFFSET]

    if isinstance(output_token_ids, torch.Tensor):
        flat = output_token_ids.reshape(-1).tolist()
    else:
        flat = list(output_token_ids)
    return [t for t in flat if t >= KIMIA_TOKEN_OFFSET]


def kimi2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Sync handoff: collect every audio token, forward to code2wav as
    ``prompt_token_ids`` (still in global-vocab space)."""
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        audio_token_ids = _extract_audio_tokens(
            output_token_ids=getattr(output, "token_ids", None) or [],
            multimodal_output=getattr(output, "multimodal_output", None),
        )

        if not audio_token_ids:
            request_id = getattr(thinker_output, "request_id", f"unknown_{i}")
            logger.warning(
                "Skipping request %s: thinker emitted no audio tokens. "
                "Either the user prompt requested text-only or the audio "
                "head fired below threshold. Code2Wav will receive empty.",
                request_id,
            )
            audio_token_ids = []

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_token_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs


def kimi2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Streaming handoff: buffer per-request audio tokens, flush a chunk
    of ``codec_chunk_frames`` when enough arrive or at request end. Returns
    ``None`` to mean "wait for more tokens"."""
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    audio_tokens_obj = pooling_output.get("audio_tokens") if pooling_output else None
    new_tokens: list[int]
    if audio_tokens_obj is not None:
        if isinstance(audio_tokens_obj, torch.Tensor):
            raw = audio_tokens_obj.reshape(-1).to(torch.long).tolist()
        else:
            raw = [int(t) for t in audio_tokens_obj]
        # Drop kimia_text_blank (delay lag) and msg_end/media_end EOD —
        # only real codec tokens belong in the code2wav input buffer.
        new_tokens = [int(t) for t in raw if int(t) >= KIMIA_TOKEN_OFFSET]
    else:
        step_token_ids = pooling_output.get("token_ids") or [] if pooling_output else []
        if isinstance(step_token_ids, torch.Tensor):
            step_token_ids = step_token_ids.reshape(-1).tolist()
        new_tokens = [int(t) for t in step_token_ids if int(t) >= KIMIA_TOKEN_OFFSET]

    request_id = request.external_req_id
    buffer = transfer_manager.code_prompt_token_ids[request_id]
    if new_tokens:
        buffer.extend(new_tokens)

    length = len(buffer)
    if length == 0:
        if is_finished:
            return {
                "code_predictor_codes": [],
                "left_context_size": 0,
                "finished": torch.tensor(True, dtype=torch.bool),
            }
        return None

    chunk_remainder = length % chunk_size
    if chunk_remainder != 0 and not is_finished:
        return None

    context_length = chunk_remainder if chunk_remainder != 0 else chunk_size
    left_context_size = max(0, min(length - context_length, left_context_size_config))
    end_index = min(length, left_context_size + context_length)
    codes = buffer[-end_index:]

    return {
        "code_predictor_codes": list(codes),
        "left_context_size": left_context_size,
        "codec_chunk_frames": chunk_size,
        "codec_left_context_frames": left_context_size_config,
        "finished": torch.tensor(is_finished, dtype=torch.bool),
    }

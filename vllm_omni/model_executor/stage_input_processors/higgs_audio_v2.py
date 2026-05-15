# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-input processor for higgs-audio v2: Talker -> Code2Wav.

Mirrors the qwen3_tts contract:

- ``talker2code2wav`` is the sync (non-async-chunk) adapter that runs once,
  after Stage 0 has finished, collecting every audio frame into a single
  ``OmniTokensPrompt`` for Stage 1.
- ``talker2code2wav_async_chunk`` is the streaming adapter invoked once per
  emitted codec frame; it accumulates frames in a per-request buffer and
  flushes an ``OmniPayloadStruct`` once a chunk is ready (or at end-of-stream).

Stage 1 strictly rejects any code id >= ``audio_stream_bos_id`` (1024). This
processor filters frames that still carry stream specials so the codec sees
only real codes in [0, 1023], regardless of where Stage 0 emitted them.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
    to_dict,
)

__all__ = ["talker2code2wav", "talker2code2wav_async_chunk"]

logger = init_logger(__name__)

# Default codec chunk-shape knobs for async_chunk mode.  These match the
# canonical Qwen3-TTS defaults (25 frames = 1 s at 25 fps).  Real values
# come from the connector config; these are only fallbacks.
_DEFAULT_CODEC_CHUNK_FRAMES = 25
_DEFAULT_CODEC_LEFT_CONTEXT_FRAMES = 25

# Boson-ai constants. Kept local (rather than imported from the model package)
# so this module remains usable without circular imports during
# stage-input-processor discovery.
_NUM_CODEBOOKS = 8
_AUDIO_STREAM_BOS_ID = 1024
_AUDIO_STREAM_EOS_ID = 1025
_NUM_REAL_CODES = _AUDIO_STREAM_BOS_ID  # codes in [0, 1023] are real


def _filter_real_code_frames(audio_codes: torch.Tensor) -> torch.Tensor:
    """Keep only frames whose codes are entirely in [0, 1023].

    ``audio_codes`` has shape ``[num_frames, num_codebooks]`` after the
    standard Stage-0 transpose. We drop frames containing stream specials
    or anything padded outside the real-code range.
    """
    if audio_codes.numel() == 0:
        return audio_codes
    if audio_codes.ndim != 2:
        raise ValueError(
            f"expected [num_frames, num_codebooks] audio_codes; got shape {tuple(audio_codes.shape)}"
        )
    valid = (audio_codes >= 0).all(dim=1) & (audio_codes < _NUM_REAL_CODES).all(dim=1)
    return audio_codes[valid]


def talker2code2wav(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Sync: collect all talker codes, then pass to code2wav at once."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    code2wav_inputs: list[OmniTokensPrompt] = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output
        mm_codes = mm.get("codes", {})

        audio_codes = mm_codes.get("audio")
        if audio_codes is None or not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
            # Nothing to decode for this request; emit an empty payload so the
            # downstream stage can still close the response.
            code2wav_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                    additional_information=None,
                )
            )
            continue

        audio_codes = audio_codes.to(torch.long)
        if audio_codes.ndim == 1:
            # Stage 0 emitted a flat sequence; reshape to [num_frames, num_codebooks].
            if audio_codes.numel() % _NUM_CODEBOOKS != 0:
                raise ValueError(
                    f"flat audio_codes length {audio_codes.numel()} not divisible by num_codebooks={_NUM_CODEBOOKS}"
                )
            audio_codes = audio_codes.reshape(-1, _NUM_CODEBOOKS)

        if audio_codes.ndim != 2:
            raise ValueError(
                f"audio_codes must be 1D or 2D; got shape {tuple(audio_codes.shape)}"
            )

        audio_codes = _filter_real_code_frames(audio_codes)
        # Code2Wav expects codebook-major flat: [Q * num_frames].
        codec_codes = audio_codes.transpose(0, 1).cpu().reshape(-1).tolist()

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )
    return code2wav_inputs


def _extract_last_frame(pooling_output: OmniPayload) -> torch.Tensor | None:
    audio_codes = pooling_output.get("codes", {}).get("audio")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
    elif audio_codes.ndim == 1:
        frame = audio_codes
    else:
        raise ValueError(
            f"unexpected audio_codes shape for higgs_audio_v2 async_chunk: {tuple(audio_codes.shape)}"
        )
    if frame.numel() == 0:
        return None
    frame = frame.to(torch.long).reshape(-1)
    # Filter frames that still carry stream specials.
    if int(frame.min().item()) < 0 or int(frame.max().item()) >= _NUM_REAL_CODES:
        return None
    return frame


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Streaming adapter: buffer frames per request and flush a chunk at a time.

    Emits codes in codebook-major flat layout (``[Q * num_frames]``) plus a
    ``MetaStruct(left_context_size, finished)`` so Stage 1 can stitch chunks
    together with the conventional left-context overlap.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            transfer_manager.code_prompt_token_ids[request_id].append(frame.cpu().tolist())
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", _DEFAULT_CODEC_CHUNK_FRAMES))
    left_context_size_config = int(
        cfg.get("codec_left_context_frames", _DEFAULT_CODEC_LEFT_CONTEXT_FRAMES)
    )

    if chunk_size <= 0 or left_context_size_config < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}"
        )

    length = len(transfer_manager.code_prompt_token_ids[request_id])
    if length <= 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    chunk_remainder = length % chunk_size
    if not finished and chunk_remainder != 0:
        # Wait until we have a clean chunk boundary.
        return None

    context_length = chunk_remainder if (finished and chunk_remainder > 0) else chunk_size
    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, end_index - context_length)
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    if not window_frames:
        # Defensive: should not happen because length > 0, but keeps the contract.
        return OmniPayloadStruct(
            codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
            meta=MetaStruct(finished=torch.tensor(finished, dtype=torch.bool)),
        )

    num_codebooks = len(window_frames[0])
    if num_codebooks != _NUM_CODEBOOKS:
        raise ValueError(
            f"expected {_NUM_CODEBOOKS} codebooks per frame; got {num_codebooks}"
        )
    num_frames = len(window_frames)
    flat = torch.tensor(
        [window_frames[f][q] for q in range(num_codebooks) for f in range(num_frames)],
        dtype=torch.long,
    )

    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
    )

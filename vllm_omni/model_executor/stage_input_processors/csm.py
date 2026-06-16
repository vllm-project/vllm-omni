"""Stage input processor for CSM-1B: backbone (Stage 0) -> Mimi (Stage 1).

Bridges the forward-flowing latent (32-code frames) Stage 0 emits in
``codes.audio`` to the flat CODEBOOK-MAJOR ``[32 * F]`` ``input_ids`` Stage 1's
Mimi vocoder consumes (A3 design §4.4). Two modes, picked by
``deploy.async_chunk`` (``stage_config._select_processor_funcs``):

  * ``backbone2mimi`` (sync / non-async): collect all of a request's frames once
    it finishes, trim invalid/EOS frames, flatten codebook-major.
  * ``backbone2mimi_async_chunk``: window frames into chunks (+ left context) as
    Stage 0 streams, for low-latency streaming audio.

Modeled on ``stage_input_processors/qwen3_tts.py`` (talker2code2wav*), with
CSM specifics: 32 codebooks; CSM-tuned valid mask (frames with any nonzero code
and all codes < 2048 -- excludes the all-zero EOS frame and reserved codec ids);
no ref_code / voice-cloning plumbing.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
)

logger = init_logger(__name__)

_NUM_CODEBOOKS = 32
_CODEBOOK_SIZE = 2048  # codes >= 2048 are reserved ids, never valid frames


def backbone2mimi(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: collect all Stage-0 frames, then decode in Stage 1 at once."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    backbone_outputs = source_outputs
    mimi_inputs: list[OmniTokensPrompt] = []
    for _i, backbone_output in enumerate(backbone_outputs):
        if not backbone_output.finished:
            # Non-async decode runs once, after Stage 0 has accumulated the frames.
            continue
        output = backbone_output.outputs[0]
        mm = output.multimodal_output
        mm_codes = mm.get("codes", {})

        # audio_codes: [num_frames, 32] (frame-major), accumulated across steps.
        audio_codes = mm_codes.get("audio")
        if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
            mimi_inputs.append(OmniTokensPrompt(prompt_token_ids=[], multi_modal_data=None))
            continue
        audio_codes = audio_codes.to(torch.long)
        if audio_codes.ndim == 1:
            audio_codes = audio_codes.reshape(-1, _NUM_CODEBOOKS)

        # CSM valid mask: drop zero-padded / all-zero EOS frames and frames with
        # out-of-range (reserved) codes that must not reach Mimi.
        valid_mask = audio_codes.any(dim=1) & (audio_codes.max(dim=1).values < _CODEBOOK_SIZE)
        audio_codes = audio_codes[valid_mask]

        # Mimi expects codebook-major flat [32 * num_frames].
        codec_codes = audio_codes.transpose(0, 1).cpu().reshape(-1).tolist()
        mimi_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )
    return mimi_inputs


def _extract_last_frame(pooling_output: OmniPayload) -> torch.Tensor | None:
    audio_codes = pooling_output.get("codes", {}).get("audio")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None  # all-zero EOS frame -> nothing to decode
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for CSM async_chunk: {tuple(audio_codes.shape)}")


def backbone2mimi_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Window Stage-0 frames into Mimi decode chunks (+ left context).

    Mirrors ``qwen3_tts.talker2code2wav_async_chunk`` minus ref_code; CSM has 32
    codebooks and an all-zero-frame EOS (filtered by ``_extract_last_frame``).
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
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)

    if chunk_size <= 0 or left_context_size_config < 0 or initial_chunk_size < 0:
        raise ValueError(
            f"Invalid CSM codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}"
        )
    if initial_chunk_size > chunk_size:
        logger.warning(
            "initial_codec_chunk_frames=%d > codec_chunk_frames=%d, clamping.",
            initial_chunk_size,
            chunk_size,
        )
        initial_chunk_size = chunk_size

    length = len(transfer_manager.code_prompt_token_ids[request_id])
    if length <= 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    use_first_chunk = 0 < initial_chunk_size < chunk_size
    if use_first_chunk and length <= initial_chunk_size:
        if not finished and length < initial_chunk_size:
            return None
        context_length = length if finished and length < initial_chunk_size else initial_chunk_size
    else:
        initial_coverage = initial_chunk_size if use_first_chunk else 0
        adjusted = length - initial_coverage
        if not finished and adjusted % chunk_size != 0:
            return None
        chunk_length = adjusted % chunk_size
        context_length = chunk_length if chunk_length != 0 else chunk_size

    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, end_index - context_length)
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    num_quantizers = len(window_frames[0])
    num_frames = len(window_frames)
    # Codebook-major flat [Q * num_frames].
    codec_codes = torch.tensor(
        [window_frames[f][qi] for qi in range(num_quantizers) for f in range(num_frames)],
        dtype=torch.long,
    )

    return OmniPayloadStruct(
        codes=CodesStruct(audio=codec_codes),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
    )

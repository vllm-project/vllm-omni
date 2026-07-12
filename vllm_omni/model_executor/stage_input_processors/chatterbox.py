"""Stage input processor for Chatterbox TTS: T3 -> S3Gen.

Handles buffering of T3 speech tokens and sending chunks to S3Gen.
Chatterbox uses a single codebook (not multi-quantizer), so each token
is a scalar rather than a frame vector.

Fixes over PR #1517:
- Filters EOS/SOS tokens (>= 6561) before buffering
- Forwards ref_dict (speaker embedding) to S3Gen via the payload speaker field
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
)

logger = logging.getLogger(__name__)

# Valid speech tokens are [0, 6561); SOS=6561, EOS=6562
SPEECH_VOCAB_SIZE = 6561


def _get_ref_audio_from_request(request: Any) -> str | None:
    """Extract ref_audio path from the original request's additional_information.

    The additional_information on the request is an AdditionalInformationPayload
    with entries that can be tensors, lists, or scalars.  The example script
    passes ``{"ref_audio": ["/path/to/ref.wav"]}`` which becomes a list entry.
    """
    ai = getattr(request, "additional_information", None)
    if ai is None:
        return None
    entries = getattr(ai, "entries", None)
    if not isinstance(entries, dict):
        return None
    entry = entries.get("ref_audio")
    if entry is None:
        return None
    # List form: e.g. ["/path/to/ref.wav"]
    if getattr(entry, "list_data", None) is not None:
        data = entry.list_data
        if isinstance(data, list) and data:
            return str(data[0])
    # Scalar form: e.g. "/path/to/ref.wav"
    if getattr(entry, "scalar_data", None) is not None:
        return str(entry.scalar_data)
    return None


def _extract_speech_token(multimodal_output: Mapping[str, Any]) -> int | None:
    """Extract the latest speech token from T3's multimodal output.

    Returns None if no valid speech token is found (e.g., EOS/SOS/prefill placeholder).
    """
    speech_tokens = multimodal_output.get("speech_tokens")
    if not isinstance(speech_tokens, torch.Tensor) or speech_tokens.numel() == 0:
        return None
    token = int(speech_tokens.reshape(-1)[-1].item())
    # Filter out EOS/SOS/OOV tokens — only valid speech tokens pass through.
    if token >= SPEECH_VOCAB_SIZE:
        return None
    return token


def t3_to_s3gen_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Buffer T3 speech tokens and send chunks to S3Gen.

    Follows the same pattern as ``qwen3_tts.talker2code2wav_async_chunk``.
    Chatterbox uses a single codebook (not multi-quantizer), so each token
    is a scalar rather than a frame vector.

    Chunks are sent when ``chunk_size`` tokens accumulate (25 tokens = 1 second
    at 25 tokens/sec).  Each chunk includes ``left_context_size`` overlap tokens.

    The emitted ``codes.audio`` tensor is ``[ctx_frames, *speech_tokens]`` —
    the chunk adapter turns it into S3Gen's ``prompt_token_ids`` verbatim.
    ``speaker`` carries the reference-audio path for voice cloning.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if isinstance(multimodal_output, Mapping):
        token = _extract_speech_token(multimodal_output)
        if token is not None:
            # Store as single-element list for compatibility with the
            # code_prompt_token_ids interface.
            transfer_manager.code_prompt_token_ids[request_id].append([token])
    elif not finished:
        # Flush calls without a payload only matter once the request is done.
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size = int(cfg.get("codec_left_context_frames", 25))
    if chunk_size <= 0 or left_context_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size}"
        )

    # Forward ref_audio_path from the original request to S3Gen.
    # multimodal_output only passes tensors, so we read the ref audio path
    # from the request's additional_information instead.
    if not hasattr(transfer_manager, "_chatterbox_ref_audio_paths"):
        transfer_manager._chatterbox_ref_audio_paths = {}
    if request_id not in transfer_manager._chatterbox_ref_audio_paths:
        ref_audio_path = _get_ref_audio_from_request(request)
        if ref_audio_path is not None:
            logger.debug("Stage processor: extracted ref_audio_path=%s for request %s", ref_audio_path, request_id)
            transfer_manager._chatterbox_ref_audio_paths[request_id] = ref_audio_path

    length = len(transfer_manager.code_prompt_token_ids[request_id])
    chunk_length = length % chunk_size

    if chunk_length != 0 and not finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size

    if length <= 0:
        if finished:
            # Empty-but-finished payload so the Stage-1 wait gate releases
            # (S3Gen handles empty codec input -> 0-sample audio) rather than
            # the connector silently dropping the request.
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    end_index = min(length, left_context_size + context_length)
    ctx_frames = max(0, int(end_index - context_length))
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Flatten: each element is [token], so just take the scalar.
    flat_tokens = [frame[0] for frame in window_frames]

    # S3Gen's expected input_ids layout: [ctx_frames, *flat_tokens]
    payload = OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor([int(ctx_frames)] + flat_tokens, dtype=torch.long)),
        request_id=request_id,
    )

    # Attach the ref audio path so S3Gen can build its ref_dict for speaker
    # conditioning. OmniPayloadStruct forbids unknown fields; ``speaker`` is
    # the designated voice-identity channel (same as qwen3_tts).
    cached_path = transfer_manager._chatterbox_ref_audio_paths.get(request_id)
    if cached_path is not None:
        payload.speaker = cached_path
        logger.debug("Stage processor: attaching ref_audio_path=%s to chunk for request %s", cached_path, request_id)

    return payload

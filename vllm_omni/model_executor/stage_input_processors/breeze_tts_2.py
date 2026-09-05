"""Stage-0 to stage-1 input conversion for Breeze-TTS-2.

The prompt builder and the reference-audio tokenizer live under the Breeze
model directory.  This module intentionally contains only the inter-stage
contract: flatten the generated ``(T, Q)`` codec matrix into the codebook-major
token sequence consumed by the stage-1 codec model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)

# ``BreezeTTS2TalkerForGeneration`` emits the complete sequence accumulated
# so far on every decode step.  The generic full-payload accumulator normally
# concatenates rank-2 tensors, which would duplicate all earlier frames.
# Replace the latest complete sequence instead.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset({"codes.audio"})
_DEFAULT_CODEC_CHUNK_FRAMES = 8


def _extract_audio_codes(output: Any) -> torch.Tensor | None:
    """Extract ``codes.audio`` from the supported Omni output shapes."""
    # Depending on the engine path, source_outputs contains either the
    # per-request output itself or an EngineCoreOutput with ``outputs[0]``.
    if isinstance(output, Mapping):
        multimodal = output.get("multimodal_output", output.get("multimodal_outputs"))
        if multimodal is None and "codes" in output:
            multimodal = output
        if multimodal is None and "codes.audio" in output:
            audio = output["codes.audio"]
            if isinstance(audio, (list, tuple)):
                if len(audio) != 1:
                    raise ValueError(f"expected one Breeze audio-code tensor, got {len(audio)}")
                audio = audio[0]
            audio = torch.as_tensor(audio)
            if audio.ndim not in (1, 2):
                raise ValueError(f"Breeze audio codes must be rank 1 or 2, got {tuple(audio.shape)}")
            return audio.to(device="cpu", dtype=torch.long).contiguous()
    else:
        multimodal = getattr(output, "multimodal_output", None)
        if multimodal is None:
            multimodal = getattr(output, "multimodal_outputs", None)
        if multimodal is None:
            candidates = getattr(output, "outputs", None)
            if isinstance(candidates, Sequence) and candidates:
                return _extract_audio_codes(candidates[0])
    if not isinstance(multimodal, Mapping):
        return None
    codes = multimodal.get("codes")
    audio = codes.get("audio") if isinstance(codes, Mapping) else multimodal.get("codes.audio")
    if audio is None:
        return None
    # ``OmniOutput`` stores one tensor per request under ``codes.audio``.
    # ``process_engine_inputs`` invokes this function once per source output,
    # so unwrap that single-item container before tensor conversion.
    if isinstance(audio, (list, tuple)):
        if len(audio) != 1:
            raise ValueError(f"expected one Breeze audio-code tensor, got {len(audio)}")
        audio = audio[0]
    audio = torch.as_tensor(audio)
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim not in (1, 2):
        raise ValueError(
            "Breeze audio codes must have shape (T, Q) or codebook-major (Q*T,), "
            f"got {tuple(audio.shape)}"
        )
    return audio.to(device="cpu", dtype=torch.long).contiguous()


def talker2codec(
    source_outputs: Sequence[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build synchronous stage-1 prompts from completed stage-0 outputs.

    Breeze's Mimi decoder receives one flat codebook-major sequence.  The
    generated matrix is therefore transposed from ``(T, Q)`` to ``(Q, T)``
    before flattening.  No audio decoding, text tokenization, or streaming
    state is handled here.
    """
    del prompt, _requires_multimodal_data
    results: list[OmniTokensPrompt] = []
    for source_index, output in enumerate(source_outputs):
        codes = _extract_audio_codes(output)
        if codes is None or codes.numel() == 0:
            logger.warning("Breeze stage output %d contains no audio codes", source_index)
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue
        # Connector full-payload builders send a codebook-major flat tensor;
        # direct in-process stage outputs carry frame-major ``(T, Q)``.
        if codes.ndim == 1:
            codebook_major_data = codes.contiguous()
            codebook_major = codebook_major_data
        else:
            codebook_major_data = codes.transpose(0, 1).contiguous()
            codebook_major = codebook_major_data.reshape(-1)
        results.append(
            OmniTokensPrompt(
                prompt_token_ids=codebook_major.tolist(),
                multi_modal_data={"codes": {"audio": codebook_major_data}},
            )
        )
    return results


def talker2codec_full_payload(
    transfer_manager: Any = None,
    pooling_output: Mapping[str, Any] | None = None,
    request: Any = None,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Build the final Stage-1 payload from accumulated talker codes.

    ``pooling_output`` is the AR runner's flattened per-request payload. Mimi
    consumes codebook-major IDs, so the frame-major talker matrix is transposed
    before flattening.
    """
    del transfer_manager, request, is_finished
    if not isinstance(pooling_output, Mapping):
        return None
    audio = pooling_output.get("codes.audio")
    if audio is None:
        return None
    codes = torch.as_tensor(audio, dtype=torch.long)
    if codes.numel() == 0:
        return None
    if codes.ndim == 2:
        flat = codes.transpose(0, 1).contiguous().reshape(-1)
    elif codes.ndim == 1:
        flat = codes.contiguous()
    else:
        raise ValueError(f"Breeze audio codes must be rank 1 or 2, got {tuple(codes.shape)}")
    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(code_flat_numel=int(flat.numel())),
    )


def _extract_current_audio_frame(
    multimodal_output: Mapping[str, Any] | None,
) -> torch.Tensor | None:
    """Extract the single ``(Q,)`` frame emitted by async Stage 0."""
    if not isinstance(multimodal_output, Mapping):
        return None
    codes = multimodal_output.get("codes")
    audio = codes.get("audio") if isinstance(codes, Mapping) else None
    if audio is None:
        return None
    frame = torch.as_tensor(audio, dtype=torch.long).reshape(-1)
    if frame.numel() == 0:
        return None
    return frame.cpu().contiguous()


def _codec_chunk_frames(transfer_manager: Any) -> int:
    connector = getattr(transfer_manager, "connector", None)
    raw_config = getattr(connector, "config", {}) or {}
    config = raw_config.get("extra", {}) if isinstance(raw_config, Mapping) else {}
    value = config.get("breeze_codec_chunk_frames")
    try:
        chunk_frames = int(value) if value is not None else _DEFAULT_CODEC_CHUNK_FRAMES
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Breeze codec_chunk_frames={value!r}") from exc
    if chunk_frames <= 0:
        raise ValueError(f"Breeze codec_chunk_frames must be positive, got {chunk_frames}")
    return chunk_frames


def talker2codec_async_chunk(
    transfer_manager: Any,
    multimodal_output: Mapping[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Batch Stage-0 frames and send only the un-emitted codebook-major tail."""
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    frame = _extract_current_audio_frame(multimodal_output)
    if frame is not None:
        transfer_manager.code_prompt_token_ids[request_id].append(frame)

    pending = transfer_manager.code_prompt_token_ids[request_id]
    chunk_frames = _codec_chunk_frames(transfer_manager)
    if not pending and not finished:
        return None
    if len(pending) < chunk_frames and not finished:
        return None

    if pending:
        # Stage 0 supplies frame-major ``(T, Q)`` rows. The Qwen3-TTS decoder
        # consumes ``(Q, T)``, so transpose before flattening the emitted tail.
        tail = torch.stack(pending, dim=0)
        flat = tail.transpose(0, 1).contiguous().reshape(-1)
    else:
        flat = torch.empty(0, dtype=torch.long)
    pending.clear()

    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(
            request_id=request_id,
            code_flat_numel=int(flat.numel()),
            codec_streaming=True,
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
    )


__all__ = ["talker2codec", "talker2codec_async_chunk", "talker2codec_full_payload"]

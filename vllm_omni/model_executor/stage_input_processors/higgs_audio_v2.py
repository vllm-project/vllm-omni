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
)
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    emit_threshold,
    stack_frames,
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


def _real_code_frame_mask(frames: torch.Tensor) -> torch.Tensor:
    """Per-frame keep mask: every code in ``[0, _NUM_REAL_CODES)``.

    ``frames`` has shape ``[num_frames, num_codebooks]``. This is the single
    definition of "carries no stream specials", shared by the sync filter below
    and the batched async-chunk resolve; the async path used to spell it as a
    scalar ``frame.min() < 0 or frame.max() >= _NUM_REAL_CODES`` per frame.
    """
    return (frames >= 0).all(dim=1) & (frames < _NUM_REAL_CODES).all(dim=1)


def _filter_real_code_frames(audio_codes: torch.Tensor) -> torch.Tensor:
    """Keep only frames whose codes are entirely in [0, 1023].

    ``audio_codes`` has shape ``[num_frames, num_codebooks]`` after the
    standard Stage-0 transpose. We drop frames containing stream specials
    or anything padded outside the real-code range.
    """
    if audio_codes.numel() == 0:
        return audio_codes
    if audio_codes.ndim != 2:
        raise ValueError(f"expected [num_frames, num_codebooks] audio_codes; got shape {tuple(audio_codes.shape)}")
    return audio_codes[_real_code_frame_mask(audio_codes)]


def _revert_delay_pattern(audio_codes_qt: torch.Tensor) -> torch.Tensor:
    """Reverse the delay-pattern shift, mirroring upstream's
    ``boson_multimodal.model.higgs_audio.utils.revert_delay_pattern``.

    Input shape: ``[num_codebooks, seq_len + num_codebooks - 1]``.
    Output shape: ``[num_codebooks, seq_len]``.

    For each codebook ``i``, slice rows ``[i:i+1]`` from columns
    ``[i : seq_len + i]`` to remove ``i`` leading BOS pads and ``Q-1-i``
    trailing PAD entries.
    """
    if audio_codes_qt.ndim != 2:
        raise ValueError(f"_revert_delay_pattern expects [Q, T] input; got {tuple(audio_codes_qt.shape)}")
    q, t = audio_codes_qt.shape
    if t < q:
        # Not enough frames to revert delay pattern; return as-is.
        return audio_codes_qt
    seq_len = t - q + 1
    out_l = []
    for i in range(q):
        out_l.append(audio_codes_qt[i : i + 1, i : seq_len + i])
    return torch.cat(out_l, dim=0)


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
            raise ValueError(f"audio_codes must be 1D or 2D; got shape {tuple(audio_codes.shape)}")

        # Audio_codes from Stage-0 are emitted in DELAY-PATTERN layout.
        # Revert the shift to recover the canonical ``[Q, seq_len]`` form,
        # then clip to the real-code range and trim the first/last frame
        # (the codec doesn't consume the BOS/EOS edge frames). This mirrors
        # the upstream ``HiggsAudioServeEngine.generate`` decode line:
        #     vq_code = revert_delay_pattern(out).clip(0, real-1)[:, 1:-1]
        codes_qt = audio_codes.transpose(0, 1).contiguous().cpu()
        codes_qt = _revert_delay_pattern(codes_qt)
        codes_qt = codes_qt.clamp_(min=0, max=_NUM_REAL_CODES - 1)
        if codes_qt.shape[-1] >= 3:
            codes_qt = codes_qt[:, 1:-1]
        # Code2Wav expects codebook-major flat: [Q * num_frames].
        codec_codes = codes_qt.reshape(-1).tolist()

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
        raise ValueError(f"unexpected audio_codes shape for higgs_audio_v2 async_chunk: {tuple(audio_codes.shape)}")
    if frame.numel() == 0:
        return None
    # Deliberately no content inspection. The stream-special test that used to
    # live here cost two `.item()` device syncs on every AR step; it is batched
    # into `_commit_pending` below, which applies the same predicate to a whole
    # pending batch at once.
    return frame.to(torch.long).reshape(-1)


def _commit_pending(committed: list, pending: list) -> None:
    """Move real frames from ``pending`` into ``committed``, dropping specials.

    One `.tolist()` for the whole pending batch, so the filter costs a single
    device sync per resolve rather than two per AR step. Filtering happens here
    rather than at emit so ``len(committed)`` keeps meaning "real frames so
    far": it drives the ``length % chunk_size`` gate below, and counting
    specials would move every chunk boundary.
    """
    if not pending:
        return
    keep = _real_code_frame_mask(stack_frames(pending)).tolist()
    committed.extend(frame for frame, ok in zip(pending, keep) if ok)
    pending.clear()


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | None,
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
    pooling_output = multimodal_output

    pending_frames = getattr(transfer_manager, "pending_frames", None)
    if pending_frames is None:
        pending_frames = {}
        transfer_manager.pending_frames = pending_frames

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            # On-device and unclassified; the stream-special test and the host
            # copy both happen once per resolve below.
            pending_frames.setdefault(request_id, []).append(frame)
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", _DEFAULT_CODEC_CHUNK_FRAMES))
    left_context_size_config = int(cfg.get("codec_left_context_frames", _DEFAULT_CODEC_LEFT_CONTEXT_FRAMES))

    if chunk_size <= 0 or left_context_size_config < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}"
        )

    # Resolve pending frames only when they could change the emission decision:
    # on finish, or when committing all of them could reach the next chunk
    # boundary. Below that no sync happens, which is what turns a per-step
    # device sync into roughly one per chunk. This gate has no ramp and no
    # initial-chunk phase, so the shared threshold reduces to the next multiple
    # of chunk_size -- the same rule as the `length % chunk_size` test below.
    committed = transfer_manager.code_prompt_token_ids[request_id]
    pending = pending_frames.get(request_id)
    if pending and (
        finished
        or len(committed) + len(pending)
        >= emit_threshold(
            len(committed),
            ramp=None,
            chunk_index=0,
            chunk_size=chunk_size,
            initial_chunk_size=0,
        )
    ):
        _commit_pending(committed, pending)

    length = len(committed)
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
        raise ValueError(f"expected {_NUM_CODEBOOKS} codebooks per frame; got {num_codebooks}")
    # Committed frames are device tensors and already free of stream specials,
    # so the codebook-major flatten is one transpose instead of a Python double
    # loop over num_codebooks x num_frames, with a single host transfer.
    flat = stack_frames(window_frames).t().reshape(-1).to(device="cpu", dtype=torch.long)

    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
    )

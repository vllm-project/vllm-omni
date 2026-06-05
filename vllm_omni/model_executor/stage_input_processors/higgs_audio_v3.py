# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-input processor for higgs-audio v3: Talker -> Code2Wav.

Two adapters:

* ``talker2code2wav`` is the sync (non-async_chunk) path that runs once,
  after Stage 0 has finished, collecting every audio frame into a single
  ``OmniTokensPrompt`` for Stage 1.
* ``talker2code2wav_async_chunk`` is the streaming adapter invoked once
  per emitted talker step. It accumulates raw delay-pattern rows per
  request and flushes a chunk to Stage 1 each time enough rows are
  available to recover ``chunk_size`` de-delayed frames (with ``Q-1``
  lookahead). Stage 1 sees clean ``[Q, num_frames]`` codes; the
  delay-pattern reversal stays on the orchestrator side.

Key difference from v2: BOC/EOC filtering happens AFTER delay pattern reversal
to avoid corrupting valid tail content.
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

__all__ = ["talker2code2wav", "talker2code2wav_async_chunk"]

logger = init_logger(__name__)

_NUM_CODEBOOKS = 8
_NUM_REAL_CODES = 1024  # codes in [0, 1023] are real

# Default codec chunk-shape knobs used by ``talker2code2wav_async_chunk``
# when the connector config does not override them. 25 frames at 25 fps is
# ~1 s of audio per chunk; left context of 25 frames keeps the decoder's
# sliding-window history populated across chunk boundaries.
_DEFAULT_CODEC_CHUNK_FRAMES = 25
_DEFAULT_CODEC_LEFT_CONTEXT_FRAMES = 25


def _revert_delay_pattern(audio_codes_qt: torch.Tensor) -> torch.Tensor:
    """Reverse the MusicGen-style delay pattern.

    Input shape: [num_codebooks, seq_len + num_codebooks - 1].
    Output shape: [num_codebooks, seq_len].

    For each codebook i, extract delayed[i, i : i + seq_len] to remove
    the i leading BOC pads and Q-1-i trailing EOC entries.
    """
    if audio_codes_qt.ndim != 2:
        raise ValueError(f"_revert_delay_pattern expects [Q, T] input; got {tuple(audio_codes_qt.shape)}")
    q, t = audio_codes_qt.shape
    if q != _NUM_CODEBOOKS:
        raise ValueError(f"Expected exactly {_NUM_CODEBOOKS} codebook rows, got {q}. Input shape: [{q}, {t}]")
    if t < q:
        raise ValueError(f"Not enough frames to revert delay pattern: T={t} < Q={q}")
    seq_len = t - q + 1
    out_l = []
    for i in range(q):
        out_l.append(audio_codes_qt[i : i + 1, i : seq_len + i])
    return torch.cat(out_l, dim=0)


def _filter_real_code_frames(audio_codes_qt: torch.Tensor) -> torch.Tensor:
    """Keep only frames where ALL codebook values are in [0, 1023].

    Input shape: [num_codebooks, num_frames].
    Called AFTER delay pattern reversal.
    """
    if audio_codes_qt.numel() == 0:
        return audio_codes_qt
    # Transpose to [num_frames, num_codebooks] for per-frame filtering
    frames = audio_codes_qt.t()
    valid = (frames >= 0).all(dim=1) & (frames < _NUM_REAL_CODES).all(dim=1)
    return frames[valid].t().contiguous()


def talker2code2wav(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Sync: collect all talker codes, revert delay pattern, filter, pass to code2wav."""
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
            if audio_codes.numel() % _NUM_CODEBOOKS != 0:
                raise ValueError(
                    f"flat audio_codes length {audio_codes.numel()} not divisible by num_codebooks={_NUM_CODEBOOKS}"
                )
            audio_codes = audio_codes.reshape(-1, _NUM_CODEBOOKS)

        if audio_codes.ndim != 2:
            raise ValueError(f"audio_codes must be 1D or 2D; got shape {tuple(audio_codes.shape)}")

        # Validate codebook count
        if audio_codes.shape[1] != _NUM_CODEBOOKS:
            raise ValueError(
                f"Expected {_NUM_CODEBOOKS} codebooks per frame, "
                f"got {audio_codes.shape[1]}. Audio codes shape: {tuple(audio_codes.shape)}"
            )

        # Transpose to [Q, T] for delay pattern reversal
        codes_qt = audio_codes.transpose(0, 1).contiguous().cpu()

        # Step 1: Revert delay pattern
        codes_qt = _revert_delay_pattern(codes_qt)

        # Step 2: Replace out-of-range codes (BOC=1024, EOC=1025, -1) with 0.
        # Must use torch.where, NOT clamp: clamp(max=1023) turns 1025→1023
        # which is a valid codec code and decodes to audio artifacts.
        # Matches sglang's: torch.where(codes >= codec_vocab, 0, codes)
        codes_qt = torch.where(
            (codes_qt >= _NUM_REAL_CODES) | (codes_qt < 0),
            torch.zeros_like(codes_qt),
            codes_qt,
        )

        # Step 3: Trim the last frame. After de-delay, the final frame
        # contains residual ramp-down codes (EOC→0 substituted) that
        # decode to a brief noise artifact at the end of the audio.
        if codes_qt.shape[-1] >= 2:
            codes_qt = codes_qt[:, :-1]

        if codes_qt.numel() == 0:
            code2wav_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                    additional_information=None,
                )
            )
            continue

        # Code2Wav expects codebook-major flat: [Q * num_frames]
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


# ---------------------------------------------------------------------------
# Async-chunk streaming adapter.
# ---------------------------------------------------------------------------


def _extract_last_step_row(pooling_output: OmniPayload) -> torch.Tensor | None:
    """Return the talker's emit row for this AR step as a flat ``[Q]`` tensor.

    The talker's postprocess emits ``codes.audio`` as either ``[1, Q]`` (a
    single new audio row) or ``[N, Q]`` for multi-active stages, but in the
    async-chunk path ``pooling_output`` is already request-scoped so the
    typical shape is ``[1, Q]`` (or ``[Q]`` if upstream squeezes the slot
    dim). Returns ``None`` if there is no row to append at this step.
    """
    codes = pooling_output.get("codes", {}) if isinstance(pooling_output, dict) else {}
    audio_codes = codes.get("audio") if isinstance(codes, dict) else None
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        row = audio_codes[-1]
    elif audio_codes.ndim == 1:
        row = audio_codes
    else:
        raise ValueError(f"unexpected audio_codes shape for higgs_audio_v3 async_chunk: {tuple(audio_codes.shape)}")
    if row.numel() != _NUM_CODEBOOKS:
        raise ValueError(
            f"talker emit row has {row.numel()} codebooks; expected {_NUM_CODEBOOKS} for higgs_audio_v3 async_chunk."
        )
    return row.to(torch.long).reshape(-1)


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Streaming adapter: per-step accumulator + de-delay flush.

    The talker emits raw delay-pattern rows one per AR step. With ``Q=8``
    codebooks the first complete de-delayed frame requires ``Q`` rows
    (the leading ``Q-1`` rows have BOC pads in trailing codebooks);
    every subsequent row adds exactly one de-delayed frame. We keep a
    per-request rolling buffer of raw rows and flush a chunk to Stage 1
    once enough rows are available to recover ``chunk_size`` de-delayed
    frames plus ``Q-1`` lookahead. Left context is appended at the front
    of each chunk so Stage 1's sliding-window codec keeps continuity.

    Stage 1 sees clean ``[Q, num_frames]`` codes in codebook-major flat
    layout - the same shape it would see in the sync path - plus a
    ``MetaStruct(left_context_size, finished)`` so it can trim the
    context overlap before stitching audio.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    # Per-request rolling state. ``code_prompt_token_ids`` is the
    # framework-provided per-request list; ``higgs_v3_emitted_frames``
    # tracks how many de-delayed frames we have already flushed for this
    # request so we know where the next chunk starts.
    emitted_frames = getattr(transfer_manager, "higgs_v3_emitted_frames", None)
    if emitted_frames is None:
        emitted_frames = {}
        transfer_manager.higgs_v3_emitted_frames = emitted_frames

    if isinstance(pooling_output, dict):
        row = _extract_last_step_row(pooling_output)
        if row is not None:
            transfer_manager.code_prompt_token_ids[request_id].append(row.cpu().tolist())
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", _DEFAULT_CODEC_CHUNK_FRAMES))
    left_context_size_config = int(cfg.get("codec_left_context_frames", _DEFAULT_CODEC_LEFT_CONTEXT_FRAMES))
    configured_initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)

    if chunk_size <= 0 or left_context_size_config < 0 or configured_initial_chunk_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={configured_initial_chunk_size}"
        )

    n_rows = len(transfer_manager.code_prompt_token_ids[request_id])
    de_delayed_available = max(0, n_rows - (_NUM_CODEBOOKS - 1))
    emitted = int(emitted_frames.get(request_id, 0))
    pending = de_delayed_available - emitted

    # First-chunk fast path for TTFA: emit a small initial chunk if the
    # connector asks for one. After the first emit we revert to the
    # configured ``chunk_size`` for steady-state throughput.
    if emitted == 0 and 0 < configured_initial_chunk_size < chunk_size:
        target_chunk = configured_initial_chunk_size
    else:
        target_chunk = chunk_size

    if not finished and pending < target_chunk:
        return None

    if pending <= 0:
        if finished:
            # Free per-request state; ``code_prompt_token_ids`` is owned
            # by the connector and cleared elsewhere on request finish.
            emitted_frames.pop(request_id, None)
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    actual_chunk = pending if finished and pending < target_chunk else target_chunk
    desired_left_context = min(left_context_size_config, emitted)

    # Window in AR-row space: include left-context tail behind the current
    # cursor, plus ``actual_chunk + Q - 1`` rows ahead so de-delay yields
    # ``desired_left_context + actual_chunk`` de-delayed frames.
    window_start = emitted - desired_left_context
    window_end = emitted + actual_chunk + (_NUM_CODEBOOKS - 1)
    if window_end > n_rows:
        if not finished:
            # Defensive guard - we already checked ``pending`` above; this
            # should be unreachable. Drop the chunk silently if it ever
            # fires so we do not over-emit at the boundary.
            return None
        window_end = n_rows

    window_rows = transfer_manager.code_prompt_token_ids[request_id][window_start:window_end]
    if len(window_rows) < _NUM_CODEBOOKS:
        if finished:
            emitted_frames.pop(request_id, None)
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    codes_qt = torch.tensor(window_rows, dtype=torch.long).t().contiguous()
    de_delayed = _revert_delay_pattern(codes_qt)
    de_delayed = torch.where(
        (de_delayed >= _NUM_REAL_CODES) | (de_delayed < 0),
        torch.zeros_like(de_delayed),
        de_delayed,
    )

    # On the final chunk, trim the trailing residual frame (the last
    # de-delayed frame still carries EOS-substituted codes from ramp-down,
    # which decode to a brief artifact - matches the sync-path trim at
    # line 128-129 of this file).
    if finished and window_end == n_rows and de_delayed.shape[-1] >= 2:
        de_delayed = de_delayed[:, :-1]
        actual_chunk = max(actual_chunk - 1, 0)

    codec_codes = de_delayed.reshape(-1)
    left_context_emitted = desired_left_context

    emitted_frames[request_id] = emitted + actual_chunk
    if finished:
        emitted_frames.pop(request_id, None)

    meta = MetaStruct(
        left_context_size=left_context_emitted,
        finished=torch.tensor(finished, dtype=torch.bool),
    )
    return OmniPayloadStruct(
        codes=CodesStruct(audio=codec_codes),
        meta=meta,
    )

# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Stage input processors: MOSS-TTS talker (Stage 0) → codec (Stage 1)."""

from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TokensPrompt as OmniTokensPrompt
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_audio_codes(stage_output: Any) -> torch.Tensor | None:
    """Pull audio codes from a Stage-0 OmniOutput or raw tensor."""
    if stage_output is None:
        return None

    # OmniOutput
    mm = getattr(stage_output, "multimodal_outputs", None)
    if mm is not None:
        codes_dict = mm.get("codes", {})
        if isinstance(codes_dict, dict):
            ac = codes_dict.get("audio")
            if isinstance(ac, torch.Tensor):
                return ac

    return None


# ---------------------------------------------------------------------------
# Non-streaming (sync): called once after Stage 0 finishes
# ---------------------------------------------------------------------------


def talker2codec(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Convert all talker codes to a single Stage-1 token sequence.

    Stage 0 output contains ``codes["audio"]`` shaped ``(T, NQ)`` where T is
    the number of generated audio frames and NQ is n_vq.  We flatten to
    ``[NQ * T]`` as the Stage-1 ``input_ids`` so the codec can reshape back
    to ``(NQ, T)`` for decoding.
    """
    results: list[Any] = []

    for src_idx in engine_input_source:
        if src_idx >= len(stage_list):
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue

        stage_out = stage_list[src_idx]
        audio_codes = _extract_audio_codes(stage_out)

        if audio_codes is None or audio_codes.numel() == 0:
            logger.warning("talker2codec: no audio codes in stage output %d; emitting silence.", src_idx)
            results.append(OmniTokensPrompt(prompt_token_ids=[]))
            continue

        # audio_codes: (T, NQ) → flatten to [NQ, T] → list[int]
        codes_nq_t = audio_codes.transpose(0, 1).contiguous()  # (NQ, T)
        flat = codes_nq_t.reshape(-1).tolist()

        results.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data={"codes": {"audio": codes_nq_t}},
            )
        )

    return results


# ---------------------------------------------------------------------------
# Streaming (async chunk): called each time Stage 0 emits a new chunk
# ---------------------------------------------------------------------------


def talker2codec_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Emit accumulated audio codes to Stage 1 as they arrive from Stage 0.

    State is maintained in ``transfer_manager`` keyed by request ID.
    A chunk is forwarded to Stage 1 when either:
      (a) ``is_finished`` is True (flush all remaining codes), or
      (b) the accumulated frame count reaches ``chunk_frames`` (default 25).

    Returns a dict compatible with the Stage-1 input format, or None to
    signal "not enough data yet — wait for more frames".
    """
    req_id: str = str(getattr(request, "request_id", id(request)))
    pooling_output = multimodal_output

    # Initialise per-request accumulation state
    if not hasattr(transfer_manager, "_moss_tts_state"):
        transfer_manager._moss_tts_state = {}
    state = transfer_manager._moss_tts_state

    if req_id not in state:
        state[req_id] = {
            "accumulated": None,  # (T_acc, NQ) tensor or None
            "total_emitted": 0,
        }
    req_state = state[req_id]

    # Extract new codes from this chunk. The talker emits the full per-request
    # ``audio_codes["accumulated"]`` snapshot every step, so we only append the
    # *new* tail rows (otherwise we'd duplicate history quadratically).
    # ``pooling_output`` carries the unflattened OmniPayload at the top level
    # (``codes.audio``), matching the qwen3_tts pattern.
    if pooling_output is not None:
        codes_dict = pooling_output.get("codes", {}) or {}
        snapshot = codes_dict.get("audio")
        if isinstance(snapshot, torch.Tensor) and snapshot.numel() > 0:
            snapshot_cpu = snapshot.cpu()
            prev_t = 0 if req_state["accumulated"] is None else int(req_state["accumulated"].shape[0])
            new_rows = snapshot_cpu[prev_t:]
            if new_rows.numel() > 0:
                if req_state["accumulated"] is None:
                    req_state["accumulated"] = new_rows
                else:
                    req_state["accumulated"] = torch.cat([req_state["accumulated"], new_rows], dim=0)
        # Realtime variant emits raw (un-delay-patterned) codes; record that
        # so the emit step below skips the apply_de_delay_pattern transform.
        # Realtime sets ``meta.finished`` to a 1-D bool tensor; the delay
        # variant leaves ``meta`` empty.
        meta_in = pooling_output.get("meta", {}) or {}
        flag = meta_in.get("finished")
        if isinstance(flag, torch.Tensor) and flag.numel() >= 1 and bool(flag.reshape(-1)[0].item()):
            req_state["skip_dedelay"] = True

    acc = req_state["accumulated"]
    if acc is None or acc.numel() == 0:
        if is_finished:
            del state[req_id]
        return None

    # The MOSS audio tokenizer's causal decoder doesn't yet have left-context
    # plumbing in this port, and a streaming chunk of 25 frames trips an
    # internal patched-pretransform reshape on the first chunk. Until we wire
    # left-context properly, accumulate all codes and emit only on finish.
    chunk_frames: int = 1 << 30
    left_context: int = 0

    t_acc = int(acc.shape[0])
    should_emit = is_finished or (t_acc - req_state["total_emitted"] >= chunk_frames)

    if not should_emit:
        return None

    # Determine the slice to emit
    emit_start = max(0, req_state["total_emitted"] - left_context)
    chunk_codes = acc[emit_start:]  # (T_chunk, NQ)
    req_state["total_emitted"] = t_acc

    if is_finished:
        del state[req_id]

    # Mirror upstream ``MossTTSDelayProcessor._parse_audio_codes``: the delay
    # talker samples codes in a delay pattern (a row is emitted every step, but
    # only the slot ``i == arange < audio_lengths`` carries a real code; the
    # rest is ``audio_pad_code``). Before sending to the codec we must
    #   1. de-delay   ``(T+nq-1, nq)`` → ``(T, nq)``
    #   2. drop rows that are entirely pad (separators between audio segments,
    #      and the leading text-mode rows that precede the first audio_start).
    # The realtime talker emits raw codes (no delay), so steps 1 is skipped.
    chunk_codes_long = chunk_codes.to(torch.long).cpu().contiguous()  # (T_chunk, NQ)
    nq = int(chunk_codes_long.shape[1])
    t_chunk = int(chunk_codes_long.shape[0])
    audio_pad_code = 1024  # MOSS-TTS audio_pad_code; same value across variants.

    if req_state.get("skip_dedelay"):
        de_delayed = chunk_codes_long
    elif t_chunk > nq:
        de_delayed = chunk_codes_long.new_zeros((t_chunk - nq + 1, nq))
        for i in range(nq):
            de_delayed[:, i] = chunk_codes_long[i : i + de_delayed.shape[0], i]
    else:
        de_delayed = chunk_codes_long.new_zeros((0, nq))

    if de_delayed.shape[0] > 0:
        is_pad = (de_delayed == audio_pad_code).all(dim=1)
        non_pad = ~is_pad
        if bool(non_pad.any()):
            de_delayed = de_delayed[non_pad]
        else:
            de_delayed = de_delayed.new_zeros((0, nq))

    if de_delayed.shape[0] == 0:
        # Nothing left after filtering — emit silence sentinel so the codec
        # request still completes cleanly.
        codec_flat: list[int] = []
    else:
        # Stage 1 (LLM_GENERATION codec) consumes ``codes.audio`` as a flat
        # codebook-major int list — chunk_transfer_adapter assigns it to
        # ``request.prompt_token_ids`` and the codec rebuilds the (NQ, T) grid.
        # Keep it a list (not a tensor): the receive path treats a list as the
        # token-id sequence directly, while a 1-D tensor would be ``.tolist()``-ed
        # anyway — the list keeps the downstream ``if not new_ids`` checks intact.
        codec_flat = de_delayed.transpose(0, 1).contiguous().reshape(-1).tolist()

    # main migrated the inter-stage connector from plain dicts to typed
    # ``OmniPayloadStruct`` (the chunk_transfer_adapter sender reads
    # ``payload_data.meta``); return the struct so MOSS matches that schema.
    return OmniPayloadStruct(
        codes=CodesStruct(audio=codec_flat),
        meta=MetaStruct(
            left_context_size=left_context,
            finished=torch.tensor(is_finished, dtype=torch.bool),
        ),
    )


__all__ = ["talker2codec", "talker2codec_async_chunk"]

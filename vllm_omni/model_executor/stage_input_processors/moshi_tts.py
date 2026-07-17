# SPDX-License-Identifier: Apache-2.0
"""Stage input processor for Moshi TTS: DSM Talker → Mimi Decoder.

Handles both sync (non-streaming) and async (streaming chunk) modes.

Sync path  (``moshi_tts_to_mimi``):
    Collect all audio codes produced by Stage 0 after the request finishes,
    strip the forced-zero delay-window frames, and pass the remainder to
    the Mimi decoder as a flat codebook-major token list.

Async path  (``moshi_tts_to_mimi_async_chunk``):
    Accumulate per-step audio codes and emit fixed-size chunks so the Mimi
    decoder can start decoding before Stage 0 finishes.  Moshi's ``audio_delay_steps``
    produces forced-zero frames at the start; the _extract_last_frame helper
    skips those (any() == False) so they are never forwarded to Mimi.
"""

from __future__ import annotations

from typing import Any

import torch


def _extract_last_frame(pooling_output: dict[str, Any]) -> torch.Tensor | None:
    """Extract the latest audio-code frame from a Stage-0 pooling output.

    Frames whose every codebook value is zero correspond to the DSM audio-delay
    window and are skipped — they would produce silent audio if decoded.
    """
    audio_codes = (pooling_output.get("codes") or {}).get("audio")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        if not bool(audio_codes.any().item()):
            return None
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Unexpected audio_codes shape: {tuple(audio_codes.shape)}")


# ---------------------------------------------------------------------------
# Sync (offline) path
# ---------------------------------------------------------------------------


def moshi_tts_to_mimi(
    source_outputs: list[Any],
    prompt: Any = None,
    **_: Any,
) -> list[Any]:
    """Non-async: collect all talker codes, then pass to Mimi at once.

    Called once after Stage 0 finishes.  Returns one ``OmniTokensPrompt``
    per finished output with a flat codebook-major codec token list.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    mimi_inputs: list[OmniTokensPrompt] = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output or {}
        audio_codes = (mm.get("codes") or {}).get("audio")

        if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
            mimi_inputs.append(OmniTokensPrompt(prompt_token_ids=[], multi_modal_data=None, mm_processor_kwargs=None))
            continue

        # audio_codes: [num_frames, n_q] — drop forced-zero delay-window frames.
        if audio_codes.ndim == 2:
            valid = audio_codes.any(dim=1)
            audio_codes = audio_codes[valid]
        elif audio_codes.ndim == 1:
            audio_codes = audio_codes.unsqueeze(0)

        # Flatten to codebook-major: [n_q * num_frames]
        flat = audio_codes.T.reshape(-1).cpu().tolist()
        mimi_inputs.append(OmniTokensPrompt(prompt_token_ids=flat, multi_modal_data=None, mm_processor_kwargs=None))
    return mimi_inputs


# ---------------------------------------------------------------------------
# Async (streaming) path
# ---------------------------------------------------------------------------


def moshi_tts_to_mimi_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Async streaming processor for Moshi TTS → Mimi.

    Called once per Stage-0 decode step.  Accumulates audio code frames and
    emits a chunk when ``codec_chunk_frames`` threshold is met or the request
    finishes.  Returns ``None`` when there is nothing to emit yet.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            transfer_manager.code_prompt_token_ids[request_id].append(frame.detach().cpu().to(torch.long))
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_cfg = int(cfg.get("codec_left_context_frames", 0))

    if chunk_size <= 0 or left_context_size_cfg < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_cfg}"
        )

    length = len(transfer_manager.code_prompt_token_ids[request_id])

    if length <= 0:
        if finished:
            return {"codes": {"audio": []}, "meta": {"finished": torch.tensor(True, dtype=torch.bool)}}
        return None

    chunk_length = length % chunk_size
    if chunk_length != 0 and not finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size
    end_index = min(length, left_context_size_cfg + context_length)
    left_context_size = max(0, end_index - context_length)
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Codebook-major flat list: [cb0_f0, cb0_f1, ..., cb1_f0, ...]
    code_predictor_codes = torch.stack(window_frames).T.reshape(-1).tolist()

    return {
        "codes": {"audio": code_predictor_codes},
        "meta": {
            "left_context_size": left_context_size,
            "finished": torch.tensor(finished, dtype=torch.bool),
        },
    }

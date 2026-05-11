# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage input processors for the ``nemotron_voicechat`` pipeline
(NemotronDuplexH → EarTTS), async-chunk streaming mode only.

Stage 1 (EarTTS) is pre-armed at request start by the orchestrator via
:func:`eartts_prewarm_input`, then driven step-by-step over the
shared-memory connector by :func:`nemotron2eartts_async_chunk` (the
``custom_process_next_stage_input_func`` registered on stage 0).

EarTTS payload schema:

* **Chunk 0 (prefill).** ``{"speaker_latent": (Tref, hidden) tensor}``.
  ``EarTTSForCausalLM._preprocess_prefill`` builds its prefill tensors
  from this; text positions are filled with PAD/EOS.
* **Chunk k ≥ 1 (decode #k).** ``{"input_text_tokens": [t_{k-1}]}``.
  The chunk-transfer adapter concatenates list values across chunks,
  so EarTTS sees the cumulative ``[t_0, ..., t_{k-1}]`` list at decode
  step ``k`` and indexes it at ``ear_decode_offset`` — yielding the
  expected lag-by-one cadence (EarTTS' k-th decode consumes Nemotron's
  (k-1)-th sampled token).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from vllm_omni.engine.serialization import deserialize_additional_information

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _ensure_list(x: Any) -> list[Any]:
    """Convert ConstantList / tensor-like to a Python list of ints."""
    if hasattr(x, "_x"):
        return list(x._x)
    if isinstance(x, list):
        return list(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).tolist()
    return list(x)


def _coerce_speaker_latent(value: Any) -> torch.Tensor | None:
    """Pull ``speaker_latent`` out of a prompt dict, an ``additional_information``
    dict, or a serialized ``AdditionalInformationPayload``. Returns ``None``
    for anything else (including ``None``)."""
    if isinstance(value, dict):
        # Prompt dict wraps add_info under "additional_information"; unwrap it.
        nested = value.get("additional_information")
        if isinstance(nested, dict):
            value = nested
        latent = value.get("speaker_latent")
    elif value is not None:
        info = deserialize_additional_information(value)
        latent = info.get("speaker_latent") if isinstance(info, dict) else None
    else:
        return None
    if isinstance(latent, torch.Tensor) and latent.numel() > 0 and latent.ndim >= 1:
        return latent.detach().cpu().contiguous()
    return None


def _get_async_chunk_state(transfer_manager: Any) -> dict[str, dict[str, Any]]:
    """Per-request scratch dict on the chunk transfer adapter.

    Survives across ``_send_single_request`` calls because the adapter
    is shared between the scheduler thread (``save_async``) and the
    background save_loop thread (which calls our hook). Used to cache
    ``speaker_latent`` so the chunk-0 send is robust against stage-0's
    ``request.additional_information`` being overwritten by a later
    ``StreamingInput`` chunk before save_loop dispatches.
    """
    cache = getattr(transfer_manager, "_nemotron_voicechat_state", None)
    if cache is None:
        cache = {}
        transfer_manager._nemotron_voicechat_state = cache
    return cache


# =============================================================================
# Async-chunk streaming pipeline
# =============================================================================


def eartts_prewarm_input(
    *,
    stage_id: int,
    stage0_request: Any,
    original_prompt: Any,
) -> dict[str, Any] | None:
    """``prewarm_input_func`` for stage 1 (EarTTS).

    The default prewarm arms downstream stages with stage-0's prompt
    length; EarTTS' prefill length is independent
    (``Tref = speaker_latent.shape[0]``), so we override here. Returns
    a placeholder prompt of length ``Tref`` plus a copy of
    ``speaker_latent``. The chunk transfer adapter replaces
    ``additional_information`` with the first received chunk's payload
    before stage 1 runs prefill, so the placeholder is effectively
    unused — populated for well-formedness only.

    Returns ``None`` if no ``speaker_latent`` is on the prompt, falling
    back to the default placeholder; EarTTS prefill will then raise.
    """
    speaker_latent = _coerce_speaker_latent(original_prompt)
    if speaker_latent is None:
        logger.warning(
            "[nemotron_voicechat.eartts_prewarm_input] no ``speaker_latent`` on "
            "original prompt; falling back to default placeholder."
        )
        return None

    return {
        "prompt_token_ids": [0] * int(speaker_latent.shape[0]),
        "additional_information": {"speaker_latent": speaker_latent},
        "multi_modal_data": None,
        "mm_processor_kwargs": None,
    }


def nemotron2eartts_async_chunk(
    *,
    transfer_manager: Any,
    pooling_output: Any,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Stage-0 chunk-transfer-adapter sender hook.

    Invoked by ``OmniChunkTransferAdapter._send_single_request`` once
    per stage-0 sampling step (Nemotron runs with ``max_tokens=1`` per
    ``StreamingInput`` chunk).

    Per-step payload:

    * **Chunk 0.** ``{"speaker_latent": ..., "finished": ...}``.
    * **Chunk k ≥ 1.** ``{"input_text_tokens": [t_{k-1}], "finished": ...}``.
      The chunk transfer adapter's ``_update_request_payload``
      concatenates list values across chunks, so EarTTS receives the
      cumulative ``[t_0, ..., t_{k-1}]`` list each step.

    Source of ``t_{k-1}``: stage 0 runs with ``max_tokens=1`` driven by
    per-step ``StreamingInput`` chunks, so vLLM's
    ``Scheduler._update_request_as_session`` extends the session prompt
    each step (and clears ``output_token_ids``). At chunk ``k`` we have
    ``prompt_token_ids = [0]*T_PREFILL + [t_0, ..., t_{k-1}]``, so the
    needed token sits at ``prompt_token_ids[-1]``.

    Race caveat: the live ``request`` may be mutated by the next
    ``StreamingInput`` chunk before save_loop dispatches the current
    task. The race-detection assert below catches this; a per-step
    snapshot in the chunk transfer adapter would be the fix.
    """
    external_req_id = getattr(request, "external_req_id", None)
    if external_req_id is None:
        external_req_id = getattr(request, "request_id", None)
    if external_req_id is None:
        logger.error(
            "[nemotron_voicechat.nemotron2eartts_async_chunk] request has no "
            "external_req_id / request_id; cannot route chunk."
        )
        return None

    chunk_id = int(transfer_manager.put_req_chunk[external_req_id])
    finished_tensor = torch.tensor(bool(is_finished), dtype=torch.bool)
    state_cache = _get_async_chunk_state(transfer_manager)
    req_state = state_cache.setdefault(external_req_id, {})

    if chunk_id == 0:
        speaker_latent = req_state.get("speaker_latent")
        if speaker_latent is None:
            speaker_latent = _coerce_speaker_latent(getattr(request, "additional_information", None))
            if speaker_latent is not None:
                req_state["speaker_latent"] = speaker_latent
        if speaker_latent is None:
            logger.error(
                "[nemotron_voicechat.nemotron2eartts_async_chunk] chunk-0 "
                "send for req=%s missing ``speaker_latent``; EarTTS prefill "
                "will fail.",
                external_req_id,
            )
            return None
        # Capture T_PREFILL for the race-detection invariant on decode chunks.
        prefill_len = len(_ensure_list(getattr(request, "prompt_token_ids", [])))
        if prefill_len > 0:
            req_state["t_prefill"] = prefill_len
        return {
            "speaker_latent": speaker_latent,
            "finished": finished_tensor,
        }

    prompt_tokens = _ensure_list(getattr(request, "prompt_token_ids", []))
    if not prompt_tokens:
        logger.warning(
            "[nemotron_voicechat.nemotron2eartts_async_chunk] req=%s "
            "chunk_id=%s has empty prompt_token_ids; skipping.",
            external_req_id,
            chunk_id,
        )
        return None

    # At chunk k we expect ``len(prompt) == T_PREFILL + k``; a mismatch
    # means a follow-up StreamingInput landed before save_loop dispatched.
    t_prefill = req_state.get("t_prefill")
    if t_prefill is not None:
        expected_len = t_prefill + chunk_id
        actual_len = len(prompt_tokens)
        assert actual_len == expected_len, (
            "[nemotron_voicechat.nemotron2eartts_async_chunk] save_async "
            f"↔ save_loop race detected for req={external_req_id} "
            f"chunk_id={chunk_id}: expected len(prompt_token_ids) == "
            f"{expected_len} (T_PREFILL={t_prefill} + chunk_id), got "
            f"{actual_len}."
        )

    new_text_token = int(prompt_tokens[-1])

    payload: dict[str, Any] = {
        "input_text_tokens": [new_text_token],
        "finished": finished_tensor,
    }

    # Terminal chunk: also forward the LATEST sampled token so EarTTS
    # gets the final t_k before stopping.
    if is_finished:
        output_tokens = _ensure_list(getattr(request, "output_token_ids", []))
        if not output_tokens:
            logger.warning(
                "[nemotron_voicechat.nemotron2eartts_async_chunk] req=%s "
                "terminal chunk has empty output_token_ids; sending only "
                "previous token.",
                external_req_id,
            )
        else:
            last_text_token = int(output_tokens[-1])
            payload["input_text_tokens"] = [new_text_token, last_text_token]
        state_cache.pop(external_req_id, None)

    return payload

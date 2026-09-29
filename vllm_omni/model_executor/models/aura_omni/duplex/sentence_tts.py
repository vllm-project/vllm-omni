# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage1 sentence handoff onto Talker. AURA-only; the orchestrator just calls it."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.engine.duplex.plugin import PartialStageForward

logger = init_logger(__name__)


class SentenceTtsOutput:
    """Stage1 view that exposes only the sentence Talker should speak."""

    def __init__(self, request_id: str, text: str) -> None:
        self.request_id = request_id
        self.finished = False
        self.text = text
        self.cumulative_text = text
        self.outputs = [self]
        self.token_ids: list[int] = []


def _call_decode(decode: Any, token_ids: list[int]) -> str:
    try:
        decoded = decode(token_ids, skip_special_tokens=False)
    except TypeError:
        decoded = decode(token_ids)
    return decoded if isinstance(decoded, str) else ""


def decode_growing_token_ids(decode: Any, token_ids: list[int], cache: dict[str, object]) -> str:
    """Decode a growing id list, re-decoding only the unstable tail.

    The returned text matches ``decode(token_ids)`` when the tail window is a
    suffix of that string. Otherwise the full id list is decoded.
    """
    ids = list(token_ids)
    prev_ids = cache.get("decoded_ids")
    prev_text = cache.get("decoded_text")
    overlap_text = cache.get("decoded_overlap_text")
    candidate: str | None = None
    if (
        isinstance(prev_ids, list)
        and isinstance(prev_text, str)
        and isinstance(overlap_text, str)
        and prev_ids
        and len(ids) > len(prev_ids)
        and ids[: len(prev_ids)] == prev_ids
        and prev_text.endswith(overlap_text)
        and overlap_text
    ):
        extended = _call_decode(decode, [*prev_ids[-1:], *ids[len(prev_ids) :]])
        if extended.startswith(overlap_text):
            candidate = prev_text[: len(prev_text) - len(overlap_text)] + extended
            window_ids = [*prev_ids[-2:], *ids[len(prev_ids) :]] if len(prev_ids) >= 2 else ids
            window_text = _call_decode(decode, window_ids)
            if window_text and not candidate.endswith(window_text):
                candidate = None
    if candidate is None:
        candidate = _call_decode(decode, ids)
    cache["decoded_ids"] = ids
    cache["decoded_text"] = candidate
    cache["decoded_overlap_text"] = _call_decode(decode, ids[-1:]) if ids else ""
    return candidate


def stage1_tts_text(orchestrator: Any, output: Any, *, cache: dict[str, object] | None = None) -> str:
    """Stage1 text for sentence TTS.

    ``cumulative_text`` is attached only when the request finishes.
    Mid-generation chunks still carry ``cumulative_token_ids``.
    """
    from vllm_omni.model_executor.stage_input_processors.aura_omni import (
        _extract_output,
        _extract_text,
    )

    completion = _extract_output(output)
    cumulative = getattr(completion, "cumulative_text", None)
    if isinstance(cumulative, str) and cumulative:
        return cumulative
    token_ids = getattr(completion, "cumulative_token_ids", None)
    if isinstance(token_ids, list) and token_ids:
        processor = orchestrator.stage_pools[1].output_processor
        tokenizer = getattr(processor, "tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            if cache is None:
                decoded = _call_decode(decode, list(token_ids))
            else:
                decoded = decode_growing_token_ids(decode, list(token_ids), cache)
            if decoded:
                return decoded
    return _extract_text(output)


def plan_partial_stage_output(
    orchestrator: Any,
    stage_id: int,
    replica_id: int,
    output: Any,
    req_state: Any,
) -> PartialStageForward | None:
    """Hand a finished sentence to Talker before Stage1 finishes.

    Same request id as the turn (Code2Wav was prewarmed on it). Text goes
    through ``aura2tts`` in the orchestrator, not the ``from_stage_1`` SHM
    edge that clears ``additional_information.text``. Talker stays resumable
    until Stage1 finishes. The orchestrator submits the returned plan.
    """
    del replica_id
    if not req_state.session_owned or stage_id != 1:
        return
    if stage_id + 1 > req_state.final_stage_id:
        return
    next_client = orchestrator.stage_pools[stage_id + 1].stage_client
    processor = getattr(next_client, "custom_process_input_func", None)
    if getattr(processor, "__name__", "") != "aura2tts":
        return
    if orchestrator._stage_receives_async_chunks(stage_id + 1):
        return

    from vllm_omni.model_executor.stage_input_processors.aura_omni import (
        _sentence_tts_enabled,
        next_duplex_sentence_chunk,
    )

    if not _sentence_tts_enabled():
        return

    bridge = req_state.streaming.bridge_states.setdefault("aura_sentence_tts", {})
    if not isinstance(bridge, dict):
        bridge = {}
        req_state.streaming.bridge_states["aura_sentence_tts"] = bridge
    if bridge.get("closed"):
        return

    finished = bool(getattr(output, "finished", False))
    raw_text = stage1_tts_text(orchestrator, output, cache=bridge)
    # Captured before the chunk helper increments ``emits``.
    prior_emits = int(bridge.get("emits", 0))
    chunk = next_duplex_sentence_chunk(bridge, raw_text, finished=finished)
    close_only = False
    if chunk is None:
        if not (finished and int(bridge.get("emits", 0))):
            return
        chunk = ""
        close_only = True
    if not chunk and not close_only:
        return

    prompt = req_state.prompt if isinstance(req_state.prompt, dict) else None
    if prompt is not None:
        raw_info = prompt.get("additional_information")
        if not isinstance(raw_info, dict):
            raw_info = {}
            prompt["additional_information"] = raw_info
        # History is committed by the session runner from model_context_text.
        # Partial and close-only updates must not commit inside aura2tts.
        raw_info["aura_tts_partial"] = True
        raw_info["aura_tts_close_only"] = close_only

    view = SentenceTtsOutput(str(getattr(output, "request_id", req_state.request_id)), chunk)
    # A later sentence on an already-running Talker must stay resumable.
    # is_final_update=True is a non-resumable end sentinel: it discards this
    # text and sets streaming_input=False on the in-flight sentence, so
    # Code2Wav sees a short codec and the user hears a blip.
    queue_close_after = bool(chunk) and finished and not close_only and prior_emits > 0
    logger.info(
        "[AURA] sentence TTS req=%s finished=%s close_only=%s text_len=%d queue_close_after=%s",
        req_state.request_id,
        finished,
        close_only,
        len(chunk),
        queue_close_after,
    )
    if finished:
        bridge["closed"] = True
    return PartialStageForward(
        output=view,
        is_final_update=finished and not queue_close_after,
        close_only=close_only,
        queue_close_after=queue_close_after,
    )

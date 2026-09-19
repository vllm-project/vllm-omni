# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage1 sentence handoff onto Talker. AURA-only; the orchestrator just calls it."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

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


def stage1_tts_text(orchestrator: Any, output: Any) -> str:
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
            try:
                decoded = decode(list(token_ids), skip_special_tokens=False)
            except TypeError:
                decoded = decode(list(token_ids))
            if isinstance(decoded, str) and decoded:
                return decoded
    return _extract_text(output)


async def forward_partial_stage_output(
    orchestrator: Any,
    stage_id: int,
    replica_id: int,
    output: Any,
    req_state: Any,
) -> None:
    """Hand a finished sentence to Talker before Stage1 finishes.

    Same request id as the turn (Code2Wav was prewarmed on it). Text goes
    through ``aura2tts`` in the orchestrator, not the ``from_stage_1`` SHM
    edge that clears ``additional_information.text``. Talker stays resumable
    until Stage1 finishes.
    """
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
        _strip_assistant_text,
        commit_duplex_stage1_history,
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
    raw_text = stage1_tts_text(orchestrator, output)
    chunk = next_duplex_sentence_chunk(bridge, raw_text, finished=finished)
    close_only = False
    if chunk is None:
        if not (finished and int(bridge.get("emits", 0)) and bridge.get("last")):
            return
        chunk = str(bridge.get("last") or "")
        close_only = True
    if not chunk:
        return

    prompt = req_state.prompt if isinstance(req_state.prompt, dict) else None
    if prompt is not None:
        raw_info = prompt.get("additional_information")
        if not isinstance(raw_info, dict):
            raw_info = {}
            prompt["additional_information"] = raw_info
        # Partial chunks must not commit history. The finish path commits
        # the full stripped turn once, then aura2tts sees this flag.
        raw_info["aura_tts_partial"] = True
        if finished and not bridge.get("history_committed"):
            raw_info["aura_tts_partial"] = False
            commit_duplex_stage1_history(raw_info, _strip_assistant_text(raw_text) or raw_text)
            raw_info["aura_tts_partial"] = True
            bridge["history_committed"] = True

    view = SentenceTtsOutput(str(getattr(output, "request_id", req_state.request_id)), chunk)
    logger.info(
        "[AURA] sentence TTS req=%s finished=%s close_only=%s text_len=%d",
        req_state.request_id,
        finished,
        close_only,
        len(chunk),
    )
    await orchestrator._forward_to_next_stage(
        req_state.request_id,
        stage_id,
        view,
        req_state,
        src_replica_id=replica_id,
        is_streaming_session=True,
        is_final_update=finished,
    )
    if finished:
        bridge["closed"] = True

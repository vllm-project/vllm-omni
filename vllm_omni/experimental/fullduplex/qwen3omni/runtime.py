# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-Omni duplex runtime extension (engine-side model policy).

Implements ``DuplexRuntimeExtension``
(``vllm_omni/experimental/fullduplex/engine/contracts.py``) for the
thinker -> talker -> code2wav pipeline.

Scope: client-signalled barge-in over a persistent session. Qwen3-Omni has
no learned listen/speak control token, so unlike MiniCPM-o 4.5 this
extension implements no model-owned turn policy -- see ``decide_output``.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputDecision,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy

#: Input modes this model accepts. Anything else is rejected rather than
#: silently degraded -- per Sy0307's RFC #3745 review point that append mode
#: must be an explicit per-model capability.
logger = init_logger(__name__)

#: float32 little-endian PCM.
_BYTES_PER_SAMPLE = 4

_THINKER_STAGE_ID = 0


def _completion_text(output: object) -> str:
    """Text from a stage output, tolerating both output shapes.

    Stage 0 hands over a raw vllm ``RequestOutput`` (completions on
    ``.outputs``); wrapped stages nest it under ``.request_output``.
    """
    request_output = getattr(output, "request_output", None) or output
    completions = getattr(request_output, "outputs", None)
    if not completions:
        return ""
    text = getattr(completions[0], "text", None)
    return text if isinstance(text, str) else ""


SUPPORTED_INPUT_MODES = frozenset(
    {
        DuplexInputMode.APPEND_AUDIO_CHUNK,
        DuplexInputMode.TURN_COMMIT_ONLY,
    }
)


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _stage_config_value(runtime_config: dict[str, Any], key: str, stage_id: int) -> object | None:
    """Read a per-stage override addressed as ``cfg[key][stage_id]``."""
    raw = runtime_config.get(key)
    if isinstance(raw, dict):
        value = raw.get(stage_id)
        return raw.get(str(stage_id)) if value is None else value
    if isinstance(raw, (list, tuple)) and stage_id < len(raw):
        return raw[stage_id]
    return None


def duplex_audio_token_count(payload: object) -> int:
    """Thinker embeddings the appended audio will produce.

    The worker must produce exactly this many. If the two disagree the runner
    truncates or pads silently, so this and
    ``Qwen3OmniStage0DuplexRuntime.expected_embedding_count`` both defer to
    ``Qwen3OmniDuplexPolicy.audio_tokens_for_samples``.
    """
    num_samples = _payload_num_samples(payload)
    if num_samples > 0:
        return max(1, Qwen3OmniDuplexPolicy.audio_tokens_for_samples(num_samples))
    if isinstance(payload, dict) and "audio" in payload:
        # Explicitly empty audio, e.g. a commit whose buffer was already
        # drained by earlier appends. Reserve nothing; the prompt is the
        # turn-closing scaffolding alone.
        return 0
    return Qwen3OmniDuplexPolicy.tokens_per_chunk()


def _payload_num_samples(payload: object) -> int:
    """Sample count of an audio payload, measured from the audio itself.

    Deliberately does not trust a ``num_samples`` key: the serving layer may
    concatenate two payloads (``serving.py:_merge_native_audio_payloads``),
    and that merge rebuilds ``audio`` while copying the second payload's
    other keys verbatim -- so a carried-over ``num_samples`` would describe
    only the tail. Under-counting here under-reserves scheduler slots, which
    the model runner absorbs by silently truncating embeddings.
    """
    if not isinstance(payload, dict):
        return 0
    audio = payload.get("audio")
    if isinstance(audio, str) and audio:
        try:
            return len(base64.b64decode(audio, validate=True)) // _BYTES_PER_SAMPLE
        except (binascii.Error, ValueError):
            return 0
    if isinstance(audio, (bytes, bytearray)):
        return len(audio) // _BYTES_PER_SAMPLE
    return _coerce_int(payload.get("num_samples")) or 0


def _token_ids(runtime_config: dict[str, Any], key: str) -> list[int]:
    raw = runtime_config.get(key)
    if not isinstance(raw, (list, tuple)):
        return []
    return [int(token_id) for token_id in raw]


def build_duplex_prompt_token_ids(
    *,
    runtime_config: dict[str, Any],
    payload: object,
    seq: int,
    turn_seq: int,
    final: bool,
) -> tuple[list[int], int, int]:
    """Assemble the stage-0 prompt for one append.

    Returns ``(prompt_token_ids, audio_offset, audio_token_count)``.

    Layout, so the thinker sees a well-formed chat turn rather than a bare
    blob of audio embeddings:

    * first append of the session -- system block, then the user turn opener
    * first append of any later turn -- the user turn opener, then
      ``<|audio_start|>``
    * every append -- ``<|audio_pad|>`` placeholders that stage 0 overwrites
      with audio embeddings
    * final append of a turn -- ``<|audio_end|>``, close the user turn, open
      the assistant's, which is what actually prompts a reply

    The ``<|audio_start|>`` / ``<|audio_end|>`` delimiters are not optional:
    without them the thinker does not treat the span as audio and emits
    role-marker garbage.

    Real token ids are used for the scaffolding (not filler), so the worker
    can embed those positions through the ordinary embedding lookup and only
    needs to replace the audio span. ``audio_offset`` tells it where that
    span begins.
    """
    audio_tokens = duplex_audio_token_count(payload)
    closes_turn = final or (isinstance(payload, dict) and payload.get(Qwen3OmniDuplexPolicy.TURN_FINAL_KEY) is True)

    # `turn_seq` counts turns in the session, not appends within one turn, so
    # gating the opener on `turn_seq <= 1` framed only the session's first
    # turn and left every later turn as bare audio with no <|im_start|>user
    # and no <|audio_start|>. Observed: turn 1 answered correctly, turns 2 and
    # 3 came through as prefix=0 and produced nothing usable.
    #
    # Audio is held until commit, so an append that closes a turn carries the
    # whole turn and needs the opener as well as the closer. The `turn_seq`
    # arm remains for a mid-turn append, which the current buffer never emits.
    starts_turn = closes_turn or turn_seq <= 1

    prefix: list[int] = []
    if seq <= 1:
        prefix += _token_ids(runtime_config, Qwen3OmniDuplexPolicy.SESSION_PREFIX_IDS_KEY)
    elif starts_turn:
        # Close the assistant's previous turn before opening a new user turn.
        #
        # Generation stops *at* <|im_end|>, so the stop token is not written
        # back into the KV. Without this the conversation reads as a run-on
        # assistant message followed by a user turn, and from the second turn
        # on the model answers generically regardless of what was said.
        prefix.append(Qwen3OmniDuplexPolicy.IM_END_TOKEN_ID)
        prefix += _token_ids(runtime_config, Qwen3OmniDuplexPolicy.NEWLINE_IDS_KEY)
    if starts_turn:
        prefix += _token_ids(runtime_config, Qwen3OmniDuplexPolicy.TURN_PREFIX_IDS_KEY)
        prefix.append(Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID)

    suffix: list[int] = []
    if closes_turn:
        suffix.append(Qwen3OmniDuplexPolicy.AUDIO_END_TOKEN_ID)
        suffix += _token_ids(runtime_config, Qwen3OmniDuplexPolicy.TURN_SUFFIX_IDS_KEY)

    prompt_token_ids = prefix + [Qwen3OmniDuplexPolicy.AUDIO_PAD_TOKEN_ID] * audio_tokens + suffix
    logger.info(
        "[qwen3omni-duplex] plan seq=%s turn_seq=%s final=%s closes_turn=%s "
        "prefix=%d audio=%d suffix=%d total=%d scaffold_keys=%s",
        seq,
        turn_seq,
        final,
        closes_turn,
        len(prefix),
        audio_tokens,
        len(suffix),
        len(prompt_token_ids),
        sorted(k for k in runtime_config if "token_ids" in k),
    )
    return prompt_token_ids, len(prefix), audio_tokens


def build_duplex_data_plane_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: dict[str, Any],
    runtime_config: dict[str, Any],
    seq: int,
    turn_seq: int,
    mode: DuplexInputMode,
    payload: object,
    final: bool,
) -> dict[str, Any]:
    """Build the stage-0 append prompt.

    Only ``prompt_token_ids`` and ``model_intermediate_buffer`` survive
    ``build_engine_core_request_from_tokens``
    (``vllm_omni/engine/orchestrator.py:118-158``) -- every other top-level
    key is dropped there with no error. Do not add keys here expecting them
    to reach the worker.

    Note ``multi_modal_data`` is among the dropped keys, and the duplex
    submit path passes no ``mm_features``
    (``orchestrator.py:289-302``). Audio therefore CANNOT travel by
    Qwen3-Omni's normal multimodal route; it rides inside
    ``model_intermediate_buffer["duplex"]["payload"]`` as base64 PCM and must
    be turned into embeddings by the model's own ``preprocess`` hook.

    All ``duplex`` sub-keys are emitted on every append, unconditionally. The
    worker-side merge is additive per sub-key
    (``gpu_model_runner.py:2035-2042``), so a conditionally-omitted key would
    leave the previous append's value in place rather than clearing it.
    """
    prompt_token_ids, audio_offset, audio_tokens = build_duplex_prompt_token_ids(
        runtime_config=runtime_config,
        payload=payload,
        seq=seq,
        turn_seq=turn_seq,
        final=final,
    )
    return {
        "prompt_token_ids": prompt_token_ids,
        "model_intermediate_buffer": {
            "request_id": request_id,
            # Session-scoped id used for cross-stage chunk routing
            # (gpu_ar_model_runner.py:2155-2168).
            "global_request_id": [fence.session_id],
            "duplex": {
                # Mandatory gate: every worker-side duplex branch checks this
                # and silently no-ops when it is absent or not True.
                "data_plane": True,
                "session_id": fence.session_id,
                "incarnation": fence.incarnation,
                "epoch": fence.epoch,
                "seq": seq,
                "turn_id": fence.turn_id,
                "response_seq": fence.response_seq,
                "turn_seq": turn_seq,
                "mode": mode.value,
                "payload": payload,
                "final": final,
                "session_config": dict(session_config),
                "runtime_config": dict(runtime_config),
                "scheduler_token_budget": len(prompt_token_ids),
                # Where stage 0 must splice the audio embeddings, and how
                # many. Everything outside this span is real scaffolding
                # tokens that embed through the ordinary lookup.
                #
                # `audio_offset` is relative to THIS APPEND's token span, not
                # to the session. The consumer compares it against
                # `duplex_token_offset`, which is absolute, so it must first
                # rebase using `append_token_count`. Emitted explicitly rather
                # than reusing `scheduler_token_budget` (numerically equal
                # here) because conflating "how many slots to reserve" with
                # "how long this append is" is what made the frame mismatch
                # invisible in the first place.
                "audio_offset": audio_offset,
                "audio_tokens": audio_tokens,
                "append_token_count": len(prompt_token_ids),
            },
        },
    }


class Qwen3OmniDuplexRuntimeExtension:
    """Pure model policy for Qwen3-Omni thinker -> talker -> code2wav."""

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        """Apply per-stage sampling overrides.

        ``defaults`` is one entry per pipeline stage (3 for Qwen3-Omni); the
        returned tuple must match in length and order. Stages without an
        override are passed through unchanged, which keeps codec sampling for
        the talker and code2wav sourced from the checkpoint's own config.

        The stage-0 ``max_tokens`` bound matters more here than it does for
        MiniCPM: Qwen3-Omni emits no ``<|chunk_eos|>``, so the token budget is
        the only thing stopping the thinker from running past the user's next
        utterance.
        """
        configured: list[object] = []
        for stage_id, default in enumerate(defaults):
            max_tokens = _coerce_int(_stage_config_value(runtime_config, "duplex_stage_max_tokens", stage_id))
            raw_overrides = _stage_config_value(runtime_config, "duplex_stage_sampling_params", stage_id)
            overrides = dict(raw_overrides) if isinstance(raw_overrides, dict) else {}
            if not isinstance(default, SamplingParams) or (not overrides and (max_tokens is None or max_tokens <= 0)):
                configured.append(default)
                continue
            params = default.clone()
            if max_tokens is not None and max_tokens > 0:
                params.max_tokens = max_tokens
            for name, value in overrides.items():
                if not hasattr(params, name):
                    continue
                setattr(params, name, value)
                if name == "stop_token_ids":
                    all_stop_token_ids = getattr(params, "_all_stop_token_ids", None)
                    if isinstance(all_stop_token_ids, set):
                        all_stop_token_ids.update(int(token_id) for token_id in value)
            configured.append(params)
        return tuple(configured)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, Any],
        runtime_config: dict[str, Any],
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        """Build the stage-0 append plan for one input chunk.

        There is no ``stage_id`` parameter: the control plane hard-codes
        stage 0 for appends (``duplex_control_plane.py:444``). Audio enters at
        the thinker; the talker and code2wav are fed by the orchestrator's
        async-chunk forwarding.
        """
        del sampling_params  # stage sampling is applied by the control plane
        if mode not in SUPPORTED_INPUT_MODES:
            raise ValueError(
                f"Qwen3-Omni duplex does not support input mode {mode.value!r}; "
                f"supported: {sorted(m.value for m in SUPPORTED_INPUT_MODES)}"
            )
        return DuplexAppendPlan(
            prompt=build_duplex_data_plane_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                mode=mode,
                payload=payload,
                final=final,
            )
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, Any],
        output: object,
    ) -> DuplexOutputDecision | None:
        r"""Surface the thinker's text to the client.

        The duplex output collector only accepts outputs whose stage_id is at
        or past ``response_stage_id`` -- the final stage, code2wav
        (``request_client.py:299``). Stage-0 text would therefore never reach
        the client and ``response.audio_transcript.delta`` would always be
        empty. Note this transcript is not ASR: the thinker *produces* the
        text that the talker then speaks, so it is already available here.

        Returning a decision would mark the output ``duplex_direct_response``
        and bypass that filter (``request_client.py:348-354``) -- but the
        orchestrator ``return``\ s immediately after emitting a direct
        response (``orchestrator.py:1284-1295``), skipping
        ``_forward_to_next_stage``. Doing that for thinker output delivers the
        transcript and *no audio*: verified live, stage 2 never ran.

        Direct response and stage advancement are therefore mutually
        exclusive for the same output, and audio matters more. Returning
        ``None`` keeps the thinker feeding the talker.

        This is why ``response.audio_transcript.delta`` is currently empty.
        The text exists and is correct -- the collector only admits
        ``stage_id >= response_stage_id`` (2) and there is no per-output way
        to both forward and surface. A fix belongs in the framework, not
        here: either the collector accepts the pipeline's declared text
        ``final_output`` stage, or a decision gains a "forward as well" mode.

        No turn policy is expressed here either -- Qwen3-Omni has no learned
        listen/speak token, so this never reports a model listen decision the
        way MiniCPM's extension does.
        """
        del segment_token_ids, segment_output_metadata, segment_finished
        del stage_id, final_stage_id, output
        return None

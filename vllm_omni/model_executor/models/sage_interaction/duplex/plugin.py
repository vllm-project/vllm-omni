# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-duplex model plugin for InternLM Sage-Interaction."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputAction,
    DuplexOutputDecision,
    ToolCallAbandonReason,
)
from vllm_omni.engine.duplex.intermediate import build_duplex_append_prompt
from vllm_omni.engine.duplex.plugin import (
    EncodeAudio,
)
from vllm_omni.model_executor.common.duplex.payload import payload_sample_count as _duplex_pcm_sample_count
from vllm_omni.model_executor.common.request_outputs import coerce_int as _coerce_int
from vllm_omni.model_executor.common.request_outputs import coerce_int_list as _coerce_int_list
from vllm_omni.model_executor.common.request_outputs import first_completion as _first_completion
from vllm_omni.model_executor.common.request_outputs import multimodal_output as _multimodal_output
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.capabilities import (
    minicpmo45_native_capabilities,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import (
    PRIVATE_RUNTIME_CONFIG_KEYS as _MINICPMO45_PRIVATE_KEYS,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import (
    TOKENIZER_KEY,
    MiniCPMO45ClientRuntimeConfigError,
    MiniCPMO45DuplexPlugin,
    _completion_token_ids,
    _duplex_force_listen_count,
    _duplex_payload_is_vision_only,
    _duplex_vision_tile_pixels,
    _duplex_vision_tokens,
    _special_token_ids,
    duplex_first_append_context_reserve,
    duplex_first_append_unit_count,
    duplex_payload_is_exact_chunks,
)
from vllm_omni.model_executor.models.sage_interaction.duplex.data_plane import (
    SageInteractionDataPlane,
)
from vllm_omni.model_executor.models.sage_interaction.duplex.delegation import (
    DEFAULT_MAX_DELEGATE_QUERY_TOKENS,
    DEFAULT_MAX_SPEAK_TOKENS_PER_CHUNK,
    DELEGATION_TOOL_NAME,
    MAX_QUEUED_TOOL_RESULTS,
    MAX_TOOL_RESULT_CHARS,
    SAGE_INTERACTION_DEFAULT_INSTRUCTIONS,
    encode_tool_result_tokens,
    format_delegate_result_payload,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig

_DUPLEX_CHUNK_SAMPLES = 16000
_DUPLEX_SAMPLES_PER_AUDIO_TOKEN = 1600

SAGE_PENDING_TOOL_RESULTS_KEY = "sage_pending_tool_results"
SAGE_TOOL_RESULT_GENERATION_KEY = "sage_tool_result_generation"
SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY = "sage_submitted_tool_result_generation"
SAGE_MODEL_PATH_KEY = "sage_model_path"
SAGE_WRAP_AUDIO_MARKERS_KEY = "duplex_wrap_unit_audio_markers"
SAGE_ALLOW_INTERRUPTION_KEY = "duplex_allow_speech_interruption"
SAGE_MAX_SPEAK_TOKENS_KEY = "duplex_max_speak_tokens_per_chunk"
SAGE_MAX_DELEGATE_TOKENS_KEY = "duplex_max_delegate_query_tokens"

SAGE_PRIVATE_RUNTIME_CONFIG_KEYS = _MINICPMO45_PRIVATE_KEYS | frozenset(
    {
        SAGE_PENDING_TOOL_RESULTS_KEY,
        SAGE_TOOL_RESULT_GENERATION_KEY,
        SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY,
        SAGE_MODEL_PATH_KEY,
        SAGE_WRAP_AUDIO_MARKERS_KEY,
        SAGE_ALLOW_INTERRUPTION_KEY,
        SAGE_MAX_SPEAK_TOKENS_KEY,
        SAGE_MAX_DELEGATE_TOKENS_KEY,
    }
)


def _sage_audio_token_budget(payload: object, *, wrap_audio_markers: bool, default: int = 64) -> int:
    sample_count = _duplex_pcm_sample_count(payload)
    if sample_count is None:
        return max(1, int(default))
    sample_count = max(1, sample_count)
    per_unit_overhead = 4 if wrap_audio_markers else 2
    if sample_count % _DUPLEX_CHUNK_SAMPLES == 0:
        units = sample_count // _DUPLEX_CHUNK_SAMPLES
        return units * (per_unit_overhead + _DUPLEX_CHUNK_SAMPLES // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN)
    return max(16, min(768, sample_count // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8 + (2 if wrap_audio_markers else 0)))


def _sage_unit_size(*, wrap_audio_markers: bool) -> int:
    # Stock MiniCPM-o 4.5 unit: <unit> + 10 audio + </unit> = 12.
    # Sage-Interaction wraps audio in <|audio_start|> ... <|audio_end|> = 14.
    return 14 if wrap_audio_markers else 12


def build_sage_data_plane_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: dict[str, object],
    runtime_config: dict[str, object],
    seq: int,
    turn_seq: int,
    payload: object,
    final: bool,
    prepend_text_token_ids: list[int] | None = None,
    submitted_tool_result_generation: int | None = None,
) -> dict[str, object]:
    """Build Stage 0 append prompt with exact Sage-Interaction token budget."""
    first_audio_window = isinstance(payload, dict) and payload.get("first_audio_window") is True
    if first_audio_window:
        payload = {key: value for key, value in payload.items() if key != "first_audio_window"}

    wrap_audio_markers = bool(runtime_config.get(SAGE_WRAP_AUDIO_MARKERS_KEY, True))
    unit_size = _sage_unit_size(wrap_audio_markers=wrap_audio_markers)
    vision_tokens = _duplex_vision_tokens(payload, tile_pixels=_duplex_vision_tile_pixels(runtime_config))
    token_budget = _sage_audio_token_budget(payload, wrap_audio_markers=wrap_audio_markers) + vision_tokens

    if isinstance(payload, dict) and payload.get("type") == "text":
        text_ids = payload["text_token_ids"]
        if not isinstance(text_ids, list) or not text_ids:
            raise ValueError("Text append requires encoded text tokens")
        lead_tokens = duplex_first_append_context_reserve(runtime_config) if seq <= 1 else 2
        token_budget = lead_tokens + 1 + len(text_ids)
    elif _duplex_payload_is_vision_only(payload):
        lead_tokens = duplex_first_append_context_reserve(runtime_config) if seq <= 1 else 2
        token_budget = lead_tokens + 1 + vision_tokens
    elif seq <= 1 or first_audio_window:
        lead_tokens = 2 if seq > 1 else duplex_first_append_context_reserve(runtime_config)
        first_units = duplex_first_append_unit_count(payload)
        if first_units is not None:
            token_budget = lead_tokens + first_units * unit_size - 1 + vision_tokens
        else:
            token_budget += lead_tokens
    if seq > 1 and not first_audio_window and duplex_payload_is_exact_chunks(payload):
        token_budget += 1

    if prepend_text_token_ids:
        token_budget += len(prepend_text_token_ids)
        if isinstance(payload, dict):
            payload = {**payload, "prepend_text_token_ids": list(prepend_text_token_ids)}

    extra_body = session_config.get("extra_body")
    raw_token_id = runtime_config.get("duplex_scheduler_token_id")
    try:
        token_id = 0 if raw_token_id is None else max(0, int(raw_token_id))
    except (TypeError, ValueError):
        token_id = 0
    force_listen_count = _duplex_force_listen_count(extra_body)
    if (
        force_listen_count > 0
        and turn_seq <= force_listen_count
        and isinstance(payload, dict)
        and payload.get("force_listen") is not True
    ):
        payload = {**payload, "force_listen": True}

    raw_listen_scale = runtime_config.get("duplex_listen_prob_scale")
    try:
        listen_scale = 1.0 if raw_listen_scale is None else float(raw_listen_scale)
    except (TypeError, ValueError):
        listen_scale = 1.0
    raw_listen_top_k = runtime_config.get("duplex_listen_top_k")
    try:
        listen_top_k = max(0, int(raw_listen_top_k)) if raw_listen_top_k is not None else 0
    except (TypeError, ValueError):
        listen_top_k = 0
    if isinstance(payload, dict) and (listen_scale != 1.0 or listen_top_k > 0):
        payload = {**payload, "listen_prob_scale": listen_scale, "listen_top_k": listen_top_k}

    # Sage uses the shared in-request window, but has different delegate and
    # listen boundaries from the stock MiniCPM-o replay protocol.
    model_fields: dict[str, object] = {"scheduler_token_id": token_id, "session_window_replay": False}
    if submitted_tool_result_generation is not None:
        model_fields[SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY] = int(submitted_tool_result_generation)

    return build_duplex_append_prompt(
        request_id=request_id,
        fence=fence,
        session_config=session_config,
        runtime_config=runtime_config,
        seq=seq,
        turn_seq=turn_seq,
        payload=payload,
        final=final,
        prompt_token_ids=[token_id] * token_budget,
        model_fields=model_fields,
    )


def _current_unit_tokens(token_ids: list[int], special_token_ids: Mapping[str, int]) -> list[int]:
    """Slice tokens belonging to the most recent unit (after the prior terminator)."""
    if not token_ids:
        return []
    terminators = {
        tid
        for key in ("listen_token_id", "chunk_eos_token_id", "chunk_tts_eos_token_id", "unit_end_token_id")
        if (tid := special_token_ids.get(key)) is not None
    }
    end = len(token_ids)
    # Ignore only the current unit's own single trailing terminator so a
    # listen-only unit after a prior delegate unit never walks back into the
    # previous unit.
    scan_end = end - 1 if token_ids[end - 1] in terminators else end
    start = 0
    for idx in range(scan_end - 1, -1, -1):
        if token_ids[idx] in terminators:
            start = idx + 1
            break
    return token_ids[start:end]


class SageInteractionDuplexPlugin(MiniCPMO45DuplexPlugin):
    """Engine and session duplex policy for InternLM Sage-Interaction."""

    plugin_id = "sage_interaction"
    private_runtime_config_keys = SAGE_PRIVATE_RUNTIME_CONFIG_KEYS
    abandoned_tool_call_reasons = frozenset(
        {
            ToolCallAbandonReason.TIMEOUT,
            ToolCallAbandonReason.SUPPRESSED,
        }
    )

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane: SageInteractionDataPlane = SageInteractionDataPlane(encode_audio)

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return replace(
            minicpmo45_native_capabilities(max_sessions=max_sessions),
            tool_mode="delegate_only",
            function_call_output_continuation="automatic",
            max_tools=1,
            clears_function_call_block_on_user_commit=True,
        )

    @staticmethod
    def declares_delegation(tools: object) -> bool:
        """Whether ``realtime_tools`` declares only Sage-Interaction's ``delegate`` tool."""
        if not isinstance(tools, list) or len(tools) != 1:
            return False
        tool = tools[0]
        if not isinstance(tool, Mapping):
            return False
        return tool.get("type", "function") == "function" and tool.get("name") == DELEGATION_TOOL_NAME

    def validate_client_extra_body(self, extra_body: object) -> None:
        super().validate_client_extra_body(extra_body)
        if not isinstance(extra_body, Mapping):
            return
        tools = extra_body.get("realtime_tools")
        if tools is not None and not self.declares_delegation(tools):
            raise MiniCPMO45ClientRuntimeConfigError(
                "Sage-Interaction sessions only support the delegate tool; other tool declarations are unsupported",
                code="unsupported_tools",
            )

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        extra_body = dict(config.extra_body)
        allow_interruption = bool(extra_body.pop("allow_speech_interruption", True))
        raw_delegate_cap = extra_body.pop("max_delegate_query_tokens", DEFAULT_MAX_DELEGATE_QUERY_TOKENS)
        try:
            max_delegate_tokens = max(32, int(raw_delegate_cap))
        except (TypeError, ValueError) as exc:
            raise MiniCPMO45ClientRuntimeConfigError(
                "max_delegate_query_tokens must be a positive integer",
                code="invalid_delegate_config",
            ) from exc
        realtime_tools = extra_body.get("realtime_tools")
        if realtime_tools is not None and not self.declares_delegation(realtime_tools):
            raise MiniCPMO45ClientRuntimeConfigError(
                "Sage-Interaction sessions only support the delegate tool; other tool declarations are unsupported",
                code="unsupported_tools",
            )

        effective_config = deepcopy(config)
        effective_config.extra_body = extra_body
        if not (isinstance(effective_config.instructions, str) and effective_config.instructions.strip()):
            effective_config.instructions = SAGE_INTERACTION_DEFAULT_INSTRUCTIONS

        runtime_config = await super().prepare_runtime_config(effective_config, model_config=model_config)
        # Propagate mutated extra_body / cleared ref_audio back to caller's config.
        config.extra_body = effective_config.extra_body
        config.ref_audio = effective_config.ref_audio

        speak_tokens = (
            config.max_tokens
            if isinstance(config.max_tokens, int) and config.max_tokens > 0
            else DEFAULT_MAX_SPEAK_TOKENS_PER_CHUNK
        )
        runtime_config[SAGE_WRAP_AUDIO_MARKERS_KEY] = True
        runtime_config[SAGE_ALLOW_INTERRUPTION_KEY] = allow_interruption
        runtime_config[SAGE_MAX_SPEAK_TOKENS_KEY] = speak_tokens
        runtime_config[SAGE_MAX_DELEGATE_TOKENS_KEY] = max_delegate_tokens
        runtime_config[SAGE_PENDING_TOOL_RESULTS_KEY] = []
        runtime_config[SAGE_TOOL_RESULT_GENERATION_KEY] = 0
        runtime_config[SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY] = 0
        model_path = getattr(model_config, "model", None)
        if isinstance(model_path, str) and model_path:
            runtime_config[SAGE_MODEL_PATH_KEY] = model_path
        if realtime_tools is not None:
            runtime_config["realtime_tools"] = deepcopy(realtime_tools)

        # Stage 0 SamplingParams.max_tokens must accommodate full <|delegate|>
        # JSON queries; normal <|speak|> units stay capped at speak_tokens inside
        # Stage 0's per-row duplex policy.
        stage_max_tokens = dict(runtime_config.get("duplex_stage_max_tokens") or {})
        stage_max_tokens["0"] = max(speak_tokens, max_delegate_tokens)
        stage_max_tokens.setdefault("1", 8192)
        runtime_config["duplex_stage_max_tokens"] = stage_max_tokens
        return runtime_config

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        effective_config = deepcopy(config)
        if not (isinstance(effective_config.instructions, str) and effective_config.instructions.strip()):
            current_instructions = current.get("instructions")
            if isinstance(current_instructions, str) and current_instructions:
                effective_config.instructions = current_instructions
        updated = super().runtime_config_for_update(effective_config, current)
        config.extra_body = effective_config.extra_body

        speak_tokens = (
            config.max_tokens
            if isinstance(config.max_tokens, int) and config.max_tokens > 0
            else int(current.get(SAGE_MAX_SPEAK_TOKENS_KEY, DEFAULT_MAX_SPEAK_TOKENS_PER_CHUNK))
        )
        max_delegate_tokens = int(current.get(SAGE_MAX_DELEGATE_TOKENS_KEY, DEFAULT_MAX_DELEGATE_QUERY_TOKENS))
        updated[SAGE_MAX_SPEAK_TOKENS_KEY] = speak_tokens
        updated[SAGE_MAX_DELEGATE_TOKENS_KEY] = max_delegate_tokens
        for key in (
            SAGE_WRAP_AUDIO_MARKERS_KEY,
            SAGE_ALLOW_INTERRUPTION_KEY,
            SAGE_PENDING_TOOL_RESULTS_KEY,
            SAGE_TOOL_RESULT_GENERATION_KEY,
            SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY,
            SAGE_MODEL_PATH_KEY,
        ):
            if key in current:
                updated[key] = deepcopy(current[key])

        stage_max_tokens = dict(updated.get("duplex_stage_max_tokens") or {})
        stage_max_tokens["0"] = max(speak_tokens, max_delegate_tokens)
        stage_max_tokens.setdefault("1", 8192)
        updated["duplex_stage_max_tokens"] = stage_max_tokens
        return updated

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        prepend_ids: list[int] = []
        submitted_gen: int | None = None
        raw_pending = runtime_config.get(SAGE_PENDING_TOOL_RESULTS_KEY)
        last_submitted = _coerce_int(runtime_config.get(SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY)) or 0
        if isinstance(raw_pending, list) and raw_pending and self.data_plane.backend_ready(request_id):
            for entry in raw_pending:
                if not isinstance(entry, Mapping):
                    continue
                gen = _coerce_int(entry.get("generation"))
                tids = entry.get("token_ids")
                if gen is None or gen <= last_submitted or not isinstance(tids, list):
                    continue
                prepend_ids.extend(int(token_id) for token_id in tids if isinstance(token_id, int))
                submitted_gen = gen if submitted_gen is None else max(submitted_gen, gen)

        return DuplexAppendPlan(
            prompt=build_sage_data_plane_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                payload=payload,
                final=final,
                prepend_text_token_ids=prepend_ids or None,
                submitted_tool_result_generation=submitted_gen,
            )
        )

    def runtime_patch_for_submitted_prompt(self, prompt: Mapping[str, object]) -> dict[str, object] | None:
        intermediate = prompt.get("model_intermediate_buffer")
        if not isinstance(intermediate, Mapping):
            return None
        duplex = intermediate.get("duplex")
        if not isinstance(duplex, Mapping):
            return None
        submitted_gen = _coerce_int(duplex.get(SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY))
        if submitted_gen is None:
            return None
        return {SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY: submitted_gen}

    def reconcile_submitted_runtime(
        self,
        current: Mapping[str, object],
        updated: Mapping[str, object],
    ) -> dict[str, object] | None:
        submitted_gen = _coerce_int(updated.get(SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY))
        if submitted_gen is None:
            return None
        reconciled = dict(updated)
        current_pending = current.get(SAGE_PENDING_TOOL_RESULTS_KEY)
        if isinstance(current_pending, list):
            reconciled[SAGE_PENDING_TOOL_RESULTS_KEY] = [
                deepcopy(entry)
                for entry in current_pending
                if isinstance(entry, Mapping) and (_coerce_int(entry.get("generation")) or 0) > submitted_gen
            ]
        current_gen = _coerce_int(current.get(SAGE_TOOL_RESULT_GENERATION_KEY)) or 0
        reconciled[SAGE_TOOL_RESULT_GENERATION_KEY] = max(current_gen, submitted_gen)
        return reconciled

    def retire_pending_function_output(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object] | None:
        del config
        pending = current.get(SAGE_PENDING_TOOL_RESULTS_KEY)
        if not isinstance(pending, list) or not pending:
            return None
        updated = deepcopy(dict(current))
        current_gen = _coerce_int(updated.get(SAGE_TOOL_RESULT_GENERATION_KEY)) or 0
        updated[SAGE_PENDING_TOOL_RESULTS_KEY] = []
        updated[SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY] = max(
            _coerce_int(updated.get(SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY)) or 0,
            current_gen,
        )
        return updated

    def runtime_config_for_function_output(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
        item: Mapping[str, object],
        *,
        history_len: int | None = None,
    ) -> dict[str, object]:
        del config, history_len
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise MiniCPMO45ClientRuntimeConfigError(
                "function_call_output requires a non-empty call_id",
                code="invalid_function_call_output",
            )
        payload = format_delegate_result_payload(item)
        content = str(payload.get("content") or "")
        if len(content) > MAX_TOOL_RESULT_CHARS:
            raise MiniCPMO45ClientRuntimeConfigError(
                f"function_call_output exceeds {MAX_TOOL_RESULT_CHARS} characters",
                code="function_call_output_too_large",
            )
        pending = list(current.get(SAGE_PENDING_TOOL_RESULTS_KEY) or [])
        if len(pending) >= MAX_QUEUED_TOOL_RESULTS:
            raise MiniCPMO45ClientRuntimeConfigError(
                f"too many queued delegate results (max {MAX_QUEUED_TOOL_RESULTS})",
                code="resource_exhausted",
            )

        tokenizer_key = current.get(TOKENIZER_KEY)
        tokenizer = self._tokenizers.get(tokenizer_key) if isinstance(tokenizer_key, str) else None
        if tokenizer is None:
            model_path = current.get(SAGE_MODEL_PATH_KEY)
            tokenizer = self._tokenizers.get(model_path if isinstance(model_path, str) else "")
        if tokenizer is None and self._tokenizers:
            tokenizer = next((tok for tok in self._tokenizers.values() if tok is not None), None)
        if tokenizer is None:
            raise MiniCPMO45ClientRuntimeConfigError(
                "Sage-Interaction tokenizer is not initialized for function_call_output encoding",
                code="tokenizer_unavailable",
            )
        try:
            token_ids = encode_tool_result_tokens(tokenizer, payload)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise MiniCPMO45ClientRuntimeConfigError(
                f"failed to encode Sage-Interaction tool result: {exc}",
                code="invalid_function_call_output",
            ) from exc

        updated = deepcopy(dict(current))
        next_gen = (_coerce_int(updated.get(SAGE_TOOL_RESULT_GENERATION_KEY)) or 0) + 1
        pending = list(updated.get(SAGE_PENDING_TOOL_RESULTS_KEY) or [])
        pending.append({"generation": next_gen, "call_id": call_id, "token_ids": token_ids})
        updated[SAGE_PENDING_TOOL_RESULTS_KEY] = pending
        updated[SAGE_TOOL_RESULT_GENERATION_KEY] = next_gen
        return updated

    def parse_function_call(self, model_result: Mapping[str, object]) -> dict[str, object] | None:
        if model_result.get("function_call") is not True:
            return None
        call_id = model_result.get("call_id")
        name = model_result.get("name")
        arguments = model_result.get("arguments", "")
        if not isinstance(call_id, str) or not call_id or name != DELEGATION_TOOL_NAME:
            return None
        return {"call_id": call_id, "name": name, "arguments": arguments}

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
        streaming_context: object | None = None,
    ) -> DuplexOutputDecision | None:
        if stage_id < final_stage_id and segment_finished:
            completion = _first_completion(output)
            output_metadata = _multimodal_output(output, completion)
            special_token_ids = _special_token_ids(segment_output_metadata)
            special_token_ids.update(_special_token_ids(output_metadata))
            delegate_id = special_token_ids.get("delegate_token_id")
            if delegate_id is not None:
                token_ids = _completion_token_ids(completion) or list(segment_token_ids)
                unit_ids = _current_unit_tokens(
                    max(
                        (
                            token_ids,
                            _coerce_int_list(getattr(completion, "cumulative_token_ids", None)),
                            list(segment_token_ids),
                        ),
                        key=len,
                    ),
                    special_token_ids,
                )
                if delegate_id in unit_ids:
                    delegate_pos = unit_ids.index(delegate_id)
                    terminators = {
                        tid
                        for key in (
                            "listen_token_id",
                            "chunk_eos_token_id",
                            "chunk_tts_eos_token_id",
                            "unit_end_token_id",
                        )
                        if (tid := special_token_ids.get(key)) is not None
                    }
                    post_ids = [tid for tid in unit_ids[delegate_pos + 1 :] if tid not in terminators]
                    delegate_text = self._decode_delegate_tokens(post_ids, completion=completion)
                    metadata = dict(output_metadata)
                    for key, value in special_token_ids.items():
                        metadata.setdefault(f"meta.{key}", value)
                    metadata.update(
                        {
                            "duplex_direct_response": True,
                            "duplex_native_decision": "delegate",
                            "delegate_token_ids": post_ids,
                            "delegate_text": delegate_text,
                        }
                    )
                    return DuplexOutputDecision(
                        action=DuplexOutputAction.DIRECT_RESPONSE,
                        metadata=metadata,
                    )
        return super().decide_output(
            stage_id=stage_id,
            final_stage_id=final_stage_id,
            segment_finished=segment_finished,
            segment_token_ids=segment_token_ids,
            segment_output_metadata=segment_output_metadata,
            output=output,
            streaming_context=streaming_context,
        )

    def _decode_delegate_tokens(self, token_ids: list[int], *, completion: object | None) -> str:
        if token_ids:
            tokenizer = next((tok for tok in self._tokenizers.values() if tok is not None), None)
            decode = getattr(tokenizer, "decode", None)
            if callable(decode):
                try:
                    return str(decode(token_ids, skip_special_tokens=True))
                except TypeError:
                    return str(decode(token_ids))
                except (ValueError, RuntimeError):
                    pass
        if completion is not None:
            raw_text = getattr(completion, "text", None)
            if isinstance(raw_text, str):
                return raw_text
        return ""


__all__ = [
    "SAGE_ALLOW_INTERRUPTION_KEY",
    "SAGE_MAX_DELEGATE_TOKENS_KEY",
    "SAGE_MAX_SPEAK_TOKENS_KEY",
    "SAGE_MODEL_PATH_KEY",
    "SAGE_PENDING_TOOL_RESULTS_KEY",
    "SAGE_PRIVATE_RUNTIME_CONFIG_KEYS",
    "SAGE_SUBMITTED_TOOL_RESULT_GENERATION_KEY",
    "SAGE_TOOL_RESULT_GENERATION_KEY",
    "SAGE_WRAP_AUDIO_MARKERS_KEY",
    "SageInteractionDuplexPlugin",
    "build_sage_data_plane_prompt",
]

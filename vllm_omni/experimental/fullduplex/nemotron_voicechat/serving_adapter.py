# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Realtime serving adapter for Nemotron VoiceChat native duplex."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from vllm_omni.experimental.fullduplex.minicpmo45.session import (
    MiniCPMO45ServingSessionState,
)
from vllm_omni.experimental.fullduplex.nemotron_voicechat.data_plane import (
    NemotronVoiceChatDataPlaneContext,
    NemotronVoiceChatDataPlaneSession,
)
from vllm_omni.experimental.fullduplex.nemotron_voicechat.input import (
    NemotronVoiceChatPcmAppendBuffer,
)
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexCapabilities
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import ServingRuntimeConfigError

EncodeAudio = Callable[[object, int, str, float | None], str | None]

_DEFAULT_SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again. "
    "Start the conversation by greeting the user."
)
_PRIVATE_KEYS = frozenset(
    {
        "nvc_prompt_token_ids",
        "nvc_text_bos_id",
        "nvc_text_eos_id",
        "nvc_text_pad_id",
        "nvc_function_sotc_id",
        "nvc_function_eotc_id",
        "nvc_function_eotr_id",
        "nvc_max_model_len",
        "nvc_tokenizer_ref",
        "nvc_tools_signature",
        "nvc_function_response_generation",
        "nvc_function_response_token_ids",
        "nvc_function_response_call_id",
        "nvc_function_response_batches",
    }
)


def _stt_config(model_config: Any) -> dict[str, Any]:
    hf_config = getattr(model_config, "hf_config", None)
    stt_cfg = getattr(hf_config, "stt_cfg", None)
    if not isinstance(stt_cfg, dict):
        raise ServingRuntimeConfigError("Nemotron VoiceChat checkpoint STT configuration is unavailable")
    return stt_cfg


def _normalized_tools(config: object) -> tuple[list[dict[str, object]], str]:
    extra_body = getattr(config, "extra_body", None)
    raw_tools = extra_body.get("realtime_tools") if isinstance(extra_body, dict) else None
    if raw_tools is None:
        return [], "[]"
    if not isinstance(raw_tools, list):
        raise ServingRuntimeConfigError("Nemotron VoiceChat tools must be a list")
    if len(raw_tools) > 5:
        raise ServingRuntimeConfigError("Nemotron VoiceChat supports at most 5 tools per session")

    normalized: list[dict[str, object]] = []
    for index, tool in enumerate(raw_tools):
        if not isinstance(tool, dict):
            raise ServingRuntimeConfigError(f"Nemotron VoiceChat tool {index} must be an object")
        function = tool.get("function", tool)
        if not isinstance(function, dict):
            raise ServingRuntimeConfigError(f"Nemotron VoiceChat tool {index} has no function definition")
        definition = {key: value for key, value in function.items() if key != "type"}
        if not isinstance(definition.get("name"), str) or not str(definition["name"]).strip():
            raise ServingRuntimeConfigError(f"Nemotron VoiceChat tool {index} requires a name")
        normalized.append(definition)
    signature = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return normalized, signature


def _render_tool_prompt(instructions: str, tools: list[dict[str, object]]) -> str:
    if not tools:
        return instructions
    available = json.dumps(tools, ensure_ascii=False, sort_keys=True)
    return (
        f"{instructions}\n\n"
        "You can use the following tools to assist the user if required:"
        f"\n<AVAILABLE_TOOLS>{available}</AVAILABLE_TOOLS>\n\n"
        "If you decide to call any tool(s), use the following format:\n"
        '<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}, '
        '{"name": "tool_name2", "arguments": "tool_args2"}]</TOOLCALL>\n\n'
        "The user will execute tool-calls and return responses from tool(s) in this format:\n"
        '<TOOL_RESPONSE>[{"tool_response1"}, {"tool_response2"}]</TOOL_RESPONSE>\n\n'
        "Based on the tool responses, you can call additional tools if needed, correct tool calls if any "
        "errors are found, or just respond to the user."
    )


def _render_tool_response(output: str) -> str:
    """Serialize one Realtime function result using NVIDIA's function channel."""
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        value = output
    payload = json.dumps([value], ensure_ascii=True, separators=(",", ":"))
    return f"<TOOL_RESPONSE>{payload}</TOOL_RESPONSE>"


def _require_native_full_duplex(config: object) -> None:
    extra_body = getattr(config, "extra_body", None)
    enabled = isinstance(extra_body, dict) and (
        extra_body.get("auto_response") is True or extra_body.get("full_duplex") is True
    )
    if not enabled:
        raise ServingRuntimeConfigError(
            "Nemotron VoiceChat currently supports model-native full-duplex streaming only; "
            "set extra_body.auto_response=true",
            code="unsupported_nemotron_duplex_mode",
        )


def _tokenize_runtime(model_config: Any, instructions: str) -> tuple[dict[str, object], Any]:
    from transformers import AutoTokenizer

    stt_cfg = _stt_config(model_config)
    tokenizer_ref = os.environ.get("NEMOTRON_VOICECHAT_LLM_PATH") or stt_cfg.get(
        "pretrained_llm", "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=False)

    def token(name: str, default: str) -> int:
        value = tokenizer.convert_tokens_to_ids(stt_cfg.get(name, default))
        if value is None:
            raise ServingRuntimeConfigError(f"Nemotron VoiceChat tokenizer does not define {name}")
        return int(value)

    bos_id = token("bos_token", "<s>")
    eos_id = token("eos_token", "</s>")
    pad_id = token("pad_token", "<SPECIAL_12>")
    prompt_ids = [bos_id] + list(tokenizer.encode(instructions, add_special_tokens=False)) + [eos_id]
    runtime = {
        "instructions": instructions,
        "nvc_prompt_token_ids": prompt_ids,
        "nvc_text_bos_id": bos_id,
        "nvc_text_eos_id": eos_id,
        "nvc_text_pad_id": pad_id,
        "nvc_function_sotc_id": int(tokenizer.convert_tokens_to_ids("<SPECIAL_20>")),
        "nvc_function_eotc_id": int(tokenizer.convert_tokens_to_ids("<SPECIAL_21>")),
        "nvc_function_eotr_id": int(tokenizer.convert_tokens_to_ids("<SPECIAL_22>")),
        "nvc_tokenizer_ref": str(tokenizer_ref),
    }
    return runtime, tokenizer


class NemotronVoiceChatServingRuntimeAdapter:
    adapter_id = "nemotron_voicechat"
    silence_continuation_samples = 1280
    # The session-owned persistent drain is the sole data-plane output
    # consumer. Collecting on append would race that drain and can consume a
    # Stage2 audio output before the realtime projector observes it.
    collect_outputs_on_append = False
    clean_response_done_prefix = ""
    interrupted_tts_prefix = ""
    private_runtime_config_keys = _PRIVATE_KEYS

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self.session_states: dict[str, MiniCPMO45ServingSessionState] = {}
        self.data_plane = NemotronVoiceChatDataPlaneSession(encode_audio)
        self._tokenizer: Any | None = None

    def create_session_state(self) -> MiniCPMO45ServingSessionState:
        return MiniCPMO45ServingSessionState(
            audio_buffer=NemotronVoiceChatPcmAppendBuffer(),
        )

    def session_state(self, session_id: str) -> MiniCPMO45ServingSessionState:
        return self.session_states.setdefault(session_id, self.create_session_state())

    def remove_session_state(self, session_id: str) -> None:
        self.session_states.pop(session_id, None)

    @staticmethod
    def is_enabled(config: object) -> bool:
        del config
        return True

    @staticmethod
    def capabilities(*, max_sessions: int) -> DuplexCapabilities:
        supports_multi_session = max_sessions > 1
        return DuplexCapabilities(
            supports_model_native_turn_policy=True,
            supports_external_turn_signal=False,
            supports_client_commit=True,
            supports_barge_in=False,
            supports_playback_ack=True,
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=False,
            supports_rollback_to_checkpoint=False,
            supports_turn_commit_only=False,
            supports_kv_lease=False,
            supports_core_kv_lease=False,
            supports_model_internal_state=True,
            supports_stage_resumption=True,
            supports_scheduler_native_append=False,
            supports_core_resumable_request=True,
            supports_stage_connector_handoff=True,
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=supports_multi_session,
            supports_multi_session_same_replica=False,
            supports_session_lease=True,
            supports_session_resume=True,
            session_admission_mode="engine_managed",
            supports_audio_truncate=False,
            requires_model_runner_kv=True,
            requires_native_stage_role=True,
            implementation_level="model_native_duplex",
            adapter_patterns=["scheduler_data_plane"],
            input_modes=["append_audio_chunk"],
            signal_sources=["model_native", "client_event"],
            stage_handoff_transport="scheduler_data_plane",
            chunk_period_ms=80,
            target_barge_in_latency_ms=None,
        )

    @staticmethod
    def validate_client_extra_body(extra_body: object) -> None:
        if not isinstance(extra_body, dict):
            return
        private = sorted(_PRIVATE_KEYS.intersection(extra_body))
        if private:
            raise ServingRuntimeConfigError(
                "Nemotron VoiceChat runtime configuration is server-owned: " + ", ".join(private)
            )

    async def prepare_runtime_config(
        self,
        config: object,
        *,
        model_config: Any,
    ) -> dict[str, object]:
        self.validate_client_extra_body(getattr(config, "extra_body", None))
        _require_native_full_duplex(config)
        instructions = str(getattr(config, "instructions", None) or _DEFAULT_SYSTEM_PROMPT)
        tools, tools_signature = _normalized_tools(config)
        rendered_prompt = _render_tool_prompt(instructions, tools)
        runtime, tokenizer = await asyncio.to_thread(_tokenize_runtime, model_config, rendered_prompt)
        max_model_len = getattr(model_config, "max_model_len", None)
        if not isinstance(max_model_len, int) or max_model_len <= 0:
            raise ServingRuntimeConfigError("Nemotron VoiceChat requires a positive Stage-0 max_model_len")
        runtime["nvc_max_model_len"] = max_model_len
        # Keep raw session values for immutable-in-incarnation update checks;
        # only the rendered prompt token ids enter Stage-0 KV.
        runtime["instructions"] = instructions
        runtime["nvc_tools_signature"] = tools_signature
        self.data_plane.configure_runtime(runtime, tokenizer=tokenizer)
        self._tokenizer = tokenizer
        return runtime

    def runtime_config_for_function_output(
        self,
        item: Mapping[str, object],
        current: Mapping[str, object],
    ) -> dict[str, object]:
        """Queue a client tool result for frame-locked function-channel injection."""
        call_id = item.get("call_id")
        output = item.get("output")
        if not isinstance(call_id, str) or not call_id:
            raise ServingRuntimeConfigError(
                "function_call_output requires call_id",
                code="invalid_function_call_output",
            )
        if not isinstance(output, str):
            raise ServingRuntimeConfigError(
                "function_call_output requires a string output",
                code="invalid_function_call_output",
            )
        if self._tokenizer is None:
            raise ServingRuntimeConfigError(
                "Nemotron VoiceChat tokenizer is unavailable for function output",
                code="invalid_function_call_output",
            )
        response_token_ids = list(
            self._tokenizer.encode(
                _render_tool_response(output),
                add_special_tokens=False,
            )
        )
        if not response_token_ids:
            raise ServingRuntimeConfigError(
                "function_call_output tokenized to an empty response",
                code="invalid_function_call_output",
            )
        runtime = deepcopy(dict(current))
        generation = int(runtime.get("nvc_function_response_generation", 0)) + 1
        token_ids = [int(token_id) for token_id in response_token_ids]
        batches = runtime.get("nvc_function_response_batches")
        if not isinstance(batches, list):
            batches = []
        batches = [*batches, {"generation": generation, "call_id": call_id, "token_ids": token_ids}]
        runtime["nvc_function_response_generation"] = generation
        runtime["nvc_function_response_token_ids"] = token_ids
        runtime["nvc_function_response_call_id"] = call_id
        runtime["nvc_function_response_batches"] = batches
        return runtime

    @staticmethod
    def runtime_config_for_update(
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        NemotronVoiceChatServingRuntimeAdapter.validate_client_extra_body(getattr(config, "extra_body", None))
        _require_native_full_duplex(config)
        runtime = deepcopy(dict(current))
        instructions = str(getattr(config, "instructions", None) or _DEFAULT_SYSTEM_PROMPT)
        _, tools_signature = _normalized_tools(config)
        # The system/tool prompt is already resident in Stage-0 KV. Updating it
        # without a new incarnation would make the advertised config disagree
        # with model state, so reject such updates explicitly.
        if runtime and instructions != runtime.get("instructions"):
            raise ServingRuntimeConfigError(
                "Nemotron VoiceChat instructions cannot change inside an active duplex incarnation"
            )
        if runtime and tools_signature != runtime.get("nvc_tools_signature", "[]"):
            raise ServingRuntimeConfigError(
                "Nemotron VoiceChat tools cannot change inside an active duplex incarnation"
            )
        return runtime

    @staticmethod
    def data_plane_context(
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> NemotronVoiceChatDataPlaneContext:
        del active_response_turn_id, active_response_id
        return NemotronVoiceChatDataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )


__all__ = ["NemotronVoiceChatServingRuntimeAdapter"]

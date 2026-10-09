# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Nemotron VoiceChat full-duplex model plugin: engine policy and session policy in one class.

Engine policy (frame-locked resumable Stage-0 prompt, greedy 1-token sampling,
Stage-2 direct audio output) and session policy (tool rendering, function-channel
injection, per-session PCM state) both run engine-side, so one class owns them.

Timeline contract (see ``nemotron_voicechat_thinker.py``): the Stage-0 prompt is
``[bos] + instructions + [eos]`` with the PAD text token fused over acoustic
frame 0; every scheduler append afterwards carries exactly one 80 ms acoustic
frame and decodes exactly one token. ``ignore_eos`` keeps the request alive;
the tokenizer's EOS *is* the model's "end of speech" signal, and the PAD token
(``<SPECIAL_12>``) its "keep listening" signal.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputAction,
    DuplexOutputDecision,
)
from vllm_omni.engine.duplex.intermediate import build_duplex_append_prompt
from vllm_omni.engine.duplex.plugin import (
    DefaultDuplexModelSessionState,
    DuplexModelPlugin,
    DuplexRuntimeConfigError,
    EncodeAudio,
    reject_private_runtime_keys,
)
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.capabilities import (
    nemotron_voicechat_capabilities,
)
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.data_plane import (
    NemotronVoiceChatDataPlaneContext,
    NemotronVoiceChatDataPlaneSession,
)
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.input import (
    NEMOTRON_VOICECHAT_FRAME_SAMPLES,
    NemotronVoiceChatPcmAppendBuffer,
    decode_pcm_f32le,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from vllm.config import ModelConfig

_DEFAULT_SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again. "
    "Start the conversation by greeting the user."
)

#: Server-owned runtime_config keys a client must not override through extra_body.
PRIVATE_RUNTIME_CONFIG_KEYS = frozenset(
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
        "nvc_function_response_batches",
    }
)


class NemotronVoiceChatClientRuntimeConfigError(DuplexRuntimeConfigError):
    pass


#: Bound batches awaiting worker ownership so per-append runtime snapshots
#: stay bounded. Once acknowledged, their tokens remain in the model-owned
#: queue and drain at one forced token per 80 ms frame.
_MAX_PENDING_FUNCTION_RESPONSE_BATCHES = 8


def _stt_config(model_config: Any) -> dict[str, Any]:
    hf_config = getattr(model_config, "hf_config", None)
    stt_cfg = getattr(hf_config, "stt_cfg", None)
    if not isinstance(stt_cfg, dict):
        raise NemotronVoiceChatClientRuntimeConfigError(
            "Nemotron VoiceChat checkpoint STT configuration is unavailable"
        )
    return stt_cfg


def _normalized_tools(config: DuplexSessionConfig) -> tuple[list[dict[str, object]], str]:
    raw_tools = config.extra_body.get("realtime_tools")
    if raw_tools is None:
        return [], "[]"
    if not isinstance(raw_tools, list):
        raise NemotronVoiceChatClientRuntimeConfigError("Nemotron VoiceChat tools must be a list")
    if len(raw_tools) > 5:
        raise NemotronVoiceChatClientRuntimeConfigError("Nemotron VoiceChat supports at most 5 tools per session")

    normalized: list[dict[str, object]] = []
    for index, tool in enumerate(raw_tools):
        if not isinstance(tool, dict):
            raise NemotronVoiceChatClientRuntimeConfigError(f"Nemotron VoiceChat tool {index} must be an object")
        function = tool.get("function", tool)
        if not isinstance(function, dict):
            raise NemotronVoiceChatClientRuntimeConfigError(
                f"Nemotron VoiceChat tool {index} has no function definition"
            )
        definition = {key: value for key, value in function.items() if key != "type"}
        if not isinstance(definition.get("name"), str) or not str(definition["name"]).strip():
            raise NemotronVoiceChatClientRuntimeConfigError(f"Nemotron VoiceChat tool {index} requires a name")
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


def _require_native_full_duplex(config: DuplexSessionConfig) -> None:
    extra_body = config.extra_body
    enabled = extra_body.get("auto_response") is True or extra_body.get("full_duplex") is True
    if not enabled:
        raise NemotronVoiceChatClientRuntimeConfigError(
            "Nemotron VoiceChat currently supports model-native full-duplex streaming only; "
            "set extra_body.auto_response=true",
            code="unsupported_nemotron_duplex_mode",
        )


def _load_tokenizer_runtime(model_config: Any) -> tuple[dict[str, object], Any]:
    """Load the session tokenizer and resolve the checkpoint's special-token ids."""
    from transformers import AutoTokenizer

    stt_cfg = _stt_config(model_config)
    tokenizer_ref = os.environ.get("NEMOTRON_VOICECHAT_LLM_PATH") or stt_cfg.get(
        "pretrained_llm", "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=False)

    def token(name: str, default: str) -> int:
        value = tokenizer.convert_tokens_to_ids(stt_cfg.get(name, default))
        if value is None:
            raise NemotronVoiceChatClientRuntimeConfigError(f"Nemotron VoiceChat tokenizer does not define {name}")
        return int(value)

    runtime = {
        "nvc_text_bos_id": token("bos_token", "<s>"),
        "nvc_text_eos_id": token("eos_token", "</s>"),
        "nvc_text_pad_id": token("pad_token", "<SPECIAL_12>"),
        "nvc_function_sotc_id": int(tokenizer.convert_tokens_to_ids("<SPECIAL_20>")),
        "nvc_function_eotc_id": int(tokenizer.convert_tokens_to_ids("<SPECIAL_21>")),
        "nvc_function_eotr_id": int(tokenizer.convert_tokens_to_ids("<SPECIAL_22>")),
        "nvc_tokenizer_ref": str(tokenizer_ref),
    }
    return runtime, tokenizer


def _render_prompt_ids(
    tokenizer: PreTrainedTokenizerBase,
    rendered_prompt: str,
    bos_id: int,
    eos_id: int,
) -> list[int]:
    return [bos_id] + list(tokenizer.encode(rendered_prompt, add_special_tokens=False)) + [eos_id]


def _plain_token_ids(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"Nemotron VoiceChat runtime requires non-empty {name}")
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Nemotron VoiceChat {name} must contain token ids") from exc


def _positive_int(value: object, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Nemotron VoiceChat requires a positive {name}") from exc
    if parsed <= 0:
        raise ValueError(f"Nemotron VoiceChat requires a positive {name}")
    return parsed


class NemotronVoiceChatDuplexPlugin(DuplexModelPlugin):
    """Nemotron-owned sampling policy, append planning, session state and output projection."""

    plugin_id = "nemotron_voicechat"
    private_runtime_config_keys = PRIVATE_RUNTIME_CONFIG_KEYS
    #: One silence continuation unit is one 80 ms acoustic frame at 16 kHz.
    silence_continuation_samples = NEMOTRON_VOICECHAT_FRAME_SAMPLES

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = NemotronVoiceChatDataPlaneSession(encode_audio)
        # The tokenizer only resolves the special-token ids and renders tool
        # prompts; it is the same for every session of a checkpoint, so load it
        # once per tokenizer ref, off the orchestrator loop.
        self._tokenizers: dict[str, Any] = {}
        self._tokenizer_lock = asyncio.Lock()

    # ---- engine policy (the frame-locked resumable Stage-0 request) ----

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        if not defaults:
            return defaults
        # Every scheduler segment is one transport wake of a long-lived
        # request.  CUMULATIVE output makes the multimodal output processor
        # concatenate all earlier Stage-2 waveforms and resend them on every
        # wake (80, 160, 240, ... ms), causing O(n^2) audio replay.  Duplex
        # consumers require per-wake deltas at every stage.
        configured = [params.clone() if isinstance(params, SamplingParams) else params for params in defaults]
        for params in configured:
            if isinstance(params, SamplingParams):
                params.output_kind = RequestOutputKind.DELTA
        stage0 = configured[0]
        if isinstance(stage0, SamplingParams):
            stage0.temperature = 0.0
            stage0.top_p = 1.0
            stage0.top_k = 0
            stage0.max_tokens = 1
            stage0.ignore_eos = True
        return tuple(configured)

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
        decode_pcm_f32le(payload, exact_frame=True)
        normalized_payload = dict(payload)
        prompt_ids = _plain_token_ids(
            runtime_config.get("nvc_prompt_token_ids"),
            name="nvc_prompt_token_ids",
        )
        max_model_len = _positive_int(
            runtime_config.get("nvc_max_model_len"),
            name="nvc_max_model_len",
        )
        # ``seq`` is the per-epoch frame counter; the worker maps it 1:1 onto
        # timeline positions of the resumable Stage-0 request.
        required_model_len = len(prompt_ids) + seq + 1
        if required_model_len > max_model_len:
            raise ValueError(
                "Nemotron VoiceChat native duplex session exceeds the Stage-0 "
                f"max_model_len: prompt_tokens={len(prompt_ids)} + input_frames={seq} + "
                f"sampled_token=1 gives {required_model_len} > {max_model_len}; "
                "start a new session or raise Stage 0 max_model_len"
            )
        try:
            pad_id = int(runtime_config["nvc_text_pad_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Nemotron VoiceChat runtime requires nvc_text_pad_id") from exc
        scheduler_prompt = prompt_ids + [pad_id] if seq <= 1 else [pad_id]
        return DuplexAppendPlan(
            prompt=build_duplex_append_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                payload=normalized_payload,
                final=final,
                prompt_token_ids=scheduler_prompt,
                model_fields={"source_input_seq": seq},
            )
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None:
        del final_stage_id, segment_finished, output
        if stage_id != 0:
            return None
        metadata = dict(segment_output_metadata)
        metadata["nvc_text_token_ids"] = list(segment_token_ids)
        # Stage 0 is a client-visible side channel even though Stage 2 is the
        # configured audio response stage. Mark it explicitly so the shared
        # collector does not discard it while waiting for the final stage.
        metadata["duplex_direct_response"] = True
        return DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata=metadata,
            final_output_type="text",
        )

    # ---- session policy ----

    def create_session_state(self) -> DefaultDuplexModelSessionState:
        return DefaultDuplexModelSessionState(audio_buffer=NemotronVoiceChatPcmAppendBuffer())

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return nemotron_voicechat_capabilities(max_sessions=max_sessions)

    def validate_client_extra_body(self, extra_body: object) -> None:
        reject_private_runtime_keys(
            extra_body,
            self.private_runtime_config_keys,
            message="Nemotron VoiceChat runtime configuration is server-owned: ",
            error_cls=NemotronVoiceChatClientRuntimeConfigError,
        )

    async def prepare_runtime_config(
        self,
        config: DuplexSessionConfig,
        *,
        model_config: ModelConfig | None,
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        _require_native_full_duplex(config)
        instructions = str(config.instructions or _DEFAULT_SYSTEM_PROMPT)
        tools, tools_signature = _normalized_tools(config)
        rendered_prompt = _render_tool_prompt(instructions, tools)
        special, tokenizer = await self._tokenizer_runtime(model_config)
        prompt_ids = await asyncio.to_thread(
            _render_prompt_ids,
            tokenizer,
            rendered_prompt,
            int(special["nvc_text_bos_id"]),
            int(special["nvc_text_eos_id"]),
        )
        max_model_len = getattr(model_config, "max_model_len", None)
        if not isinstance(max_model_len, int) or max_model_len <= 0:
            raise NemotronVoiceChatClientRuntimeConfigError(
                "Nemotron VoiceChat requires a positive Stage-0 max_model_len"
            )
        runtime: dict[str, object] = dict(special)
        runtime["nvc_prompt_token_ids"] = prompt_ids
        runtime["nvc_max_model_len"] = max_model_len
        # Keep raw session values for immutable-in-epoch update checks; only
        # the rendered prompt token ids enter Stage-0 KV.
        runtime["instructions"] = instructions
        runtime["nvc_tools_signature"] = tools_signature
        self.data_plane.configure_runtime(runtime, tokenizer=tokenizer)
        return runtime

    async def _tokenizer_runtime(self, model_config: ModelConfig | None) -> tuple[dict[str, object], Any]:
        """Return the (special ids, tokenizer) pair for this checkpoint, cached per tokenizer ref."""
        stt_cfg = _stt_config(model_config)
        tokenizer_ref = str(
            os.environ.get("NEMOTRON_VOICECHAT_LLM_PATH")
            or stt_cfg.get("pretrained_llm", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
        )
        cached = self._tokenizers.get(tokenizer_ref)
        if cached is not None:
            return cached
        async with self._tokenizer_lock:
            cached = self._tokenizers.get(tokenizer_ref)
            if cached is None:
                cached = await asyncio.to_thread(_load_tokenizer_runtime, model_config)
                self._tokenizers[tokenizer_ref] = cached
        return cached

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        _require_native_full_duplex(config)
        runtime = deepcopy(dict(current))
        instructions = str(config.instructions or _DEFAULT_SYSTEM_PROMPT)
        _, tools_signature = _normalized_tools(config)
        # The system/tool prompt is already resident in Stage-0 KV. Updating it
        # without a new epoch would make the advertised config disagree with
        # model state, so reject such updates explicitly.
        if runtime and instructions != runtime.get("instructions"):
            raise NemotronVoiceChatClientRuntimeConfigError(
                "Nemotron VoiceChat instructions cannot change inside an active duplex session"
            )
        if runtime and tools_signature != runtime.get("nvc_tools_signature", "[]"):
            raise NemotronVoiceChatClientRuntimeConfigError(
                "Nemotron VoiceChat tools cannot change inside an active duplex session"
            )
        return runtime

    def runtime_config_for_function_output(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
        item: Mapping[str, object],
    ) -> dict[str, object]:
        """Queue a client tool result for frame-locked function-channel injection."""
        del config
        call_id = item.get("call_id")
        output = item.get("output")
        if not isinstance(call_id, str) or not call_id:
            raise NemotronVoiceChatClientRuntimeConfigError(
                "function_call_output requires call_id",
                code="invalid_function_call_output",
            )
        if not isinstance(output, str):
            raise NemotronVoiceChatClientRuntimeConfigError(
                "function_call_output requires a string output",
                code="invalid_function_call_output",
            )
        tokenizer_ref = current.get("nvc_tokenizer_ref")
        cached = self._tokenizers.get(str(tokenizer_ref)) if tokenizer_ref else None
        tokenizer = cached[1] if cached is not None else None
        if tokenizer is None:
            raise NemotronVoiceChatClientRuntimeConfigError(
                "Nemotron VoiceChat tokenizer is unavailable for function output",
                code="invalid_function_call_output",
            )
        response_token_ids = list(
            tokenizer.encode(
                _render_tool_response(output),
                add_special_tokens=False,
            )
        )
        if not response_token_ids:
            raise NemotronVoiceChatClientRuntimeConfigError(
                "function_call_output tokenized to an empty response",
                code="invalid_function_call_output",
            )
        runtime = deepcopy(dict(current))
        generation = int(runtime.get("nvc_function_response_generation", 0)) + 1
        token_ids = [int(token_id) for token_id in response_token_ids]
        batches = runtime.get("nvc_function_response_batches")
        if not isinstance(batches, list):
            batches = []
        if len(batches) >= _MAX_PENDING_FUNCTION_RESPONSE_BATCHES:
            raise NemotronVoiceChatClientRuntimeConfigError(
                "Nemotron VoiceChat received function results faster than the frame-locked "
                f"channel drains them (>{_MAX_PENDING_FUNCTION_RESPONSE_BATCHES} pending batches); "
                "return tool results after the model has consumed the previous one",
                code="function_response_backlog",
            )
        batches = [*batches, {"generation": generation, "token_ids": token_ids}]
        runtime["nvc_function_response_generation"] = generation
        runtime["nvc_function_response_batches"] = batches
        return runtime

    def runtime_config_after_model_output(
        self,
        current: Mapping[str, object],
        output_metadata: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Retire batches after the thinker takes ownership of their tokens."""
        consumed = output_metadata.get("nvc_function_response_consumed_generation")
        try:
            consumed_generation = int(consumed)
        except (TypeError, ValueError):
            return None
        if consumed_generation <= 0:
            return None
        batches = current.get("nvc_function_response_batches")
        if not isinstance(batches, list):
            return None
        remaining = [
            batch
            for batch in batches
            if isinstance(batch, dict) and int(batch.get("generation", 0)) > consumed_generation
        ]
        if len(remaining) == len(batches):
            return None
        runtime = deepcopy(dict(current))
        runtime["nvc_function_response_batches"] = remaining
        return runtime

    def data_plane_context(
        self,
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


__all__ = [
    "PRIVATE_RUNTIME_CONFIG_KEYS",
    "NemotronVoiceChatClientRuntimeConfigError",
    "NemotronVoiceChatDuplexPlugin",
]

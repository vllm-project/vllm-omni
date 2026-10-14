# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in Qwen3-Omni turn-based chat fallback for the duplex engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence, DuplexOutputDecision
from vllm_omni.engine.duplex.plugin import (
    DuplexModelPlugin,
    DuplexModelSessionState,
    DuplexRuntimeConfigError,
    EncodeAudio,
)
from vllm_omni.model_executor.models.qwen3_omni.duplex.data_plane import (
    Qwen3OmniDataPlaneContext,
    Qwen3OmniDataPlaneSession,
)
from vllm_omni.model_executor.models.qwen3_omni.duplex.fallback_session import Qwen3OmniFallbackSessionState
from vllm_omni.model_executor.models.qwen3_omni.duplex.policy import INTERRUPTION_NOTE, SYSTEM_PROMPT

if TYPE_CHECKING:
    from vllm.config import ModelConfig

PRIVATE_RUNTIME_CONFIG_KEYS = frozenset({"auto_commit_silence_ms"})


class Qwen3OmniClientRuntimeConfigError(DuplexRuntimeConfigError):
    """A client attempted to change a server-owned Qwen3 runtime setting."""


class Qwen3OmniChatFallbackPlugin(DuplexModelPlugin):
    """Bind chat fallback only to the explicit Qwen3 fallback pipeline."""

    plugin_id = "qwen3omni_chat_fallback"
    private_runtime_config_keys = PRIVATE_RUNTIME_CONFIG_KEYS
    default_auto_response = True

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = Qwen3OmniDataPlaneSession(encode_audio)

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        return tuple(defaults)

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
        del request_id, fence, session_config, runtime_config, seq, turn_seq, payload, final, sampling_params
        raise RuntimeError("Qwen3-Omni uses the chat fallback; native append is disabled")

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
        del stage_id, final_stage_id, segment_finished, segment_token_ids, segment_output_metadata, output
        return None

    def create_session_state(self) -> DuplexModelSessionState:
        return Qwen3OmniFallbackSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        del max_sessions
        return DuplexCapabilities(
            supports_model_native_turn_policy=False,
            supports_external_turn_signal=True,
            supports_client_commit=True,
            supports_barge_in=True,
            supports_playback_ack=True,
            supports_input_append=False,
            supports_replace_latest_chunk=False,
            supports_reencode_context=False,
            supports_turn_commit_only=True,
            supports_realtime_endpoint=True,
            supports_multi_session=False,
            supports_multi_session_same_replica=False,
            supports_session_lease=False,
            supports_session_resume=False,
            supports_chat_completions=True,
            supports_chat_fallback=True,
            implementation_level="chat_fallback",
            adapter_patterns=["turn_commit_only", "chat_fallback"],
            input_modes=["turn_commit_only"],
            signal_sources=["client_event", "server_policy"],
            chunk_period_ms=None,
            target_barge_in_latency_ms=1000,
        )

    def validate_client_extra_body(self, extra_body: object) -> None:
        if not isinstance(extra_body, Mapping):
            return
        private_keys = sorted(PRIVATE_RUNTIME_CONFIG_KEYS.intersection(extra_body))
        if private_keys:
            raise Qwen3OmniClientRuntimeConfigError(
                "duplex runtime configuration is server-owned: " + ", ".join(private_keys),
                code="private_runtime_config",
            )
        self._validate_tools(extra_body)

    @staticmethod
    def _validate_tools(extra_body: Mapping[str, object]) -> None:
        if any(
            extra_body.get(key)
            for key in (
                "tools",
                "tool_choice",
                "realtime_tools",
                "realtime_tool_choice",
                "realtime_response_tools",
                "realtime_response_tool_choice",
            )
        ):
            raise DuplexRuntimeConfigError("Qwen duplex tool calls are not implemented", code="unsupported_tools")

    async def prepare_runtime_config(
        self,
        config: DuplexSessionConfig,
        *,
        model_config: ModelConfig | None,
    ) -> dict[str, object]:
        del model_config
        return self.runtime_config_for_update(config, {})

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        source: object = config if isinstance(config, Mapping) else config.extra_body
        merged = dict(current)
        if isinstance(source, Mapping):
            self._validate_tools(source)
            for key in PRIVATE_RUNTIME_CONFIG_KEYS:
                if key in source:
                    merged[key] = source[key]
        return merged

    def runtime_config_for_function_output(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
        item: Mapping[str, object],
    ) -> dict[str, object] | None:
        del config, current, item
        raise DuplexRuntimeConfigError("Qwen duplex tool calls are not implemented", code="unsupported_tools")

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
    ) -> Qwen3OmniDataPlaneContext:
        return Qwen3OmniDataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            active_response_turn_id=active_response_turn_id,
            active_response_id=active_response_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )

    def fallback_policy_messages(self, state: DuplexModelSessionState) -> tuple[Mapping[str, object], ...]:
        messages: list[Mapping[str, object]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if isinstance(state, Qwen3OmniFallbackSessionState) and state.last_turn_interrupted:
            messages.append({"role": "system", "content": INTERRUPTION_NOTE})
        return tuple(messages)

    def on_fallback_started(self, state: DuplexModelSessionState) -> None:
        if isinstance(state, Qwen3OmniFallbackSessionState):
            state.last_turn_interrupted = False

    def on_barge_in(self, state: DuplexModelSessionState) -> None:
        if isinstance(state, Qwen3OmniFallbackSessionState):
            state.last_turn_interrupted = True

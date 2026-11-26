# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Venus delegation over the shared MiniCPM native-duplex execution path."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, TypedDict, cast
from uuid import uuid4

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence, ToolCallAbandonReason
from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext, DuplexRuntimeConfigError, EncodeAudio
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import MiniCPMO45DataPlaneSession
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import TOKENIZER_KEY, MiniCPMO45DuplexPlugin
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import APPEND_TEXT_TOKEN_IDS_KEY
from vllm_omni.model_executor.models.realtime_venus.delegation import DelegateParser

if TYPE_CHECKING:
    from vllm.config import ModelConfig

_RESULTS = "venus_function_results"
_GENERATION = "venus_function_generation"
_SUBMITTED = "_venus_submitted_result_generation"
_MAX_RESULT_CHARS = 4000
_MAX_QUEUED_TOKENS = 4096


class _FunctionResult(TypedDict):
    generation: int
    token_ids: list[int]


def _records(runtime: Mapping[str, object]) -> list[_FunctionResult]:
    # This private runtime key is written only by this plugin.
    return cast(list[_FunctionResult], runtime.get(_RESULTS, []))


class VenusDataPlane(MiniCPMO45DataPlaneSession):
    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self._delegations: dict[str, tuple[object, DelegateParser]] = {}
        self._backend_ready: dict[str, bool] = {}

    def backend_ready(self, request_id: str) -> bool:
        return self._backend_ready.get(request_id, True)

    def close_request(self, request_id: str) -> None:
        self._delegations.pop(request_id, None)
        self._backend_ready.pop(request_id, None)
        super().close_request(request_id)

    def project_output(
        self, output: object, *, context: DuplexDataPlaneContext | None = None
    ) -> Iterator[dict[str, object]]:
        for result in super().project_output(output, context=context):
            request_id = result.get("data_plane_request_id")
            if isinstance(request_id, str):
                # A closed delegate span can precede the foreground turn's
                # final tokens. Injecting its result before that boundary
                # lets the old turn's EOS consume the new answer context.
                # Synthetic segment-complete listens only schedule another
                # unit; they do not close the model's foreground turn.
                self._backend_ready[request_id] = (
                    result.get("model_listen") is True or result.get("end_of_turn") is True
                )
                turn_id = result.get("model_turn_id", context.turn_id if context is not None else None)
                previous = self._delegations.get(request_id)
                if previous is None or previous[0] != turn_id:
                    previous = (turn_id, DelegateParser())
                    self._delegations[request_id] = previous
                text = result.get("text")
                completed_before = previous[1].completed_spans
                try:
                    query = previous[1].feed(
                        text if isinstance(text, str) else "", end_of_turn=result.get("end_of_turn") is True
                    )
                except ValueError as exc:
                    yield {"data_plane_request_id": request_id, "error_code": "invalid_delegate", "error": str(exc)}
                else:
                    if completed_before and previous[1].completed_spans > completed_before:
                        # A model waiting for its backend may keep repeating
                        # the completed task instead of emitting turn_eos.
                        # Resume at that control boundary without executing it
                        # again or stranding an already accepted result.
                        self._backend_ready[request_id] = True
                    if query is not None:
                        yield {
                            "data_plane_request_id": request_id,
                            "function_call": True,
                            "call_id": f"call_{uuid4().hex}",
                            "name": "delegate",
                            "arguments": json.dumps({"question": query}, ensure_ascii=False),
                        }
            # Keep the existing text/audio cursor and turn-boundary semantics.
            yield result


class RealtimeVenusDuplexPlugin(MiniCPMO45DuplexPlugin):
    plugin_id = "realtime_venus_omni"
    private_runtime_config_keys = MiniCPMO45DuplexPlugin.private_runtime_config_keys | {
        _RESULTS,
        _GENERATION,
    }
    abandoned_tool_call_reasons = frozenset(
        {
            ToolCallAbandonReason.TIMEOUT,
            ToolCallAbandonReason.CLIENT_CANCELLED,
            ToolCallAbandonReason.CLIENT_FAILED,
            ToolCallAbandonReason.SUPPRESSED,
        }
    )

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = VenusDataPlane(encode_audio)

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return replace(
            super().capabilities(max_sessions=max_sessions),
            tool_mode="delegate_only",
            max_tools=1,
            function_call_output_continuation="automatic",
        )

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        return await super().prepare_runtime_config(config, model_config=model_config)

    def validate_client_extra_body(self, extra_body: object) -> None:
        super().validate_client_extra_body(extra_body)
        tools = extra_body.get("realtime_tools") if isinstance(extra_body, Mapping) else None
        if tools and (
            not isinstance(tools, list)
            or len(tools) != 1
            or not isinstance(tools[0], Mapping)
            or tools[0].get("type", "function") != "function"
            or tools[0].get("name") != "delegate"
        ):
            raise DuplexRuntimeConfigError("Venus only supports the delegate tool", code="tools_unsupported")

    def runtime_config_for_update(
        self, config: DuplexSessionConfig, current: Mapping[str, object]
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        return super().runtime_config_for_update(config, current)

    def runtime_config_for_function_output(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
        item: Mapping[str, object],
        *,
        history_len: int | None = None,
    ) -> dict[str, object]:
        del config, history_len
        output = item.get("output")
        if item.get("name") != "delegate" or not isinstance(output, str):
            raise DuplexRuntimeConfigError(
                "expected a delegate function_call_output", code="invalid_function_call_output"
            )
        if len(output) > _MAX_RESULT_CHARS:
            raise DuplexRuntimeConfigError(
                "delegate output exceeds 4000 characters", code="invalid_function_call_output"
            )
        tokenizer = self._tokenizers.get(str(current.get(TOKENIZER_KEY, "")))
        if tokenizer is None:
            raise DuplexRuntimeConfigError("Venus tokenizer is unavailable", code="invalid_function_call_output")
        # Quoted protocol tags in tool data must remain ordinary text. Encode
        # the trusted frame separately so only its delimiters are control IDs.
        token_ids = [
            *tokenizer.encode("<backend>", add_special_tokens=False),
            *tokenizer.encode(output, add_special_tokens=False, split_special_tokens=True),
            *tokenizer.encode("</backend>", add_special_tokens=False),
        ]
        records = _records(current)
        if (
            len(records) >= 8
            or sum(len(record["token_ids"]) for record in records) + len(token_ids) > _MAX_QUEUED_TOKENS
        ):
            raise DuplexRuntimeConfigError(
                "delegate results exceed the pending token budget", code="function_response_backlog"
            )
        generation = cast(int, current.get(_GENERATION, 0)) + 1
        return {
            **current,
            _GENERATION: generation,
            _RESULTS: [*records, {"generation": generation, "token_ids": token_ids}],
        }

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
        records = _records(runtime_config) if self.data_plane.backend_ready(request_id) else []
        if records:
            runtime_config = {
                **runtime_config,
                APPEND_TEXT_TOKEN_IDS_KEY: [token_id for record in records for token_id in record["token_ids"]],
            }
        plan = super().plan_append(
            request_id=request_id,
            fence=fence,
            session_config=session_config,
            runtime_config=runtime_config,
            seq=seq,
            turn_seq=turn_seq,
            payload=payload,
            final=final,
            sampling_params=sampling_params,
        )
        if records:
            plan.prompt[_SUBMITTED] = records[-1]["generation"]
        return plan

    def runtime_patch_for_submitted_prompt(self, prompt: dict[str, object]) -> Mapping[str, object] | None:
        generation = prompt.pop(_SUBMITTED, None)
        return {_SUBMITTED: generation} if generation is not None else None

    def reconcile_submitted_runtime(
        self, current: Mapping[str, object], updated: Mapping[str, object]
    ) -> Mapping[str, object]:
        # Transfer ownership only after successful submission. Preserve results
        # accepted while prompt preparation/submission was awaiting the stage.
        generation = cast(int, updated[_SUBMITTED])
        return {**current, _RESULTS: [record for record in _records(current) if record["generation"] > generation]}

    def retire_pending_function_output(
        self, config: DuplexSessionConfig, current: Mapping[str, object]
    ) -> dict[str, object]:
        del config
        return {**current, _RESULTS: []}

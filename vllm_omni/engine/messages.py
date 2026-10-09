# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Literal

import msgspec
from vllm.inputs import PromptType
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.inputs.data import OmniInteractionPrompt, OmniPromptType, OmniSamplingParams
from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.outputs import OmniRequestOutput


class EngineQueueMessage(msgspec.Struct, forbid_unknown_fields=True):
    pass


class StageSubmissionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["add_request", "streaming_update"]
    request_id: str
    prompt: EngineCoreRequest | PromptType
    original_prompt: EngineCoreRequest | PromptType
    output_prompt_text: object | None
    sampling_params_list: list[OmniSamplingParams]
    final_stage_id: int
    preprocess_ms: float
    request_timestamp: float
    enqueue_ts: float
    final_output_stage_ids: list[int] | None = None
    request_artifact_dirs: list[str] | None = None
    # Stage ID to provide the output of current request for use in external routing.
    # Only set when that stage has a next stage; otherwise it's a normal request.
    yield_stage_id: int | None = None


class AddCompanionRequestMessage(EngineQueueMessage, kw_only=True):
    type: Literal["add_companion_request"] = "add_companion_request"
    companion_id: str
    parent_id: str
    role: str
    prompt: EngineCoreRequest
    companion_prompt_text: object | None
    sampling_params_list: list[OmniSamplingParams]


class AbortRequestMessage(EngineQueueMessage, kw_only=True):
    type: Literal["abort"] = "abort"
    request_ids: list[str]
    # When set, Orchestrator emits AbortResultMessage on rpc_async_queue so
    # callers can await acknowledgment. Omit for fire-and-forget abort().
    rpc_id: str | None = None


class InteractionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["interaction"] = "interaction"
    request_id: str
    interaction: OmniInteractionPrompt


class CollectiveRPCRequestMessage(EngineQueueMessage, kw_only=True):
    type: Literal["collective_rpc"] = "collective_rpc"
    rpc_id: str
    method: str
    timeout: float | None = None
    args: tuple[object, ...]
    kwargs: dict[str, object]
    stage_ids: list[int] | None


class ShutdownRequestMessage(EngineQueueMessage, kw_only=True):
    type: Literal["shutdown"] = "shutdown"


class RegisterRemoteReplicaMessage(EngineQueueMessage, kw_only=True):
    type: Literal["register_remote_replica"] = "register_remote_replica"
    stage_id: int
    replica_id: int


class UnregisterRemoteReplicaMessage(EngineQueueMessage, kw_only=True):
    type: Literal["unregister_remote_replica"] = "unregister_remote_replica"
    stage_id: int
    input_addr: str


class ErrorMessage(EngineQueueMessage, kw_only=True):
    type: Literal["error"] = "error"
    error: str
    status_code: int | None = None
    error_type: str | None = None
    fatal: bool = False
    request_id: str | None = None
    stage_id: int | None = None
    event_id: str | None = None  # for interactions on diffusion generation requests


class OutputMessage(EngineQueueMessage, kw_only=True):
    type: Literal["output"] = "output"
    request_id: str
    stage_id: int
    replica_id: int | None = None
    engine_outputs: OmniRequestOutput
    metrics: StageRequestMetrics | None = None
    finished: bool
    stage_submit_ts: float | None = None


class NextStageInputMessage(EngineQueueMessage, kw_only=True):
    """Input for the next stage, to support external routing.

    NOTE: This encapsulates everything needed to submit to the receiver
    in the next request since in this case, each stage call is a separate
    request. This message should not be used in normal cases, since
    most of the state lives on objects held by the orchestrator and connectors.
    """

    type: Literal["next_stage_input"] = "next_stage_input"
    request_id: str
    source_stage_id: int
    receiver_stage_id: int
    requests: list[EngineCoreRequest] | list[OmniPromptType]
    submit_kwargs: dict[str, object] | None
    stage_output: OmniRequestOutput | None
    # For now, callers pass the full list of params on every call
    sampling_params_list: list[OmniSamplingParams]
    final_stage_id: int
    final_output_stage_ids: list[int]


class AbortResultMessage(EngineQueueMessage, kw_only=True):
    type: Literal["abort_result"] = "abort_result"
    rpc_id: str
    success: bool
    error: str | None = None
    # Final-stage AR abort outputs (partial tokens) for frontend delivery.
    # Empty for diffusion / requests with no output-processor state.
    abort_outputs: list[OutputMessage] | None = None

    @property
    def rpc_correlation_key(self) -> tuple[str, str]:
        return ("abort", self.rpc_id)


class StageMetricsMessage(EngineQueueMessage, kw_only=True):
    type: Literal["stage_metrics"] = "stage_metrics"
    request_id: str
    stage_id: int
    replica_id: int | None = None
    metrics: StageRequestMetrics
    stage_submit_ts: float | None = None


class CollectiveRPCResultMessage(EngineQueueMessage, kw_only=True):
    type: Literal["collective_rpc_result"] = "collective_rpc_result"
    rpc_id: str
    method: str
    stage_ids: list[int]
    results: list[object]

    @property
    def rpc_correlation_key(self) -> tuple[str, str]:
        return ("collective", self.rpc_id)

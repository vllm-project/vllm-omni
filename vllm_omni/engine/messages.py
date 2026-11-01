# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

import msgspec
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.stage_client import DiffusionSubmitKwargs
from vllm_omni.inputs.data import (
    OmniDiffusionSamplingParams,
    OmniInteractionPrompt,
    OmniPromptType,
    OmniSamplingParams,
)
from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.outputs import OmniRequestOutput


class EngineQueueMessage(msgspec.Struct, forbid_unknown_fields=True):
    pass


class OutputQueueWriter(Protocol):
    """Where the orchestrator thread puts messages for the serving frontend."""

    async def put(self, item: EngineQueueMessage) -> None: ...

    def put_nowait(self, item: EngineQueueMessage) -> None: ...


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
    entry_stage_id: int = 0
    lora_request: LoRARequest | None = None


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
    mm_cache_miss_hashes: list[str] | None = None
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
    submit_kwargs: DiffusionSubmitKwargs | None
    stage_output: OmniRequestOutput | None
    prompt: OmniPromptType
    # Currently, sampling params are set by the stage 0 call & carried to every later call
    sampling_params_list: list[OmniSamplingParams]

    def ensure_valid(self, stage_types: Sequence[str | None]) -> None:
        """Raise ValueError if this input can't be submitted to its receiver stage correctly.

        Fields come from the caller, so they may be any type.
        """
        self.check_stage_ids(len(stage_types))
        receiver_type = stage_types[self.receiver_stage_id]
        self._check_requests(receiver_type)
        self._check_sampling_params(stage_types)
        self._check_submit_kwargs(receiver_type)

    def check_stage_ids(self, num_stages: int) -> None:
        """Raise ValueError unless the source stage precedes a receiver stage in the pipeline."""
        source = self.source_stage_id
        receiver = self.receiver_stage_id
        if not (isinstance(source, int) and isinstance(receiver, int) and 0 <= source < receiver < num_stages):
            raise ValueError(
                f"stage_input has source stage {source!r} and receiver stage {receiver!r}, "
                f"which does not fit the {num_stages}-stage pipeline"
            )

    def _check_requests(self, receiver_type: str | None) -> None:
        """Raise ValueError if the requests are incompatible with the receiver type."""
        requests = self.requests
        if receiver_type == "diffusion":
            # StagePool.submit_initial rejects list prompts.
            requests_fit = isinstance(requests, list) and len(requests) == 1 and isinstance(requests[0], (dict, str))
        else:
            requests_fit = (
                isinstance(requests, list)
                and len(requests) > 0
                and all(isinstance(request, OmniEngineCoreRequest) for request in requests)
            )
        if not requests_fit:
            raise ValueError(
                f"stage_input has requests that stage {self.receiver_stage_id} ({receiver_type}) can't take"
            )

    def _check_sampling_params(self, stage_types: Sequence[str | None]) -> None:
        """Raise ValueError if sampling params aren't well aligned with stages."""
        params_list = self.sampling_params_list
        if not (
            isinstance(params_list, list)
            and len(params_list) == len(stage_types)
            and all(self._params_fit_stage(params, stage_type) for params, stage_type in zip(params_list, stage_types))
        ):
            raise ValueError("stage_input has sampling params that don't match the pipeline's stages")

    def _check_submit_kwargs(self, receiver_type: str | None) -> None:
        """Raise ValueError if the submit kwargs are incompatible with this receiver's type."""
        submit_kwargs = self.submit_kwargs
        if receiver_type == "diffusion":
            submit_kwargs_fit = submit_kwargs is None or (
                isinstance(submit_kwargs, dict) and submit_kwargs.keys() <= DiffusionSubmitKwargs.__optional_keys__
            )
        else:
            # LLM stage clients take no submit kwargs.
            submit_kwargs_fit = submit_kwargs is None
        if not submit_kwargs_fit:
            raise ValueError(
                f"stage_input has submit kwargs {submit_kwargs!r} that stage {self.receiver_stage_id} "
                f"({receiver_type}) can't take"
            )

    @staticmethod
    def _params_fit_stage(params: OmniSamplingParams, stage_type: str | None) -> bool:
        """Return whether a stage of this type can take these sampling params."""
        if stage_type == "diffusion":
            return isinstance(params, (OmniDiffusionSamplingParams, SamplingParams))
        return isinstance(params, SamplingParams)


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

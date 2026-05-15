from __future__ import annotations

from typing import Literal

import msgspec
from vllm.inputs import PromptType
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.inputs.data import OmniSamplingParams
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
    enqueue_ts: float


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


class ErrorMessage(EngineQueueMessage, kw_only=True):
    type: Literal["error"] = "error"
    error: str
    fatal: bool = False
    request_id: str | None = None
    stage_id: int | None = None


class OutputMessage(EngineQueueMessage, kw_only=True):
    type: Literal["output"] = "output"
    request_id: str
    stage_id: int
    engine_outputs: OmniRequestOutput
    metrics: StageRequestMetrics | None = None
    finished: bool
    stage_submit_ts: float | None = None


class StageMetricsMessage(EngineQueueMessage, kw_only=True):
    type: Literal["stage_metrics"] = "stage_metrics"
    request_id: str
    stage_id: int
    metrics: StageRequestMetrics
    stage_submit_ts: float | None = None


class CollectiveRPCResultMessage(EngineQueueMessage, kw_only=True):
    type: Literal["collective_rpc_result"] = "collective_rpc_result"
    rpc_id: str
    method: str
    stage_ids: list[int]
    results: list[object]

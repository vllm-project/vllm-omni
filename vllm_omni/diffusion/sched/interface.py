# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypedDict

from vllm_omni.diffusion.request import OmniDiffusionRequest


class DiffusionRequestStatus(enum.IntEnum):
    """Request status tracked by diffusion scheduler."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()

    # if any status is after or equal to FINISHED_COMPLETED, it is considered finished
    FINISHED_COMPLETED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_ERROR = enum.auto()

    @staticmethod
    def is_finished(status: DiffusionRequestStatus) -> bool:
        return status >= DiffusionRequestStatus.FINISHED_COMPLETED


@dataclass(frozen=True, eq=True)
class StepBatchSamplingParamsKey:
    """Denoise step level Batch-compatibility key derived from ``OmniDiffusionSamplingParams``.

    Only requests with the same key can be batched together.
    Fields not included here are treated as request-local and do not
    participate in the current homogeneous batching policy.
    """

    # Spatial / temporal shape.
    height: int | None = None
    width: int | None = None
    num_frames: int = 1
    resolution: int | str | None = None
    fps: int | None = None
    frame_rate: float | None = None
    boundary_ratio: float | None = None

    # CFG / guidance.
    do_classifier_free_guidance: bool = False
    guidance_scale: float = 0.0
    guidance_scale_provided: bool = False
    guidance_scale_2: float | None = None
    guidance_rescale: float = 0.0
    true_cfg_scale: float | None = None
    cfg_normalize: bool = False

    # Output count. Requests with different num_outputs_per_prompt produce
    # differently shaped outputs and cannot share a batch.
    num_outputs_per_prompt: int = 1

    # LoRA identity. Requests with different adapters or scales must run in
    # separate batches so the worker can activate exactly one adapter per step.
    lora_int_id: int | None = None
    lora_scale: float = 1.0


@dataclass(frozen=True, eq=True)
class RequestBatchSamplingParamsKey:
    """Request level Batch-compatibility key derived from ``OmniDiffusionSamplingParams``.

    Only request-batch-wide fields belong here. Request-local values such as
    seeds, generators, latent tensors, timesteps, and pipeline-specific
    ``extra_args`` are read per request from
    ``DiffusionRequestBatch.sampling_params_list``.
    """

    # Spatial / temporal shape.
    height: int | None = None
    width: int | None = None
    num_frames: int = 1
    resolution: int = 640
    fps: int | None = None
    frame_rate: float | None = None
    boundary_ratio: float | None = None

    # CFG / guidance.
    do_classifier_free_guidance: bool = False
    guidance_scale: float = 0.0
    guidance_scale_provided: bool = False
    guidance_scale_2: float | None = None
    guidance_rescale: float = 0.0
    true_cfg_scale: float | None = None
    cfg_normalize: bool = False
    strength: float | None = None

    # Scheduling / output shape.
    num_inference_steps: int | None = None
    sigmas: list[float] | None = None
    max_sequence_length: int | None = None
    num_outputs_per_prompt: int = 1
    eta: float = 0.0
    decode_timestep: float | list[float] | None = None
    decode_noise_scale: float | list[float] | None = None
    output_type: str | None = None

    # Model-specific batch defaults used by request-mode pipelines.
    layers: int = 4
    use_en_prompt: bool = False

    # LoRA identity.
    lora_int_id: int | None = None
    lora_scale: float = 1.0


@dataclass
class SchedulerRequestState:
    """Scheduler-owned state for one queued OmniDiffusionRequest."""

    request_id: str
    req: OmniDiffusionRequest
    sampling_params_key: StepBatchSamplingParamsKey | RequestBatchSamplingParamsKey | None = None
    status: DiffusionRequestStatus = DiffusionRequestStatus.WAITING
    error: str | None = None

    def is_finished(self) -> bool:
        return DiffusionRequestStatus.is_finished(self.status)


@dataclass
class NewRequestData:
    """Payload for a newly scheduled diffusion request.

    Carries the already-initialized request object so executors and workers do
    not re-run ``OmniDiffusionRequest.__post_init__`` and mutate sentinel-based
    fields like ``guidance_scale_provided``.
    """

    request_id: str
    req: OmniDiffusionRequest

    @classmethod
    def from_state(cls, state: SchedulerRequestState) -> NewRequestData:
        return cls(request_id=state.request_id, req=state.req)


@dataclass
class CachedRequestData:
    """Cached diffusion requests that only need their request ids resent."""

    request_ids: list[str]

    @classmethod
    def make_empty(cls) -> CachedRequestData:
        return cls(request_ids=[])


class KVPrefetchJob(TypedDict):
    """Descriptor for prefetching the next request's received KV cache."""

    request_id: str
    kv_sender_info: dict[str, Any]


@dataclass
class DiffusionSchedulerOutput:
    """Output of a single scheduling cycle."""

    # Stable scheduler-cycle diagnostics for engines, injected schedulers, and
    # instrumentation; they are intentionally retained even without a direct
    # production consumer for every field.
    step_id: int  # global step index
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    finished_req_ids: set[str]
    num_running_reqs: int
    num_waiting_reqs: int
    # next request to background-prefetch KV
    kv_prefetch_job: KVPrefetchJob | None = None

    @cached_property
    def scheduled_request_ids(self) -> list[str]:
        """
        All scheduled request ids in this cycle, including both new and cached ones.
        """
        return [
            *(req.request_id for req in self.scheduled_new_reqs),
            *self.scheduled_cached_reqs.request_ids,
        ]

    @property
    def num_scheduled_reqs(self) -> int:
        return len(self.scheduled_request_ids)

    @property
    def is_empty(self) -> bool:
        return self.num_scheduled_reqs == 0

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypedDict

from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.request import OmniDiffusionRequest

if TYPE_CHECKING:
    from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest


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


@dataclass(frozen=True)
class _AdmissionWaitDecision:
    """Internal scheduler policy for delaying the next admission wave."""

    should_wait: bool
    deadline: float | None = None
    stable_window_s: float = 0.0
    max_batch: int = 1


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

    # Request-scoped execution policy. A batch must use one quality mode
    # because model acceleration hooks are shared by the whole worker batch.
    quality: str | None = None

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

    # Request-scoped execution policy. Keep accelerated and reference-path
    # requests in separate worker batches. ``None`` delegates the default to
    # the model and must remain distinct from explicit ``lossless``.
    quality: str | None = None

    # LoRA identity.
    lora_int_id: int | None = None
    lora_scale: float = 1.0


@dataclass
class SchedulerRequestState:
    """Scheduler-owned state for one queued OmniDiffusionRequest."""

    request_id: str
    req: OmniDiffusionRequest
    sampling_params_key: StepBatchSamplingParamsKey | RequestBatchSamplingParamsKey | None = None
    diffusion_kv_requests: tuple[DiffusionKVRequest, ...] = ()
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
    diffusion_kv_metadata: DiffusionKVMetadata | None = None

    @classmethod
    def from_state(
        cls,
        state: SchedulerRequestState,
        *,
        diffusion_kv_metadata: DiffusionKVMetadata | None = None,
    ) -> NewRequestData:
        return cls(
            request_id=state.request_id,
            req=state.req,
            diffusion_kv_metadata=diffusion_kv_metadata,
        )


def validate_new_request_data_identity(new_req: NewRequestData) -> None:
    """Ensure an envelope and its forwarded request describe the same request."""
    forwarded_request_id = new_req.req.request_id
    if new_req.request_id != forwarded_request_id:
        raise ValueError(
            "Diffusion request identity mismatch: "
            f"envelope request_id={new_req.request_id!r}, "
            f"forwarded request_id={forwarded_request_id!r}"
        )

    metadata = new_req.diffusion_kv_metadata
    if metadata is not None and metadata.request_id != forwarded_request_id:
        raise ValueError(
            "Diffusion request identity mismatch: "
            f"metadata request_id={metadata.request_id!r}, "
            f"forwarded request_id={forwarded_request_id!r}"
        )


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

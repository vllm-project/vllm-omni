# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

if TYPE_CHECKING:
    from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest

DUMMY_DIFFUSION_REQUEST_ID = "dummy_req_id"
_DUMMY_DIFFUSION_REQUEST_ID_PREFIX = f"{DUMMY_DIFFUSION_REQUEST_ID}/"


def resolve_video_num_frames(
    num_frames: int | None,
    *,
    default_num_frames: int,
    is_dummy_run: bool,
) -> int:
    """Resolve the shared image-model frame sentinel for a video pipeline.

    ``OmniDiffusionSamplingParams`` defaults ``num_frames`` to one for image
    models, so an omitted video API field reaches model code as ``1``. Video
    pipelines with a different default must materialize it at their boundary.
    Startup profiling intentionally requests one frame, however, and must stay
    lightweight.
    """
    if num_frames is None or (num_frames == 1 and not is_dummy_run):
        return default_num_frames
    return num_frames


@dataclass
class OmniDiffusionRequest:
    """
    Input payload for a single diffusion request.

    This dataclass contains the prompt and sampling parameters for the diffusion pipeline
    execution. It also contains a request_id for other components to trace this request and its outputs.
    The runner wraps one or more requests into a DiffusionRequestBatch before pipeline execution.
    """

    # TODO(will): double check that args are separate from server_args
    # properly. Also maybe think about providing an abstraction for pipeline
    # specific arguments.
    # data_type: DataType

    prompt: OmniPromptType
    sampling_params: OmniDiffusionSamplingParams
    request_id: str
    kv_sender_info: dict[str, Any] | None = None
    # Optional opaque, model-owned input prepared before Scheduler admission.
    # Model code validates its concrete type when consuming it on the Worker.
    prepared_layout: Any | None = None
    # Scheduler-only native-compatible KV requests produced by model
    # preprocessing.
    # BaseScheduler takes ownership and clears this field before the request is
    # sent to a Worker.
    diffusion_kv_requests: tuple[DiffusionKVRequest, ...] | None = None
    # Optional model-specific condition structure used by request-batch admission.
    # This is populated by a pipeline preprocessor before the request reaches
    # the scheduler; ``None`` keeps the default behavior for other pipelines.
    batch_compatibility_key: tuple[Any, ...] | None = None
    # KV-recv wall-clock (ms), set by the runner's _prepare_request_for_forward
    # and carried to DiffusionOutput for the vllm_omni:diffusion_kv_load_s metric.
    kv_recv_ms: float = 0.0
    # Time spent waiting for initial admission by the diffusion scheduler.
    scheduler_queue_wait_ms: float | None = None

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("OmniDiffusionRequest.request_id must be a non-empty string.")

        # When neither a generator nor a seed is provided, assign a random seed
        # so that all ranks derive the same generator state.
        if self.sampling_params.generator is None and self.sampling_params.seed is None:
            self.sampling_params.seed = random.randint(0, 2**31 - 1)

        # Detect whether the caller explicitly provided guidance_scale before
        # resolving the omitted value.  ``0.0`` is API-valid and must not be
        # treated as omission because that would unexpectedly enable CFG in
        # pipelines with a model-specific default above zero.
        if self.sampling_params.guidance_scale is not None:
            self.sampling_params.guidance_scale_provided = True
        else:
            self.sampling_params.guidance_scale = 1.0

        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.sampling_params.guidance_scale > 1.0 and (
            not isinstance(self.prompt, str) and self.prompt.get("negative_prompt")
        ):
            self.sampling_params.do_classifier_free_guidance = True

        # Detect whether the caller explicitly provided guidance_scale_2
        # BEFORE auto-filling it. Pipelines with a model-specific second
        # guidance default (e.g. Boogu image_guidance_scale=1.0) must be able
        # to tell an explicit value apart from the guidance_scale fallback.
        if self.sampling_params.guidance_scale_2 is not None:
            self.sampling_params.guidance_scale_2_provided = True

        # Auto-fill guidance_scale_2 from the (now-resolved) guidance_scale
        # so downstream code always has a valid value.
        if self.sampling_params.guidance_scale_2 is None:
            self.sampling_params.guidance_scale_2 = self.sampling_params.guidance_scale

    def is_dummy_run(self) -> bool:
        return self.is_dummy_run_request_id(self.request_id)

    @classmethod
    def is_dummy_run_request_id(cls, request_id: str | None) -> bool:
        return request_id == DUMMY_DIFFUSION_REQUEST_ID or (
            isinstance(request_id, str) and request_id.startswith(_DUMMY_DIFFUSION_REQUEST_ID_PREFIX)
        )

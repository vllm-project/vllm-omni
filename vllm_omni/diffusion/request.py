# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from dataclasses import dataclass

from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

DUMMY_DIFFUSION_REQUEST_ID = "dummy_req_id"


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
    kv_sender_info: dict | None = None

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("OmniDiffusionRequest.request_id must be a non-empty string.")

        # When neither a generator nor a seed is provided, assign a random seed
        # so that all ranks derive the same generator state.
        if self.sampling_params.generator is None and self.sampling_params.seed is None:
            self.sampling_params.seed = random.randint(0, 2**31 - 1)

        # Detect whether user explicitly provided guidance_scale.
        # The sentinel default is 0.0 (false-like); any truthy value means
        # the caller set it intentionally.  We must resolve this BEFORE
        # auto-filling guidance_scale_2, otherwise the sentinel leaks into
        # guidance_scale_2.
        if self.sampling_params.guidance_scale:
            self.sampling_params.guidance_scale_provided = True
        else:
            self.sampling_params.guidance_scale = 1.0

        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.sampling_params.guidance_scale > 1.0 and (
            not isinstance(self.prompt, str) and self.prompt.get("negative_prompt")
        ):
            self.sampling_params.do_classifier_free_guidance = True

        # Auto-fill guidance_scale_2 from the (now-resolved) guidance_scale
        # so downstream code always has a valid value.
        if self.sampling_params.guidance_scale_2 is None:
            self.sampling_params.guidance_scale_2 = self.sampling_params.guidance_scale

    def is_dummy_run(self) -> bool:
        return self.is_dummy_run_request_id(self.request_id)

    @classmethod
    def is_dummy_run_request_id(cls, request_id: str | None) -> bool:
        return request_id == DUMMY_DIFFUSION_REQUEST_ID

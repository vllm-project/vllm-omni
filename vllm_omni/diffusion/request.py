# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


@dataclass
class OmniDiffusionRequest:
    """
    Complete state passed through the pipeline execution.

    This dataclass contains the prompts and sampling parameters for the diffusion pipeline
    execution. It also contains a request_id for other components to trace this request and its outputs.
    """

    # TODO(will): double check that args are separate from server_args
    # properly. Also maybe think about providing an abstraction for pipeline
    # specific arguments.
    # data_type: DataType

    prompts: list[OmniPromptType]  # Actually supporting str-based prompts
    sampling_params: OmniDiffusionSamplingParams

    request_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""
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
        if self.sampling_params.guidance_scale > 1.0 and any(
            (not isinstance(p, str) and p.get("negative_prompt")) for p in self.prompts
        ):
            self.sampling_params.do_classifier_free_guidance = True

        # Auto-fill guidance_scale_2 from the (now-resolved) guidance_scale
        # so downstream code always has a valid value.
        if self.sampling_params.guidance_scale_2 is None:
            self.sampling_params.guidance_scale_2 = self.sampling_params.guidance_scale

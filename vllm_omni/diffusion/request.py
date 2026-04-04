# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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
    request_id: str | None = None
    kv_sender_info: dict | None = None
    canonical_prompts: list[dict[str, Any]] = field(init=False, repr=False)

    @staticmethod
    def _canonicalize_prompt_item(prompt: OmniPromptType) -> dict[str, Any]:
        """Return a stable dict view for downstream diffusion pipelines.

        Diffusion serving paths may hand over prompt dicts where keys exist but
        values are ``None``. Normalize those cases once at request construction
        time so pipelines do not need to carry model-local online-mode hacks.
        """
        if isinstance(prompt, str):
            return {"prompt": prompt, "negative_prompt": None}

        if not isinstance(prompt, Mapping):
            raise TypeError(
                f"Diffusion prompts must be strings or mapping-like prompt objects, got {type(prompt).__name__}."
            )

        normalized = dict(prompt)
        prompt_text = normalized.get("prompt")
        if prompt_text is None:
            prompt_text = normalized.get("prompts")
        normalized["prompt"] = prompt_text or ""
        normalized["negative_prompt"] = normalized.get("negative_prompt")
        return normalized

    def get_prompt_texts(self) -> list[str]:
        """Return canonicalized prompt text for each request item."""
        return [prompt["prompt"] for prompt in self.canonical_prompts]

    def get_negative_prompt_texts(self) -> list[str] | None:
        """Return canonicalized negative prompts or ``None`` when absent."""
        if all(prompt.get("negative_prompt") is None for prompt in self.canonical_prompts):
            return None
        return [
            "" if prompt.get("negative_prompt") is None else prompt["negative_prompt"]
            for prompt in self.canonical_prompts
        ]

    def refresh_canonical_prompts(self) -> None:
        """Refresh canonical prompts after in-place prompt preprocessing."""
        self.canonical_prompts = [self._canonicalize_prompt_item(prompt) for prompt in self.prompts]

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""
        self.refresh_canonical_prompts()

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
        if self.sampling_params.guidance_scale > 1.0 and any(
            prompt.get("negative_prompt") for prompt in self.canonical_prompts
        ):
            self.sampling_params.do_classifier_free_guidance = True

        # Auto-fill guidance_scale_2 from the (now-resolved) guidance_scale
        # so downstream code always has a valid value.
        if self.sampling_params.guidance_scale_2 is None:
            self.sampling_params.guidance_scale_2 = self.sampling_params.guidance_scale

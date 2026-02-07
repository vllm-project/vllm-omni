# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


@dataclass
class OmniDiffusionExecutionState:
    """State for step-granularity resume."""

    step_index: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


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
    priority: float | None = None
    execution_state: OmniDiffusionExecutionState | None = None
    preempt_step_chunk_size: int = 1
    preempt_enabled: bool = False
    _request_key: str = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize dependent fields after dataclass initialization."""
        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.sampling_params.guidance_scale > 1.0 and any(
            (not isinstance(p, str) and p.get("negative_prompt")) for p in self.prompts
        ):
            self.sampling_params.do_classifier_free_guidance = True
        if self.sampling_params.guidance_scale_2 is None:
            self.sampling_params.guidance_scale_2 = self.sampling_params.guidance_scale

        # The dataclass default value is 0 (false-like), used to detect whether user explicitly provides this value
        # After this check is done, reset this value to old default 1
        if self.sampling_params.guidance_scale:
            self.sampling_params.guidance_scale_provided = True
        else:
            self.sampling_params.guidance_scale = 1.0

        self._request_key = self.request_ids[0] if self.request_ids else f"diff-{uuid.uuid4().hex[:16]}"

    @property
    def request_key(self) -> str:
        return self._request_key

    def get_or_assign_priority(self) -> float:
        """Assign a reproducible random priority when not provided."""
        if self.priority is not None:
            return float(self.priority)

        seed_hex = hashlib.sha256(self.request_key.encode("utf-8")).hexdigest()[:16]
        self.priority = random.Random(int(seed_hex, 16)).random()
        return float(self.priority)

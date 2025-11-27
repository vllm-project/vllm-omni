from dataclasses import dataclass
from typing import Optional

import torch
from vllm.outputs import RequestOutput
from vllm.v1.outputs import ModelRunnerOutput


class OmniModelRunnerOutput(ModelRunnerOutput):
    """Model runner output for omni models.

    Extends the base ModelRunnerOutput with support for multimodal outputs
    that may be produced by non-autoregressive stages.

    Attributes:
        multimodal_outputs: Optional dictionary mapping modality names to
            output tensors (e.g., {"image": tensor, "audio": tensor})
    """

    multimodal_outputs: Optional[dict[str, torch.Tensor]] = None


@dataclass
class OmniRequestOutput(RequestOutput):
    """Request output for omni pipeline stages.

    Wraps a standard RequestOutput with stage-specific metadata,
    indicating which stage produced the output and what type of
    final output it represents.

    Attributes:
        stage_id: Identifier of the stage that produced this output
        final_output_type: Type of final output (e.g., "text", "image",
            "audio", "latents")
        request_output: The underlying RequestOutput from the stage
    """

    stage_id: int
    final_output_type: str
    request_output: RequestOutput

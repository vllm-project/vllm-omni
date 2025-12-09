from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image
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

    multimodal_outputs: dict[str, torch.Tensor] | None = None


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


@dataclass
class OmniDiffusionRequestOutput:
    """Request output for diffusion model inference.

    Wraps diffusion model outputs with request metadata for tracking
    and processing in the API server.

    Attributes:
        request_id: Unique identifier for this request
        images: List of generated PIL images
        latents: Optional tensor of latent representations
        prompt: The prompt used for generation
        finished: Whether generation is complete
        metrics: Optional dictionary of generation metrics
            (e.g., inference time, steps completed)
    """

    request_id: str
    images: list[Image.Image] = field(default_factory=list)
    latents: Optional[torch.Tensor] = None
    prompt: Optional[str] = None
    finished: bool = True
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def num_images(self) -> int:
        """Return the number of generated images."""
        return len(self.images)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "num_images": self.num_images,
            "prompt": self.prompt,
            "finished": self.finished,
            "metrics": self.metrics,
        }

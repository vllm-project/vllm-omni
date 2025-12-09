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
class OmniRequestOutput:
    """Unified request output for both pipeline stages and diffusion models.

    This class handles outputs from:
    1. Multi-stage LLM pipelines (with stage_id, final_output_type, request_output)
    2. Diffusion models (with images, prompt, metrics)

    Attributes:
        request_id: Unique identifier for this request
        finished: Whether generation is complete

        # Pipeline stage fields
        stage_id: Identifier of the stage that produced this output (pipeline mode)
        final_output_type: Type of output ("text", "image", "audio", "latents")
        request_output: The underlying RequestOutput from the stage (pipeline mode)

        # Diffusion model fields
        images: List of generated PIL images (diffusion mode)
        prompt: The prompt used for generation (diffusion mode)
        latents: Optional tensor of latent representations (diffusion mode)
        metrics: Optional dictionary of generation metrics
    """

    request_id: str = ""
    finished: bool = True

    # Pipeline stage fields
    stage_id: Optional[int] = None
    final_output_type: str = "text"
    request_output: Optional[RequestOutput] = None

    # Diffusion model fields
    images: list[Image.Image] = field(default_factory=list)
    prompt: Optional[str] = None
    latents: Optional[torch.Tensor] = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pipeline(
        cls,
        stage_id: int,
        final_output_type: str,
        request_output: RequestOutput,
    ) -> "OmniRequestOutput":
        """Create output from pipeline stage.

        Args:
            stage_id: Stage identifier
            final_output_type: Type of output
            request_output: The stage's output

        Returns:
            OmniRequestOutput configured for pipeline mode
        """
        return cls(
            request_id=getattr(request_output, "request_id", ""),
            stage_id=stage_id,
            final_output_type=final_output_type,
            request_output=request_output,
            finished=True,
        )

    @classmethod
    def from_diffusion(
        cls,
        request_id: str,
        images: list[Image.Image],
        prompt: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> "OmniRequestOutput":
        """Create output from diffusion model.

        Args:
            request_id: Request identifier
            images: Generated images
            prompt: The prompt used
            metrics: Generation metrics
            latents: Optional latent tensors

        Returns:
            OmniRequestOutput configured for diffusion mode
        """
        return cls(
            request_id=request_id,
            final_output_type="image",
            images=images,
            prompt=prompt,
            latents=latents,
            metrics=metrics or {},
            finished=True,
        )

    @property
    def num_images(self) -> int:
        """Return the number of generated images."""
        return len(self.images)

    @property
    def is_diffusion_output(self) -> bool:
        """Check if this is a diffusion model output."""
        return len(self.images) > 0 or self.final_output_type == "image"

    @property
    def is_pipeline_output(self) -> bool:
        """Check if this is a pipeline stage output."""
        return self.stage_id is not None and self.request_output is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "request_id": self.request_id,
            "finished": self.finished,
            "final_output_type": self.final_output_type,
        }

        if self.is_diffusion_output:
            result.update(
                {
                    "num_images": self.num_images,
                    "prompt": self.prompt,
                    "metrics": self.metrics,
                }
            )

        if self.is_pipeline_output:
            result.update(
                {
                    "stage_id": self.stage_id,
                }
            )

        return result

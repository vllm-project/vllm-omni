from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from vllm.outputs import RequestOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_omni.inputs.data import OmniPromptType


class OmniModelRunnerOutput(ModelRunnerOutput):
    """Model runner output for omni models.

    Extends the base ModelRunnerOutput with support for multimodal outputs
    that may be produced by non-autoregressive stages.

    Attributes:
        multimodal_outputs: Optional dictionary mapping modality names to
            output tensors (e.g., {"image": tensor, "audio": tensor})
    """

    multimodal_outputs: dict[str, torch.Tensor] | None = None
    # IDs of requests whose KV cache has been extracted from GPU/NPU to CPU.
    # The Scheduler can safely free the block tables for these requests.
    kv_extracted_req_ids: list[str] | None = None


@dataclass
class OmniRequestOutput:
    """Unified request output for both pipeline stages and diffusion models.

    This class handles outputs from:
    1. Multi-stage LLM pipelines (with stage_id, final_output_type, request_output)
    2. Diffusion models (with images, prompt, metrics)

    Attributes:
        request_id: Unique identifier for this request
        finished: Whether generation is complete
        stage_id: Identifier of the stage that produced this output (pipeline mode)
        final_output_type: Type of output ("text", "image", "audio", "latents")
        request_output: The underlying RequestOutput from the stage (pipeline mode)
        images: List of generated PIL images (diffusion mode)
        prompt: The prompt used for generation (diffusion mode)
        latents: Optional tensor of latent representations (diffusion mode)
        metrics: Optional dictionary of generation metrics
    """

    request_id: str = ""
    finished: bool = True

    # Pipeline stage fields
    stage_id: int | None = None
    final_output_type: str = "text"
    request_output: RequestOutput | None = None

    # Diffusion model fields
    images: list[Image.Image] = field(default_factory=list)
    prompt: OmniPromptType | None = None
    latents: torch.Tensor | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    _multimodal_output: dict[str, Any] = field(default_factory=dict)

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
        prompt: OmniPromptType | None = None,
        metrics: dict[str, Any] | None = None,
        latents: torch.Tensor | None = None,
        multimodal_output: dict[str, Any] | None = None,
        final_output_type: str = "image",
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
            final_output_type=final_output_type,
            images=images,
            prompt=prompt,
            latents=latents,
            metrics=metrics or {},
            _multimodal_output=multimodal_output or {},
            finished=True,
        )

    @property
    def multimodal_output(self) -> dict[str, Any]:
        """Return multimodal output from the underlying request output or local field.

        For pipeline outputs, this checks completion outputs first, then request_output.
        For diffusion outputs, this returns the local _multimodal_output field.
        """
        if self.request_output is not None:
            # Check completion outputs first (where multimodal_output is attached).
            outputs = getattr(self.request_output, "outputs", None)
            if isinstance(outputs, list) and outputs:
                for output in outputs:
                    mm = getattr(output, "multimodal_output", None)
                    if mm:
                        return mm
            return getattr(self.request_output, "multimodal_output", {})
        return self._multimodal_output

    @property
    def num_images(self) -> int:
        """Return the number of generated images."""
        return len(self.images)

    # Pass-through properties keep vLLM serving codepaths compatible with
    # OmniRequestOutput for pipeline outputs (Issue #345).
    @property
    def prompt_token_ids(self) -> list[int] | None:
        """Return prompt token IDs from the underlying request output.

        This property is required for compatibility with vLLM's streaming
        chat completion generator which checks res.prompt_token_ids.
        """
        if self.request_output is not None:
            return getattr(self.request_output, "prompt_token_ids", None)
        return None

    @property
    def outputs(self) -> list[Any]:
        """Return outputs from the underlying request output.

        This property is required for compatibility with vLLM's streaming
        and non-streaming chat completion generators.
        """
        if self.request_output is not None:
            return getattr(self.request_output, "outputs", [])
        return []

    @property
    def encoder_prompt_token_ids(self) -> list[int] | None:
        """Return encoder prompt token IDs from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "encoder_prompt_token_ids", None)
        return None

    @property
    def prompt_logprobs(self) -> Any:
        """Return prompt logprobs from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "prompt_logprobs", None)
        return None

    @property
    def num_cached_tokens(self) -> int | None:
        """Return number of cached tokens from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "num_cached_tokens", None)
        return None

    @property
    def kv_transfer_params(self) -> Any:
        """Return KV transfer params from the underlying request output."""
        if self.request_output is not None:
            return getattr(self.request_output, "kv_transfer_params", None)
        return None

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

    def __repr__(self) -> str:
        """Custom repr to properly show image count instead of image objects."""
        # For images, show count instead of full list
        images_repr = f"[{len(self.images)} PIL Images]" if self.images else "[]"

        # Build repr string
        parts = [
            f"request_id={self.request_id!r}",
            f"finished={self.finished}",
            f"stage_id={self.stage_id}",
            f"final_output_type={self.final_output_type!r}",
            f"request_output={self.request_output}",
            f"images={images_repr}",
            f"prompt={self.prompt!r}",
            f"latents={self.latents}",
            f"metrics={self.metrics}",
            f"multimodal_output={self._multimodal_output}",
        ]

        return f"OmniRequestOutput({', '.join(parts)})"

"""
Diffusers-backed executor for DiT stages.

Provides a minimal wrapper around HuggingFace diffusers pipelines to support
text-to-image generation as the DiT stage backend.
"""

from __future__ import annotations

from typing import Any, Optional

from vllm_omni.worker.gpu_diffusion_worker import (
    DiffusionGPUWorker,
    DiffusionRunnerOutput,
)


class DiffusersPipelineExecutor:
    """Thin executor that delegates to a diffusion worker."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        pipeline_name: Optional[str] = None,
    ) -> None:
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "diffusers and torch are required for DiT diffusers backend."
            ) from e

        self.worker = DiffusionGPUWorker(
            model_path,
            pipeline_name=pipeline_name,
            device=device,
            dtype=dtype,
        )

    def generate(
        self,
        prompt: str,
        *,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        image: Optional[Any] = None,
    ) -> DiffusionRunnerOutput:
        return self.worker.generate(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            image=image,
        )

"""Worker abstraction for executing diffusion model runners."""

from __future__ import annotations

from typing import Any, Optional

from vllm.v1.worker.worker_base import WorkerBase

from .gpu_diffusion_model_runner import DiffusionModelRunner, DiffusionRunnerOutput


class DiffusionGPUWorker(WorkerBase):
    """Minimal worker wrapping DiffusionModelRunner.

    The worker interface mirrors the shape expected by vLLM executors: it owns
    a model runner instance and exposes a `generate` method that can be called
    by an executor implementation.
    """

    def __init__(
        self,
        model_path: str,
        *,
        pipeline_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        self.model_runner = DiffusionModelRunner(
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
        return self.model_runner.generate(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            image=image,
        )

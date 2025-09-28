"""Model runner that wraps a diffusers text-to-image pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class DiffusionRunnerOutput:
    """Structured output returned by the diffusion runner."""

    prompt: str
    images: list
    request_id: str = "diffusion_request"
    finished: bool = True
    output_type: str = "image"


class DiffusionModelRunner:
    """Lightweight runner that loads and executes diffusers pipelines."""

    def __init__(
        self,
        model_path: str,
        *,
        pipeline_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.pipeline_name = pipeline_name or "auto"
        self.device, self.torch_dtype = self._resolve_device_and_dtype(device, dtype)
        self._pipeline = self._load_pipeline()

    def _resolve_device_and_dtype(
        self,
        device: Optional[str],
        dtype: Optional[str],
    ):
        import torch

        resolved_device = device
        if resolved_device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            elif torch.cuda.is_available():
                resolved_device = "cuda"
            else:
                resolved_device = "cpu"

        if dtype is not None:
            resolved_dtype = getattr(torch, dtype, None)
        else:
            if resolved_device in {"cuda", "mps"}:
                resolved_dtype = torch.float16
            else:
                resolved_dtype = torch.float32

        return resolved_device, resolved_dtype

    def _load_pipeline(self):
        import diffusers

        if self.pipeline_name != "auto":
            PipelineCls = getattr(diffusers, self.pipeline_name)
        else:
            PipelineCls = diffusers.AutoPipelineForText2Image

        pipeline = PipelineCls.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
        )

        try:
            pipeline = pipeline.to(self.device)
        except Exception:
            pipeline = pipeline.to("cpu")

        self._apply_optimisations(pipeline)
        return pipeline

    @staticmethod
    def _apply_optimisations(pipeline) -> None:
        # Enable attention slicing / VAE tiling where available to reduce memory.
        try:
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
        except Exception:
            pass
        try:
            vae = getattr(pipeline, "vae", None)
            if vae and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
        except Exception:
            pass

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
        import torch

        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
            except Exception:
                # Some backends (e.g., CPU without full support) may not accept the device.
                generator = torch.Generator().manual_seed(int(seed))

        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            height=int(height),
            width=int(width),
            generator=generator,
            image=image,
        )

        return DiffusionRunnerOutput(
            prompt=prompt,
            images=getattr(output, "images", []) or [],
        )

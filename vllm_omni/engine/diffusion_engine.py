"""Diffusion engine adapters for diffusers-backed DiT stages."""

from __future__ import annotations

from typing import Any, Optional

from ..config import DiTConfig
from ..worker.gpu_diffusion_worker import (
    DiffusionGPUWorker,
    DiffusionRunnerOutput,
)


class DiffusionEngine:
    """Base diffusion engine storing shared DiT configuration."""

    def __init__(
        self,
        dit_config: DiTConfig,
        model_path: Optional[str] = None,
        log_stats: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        self.dit_config = dit_config

        model_cfg = getattr(dit_config, "model_config", None)
        config_model_path = None
        if model_cfg is not None:
            if isinstance(model_cfg, dict):
                config_model_path = (
                    model_cfg.get("model")
                    or model_cfg.get("model_path")
                )
            else:
                config_model_path = (
                    getattr(model_cfg, "model", None)
                    or getattr(model_cfg, "model_path", None)
                )

        self.model_path = model_path or config_model_path
        self.model_config = dit_config.model_config
        self.cache_config = dit_config.dit_cache_config

        self.height = dit_config.height
        self.width = dit_config.width
        self.num_inference_steps = dit_config.num_inference_steps
        self.guidance_scale = dit_config.guidance_scale

        self.log_stats = log_stats
        self.multiprocess_mode = multiprocess_mode

    def generate(self, *args, **kwargs):  # pragma: no cover - placeholder
        raise NotImplementedError("This method should be implemented in subclasses.")


class DiffusersPipelineEngine(DiffusionEngine):
    """Adapter that invokes a diffusers pipeline via DiffusionGPUWorker."""

    def __init__(
        self,
        dit_config: DiTConfig,
        model_path: Optional[str] = None,
        log_stats: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        super().__init__(dit_config, model_path, log_stats, multiprocess_mode)

        resolved_model_path = self.model_path
        if resolved_model_path is None:
            raise ValueError("Model path must be provided for DiffusersPipelineEngine.")

        device_cfg = getattr(dit_config, "device_config", None)
        model_cfg = getattr(dit_config, "model_config", None)

        device = None
        dtype = None

        if device_cfg:
            if isinstance(device_cfg, dict):
                device = device_cfg.get("device")
                dtype = device_cfg.get("dtype")
            else:
                device = getattr(device_cfg, "device", None)
                dtype = getattr(device_cfg, "dtype", None)

        if dtype is None and model_cfg:
            if isinstance(model_cfg, dict):
                dtype = model_cfg.get("dtype")
            else:
                dtype = getattr(model_cfg, "dtype", None)

        self.worker = DiffusionGPUWorker(
            resolved_model_path,
            pipeline_name=dit_config.diffusers_pipeline,
            device=device,
            dtype=dtype,
        )

    def generate(
        self,
        prompt: str,
        *,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        image: Optional[Any] = None,
    ) -> DiffusionRunnerOutput:
        height = int(height) if height is not None else self.height
        width = int(width) if width is not None else self.width
        num_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else self.num_inference_steps
        )
        guidance = (
            float(guidance_scale)
            if guidance_scale is not None
            else self.guidance_scale
        )

        return self.worker.generate(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt,
            seed=seed,
            image=image,
        )

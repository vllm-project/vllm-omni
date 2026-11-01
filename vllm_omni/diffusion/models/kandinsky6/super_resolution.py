# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional HD/Full-HD refinement after Kandinsky6 video/audio generation."""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform

DEFAULT_SR_MODEL = "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"
TARGET_RESOLUTIONS = {"hd": (720, 1280), "fhd": (1080, 1920), "fullhd": (1080, 1920)}


def validate_super_resolution_execution(config: Any) -> None:
    if config is None:
        return
    parallel = config.parallel_config
    if parallel.world_size != 1 or parallel.use_hsdp or config.enable_layerwise_offload:
        raise ValueError("Kandinsky6 HD/FHD refinement requires one GPU and supports component CPU offload")


@dataclass(frozen=True)
class SuperResolutionSettings:
    height: int
    width: int
    model: str
    steps: int | None
    min_overlap: float
    tiles_batch_size: int


def resolve_super_resolution(sampling: Any) -> SuperResolutionSettings | None:
    extra = sampling.extra_args or {}
    target = extra.get("target_resolution")
    height, width = sampling.height, sampling.width
    if target is not None:
        if target not in TARGET_RESOLUTIONS:
            raise ValueError("target_resolution must be hd, fhd or fullhd")
        target_height, target_width = TARGET_RESOLUTIONS[target]
        if height not in (None, target_height) or width not in (None, target_width):
            raise ValueError("Explicit dimensions conflict with target_resolution")
        height, width = target_height, target_width
    if (height, width) not in TARGET_RESOLUTIONS.values():
        if any(key.startswith("sr_") for key in extra):
            raise ValueError("SR options require target_resolution or explicit HD/FHD dimensions")
        return None
    if sampling.output_type == "latent":
        raise ValueError("HD/FHD refinement requires decoded video output, not output_type=latent")
    steps = extra.get("sr_num_inference_steps")
    if steps is not None and (not isinstance(steps, int) or steps < 1):
        raise ValueError("sr_num_inference_steps must be a positive integer")
    overlap = float(extra.get("sr_min_overlap", 0.2))
    tile_batch = extra.get("sr_tiles_batch_size", 1)
    if not 0 <= overlap < 1:
        raise ValueError("sr_min_overlap must be in [0,1)")
    if not isinstance(tile_batch, int) or tile_batch < 1:
        raise ValueError("sr_tiles_batch_size must be a positive integer")
    model = extra.get("sr_model", DEFAULT_SR_MODEL)
    if not isinstance(model, str) or not model:
        raise ValueError("sr_model must be a Hugging Face checkpoint ID or a local directory")
    return SuperResolutionSettings(height, width, model, steps, overlap, tile_batch)


@contextmanager
def stage_generation_modules_on_cpu(modules: list[nn.Module]):
    """Release base-model GPU weights and restore their previous residency."""
    snapshots = []
    for module in modules:
        devices = {name: value.device for name, value in module.named_parameters()}
        buffers = {name: value.device for name, value in module.named_buffers()}
        snapshots.append((module, devices, buffers))
    try:
        for module, _, _ in snapshots:
            module.to("cpu")
        current_omni_platform.empty_cache()
        yield
    finally:
        for module, devices, buffers in snapshots:
            for name, parameter in module.named_parameters():
                parameter.data = parameter.data.to(devices[name])
            for name, buffer in module.named_buffers():
                buffer.data = buffer.data.to(buffers[name])


class Kandinsky6SuperResolution:
    """Lazy SR stage; base generation and refinement occupy the GPU in turn."""

    def __init__(self, pipeline: nn.Module):
        self._pipeline = weakref.ref(pipeline)
        self._refiner = None
        self._model = None

    def _load_refiner(self, model: str):
        from vllm.config import LoadConfig

        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
        from vllm_omni.diffusion.models.kandinsky6_sr import Kandinsky6SRRefiner

        if self._model != model:
            self._refiner = None
            self._model = None
            config = OmniDiffusionConfig(model=model, dtype=torch.bfloat16, enable_cpu_offload=True)
            refiner = Kandinsky6SRRefiner(od_config=config)
            DiffusersPipelineLoader(LoadConfig(), config).load_weights(refiner)
            self._refiner = refiner.eval()
            self._model = model
        return self._refiner

    def refine(self, frames: np.ndarray, sampling: Any, settings: SuperResolutionSettings) -> np.ndarray:
        pipeline = self._pipeline()
        if pipeline is None:
            raise RuntimeError("Kandinsky6 generation pipeline has been released")
        validate_super_resolution_execution(pipeline.od_config)
        pipeline.transformer.clear_text_proj_cache()
        modules = [
            module
            for name in ("transformer", "vae", "audio_vae", "text_encoder", "text_encoder_2")
            if isinstance((module := getattr(pipeline, name, None)), nn.Module)
        ]
        with stage_generation_modules_on_cpu(modules):
            refiner = self._load_refiner(settings.model)
            try:
                refiner.to(pipeline.device)
                return refiner.refine(
                    frames,
                    height=settings.height,
                    width=settings.width,
                    seed=sampling.seed if sampling.seed is not None else 42,
                    num_inference_steps=settings.steps,
                    min_overlap=settings.min_overlap,
                    tiles_batch_size=settings.tiles_batch_size,
                )
            finally:
                refiner.to("cpu")
                current_omni_platform.empty_cache()

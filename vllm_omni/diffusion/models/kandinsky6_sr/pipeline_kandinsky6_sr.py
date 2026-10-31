# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Internal tiled SR model for Kandinsky 6 text/image-to-video generation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.modeling_utils import no_init_weights
from PIL import Image
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.offloader.config import offload_enabled
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

from .autoencoder_kandinsky6_sr import Kandinsky6SRVAE
from .config import OutputPlan, default_steps, output_plan, transformer_config, upscaler_config, vae_config
from .latent_upscaler_kandinsky6_sr import Kandinsky6SRLatentUpscalerBank
from .scheduling_piflow import PiflowScheduler
from .sr_inference import Kandinsky6SRInference
from .transformer_kandinsky6_sr import Kandinsky6SRTransformer3DModel


def normalize_frames(video: Any) -> torch.Tensor:
    if isinstance(video, (list, tuple)) and video and isinstance(video[0], Image.Image):
        video = np.stack([np.asarray(frame.convert("RGB")) for frame in video])
    frames = torch.as_tensor(video).detach().cpu()
    if frames.ndim == 5:
        if frames.shape[0] != 1:
            raise ValueError("Kandinsky VSR accepts one input video per request")
        frames = frames[0]
        if frames.shape[-1] != 3 and frames.shape[0] == 3:
            frames = frames.permute(1, 2, 3, 0)
    if frames.ndim == 4 and frames.shape[-1] != 3 and frames.shape[1] == 3:
        frames = frames.permute(0, 2, 3, 1)
    if frames.ndim != 4 or frames.shape[-1] != 3 or min(frames.shape) < 1:
        raise ValueError("Kandinsky VSR requires nonempty RGB frames [T,H,W,3] or [T,3,H,W]")
    if frames.dtype != torch.uint8:
        if not frames.is_floating_point() or not torch.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError("Floating-point video frames must be finite and in [0,1]")
        frames = (frames.float() * 255).round().to(torch.uint8)
    return frames.contiguous()


def prepare_pixels(frames: torch.Tensor, plan: OutputPlan, device: torch.device) -> torch.Tensor:
    # Match the delivery aspect ratio with a center crop before resampling.
    source_height, source_width = frames.shape[1:3]
    if source_width / source_height > plan.width / plan.height:
        crop_width = max(1, round(source_height * plan.width / plan.height))
        left = (source_width - crop_width) // 2
        frames = frames[:, :, left : left + crop_width]
    else:
        crop_height = max(1, round(source_width * plan.height / plan.width))
        top = (source_height - crop_height) // 2
        frames = frames[:, top : top + crop_height]
    chunks = []
    for start in range(0, len(frames), 8):
        pixels = frames[start : start + 8].to(device).permute(0, 3, 1, 2).float() / 255
        pixels = F.interpolate(
            pixels, size=(plan.content_height, plan.content_width), mode="bilinear", align_corners=False, antialias=True
        )
        chunks.append(pixels)
    pixels = torch.cat(chunks).permute(1, 0, 2, 3).unsqueeze(0)
    return F.pad(
        pixels,
        (
            0,
            plan.prepared_width - plan.content_width,
            0,
            plan.prepared_height - plan.content_height,
            0,
            (1 - len(frames)) % 4,
        ),
        mode="replicate",
    )


def _cast_component(module: nn.Module, dtype: torch.dtype) -> None:
    module.to(dtype=dtype)
    keep = getattr(module, "_keep_in_fp32_modules", ()) or ()
    for name, parameter in module.named_parameters():
        if any(pattern in name for pattern in keep):
            parameter.data = parameter.data.float()
    if hasattr(module, "visual_rope_embeddings"):
        module.visual_rope_embeddings.float()


class Kandinsky6SRRefiner(nn.Module):
    """Internal refinement stage for Kandinsky6 text/image-to-video generation."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _encoder_modules: ClassVar[list[str]] = ["latent_upscaler"]
    _resident_modules: ClassVar[list[str]] = []

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.od_config, self.device = od_config, get_local_device()
        parallel = od_config.parallel_config
        for name in (
            "tensor_parallel_size",
            "cfg_parallel_size",
            "ulysses_degree",
            "ring_degree",
            "pipeline_parallel_size",
            "vae_patch_parallel_size",
        ):
            if getattr(parallel, name) != 1:
                raise ValueError("Kandinsky VSR currently runs on one GPU; all parallel degrees must be 1")
        if parallel.use_hsdp or parallel.enable_expert_parallel or od_config.enable_layerwise_offload:
            raise ValueError("Kandinsky VSR supports component CPU offload, not HSDP, expert or layerwise offload")
        if od_config.quantization_config is not None or od_config.lora_path is not None:
            raise ValueError("Kandinsky VSR does not support quantization or LoRA adapters")
        root = Path(od_config.model)
        if not root.is_dir():
            root = Path(download_weights_from_hf_specific(od_config.model, None, ["*"], revision=od_config.revision))

        def config(component: str, filename: str = "config.json") -> dict[str, Any]:
            return json.loads((root / component / filename).read_text())

        dit_config = config("transformer")
        scheduler_config = config("scheduler", "scheduler_config.json")
        self.default_num_inference_steps = default_steps(scheduler_config)
        self.lq_noise_scale = float(dit_config.get("sr_params", {}).get("lq_noise_scale", 0.7))
        upscaler_keys = []
        for weights_file in (root / "latent_upscaler").glob("*.safetensors"):
            with safe_open(weights_file, framework="pt", device="cpu") as weights:
                upscaler_keys.extend(weights.keys())
        load_device = torch.device("cpu") if offload_enabled(od_config) else self.device
        with torch.device(load_device), no_init_weights():
            self.transformer = Kandinsky6SRTransformer3DModel(**transformer_config(dit_config))
            self.vae = Kandinsky6SRVAE(**vae_config(config("vae")))
            self.latent_upscaler = Kandinsky6SRLatentUpscalerBank(
                **upscaler_config(config("latent_upscaler"), upscaler_keys)
            )
        for component in (self.transformer, self.vae, self.latent_upscaler):
            _cast_component(component, od_config.dtype)
        scheduler_type = (
            PiflowScheduler if scheduler_config["_class_name"] == "PiflowScheduler" else FlowMatchEulerDiscreteScheduler
        )
        scheduler = scheduler_type.from_config(scheduler_config)
        self._inference = Kandinsky6SRInference(self.transformer, self.vae, scheduler, self.latent_upscaler)
        self._inference.omni_device = self.device
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(root),
                subfolder=name,
                revision=None,
                prefix=name + ".",
                fall_back_to_pt=False,
            )
            for name in ("transformer", "vae", "latent_upscaler")
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.models.utils import AutoWeightsLoader

        loaded = AutoWeightsLoader(self).load_weights(weights)
        missing = self.state_dict().keys() - loaded
        if missing:
            raise ValueError(f"Missing Kandinsky VSR weights: {sorted(missing)}")
        self.transformer.set_attention_backend("flex")
        self.transformer.compile_repeated_blocks(fullgraph=True, options={"max_autotune": True})
        return loaded

    @torch.inference_mode()
    def refine(
        self,
        frames: np.ndarray | torch.Tensor,
        *,
        height: int,
        width: int,
        seed: int,
        num_inference_steps: int | None = None,
        min_overlap: float = 0.2,
        tiles_batch_size: int = 1,
    ) -> np.ndarray:
        frames = normalize_frames(frames)
        plan = output_plan(frames.shape[1], frames.shape[2], height, width, {})
        pixels = prepare_pixels(frames, plan, self.device)
        steps = self.default_num_inference_steps if num_inference_steps is None else num_inference_steps
        result = self._inference(
            video=pixels.permute(0, 2, 1, 3, 4),
            resolution_scale=plan.tiling_scale,
            num_inference_steps=steps,
            lq_noise_scale=self.lq_noise_scale,
            min_overlap=min_overlap,
            tiles_batch_size=tiles_batch_size,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            output_type="np",
        ).frames
        video = result[:, : len(frames), : plan.height, : plan.width]
        return (np.clip(video, 0, 1) * 255).round().astype(np.uint8)

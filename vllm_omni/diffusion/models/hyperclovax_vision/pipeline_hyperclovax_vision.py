# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from NAVER Cloud Corp. vision-decoder-api

"""
HyperCLOVAX Vision Pipeline for vLLM-Omni.

This pipeline converts vision tokens to images using a VisionTransformer
diffusion model. It supports:
- Vision token embedding
- Flow matching diffusion
- Autoguidance (optional transformer2)
- xDiT USP sequence parallelism
"""

import json
import logging
import os
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from einops import rearrange, repeat
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .hyperclovax_vision_transformer import HyperCLOVAXVisionTransformer2DModel
from .vision_token_embedder import VisionTokenEmbedder

logger = logging.getLogger(__name__)


def get_hyperclovax_vision_post_process_func(od_config: OmniDiffusionConfig):
    """
    Get post-processing function for HyperCLOVAX Vision pipeline.

    Returns a function that converts model output tensors to PIL images.
    """
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        from vllm_omni.model_executor.model_loader.weight_utils import (
            download_weights_from_hf_specific,
        )

        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    # VaeImageProcessor expects the spatial downsampling ratio (e.g. 8),
    # not the latent scaling_factor (e.g. 0.13025). Read from vae config if available.
    vae_spatial_factor = 8  # default for AutoencoderKL (2^3 downsampling)
    vae_config_path = os.path.join(model_path, "vae/config.json")
    if os.path.exists(vae_config_path):
        with open(vae_config_path) as f:
            config = json.load(f)
            # block_out_channels determines spatial downsampling: 2^(len-1)
            block_out_channels = config.get("block_out_channels")
            if block_out_channels:
                vae_spatial_factor = 2 ** (len(block_out_channels) - 1)

    image_processor = VaeImageProcessor(vae_scale_factor=vae_spatial_factor)

    def post_process_func(images: torch.Tensor):
        """Convert tensor images to PIL images."""
        return image_processor.postprocess(images)

    return post_process_func


class HyperCLOVAXVisionPipeline(nn.Module):
    """
    HyperCLOVAX Vision Pipeline for vision token to image generation.

    This pipeline:
    1. Embeds vision tokens using VisionTokenEmbedder
    2. Runs flow matching diffusion with VisionTransformer
    3. Decodes latents to images using VAE
    4. Optionally applies autoguidance with transformer2

    Args:
        od_config: OmniDiffusionConfig containing model configuration
        prefix: Prefix for weight loading (default: "")
    """

    @staticmethod
    def get_dummy_extra() -> dict:
        """Return dummy extra dict for warmup dummy run."""
        import numpy as np

        # token_length=729, vocab_size=65536 per token_embedder/config.json
        return {"vision_tokens": np.zeros((1, 729), dtype=np.int64)}

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

        model = od_config.model
        local_files_only = os.path.exists(model)

        def _load_component_config(subfolder: str) -> dict:
            if os.path.isdir(model):
                cfg_path = os.path.join(model, subfolder, "config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        return json.load(f)
                return {}
            cfg = get_hf_file_to_dict(f"{subfolder}/config.json", model)
            return cfg or {}

        def _build_transformer_kwargs(cfg: dict) -> dict:
            axes_dim = cfg.get("axes_dim", [8, 36, 36])
            return {
                "in_channels": cfg.get("in_channels", 16),
                "vec_in_dim": cfg.get("vec_in_dim", 1536),
                "context_in_dim": cfg.get("context_in_dim", 1536),
                "hidden_size": cfg.get("hidden_size", 1920),
                "mlp_ratio": cfg.get("mlp_ratio", 4.0),
                "num_heads": cfg.get("num_heads", 24),
                "depth_single_blocks": cfg.get("depth_single_blocks", 35),
                "axes_dim": tuple(axes_dim),
                "theta": cfg.get("theta", 10000),
                "use_patchify": cfg.get("use_patchify", False),
            }

        transformer_cfg = _load_component_config("transformer")

        # 1. Load scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        # 2. Load VAE
        self.vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_files_only).to(
            self.device
        )

        # 3. Initialize token embedder — read dims from token_embedder/config.json
        token_embedder_cfg = _load_component_config("token_embedder")
        self.token_embedder = VisionTokenEmbedder(
            vocab_size=token_embedder_cfg.get("vocab_size", 65536),
            embedding_dim=token_embedder_cfg.get("embedding_dim", 1536),
            token_length=token_embedder_cfg.get("token_length", 729),
        )

        # 4. Initialize transformer
        self.transformer = HyperCLOVAXVisionTransformer2DModel(
            od_config=od_config,
            **_build_transformer_kwargs(transformer_cfg),
        )

        # 5. Initialize transformer2 for autoguidance (if available)
        transformer2_exists = False
        if os.path.isdir(model):
            # Local path: check filesystem
            transformer2_path = os.path.join(model, "transformer2")
            transformer2_exists = os.path.exists(transformer2_path)
        else:
            # Remote HF repo: check if transformer2 subfolder exists
            try:
                from huggingface_hub import HfFileSystem

                fs = HfFileSystem()
                transformer2_exists = fs.exists(f"{model}/transformer2")
            except Exception:
                transformer2_exists = False

        if transformer2_exists:
            transformer2_cfg = _load_component_config("transformer2")
            if not transformer2_cfg:
                transformer2_cfg = transformer_cfg
            self.transformer2 = HyperCLOVAXVisionTransformer2DModel(
                od_config=od_config,
                **_build_transformer_kwargs(transformer2_cfg),
            )
        else:
            self.transformer2 = None

        # Weight sources for vLLM loader
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="token_embedder",
                revision=None,
                prefix="token_embedder.",
                fall_back_to_pt=True,
            ),
        ]

        # Add transformer2 weights if available
        if self.transformer2 is not None:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer2",
                    revision=None,
                    prefix="transformer2.",
                    fall_back_to_pt=True,
                )
            )

        # VAE configuration
        self.vae_scale_factor = 8
        self.vae_scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        self.vae_shift_factor = getattr(self.vae.config, "shift_factor", 0.0)

        # Apply USP parallelization if configured
        if od_config.parallel_config.sequence_parallel_size > 1:
            try:
                from .transformer_usp import parallelize_transformer

                self.transformer = parallelize_transformer(self.transformer)
                if self.transformer2 is not None:
                    self.transformer2 = parallelize_transformer(self.transformer2)
                logger.info("USP parallelization applied successfully")
            except ImportError:
                logger.warning("xDiT not available, skipping USP parallelization")

        self.to(self.device)

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Prepare random latents for diffusion."""
        dtype = dtype or self.od_config.dtype

        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = 16  # VAE has 16 latent channels

        shape = (batch_size, latent_channels, latent_h, latent_w)
        latents = torch.randn(shape, device=self.device, dtype=dtype, generator=generator)

        return latents

    def _prepare_img_ids(
        self,
        batch_size: int,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """Prepare position IDs for the transformer."""
        img_ids = torch.zeros(img_h, img_w, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(img_h)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(img_w)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
        return img_ids.to(device=self.device, dtype=self.od_config.dtype)

    def _prepare_vision_spatial(
        self,
        vision_hidden: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """
        Prepare vision spatial features for concatenation with latents.

        Interpolates vision hidden states to match latent spatial dimensions.
        """
        # vision_hidden: (B, L, C) where L is typically 729 (27x27)
        cond_h = cond_w = int(vision_hidden.shape[1] ** 0.5)

        # Reshape to spatial format
        vision_spatial = rearrange(vision_hidden, "b (h w) c -> b c h w", h=cond_h, w=cond_w)

        # Interpolate to match latent size
        vision_spatial = F.interpolate(vision_spatial, size=(img_h, img_w), mode="bilinear", align_corners=False)

        # Reshape back to sequence format
        vision_spatial = rearrange(vision_spatial, "b c h w -> b (h w) c")

        return vision_spatial

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images using VAE."""
        latents = latents / self.vae_scaling_factor + self.vae_shift_factor
        images = self.vae.decode(latents).sample
        return images

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Generate images from vision tokens.

        Args:
            req: OmniDiffusionRequest containing:
                - extra["vision_tokens"]: Vision token IDs (B, L) or (L,)
                - height: Output image height (default: 768)
                - width: Output image width (default: 768)
                - num_inference_steps: Number of diffusion steps (default: 50)
                - guidance_scale: Autoguidance scale (default: 0.0)
                - seed: Random seed (optional)

        Returns:
            DiffusionOutput with generated images
        """
        # Extract vision tokens from request
        vision_tokens = req.extra.get("vision_tokens")
        if vision_tokens is None:
            return DiffusionOutput(output=None, error="vision_tokens required in req.extra")

        # Convert to tensor if needed
        if isinstance(vision_tokens, list):
            vision_tokens = torch.tensor(vision_tokens, dtype=torch.long)
        elif isinstance(vision_tokens, np.ndarray):
            vision_tokens = torch.from_numpy(vision_tokens).long()

        if vision_tokens.ndim == 1:
            vision_tokens = vision_tokens.unsqueeze(0)

        vision_tokens = vision_tokens.to(self.device)
        batch_size = vision_tokens.shape[0]

        # Get parameters from request sampling_params
        sp = req.sampling_params
        height = sp.height if sp.height else 768
        width = sp.width if sp.width else 768
        num_steps = sp.num_inference_steps if sp.num_inference_steps else 50
        guidance_scale = sp.guidance_scale if sp.guidance_scale else 0.0

        # Setup generator for reproducibility
        generator = sp.generator
        if generator is None and sp.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sp.seed)

        dtype = self.od_config.dtype

        # 1. Vision Token Embedding
        vision_cond = self.token_embedder(vision_tokens)
        vision_hidden = vision_cond["vision_last_hidden_state"].to(dtype)
        vision_pooler = vision_cond["vision_pooler_output"].to(dtype)

        # 2. Prepare latents
        latents = self._prepare_latents(batch_size, height, width, dtype=dtype, generator=generator)

        # 3. Prepare position IDs
        img_h = height // self.vae_scale_factor
        img_w = width // self.vae_scale_factor
        img_ids = self._prepare_img_ids(batch_size, img_h, img_w)

        # 4. Prepare vision spatial features
        vision_spatial = self._prepare_vision_spatial(vision_hidden, img_h, img_w)

        # 5. Set timesteps
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Determine if using autoguidance
        use_autoguidance = self.transformer2 is not None and guidance_scale > 0

        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            # Prepare input: concatenate latents with vision spatial
            if self.transformer.use_patchify:
                x_t = rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            else:
                x_t = rearrange(latents, "b c h w -> b (h w) c")

            x_t = torch.cat((x_t, vision_spatial), dim=2)

            # FlowMatchEulerDiscreteScheduler.timesteps are already float sigmas in [0, 1].
            # Do not divide by num_train_timesteps again.
            sigma = torch.full((batch_size,), t.item(), device=self.device, dtype=torch.float32)

            # Forward pass
            pred = self.transformer(
                img=x_t,
                img_ids=img_ids,
                timesteps=sigma,
                y=vision_pooler,
            )

            # Apply autoguidance
            if use_autoguidance:
                pred2 = self.transformer2(
                    img=x_t,
                    img_ids=img_ids,
                    timesteps=sigma,
                    y=vision_pooler,
                )
                pred = pred + guidance_scale * (pred - pred2)

            # Unpatchify prediction
            if self.transformer.use_patchify:
                pred = rearrange(
                    pred,
                    "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=img_h // 2,
                    w=img_w // 2,
                    ph=2,
                    pw=2,
                )
            else:
                pred = rearrange(pred, "b (h w) c -> b c h w", h=img_h, w=img_w)

            # Scheduler step
            latents = self.scheduler.step(pred, t, latents, generator=generator).prev_sample

        # 7. Decode latents
        images = self._decode_latents(latents)

        return DiffusionOutput(output=images)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights using AutoWeightsLoader."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

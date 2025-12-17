"""
Diffusion (DiT + VAE) stage for MammothModa2 in vLLM-Omni.

依赖 mammothmoda2 的 DiT/流匹配逻辑，将 AR 阶段提供的条件特征转为最终图像。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers import AutoConfig
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.mammoth_moda2.mammothmoda2_dit.diffusion_transformer import (
    Transformer2DModel,
)
from vllm_omni.model_executor.models.mammoth_moda2.mammothmoda2_dit.rope_real import RotaryPosEmbedReal
from vllm_omni.model_executor.models.mammoth_moda2.mammothmoda2_dit.schedulers import (
    FlowMatchEulerDiscreteScheduler,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput


class MammothModa2DiTForConditionalGeneration(nn.Module):
    """Diffusion 解码阶段。期望从 additional_information 中获取 AR 条件隐藏态等信息。"""

    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.model_stage = "dit"

        mc = vllm_config.model_config
        trust_remote_code = getattr(mc, "trust_remote_code", True)

        # 组合配置：优先使用已有 hf_config，否则回退下载
        hf_combined = getattr(mc, "hf_config", None)
        if hf_combined is None:
            hf_combined = AutoConfig.from_pretrained(
                mc.model, trust_remote_code=trust_remote_code, revision=mc.revision
            )

        self.gen_dit_config = getattr(hf_combined, "gen_dit_config", None)
        self.gen_vae_config = getattr(hf_combined, "gen_vae_config", None)
        if self.gen_dit_config is None or self.gen_vae_config is None:
            raise ValueError("MammothModa2 config must contain gen_dit_config and gen_vae_config for DiT stage.")

        # 初始化 DiT 与 VAE
        self.gen_transformer = Transformer2DModel.from_config(self.gen_dit_config)
        self.gen_vae = AutoencoderKL.from_config(self.gen_vae_config)

        # 预计算 rope 频率
        self.freqs_cis = RotaryPosEmbedReal.get_freqs_real(
            self.gen_dit_config.axes_dim_rope,
            self.gen_dit_config.axes_lens,
            theta=10000,
        )

        self.latent_channels = self.gen_transformer.config.in_channels
        self.default_vae_scale_factor = getattr(self.gen_vae.config, "scaling_factor", 16) or 16

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        vae_scale_factor: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        h = math.ceil(height / vae_scale_factor)
        w = math.ceil(width / vae_scale_factor)
        shape = (batch_size, self.latent_channels, h, w)
        return torch.randn(shape, device=device, dtype=dtype)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, Any] | None = None,
        **kwargs,
    ) -> OmniOutput:
        """
        期望 additional_information 包含：
        - text_condition_tokens: Tensor [B,L,D]
        - text_condition_attention_mask: Tensor [B,L]
        - image_condition_tokens/image_condition_attention_mask (可选)
        - negative_condition_tokens/negative_attention_mask (可选)
        - ref_latents (可选，列表或张量)
        其他可选 kwargs：num_inference_steps, text_guidance_scale, image_guidance_scale, cfg_range, height, width, vae_scale_factor
        """
        additional_information = additional_information or {}

        text_prompt_embeds = additional_information.get("text_condition_tokens")
        text_prompt_attention_mask = additional_information.get("text_condition_attention_mask")
        if text_prompt_embeds is None or text_prompt_attention_mask is None:
            raise ValueError("DiT forward requires text_condition_tokens and text_condition_attention_mask.")

        image_prompt_embeds = additional_information.get("image_condition_tokens")
        image_prompt_attention_mask = additional_information.get("image_condition_attention_mask")
        negative_prompt_embeds = additional_information.get("negative_condition_tokens")
        negative_attention_mask = additional_information.get("negative_attention_mask")
        ref_latents = additional_information.get("ref_latents")

        device = text_prompt_embeds.device
        dtype = text_prompt_embeds.dtype
        batch_size = text_prompt_embeds.shape[0]

        vae_scale_factor = float(kwargs.get("vae_scale_factor", self.default_vae_scale_factor))
        height = int(kwargs.get("height", 1024))
        width = int(kwargs.get("width", 1024))
        num_inference_steps = int(kwargs.get("num_inference_steps", 50))
        text_guidance_scale = float(kwargs.get("text_guidance_scale", 5.0))
        image_guidance_scale = float(kwargs.get("image_guidance_scale", 2.0))
        cfg_range = tuple(kwargs.get("cfg_range", (0.0, 1.0)))

        latents = kwargs.get("latents", None)
        if latents is None:
            latents = self._prepare_latents(batch_size, height, width, vae_scale_factor, device, dtype)

        scheduler = FlowMatchEulerDiscreteScheduler()
        timesteps = scheduler.set_timesteps(num_inference_steps, device=device)
        freqs_cis = self.freqs_cis.to(device=device, dtype=dtype)

        for i, t in enumerate(timesteps):
            timestep = torch.full((batch_size,), t, device=device, dtype=dtype)

            model_pred = self.gen_transformer(
                hidden_states=latents,
                timestep=timestep,
                text_hidden_states=text_prompt_embeds,
                text_attention_mask=text_prompt_attention_mask,
                ar_image_hidden_states=image_prompt_embeds,
                ar_image_attention_mask=image_prompt_attention_mask,
                ref_image_hidden_states=ref_latents,
                freqs_cis=freqs_cis,
            )

            # 简化 CFG：仅文本/图像引导
            if text_guidance_scale > 1.0 and negative_prompt_embeds is not None:
                model_pred_uncond = self.gen_transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    text_hidden_states=negative_prompt_embeds,
                    text_attention_mask=negative_attention_mask,
                    ref_image_hidden_states=None,
                    freqs_cis=freqs_cis,
                )
                model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)

            latents = scheduler.step(model_pred, t, latents, return_dict=False)[0].to(dtype=dtype)

        # VAE decode
        scale = getattr(self.gen_vae.config, "scaling_factor", None)
        shift = getattr(self.gen_vae.config, "shift_factor", None)
        if scale:
            latents = latents / scale
        if shift:
            latents = latents + shift
        images = self.gen_vae.decode(latents, return_dict=False)[0]

        # multimodal_outputs 返回图像张量列表；text_hidden_states 填充占位
        return OmniOutput(text_hidden_states=torch.zeros(1, device=device), multimodal_outputs=images)

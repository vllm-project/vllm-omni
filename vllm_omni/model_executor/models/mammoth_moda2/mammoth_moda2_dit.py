from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.mammoth_moda2.configuration_mammothmoda2 import Mammothmoda2Config

from .mammothmoda2_dit import (
    FlowMatchEulerDiscreteScheduler,
    RotaryPosEmbedReal,
    SimpleQFormerImageRefiner,
    Transformer2DModel,
)
from .mammothmoda2_dit.rmsnorm import Qwen2RMSNorm


class MammothModa2DiTForConditionalGeneration(nn.Module):
    """
    MammothModa2 的 DiT + VAE 生成阶段（非自回归）。

    该 stage 期望从上游 AR stage 拿到“图像条件 token 的 hidden states”，并通过
    diffusion transformer + VAE decode 输出图像张量。

    说明：
    - vLLM-Omni 的 `GPUGenerationModelRunner` 会调用 `forward(...)` 并将结果
      作为 pooling_output 透传给上层。
    - 为兼容 runner 的解包逻辑，这里使用 `OmniOutput(multimodal_outputs=...)`
      返回生成结果，`text_hidden_states` 仅作为占位张量。
    """

    have_multimodal_outputs = True

    # 只加载 gen_* 权重；忽略 llm_model.*（避免 DiT stage 把整套 LLM 权重也加载进来）
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llm_model.": None,
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix

        hf_config = vllm_config.model_config.hf_config
        if not isinstance(hf_config, Mammothmoda2Config):
            raise TypeError(f"Expected Mammothmoda2Config, got {type(hf_config)}")

        self.config = hf_config

        # --- Build DiT / VAE modules (names must match checkpoint keys) ---
        if self.config.gen_vae_config is None or self.config.gen_dit_config is None:
            raise ValueError("Mammothmoda2Config.gen_vae_config / gen_dit_config must not be None")

        self.gen_vae = AutoencoderKL.from_config(self.config.gen_vae_config)
        self.gen_transformer = Transformer2DModel.from_config(self.config.gen_dit_config)

        llm_hidden_size = int(getattr(self.config.llm_config, "hidden_size", 0) or 0)
        if llm_hidden_size <= 0:
            raise ValueError("Failed to infer llm hidden_size from Mammothmoda2Config.llm_config.hidden_size")
        self._reinit_caption_embedder(llm_hidden_size)

        # Optional: image condition refiner (Q-Former)
        if self.config.gen_image_condition_refiner_config is not None:
            self.gen_image_condition_refiner = SimpleQFormerImageRefiner(
                hidden_size=llm_hidden_size,
                **self.config.gen_image_condition_refiner_config,
            )
        else:
            self.gen_image_condition_refiner = None

        # Precompute rotary freqs for diffusion transformer
        # IMPORTANT: follow upstream mammothmoda: use top-level `config.gen_axes_*`
        # (the checkpoint's `gen_dit_config.axes_lens` can be as small as 1024,
        # which is insufficient for vLLM dummy-run/cudagraph warmup).
        self.gen_freqs_cis = RotaryPosEmbedReal.get_freqs_real(
            tuple(self.config.gen_axes_dim_rope),
            tuple(self.config.gen_axes_lens),
            theta=10000,
        )

        # vLLM PP interface compatibility
        self.make_empty_intermediate_tensors = lambda: None

    def _reinit_caption_embedder(self, in_features: int) -> None:
        # 与上游 Mammothmoda2Model 的 `reinit_caption_embedder` 对齐：
        # 用 Qwen2RMSNorm(in_features) + Linear(in_features -> out_features)
        out_features = int(getattr(self.gen_transformer, "hidden_size", 0) or self.gen_transformer.config.hidden_size)
        self.gen_transformer.time_caption_embed.caption_embedder = nn.Sequential(
            Qwen2RMSNorm(in_features, eps=1e-5),
            nn.Linear(in_features, out_features, bias=True),
        )

    @torch.inference_mode()
    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> OmniOutput:
        
        runtime_addi = kwargs.get("runtime_additional_information", None)
        text_cond = None
        image_cond = None
        negative_cond = None
        negative_attention_mask = None
        legacy_cond = None
        image_hw = (512, 512)
        text_guidance_scale = 1.0
        cfg_range = (0.0, 1.0)
        num_inference_steps = 50

        if isinstance(runtime_addi, list) and runtime_addi and isinstance(runtime_addi[0], dict):
            # runtime_addi will be none in dummy/profile runs
            info = runtime_addi[0]
            text_cond = info["text_prompt_embeds"]
            image_cond = info["image_prompt_embeds"]
            negative_cond = info.get("negative_prompt_embeds")
            negative_attention_mask = info.get("negative_prompt_attention_mask")
            image_hw = info["image_height"][0], info["image_width"][0]
            text_guidance_scale = info["text_guidance_scale"][0]
            cfg_range = info["cfg_range"][0], info["cfg_range"][1]
            num_inference_steps = info["num_inference_steps"][0]

        # Dummy/profile fallback: use inputs_embeds (often zeros)
        if text_cond is None and image_cond is None and legacy_cond is None:
            # runtime_addi will be none in dummy/profile runs
            legacy_cond = inputs_embeds

        # Move to model device/dtype.
        #
        # NOTE: The DiT weights are typically bf16 in vLLM-Omni runs; forcing
        # fp16 here will cause matmul dtype mismatch inside the refiner / DiT.
        model_device = next(self.parameters()).device
        if self.gen_image_condition_refiner is not None:
            target_dtype = next(self.gen_image_condition_refiner.parameters()).dtype
        else:
            target_dtype = next(self.gen_transformer.parameters()).dtype

        def _ensure_2d(x: torch.Tensor, name: str) -> torch.Tensor:
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            if x.ndim != 2:
                raise ValueError(f"Expected {name} to be 2D [T,H], got shape={tuple(x.shape)}")
            return x

        if text_cond is not None or image_cond is not None:
            # Preferred path: separate text/image conditions from preprocess.
            text_cond = _ensure_2d(text_cond, "text_prompt_embeds")
            image_cond = _ensure_2d(image_cond, "image_prompt_embeds")
            text_cond = text_cond.to(device=model_device, dtype=target_dtype, non_blocking=True).contiguous()
            image_cond = image_cond.to(device=model_device, dtype=target_dtype, non_blocking=True).contiguous()

            text_embeds = text_cond.unsqueeze(0)  # [1, T_text, H]
            text_attention_mask = torch.ones(
                (1, text_embeds.shape[1]),
                dtype=torch.bool,
                device=text_embeds.device,
            )

            image_embeds = image_cond.unsqueeze(0)  # [1, T_img, H]
            image_attention_mask = torch.ones(
                (1, image_embeds.shape[1]),
                dtype=torch.bool,
                device=image_embeds.device,
            )

            # Apply optional refiner ONLY on image condition tokens.
            if self.gen_image_condition_refiner is not None and image_embeds.shape[1] > 0:
                image_embeds = self.gen_image_condition_refiner(image_embeds, ~image_attention_mask.bool())
                image_attention_mask = torch.ones(
                    image_embeds.shape[:2],
                    dtype=torch.bool,
                    device=image_embeds.device,
                )

            prompt_embeds = torch.cat([text_embeds, image_embeds], dim=1)
            prompt_attention_mask = torch.cat([text_attention_mask, image_attention_mask], dim=1)
        else:
            # Legacy path: a single concatenated condition tensor.
            # use in dummy/profile runs
            if legacy_cond is None:
                raise ValueError("DiT stage expects condition embeddings, but none were provided.")
            legacy_cond = _ensure_2d(legacy_cond, "prompt_embeds")
            legacy_cond = legacy_cond.to(device=model_device, dtype=target_dtype, non_blocking=True).contiguous()
            prompt_embeds = legacy_cond.unsqueeze(0)
            prompt_attention_mask = torch.ones(
                (1, prompt_embeds.shape[1]),
                dtype=torch.bool,
                device=prompt_embeds.device,
            )

        # Prepare negative prompt (for CFG). If none provided, fall back to unconditional.
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        if text_guidance_scale > 1.0:
            if negative_cond is not None:
                negative_cond = _ensure_2d(negative_cond, "negative_prompt_embeds")
                negative_prompt_embeds = negative_cond.to(
                    device=model_device, dtype=target_dtype, non_blocking=True
                ).contiguous().unsqueeze(0)
                if isinstance(negative_attention_mask, torch.Tensor):
                    neg_mask = negative_attention_mask
                elif isinstance(negative_attention_mask, list):
                    neg_mask = torch.tensor(negative_attention_mask, dtype=torch.bool)
                else:
                    neg_mask = None
                if neg_mask is None:
                    negative_prompt_attention_mask = torch.ones(
                        (1, negative_prompt_embeds.shape[1]),
                        dtype=torch.bool,
                        device=negative_prompt_embeds.device,
                    )
                else:
                    neg_mask = neg_mask.to(device=negative_prompt_embeds.device, dtype=torch.bool)
                    if neg_mask.ndim == 1:
                        neg_mask = neg_mask.unsqueeze(0)
                    negative_prompt_attention_mask = neg_mask
            else:
                hidden_size = int(prompt_embeds.shape[-1])
                negative_prompt_embeds = torch.zeros(
                    (1, 0, hidden_size),
                    dtype=target_dtype,
                    device=prompt_embeds.device,
                )
                negative_prompt_attention_mask = torch.zeros(
                    (1, 0),
                    dtype=torch.bool,
                    device=prompt_embeds.device,
                )

        # Output image size (px), passed from stage input processor.
        height, width = image_hw
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image size: {height}x{width}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Image size must be multiples of 16, got {height}x{width}")
        vae_scale_factor = 16

        latent_channels = int(self.gen_transformer.config.in_channels)
        shape = (1, latent_channels, 2 * height // vae_scale_factor, 2 * width // vae_scale_factor)
        latents = randn_tensor(shape, device=prompt_embeds.device, dtype=prompt_embeds.dtype)

        scheduler = FlowMatchEulerDiscreteScheduler()

        scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=prompt_embeds.device,
            num_tokens=latents.shape[-2] * latents.shape[-1],
        )

        # Run diffusion loop (CFG supported when text_guidance_scale > 1.0)
        total_steps = max(1, len(scheduler.timesteps))
        for i, t in enumerate(scheduler.timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            model_pred = self.gen_transformer(
                hidden_states=latents,
                timestep=timestep,
                text_hidden_states=prompt_embeds,
                text_attention_mask=prompt_attention_mask,
                ref_image_hidden_states=None,
                freqs_cis=self.gen_freqs_cis,
            )
            guidance_scale = (
                text_guidance_scale
                if cfg_range[0] <= i / total_steps <= cfg_range[1]
                else 1.0
            )
            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                model_pred_uncond = self.gen_transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    text_hidden_states=negative_prompt_embeds,
                    text_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=None,
                    freqs_cis=self.gen_freqs_cis,
                )
                model_pred = model_pred_uncond + guidance_scale * (model_pred - model_pred_uncond)
            latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]
            latents = latents.to(dtype=prompt_embeds.dtype)

        # VAE decode
        if self.gen_vae.config.scaling_factor is not None:
            latents = latents / self.gen_vae.config.scaling_factor
        if self.gen_vae.config.shift_factor is not None:
            latents = latents + self.gen_vae.config.shift_factor
        image = self.gen_vae.decode(latents, return_dict=False)[0]

        return OmniOutput(
            text_hidden_states=inputs_embeds,  # 占位，runner 不会用到
            multimodal_outputs=image,
            intermediate_tensors=None,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:  # noqa: ARG002
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

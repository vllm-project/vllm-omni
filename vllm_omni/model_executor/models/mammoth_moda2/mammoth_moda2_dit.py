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


class MammothModa2DiTForConditionalGeneration(nn.Module, SupportsPP):
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
        axes_dim_rope = getattr(self.gen_transformer.config, "axes_dim_rope", None) or self.config.gen_axes_dim_rope
        axes_lens = getattr(self.gen_transformer.config, "axes_lens", None) or self.config.gen_axes_lens
        self.gen_freqs_cis = RotaryPosEmbedReal.get_freqs_real(axes_dim_rope, axes_lens, theta=10000)

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
        input_ids: torch.Tensor | None = None,  # noqa: ARG002
        positions: torch.Tensor | None = None,  # noqa: ARG002
        intermediate_tensors: Any | None = None,  # noqa: ARG002
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, object] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> OmniOutput:
        # MammothModa2 follows the Qwen2.5-Omni pattern: pass condition embeddings
        # via `additional_information["prompt_embeds"]` to avoid vLLM V1 treating
        # prompt_embeds as a serialized payload and calling `len()` on it during
        # request state initialization.
        cond: torch.Tensor | None = None
        if isinstance(additional_information, dict):
            pe = additional_information.get("prompt_embeds")
            if isinstance(pe, torch.Tensor):
                cond = pe

        # Preferred: per-step request-scoped mapping from runner
        if cond is None:
            addi_by_req = kwargs.get("additional_information_by_req_id")
            req_ids = kwargs.get("request_ids")
            info = None
            if isinstance(addi_by_req, dict) and isinstance(req_ids, list) and req_ids and isinstance(req_ids[0], str):
                info = addi_by_req.get(req_ids[0])
            elif isinstance(addi_by_req, dict) and addi_by_req:
                # Best-effort: take the first entry
                info = next(iter(addi_by_req.values()))
            if isinstance(info, dict):
                pe = info.get("prompt_embeds")
                if isinstance(pe, torch.Tensor):
                    cond = pe

        # Fallback: runtime_additional_information (per-request dicts)
        if cond is None:
            runtime_addi = kwargs.get("runtime_additional_information")
            if isinstance(runtime_addi, list) and runtime_addi and isinstance(runtime_addi[0], dict):
                pe = runtime_addi[0].get("prompt_embeds")
                if isinstance(pe, torch.Tensor):
                    cond = pe
            elif isinstance(runtime_addi, dict):
                # Map form: {req_id: {...}} (best-effort: take first entry)
                for v in runtime_addi.values():
                    if isinstance(v, dict) and isinstance(v.get("prompt_embeds"), torch.Tensor):
                        cond = v["prompt_embeds"]
                        break

        # Dummy/profile fallback: use inputs_embeds (often zeros)
        if cond is None:
            cond = inputs_embeds

        if cond is None:
            raise ValueError("DiT stage expects condition embeddings, but none were provided.")

        # Normalize to token-major [T,H]
        if cond.ndim == 3 and cond.shape[0] == 1:
            cond = cond[0]
        if cond.ndim != 2:
            raise ValueError(f"Expected condition embeddings to be 2D [T,H], got shape={tuple(cond.shape)}")

        # Move to model device/dtype.
        #
        # NOTE: The DiT weights are typically bf16 in vLLM-Omni runs; forcing
        # fp16 here will cause matmul dtype mismatch inside the refiner / DiT.
        model_device = next(self.parameters()).device
        if self.gen_image_condition_refiner is not None:
            target_dtype = next(self.gen_image_condition_refiner.parameters()).dtype
        else:
            target_dtype = next(self.gen_transformer.parameters()).dtype
        cond = cond.to(device=model_device, dtype=target_dtype, non_blocking=True).contiguous()

        prompt_embeds = cond.unsqueeze(0)  # [1, T, H]
        prompt_attention_mask = torch.ones(
            (1, prompt_embeds.shape[1]),
            dtype=torch.bool,
            device=prompt_embeds.device,
        )

        # Apply optional refiner (keeps shape)
        if self.gen_image_condition_refiner is not None:
            prompt_embeds = self.gen_image_condition_refiner(prompt_embeds, ~prompt_attention_mask.bool())
            prompt_attention_mask = torch.ones(
                prompt_embeds.shape[:2],
                dtype=torch.bool,
                device=prompt_embeds.device,
            )

        # TODO: 后续从上游 prompt 解析分辨率；先固定 512x512 以跑通端到端。
        height = 512
        width = 512
        vae_scale_factor = 16

        latent_channels = int(self.gen_transformer.config.in_channels)
        shape = (1, latent_channels, 2 * height // vae_scale_factor, 2 * width // vae_scale_factor)
        latents = randn_tensor(shape, device=prompt_embeds.device, dtype=prompt_embeds.dtype)

        scheduler = FlowMatchEulerDiscreteScheduler()
        num_inference_steps = 50

        timesteps = scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=prompt_embeds.device,
            num_tokens=latents.shape[-2] * latents.shape[-1],
        )
        # diffusers 风格：set_timesteps 返回 None，但会写入 scheduler.timesteps
        _ = timesteps

        # Run diffusion loop (no CFG for now)
        for t in scheduler.timesteps:
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            model_pred = self.gen_transformer(
                hidden_states=latents,
                timestep=timestep,
                text_hidden_states=prompt_embeds,
                text_attention_mask=prompt_attention_mask,
                ref_image_hidden_states=None,
                freqs_cis=self.gen_freqs_cis,
            )
            latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]

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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CheersGenerationPipeline — diffusion pipeline for the Cheers (UMM) model.

Ports the image generation path from HF modeling_umm.py into vLLM-Omni's
diffusion engine.  Uses standalone re-implementations of Siglip2VisionTransformer,
UMMTextModel (Qwen2 with bool-mask SDPA), and all generation modules to avoid
version-specific dependencies on the HF custom code.

The denoising loop follows the flow-matching formulation:

    For each timestep t in schedule:
        1. x_t → VAEDecoderProjector → pixel-like
        2. pixel-like → SigLIP2 vision encoder → features
        3. features → UndProjector → LLM-space tokens
        4. LM forward (with cached KV from AR stage) → hidden states
        5. hidden → GenProjector (7-layer DiT) → drift (patch)
        6. Unpatchify 2×2
        7. SigLIP2 patch embeddings of x_t (semantic residual)
        8. HiGate(drift, patch_embeddings) → HiProjector (3-layer DiT) → velocity
        9. Euler step: x_{t+1} = x_t + velocity * dt
    VAE decode final latent → PIL image
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from PIL import Image
from torch import nn
from transformers import AutoTokenizer
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.cheers.cheers_modules import (
    CheersGenProjector,
    CheersHiProjector,
    CheersQwen2Config,
    CheersUndProjector,
    CheersVAEDecoderProjector,
    CheersVAEModel,
    HiGate,
    Siglip2VisionConfig,
    Siglip2VisionTransformer,
    TimestepEmbedder,
    UMMTextModel,
    _crop_kv_cache,
    _kv_seq_len,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

IM_START_ID = 151667
IM_END_ID = 151668
NO_MEAN_ID = 151669
EOS_TOKEN_ID = 151645


@dataclass
class CheersGenParams:
    num_timesteps: int = 50
    cfg_scale: float = 9.5
    alpha: float = 1.0


def get_cheers_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(x):
        return x

    return post_process_func


def _parse_config(model_path: str):
    """Parse the Cheers config.json and return structured sub-configs."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        raw = json.load(f)

    text_cfg = CheersQwen2Config.from_dict(raw.get("text_config", {}))
    vit_cfg = Siglip2VisionConfig.from_dict(raw.get("vision_representation_config", {}))

    vae_enc_cfg = raw.get("vae_encoder_config", {})
    vae_dec_cfg = raw.get("vae_decoder_config", {})
    z_channels = vae_enc_cfg.get("z_channels", 32)

    return text_cfg, vit_cfg, vae_enc_cfg, vae_dec_cfg, z_channels


class CheersGenerationPipeline(nn.Module):
    """Cheers diffusion pipeline for T2I/I2I generation via vLLM-Omni."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

        model = od_config.model
        local_files_only = os.path.exists(model)
        if local_files_only:
            model_path = model
        else:
            from vllm_omni.model_executor.model_loader.weight_utils import (
                download_weights_from_hf_specific,
            )

            model_path = download_weights_from_hf_specific(model, od_config.revision, ["*"])

        text_config, vit_config, vae_enc_cfg, vae_dec_cfg, z_channels = _parse_config(model_path)

        llm_hidden_size = text_config.hidden_size
        vit_hidden_size = vit_config.hidden_size
        latent_channels = 4 * z_channels  # 128

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        # Build model components using standalone modules
        self.language_model = UMMTextModel(text_config)
        self.vision_representation = Siglip2VisionTransformer(vit_config)
        self.und_projector = CheersUndProjector(
            image_embed_dim=vit_hidden_size,
            text_embed_dim=llm_hidden_size,
        )
        self.vae_model = CheersVAEModel(vae_enc_cfg, vae_dec_cfg, z_channels)
        self.vae_decoder_projector = CheersVAEDecoderProjector(vae_dec_cfg, z_channels)
        self.time_embed = TimestepEmbedder(
            hidden_size_1=llm_hidden_size,
            hidden_size_2=vit_hidden_size,
        )
        self.gen_projector = CheersGenProjector(
            embed_dim=llm_hidden_size,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            patch_size=2,
            output_dim=vit_hidden_size,
            layers_num=7,
        )
        self.hi_gate = HiGate(embed_dim=vit_hidden_size)
        self.hi_projector = CheersHiProjector(
            embed_dim=vit_hidden_size,
            num_attention_heads=vit_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            patch_size=1,
            output_dim=vae_dec_cfg.get("ch", 128) if isinstance(vae_dec_cfg, dict) else getattr(vae_dec_cfg, "ch", 128),
            layers_num=3,
        )

        self.latent_channels = latent_channels
        self.llm_hidden_size = llm_hidden_size
        self.vit_hidden_size = vit_hidden_size
        self.model_path = model_path

        # Weight sources for vLLM-Omni loader
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=False,
            )
        ]

        self.to(device=self.device, dtype=self.od_config.dtype)

    def _load_pretrained_weights(self, model_path: str) -> None:
        """Load pretrained weights from safetensors files."""
        import glob

        from safetensors import safe_open

        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not safetensor_files:
            logger.warning("No safetensors files found in %s", model_path)
            return

        def _weight_iter():
            for sf_path in safetensor_files:
                with safe_open(sf_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        yield key, f.get_tensor(key)

        self.load_weights(_weight_iter())

    def _drift_fn(
        self,
        x_t: torch.Tensor,
        t: float,
        attention_mask: torch.Tensor,
        past_key_values,
    ) -> tuple[torch.Tensor, object]:
        """Single denoising step following HF Cheers._drift_fn."""
        t_tensor = torch.full((x_t.size(0), 1), t, device=x_t.device, dtype=x_t.dtype)
        t_embeds_1, t_embeds_2 = self.time_embed(t_tensor.squeeze(-1), x_t.dtype)

        x_pixel = self.vae_decoder_projector(x_t)

        interpolate_pos_encoding = x_pixel.size(-1) > 512

        image_feature = self.vision_representation(
            x_pixel, interpolate_pos_encoding=interpolate_pos_encoding
        ).last_hidden_state

        projected_image_feature = self.und_projector(image_feature)
        h_w = int(projected_image_feature.size(1) ** 0.5)

        attn_mask = torch.ones(
            (1, 1, projected_image_feature.size(1), attention_mask.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        step_output = self.language_model(
            inputs_embeds=projected_image_feature,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )

        image_feature_pre = step_output.last_hidden_state[:, -projected_image_feature.size(1) :, :]

        drift = self.gen_projector(image_feature_pre, t_embeds_1)

        # Unpatchify 2×2
        drift = drift.reshape(drift.size(0), h_w, h_w, drift.size(-1))
        B, H, W, C = drift.shape
        P = 2
        D = C // (P * P)
        drift = drift.view(B, H, W, P, P, D)
        drift = drift.permute(0, 1, 3, 2, 4, 5).contiguous()
        drift = drift.view(B, H * P, W * P, D)
        drift = drift.view(B, H * P * W * P, D)

        patch_embedding_res = self.vision_representation.embeddings(
            x_pixel, interpolate_pos_encoding=interpolate_pos_encoding
        )

        hi_input = self.hi_gate(drift, patch_embedding_res)

        velocity = self.hi_projector(hi_input, t_embeds_2)
        velocity = velocity.view(B, h_w * 2, h_w * 2, velocity.size(-1))
        velocity = velocity.permute(0, 3, 1, 2)

        return velocity, step_output

    @staticmethod
    def _time_shift(ts: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return (alpha * ts) / (1.0 + (alpha - 1.0) * ts)

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning("Cheers pipeline only supports single prompt; using first.")

        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")

        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}

        height = int(req.sampling_params.height) if req.sampling_params.height else 512
        width = int(req.sampling_params.width) if req.sampling_params.width else 512
        height = max(256, (height // 16) * 16)
        width = max(256, (width // 16) * 16)

        gen_params = CheersGenParams(
            num_timesteps=int(req.sampling_params.num_inference_steps or 50),
            cfg_scale=float(extra_args.get("cfg_scale", 9.5)),
            alpha=float(extra_args.get("alpha", 1.0)),
        )

        image_h = height // 16
        image_w = width // 16
        per_image_token = (image_h * image_w) // 4

        injected_kv = req.sampling_params.past_key_values

        if injected_kv is not None:
            past_key_values = injected_kv
            logger.info(
                "Using injected KV Cache from AR stage, seq_len=%d",
                _kv_seq_len(past_key_values),
            )
            use_cfg = False
            seq_len = _kv_seq_len(past_key_values)
            attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
        else:
            logger.info("Standalone diffusion mode: prefilling text prompt")
            past_key_values, attention_mask, use_cfg = self._standalone_prefill(
                prompt,
                gen_params,
                image_h,
                image_w,
            )
            if use_cfg:
                uncond_past_key_values = self._uncond_kv_cache

        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        num_inference_steps = gen_params.num_timesteps
        t_list = torch.linspace(0, 1, num_inference_steps)
        t_list = self._time_shift(t_list, alpha=gen_params.alpha)
        x_t = torch.randn(
            (1, self.latent_channels, image_h, image_w),
            dtype=next(self.parameters()).dtype,
            device=self.device,
        )

        last_step_size = 1.0 / num_inference_steps

        z_mask = torch.ones((1, per_image_token), dtype=attention_mask.dtype, device=self.device)
        attention_mask = torch.cat([attention_mask, z_mask], dim=1)

        for n in range(num_inference_steps):
            ti = t_list[n].item()

            velocity, step_output = self._drift_fn(x_t, ti, attention_mask, past_key_values)

            if use_cfg:
                uncond_velocity, uncond_step_output = self._drift_fn(
                    x_t,
                    ti,
                    attention_mask,
                    uncond_past_key_values,
                )
                velocity = uncond_velocity + gen_params.cfg_scale * (velocity - uncond_velocity)

            if ti != 1 and n < num_inference_steps - 1:
                dt = t_list[n + 1].item() - ti
                x_t = x_t + velocity * dt
                _crop_kv_cache(past_key_values, per_image_token)
                if use_cfg:
                    _crop_kv_cache(uncond_past_key_values, per_image_token)
            else:
                x_t = x_t + velocity * last_step_size

        with torch.no_grad():
            vae_dtype = next(self.vae_model.parameters()).dtype
            decoded = self.vae_model.decode(x_t.to(vae_dtype))
            image_array = decoded.clamp(0, 1)[0].permute(1, 2, 0) * 255
            pil_image = Image.fromarray(image_array.to(torch.uint8).cpu().numpy())

        return DiffusionOutput(output=pil_image)

    def _standalone_prefill(
        self,
        prompt: str,
        gen_params: CheersGenParams,
        image_h: int,
        image_w: int,
    ) -> tuple[object, torch.Tensor, bool]:
        """Replicate HF Cheers model's T2I prefill: tokenize + LLM forward + build KV cache."""
        messages = [{"role": "user", "content": prompt}]

        # Build input_ids using tokenizer chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

        inputs_embeds = self.language_model.embed_tokens(input_ids)
        img_len = 0
        txt_len = inputs_embeds.size(1)

        total_len = img_len + txt_len
        head_num = self.language_model.config.num_attention_heads

        omni_mask = torch.tril(torch.ones((1, head_num, total_len, total_len), dtype=torch.long, device=self.device))

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=omni_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values

        use_cfg = gen_params.cfg_scale > 1.0
        if use_cfg:
            uncond_ids = self._build_uncond_ids(input_ids)
            uncond_text_embeds = self.language_model.embed_tokens(uncond_ids)
            uncond_outputs = self.language_model(
                inputs_embeds=uncond_text_embeds,
                attention_mask=omni_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            self._uncond_kv_cache = uncond_outputs.past_key_values

        attention_mask = torch.ones((1, total_len), dtype=torch.long, device=self.device)
        return past_key_values, attention_mask, use_cfg

    def _build_uncond_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build unconditional input_ids for CFG by replacing text with NO_MEAN_ID."""
        new_ids = input_ids.clone()
        bsz, seqlen = new_ids.shape
        for b in range(bsz):
            in_img_block = False
            for t_idx in range(seqlen):
                tok = int(new_ids[b, t_idx].item())
                if tok == IM_START_ID:
                    in_img_block = True
                elif tok == IM_END_ID and in_img_block:
                    in_img_block = False
                elif not in_img_block:
                    new_ids[b, t_idx] = NO_MEAN_ID
        return new_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        state = self.state_dict()
        allowed = set(state.keys())
        shapes = {k: tuple(v.shape) for k, v in state.items()}
        loaded: set[str] = set()
        skipped: set[str] = set()

        _prefix_map = {
            "model.language_model.": "language_model.",
            "model.vision_representation.": "vision_representation.",
            "model.und_projector.": "und_projector.",
            "model.vae_model.": "vae_model.",
            "model.vae_decoder_projector.": "vae_decoder_projector.",
            "model.time_embed.": "time_embed.",
            "model.gen_projector.": "gen_projector.",
            "model.hi_gate.": "hi_gate.",
            "model.hi_projector.": "hi_projector.",
        }

        def _normalize(name: str) -> str:
            for pfx_src, pfx_dst in _prefix_map.items():
                if name.startswith(pfx_src):
                    return pfx_dst + name[len(pfx_src) :]
            return name

        for orig_name, tensor in weights:
            name = _normalize(orig_name)
            if name not in allowed:
                skipped.add(orig_name)
                continue
            expected_shape = shapes.get(name)
            if expected_shape and tuple(tensor.shape) != expected_shape:
                logger.warning(
                    "Shape mismatch for %s: expected %s, got %s; skipping",
                    name,
                    expected_shape,
                    tuple(tensor.shape),
                )
                skipped.add(orig_name)
                continue
            param = self
            parts = name.split(".")
            for part in parts[:-1]:
                param = getattr(param, part)
            leaf = parts[-1]
            target = getattr(param, leaf)
            if isinstance(target, nn.Parameter):
                target.data.copy_(tensor)
            else:
                target.copy_(tensor)
            loaded.add(name)

        if skipped:
            logger.info("Skipped %d weights not matching pipeline state_dict", len(skipped))
        logger.info("Loaded %d / %d weights", len(loaded), len(loaded) + len(skipped))
        return loaded

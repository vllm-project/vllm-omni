# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal Bagel generation core, adapted from Bagel `modeling/bagel/bagel.py`.
# This is intentionally self-contained to avoid importing external Bagel package.

from __future__ import annotations

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from .modeling_utils import PositionEmbedding, TimestepEmbedder
from .qwen2_navit import NaiveCache, Qwen2ForCausalLM


class BagelConfig(PretrainedConfig):
    model_type = "bagel"

    def __init__(
        self,
        *,
        visual_gen: bool = True,
        llm_config=None,
        vae_config=None,
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
        timestep_shift: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.llm_config = llm_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.timestep_shift = timestep_shift


class Bagel(nn.Module):
    """Bagel core model for visual generation (flow in latent space)."""

    def __init__(self, language_model: Qwen2ForCausalLM, config: BagelConfig):
        super().__init__()
        self.language_model = language_model
        self.config = config

        # Visual generation params
        assert config.visual_gen, "Bagel core in vllm-omni currently supports visual_gen only."
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in getattr(config.llm_config, "layer_module", "")
        self.latent_patch_size = config.latent_patch_size
        self.timestep_shift = config.timestep_shift
        self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
        self.max_latent_size = config.max_latent_size
        self.latent_channel = config.vae_config.z_channels
        self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel

        self.time_embedder = TimestepEmbedder(self.hidden_size)
        self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
        self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
        self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = []
        packed_text_position_ids = []
        text_token_lens = []
        packed_text_indexes = []
        packed_key_value_indexes = []

        curr = 0
        newlens, new_rope = [], []
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids["bos_token_id"]] + text_ids + [new_token_ids["eos_token_id"]]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        return generation_input, newlens, new_rope

    @torch.no_grad()
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ) -> NaiveCache:
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        extra_inputs = {"mode": "und"} if self.use_moe else {}
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        return output.past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = [], []
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = [], [], []
        packed_seqlens, packed_position_ids, packed_indexes = [], [], []
        packed_key_value_indexes = []

        p = self.latent_patch_size
        curr = _curr = 0
        newlens, new_rope = [], []
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_vae_token_indexes.extend(range(_curr, _curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            _curr += num_image_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_seqlens.append(num_image_tokens + 2)
            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

            # latent position ids: 2D flattened
            coords_h = torch.arange(0, h)
            coords_w = torch.arange(0, w)
            pos_ids = (coords_h[:, None] * self.max_latent_size + coords_w).flatten()
            packed_vae_position_ids.append(pos_ids)

            # init noise in packed latent space (per-token patch vector)
            packed_init_noises.append(torch.randn((num_image_tokens, p * p * self.latent_channel)))

            newlens.append(curr_kvlen + num_image_tokens + 2)
            new_rope.append(curr_position_id + (num_image_tokens + 2))

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0).to(torch.long),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        return generation_input, newlens, new_rope

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = [], [], []
        curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        return {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

    @torch.no_grad()
    def generate_image(
        self,
        *,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 50,
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: tuple[float, float] = (0.4, 1.0),
        cfg_text_scale: float = 4.0,
        cfg_text_packed_query_indexes: torch.LongTensor | None = None,
        cfg_text_packed_position_ids: torch.LongTensor | None = None,
        cfg_text_past_key_values: NaiveCache | None = None,
        cfg_text_key_values_lens: torch.IntTensor | None = None,
        cfg_text_packed_key_value_indexes: torch.LongTensor | None = None,
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: torch.LongTensor | None = None,
        cfg_img_packed_position_ids: torch.LongTensor | None = None,
        cfg_img_past_key_values: NaiveCache | None = None,
        cfg_img_key_values_lens: torch.IntTensor | None = None,
        cfg_img_packed_key_value_indexes: torch.LongTensor | None = None,
        cfg_type: str = "parallel",
    ):
        # Keep dtype consistent with model parameters. In some environments autocast
        # may not cover all paths, and BF16 weights + FP32 inputs will error:
        # "Input type(float) and bias type (c10::BFloat16) should be the same".
        param_dtype = self.vae2llm.weight.dtype
        x_t = packed_init_noises.to(dtype=param_dtype)
        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in enumerate(timesteps):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device, dtype=param_dtype)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
            )
            x_t = x_t - v_t.to(x_t.device) * dts[i]

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    def _forward_flow(
        self,
        *,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: torch.LongTensor | None = None,
        cfg_text_packed_query_indexes: torch.LongTensor | None = None,
        cfg_text_key_values_lens: torch.IntTensor | None = None,
        cfg_text_past_key_values: NaiveCache | None = None,
        cfg_text_packed_key_value_indexes: torch.LongTensor | None = None,
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: torch.LongTensor | None = None,
        cfg_img_packed_query_indexes: torch.LongTensor | None = None,
        cfg_img_key_values_lens: torch.IntTensor | None = None,
        cfg_img_past_key_values: NaiveCache | None = None,
        cfg_img_packed_key_value_indexes: torch.LongTensor | None = None,
        cfg_type: str = "parallel",
    ) -> torch.Tensor:
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((int(packed_seqlens.sum().item()), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t_emb = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t_emb.dtype != packed_sequence.dtype:
            x_t_emb = x_t_emb.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t_emb

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)[packed_vae_token_indexes]

        # text cfg
        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)[packed_vae_token_indexes]
        else:
            cfg_text_v_t = None

        # img cfg (optional)
        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)[packed_vae_token_indexes]
        else:
            cfg_img_v_t = None

        if cfg_text_scale > 1.0:
            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
            if cfg_img_scale > 1.0:
                v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
            else:
                v_t_ = v_t_text_

            if cfg_renorm_type == "global":
                norm_v_t = torch.norm(v_t)
                norm_v_t_ = torch.norm(v_t_)
            elif cfg_renorm_type == "channel":
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
            else:
                raise NotImplementedError(f"{cfg_renorm_type} not supported")
            scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            v_t = v_t_ * scale

        return v_t

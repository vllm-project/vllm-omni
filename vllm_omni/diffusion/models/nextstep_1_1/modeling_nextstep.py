# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from NextStep-1.1 (https://huggingface.co/stepfun-ai/NextStep-1.1)
# Original: models/nextstep_model.py — local version with TP-aware layers.

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.models.nextstep_1_1.modeling_nextstep_heads import (
    FlowMatchingHead,
)
from vllm_omni.diffusion.models.nextstep_1_1.modeling_nextstep_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

# ---------------------------------------------------------------------------
# Positional embedding utilities (inlined from remote utils/model_utils.py)
# ---------------------------------------------------------------------------


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


# ---------------------------------------------------------------------------
# NextStepConfig — extends LlamaConfig with NextStep-specific fields.
# This mirrors the remote models/config.py.
# ---------------------------------------------------------------------------


class NextStepConfig(LlamaConfig):
    model_type = "nextstep"

    def __init__(
        self,
        vae_name_or_path: str | None = None,
        latent_size: int = 32,
        latent_patch_size: int = 2,
        latent_channels: int = 16,
        boi: int | None = None,
        eoi: int | None = None,
        image_placeholder_id: int | None = None,
        pad_token_id_added: int | None = None,
        lm_loss_weight: float = 0.01,
        im_loss_weight: float = 1.0,
        fm_head_dim: int = 1536,
        fm_head_layers: int = 12,
        fm_head_batch_mul: int = 4,
        o_attention_bias: bool | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vae_name_or_path = vae_name_or_path
        self.latent_size = latent_size
        self.latent_patch_size = latent_patch_size
        self.latent_channels = latent_channels
        self.boi = boi
        self.eoi = eoi
        self.image_placeholder_id = image_placeholder_id
        self.pad_token_id_added = pad_token_id_added
        self.lm_loss_weight = lm_loss_weight
        self.im_loss_weight = im_loss_weight
        self.fm_head_dim = fm_head_dim
        self.fm_head_layers = fm_head_layers
        self.fm_head_batch_mul = fm_head_batch_mul
        self.o_attention_bias = self.attention_bias if o_attention_bias is None else o_attention_bias

    @classmethod
    def from_json(cls, path: str) -> NextStepConfig:
        with open(path) as f:
            data = json.load(f)
        # Remove keys that are not constructor parameters (accept extra keys)
        # LlamaConfig.__init__ uses **kwargs to absorb extras.
        return cls(**data)


# ---------------------------------------------------------------------------
# NextStepModel — main model class
# ---------------------------------------------------------------------------


class NextStepModel(nn.Module):
    def __init__(self, config: NextStepConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # lm_head is part of the checkpoint but not used during image generation
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Image projectors.
        token_dim = config.latent_channels * config.latent_patch_size**2
        self.image_in_projector = nn.Linear(token_dim, config.hidden_size)
        self.image_out_projector = nn.Linear(config.hidden_size, config.hidden_size)

        # Flow-matching head (no TP — tiny network)
        self.image_head = FlowMatchingHead(
            input_dim=token_dim,
            cond_dim=config.hidden_size,
            dim=config.fm_head_dim,
            layers=config.fm_head_layers,
        )

        # Optional generation position embeddings
        if getattr(config, "use_gen_pos_embed", False):
            self._init_gen_pos_embed()

    # ------------------------------------------------------------------
    # Generation positional embeddings
    # ------------------------------------------------------------------

    def _init_gen_pos_embed(self):
        self.register_buffer(
            "gen_pos_embed",
            torch.from_numpy(get_2d_sincos_pos_embed(self.config.hidden_size, self.config.base_image_grid_size))
            .float()
            .unsqueeze(0),
        )

    def gen_pos_embed_with_ar(self, h: int, w: int) -> torch.Tensor:
        bsz, hw, dim = self.gen_pos_embed.shape
        side = int(hw**0.5)
        gen_pos_embed = self.gen_pos_embed.reshape(bsz, side, side, dim)
        gen_pos_embed = gen_pos_embed[:, :h, :w, :]
        return gen_pos_embed.reshape(bsz, -1, dim)

    # ------------------------------------------------------------------
    # Patchify / Unpatchify
    # ------------------------------------------------------------------

    def patchify(self, img: torch.Tensor) -> torch.Tensor:
        bsz, c, h, w = img.shape
        p = self.config.latent_patch_size
        h_, w_ = h // p, w // p
        img = img.reshape(bsz, c, h_, p, w_, p)
        img = torch.einsum("nchpwq->nhwcpq", img)
        return img.reshape(bsz, h_ * w_, c * p**2)

    def unpatchify(
        self,
        x: torch.Tensor,
        h: int | None = None,
        w: int | None = None,
    ) -> torch.Tensor:
        bsz = x.shape[0]
        p = self.config.latent_patch_size
        c = self.config.latent_channels
        if h is None and w is None:
            h_ = w_ = int(x.shape[1] ** 0.5)
        else:
            h_, w_ = h, w
        assert h_ * w_ == x.shape[1], f"Invalid sequence length {x.shape[1]}."
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        return x.reshape(bsz, c, h_ * p, w_ * p)

    # ------------------------------------------------------------------
    # Input embedding preparation
    # ------------------------------------------------------------------

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        latents: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        if latents is None:
            return self.embed_tokens(input_ids)

        bs, seq_length = input_ids.shape
        inputs_embeds = torch.zeros(
            (bs, seq_length, self.config.hidden_size),
            device=self.embed_tokens.weight.device,
            dtype=self.embed_tokens.weight.dtype,
        )
        im_indices = input_ids == self.config.image_placeholder_id
        lm_indices = ~im_indices

        if isinstance(latents, list):
            tokens = torch.cat([self.patchify(latent) for latent in latents], dim=1)
        else:
            tokens = self.patchify(latents)

        image_embeds = self.image_in_projector(tokens)
        image_embeds = image_embeds.view(-1, self.config.hidden_size)

        token_embeds = self.embed_tokens(input_ids[lm_indices])

        inputs_embeds[im_indices] = image_embeds.to(inputs_embeds.dtype)
        inputs_embeds[lm_indices] = token_embeds

        return inputs_embeds

    # ------------------------------------------------------------------
    # Causal mask utilities
    # ------------------------------------------------------------------

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ) -> torch.Tensor | None:
        # For SDPA we build the 4D mask; flash_attention_2 uses None.
        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        if attn_impl == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if attn_impl == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            attn_impl == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

        return causal_mask

    # ------------------------------------------------------------------
    # Forward through decoder layers
    # ------------------------------------------------------------------

    def forward_model(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", True)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # ------------------------------------------------------------------
    # Weight loading with TP sharding support
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name_in_model, weight_name_in_checkpoint, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params

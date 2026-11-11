# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SAM3 memory tracker for video concept segmentation.

Ported from the Hugging Face Transformers SAM3 tracker
(``transformers/models/sam3_tracker_video/modeling_sam3_tracker_video.py``,
Apache-2.0) and kept numerically equivalent to it. Module and parameter names
match the released ``tracker_model.*`` checkpoint tensors. Only the path the
video pipeline drives is kept: objects enter as detector masks on their first
frame and are then propagated through a memory bank of past frames.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.models.sam3.sam3_model import _fields

# Mask logit written where the tracker sees no object.
NO_OBJ_SCORE = -1024.0

_ACT = {"gelu": nn.GELU, "relu": nn.ReLU}


@dataclass
class Sam3TrackerConfig:
    image_size: int = 1008
    patch_size: int = 14
    hidden_dim: int = 256
    backbone_feature_sizes: list[list[int]] = field(default_factory=lambda: [[288, 288], [144, 144], [72, 72]])
    num_maskmem: int = 7
    max_cond_frame_num: int = 4
    max_object_pointers_in_encoder: int = 16
    enable_occlusion_spatial_embedding: bool = True
    enable_temporal_pos_encoding_for_object_pointers: bool = True
    multimask_output_in_sam: bool = True
    multimask_output_for_tracking: bool = True
    multimask_min_pt_num: int = 0
    multimask_max_pt_num: int = 1
    sigmoid_scale_for_mem_enc: float = 20.0
    sigmoid_bias_for_mem_enc: float = -10.0
    # Memory attention.
    memory_attention_hidden_size: int = 256
    memory_attention_num_layers: int = 4
    memory_attention_num_attention_heads: int = 1
    memory_attention_downsample_rate: int = 1
    memory_attention_feed_forward_hidden_size: int = 2048
    memory_attention_feed_forward_hidden_act: str = "relu"
    memory_attention_rope_theta: float = 10000.0
    memory_attention_rope_feat_sizes: list[int] = field(default_factory=lambda: [72, 72])
    # Memory encoder.
    memory_encoder_hidden_size: int = 256
    memory_encoder_output_channels: int = 64
    mask_downsampler_embed_dim: int = 256
    mask_downsampler_kernel_size: int = 3
    mask_downsampler_stride: int = 2
    mask_downsampler_padding: int = 1
    mask_downsampler_total_stride: int = 16
    mask_downsampler_hidden_act: str = "gelu"
    memory_fuser_num_layers: int = 2
    memory_fuser_embed_dim: int = 256
    memory_fuser_intermediate_dim: int = 1024
    memory_fuser_kernel_size: int = 7
    memory_fuser_padding: int = 3
    memory_fuser_hidden_act: str = "gelu"
    # Prompt encoder.
    mask_input_channels: int = 16
    num_point_embeddings: int = 4
    prompt_hidden_act: str = "gelu"
    prompt_layer_norm_eps: float = 1e-6
    # SAM mask decoder.
    mask_decoder_mlp_dim: int = 2048
    mask_decoder_num_hidden_layers: int = 2
    mask_decoder_num_attention_heads: int = 8
    mask_decoder_attention_downsample_rate: int = 2
    num_multimask_outputs: int = 3
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 256

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Sam3TrackerConfig:
        prompt = values.get("prompt_encoder_config") or {}
        decoder = values.get("mask_decoder_config") or {}
        vision = values.get("vision_config") or {}
        rope = values.get("rope_parameters") or {}
        merged = {
            **_fields(cls, values),
            "patch_size": prompt.get("patch_size", cls.patch_size),
            "hidden_dim": vision.get("fpn_hidden_size", cls.hidden_dim),
            "mask_input_channels": prompt.get("mask_input_channels", cls.mask_input_channels),
            "num_point_embeddings": prompt.get("num_point_embeddings", cls.num_point_embeddings),
            "prompt_hidden_act": prompt.get("hidden_act", cls.prompt_hidden_act),
            "prompt_layer_norm_eps": prompt.get("layer_norm_eps", cls.prompt_layer_norm_eps),
            "mask_decoder_mlp_dim": decoder.get("mlp_dim", cls.mask_decoder_mlp_dim),
            "mask_decoder_num_hidden_layers": decoder.get("num_hidden_layers", cls.mask_decoder_num_hidden_layers),
            "mask_decoder_num_attention_heads": decoder.get(
                "num_attention_heads", cls.mask_decoder_num_attention_heads
            ),
            "mask_decoder_attention_downsample_rate": decoder.get(
                "attention_downsample_rate", cls.mask_decoder_attention_downsample_rate
            ),
            "num_multimask_outputs": decoder.get("num_multimask_outputs", cls.num_multimask_outputs),
            "iou_head_depth": decoder.get("iou_head_depth", cls.iou_head_depth),
            "iou_head_hidden_dim": decoder.get("iou_head_hidden_dim", cls.iou_head_hidden_dim),
        }
        if "backbone_feature_sizes" in vision:
            merged["backbone_feature_sizes"] = [list(size) for size in vision["backbone_feature_sizes"]]
        if "rope_theta" in rope:
            merged["memory_attention_rope_theta"] = rope["rope_theta"]
        return cls(**merged)

    @property
    def image_embedding_size(self) -> tuple[int, int]:
        size = self.image_size // self.patch_size
        return size, size

    @property
    def mask_input_size(self) -> tuple[int, int]:
        size = 4 * self.image_size // self.patch_size
        return size, size


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Sam3TrackerLayerNorm(nn.LayerNorm):
    """LayerNorm over the channel axis of an NCHW tensor."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return super().forward(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Sam3TrackerFeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output=False):
        super().__init__()
        self.activation = nn.ReLU()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.proj_in(hidden_states))
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.sigmoid() if self.sigmoid_output else hidden_states


def _sdpa(query, key, value, num_heads: int, scaling: float) -> torch.Tensor:
    """``query``/``key``/``value``: [batch, points, tokens, heads * head_dim] -> same layout."""
    batch, points = query.shape[:2]
    shape = (batch * points, -1, num_heads, query.shape[-1] // num_heads)
    query, key, value = (x.reshape(shape).transpose(1, 2) for x in (query, key, value))
    out = F.scaled_dot_product_attention(query, key, value, scale=scaling)
    return out.transpose(1, 2).reshape(batch, points, -1, shape[2] * shape[3])


class Sam3TrackerAttention(nn.Module):
    """SAM two-way-transformer attention with an optional channel downsample."""

    def __init__(self, hidden_size: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.internal_dim = hidden_size // downsample_rate
        self.num_attention_heads = num_heads
        self.scaling = (self.internal_dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(hidden_size, self.internal_dim)
        self.o_proj = nn.Linear(self.internal_dim, hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        out = _sdpa(self.q_proj(query), self.k_proj(key), self.v_proj(value), self.num_attention_heads, self.scaling)
        return self.o_proj(out)


class Sam3TrackerTwoWayAttentionBlock(nn.Module):
    def __init__(self, config: Sam3TrackerConfig, skip_first_layer_pe: bool):
        super().__init__()
        hidden = config.hidden_dim
        heads = config.mask_decoder_num_attention_heads
        rate = config.mask_decoder_attention_downsample_rate
        self.self_attn = Sam3TrackerAttention(hidden, heads, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.cross_attn_token_to_image = Sam3TrackerAttention(hidden, heads, rate)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.mlp = Sam3TrackerFeedForward(
            hidden, config.mask_decoder_mlp_dim, hidden, num_layers=config.mask_decoder_num_hidden_layers
        )
        self.layer_norm3 = nn.LayerNorm(hidden)
        self.layer_norm4 = nn.LayerNorm(hidden)
        self.cross_attn_image_to_token = Sam3TrackerAttention(hidden, heads, rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_point_embedding, key_point_embedding):
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            query = queries + query_point_embedding
            queries = queries + self.self_attn(query, query, queries)
        queries = self.layer_norm1(queries)

        query = queries + query_point_embedding
        key = keys + key_point_embedding
        queries = self.layer_norm2(queries + self.cross_attn_token_to_image(query, key, keys))

        queries = self.layer_norm3(queries + self.mlp(queries))

        query = queries + query_point_embedding
        key = keys + key_point_embedding
        keys = self.layer_norm4(keys + self.cross_attn_image_to_token(key, query, queries))
        return queries, keys


class Sam3TrackerTwoWayTransformer(nn.Module):
    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Sam3TrackerTwoWayAttentionBlock(config, skip_first_layer_pe=(index == 0))
                for index in range(config.mask_decoder_num_hidden_layers)
            ]
        )
        self.final_attn_token_to_image = Sam3TrackerAttention(
            config.hidden_dim, config.mask_decoder_num_attention_heads, config.mask_decoder_attention_downsample_rate
        )
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_dim)

    def forward(self, point_embeddings, image_embeddings, image_positional_embeddings):
        image_embeddings = image_embeddings.flatten(2).transpose(1, 2).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(2).transpose(1, 2).unsqueeze(1)
        queries, keys = point_embeddings, image_embeddings
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embeddings, image_positional_embeddings)
        query = queries + point_embeddings
        key = keys + image_positional_embeddings
        queries = self.layer_norm_final_attn(queries + self.final_attn_token_to_image(query, key, keys))
        return queries, keys


class Sam3TrackerMaskDecoder(nn.Module):
    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.num_mask_tokens = config.num_multimask_outputs + 1
        self.iou_token = nn.Embedding(1, hidden)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, hidden)
        self.transformer = Sam3TrackerTwoWayTransformer(config)
        self.upscale_conv1 = nn.ConvTranspose2d(hidden, hidden // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(hidden // 4, hidden // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = Sam3TrackerLayerNorm(hidden // 4, eps=1e-6)
        self.activation = nn.GELU()
        self.output_hypernetworks_mlps = nn.ModuleList(
            [Sam3TrackerFeedForward(hidden, hidden, hidden // 8, 3) for _ in range(self.num_mask_tokens)]
        )
        self.iou_prediction_head = Sam3TrackerFeedForward(
            hidden, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth, sigmoid_output=True
        )
        self.conv_s0 = nn.Conv2d(hidden, hidden // 8, kernel_size=1, stride=1)
        self.conv_s1 = nn.Conv2d(hidden, hidden // 4, kernel_size=1, stride=1)
        self.obj_score_token = nn.Embedding(1, hidden)
        self.pred_obj_score_head = Sam3TrackerFeedForward(hidden, hidden, 1, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_resolution_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multimask outputs: (masks [b, p, 3, h, w], iou [b, p, 3], mask tokens [b, p, 3, c], object logits)."""
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        output_tokens = torch.cat([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight])
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2).to(self.iou_token.weight.dtype)

        image_embeddings = (image_embeddings + dense_prompt_embeddings).repeat_interleave(point_batch_size, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)
        tokens, image_embeddings = self.transformer(tokens, image_embeddings, image_positional_embeddings)
        iou_token_out = tokens[:, :, 1, :]
        mask_tokens_out = tokens[:, :, 2 : 2 + self.num_mask_tokens, :]

        image_embeddings = image_embeddings.transpose(2, 3).view(
            batch_size * point_batch_size, num_channels, height, width
        )
        feat_s0, feat_s1 = (feature.repeat_interleave(point_batch_size, dim=0) for feature in high_resolution_features)
        upscaled = self.upscale_conv1(image_embeddings) + feat_s1
        upscaled = self.activation(self.upscale_layer_norm(upscaled))
        upscaled = self.activation(self.upscale_conv2(upscaled) + feat_s0)

        hyper_in = torch.stack(
            [mlp(mask_tokens_out[:, :, index, :]) for index, mlp in enumerate(self.output_hypernetworks_mlps)], dim=2
        )
        _, num_channels, height, width = upscaled.shape
        upscaled = upscaled.view(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled).view(batch_size, point_batch_size, -1, height, width)
        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(tokens[:, :, 0, :])
        # The video path always asks for the three multimask outputs.
        return masks[:, :, 1:], iou_pred[:, :, 1:], mask_tokens_out[:, :, 1:], object_score_logits


class Sam3TrackerPositionalEmbedding(nn.Module):
    """Random-Fourier positional embedding of normalized point coordinates."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.register_buffer("positional_embedding", torch.randn((2, hidden_size // 2)))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        coordinates = (2 * coordinates - 1).to(self.positional_embedding.dtype)
        coordinates = 2 * math.pi * (coordinates @ self.positional_embedding)
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)


class Sam3TrackerMaskEmbedding(nn.Module):
    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        channels = config.mask_input_channels // 4
        self.activation = _ACT[config.prompt_hidden_act]()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_dim, kernel_size=1)
        self.layer_norm1 = Sam3TrackerLayerNorm(channels, eps=config.prompt_layer_norm_eps)
        self.layer_norm2 = Sam3TrackerLayerNorm(channels * 4, eps=config.prompt_layer_norm_eps)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.layer_norm1(self.conv1(masks)))
        hidden_states = self.activation(self.layer_norm2(self.conv2(hidden_states)))
        return self.conv3(hidden_states)


class Sam3TrackerPromptEncoder(nn.Module):
    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        self.shared_embedding = Sam3TrackerPositionalEmbedding(config.hidden_dim)
        self.mask_embed = Sam3TrackerMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_dim)
        self.point_embed = nn.Embedding(config.num_point_embeddings, config.hidden_dim)
        self.not_a_point_embed = nn.Embedding(1, config.hidden_dim)
        self.image_embedding_size = config.image_embedding_size
        self.mask_input_size = config.mask_input_size
        self.input_image_size = config.image_size

    def forward(self, input_masks: torch.Tensor | None, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed the padding point the tracker uses in place of clicks, plus an optional dense mask prompt."""
        device = self.not_a_point_embed.weight.device
        # One empty point (label -1) and its pad (label -1): both become ``not_a_point_embed``.
        sparse = self.not_a_point_embed.weight.reshape(1, 1, 1, -1).expand(batch_size, 1, 2, -1).to(device)
        if input_masks is not None:
            dense = self.mask_embed(input_masks)
        else:
            dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(batch_size, -1, *self.image_embedding_size)
        return sparse, dense


class Sam3TrackerVisionRotaryEmbedding(nn.Module):
    """Axial 2D RoPE over the top-level 72x72 feature grid."""

    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        dim = config.memory_attention_hidden_size // (
            config.memory_attention_downsample_rate * config.memory_attention_num_attention_heads
        )
        spatial_dim = dim // 2
        inv_freq = 1.0 / (
            config.memory_attention_rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = position_ids[..., None].float() * self.inv_freq.float()
            cos, sin = freqs.cos(), freqs.sin()

        def recompose(freq: torch.Tensor) -> torch.Tensor:
            return torch.cat([freq[:, 0], freq[:, 1]], dim=-1)[None, ...].repeat_interleave(2, dim=-1)

        return recompose(cos), recompose(sin)


def rotate_pairwise(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.view(*x.shape[:-1], -1, 2).unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rotary_pos_emb_2d(q, k, cos, sin, num_k_exclude_rope: int = 0, repeat_freqs_k: bool = False):
    k_rot, k_pass = k[..., : k.shape[-2] - num_k_exclude_rope, :], k[..., k.shape[-2] - num_k_exclude_rope :, :]
    q_embed = q.float()
    q_embed = (q_embed * cos) + (rotate_pairwise(q_embed) * sin)
    if k_rot.shape[-2] == 0:
        return q_embed.type_as(q), torch.cat([k_rot, k_pass], dim=-2)
    if repeat_freqs_k and k_rot.shape[-2] != q.shape[-2]:
        repeat_factor = k_rot.shape[-2] // q.shape[-2]
        cos_k, sin_k = cos.repeat(1, 1, repeat_factor, 1), sin.repeat(1, 1, repeat_factor, 1)
    else:
        cos_k, sin_k = cos, sin
    k_embed = k_rot.float()
    k_embed = (k_embed * cos_k) + (rotate_pairwise(k_embed) * sin_k)
    return q_embed.type_as(q), torch.cat([k_embed.type_as(k), k_pass], dim=-2)


class Sam3TrackerRoPEAttention(nn.Module):
    def __init__(self, config: Sam3TrackerConfig, kv_in_dim: int | None = None, rope_k_repeat: bool = False):
        super().__init__()
        hidden = config.memory_attention_hidden_size
        self.internal_dim = hidden // config.memory_attention_downsample_rate
        self.num_attention_heads = config.memory_attention_num_attention_heads
        self.head_dim = self.internal_dim // self.num_attention_heads
        self.scaling = self.head_dim**-0.5
        kv_in_dim = kv_in_dim if kv_in_dim is not None else hidden
        self.q_proj = nn.Linear(hidden, self.internal_dim)
        self.k_proj = nn.Linear(kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(kv_in_dim, self.internal_dim)
        self.o_proj = nn.Linear(self.internal_dim, hidden)
        self.rope_k_repeat = rope_k_repeat

    def forward(self, query, key, value, position_embeddings, num_k_exclude_rope: int = 0) -> torch.Tensor:
        batch, points = query.shape[:2]
        shape = (batch * points, -1, self.num_attention_heads, self.head_dim)
        query = self.q_proj(query).view(*shape).transpose(1, 2)
        key = self.k_proj(key).view(*shape).transpose(1, 2)
        value = self.v_proj(value).view(*shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_2d(
            query, key, cos, sin, repeat_freqs_k=self.rope_k_repeat, num_k_exclude_rope=num_k_exclude_rope
        )
        out = F.scaled_dot_product_attention(query, key, value, scale=self.scaling)
        out = out.transpose(1, 2).reshape(batch, points, -1, self.num_attention_heads * self.head_dim)
        return self.o_proj(out)


class Sam3TrackerMemoryAttentionLayer(nn.Module):
    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        hidden = config.memory_attention_hidden_size
        self.self_attn = Sam3TrackerRoPEAttention(config)
        self.cross_attn_image = Sam3TrackerRoPEAttention(
            config, kv_in_dim=config.memory_encoder_output_channels, rope_k_repeat=True
        )
        self.linear1 = nn.Linear(hidden, config.memory_attention_feed_forward_hidden_size)
        self.linear2 = nn.Linear(config.memory_attention_feed_forward_hidden_size, hidden)
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.layer_norm3 = nn.LayerNorm(hidden)
        self.activation = _ACT[config.memory_attention_feed_forward_hidden_act]()

    def forward(self, queries, keys, key_point_embedding, rope_position_embeddings, num_k_exclude_rope: int = 0):
        query = self.layer_norm1(queries)
        queries = queries + self.self_attn(query, query, query, rope_position_embeddings)
        query = self.layer_norm2(queries)
        queries = queries + self.cross_attn_image(
            query, keys + key_point_embedding, keys, rope_position_embeddings, num_k_exclude_rope=num_k_exclude_rope
        )
        query = self.layer_norm3(queries)
        return queries + self.linear2(self.activation(self.linear1(query)))


class Sam3TrackerMemoryAttention(nn.Module):
    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Sam3TrackerMemoryAttentionLayer(config) for _ in range(config.memory_attention_num_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.memory_attention_hidden_size)
        self.rotary_emb = Sam3TrackerVisionRotaryEmbedding(config)
        width, height = config.memory_attention_rope_feat_sizes
        hpos, wpos = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        self.register_buffer("position_ids", torch.stack([wpos.flatten(), hpos.flatten()], dim=-1), persistent=False)

    def forward(
        self,
        current_vision_features: torch.Tensor,
        memory: torch.Tensor,
        current_vision_position_embeddings: torch.Tensor,
        memory_position_embeddings: torch.Tensor,
        num_object_pointer_tokens: int,
    ) -> torch.Tensor:
        output = (current_vision_features + 0.1 * current_vision_position_embeddings).transpose(0, 1)
        memory = memory.transpose(0, 1).unsqueeze(1)
        memory_position_embeddings = memory_position_embeddings.transpose(0, 1).unsqueeze(1)
        rope = self.rotary_emb(self.position_ids)
        for layer in self.layers:
            output = layer(
                output.unsqueeze(1) if output.ndim == 3 else output,
                memory,
                memory_position_embeddings,
                rope,
                num_k_exclude_rope=num_object_pointer_tokens,
            )
        return self.layer_norm(output).transpose(0, 1)


class Sam3TrackerMemoryFuserCXBlock(nn.Module):
    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        dim = config.memory_fuser_embed_dim
        self.depthwise_conv = nn.Conv2d(
            dim, dim, kernel_size=config.memory_fuser_kernel_size, padding=config.memory_fuser_padding, groups=dim
        )
        self.layer_norm = Sam3TrackerLayerNorm(dim, eps=1e-6)
        self.activation = _ACT[config.memory_fuser_hidden_act]()
        self.pointwise_conv1 = nn.Linear(dim, config.memory_fuser_intermediate_dim)
        self.pointwise_conv2 = nn.Linear(config.memory_fuser_intermediate_dim, dim)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm(self.depthwise_conv(hidden_states)).permute(0, 2, 3, 1)
        hidden_states = self.pointwise_conv2(self.activation(self.pointwise_conv1(hidden_states)))
        return residual + (self.scale * hidden_states).permute(0, 3, 1, 2)


class Sam3TrackerMemoryFuser(nn.Module):
    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [Sam3TrackerMemoryFuserCXBlock(config) for _ in range(config.memory_fuser_num_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Sam3TrackerMaskDownSamplerLayer(nn.Module):
    def __init__(self, config: Sam3TrackerConfig, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=config.mask_downsampler_kernel_size,
            stride=config.mask_downsampler_stride,
            padding=config.mask_downsampler_padding,
        )
        self.layer_norm = Sam3TrackerLayerNorm(out_channels, eps=1e-6)
        self.activation = _ACT[config.mask_downsampler_hidden_act]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layer_norm(self.conv(x)))


class Sam3TrackerMaskDownSampler(nn.Module):
    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        num_layers = int(math.log2(config.mask_downsampler_total_stride) // math.log2(config.mask_downsampler_stride))
        self.layers = nn.ModuleList()
        in_channels = out_channels = 1
        for _ in range(num_layers):
            out_channels = in_channels * config.mask_downsampler_stride**2
            self.layers.append(Sam3TrackerMaskDownSamplerLayer(config, in_channels, out_channels))
            in_channels = out_channels
        self.final_conv = nn.Conv2d(out_channels, config.mask_downsampler_embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


class Sam3TrackerMemoryEncoder(nn.Module):
    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        hidden = config.memory_encoder_hidden_size
        self.mask_downsampler = Sam3TrackerMaskDownSampler(config)
        self.feature_projection = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.memory_fuser = Sam3TrackerMemoryFuser(config)
        self.projection = nn.Conv2d(hidden, config.memory_encoder_output_channels, kernel_size=1)
        self.num_position_features = config.memory_encoder_output_channels // 2

    def forward(self, vision_features: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_omni.diffusion.models.sam3.sam3_model import _sine_position_embedding

        features = self.feature_projection(vision_features) + self.mask_downsampler(masks)
        features = self.projection(self.memory_fuser(features))
        position = _sine_position_embedding(
            torch.Size(features.shape), features.device, features.dtype, self.num_position_features, True, 2 * math.pi
        )
        return features, position


def get_1d_sine_pe(positions: torch.Tensor, dim: int, temperature: float = 10000) -> torch.Tensor:
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=positions.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = positions.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Per-object tracking state
# ---------------------------------------------------------------------------


@dataclass
class Sam3TrackerObjectState:
    """Memory bank of one tracked object: per-frame outputs of conditioning and tracked frames."""

    cond_frame_outputs: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    non_cond_frame_outputs: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    frames_tracked: set[int] = field(default_factory=set)
    # Detector masks that seed the object on a frame (consumed once).
    mask_inputs: dict[int, torch.Tensor] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class Sam3Tracker(nn.Module):
    """SAM2-style memory tracker; vision features come from the shared SAM3 backbone."""

    def __init__(self, config: Sam3TrackerConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.hidden_dim = hidden
        self.mem_dim = config.memory_encoder_output_channels
        self.num_maskmem = config.num_maskmem
        self.image_size = config.image_size
        self.backbone_feature_sizes = [tuple(size) for size in config.backbone_feature_sizes]
        self.shared_image_embedding = Sam3TrackerPositionalEmbedding(hidden)
        self.prompt_encoder = Sam3TrackerPromptEncoder(config)
        self.mask_decoder = Sam3TrackerMaskDecoder(config)
        self.no_memory_embedding = nn.Parameter(torch.zeros(1, 1, hidden))
        self.memory_attention = Sam3TrackerMemoryAttention(config)
        self.memory_encoder = Sam3TrackerMemoryEncoder(config)
        self.no_memory_positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden))
        self.memory_temporal_positional_encoding = nn.Parameter(torch.zeros(self.num_maskmem, 1, 1, self.mem_dim))
        self.no_object_pointer = nn.Parameter(torch.zeros(1, hidden))
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.object_pointer_proj = Sam3TrackerFeedForward(hidden, hidden, hidden, 3)
        self.temporal_positional_encoding_projection_layer = (
            nn.Linear(hidden, self.mem_dim) if config.enable_temporal_pos_encoding_for_object_pointers else None
        )
        self.occlusion_spatial_embedding_parameter = (
            nn.Parameter(torch.zeros(1, self.mem_dim)) if config.enable_occlusion_spatial_embedding else None
        )

    # -- SAM heads ----------------------------------------------------------

    def _image_positional_embeddings(self) -> torch.Tensor:
        size = self.config.image_embedding_size
        embedding = self.shared_image_embedding.positional_embedding
        grid = torch.ones(size, device=embedding.device, dtype=embedding.dtype)
        y_embed = (grid.cumsum(dim=0) - 0.5) / size[0]
        x_embed = (grid.cumsum(dim=1) - 0.5) / size[1]
        return self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1)).permute(2, 0, 1).unsqueeze(0)

    def _sam_heads(self, image_embeddings: list[torch.Tensor], input_masks: torch.Tensor | None) -> dict:
        """Run the SAM prompt encoder + mask decoder; keep the best of the three multimask outputs."""
        batch_size = image_embeddings[-1].shape[0]
        positional = self._image_positional_embeddings().repeat(batch_size, 1, 1, 1)
        if input_masks is not None and input_masks.shape[-2:] != self.prompt_encoder.mask_input_size:
            input_masks = F.interpolate(
                input_masks.float(),
                size=self.prompt_encoder.mask_input_size,
                align_corners=False,
                mode="bilinear",
                antialias=True,
            ).to(input_masks.dtype)
        sparse, dense = self.prompt_encoder(input_masks, batch_size)
        multimasks, iou_scores, mask_tokens, object_score_logits = self.mask_decoder(
            image_embeddings[-1], positional, sparse, dense, image_embeddings[:-1]
        )
        is_obj_appearing = object_score_logits > 0
        multimasks = torch.where(is_obj_appearing[:, None, None], multimasks, NO_OBJ_SCORE)
        high_res_multimasks = (
            F.interpolate(
                multimasks.squeeze(1).float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            .unsqueeze(1)
            .to(multimasks.dtype)
        )
        best = torch.argmax(iou_scores, dim=-1)
        batch_inds = torch.arange(batch_size, device=multimasks.device)
        object_inds = torch.arange(1, device=multimasks.device)
        low_res_masks = multimasks[batch_inds, object_inds, best]
        high_res_masks = high_res_multimasks[batch_inds, object_inds, best]
        object_pointer = self.object_pointer_proj(mask_tokens[batch_inds, object_inds, best])
        appearing = is_obj_appearing.to(object_pointer.dtype)
        object_pointer = appearing * object_pointer + (1 - appearing) * self.no_object_pointer
        return {
            "pred_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_pointer": object_pointer,
            "object_score_logits": object_score_logits,
        }

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs: torch.Tensor) -> dict:
        """Treat a detector mask as ground truth for its frame (no SAM decoding of the mask itself)."""
        out_scale, out_bias = 20.0, -10.0
        dtype = backbone_features[0].dtype
        mask_float = mask_inputs.to(dtype)
        if mask_float.shape[-2:] != (self.image_size, self.image_size):
            mask_float = F.interpolate(
                mask_float.float(),
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            ).to(mask_inputs.dtype)
        high_res_masks = mask_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks.float(),
            size=self.prompt_encoder.mask_input_size,
            align_corners=False,
            mode="bilinear",
            antialias=True,
        ).to(dtype)
        object_pointer = self._sam_heads(
            high_res_features + [backbone_features], self.mask_downsample(mask_float.to(dtype))
        )["object_pointer"]
        appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)[..., None].to(dtype)
        object_score_logits = out_scale * appearing + out_bias
        object_pointer = appearing * object_pointer + (1 - appearing) * self.no_object_pointer
        return {
            "pred_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_pointer": object_pointer,
            "object_score_logits": object_score_logits.unsqueeze(-1),
        }

    # -- memory ---------------------------------------------------------------

    @staticmethod
    def _select_closest_cond_frames(frame_idx: int, cond_outputs: dict, max_num: int) -> tuple[dict, dict]:
        if max_num == -1 or len(cond_outputs) <= max_num:
            return cond_outputs, {}
        selected = {}
        before = max((t for t in cond_outputs if t < frame_idx), default=None)
        if before is not None:
            selected[before] = cond_outputs[before]
        after = min((t for t in cond_outputs if t >= frame_idx), default=None)
        if after is not None:
            selected[after] = cond_outputs[after]
        remaining = sorted((t for t in cond_outputs if t not in selected), key=lambda t: abs(t - frame_idx))
        selected.update((t, cond_outputs[t]) for t in remaining[: max_num - len(selected)])
        return selected, {t: v for t, v in cond_outputs.items() if t not in selected}

    def _memory_conditioned_features(
        self,
        state: Sam3TrackerObjectState,
        frame_idx: int,
        is_init_cond_frame: bool,
        vision_features: torch.Tensor,
        vision_pos_embeds: torch.Tensor,
        num_frames: int,
        reverse: bool,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = vision_features.size(1)
        height, width = self.backbone_feature_sizes[-1]
        if is_init_cond_frame:
            features = vision_features + self.no_memory_embedding
            return features.permute(1, 2, 0).view(batch_size, self.hidden_dim, height, width)

        if not state.cond_frame_outputs:
            raise ValueError("SAM3 tracker has no conditioning frame for this object")
        cond_outputs, unselected = self._select_closest_cond_frames(
            frame_idx, state.cond_frame_outputs, self.config.max_cond_frame_num
        )
        memory_frames = [(0, out) for out in cond_outputs.values()]
        for offset in range(self.num_maskmem - 1, 0, -1):
            previous = frame_idx + offset if reverse else frame_idx - offset
            memory_frames.append((offset, state.non_cond_frame_outputs.get(previous, unselected.get(previous, None))))
        memories, memory_pos = [], []
        for offset, output in memory_frames:
            if output is None:
                continue
            memories.append(output["maskmem_features"])
            memory_pos.append(output["maskmem_pos_enc"] + self.memory_temporal_positional_encoding[offset - 1])

        # Object pointers from conditioning frames up to now, then recent tracked frames.
        max_pointers = min(num_frames, self.config.max_object_pointers_in_encoder)
        sign = -1 if reverse else 1
        offsets, pointers = [], []
        for t, output in state.cond_frame_outputs.items():
            if t >= frame_idx if reverse else t <= frame_idx:
                offsets.append((frame_idx - t) * sign)
                pointers.append(output["object_pointer"])
        for offset in range(1, max_pointers):
            ref = frame_idx + offset if reverse else frame_idx - offset
            if ref < 0 or ref >= num_frames:
                break
            output = state.non_cond_frame_outputs.get(ref)
            if output is not None:
                offsets.append(offset)
                pointers.append(output["object_pointer"])

        num_pointer_tokens = 0
        if pointers:
            object_pointers = torch.stack(pointers, dim=0)
            if self.temporal_positional_encoding_projection_layer is not None:
                diffs = torch.tensor(offsets, device=vision_features.device, dtype=torch.float32) / float(
                    max_pointers - 1
                )
                sine = get_1d_sine_pe(diffs, dim=self.hidden_dim).to(object_pointers.dtype)
                pointer_pos = self.temporal_positional_encoding_projection_layer(sine).unsqueeze(1)
                pointer_pos = pointer_pos.expand(-1, batch_size, self.mem_dim)
            else:
                pointer_pos = object_pointers.new_zeros(len(offsets), batch_size, self.mem_dim)
            if self.mem_dim < self.hidden_dim:
                splits = self.hidden_dim // self.mem_dim
                object_pointers = object_pointers.reshape(-1, batch_size, splits, self.mem_dim)
                object_pointers = object_pointers.permute(0, 2, 1, 3).flatten(0, 1)
                pointer_pos = pointer_pos.repeat_interleave(splits, dim=0)
            memories.append(object_pointers)
            memory_pos.append(pointer_pos)
            num_pointer_tokens = object_pointers.shape[0]

        conditioned = self.memory_attention(
            current_vision_features=vision_features,
            memory=torch.cat(memories, dim=0).to(dtype=dtype),
            current_vision_position_embeddings=vision_pos_embeds,
            memory_position_embeddings=torch.cat(memory_pos, dim=0),
            num_object_pointer_tokens=num_pointer_tokens,
        )
        return conditioned.squeeze(1).transpose(1, 2).view(batch_size, self.hidden_dim, height, width)

    def encode_memory(
        self,
        vision_features: torch.Tensor,
        high_res_masks: torch.Tensor,
        object_score_logits: torch.Tensor,
        is_mask_from_pts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode [N, 1, H, W] mask logits on top of the frame features into N memory entries."""
        batch_size = vision_features.size(1)
        height, width = self.backbone_feature_sizes[-1]
        mask_h, mask_w = self.prompt_encoder.mask_input_size
        if high_res_masks.shape[2:] != (mask_h * 4, mask_w * 4):
            high_res_masks = F.interpolate(
                high_res_masks.float(),
                size=(mask_h * 4, mask_w * 4),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            ).to(high_res_masks.dtype)
        pix_feat = vision_features.permute(1, 2, 0).view(batch_size, self.hidden_dim, height, width)
        if is_mask_from_pts:
            mask_for_mem = (high_res_masks > 0).to(high_res_masks.dtype)
        else:
            mask_for_mem = torch.sigmoid(high_res_masks)
        mask_for_mem = mask_for_mem * self.config.sigmoid_scale_for_mem_enc + self.config.sigmoid_bias_for_mem_enc
        features, position = self.memory_encoder(pix_feat, mask_for_mem)
        if self.occlusion_spatial_embedding_parameter is not None:
            appearing = (object_score_logits > 0).float()
            features += (1 - appearing[..., None]) * self.occlusion_spatial_embedding_parameter[..., None, None].expand(
                *features.shape
            )
        # The memory bank keeps bfloat16 features, as the reference does.
        features = features.to(torch.bfloat16).flatten(2).permute(2, 0, 1)
        position = position.to(high_res_masks.dtype).flatten(2).permute(2, 0, 1)
        return features, position

    # -- one frame ------------------------------------------------------------

    def track_frame(
        self,
        objects: list[Sam3TrackerObjectState],
        new_input_objects: set[int],
        frame_idx: int,
        vision_feats: list[torch.Tensor],
        vision_pos_embeds: list[torch.Tensor],
        num_frames: int,
        reverse: bool,
        dtype: torch.dtype,
        run_mem_encoder: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate every object onto ``frame_idx``; returns low-res mask logits and object-score logits.

        ``vision_feats`` / ``vision_pos_embeds`` are the tracker FPN levels flattened to [HW, 1, C].
        Objects listed in ``new_input_objects`` consume their mask input as a conditioning frame.
        """
        pred_masks, score_logits = [], []
        encode_objects, encode_masks, encode_scores = [], [], []
        any_mask_input = False
        high_res_features = [
            feature.permute(1, 2, 0).view(feature.size(1), feature.size(2), *size)
            for feature, size in zip(vision_feats[:-1], self.backbone_feature_sizes[:-1])
        ]
        for index, state in enumerate(objects):
            has_new_inputs = index in new_input_objects
            if not has_new_inputs and frame_idx in state.cond_frame_outputs:
                out = state.cond_frame_outputs[frame_idx]
                is_init_cond_frame = True
                mask_inputs = None
            else:
                is_init_cond_frame = False
                mask_inputs = None
                if has_new_inputs:
                    is_init_cond_frame = frame_idx not in state.frames_tracked
                    if is_init_cond_frame:
                        # As in the reference, this also switches the remaining objects of the frame to forward time.
                        reverse = False
                    mask_inputs = state.mask_inputs.get(frame_idx)
                    if mask_inputs is not None:
                        new_input_objects.discard(index)
                        any_mask_input = True
                if mask_inputs is not None:
                    pix_feat = (
                        vision_feats[-1].permute(1, 2, 0).view(-1, self.hidden_dim, *self.backbone_feature_sizes[-1])
                    )
                    out = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
                else:
                    pix_feat = self._memory_conditioned_features(
                        state,
                        frame_idx,
                        is_init_cond_frame,
                        vision_feats[-1],
                        vision_pos_embeds[-1],
                        num_frames,
                        reverse,
                        dtype,
                    )
                    out = self._sam_heads(high_res_features + [pix_feat], None)
                target = state.cond_frame_outputs if is_init_cond_frame else state.non_cond_frame_outputs
                target[frame_idx] = dict(out)
                if run_mem_encoder and self.num_maskmem > 0:
                    encode_objects.append(index)
                    encode_masks.append(out["high_res_masks"])
                    encode_scores.append(out["object_score_logits"])
            pred_masks.append(out["pred_masks"])
            score_logits.append(out["object_score_logits"].squeeze(-1))
            if not is_init_cond_frame:
                state.frames_tracked.add(frame_idx)

        if encode_objects:
            features, positions = self.encode_memory(
                vision_feats[-1].expand(-1, len(encode_objects), -1),
                torch.cat(encode_masks, dim=0),
                torch.cat(encode_scores, dim=0),
                is_mask_from_pts=any_mask_input,
            )
            for column, index in enumerate(encode_objects):
                state = objects[index]
                target = (
                    state.cond_frame_outputs if frame_idx in state.cond_frame_outputs else state.non_cond_frame_outputs
                )
                if frame_idx in target:
                    target[frame_idx]["maskmem_features"] = features[:, column : column + 1]
                    target[frame_idx]["maskmem_pos_enc"] = positions[:, column : column + 1]
        return torch.cat(pred_masks, dim=0), torch.cat(score_logits, dim=0)

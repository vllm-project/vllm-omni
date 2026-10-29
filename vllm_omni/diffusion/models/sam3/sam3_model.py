# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SAM3 promptable concept segmentation detector (image path).

Ported from the Hugging Face Transformers SAM3 implementation
(``transformers/models/sam3/modeling_sam3.py``, Apache-2.0) and kept
numerically equivalent to it. Module and parameter names match the released
checkpoint so its detector weights load strictly. The tracker that the video
checkpoint also carries is out of scope here.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import regex as re
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from transformers import CLIPTextConfig, CLIPTextModelWithProjection
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _fields(cls, values: Mapping[str, Any]) -> dict[str, Any]:
    names = {item.name for item in cls.__dataclass_fields__.values()}
    return {key: value for key, value in values.items() if key in names}


@dataclass
class Sam3ViTConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4736
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 1008
    patch_size: int = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    window_size: int = 24
    global_attn_indexes: list[int] = field(default_factory=lambda: [7, 15, 23, 31])
    pretrain_image_size: int = 336
    hidden_dropout: float = 0.0
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Sam3ViTConfig:
        config = cls(**_fields(cls, values))
        rope = values.get("rope_parameters") or {}
        if rope.get("rope_type", "axial") != "axial":
            raise ValueError(f"SAM3 ViT supports only axial RoPE, got {rope.get('rope_type')!r}")
        config.rope_theta = float(rope.get("rope_theta", values.get("rope_theta", config.rope_theta)))
        if values.get("global_attn_indexes") is None:
            config.global_attn_indexes = [7, 15, 23, 31]
        return config


@dataclass
class Sam3VisionConfig:
    backbone_config: Sam3ViTConfig = field(default_factory=Sam3ViTConfig)
    fpn_hidden_size: int = 256
    scale_factors: list[float] = field(default_factory=lambda: [4.0, 2.0, 1.0, 0.5])

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Sam3VisionConfig:
        return cls(
            backbone_config=Sam3ViTConfig.from_dict(values.get("backbone_config") or {}),
            fpn_hidden_size=values.get("fpn_hidden_size", 256),
            scale_factors=values.get("scale_factors") or [4.0, 2.0, 1.0, 0.5],
        )


@dataclass
class Sam3BlockConfig:
    """Shared fields of the geometry encoder, DETR encoder/decoder and mask decoder."""

    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float = 0.0
    roi_size: int = 7
    num_queries: int = 200
    num_upsampling_stages: int = 3

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], **defaults: Any) -> Sam3BlockConfig:
        return cls(**{**defaults, **_fields(cls, values)})


@dataclass
class Sam3DetectorConfig:
    vision_config: Sam3VisionConfig
    text_config: CLIPTextConfig
    geometry_encoder_config: Sam3BlockConfig
    detr_encoder_config: Sam3BlockConfig
    detr_decoder_config: Sam3BlockConfig
    mask_decoder_config: Sam3BlockConfig

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Sam3DetectorConfig:
        """Accept a ``sam3`` config or a ``sam3_video`` config nesting it under ``detector_config``."""
        values = values.get("detector_config") or values
        text = dict(values.get("text_config") or {})
        text.pop("model_type", None)
        # SAM3's text-tower defaults apply only when the config omits text_config
        # entirely; a given text_config otherwise falls back to CLIP's own defaults.
        sam3_text_defaults = {
            "vocab_size": 49408,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "projection_dim": 512,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "max_position_embeddings": 32,
            "hidden_act": "gelu",
        }
        return cls(
            vision_config=Sam3VisionConfig.from_dict(values.get("vision_config") or {}),
            text_config=CLIPTextConfig(**(text or sam3_text_defaults)),
            geometry_encoder_config=Sam3BlockConfig.from_dict(
                values.get("geometry_encoder_config") or {}, num_layers=3
            ),
            detr_encoder_config=Sam3BlockConfig.from_dict(values.get("detr_encoder_config") or {}),
            detr_decoder_config=Sam3BlockConfig.from_dict(values.get("detr_decoder_config") or {}),
            mask_decoder_config=Sam3BlockConfig.from_dict(values.get("mask_decoder_config") or {}, dropout=0.0),
        )


@dataclass
class Sam3VisionOutput:
    fpn_hidden_states: tuple[torch.Tensor, ...]
    fpn_position_encoding: tuple[torch.Tensor, ...]


@dataclass
class Sam3DetectorOutput:
    pred_masks: torch.Tensor
    pred_boxes: torch.Tensor
    pred_logits: torch.Tensor
    presence_logits: torch.Tensor
    semantic_seg: torch.Tensor


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


def concat_padded_sequences(seq1, mask1, seq2, mask2):
    """Concatenate two right-padded sequences, packing the valid tokens of ``seq2`` after those of ``seq1``."""
    batch_size, seq1_length, hidden_size = seq1.shape
    seq2_length = seq2.shape[1]
    seq1_lengths = mask1.sum(dim=-1)
    final_lengths = seq1_lengths + mask2.sum(dim=-1)
    max_length = seq1_length + seq2_length
    mask = torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1) < final_lengths[:, None]
    sequence = torch.zeros((batch_size, max_length, hidden_size), device=seq2.device, dtype=seq2.dtype)
    sequence[:, :seq1_length, :] = seq1
    index = torch.arange(seq2_length, device=seq2.device)[None].repeat(batch_size, 1) + seq1_lengths[:, None]
    sequence = sequence.scatter(1, index[:, :, None].expand(-1, -1, hidden_size), seq2)
    return sequence, mask


def key_padding_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """``[batch, kv]`` validity to an SDPA boolean mask (``True`` attends), or ``None`` without padding."""
    if mask is None or bool(mask.all()):
        return None
    return mask.bool()[:, None, None, :]


class Sam3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, dropout: float):
        super().__init__()
        self.activation_fn = _ACTIVATIONS[hidden_act]()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.dropout(self.fc1(hidden_states))))


class Sam3Attention(nn.Module):
    """Multi-head attention with separate query/key/value inputs and an optional mask or additive bias."""

    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, query_len = query.shape[:2]
        key_len = key.shape[1]
        heads, head_dim = self.num_attention_heads, self.head_dim
        query = self.q_proj(query).view(batch_size, query_len, heads, head_dim).transpose(1, 2)
        key = self.k_proj(key).view(batch_size, key_len, heads, head_dim).transpose(1, 2)
        value = self.v_proj(value).view(batch_size, key_len, heads, head_dim).transpose(1, 2)
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(query.dtype)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=self.scaling)
        output = output.transpose(1, 2).reshape(batch_size, query_len, heads * head_dim)
        return self.o_proj(output)


# ---------------------------------------------------------------------------
# Vision encoder: windowed ViT with 2D axial RoPE and a simple FPN neck
# ---------------------------------------------------------------------------


class Sam3ViTRotaryEmbedding(nn.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        spatial_dim = (config.hidden_size // config.num_attention_heads) // 2
        inv_freq = 1.0 / config.rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = position_ids[..., None].float() * self.inv_freq.float()
            cos, sin = freqs.cos(), freqs.sin()
        return self._recompose(cos), self._recompose(sin)

    @staticmethod
    def _recompose(freq: torch.Tensor) -> torch.Tensor:
        freq_hw = torch.cat([freq[:, 0], freq[:, 1]], dim=-1)[None, ...]
        return freq_hw.repeat_interleave(2, dim=-1)


def rotate_pairwise(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.view(*x.shape[:-1], -1, 2).unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rotary_pos_emb_2d(q, k, cos, sin):
    q_embed, k_embed = q.float(), k.float()
    q_embed = q_embed * cos + rotate_pairwise(q_embed) * sin
    k_embed = k_embed * cos + rotate_pairwise(k_embed) * sin
    return q_embed.type_as(q), k_embed.type_as(k)


def _qkv_shard_loader(param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str | None = None) -> None:
    """Load a fused ``[q; k; v]`` tensor whole, or one of its ``q``/``k``/``v`` thirds."""
    if shard_id is None:
        param.data.copy_(loaded_weight)
        return
    size = param.shape[0] // 3
    offset = {"q": 0, "k": size, "v": 2 * size}[shard_id]
    param.data[offset : offset + size].copy_(loaded_weight)


class Sam3ViTRoPEAttention(nn.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        # Fused projection, as in the released checkpoint; split checkpoints load per shard.
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        for param in self.qkv.parameters():
            param.weight_loader = _qkv_shard_loader  # type: ignore[attr-defined]
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, position_embeddings) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        shape = (batch_size, height * width, self.num_attention_heads, self.head_dim)
        query, key, value = (part.reshape(*shape).transpose(1, 2) for part in self.qkv(hidden_states).chunk(3, dim=-1))
        query, key = apply_rotary_pos_emb_2d(query, key, *position_embeddings)
        output = F.scaled_dot_product_attention(query, key, value, scale=self.scaling)
        output = output.transpose(1, 2).reshape(batch_size, height, width, -1)
        return self.o_proj(output)


class Sam3ViTPatchEmbeddings(nn.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        patch = config.patch_size
        self.num_patches = (config.pretrain_image_size // patch) ** 2
        self.projection = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=patch, stride=patch, bias=False
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection(pixel_values.to(self.projection.weight.dtype)).flatten(2).transpose(1, 2)


class Sam3ViTEmbeddings(nn.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.patch_embeddings = Sam3ViTPatchEmbeddings(config)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.patch_embeddings.num_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.patch_size = config.patch_size

    def _tile_position_embeddings(self, height: int, width: int) -> torch.Tensor:
        """Tile the pretraining-resolution table over the patch grid (no interpolation)."""
        position_embeddings = self.position_embeddings
        pretrain_size = int(position_embeddings.shape[1] ** 0.5)
        hidden_size = position_embeddings.shape[-1]
        if pretrain_size == height and pretrain_size == width:
            return position_embeddings.reshape(1, height * width, -1)
        pos_embed = position_embeddings.reshape(1, pretrain_size, pretrain_size, hidden_size).permute(0, 3, 1, 2)
        pos_embed = pos_embed.tile([1, 1, height // pretrain_size + 1, width // pretrain_size + 1])[
            :, :, :height, :width
        ]
        return pos_embed.permute(0, 2, 3, 1).reshape(1, height * width, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = embeddings + self._tile_position_embeddings(height // self.patch_size, width // self.patch_size)
        return self.dropout(embeddings)


def window_partition(hidden_state: torch.Tensor, window_size: int):
    batch_size, height, width, channels = hidden_state.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    hidden_state = F.pad(hidden_state, (0, 0, 0, pad_width, 0, pad_height))
    padded_height, padded_width = height + pad_height, width + pad_width
    hidden_state = hidden_state.view(
        batch_size, padded_height // window_size, window_size, padded_width // window_size, window_size, channels
    )
    windows = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)
    return windows, (padded_height, padded_width)


def window_unpartition(windows, window_size, padded_size, size):
    padded_height, padded_width = padded_size
    height, width = size
    batch_size = windows.shape[0] // (padded_height * padded_width // window_size // window_size)
    hidden_state = windows.view(
        batch_size, padded_height // window_size, padded_width // window_size, window_size, window_size, -1
    )
    hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, padded_height, padded_width, -1)
    return hidden_state[:, :height, :width, :].contiguous()


class Sam3ViTLayer(nn.Module):
    def __init__(self, config: Sam3ViTConfig, window_size: int = 0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = Sam3ViTRotaryEmbedding(config)
        self.attention = Sam3ViTRoPEAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Sam3MLP(config.hidden_size, config.intermediate_size, config.hidden_act, config.hidden_dropout)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.window_size = window_size
        self.register_buffer("position_ids", self._positions(config, window_size), persistent=False)

    @staticmethod
    def _positions(config: Sam3ViTConfig, window_size: int) -> torch.Tensor:
        if window_size > 0:
            grid = (window_size, window_size)
        else:
            grid = (config.image_size // config.patch_size, config.image_size // config.patch_size)
        hpos, wpos = torch.meshgrid(torch.arange(grid[0]), torch.arange(grid[1]), indexing="ij")
        positions = torch.stack([wpos.flatten(), hpos.flatten()], dim=-1)
        # Global layers are rescaled onto the window grid the RoPE was trained on.
        return positions * (config.window_size / grid[1])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if self.window_size > 0:
            size = (hidden_states.shape[1], hidden_states.shape[2])
            hidden_states, padded_size = window_partition(hidden_states, self.window_size)
        hidden_states = self.attention(hidden_states, self.rotary_emb(self.position_ids))
        if self.window_size > 0:
            hidden_states = window_unpartition(hidden_states, self.window_size, padded_size, size)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.layer_norm2(hidden_states))
        return residual + self.dropout(hidden_states)


class Sam3ViTModel(nn.Module):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.config = config
        self.embeddings = Sam3ViTEmbeddings(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            [
                Sam3ViTLayer(config, window_size=0 if index in config.global_attn_indexes else config.window_size)
                for index in range(config.num_hidden_layers)
            ]
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        batch_size, _, hidden_size = hidden_states.shape
        height = pixel_values.shape[-2] // self.config.patch_size
        width = pixel_values.shape[-1] // self.config.patch_size
        hidden_states = self.layer_norm(hidden_states.view(batch_size, height, width, hidden_size))
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states  # [batch, height, width, hidden]


@functools.lru_cache(maxsize=8)
def _sine_position_embedding(
    shape: torch.Size, device: torch.device, dtype: torch.dtype, features: int, normalize: bool, scale: float
) -> torch.Tensor:
    batch_size, _, height, width = shape
    y_embed = torch.arange(1, height + 1, dtype=dtype, device=device)[None, :, None].expand(batch_size, height, width)
    x_embed = torch.arange(1, width + 1, dtype=dtype, device=device)[None, None, :].expand(batch_size, height, width)
    if normalize:
        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * scale
    dim_t = torch.arange(features, dtype=torch.int64, device=device).to(dtype)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / features)
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class Sam3SinePositionEmbedding(nn.Module):
    def __init__(self, num_position_features: int = 64, normalize: bool = False):
        super().__init__()
        self.num_position_features = num_position_features
        self.normalize = normalize
        self.scale = 2 * math.pi

    def _sine(self, values: torch.Tensor) -> torch.Tensor:
        dim_t = torch.arange(self.num_position_features, dtype=torch.int64, device=values.device).to(values.dtype)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_position_features)
        pos = values[..., None] * self.scale / dim_t
        return torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(-2)

    def encode_1d_positions(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sine(x), self._sine(y)

    def encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        x, y, w, h = boxes.unbind(-1)
        return torch.cat((self._sine(y), self._sine(x), self._sine(w), self._sine(h)), dim=-1)

    def forward(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return _sine_position_embedding(
            torch.Size(shape), device, dtype, self.num_position_features, self.normalize, self.scale
        )


class Sam3FPNLayer(nn.Module):
    def __init__(self, in_channels: int, fpn_dim: int, scale_factor: float):
        super().__init__()
        self.scale_layers = nn.ModuleList()
        if scale_factor == 4.0:
            self.scale_layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            self.scale_layers.append(nn.GELU())
            self.scale_layers.append(nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2))
            intermediate_channels = in_channels // 4
        elif scale_factor == 2.0:
            self.scale_layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            intermediate_channels = in_channels // 2
        elif scale_factor == 1.0:
            intermediate_channels = in_channels
        elif scale_factor == 0.5:
            self.scale_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            intermediate_channels = in_channels
        else:
            raise NotImplementedError(f"SAM3 FPN scale_factor={scale_factor} is not supported.")
        self.proj1 = nn.Conv2d(intermediate_channels, fpn_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.proj1.weight.dtype)
        for layer in self.scale_layers:
            hidden_states = layer(hidden_states)
        return self.proj2(self.proj1(hidden_states))


class Sam3VisionNeck(nn.Module):
    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        self.position_encoding = Sam3SinePositionEmbedding(config.fpn_hidden_size // 2, normalize=True)
        # The coarsest level only feeds the video tracker, so the detector does not build it.
        self.fpn_layers = nn.ModuleList(
            [
                Sam3FPNLayer(config.backbone_config.hidden_size, config.fpn_hidden_size, scale)
                for scale in config.scale_factors[:-1]
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> Sam3VisionOutput:
        features, encodings = [], []
        for fpn_layer in self.fpn_layers:
            output = fpn_layer(hidden_states)
            features.append(output)
            encodings.append(self.position_encoding(output.shape, output.device, output.dtype))
        return Sam3VisionOutput(tuple(features), tuple(encodings))


class Sam3VisionModel(nn.Module):
    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        self.backbone = Sam3ViTModel(config.backbone_config)
        self.neck = Sam3VisionNeck(config)

    def forward(self, pixel_values: torch.Tensor) -> Sam3VisionOutput:
        spatial = self.backbone(pixel_values).permute(0, 3, 1, 2)
        return self.neck(spatial)


# ---------------------------------------------------------------------------
# Prompt encoders and the DETR-style detector
# ---------------------------------------------------------------------------


class Sam3PromptLayer(nn.Module):
    """Pre-norm self-attention, cross-attention and MLP block (geometry and DETR encoders)."""

    def __init__(self, config: Sam3BlockConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.self_attn = Sam3Attention(config.hidden_size, config.num_attention_heads)
        self.dropout = nn.Dropout(config.dropout)
        self.cross_attn = Sam3Attention(config.hidden_size, config.num_attention_heads)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = Sam3MLP(config.hidden_size, config.intermediate_size, config.hidden_act, config.hidden_dropout)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size)


class Sam3GeometryEncoderLayer(Sam3PromptLayer):
    def forward(self, prompt_feats, vision_feats, vision_pos_encoding, prompt_mask):
        residual = prompt_feats
        hidden_states = self.layer_norm1(prompt_feats)
        hidden_states = (
            self.dropout(self.self_attn(hidden_states, hidden_states, hidden_states, prompt_mask)) + residual
        )
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.cross_attn(hidden_states, vision_feats + vision_pos_encoding, vision_feats)
        hidden_states = self.dropout(hidden_states) + residual
        residual = hidden_states
        return self.dropout(self.mlp(self.layer_norm3(hidden_states))) + residual


class Sam3GeometryEncoder(nn.Module):
    """Encodes exemplar boxes (with positive/negative labels) into prompt tokens."""

    def __init__(self, config: Sam3BlockConfig):
        super().__init__()
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.roi_size = config.roi_size
        self.position_encoding = Sam3SinePositionEmbedding(hidden // 2, normalize=True)
        self.label_embed = nn.Embedding(2, hidden)
        self.cls_embed = nn.Embedding(1, hidden)
        self.boxes_direct_project = nn.Linear(4, hidden)
        self.boxes_pool_project = nn.Conv2d(hidden, hidden, self.roi_size)
        self.boxes_pos_enc_project = nn.Linear(hidden + 2, hidden)
        self.vision_layer_norm = nn.LayerNorm(hidden)
        self.final_proj = nn.Linear(hidden, hidden)
        self.prompt_layer_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([Sam3GeometryEncoderLayer(config) for _ in range(config.num_layers)])
        self.output_layer_norm = nn.LayerNorm(hidden)

    def _encode_boxes(self, boxes, boxes_labels, vision_features):
        batch_size, num_boxes = boxes.shape[:2]
        height, width = vision_features.shape[-2:]
        boxes_embed = self.boxes_direct_project(boxes)
        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device).view(1, 1, 4)
        boxes_xyxy = box_cxcywh_to_xyxy(boxes) * scale
        # roi_align has no bfloat16 kernel; float16 keeps it on device.
        dtype = torch.float16 if vision_features.dtype == torch.bfloat16 else vision_features.dtype
        pooled = torchvision.ops.roi_align(
            vision_features.to(dtype), list(boxes_xyxy.to(dtype).unbind(0)), self.roi_size
        ).to(vision_features.dtype)
        boxes_embed = boxes_embed + self.boxes_pool_project(pooled).view(batch_size, num_boxes, self.hidden_size)
        center_x, center_y, box_width, box_height = (value.flatten() for value in boxes.unbind(-1))
        pos_x, pos_y = self.position_encoding.encode_1d_positions(center_x, center_y)
        pos_enc = torch.cat((pos_y, pos_x, box_height[:, None], box_width[:, None]), dim=1)
        boxes_embed = boxes_embed + self.boxes_pos_enc_project(pos_enc.view(batch_size, num_boxes, -1))
        return self.label_embed(boxes_labels.long()) + boxes_embed

    def forward(self, boxes, box_mask, box_labels, img_feats, img_pos_embeds):
        batch_size = boxes.shape[0]
        vision_feats = img_feats[-1]
        vision_feats_flat = vision_feats.flatten(2).transpose(1, 2)
        vision_pos_flat = img_pos_embeds[-1].flatten(2).transpose(1, 2)
        normalized = self.vision_layer_norm(vision_feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        prompt_embeds = self._encode_boxes(boxes, box_labels, normalized)
        cls_embed = self.cls_embed.weight.view(1, 1, self.hidden_size).expand(batch_size, -1, -1)
        cls_mask = torch.ones(batch_size, 1, dtype=box_mask.dtype, device=box_mask.device)
        prompt_embeds, prompt_mask = concat_padded_sequences(prompt_embeds, box_mask, cls_embed, cls_mask)
        prompt_embeds = self.prompt_layer_norm(self.final_proj(prompt_embeds))
        attention_mask = key_padding_mask(prompt_mask)
        for layer in self.layers:
            prompt_embeds = layer(prompt_embeds, vision_feats_flat, vision_pos_flat, attention_mask)
        return self.output_layer_norm(prompt_embeds), prompt_mask


class Sam3DetrEncoderLayer(Sam3PromptLayer):
    def forward(self, vision_feats, prompt_feats, vision_pos_encoding, prompt_mask):
        residual = vision_feats
        hidden_states = self.layer_norm1(vision_feats)
        with_pos = hidden_states + vision_pos_encoding
        hidden_states = self.dropout(self.self_attn(with_pos, with_pos, hidden_states)) + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.dropout(self.cross_attn(hidden_states, prompt_feats, prompt_feats, prompt_mask)) + residual
        residual = hidden_states
        return self.dropout(self.mlp(self.layer_norm3(hidden_states))) + residual


class Sam3DetrEncoder(nn.Module):
    """Fuses the prompt into the coarsest-but-one FPN level."""

    def __init__(self, config: Sam3BlockConfig):
        super().__init__()
        self.layers = nn.ModuleList([Sam3DetrEncoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, vision_features, vision_pos_embeds, prompt_features, prompt_mask):
        height, width = vision_features.shape[-2:]
        hidden_states = vision_features.flatten(2).transpose(1, 2)
        pos_embeds = vision_pos_embeds.flatten(2).transpose(1, 2)
        attention_mask = key_padding_mask(prompt_mask)
        for layer in self.layers:
            hidden_states = layer(hidden_states, prompt_features, pos_embeds, attention_mask)
        return hidden_states, pos_embeds, (height, width)


class Sam3DecoderMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        if num_layers == 2:
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.layer3 = None
        elif num_layers == 3:
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Only 2 or 3 layers are supported, got {num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        if self.layer3 is None:
            return self.layer2(x)
        return self.layer3(F.relu(self.layer2(x)))


class Sam3DetrDecoderLayer(nn.Module):
    def __init__(self, config: Sam3BlockConfig):
        super().__init__()
        hidden, heads = config.hidden_size, config.num_attention_heads
        self.self_attn = Sam3Attention(hidden, heads)
        self.self_attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hidden)
        self.text_cross_attn = Sam3Attention(hidden, heads)
        self.text_cross_attn_dropout = nn.Dropout(config.dropout)
        self.text_cross_attn_layer_norm = nn.LayerNorm(hidden)
        self.vision_cross_attn = Sam3Attention(hidden, heads)
        self.vision_cross_attn_dropout = nn.Dropout(config.dropout)
        self.vision_cross_attn_layer_norm = nn.LayerNorm(hidden)
        self.mlp = Sam3MLP(hidden, config.intermediate_size, config.hidden_act, config.hidden_dropout)
        self.mlp_layer_norm = nn.LayerNorm(hidden)
        self.mlp_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, query_pos, text_features, vision_features, vision_pos, text_mask, vision_bias):
        # The presence token (index 0) carries no positional query.
        query_pos = F.pad(query_pos, (0, 0, 1, 0), value=0)
        query = hidden_states + query_pos
        hidden_states = hidden_states + self.self_attn_dropout(self.self_attn(query, query, hidden_states))
        hidden_states = self.self_attn_layer_norm(hidden_states)
        query = hidden_states + query_pos
        attended = self.text_cross_attn(query, text_features, text_features, text_mask)
        hidden_states = self.text_cross_attn_layer_norm(hidden_states + self.text_cross_attn_dropout(attended))
        query = hidden_states + query_pos
        attended = self.vision_cross_attn(query, vision_features + vision_pos, vision_features, vision_bias)
        hidden_states = self.vision_cross_attn_layer_norm(hidden_states + self.vision_cross_attn_dropout(attended))
        return self.mlp_layer_norm(hidden_states + self.mlp_dropout(self.mlp(hidden_states)))


class Sam3DetrDecoder(nn.Module):
    """Iterative box-refining query decoder with a presence token and box relative-position bias."""

    def __init__(self, config: Sam3BlockConfig):
        super().__init__()
        hidden = config.hidden_size
        self.layers = nn.ModuleList([Sam3DetrDecoderLayer(config) for _ in range(config.num_layers)])
        self.output_layer_norm = nn.LayerNorm(hidden)
        self.box_head = Sam3DecoderMLP(hidden, hidden, 4, 3)
        self.query_embed = nn.Embedding(config.num_queries, hidden)
        self.reference_points = nn.Embedding(config.num_queries, 4)
        self.presence_token = nn.Embedding(1, hidden)
        self.presence_head = Sam3DecoderMLP(hidden, hidden, 1, 3)
        self.presence_layer_norm = nn.LayerNorm(hidden)
        self.clamp_presence_logit_max_val = 10.0
        self.ref_point_head = Sam3DecoderMLP(2 * hidden, hidden, hidden, 2)
        self.box_rpb_embed_x = Sam3DecoderMLP(2, hidden, config.num_attention_heads, 2)
        self.box_rpb_embed_y = Sam3DecoderMLP(2, hidden, config.num_attention_heads, 2)
        self.position_encoding = Sam3SinePositionEmbedding(hidden // 2, normalize=False)

    def _rpb_matrix(self, reference_boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        batch_size, num_queries, _ = boxes_xyxy.shape
        coords_h = torch.arange(0, height, device=reference_boxes.device, dtype=reference_boxes.dtype) / height
        coords_w = torch.arange(0, width, device=reference_boxes.device, dtype=reference_boxes.dtype) / width
        deltas_y = (coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]).view(
            batch_size, num_queries, -1, 2
        )
        deltas_x = (coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]).view(
            batch_size, num_queries, -1, 2
        )
        deltas_x = deltas_x * 8
        deltas_x = torch.sign(deltas_x) * torch.log2(torch.abs(deltas_x) + 1.0) / math.log2(8)
        deltas_y = deltas_y * 8
        deltas_y = torch.sign(deltas_y) * torch.log2(torch.abs(deltas_y) + 1.0) / math.log2(8)
        rpb = self.box_rpb_embed_y(deltas_y).unsqueeze(3) + self.box_rpb_embed_x(deltas_x).unsqueeze(2)
        return rpb.flatten(2, 3).permute(0, 3, 1, 2).contiguous()

    def forward(self, vision_features, text_features, vision_pos, text_mask, spatial_shape):
        batch_size = vision_features.shape[0]
        query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = self.reference_points.weight.unsqueeze(0).expand(batch_size, -1, -1).sigmoid()
        presence_token = self.presence_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = torch.cat([presence_token, query_embeds], dim=1)
        attention_mask = key_padding_mask(text_mask)
        query_states, boxes, presence = [], [reference_boxes], []
        for layer in self.layers:
            query_pos = self.ref_point_head(self.position_encoding.encode_boxes(reference_boxes))
            vision_bias = F.pad(self._rpb_matrix(reference_boxes, *spatial_shape), (0, 0, 1, 0), value=0)
            hidden_states = layer(
                hidden_states, query_pos, text_features, vision_features, vision_pos, attention_mask, vision_bias
            )
            queries = self.output_layer_norm(hidden_states[:, 1:])
            reference_boxes = (self.box_head(queries) + inverse_sigmoid(reference_boxes)).sigmoid().detach()
            query_states.append(queries)
            boxes.append(reference_boxes)
            logits = self.presence_head(self.presence_layer_norm(hidden_states[:, :1])).squeeze(-1)
            presence.append(logits.clamp(-self.clamp_presence_logit_max_val, self.clamp_presence_logit_max_val))
        # Final-layer outputs: query states, the reference boxes they refined, and presence logits.
        return query_states[-1], boxes[-2], presence[-1]


class Sam3DotProductScoring(nn.Module):
    def __init__(self, config: Sam3BlockConfig):
        super().__init__()
        hidden = config.hidden_size
        self.text_mlp = Sam3DecoderMLP(hidden, config.intermediate_size, hidden, num_layers=2)
        self.text_mlp_dropout = nn.Dropout(config.dropout)
        self.text_mlp_out_norm = nn.LayerNorm(hidden)
        self.text_proj = nn.Linear(hidden, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.scale = float(1.0 / math.sqrt(hidden))
        self.clamp_max_val = 12.0

    def forward(self, decoder_hidden_states, text_features, text_mask):
        text_features = self.text_mlp_out_norm(self.text_mlp_dropout(self.text_mlp(text_features)) + text_features)
        if text_mask is None:
            pooled = text_features.mean(dim=1)
        else:
            valid = text_mask.to(text_features.dtype).unsqueeze(-1)
            pooled = (text_features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        scores = torch.matmul(self.query_proj(decoder_hidden_states), self.text_proj(pooled).unsqueeze(-1))
        return (scores * self.scale).clamp(min=-self.clamp_max_val, max=self.clamp_max_val)


class Sam3MaskEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        hidden_states = queries
        for index, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if index < len(self.layers) - 1:
                hidden_states = F.relu(hidden_states)
        return hidden_states


class Sam3PixelDecoder(nn.Module):
    def __init__(self, config: Sam3BlockConfig):
        super().__init__()
        hidden = config.hidden_size
        stages = config.num_upsampling_stages
        self.conv_layers = nn.ModuleList([nn.Conv2d(hidden, hidden, kernel_size=3, padding=1) for _ in range(stages)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden) for _ in range(stages)])

    def forward(self, backbone_features: list[torch.Tensor]) -> torch.Tensor:
        previous = backbone_features[-1]
        for index, feature in enumerate(reversed(backbone_features[:-1])):
            previous = F.interpolate(previous, size=feature.shape[-2:], mode="nearest") + feature
            previous = F.relu(self.norms[index](self.conv_layers[index](previous)))
        return previous


class Sam3MaskDecoder(nn.Module):
    def __init__(self, config: Sam3BlockConfig):
        super().__init__()
        hidden = config.hidden_size
        self.pixel_decoder = Sam3PixelDecoder(config)
        self.mask_embedder = Sam3MaskEmbedder(hidden)
        self.instance_projection = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.semantic_projection = nn.Conv2d(hidden, 1, kernel_size=1)
        self.prompt_cross_attn = Sam3Attention(hidden, config.num_attention_heads)
        self.prompt_cross_attn_norm = nn.LayerNorm(hidden)
        self.prompt_cross_attn_dropout = nn.Dropout(config.dropout)

    def forward(self, decoder_queries, backbone_features, encoder_hidden_states, prompt_features, prompt_mask):
        normed = self.prompt_cross_attn_norm(encoder_hidden_states)
        attended = self.prompt_cross_attn(normed, prompt_features, prompt_features, key_padding_mask(prompt_mask))
        encoder_hidden_states = encoder_hidden_states + self.prompt_cross_attn_dropout(attended)
        # The encoder output replaces the coarsest feature map fed to the pixel decoder.
        height, width = backbone_features[-1].shape[-2:]
        batch_size, _, hidden = encoder_hidden_states.shape
        encoder_visual = (
            encoder_hidden_states[:, : height * width].transpose(1, 2).reshape(batch_size, hidden, height, width)
        )
        pixel_embed = self.pixel_decoder([*backbone_features[:-1], encoder_visual])
        pred_masks = torch.einsum(
            "bqc,bchw->bqhw", self.mask_embedder(decoder_queries), self.instance_projection(pixel_embed)
        )
        return pred_masks, self.semantic_projection(pixel_embed)


class Sam3Detector(nn.Module):
    """Image + text/box prompts -> per-query masks, boxes and scores."""

    def __init__(self, config: Sam3DetectorConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = Sam3VisionModel(config.vision_config)
        # A stock CLIP text tower, shared with other vLLM-Omni pipelines.
        self.text_encoder = CLIPTextModelWithProjection(config.text_config)
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.detr_encoder_config.hidden_size)
        self.geometry_encoder = Sam3GeometryEncoder(config.geometry_encoder_config)
        self.detr_encoder = Sam3DetrEncoder(config.detr_encoder_config)
        self.detr_decoder = Sam3DetrDecoder(config.detr_decoder_config)
        self.mask_decoder = Sam3MaskDecoder(config.mask_decoder_config)
        self.dot_product_scoring = Sam3DotProductScoring(config.detr_decoder_config)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.text_projection(hidden)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_boxes: torch.Tensor | None = None,
        input_boxes_labels: torch.Tensor | None = None,
    ) -> Sam3DetectorOutput:
        vision = self.vision_encoder(pixel_values)
        fpn_features = vision.fpn_hidden_states
        fpn_encodings = vision.fpn_position_encoding
        prompt_features = self.encode_text(input_ids, attention_mask)
        prompt_mask = attention_mask.bool()
        if input_boxes is not None and input_boxes.numel() > 0:
            labels = input_boxes_labels
            if labels is None:
                labels = torch.ones(input_boxes.shape[:2], dtype=torch.long, device=input_boxes.device)
            box_mask = labels != -10
            box_features, box_mask = self.geometry_encoder(
                input_boxes, box_mask, torch.where(labels == -10, 0, labels), fpn_features, fpn_encodings
            )
            if prompt_features.shape[0] == 1 and box_features.shape[0] > 1:
                prompt_features = prompt_features.repeat(box_features.shape[0], 1, 1)
                prompt_mask = prompt_mask.repeat(box_features.shape[0], 1)
            prompt_features = torch.cat([prompt_features, box_features], dim=1)
            prompt_mask = torch.cat([prompt_mask, box_mask], dim=1)
        encoded, encoded_pos, spatial_shape = self.detr_encoder(
            fpn_features[-1], fpn_encodings[-1], prompt_features, prompt_mask
        )
        queries, reference_boxes, presence_logits = self.detr_decoder(
            encoded, prompt_features, encoded_pos, prompt_mask, spatial_shape
        )
        boxes = box_cxcywh_to_xyxy((inverse_sigmoid(reference_boxes) + self.detr_decoder.box_head(queries)).sigmoid())
        logits = self.dot_product_scoring(queries, prompt_features, prompt_mask).squeeze(-1)
        pred_masks, semantic_seg = self.mask_decoder(queries, list(fpn_features), encoded, prompt_features, prompt_mask)
        return Sam3DetectorOutput(pred_masks, boxes, logits, presence_logits, semantic_seg)


_DETECTOR_MODULES = "backbone|transformer|geometry_encoder|segmentation_head|dot_prod_scoring"
_LAYER = r"\.(\d+)\."
_MLP_HEADS = (
    r"detr_decoder\.(?:box_head|presence_head|ref_point_head|box_rpb_embed_x|box_rpb_embed_y)"
    r"|dot_product_scoring\.text_mlp"
)

# Released checkpoints come in three layouts: Meta's native one (``detector.*`` and
# ``tracker.*``), the combined Transformers one (``detector_model.*`` and
# ``tracker_model.*``) and the image-only Transformers one (no prefix). They all
# map onto this module; tracker and video-only tensors are dropped.
SAM3_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_regex={
        re.compile(r"^(tracker|tracker_model|tracker_neck)\."): None,
        re.compile(r"^detector_model\."): "",
        re.compile(rf"^detector\.({_DETECTOR_MODULES})\."): r"\1.",
        # Detector tensors the image path never uses: video/interactive necks,
        # point prompts, instance queries and the precomputed RoPE table.
        re.compile(
            r"(interactive_convs|propagation_convs|\.freqs_cis$"
            r"|geometry_encoder\.points_|transformer\.decoder\.instance_)"
        ): None,
        # Vision backbone.
        re.compile(
            r"^backbone\.vision_backbone\.trunk\.pos_embed"
        ): "vision_encoder.backbone.embeddings.position_embeddings",
        re.compile(r"^backbone\.vision_backbone\.trunk\.patch_embed\.proj\."): (
            "vision_encoder.backbone.embeddings.patch_embeddings.projection."
        ),
        re.compile(r"^backbone\.vision_backbone\.trunk\.ln_pre\."): "vision_encoder.backbone.layer_norm.",
        re.compile(rf"^backbone\.vision_backbone\.trunk\.blocks{_LAYER}norm(\d)\."): (
            r"vision_encoder.backbone.layers.\1.layer_norm\2."
        ),
        re.compile(rf"^backbone\.vision_backbone\.trunk\.blocks{_LAYER}attn\.qkv\."): (
            r"vision_encoder.backbone.layers.\1.attention.qkv."
        ),
        re.compile(rf"^backbone\.vision_backbone\.trunk\.blocks{_LAYER}attn\.proj\."): (
            r"vision_encoder.backbone.layers.\1.attention.o_proj."
        ),
        re.compile(
            rf"^backbone\.vision_backbone\.trunk\.blocks{_LAYER}mlp\."
        ): r"vision_encoder.backbone.layers.\1.mlp.",
        re.compile(rf"^backbone\.vision_backbone\.convs{_LAYER}dconv_2x2(_0)?\."): (
            r"vision_encoder.neck.fpn_layers.\1.scale_layers.0."
        ),
        re.compile(rf"^backbone\.vision_backbone\.convs{_LAYER}dconv_2x2_1\."): (
            r"vision_encoder.neck.fpn_layers.\1.scale_layers.2."
        ),
        re.compile(
            rf"^backbone\.vision_backbone\.convs{_LAYER}conv_1x1\."
        ): r"vision_encoder.neck.fpn_layers.\1.proj1.",
        re.compile(
            rf"^backbone\.vision_backbone\.convs{_LAYER}conv_3x3\."
        ): r"vision_encoder.neck.fpn_layers.\1.proj2.",
        # OpenCLIP text tower -> Transformers CLIP text tower.
        re.compile(r"^backbone\.language_backbone\.resizer\."): "text_projection.",
        re.compile(r"^backbone\.language_backbone\.encoder\.token_embedding\."): (
            "text_encoder.text_model.embeddings.token_embedding."
        ),
        re.compile(r"^backbone\.language_backbone\.encoder\.positional_embedding$"): (
            "text_encoder.text_model.embeddings.position_embedding.weight"
        ),
        re.compile(r"^backbone\.language_backbone\.encoder\.ln_final\."): "text_encoder.text_model.final_layer_norm.",
        re.compile(r"^backbone\.language_backbone\.encoder\.text_projection$"): "text_encoder.text_projection.weight",
        re.compile(rf"^backbone\.language_backbone\.encoder\.transformer\.resblocks{_LAYER}attn\.out_proj\."): (
            r"text_encoder.text_model.encoder.layers.\1.self_attn.out_proj."
        ),
        re.compile(rf"^backbone\.language_backbone\.encoder\.transformer\.resblocks{_LAYER}attn\.in_proj_"): (
            r"text_encoder.text_model.encoder.layers.\1.self_attn.in_proj_"
        ),
        re.compile(rf"^backbone\.language_backbone\.encoder\.transformer\.resblocks{_LAYER}ln_(\d)\."): (
            r"text_encoder.text_model.encoder.layers.\1.layer_norm\2."
        ),
        re.compile(rf"^backbone\.language_backbone\.encoder\.transformer\.resblocks{_LAYER}mlp\.c_fc\."): (
            r"text_encoder.text_model.encoder.layers.\1.mlp.fc1."
        ),
        re.compile(rf"^backbone\.language_backbone\.encoder\.transformer\.resblocks{_LAYER}mlp\.c_proj\."): (
            r"text_encoder.text_model.encoder.layers.\1.mlp.fc2."
        ),
        # Geometry encoder.
        re.compile(rf"^geometry_encoder\.encode{_LAYER}cross_attn_image\."): r"geometry_encoder.layers.\1.cross_attn.",
        re.compile(rf"^geometry_encoder\.encode{_LAYER}linear(\d)\."): r"geometry_encoder.layers.\1.mlp.fc\2.",
        re.compile(rf"^geometry_encoder\.encode{_LAYER}norm(\d)\."): r"geometry_encoder.layers.\1.layer_norm\2.",
        re.compile(rf"^geometry_encoder\.encode{_LAYER}"): r"geometry_encoder.layers.\1.",
        re.compile(r"^geometry_encoder\.img_pre_norm\."): "geometry_encoder.vision_layer_norm.",
        re.compile(r"^geometry_encoder\.norm\."): "geometry_encoder.prompt_layer_norm.",
        re.compile(r"^geometry_encoder\.encode_norm\."): "geometry_encoder.output_layer_norm.",
        # DETR encoder.
        re.compile(rf"^transformer\.encoder\.layers{_LAYER}cross_attn_image\."): r"detr_encoder.layers.\1.cross_attn.",
        re.compile(rf"^transformer\.encoder\.layers{_LAYER}linear(\d)\."): r"detr_encoder.layers.\1.mlp.fc\2.",
        re.compile(rf"^transformer\.encoder\.layers{_LAYER}norm(\d)\."): r"detr_encoder.layers.\1.layer_norm\2.",
        re.compile(rf"^transformer\.encoder\.layers{_LAYER}"): r"detr_encoder.layers.\1.",
        # DETR decoder.
        re.compile(r"^transformer\.decoder\.presence_token_head\."): "detr_decoder.presence_head.",
        re.compile(r"^transformer\.decoder\.presence_token_out_norm\."): "detr_decoder.presence_layer_norm.",
        re.compile(r"^transformer\.decoder\.norm\."): "detr_decoder.output_layer_norm.",
        re.compile(r"^transformer\.decoder\.bbox_embed\."): "detr_decoder.box_head.",
        re.compile(r"^transformer\.decoder\.ref_point_head\."): "detr_decoder.ref_point_head.",
        re.compile(r"^transformer\.decoder\.boxRPB_embed_(x|y)\."): r"detr_decoder.box_rpb_embed_\1.",
        re.compile(rf"^transformer\.decoder\.layers{_LAYER}ca_text\."): r"detr_decoder.layers.\1.text_cross_attn.",
        re.compile(rf"^transformer\.decoder\.layers{_LAYER}cross_attn\."): r"detr_decoder.layers.\1.vision_cross_attn.",
        re.compile(rf"^transformer\.decoder\.layers{_LAYER}linear(\d)\."): r"detr_decoder.layers.\1.mlp.fc\2.",
        re.compile(
            rf"^transformer\.decoder\.layers{_LAYER}norm1\."
        ): r"detr_decoder.layers.\1.vision_cross_attn_layer_norm.",
        re.compile(
            rf"^transformer\.decoder\.layers{_LAYER}catext_norm\."
        ): r"detr_decoder.layers.\1.text_cross_attn_layer_norm.",
        re.compile(rf"^transformer\.decoder\.layers{_LAYER}norm2\."): r"detr_decoder.layers.\1.self_attn_layer_norm.",
        re.compile(rf"^transformer\.decoder\.layers{_LAYER}norm3\."): r"detr_decoder.layers.\1.mlp_layer_norm.",
        re.compile(r"^transformer\.decoder\."): "detr_decoder.",
        # Scoring and mask head.
        re.compile(r"^dot_prod_scoring\.prompt_mlp\.layers\."): "dot_product_scoring.text_mlp.layers.",
        re.compile(r"^dot_prod_scoring\.prompt_mlp\.out_norm\."): "dot_product_scoring.text_mlp_out_norm.",
        re.compile(r"^dot_prod_scoring\.prompt_proj\."): "dot_product_scoring.text_proj.",
        re.compile(r"^dot_prod_scoring\.hs_proj\."): "dot_product_scoring.query_proj.",
        re.compile(r"^segmentation_head\.(mask_predictor\.)?mask_embed\."): "mask_decoder.mask_embedder.",
        re.compile(r"^segmentation_head\.instance_seg_head\."): "mask_decoder.instance_projection.",
        re.compile(r"^segmentation_head\.semantic_seg_head\."): "mask_decoder.semantic_projection.",
        re.compile(r"^segmentation_head\.cross_attend_prompt\."): "mask_decoder.prompt_cross_attn.",
        re.compile(r"^segmentation_head\.cross_attn_norm\."): "mask_decoder.prompt_cross_attn_norm.",
        re.compile(r"^segmentation_head\.pixel_decoder\."): "mask_decoder.pixel_decoder.",
        # Native MLP heads index layers from 0; Sam3DecoderMLP names them layer1..layer3.
        re.compile(rf"^({_MLP_HEADS})\.layers\.0\."): r"\1.layer1.",
        re.compile(rf"^({_MLP_HEADS})\.layers\.1\."): r"\1.layer2.",
        re.compile(rf"^({_MLP_HEADS})\.layers\.2\."): r"\1.layer3.",
        # nn.MultiheadAttention names its output projection out_proj.
        re.compile(r"^(?!text_encoder)(.*)\.out_proj\."): r"\1.o_proj.",
    },
    # Transformers checkpoints split the ViT projection; load it into the fused one.
    orig_to_new_stacked={
        ".attention.q_proj.": (".attention.qkv.", "q"),
        ".attention.k_proj.": (".attention.qkv.", "k"),
        ".attention.v_proj.": (".attention.qkv.", "v"),
    },
)


def load_sam3_weights(detector: Sam3Detector, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Load any SAM3 checkpoint layout into ``detector``; returns the loaded parameter names."""
    params = dict(detector.named_parameters())
    loaded: set[str] = set()

    tracker_level = f"vision_encoder.neck.fpn_layers.{len(detector.vision_encoder.neck.fpn_layers)}."

    def load(name: str, tensor: torch.Tensor, shard_id: str | None = None) -> None:
        if name.startswith(tracker_level):
            return  # Transformers exports keep the tracker-only FPN level.
        if name not in params:
            raise ValueError(f"SAM3 checkpoint tensor maps to unknown parameter {name!r}")
        param = params[name]
        if name.endswith("embeddings.position_embeddings") and tensor.shape[1] == param.shape[1] + 1:
            tensor = tensor[:, 1:]  # The native table carries an unused class-token slot.
        elif name == "text_encoder.text_projection.weight" and tensor.shape == param.shape[::-1]:
            tensor = tensor.t()  # OpenCLIP stores the projection as x @ W.
        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, tensor, shard_id) if shard_id is not None else loader(param, tensor)
        loaded.add(name)

    for name, tensor in SAM3_WEIGHTS_MAPPER.apply(weights):
        stem, found, kind = name.rpartition(".in_proj_")
        if found:
            # nn.MultiheadAttention packs q, k and v; this module keeps them separate.
            for part, chunk in zip(("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0)):
                load(f"{stem}.{part}.{kind}", chunk)
            continue
        load(name, tensor, getattr(tensor, "shard_id", None))
    return loaded

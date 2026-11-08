# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import inspect
import json
import math
import os
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from safetensors.torch import load_file
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention as VllmAttention


def linear_fp32(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if not hasattr(linear, "weight"):
        return linear(x.float())
    bias = None if linear.bias is None else linear.bias.float()
    return F.linear(x.float(), linear.weight.float(), bias)


def silu_linear_fp32(module: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    return linear_fp32(module[1], F.silu(x.float()))


class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm_FP32(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerNorm_FP32(nn.LayerNorm):
    def __init__(self, dim, eps, elementwise_affine):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        ).to(origin_dtype)
        return out


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, depth, height, width = x.size()
        if width % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - width % self.patch_size[2]))
        if height % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - height % self.patch_size[1]))
        if depth % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - depth % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            depth, height, width = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, depth, height, width)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


def modulate_fp32(norm_func, x, shift, scale):
    assert shift.dtype == torch.float32 and scale.dtype == torch.float32
    dtype = x.dtype
    x = norm_func(x.to(torch.float32))
    x = x * (scale + 1) + shift
    x = x.to(dtype)
    return x


class FinalLayer_FP32(nn.Module):
    def __init__(self, hidden_size, num_patch, out_channels, adaln_tembed_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patch = num_patch
        self.out_channels = out_channels
        self.adaln_tembed_dim = adaln_tembed_dim

        self.norm_final = LayerNorm_FP32(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 2 * hidden_size, bias=True))

    def forward(self, x, t, latent_shape):
        assert t.dtype == torch.float32
        batch, seq_len, channels = x.shape
        num_frames, _, _ = latent_shape

        shift, scale = silu_linear_fp32(self.adaLN_modulation, t).unsqueeze(2).chunk(2, dim=-1)
        x = modulate_fp32(self.norm_final, x.view(batch, num_frames, -1, channels), shift, scale).view(
            batch, seq_len, channels
        )
        x = linear_fp32(self.linear, x)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, t_embed_dim, frequency_embedding_size=256):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, t_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = linear_fp32(self.mlp[0], t_freq)
        t_emb = F.silu(t_emb)
        t_emb = linear_fp32(self.mlp[2], t_emb)
        return t_emb


class CaptionEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, caption):
        caption = self.y_proj(caption)
        return caption


def broadcast_concat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(map(lambda t: len(t.shape), tensors))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), "invalid broadcast concat dimensions"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def normalize_and_scale(values, source_range, target_range):
    source_min, source_max = source_range
    target_min, target_max = target_range
    denom = source_max - source_min
    if torch.is_tensor(denom):
        denom = torch.clamp(denom, min=torch.finfo(values.dtype).eps)
    else:
        denom = max(denom, torch.finfo(values.dtype).eps)
    normalized = (values - source_min) / denom
    return normalized * (target_max - target_min) + target_min


class RotaryPositionalEmbedding1D(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000

    def precompute_freqs_cis_1d(self, pos_indices):
        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        return repeat(freqs, "... n -> ... (n r)", r=2)

    def forward(self, x, pos_indices):
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)
        x_ = x.float()
        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, "n d -> 1 1 n d"), rearrange(sin, "n d -> 1 1 n d")
        x_ = (x_ * cos) + (rotate_half(x_) * sin)
        return x_.type_as(x)


def _calculate_x_ref_attn_map(
    noise_q: torch.Tensor,
    ref_k: torch.Tensor,
    ref_target_masks: torch.Tensor,
) -> torch.Tensor:
    ref_k = ref_k.to(noise_q.dtype).to(noise_q.device)
    noise_q = noise_q * (1.0 / noise_q.shape[-1] ** 0.5)
    noise_q = noise_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = noise_q @ ref_k.transpose(-2, -1)
    attn_source = attn.softmax(-1).to(noise_q.dtype)

    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(noise_q.dtype)
    for ref_target_mask in ref_target_masks:
        ref_target_mask = ref_target_mask[None, None, None, ...]
        mask_sum = torch.clamp(ref_target_mask.sum(), min=torch.finfo(noise_q.dtype).eps)
        x_ref_attn_map = (attn_source * ref_target_mask).sum(-1) / mask_sum
        x_ref_attn_map = x_ref_attn_map.permute(0, 2, 1).mean(-1)
        x_ref_attn_maps.append(x_ref_attn_map)
    return torch.cat(x_ref_attn_maps, dim=0)


def _get_attn_map_with_target(
    noise_q: torch.Tensor,
    key: torch.Tensor,
    shape: tuple[int, int, int],
    ref_target_masks: torch.Tensor,
    split_num: int = 2,
    cp_split_hw=None,
) -> torch.Tensor:
    if cp_split_hw is not None and cp_split_hw[0] * cp_split_hw[1] > 1:
        raise NotImplementedError("LongCat-Video-Avatar multi-speaker masks do not support context parallel.")
    _, num_h, num_w = shape
    ref_seq_len = num_h * num_w
    ref_k = key[:, :ref_seq_len]
    if ref_target_masks.shape[-1] != ref_seq_len:
        raise ValueError(f"Expected ref_target_masks token length {ref_seq_len}, got {ref_target_masks.shape[-1]}.")

    _, seq_len, heads, _ = noise_q.shape
    class_num = ref_target_masks.shape[0]
    x_ref_attn_maps = torch.zeros(class_num, seq_len, dtype=noise_q.dtype, device=noise_q.device)
    split_num = max(1, min(split_num, heads))
    for noise_q_chunk, ref_k_chunk in zip(
        torch.chunk(noise_q.contiguous(), split_num, dim=2),
        torch.chunk(ref_k.contiguous(), split_num, dim=2),
    ):
        x_ref_attn_maps += _calculate_x_ref_attn_map(noise_q_chunk, ref_k_chunk, ref_target_masks)
    return x_ref_attn_maps / split_num


class RotaryPositionalEmbedding(nn.Module):
    _MAX_FREQS_LRU_CACHE_SIZE = 32

    def __init__(self, head_dim, cp_split_hw=None):
        super().__init__()
        self.head_dim = head_dim
        assert self.head_dim % 8 == 0, "Dim must be a multiply of 8 for 3D RoPE."
        self.cp_split_hw = tuple(cp_split_hw or (1, 1))
        self.base = 10000
        self.freqs_dict = OrderedDict()

    def register_grid_size(self, grid_size, key_name, frame_index=None, num_ref_latents=None):
        if key_name in self.freqs_dict:
            self.freqs_dict.move_to_end(key_name)
            return
        self.freqs_dict[key_name] = self.precompute_freqs_cis_3d(grid_size, frame_index, num_ref_latents)
        if len(self.freqs_dict) > self._MAX_FREQS_LRU_CACHE_SIZE:
            self.freqs_dict.popitem(last=False)

    def precompute_freqs_cis_3d(self, grid_size, frame_index=None, num_ref_latents=None):
        num_frames, height, width = grid_size
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)
        freqs_t = 1.0 / (self.base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (self.base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (self.base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
        if frame_index is not None and num_ref_latents is not None:
            grid_t = torch.cat(
                [
                    torch.tensor([frame_index], dtype=torch.float32),
                    torch.arange(0, num_frames - num_ref_latents, dtype=torch.float32),
                ],
                dim=0,
            )
        else:
            grid_t = torch.from_numpy(np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)).float()
        grid_h = torch.from_numpy(np.linspace(0, height, height, endpoint=False, dtype=np.float32)).float()
        grid_w = torch.from_numpy(np.linspace(0, width, width, endpoint=False, dtype=np.float32)).float()
        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
        freqs = broadcast_concat(
            (freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1
        )
        freqs = rearrange(freqs, "T H W D -> (T H W) D")
        if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
            raise NotImplementedError("LongCat-Video-Avatar context parallel RoPE is not supported in native MVP.")
        return freqs

    def forward(self, q, k, grid_size, frame_index=None, num_ref_latents=None):
        key_name = (tuple(grid_size), frame_index, num_ref_latents)
        self.register_grid_size(grid_size, key_name, frame_index, num_ref_latents)

        freqs_cis = self.freqs_dict[key_name].to(q.device)
        q_, k_ = q.float(), k.float()
        freqs_cis = freqs_cis.float().to(q.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, "n d -> 1 1 n d"), rearrange(sin, "n d -> 1 1 n d")
        q_ = (q_ * cos) + (rotate_half(q_) * sin)
        k_ = (k_ * cos) + (rotate_half(k_) * sin)
        return q_.type_as(q), k_.type_as(k)


class Attention(nn.Module):
    """LongCat self-attention projections with vLLM-Omni attention execution."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict | None = None,
        cp_split_hw: tuple[int, int] | list[int] | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if enable_bsa:
            raise NotImplementedError("LongCat-Video-Avatar block-sparse attention is not supported in native MVP.")
        if cp_split_hw not in (None, [1, 1], (1, 1)):
            raise NotImplementedError("LongCat-Video-Avatar context parallel attention is not supported in native MVP.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

        self.rope_3d = RotaryPositionalEmbedding(self.head_dim, cp_split_hw=(1, 1))
        self.attn = VllmAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=self.scale,
            causal=False,
            role="longcat_video_avatar.self",
            role_category="self",
            qkv_layout="BSND",
        )

    def _process_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.shape[2] == 0:
            return q
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        x = self.attn(q, k, v)
        return x.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        shape: tuple[int, int, int] | None = None,
        num_cond_latents: int | None = None,
        return_kv: bool = False,
        num_ref_latents=None,
        ref_img_index=None,
        mask_frame_range=None,
        ref_target_masks=None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()

        q, k = self.rope_3d(q, k, shape, ref_img_index, num_ref_latents)

        num_cond_latents_thw = None
        if num_cond_latents is not None and num_cond_latents > 0:
            if shape is None:
                raise ValueError("shape is required when num_cond_latents is set.")
            tokens_per_frame = seq_len // shape[0]
            num_cond_latents_thw = num_cond_latents * tokens_per_frame
            if num_cond_latents == 1:
                q_cond = q[:, :, :num_cond_latents_thw].contiguous()
                k_cond = k[:, :, :num_cond_latents_thw].contiguous()
                v_cond = v[:, :, :num_cond_latents_thw].contiguous()
                x_cond = self._process_attn(q_cond, k_cond, v_cond)
                q_noise = q[:, :, num_cond_latents_thw:].contiguous()
                x_noise = self._process_attn(q_noise, k, v)
                x = torch.cat([x_cond, x_noise], dim=2).contiguous()
            else:
                if num_ref_latents is None or ref_img_index is None:
                    raise ValueError("Video continuation requires num_ref_latents and ref_img_index.")
                num_ref_latents_thw = num_ref_latents * tokens_per_frame
                q_ref = q[:, :, :num_ref_latents_thw].contiguous()
                k_ref = k[:, :, :num_ref_latents_thw].contiguous()
                v_ref = v[:, :, :num_ref_latents_thw].contiguous()
                q_cond = q[:, :, num_ref_latents_thw:num_cond_latents_thw].contiguous()
                k_cond = k[:, :, num_ref_latents_thw:num_cond_latents_thw].contiguous()
                v_cond = v[:, :, num_ref_latents_thw:num_cond_latents_thw].contiguous()
                x_ref = self._process_attn(q_ref, k_ref, v_ref)
                x_cond = self._process_attn(q_cond, k_cond, v_cond)
                if num_cond_latents == shape[0]:
                    x = torch.cat([x_ref, x_cond], dim=2).contiguous()
                else:
                    q_noise = q[:, :, num_cond_latents_thw:].contiguous()
                    start_noise, end_noise, num_noisy_frames = 0, 0, shape[0] - num_cond_latents
                    if mask_frame_range is not None and mask_frame_range > 0:
                        start_noise = ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents
                        end_noise = ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1

                    if start_noise >= 0 and end_noise > start_noise and end_noise <= num_noisy_frames:
                        start_pos = start_noise * tokens_per_frame
                        end_pos = end_noise * tokens_per_frame
                        q_noise_front = q_noise[:, :, :start_pos].contiguous()
                        q_noise_maskref = q_noise[:, :, start_pos:end_pos].contiguous()
                        q_noise_back = q_noise[:, :, end_pos:].contiguous()
                        k_non_ref = k[:, :, num_ref_latents_thw:].contiguous()
                        v_non_ref = v[:, :, num_ref_latents_thw:].contiguous()
                        x_noise_front = self._process_attn(q_noise_front, k, v)
                        x_noise_maskref = self._process_attn(q_noise_maskref, k_non_ref, v_non_ref)
                        x_noise_back = self._process_attn(q_noise_back, k, v)
                        x_noise = torch.cat([x_noise_front, x_noise_maskref, x_noise_back], dim=2).contiguous()
                    else:
                        x_noise = self._process_attn(q_noise, k, v)
                    x = torch.cat([x_ref, x_cond, x_noise], dim=2).contiguous()
        else:
            x = self._process_attn(q, k, v)

        x = x.transpose(1, 2).reshape(bsz, seq_len, channels)
        x = self.proj(x)
        x_ref_attn_map = None
        if ref_target_masks is not None:
            if shape is None or num_cond_latents_thw is None or num_cond_latents_thw <= 0:
                raise ValueError("Multi-speaker Avatar masks require image/video condition latents.")
            x_ref_attn_map = _get_attn_map_with_target(
                q.permute(0, 2, 1, 3)[:, num_cond_latents_thw:].type_as(x),
                k.permute(0, 2, 1, 3).type_as(x),
                shape,
                ref_target_masks,
                cp_split_hw=self.rope_3d.cp_split_hw,
            )
        if return_kv:
            return x, (k_cache, v_cache), x_ref_attn_map
        return x, x_ref_attn_map

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        shape: tuple[int, int, int] | None = None,
        num_cond_latents: int | None = None,
        kv_cache=None,
        num_ref_latents=None,
        ref_img_index=None,
        mask_frame_range=None,
        ref_target_masks=None,
    ) -> torch.Tensor:
        if shape is None:
            raise ValueError("shape is required for LongCat-Video-Avatar KV-cache attention.")
        if kv_cache is None:
            raise ValueError("kv_cache is required for forward_with_kv_cache.")

        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        k_cache, v_cache = kv_cache
        k_cache = k_cache.to(device=x.device, dtype=k.dtype)
        v_cache = v_cache.to(device=x.device, dtype=v.dtype)
        if k_cache.shape[0] == 1 and bsz > 1:
            k_cache = k_cache.repeat(bsz, 1, 1, 1)
            v_cache = v_cache.repeat(bsz, 1, 1, 1)
        if k_cache.shape[0] != bsz:
            raise ValueError(f"KV cache batch {k_cache.shape[0]} must be 1 or match input batch {bsz}.")

        k_full = torch.cat([k_cache, k], dim=2).contiguous()
        v_full = torch.cat([v_cache, v], dim=2).contiguous()
        if num_cond_latents is not None and num_cond_latents > 0:
            q_padding = torch.cat([torch.empty_like(k_cache), q], dim=2).contiguous()
            q_padding, k_full = self.rope_3d(
                q_padding,
                k_full,
                (shape[0] + num_cond_latents, shape[1], shape[2]),
                ref_img_index,
                num_ref_latents,
            )
            q = q_padding[:, :, -seq_len:].contiguous()

        tokens_per_frame = seq_len // shape[0]
        start_noise, end_noise, num_noisy_frames = 0, 0, shape[0]
        if mask_frame_range is not None and mask_frame_range > 0:
            start_noise = ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents
            end_noise = ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1

        if start_noise >= 0 and end_noise > start_noise and end_noise <= num_noisy_frames:
            num_ref_latents_thw = (num_ref_latents or 0) * tokens_per_frame
            start_pos = start_noise * tokens_per_frame
            end_pos = end_noise * tokens_per_frame
            q_noise_front = q[:, :, :start_pos].contiguous()
            q_noise_maskref = q[:, :, start_pos:end_pos].contiguous()
            q_noise_back = q[:, :, end_pos:].contiguous()
            k_non_ref = k_full[:, :, num_ref_latents_thw:].contiguous()
            v_non_ref = v_full[:, :, num_ref_latents_thw:].contiguous()
            x_noise_front = self._process_attn(q_noise_front, k_full, v_full)
            x_noise_maskref = self._process_attn(q_noise_maskref, k_non_ref, v_non_ref)
            x_noise_back = self._process_attn(q_noise_back, k_full, v_full)
            x_attn = torch.cat([x_noise_front, x_noise_maskref, x_noise_back], dim=2).contiguous()
        else:
            x_attn = self._process_attn(q, k_full, v_full)

        x_out = x_attn.transpose(1, 2).reshape(bsz, seq_len, channels)
        x_out = self.proj(x_out)

        x_ref_attn_map = None
        if ref_target_masks is not None:
            x_ref_attn_map = _get_attn_map_with_target(
                q.permute(0, 2, 1, 3).type_as(x_out),
                k_full.permute(0, 2, 1, 3).type_as(x_out),
                shape,
                ref_target_masks,
                cp_split_hw=self.rope_3d.cp_split_hw,
            )
        return x_out, x_ref_attn_map


class MultiHeadCrossAttention(nn.Module):
    """LongCat cross-attention projections with vLLM-Omni attention execution."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.attn = VllmAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=self.scale,
            causal=False,
            role="longcat_video_avatar.cross",
            role_category="cross",
            qkv_layout="BSND",
            skip_sequence_parallel=True,
            disable_kv_quant=True,
        )

    def _unpack_condition(
        self,
        cond: torch.Tensor,
        kv_seqlen: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cond.shape[0] != 1:
            return cond, None

        batch = len(kv_seqlen)
        max_len = max(kv_seqlen)
        if len(set(kv_seqlen)) == 1:
            return cond.view(batch, max_len, self.dim), None

        padded = cond.new_zeros(batch, max_len, self.dim)
        mask = torch.zeros(batch, max_len, dtype=torch.bool, device=cond.device)
        offset = 0
        flat = cond.squeeze(0)
        for idx, seq_len in enumerate(kv_seqlen):
            padded[idx, :seq_len] = flat[offset : offset + seq_len]
            mask[idx, :seq_len] = True
            offset += seq_len
        return padded, mask

    def _process_cross_attn(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        kv_seqlen: list[int],
    ) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected x channel dim {self.dim}, got {channels}.")

        cond, attn_mask = self._unpack_condition(cond, kv_seqlen)
        q = self.q_linear(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(bsz, cond.shape[1], 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)

        metadata = AttentionMetadata(attn_mask=attn_mask) if attn_mask is not None else None
        x = self.attn(q, k, v, metadata)
        x = x.reshape(bsz, seq_len, channels)
        return self.proj(x)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        kv_seqlen: list[int],
        num_cond_latents: int | None = None,
        shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        if num_cond_latents is None or num_cond_latents == 0:
            return self._process_cross_attn(x, cond, kv_seqlen)

        bsz, seq_len, channels = x.shape
        if shape is None:
            raise ValueError("shape is required when num_cond_latents is set.")
        num_cond_latents_thw = num_cond_latents * (seq_len // shape[0])
        x_noise = x[:, num_cond_latents_thw:]
        output_noise = self._process_cross_attn(x_noise, cond, kv_seqlen)
        return torch.cat(
            [
                torch.zeros(
                    (bsz, num_cond_latents_thw, channels),
                    dtype=output_noise.dtype,
                    device=output_noise.device,
                ),
                output_noise,
            ],
            dim=1,
        ).contiguous()


class AvatarSelfAttention(Attention):
    """Avatar self-attention with vLLM-Omni attention execution."""

    def forward(
        self,
        x: torch.Tensor,
        shape: tuple[int, int, int] | None = None,
        num_cond_latents: int | None = None,
        return_kv: bool = False,
        num_ref_latents=None,
        ref_img_index=None,
        mask_frame_range=None,
        ref_target_masks=None,
    ):
        out = super().forward(
            x,
            shape=shape,
            num_cond_latents=num_cond_latents,
            return_kv=return_kv,
            num_ref_latents=num_ref_latents,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            ref_target_masks=ref_target_masks,
        )
        if return_kv:
            x_out, kv_cache, x_ref_attn_map = out
            return x_out, kv_cache, x_ref_attn_map
        return out


class SingleStreamAudioAttention(nn.Module):
    """Avatar audio cross-attention using vLLM-Omni attention backend."""

    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
        class_range: int = 24,
        class_interval: int = 4,
        cp_split_hw=None,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if cp_split_hw not in (None, [1, 1], (1, 1)):
            raise NotImplementedError("LongCat-Video-Avatar context parallel is not supported in native MVP.")
        if attn_drop != 0.0 or proj_drop != 0.0:
            raise NotImplementedError("LongCat-Video-Avatar native MVP only supports zero attention/proj dropout.")

        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cp_split_hw = cp_split_hw

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Identity()
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1 = (0, self.class_interval)
        self.rope_h2 = (self.class_range - self.class_interval, self.class_range)
        self.rope_bak = int(self.class_range // 2)
        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

        self.attn = VllmAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=self.scale,
            causal=False,
            role="longcat_video.avatar_audio",
            role_category="cross",
            qkv_layout="BSND",
            skip_sequence_parallel=True,
            disable_kv_quant=True,
        )

    def _process_cross_attn(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        frames_num: int,
        x_ref_attn_map=None,
        human_num=None,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=frames_num)
        batch, seq_len, channels = x.shape
        q = self.q_linear(x).view(batch, seq_len, self.num_heads, self.head_dim)
        q = self.q_norm(q)

        if x_ref_attn_map is not None:
            if x_ref_attn_map.shape[0] < 2:
                raise ValueError("Multi-speaker Avatar requires at least two speaker attention maps.")
            human1 = normalize_and_scale(
                x_ref_attn_map[0],
                (x_ref_attn_map[0].min(), x_ref_attn_map[0].max()),
                (self.rope_h1[0], self.rope_h1[1]),
            )
            human2 = normalize_and_scale(
                x_ref_attn_map[1],
                (x_ref_attn_map[1].min(), x_ref_attn_map[1].max()),
                (self.rope_h2[0], self.rope_h2[1]),
            )
            background_pos = self.rope_bak if x_ref_attn_map.shape[0] <= 3 else 100
            background = torch.full(
                (x_ref_attn_map.size(1),),
                background_pos,
                dtype=human1.dtype,
                device=human1.device,
            )
            max_indices = x_ref_attn_map.argmax(dim=0).clamp(max=2)
            normalized_map = torch.stack([human1, human2, background], dim=1)
            normalized_pos = normalized_map[torch.arange(x_ref_attn_map.size(1), device=q.device), max_indices]
            q = rearrange(q, "(B N_t) S H D -> B H (N_t S) D", N_t=frames_num)
            q = self.rope_1d(q, normalized_pos)
            q = rearrange(q, "B H (N_t S) D -> (B N_t) S H D", N_t=frames_num)

        kv = self.kv_linear(cond).view(batch, cond.shape[1], 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = self.k_norm(k)

        if x_ref_attn_map is not None:
            per_frame = torch.zeros(cond.shape[1], dtype=k.dtype, device=k.device)
            human1_pos = (self.rope_h1[0] + self.rope_h1[1]) / 2
            human2_pos = (self.rope_h2[0] + self.rope_h2[1]) / 2
            if human_num is not None and human_num > 2:
                background_pos = self.rope_bak if x_ref_attn_map.shape[0] <= 3 else 100
                tokens_per_human = max(1, per_frame.size(0) // human_num)
                per_frame[:tokens_per_human] = human1_pos
                per_frame[tokens_per_human : 2 * tokens_per_human] = human2_pos
                per_frame[2 * tokens_per_human :] = background_pos
            else:
                midpoint = per_frame.size(0) // 2
                per_frame[:midpoint] = human1_pos
                per_frame[midpoint:] = human2_pos
            encoder_pos = torch.cat([per_frame] * frames_num, dim=0)
            k = rearrange(k, "(B N_t) S H D -> B H (N_t S) D", N_t=frames_num)
            k = self.rope_1d(k, encoder_pos)
            k = rearrange(k, "B H (N_t S) D -> (B N_t) S H D", N_t=frames_num)

        x = self.attn(q, k, v)
        x = self.proj(x.reshape(batch, seq_len, channels))
        x = self.proj_drop(x)
        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=frames_num)
        return x.to(out_dtype)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        shape: tuple[int, int, int] | None = None,
        num_cond_latents: int | None = None,
        x_ref_attn_map=None,
        human_num=None,
    ):
        if shape is None:
            raise ValueError("shape is required for LongCat-Video-Avatar audio attention.")
        batch, seq_len, channels = x.shape
        if num_cond_latents is None or num_cond_latents == 0:
            return None, self._process_cross_attn(x, cond, shape[0], x_ref_attn_map, human_num)

        num_cond_latents_thw = num_cond_latents * (seq_len // shape[0])
        x_noise = x[:, num_cond_latents_thw:]
        cond = rearrange(cond, "(B N_t) M C -> B N_t M C", B=batch)
        cond = cond[:, num_cond_latents:]
        cond = rearrange(cond, "B N_t M C -> (B N_t) M C")
        frames_num = shape[0] - num_cond_latents
        output_noise = self._process_cross_attn(x_noise, cond, frames_num, x_ref_attn_map, human_num)
        output_cond = torch.zeros(
            (batch, num_cond_latents_thw, channels),
            dtype=output_noise.dtype,
            device=output_noise.device,
        )
        return output_cond, output_noise


class AudioProjModel(nn.Module):
    """Project Whisper/Wav2Vec frame windows into Avatar audio context tokens."""

    def __init__(
        self,
        seq_len: int = 5,
        seq_len_vf: int = 12,
        blocks: int = 12,
        channels: int = 768,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        norm_output_audio: bool = True,
        enable_compile: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim
        self.enable_compile = enable_compile
        self.flops = 0.0

        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds: torch.Tensor, audio_embeds_vf: torch.Tensor) -> torch.Tensor:
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        batch, _, _, _, _ = audio_embeds.shape

        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))

        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=batch)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=batch)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)

        audio_embeds_c = audio_embeds_c.reshape(audio_embeds_c.shape[0] * audio_embeds_c.shape[1], -1)
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))
        context_tokens = self.proj3(audio_embeds_c).reshape(-1, self.context_tokens, self.output_dim)
        context_tokens = self.norm(context_tokens)
        return rearrange(context_tokens, "(bz f) m c -> bz f m c", bz=batch, f=video_length)


class QuantizedLinear(nn.Module):
    """INT8 weight-only quantized linear layer with per-channel scales."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.zeros(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = x.dtype
        weight = self.weight_int8.to(compute_dtype) * self.weight_scale.to(compute_dtype).unsqueeze(1)
        bias = self.bias.to(compute_dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


DEFAULT_SKIP_PATTERNS = {"final_layer.linear"}


def _read_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    signature = inspect.signature(LongCatVideoAvatarTransformer3DModel.__init__)
    allowed_keys = set(signature.parameters.keys())
    return {key: value for key, value in config.items() if key in allowed_keys}


def replace_linear_with_quantized(model: nn.Module, skip_patterns: set[str] | None = None) -> None:
    skip_patterns = skip_patterns or DEFAULT_SKIP_PATTERNS
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(pattern in name for pattern in skip_patterns):
            modules_to_replace[name] = QuantizedLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )

    for name, qlinear in modules_to_replace.items():
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], qlinear)


class LoRAUPParallel(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        if x.shape[-1] % len(self.blocks) != 0:
            raise ValueError("LoRAUPParallel input dim must be divisible by number of blocks.")
        xs = torch.chunk(x, len(self.blocks), dim=-1)
        return torch.cat([block(xs[idx]) for idx, block in enumerate(self.blocks)], dim=-1)


class LoRAModule(nn.Module):
    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        n_separate: int = 1,
    ):
        super().__init__()
        self.lora_name = lora_name
        if org_module.__class__.__name__ not in ("Linear", "QuantizedLinear"):
            raise TypeError(f"LoRA target must be Linear or QuantizedLinear, got {org_module.__class__.__name__}")

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.lora_dim = lora_dim
        if n_separate > 1:
            if out_dim % n_separate != 0:
                raise ValueError("LoRA separate blocks must evenly divide output dimension.")
            self.lora_down = nn.Linear(in_dim, n_separate * self.lora_dim, bias=False)
            self.lora_up = LoRAUPParallel(
                [nn.Linear(self.lora_dim, out_dim // n_separate, bias=False) for _ in range(n_separate)]
            )
        else:
            self.lora_down = nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = nn.Linear(self.lora_dim, out_dim, bias=False)

        if isinstance(alpha, torch.Tensor):
            alpha = float(alpha.detach().float().cpu().numpy())
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.register_buffer("alpha_scale", torch.tensor(alpha / self.lora_dim))
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        if n_separate > 1:
            for block in self.lora_up.blocks:
                nn.init.zeros_(block.weight)
        else:
            nn.init.zeros_(self.lora_up.weight)
        self.multiplier = multiplier
        self.use_lora = True


class LoRANetwork(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        lora_state_dict: dict[str, torch.Tensor],
        multiplier: float = 1.0,
        lora_dim: int = 128,
        alpha: float = 64,
    ):
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.loras: list[LoRAModule] = []

        lora_names = {key.split(".lora_down.weight")[0] for key in lora_state_dict if key.endswith("lora_down.weight")}
        for lora_name in sorted(lora_names):
            module_name = lora_name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
            try:
                module = model
                for part in module_name.split("."):
                    module = getattr(module, part)
            except AttributeError:
                continue
            if module.__class__.__name__ not in ("Linear", "QuantizedLinear"):
                continue
            prefix = lora_name + ".lora_up.blocks"
            n_blocks = sum(1 for key in lora_state_dict if key.startswith(prefix))
            lora = LoRAModule(
                lora_name,
                module,
                multiplier=multiplier,
                lora_dim=lora_dim,
                alpha=alpha,
                n_separate=n_blocks if n_blocks > 0 else 1,
            )
            self.loras.append(lora)
            self.add_module(lora_name, lora)


def create_lora_network(
    transformer: nn.Module,
    lora_state_dict: dict[str, torch.Tensor],
    multiplier: float,
    network_dim: int | None,
    network_alpha: float | None,
) -> LoRANetwork:
    network = LoRANetwork(
        transformer,
        lora_state_dict,
        multiplier=multiplier,
        lora_dim=network_dim or 128,
        alpha=network_alpha or 64,
    )
    network.load_state_dict(lora_state_dict, strict=True)
    return network


class LongCatAvatarSingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params=None,
        cp_split_hw=None,
        output_dim: int = 768,
        audio_prenorm: bool = True,
        class_range: int = 24,
        class_interval: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True))
        self.audio_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_tembed_dim, 3 * hidden_size, bias=True),
        )

        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)
        self.pre_video_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)
        self.pre_audio_crs_attn_norm = (
            LayerNorm_FP32(output_dim, eps=1e-6, elementwise_affine=True) if audio_prenorm else nn.Identity()
        )

        self.attn = AvatarSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw,
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
        )
        self.audio_cross_attn = SingleStreamAudioAttention(
            dim=hidden_size,
            encoder_hidden_states_dim=output_dim,
            num_heads=num_heads,
            qk_norm=True,
            qkv_bias=True,
            class_range=class_range,
            class_interval=class_interval,
            cp_split_hw=cp_split_hw,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
        )
        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))

    def forward(
        self,
        x,
        y,
        t,
        y_seqlen,
        latent_shape,
        num_cond_latents=None,
        return_kv=False,
        kv_cache=None,
        skip_crs_attn=False,
        num_ref_latents=None,
        audio_hidden_states=None,
        ref_img_index=None,
        mask_frame_range=None,
        token_ref_target_masks=None,
        human_num=None,
    ):
        x_dtype = x.dtype
        batch, seq_len, channels = x.shape
        num_frames, _, _ = latent_shape
        num_cond = int(num_cond_latents or 0)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            silu_linear_fp32(self.adaLN_modulation, t).unsqueeze(2).chunk(6, dim=-1)
        )
        x_m = modulate_fp32(self.mod_norm_attn, x.view(batch, num_frames, -1, channels), shift_msa, scale_msa)
        x_m = x_m.view(batch, seq_len, channels)
        if kv_cache is not None:
            attn_outputs = self.attn.forward_with_kv_cache(
                x_m,
                shape=latent_shape,
                num_cond_latents=num_cond_latents,
                kv_cache=kv_cache,
                num_ref_latents=num_ref_latents,
                ref_img_index=ref_img_index,
                mask_frame_range=mask_frame_range,
                ref_target_masks=token_ref_target_masks,
            )
        else:
            attn_outputs = self.attn(
                x_m,
                shape=latent_shape,
                num_cond_latents=num_cond_latents,
                return_kv=return_kv,
                num_ref_latents=num_ref_latents,
                ref_img_index=ref_img_index,
                mask_frame_range=mask_frame_range,
                ref_target_masks=token_ref_target_masks,
            )
        if return_kv:
            x_s, new_kv_cache, x_ref_attn_map = attn_outputs
        else:
            x_s, x_ref_attn_map = attn_outputs
        x = x + (gate_msa * x_s.view(batch, -1, seq_len // num_frames, channels)).view(batch, -1, channels)
        x = x.to(x_dtype)

        if not skip_crs_attn:
            cross_num_cond_latents = None if kv_cache is not None else num_cond_latents
            x = x + self.cross_attn(
                self.pre_crs_attn_norm(x),
                y,
                y_seqlen,
                num_cond_latents=cross_num_cond_latents,
                shape=latent_shape,
            )

            audio_num_cond = 0 if kv_cache is not None else num_cond
            audio_shift_mca, audio_scale_mca, audio_gate_mca = (
                silu_linear_fp32(self.audio_adaLN_modulation, t[:, audio_num_cond:]).unsqueeze(2).chunk(3, dim=-1)
            )
            audio_output_cond, audio_output_noise = self.audio_cross_attn(
                self.pre_video_crs_attn_norm(x),
                self.pre_audio_crs_attn_norm(audio_hidden_states),
                shape=latent_shape,
                num_cond_latents=audio_num_cond,
                x_ref_attn_map=x_ref_attn_map,
                human_num=human_num,
            )
            audio_output_noise = modulate_fp32(
                self.mod_norm_attn,
                audio_output_noise.view(batch, num_frames - audio_num_cond, -1, channels),
                audio_shift_mca,
                audio_scale_mca,
            ).view(batch, -1, channels)
            audio_add_x = (
                audio_gate_mca * audio_output_noise.view(batch, num_frames - audio_num_cond, -1, channels)
            ).view(batch, -1, channels)
            if audio_output_cond is not None:
                audio_add_x = torch.cat([audio_output_cond, audio_add_x], dim=1).contiguous()
            x = (x + audio_add_x).to(x_dtype)

        x_m = modulate_fp32(self.mod_norm_ffn, x.view(batch, -1, seq_len // num_frames, channels), shift_mlp, scale_mlp)
        x_m = x_m.view(batch, -1, channels)
        x_s = self.ffn(x_m)
        x = x + (gate_mlp * x_s.view(batch, -1, seq_len // num_frames, channels)).view(batch, -1, channels)
        if return_kv:
            return x.to(x_dtype), new_kv_cache
        return x.to(x_dtype)


class LongCatVideoAvatarTransformer3DModel(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        patch_size: tuple[int, int, int] | list[int] = (1, 2, 2),
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict | None = None,
        cp_split_hw: tuple[int, int] | list[int] | None = None,
        text_tokens_zero_pad: bool = False,
        audio_window: int = 5,
        audio_block: int = 12,
        audio_channel: int = 768,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        vae_scale: int = 4,
        audio_prenorm: bool = False,
        class_range: int = 24,
        class_interval: int = 4,
    ) -> None:
        super().__init__()
        if enable_bsa:
            raise NotImplementedError("LongCat-Video-Avatar BSA is not supported in native MVP.")
        if cp_split_hw not in (None, [1, 1], (1, 1)):
            raise NotImplementedError("LongCat-Video-Avatar context parallel is not supported in native MVP.")

        self.patch_size = tuple(patch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cp_split_hw = (1, 1)
        self.vae_scale = vae_scale
        self.audio_window = audio_window
        self.text_tokens_zero_pad = text_tokens_zero_pad
        self.gradient_checkpointing = False
        self.lora_dict = {}
        self.active_loras = []
        self.config = SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            caption_channels=caption_channels,
            mlp_ratio=mlp_ratio,
            adaln_tembed_dim=adaln_tembed_dim,
            frequency_embedding_size=frequency_embedding_size,
            patch_size=self.patch_size,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=None,
            text_tokens_zero_pad=text_tokens_zero_pad,
            audio_window=audio_window,
            audio_block=audio_block,
            audio_channel=audio_channel,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
            vae_scale=vae_scale,
            audio_prenorm=audio_prenorm,
            class_range=class_range,
            class_interval=class_interval,
        )

        self.x_embedder = PatchEmbed3D(self.patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(
            t_embed_dim=adaln_tembed_dim,
            frequency_embedding_size=frequency_embedding_size,
        )
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size)
        self.blocks = nn.ModuleList(
            [
                LongCatAvatarSingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_flashattn3=enable_flashattn3,
                    enable_flashattn2=enable_flashattn2,
                    enable_xformers=enable_xformers,
                    enable_bsa=enable_bsa,
                    bsa_params=bsa_params,
                    cp_split_hw=self.cp_split_hw,
                    output_dim=output_dim,
                    audio_prenorm=audio_prenorm,
                    class_range=class_range,
                    class_interval=class_interval,
                )
                for _ in range(depth)
            ]
        )
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            blocks=audio_block,
            channels=audio_channel,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
        )
        self.final_layer = FinalLayer_FP32(hidden_size, np.prod(self.patch_size), out_channels, adaln_tembed_dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.x_embedder.proj.weight.dtype

    def load_lora(self, lora_path, lora_key, multiplier=1.0, lora_network_dim=128, lora_network_alpha=64):
        lora_state_dict = load_file(lora_path, device="cpu")
        self.lora_dict[lora_key] = create_lora_network(
            self,
            lora_state_dict,
            multiplier=multiplier,
            network_dim=lora_network_dim,
            network_alpha=lora_network_alpha,
        )

    def enable_loras(self, lora_key_list=None):
        lora_key_list = lora_key_list or []
        self.disable_all_loras()
        module_loras = {}
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        for lora_key in lora_key_list:
            if lora_key not in self.lora_dict:
                continue
            for lora in self.lora_dict[lora_key].loras:
                lora.to(model_device, dtype=model_dtype, non_blocking=True)
                module_name = lora.lora_name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
                module_loras.setdefault(module_name, []).append(lora)
            self.active_loras.append(lora_key)

        # This mutates module.forward and is expected to run only during
        # single-threaded model initialization, before requests are served.
        for module_name, loras in module_loras.items():
            module = self._get_module_by_name(module_name)
            if not hasattr(module, "org_forward"):
                module.org_forward = module.forward
            module.forward = self._create_multi_lora_forward(module, loras)

    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x, *args, **kwargs):
            weight_dtype = x.dtype
            org_output = module.org_forward(x, *args, **kwargs)
            total_lora_output = 0
            for lora in loras:
                if lora.use_lora:
                    lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                    lx = lora.lora_up(lx)
                    total_lora_output = total_lora_output + lx.to(weight_dtype) * lora.multiplier * lora.alpha_scale
            return org_output + total_lora_output

        return multi_lora_forward

    def _get_module_by_name(self, module_name):
        module = self
        for part in module_name.split("."):
            module = getattr(module, part)
        return module

    def disable_all_loras(self):
        for module in self.modules():
            if hasattr(module, "org_forward"):
                module.forward = module.org_forward
                delattr(module, "org_forward")
        for lora_network in self.lora_dict.values():
            for lora in lora_network.loras:
                lora.to("cpu")
        self.active_loras.clear()

    def enable_bsa(self):
        raise NotImplementedError("LongCat-Video-Avatar BSA is not supported in native MVP.")

    def disable_bsa(self):
        return None

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        return_kv=False,
        kv_cache_dict=None,
        skip_crs_attn=False,
        offload_kv_cache=False,
        audio_embs=None,
        num_ref_latents=None,
        ref_img_index=None,
        mask_frame_range=None,
        ref_target_masks=None,
    ):
        if audio_embs is None:
            raise ValueError("LongCat-Video-Avatar requires audio_embs.")
        kv_cache_dict = kv_cache_dict or {}

        batch, _, num_frames, height, width = hidden_states.shape
        n_t = num_frames // self.patch_size[0]
        n_h = height // self.patch_size[1]
        n_w = width // self.patch_size[2]
        if self.patch_size[0] != 1:
            raise NotImplementedError("LongCat-Video-Avatar native MVP requires temporal patch size 1.")
        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, n_t)

        dtype = self.x_embedder.proj.weight.dtype
        hidden_states = self.x_embedder(hidden_states.to(dtype))
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)
        t = self.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(batch, n_t, -1)
        encoder_hidden_states = self.y_embedder(encoder_hidden_states)

        audio_cond = audio_embs.to(device=hidden_states.device, dtype=hidden_states.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...]
        latter_frame_audio_emb = audio_cond[:, 1:, ...]
        latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, : middle_index + 1, ...]
        latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
        latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index : middle_index + 1, ...]
        latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_frame_audio_emb_s = torch.concat(
            [latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb],
            dim=2,
        )
        audio_hidden_states = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
        if num_ref_latents is not None and num_ref_latents > 0:
            audio_start_ref = audio_hidden_states[:, [0], :, :]
            audio_hidden_states = torch.cat([audio_start_ref, audio_hidden_states], dim=1).contiguous()
        audio_hidden_states = audio_hidden_states[:, -n_t:]

        human_num = None
        if ref_target_masks is not None:
            if batch != 1:
                raise NotImplementedError("LongCat-Video-Avatar multi-speaker AI2V supports batch size 1.")
            human_num = len(audio_hidden_states)
            audio_hidden_states = torch.concat(audio_hidden_states.split(1), dim=2).squeeze(0)
        else:
            audio_hidden_states = rearrange(audio_hidden_states, "b t n c -> (b t) n c")

        token_ref_target_masks = None
        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0).to(device=hidden_states.device, dtype=torch.float32)
            token_ref_target_masks = F.interpolate(ref_target_masks, size=(n_h, n_w), mode="nearest")
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = (token_ref_target_masks > 0).view(token_ref_target_masks.shape[0], -1)
            token_ref_target_masks = token_ref_target_masks.to(dtype)

        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = (
                encoder_hidden_states.squeeze(1)
                .masked_select(encoder_attention_mask.unsqueeze(-1) != 0)
                .view(1, -1, hidden_states.shape[-1])
            )
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(1, -1, hidden_states.shape[-1])

        kv_cache_dict_ret = {}
        for idx, block in enumerate(self.blocks):
            block_outputs = block(
                hidden_states,
                encoder_hidden_states,
                t,
                y_seqlens,
                (n_t, n_h, n_w),
                num_cond_latents,
                return_kv,
                kv_cache_dict.get(idx),
                skip_crs_attn,
                num_ref_latents,
                audio_hidden_states,
                ref_img_index,
                mask_frame_range,
                token_ref_target_masks,
                human_num,
            )
            if return_kv:
                hidden_states, kv_cache = block_outputs
                if offload_kv_cache:
                    kv_cache_dict_ret[idx] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[idx] = (kv_cache[0].contiguous(), kv_cache[1].contiguous())
            else:
                hidden_states = block_outputs

        hidden_states = self.final_layer(hidden_states, t, (n_t, n_h, n_w))
        hidden_states = self.unpatchify(hidden_states, n_t, n_h, n_w)
        hidden_states = hidden_states.to(torch.float32)
        if return_kv:
            return hidden_states, kv_cache_dict_ret
        return hidden_states

    def unpatchify(self, x, n_t, n_h, n_w):
        t_p, h_p, w_p = self.patch_size
        return rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=n_t,
            N_h=n_h,
            N_w=n_w,
            T_p=t_p,
            H_p=h_p,
            W_p=w_p,
            C_out=self.out_channels,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


def create_quantized_avatar_dit(checkpoint_dir: str | os.PathLike[str], subfolder: str = "base_model_int8", **kwargs):
    quantized_dir = Path(checkpoint_dir) / subfolder
    config = _read_config(quantized_dir / "config.json")
    config.update(kwargs)
    model = LongCatVideoAvatarTransformer3DModel(**config)
    replace_linear_with_quantized(model)

    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            continue
        for param in module.parameters(recurse=False):
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
    return model.eval()


def create_full_precision_avatar_dit(checkpoint_dir: str | os.PathLike[str], subfolder: str = "base_model", **kwargs):
    model_dir = Path(checkpoint_dir) / subfolder
    config = _read_config(model_dir / "config.json")
    config.update(kwargs)
    model = LongCatVideoAvatarTransformer3DModel(**config)
    return model.eval()

# Copyright 2025 The HuggingFace Team and SANA-Video Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numbers
from collections.abc import Iterable
from dataclasses import dataclass, fields

import torch
from cache_dit import ForwardPattern
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.utils import set_weight_attrs

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention as OmniAttention
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig


def validate_sana_video_parallel_config(parallel_config) -> None:
    """Reject unsupported SANA-Video parallel topologies before weights load."""
    tp_size = parallel_config.tensor_parallel_size
    if tp_size not in (1, 2):
        raise NotImplementedError(
            f"SANA-Video supports tensor_parallel_size 1 or 2, got {tp_size}. Set --tensor-parallel-size to 1 or 2."
        )
    cfg_size = parallel_config.cfg_parallel_size
    if cfg_size not in (1, 2):
        raise NotImplementedError(
            f"SANA-Video supports cfg_parallel_size 1 or 2, got {cfg_size}. Set --cfg-parallel-size to 1 or 2."
        )
    sp_size = parallel_config.sequence_parallel_size
    if sp_size is not None and sp_size > 1:
        raise NotImplementedError(
            "Sequence parallel is not supported for SANA-Video: its linear attention and "
            "GLUMB temporal conv have no SANA-specific SP implementation. Use tensor and/or "
            "CFG parallel instead."
        )
    if parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError("SANA-Video does not support pipeline parallel. Set --pipeline-parallel-size to 1.")
    if parallel_config.use_hsdp:
        raise NotImplementedError("SANA-Video does not support HSDP. Remove --use-hsdp.")
    if parallel_config.text_encoder_tp_size > 1:
        raise NotImplementedError(
            "SANA-Video does not support text encoder tensor parallel. Set text_encoder_tp_size to 1."
        )


@dataclass
class SanaVideoTransformerConfig:
    in_channels: int = 16
    out_channels: int | None = 16
    num_attention_heads: int = 20
    attention_head_dim: int = 112
    num_layers: int = 20
    num_cross_attention_heads: int | None = 20
    cross_attention_head_dim: int | None = 112
    cross_attention_dim: int | None = 2240
    caption_channels: int = 2304
    mlp_ratio: float = 2.5
    dropout: float = 0.0
    attention_bias: bool = False
    sample_size: int = 30
    patch_size: tuple[int, int, int] = (1, 2, 2)
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-6
    interpolation_scale: int | None = None
    guidance_embeds: bool = False
    guidance_embeds_scale: float = 0.1
    qk_norm: str | None = "rms_norm_across_heads"
    rope_max_seq_len: int = 1024

    @classmethod
    def from_dict(cls, config: dict) -> "SanaVideoTransformerConfig":
        field_names = {field.name for field in fields(cls)}
        unknown = {key for key in config if key not in field_names and not key.startswith("_")}
        if unknown:
            raise ValueError(f"Unsupported SANA-Video transformer config fields: {sorted(unknown)}")
        values = {key: value for key, value in config.items() if key in field_names}
        if "patch_size" in values:
            values["patch_size"] = tuple(values["patch_size"])
        return cls(**values)


@dataclass
class SanaVideoTransformerOutput:
    sample: torch.Tensor


class SanaRMSNorm(nn.Module):
    """RMSNorm with the parameter and dtype semantics used by SANA checkpoints."""

    def __init__(
        self,
        dim: int | Iterable[int],
        eps: float,
        elementwise_affine: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        self.dim = torch.Size(dim)

        self.weight: nn.Parameter | None = None
        self.bias: nn.Parameter | None = None
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class SanaDistributedRMSNorm(nn.Module):
    """RMSNorm that computes global RMS across tensor parallel ranks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if param.shape == loaded_weight.shape:
            param.data.copy_(loaded_weight)
            return

        tp_size = get_tensor_model_parallel_world_size()
        if loaded_weight.shape[0] % tp_size != 0:
            raise ValueError(
                f"Cannot shard RMSNorm weight of shape {tuple(loaded_weight.shape)} across tp_size={tp_size}."
            )

        shard_size = loaded_weight.shape[0] // tp_size
        start_idx = get_tensor_model_parallel_rank() * shard_size
        shard = loaded_weight.narrow(0, start_idx, shard_size)
        if param.shape != shard.shape:
            raise ValueError(f"RMSNorm shard shape mismatch: param={tuple(param.shape)}, shard={tuple(shard.shape)}.")
        param.data.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        x_float = x.float()
        sum_sq = x_float.pow(2).sum(dim=-1, keepdim=True)
        count = x.shape[-1]

        if tp_size > 1:
            # Use vLLM's collective (custom all-reduce / symmetric-mem fast path
            # for small tensors) instead of raw torch.distributed.all_reduce, and
            # take the return value so the custom-AR path (which may return a new
            # buffer) is handled correctly. No .clone() needed.
            sum_sq = tensor_model_parallel_all_reduce(sum_sq)
            count = count * tp_size

        hidden_states = x_float * torch.rsqrt(sum_sq / count + self.eps)
        # Only cast to the weight dtype for fp16/bf16 weights; keep fp32 otherwise.
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create the sinusoidal timestep embeddings used by SANA checkpoints."""
    if timesteps.ndim != 1:
        raise ValueError(f"Timesteps must be a 1D tensor, got shape {tuple(timesteps.shape)}")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    embeddings = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    embeddings = scale * embeddings
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    if flip_sin_to_cos:
        embeddings = torch.cat([embeddings[:, half_dim:], embeddings[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        embeddings = torch.nn.functional.pad(embeddings, (0, 1, 0, 0))

    return embeddings


class SanaTimesteps(nn.Module):
    """Parameter-free timestep projection compatible with Diffusers Timesteps."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return _get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class SanaTimestepEmbedding(nn.Module):
    """Two-layer timestep MLP with Diffusers-compatible parameter names."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
    ) -> None:
        super().__init__()
        if act_fn != "silu":
            raise ValueError(f"SANA timestep embeddings only support act_fn='silu', got {act_fn!r}")
        if post_act_fn not in (None, "silu"):
            raise ValueError(f"SANA timestep embeddings only support post_act_fn=None or 'silu', got {post_act_fn!r}")

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=sample_proj_bias)
        self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False) if cond_proj_dim is not None else None
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(
            time_embed_dim,
            out_dim if out_dim is not None else time_embed_dim,
            bias=sample_proj_bias,
        )
        self.post_act = nn.SiLU() if post_act_fn == "silu" else None

    def forward(
        self,
        sample: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition is not None:
            if self.cond_proj is None:
                raise ValueError("condition was provided but cond_proj_dim was not configured")
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class SanaCombinedTimestepSizeEmbeddings(nn.Module):
    """PixArt timestep/size conditioning used by SANA's adaLN-single."""

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int,
        use_additional_conditions: bool = False,
    ) -> None:
        super().__init__()
        self.outdim = size_emb_dim
        self.time_proj = SanaTimesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_embedder = SanaTimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = SanaTimesteps(
                num_channels=256,
                flip_sin_to_cos=True,
                downscale_freq_shift=0,
            )
            self.resolution_embedder = SanaTimestepEmbedding(
                in_channels=256,
                time_embed_dim=size_emb_dim,
            )
            self.aspect_ratio_embedder = SanaTimestepEmbedding(
                in_channels=256,
                time_embed_dim=size_emb_dim,
            )

    def forward(
        self,
        timestep: torch.Tensor,
        resolution: torch.Tensor | None,
        aspect_ratio: torch.Tensor | None,
        batch_size: int | None,
        hidden_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))

        if not self.use_additional_conditions:
            return timesteps_emb
        if resolution is None or aspect_ratio is None or batch_size is None:
            raise ValueError(
                "resolution, aspect_ratio, and batch_size are required when use_additional_conditions=True"
            )

        resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
        resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
        aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
        aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
        return timesteps_emb + torch.cat([resolution_emb, aspect_ratio_emb], dim=1)


class SanaAdaLayerNormSingle(nn.Module):
    """SANA's checkpoint-compatible adaLN-single conditioning module."""

    def __init__(
        self,
        embedding_dim: int,
        use_additional_conditions: bool = False,
    ) -> None:
        super().__init__()
        self.emb = SanaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        batch_size: int | None = None,
        hidden_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        added_cond_kwargs = added_cond_kwargs or {
            "resolution": None,
            "aspect_ratio": None,
        }
        embedded_timestep = self.emb(
            timestep,
            **added_cond_kwargs,
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class _SanaFP32SiLU(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(inputs.float(), inplace=False).to(inputs.dtype)


class SanaPixArtAlphaTextProjection(nn.Module):
    """Caption projection with Diffusers-compatible parameters and activations."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
    ) -> None:
        super().__init__()
        out_features = hidden_size if out_features is None else out_features
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = _SanaFP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(hidden_size, out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        return self.linear_2(hidden_states)


def _get_1d_rotary_pos_embed(
    dim: int,
    max_seq_len: int,
    theta: float,
    freqs_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the real-valued 1D RoPE representation used by SANA-Video."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype) / dim))
    positions = torch.arange(max_seq_len, dtype=freqs_dtype)
    freqs = torch.outer(positions, freqs)
    freqs_cos = freqs.cos().float().repeat_interleave(2, dim=-1)
    freqs_sin = freqs.sin().float().repeat_interleave(2, dim=-1)
    return freqs_cos, freqs_sin


class GLUMBTempConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4,
        norm_type: str | None = None,
        residual_connection: bool = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection
        self.hidden_channels = hidden_channels

        tp_size = get_tensor_model_parallel_world_size()
        local_hidden_channels = hidden_channels // tp_size

        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, 2 * local_hidden_channels, 1, 1, 0)
        self.conv_depth = nn.Conv2d(
            2 * local_hidden_channels, 2 * local_hidden_channels, 3, 1, 1, groups=2 * local_hidden_channels
        )
        self.conv_point = nn.Conv2d(local_hidden_channels, out_channels, 1, 1, 0, bias=False)
        # conv_inverted/conv_depth carry [value(h), gate(h)]; each rank must hold a
        # slice of both segments so the local chunk(2) still splits value from gate.
        for param in (self.conv_inverted.weight, self.conv_inverted.bias, self.conv_depth.weight, self.conv_depth.bias):
            set_weight_attrs(param, {"weight_loader": self._segment_weight_loader})
        set_weight_attrs(self.conv_point.weight, {"weight_loader": self._point_weight_loader})

        self.norm = None
        if norm_type == "rms_norm":
            self.norm = SanaRMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True)

        self.conv_temp = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)

    def _segment_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        h = self.hidden_channels
        local = h // get_tensor_model_parallel_world_size()
        start = get_tensor_model_parallel_rank() * local
        value = loaded_weight.narrow(0, start, local)
        gate = loaded_weight.narrow(0, h + start, local)
        param.data.copy_(torch.cat([value, gate], dim=0))

    def _point_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        local = self.hidden_channels // get_tensor_model_parallel_world_size()
        start = get_tensor_model_parallel_rank() * local
        param.data.copy_(loaded_weight.narrow(1, start, local))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
            residual = hidden_states
        batch_size, num_frames, height, width, num_channels = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * num_frames, height, width, num_channels).permute(0, 3, 1, 2)

        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)

        hidden_states = self.conv_point(hidden_states)
        if get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        # Temporal aggregation
        hidden_states_temporal = hidden_states.view(batch_size, num_frames, num_channels, height * width).permute(
            0, 2, 1, 3
        )
        hidden_states = hidden_states_temporal + self.conv_temp(hidden_states_temporal)
        hidden_states = hidden_states.permute(0, 2, 3, 1).view(batch_size, num_frames, height, width, num_channels)

        if self.norm_type == "rms_norm":
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


class SanaLinearAttention(nn.Module):
    """SANA's ReLU-kernel linear self-attention.

    This is intentionally not routed through the unified softmax attention
    layer because doing so would change the model's attention algorithm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bias: bool,
        qk_norm: str | None,
    ) -> None:
        super().__init__()
        inner_dim = num_heads * head_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.heads = num_heads // tp_size
        local_inner_dim = inner_dim // tp_size
        self.to_q = ColumnParallelLinear(dim, inner_dim, bias=bias, gather_output=False, return_bias=False)
        self.to_k = ColumnParallelLinear(dim, inner_dim, bias=bias, gather_output=False, return_bias=False)
        self.to_v = ColumnParallelLinear(dim, inner_dim, bias=bias, gather_output=False, return_bias=False)
        self.norm_q = SanaDistributedRMSNorm(local_inner_dim, eps=1e-5) if qk_norm is not None else None
        self.norm_k = SanaDistributedRMSNorm(local_inner_dim, eps=1e-5) if qk_norm is not None else None
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(inner_dim, dim, bias=True, input_is_parallel=True, return_bias=False),
                nn.Dropout(dropout),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if rotary_emb is None:
            raise ValueError("SanaLinearAttention requires rotary_emb.")

        original_dtype = hidden_states.dtype
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.unflatten(2, (self.heads, -1))
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        # B,N,H,C

        query = torch.relu(query)
        key = torch.relu(key)

        def apply_rotary_emb(
            hidden_states: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
        ):
            x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
            cos = freqs_cos[..., 0::2]
            sin = freqs_sin[..., 1::2]
            out = torch.empty_like(hidden_states)
            out[..., 0::2] = x1 * cos - x2 * sin
            out[..., 1::2] = x1 * sin + x2 * cos
            return out.type_as(hidden_states)

        query_rotate = apply_rotary_emb(query, *rotary_emb)
        key_rotate = apply_rotary_emb(key, *rotary_emb)

        # B,H,C,N
        query = query.permute(0, 2, 3, 1)
        key = key.permute(0, 2, 3, 1)
        query_rotate = query_rotate.permute(0, 2, 3, 1)
        key_rotate = key_rotate.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)

        query_rotate, key_rotate, value = query_rotate.float(), key_rotate.float(), value.float()

        # Keep the unrotated denominator contraction in the input dtype. This
        # intentionally matches Diffusers 0.38.0's SanaLinearAttnProcessor3_0;
        # only the rotated numerator path is accumulated in FP32.
        z = 1 / (key.sum(dim=-1, keepdim=True).transpose(-2, -1) @ query + 1e-15)

        scores = torch.matmul(value, key_rotate.transpose(-1, -2))
        hidden_states = torch.matmul(scores, query_rotate)

        hidden_states = hidden_states * z
        # B,H,C,N
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = _get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype)
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


class SanaModulatedNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, scale_shift_table: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        shift, scale = (scale_shift_table[None, None] + temb[:, :, None].to(scale_shift_table.device)).unbind(dim=2)
        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class SanaCombinedTimestepGuidanceEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = SanaTimesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = SanaTimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.guidance_condition_proj = SanaTimesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.guidance_embedder = SanaTimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor = None, hidden_dtype: torch.dtype = None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        guidance_proj = self.guidance_condition_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=hidden_dtype))
        conditioning = timesteps_emb + guidance_emb

        return self.linear(self.silu(conditioning)), conditioning


class SanaCrossAttention(nn.Module):
    """SANA cross-attention backed by vLLM-Omni's unified attention layer."""

    def __init__(
        self,
        dim: int,
        cross_attention_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bias: bool,
        out_bias: bool,
        qk_norm: str | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        inner_dim = num_heads * head_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.heads = num_heads // tp_size
        self.head_dim = head_dim
        local_inner_dim = inner_dim // tp_size
        self.to_q = ColumnParallelLinear(dim, inner_dim, bias=bias, gather_output=False, return_bias=False)
        self.to_k = ColumnParallelLinear(
            cross_attention_dim, inner_dim, bias=bias, gather_output=False, return_bias=False
        )
        self.to_v = ColumnParallelLinear(
            cross_attention_dim, inner_dim, bias=bias, gather_output=False, return_bias=False
        )
        self.norm_q = SanaDistributedRMSNorm(local_inner_dim, eps=1e-5) if qk_norm is not None else None
        self.norm_k = SanaDistributedRMSNorm(local_inner_dim, eps=1e-5) if qk_norm is not None else None
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(inner_dim, dim, bias=out_bias, input_is_parallel=True, return_bias=False),
                nn.Dropout(dropout),
            ]
        )
        self.attn = OmniAttention(
            num_heads=self.heads,
            head_size=head_dim,
            num_kv_heads=self.heads,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
            role="cross",
            qkv_layout="BSND",
            prefix=prefix,
            skip_sequence_parallel=True,
            disable_kv_quant=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.unflatten(2, (self.heads, self.head_dim))
        key = key.unflatten(2, (self.heads, self.head_dim))
        value = value.unflatten(2, (self.heads, self.head_dim))

        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        if attention_mask is not None:
            # The Flash masked-varlen path applies one padding mask to Q/K/V,
            # while SANA's text mask describes only cross-attention K/V.
            # Use SDPA until Flash supports an independent key-padding mask.
            hidden_states = self.attn.sdpa_fallback.forward(query, key, value, attn_metadata)
        else:
            hidden_states = self.attn(query, key, value, None)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class SanaVideoTransformerBlock(nn.Module):
    r"""
    Transformer block introduced in [Sana-Video](https://huggingface.co/papers/2509.24695).
    """

    def __init__(
        self,
        dim: int = 2240,
        num_attention_heads: int = 20,
        attention_head_dim: int = 112,
        dropout: float = 0.0,
        num_cross_attention_heads: int | None = 20,
        cross_attention_head_dim: int | None = 112,
        cross_attention_dim: int | None = 2240,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_out_bias: bool = True,
        mlp_ratio: float = 3.0,
        qk_norm: str | None = "rms_norm_across_heads",
        rope_max_seq_len: int = 1024,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # 1. Self Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = SanaLinearAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            qk_norm=qk_norm,
        )

        # 2. Cross Attention
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn2 = None
        if cross_attention_dim is not None:
            self.attn2 = SanaCrossAttention(
                dim=dim,
                cross_attention_dim=cross_attention_dim,
                num_heads=num_cross_attention_heads,
                head_dim=cross_attention_head_dim,
                dropout=dropout,
                bias=True,
                out_bias=attention_out_bias,
                qk_norm=qk_norm,
                prefix=f"{prefix}.attn2" if prefix else "attn2",
            )

        # 3. Feed-forward
        self.ff = GLUMBTempConv(dim, dim, mlp_ratio, norm_type=None, residual_connection=False)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        frames: int = None,
        height: int = None,
        width: int = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None] + timestep.reshape(batch_size, timestep.shape[1], 6, -1)
        ).unbind(dim=2)

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        attn_output = self.attn1(norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = hidden_states + gate_msa * attn_output

        # 3. Cross Attention
        if self.attn2 is not None:
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states,
                encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        norm_hidden_states = norm_hidden_states.unflatten(1, (frames, height, width))
        ff_output = self.ff(norm_hidden_states)
        ff_output = ff_output.flatten(1, 3)
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states


class SanaVideoTransformer3DModel(nn.Module):
    r"""
    A 3D Transformer model introduced in [Sana-Video](https://huggingface.co/papers/2509.24695) family of models.

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `20`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `112`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of Transformer blocks to use.
        num_cross_attention_heads (`int`, *optional*, defaults to `20`):
            The number of heads to use for cross-attention.
        cross_attention_head_dim (`int`, *optional*, defaults to `112`):
            The number of channels in each head for cross-attention.
        cross_attention_dim (`int`, *optional*, defaults to `2240`):
            The number of channels in the cross-attention output.
        caption_channels (`int`, defaults to `2304`):
            The number of channels in the caption embeddings.
        mlp_ratio (`float`, defaults to `2.5`):
            The expansion ratio to use in the GLUMBConv layer.
        dropout (`float`, defaults to `0.0`):
            The dropout probability.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in the attention layer.
        sample_size (`int`, defaults to `32`):
            The base size of the input latent.
        patch_size (`int`, defaults to `1`):
            The size of the patches to use in the patch embedding layer.
        norm_elementwise_affine (`bool`, defaults to `False`):
            Whether to use elementwise affinity in the normalization layer.
        norm_eps (`float`, defaults to `1e-6`):
            The epsilon value for the normalization layer.
        qk_norm (`str`, *optional*, defaults to `None`):
            The normalization to use for the query and key.
    """

    _no_split_modules = ["SanaVideoTransformerBlock", "SanaModulatedNorm"]
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={
            "transformer_blocks": ForwardPattern.Pattern_3,
        },
    )
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int | None = 16,
        num_attention_heads: int = 20,
        attention_head_dim: int = 112,
        num_layers: int = 20,
        num_cross_attention_heads: int | None = 20,
        cross_attention_head_dim: int | None = 112,
        cross_attention_dim: int | None = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 30,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: int | None = None,
        guidance_embeds: bool = False,
        guidance_embeds_scale: float = 0.1,
        qk_norm: str | None = "rms_norm_across_heads",
        rope_max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        self.config = SanaVideoTransformerConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            caption_channels=caption_channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_size=sample_size,
            patch_size=tuple(patch_size),
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            interpolation_scale=interpolation_scale,
            guidance_embeds=guidance_embeds,
            guidance_embeds_scale=guidance_embeds_scale,
            qk_norm=qk_norm,
            rope_max_seq_len=rope_max_seq_len,
        )
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Additional condition embeddings
        if guidance_embeds:
            self.time_embed = SanaCombinedTimestepGuidanceEmbeddings(inner_dim)
        else:
            self.time_embed = SanaAdaLayerNormSingle(inner_dim)

        self.caption_projection = SanaPixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=inner_dim,
        )
        self.caption_norm = SanaRMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)

        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                SanaVideoTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=num_cross_attention_heads,
                    cross_attention_head_dim=cross_attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    prefix=f"transformer_blocks.{layer_idx}",
                )
                for layer_idx in range(num_layers)
            ]
        )

        # 4. Output blocks
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.norm_out = SanaModulatedNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, math.prod(patch_size) * out_channels)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_config(cls, config: dict | SanaVideoTransformerConfig) -> "SanaVideoTransformer3DModel":
        if isinstance(config, dict):
            config = SanaVideoTransformerConfig.from_dict(config)
        return cls(**{field.name: getattr(config, field.name) for field in fields(config)})

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        controlnet_block_samples: tuple[torch.Tensor] | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor, ...] | SanaVideoTransformerOutput:
        """
        The [`SanaVideoTransformer3DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, num_frames, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            guidance (`torch.Tensor`, *optional*):
                Guidance scale embedding.
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`.
            attention_mask (`torch.Tensor`, *optional*):
                Retained for Diffusers API compatibility. SANA linear self-attention does not consume this mask.
            controlnet_block_samples (`tuple` of `torch.Tensor`, *optional*):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return `SanaVideoTransformerOutput` instead of a plain tuple.

        Returns:
            If `return_dict` is True, a [`SanaVideoTransformerOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # SANA's linear self-attention does not consume ``attention_mask``;
        # retain the public argument for Diffusers-compatible call sites.
        if encoder_attention_mask is not None:
            if encoder_attention_mask.ndim != 2:
                raise ValueError(
                    "encoder_attention_mask must be a 2D padding mask with "
                    f"shape (batch_size, key_length); got {tuple(encoder_attention_mask.shape)}"
                )
            if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
                raise ValueError(
                    "encoder_attention_mask shape must match encoder_hidden_states: "
                    f"expected {tuple(encoder_hidden_states.shape[:2])}, "
                    f"got {tuple(encoder_attention_mask.shape)}"
                )
            if encoder_attention_mask.is_floating_point() or encoder_attention_mask.is_complex():
                raise TypeError(
                    "encoder_attention_mask must be a boolean or integer padding mask, "
                    "not a floating-point additive attention bias"
                )
            encoder_attention_mask = encoder_attention_mask.to(
                device=encoder_hidden_states.device,
                dtype=torch.bool,
            )

        # 1. Input
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep.flatten(), guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        timestep = timestep.view(batch_size, -1, timestep.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        for index_block, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                rotary_emb,
            )
            if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        # 3. Normalization
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)

        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return SanaVideoTransformerOutput(sample=output)

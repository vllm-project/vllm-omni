# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import (
    CombinedTimestepTextProjEmbeddings,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp_utils import is_transformer_block_module
from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput, SequenceParallelOutput
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard_with_padding
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.flux.flux_transformer import FeedForward

logger = init_logger(__name__)


class HunyuanVideo15PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCFHW -> BNC
        return hidden_states


class HunyuanVideo15AdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: int | None = None) -> None:
        super().__init__()
        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideo15TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, use_meanflow: bool = False):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_meanflow = use_meanflow
        self.time_proj_r = None
        self.timestep_embedder_r = None
        if use_meanflow:
            self.time_proj_r = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.timestep_embedder_r = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=timestep.dtype))

        if timestep_r is not None and self.timestep_embedder_r is not None:
            timesteps_proj_r = self.time_proj_r(timestep_r)
            timesteps_emb_r = self.timestep_embedder_r(timesteps_proj_r.to(dtype=timestep.dtype))
            timesteps_emb = timesteps_emb + timesteps_emb_r

        return timesteps_emb


class HunyuanVideo15RotaryPosEmbed(nn.Module):
    def __init__(self, patch_size: int, patch_size_t: int, rope_dim: list[int], theta: float = 256.0) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]

        axes_grids = []
        for i in range(len(rope_sizes)):
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")
        grid = torch.stack(grid, dim=0)  # [3, T', H', W']

        freqs = []
        for i in range(3):
            # use_real=False returns complex [seq, dim/2]; we extract cos/sin manually
            # to get half-dim tensors compatible with RotaryEmbedding (which doubles internally).
            freq_cis = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=False)
            freqs.append((freq_cis.real, freq_cis.imag))

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1).float()
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1).float()
        return freqs_cos, freqs_sin


class HunyuanVideo15IndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        # Not TP
        from diffusers.models.attention_processor import Attention as DiffusersAttention

        self.attn = DiffusersAttention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        from diffusers.models.attention import FeedForward as DiffusersFeedForward

        self.ff = DiffusersFeedForward(
            hidden_size, mult=mlp_width_ratio, activation_fn="linear-silu", dropout=mlp_drop_rate
        )

        self.norm_out = HunyuanVideo15AdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideo15IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideo15IndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideo15TokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideo15IndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states


class HunyuanVideo15ByT5TextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, out_features)
        self.act_fn = nn.GELU()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(encoder_hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        return hidden_states


class HunyuanVideo15ImageProjection(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_channels)
        self.linear_1 = nn.Linear(in_channels, in_channels)
        self.act_fn = nn.GELU()
        self.linear_2 = nn.Linear(in_channels, hidden_size)
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_in(image_embeds)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


class HunyuanVideo15Attention(nn.Module):
    """Dual-stream joint attention with TP optimization.

    Key difference from FluxAttention: RoPE is applied **only** to the video
    stream Q/K *before* concatenation with the encoder stream, whereas Flux
    applies RoPE to the concatenated Q/K.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = True,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-6,
        out_dim: int | None = None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            bias=bias,
        )

        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    self.inner_dim,
                    self.out_dim,
                    bias=out_bias,
                    input_is_parallel=True,
                    return_bias=False,
                ),
                nn.Identity(),  # placeholder for dropout (none used)
            ]
        )

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=self.added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                bias=added_proj_bias,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim,
                query_dim,
                bias=out_bias,
                input_is_parallel=True,
                return_bias=False,
            )

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        hidden_states_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_kv_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)

        if encoder_hidden_states is not None:
            encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
            add_q_size = self.add_kv_proj.num_heads * self.head_dim
            add_kv_size = self.add_kv_proj.num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size], dim=-1
            )

            encoder_query = encoder_query.unflatten(-1, (self.add_kv_proj.num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

        attn_metadata = None
        ctx = get_forward_context()
        if ctx.sp_active and encoder_hidden_states is not None:
            # Under Ulysses SP, encoder tokens are passed via joint_*
            # metadata so they can be head-sliced separately from the
            # all-to-all'd video tokens in UlyssesParallelAttention.
            attn_metadata = AttentionMetadata(
                joint_query=encoder_query,
                joint_key=encoder_key,
                joint_value=encoder_value,
                joint_strategy="rear",
            )
            if attention_mask is not None:
                attn_metadata.joint_attn_mask = attention_mask.bool()
            if hidden_states_mask is not None:
                attn_metadata.attn_mask = hidden_states_mask
            hidden_states = self.attn(query, key, value, attn_metadata)
        else:
            if encoder_hidden_states is not None:
                query = torch.cat([query, encoder_query], dim=1)
                key = torch.cat([key, encoder_key], dim=1)
                value = torch.cat([value, encoder_value], dim=1)

            if attention_mask is not None:
                seq_len = query.shape[1]
                # Pad mask to full sequence length (video + encoder tokens)
                # Keep mask 2D (batch_size, seq_len) - each attention backend
                # handles reshaping internally (flash_attn uses unpadding,
                # SDPA expands to 4D via _maybe_reshape_attn_mask).
                attention_mask = F.pad(attention_mask, (seq_len - attention_mask.shape[1], 0), value=True)
                attention_mask = attention_mask.bool()
                attn_metadata = AttentionMetadata(attn_mask=attention_mask)

            hidden_states = self.attn(query, key, value, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
                [hidden_states.shape[1] - encoder_hidden_states.shape[1], encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = self.to_out[0](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states


class HunyuanVideo15TransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = HunyuanVideo15Attention(
            query_dim=hidden_size,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=hidden_size, dim_out=hidden_size, mult=mlp_ratio)

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=hidden_size, dim_out=hidden_size, mult=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
        hidden_states_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
            hidden_states_mask=hidden_states_mask,
        )

        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class HunyuanVideo15Transformer3DModel(nn.Module):
    """
    HunyuanVideo-1.5 Transformer with TP-optimized dual-stream attention.

    Ported from diffusers ``HunyuanVideo15Transformer3DModel`` with vllm-omni
    tensor-parallel layers for the 54 main transformer blocks.
    """

    _repeated_blocks = ["HunyuanVideo15TransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    _hsdp_shard_conditions = [is_transformer_block_module]

    _sp_plan = {
        "rope": {
            0: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True, auto_pad=True),
            1: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True, auto_pad=True),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        in_channels: int = 65,
        out_channels: int = 32,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 54,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 3584,
        text_embed_2_dim: int = 1472,
        image_embed_dim: int = 1152,
        rope_theta: float = 256.0,
        rope_axes_dim: tuple[int, ...] = (16, 56, 56),
        target_size: int = 640,
        task_type: str = "i2v",
        use_meanflow: bool = False,
    ):
        super().__init__()

        # Allow tf_model_config to override num_layers
        model_config = od_config.tf_model_config
        num_layers = getattr(model_config, "num_layers", num_layers)

        self.parallel_config = od_config.parallel_config
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t

        self.x_embedder = HunyuanVideo15PatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.image_embedder = HunyuanVideo15ImageProjection(image_embed_dim, inner_dim)

        self.context_embedder = HunyuanVideo15TokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )
        self.context_embedder_2 = HunyuanVideo15ByT5TextProjection(text_embed_2_dim, 2048, inner_dim)

        self.time_embed = HunyuanVideo15TimeEmbedding(inner_dim, use_meanflow=use_meanflow)
        self.cond_type_embed = nn.Embedding(3, inner_dim)

        self.rope = HunyuanVideo15RotaryPosEmbed(patch_size, patch_size_t, list(rope_axes_dim), rope_theta)

        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideo15TransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * self.out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: torch.LongTensor | None = None,
        encoder_hidden_states_2: torch.Tensor | None = None,
        encoder_attention_mask_2: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        image_embeds_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size_t, self.patch_size, self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        image_rotary_emb = self.rope(hidden_states)

        temb = self.time_embed(timestep, timestep_r=timestep_r)

        hidden_states = self.x_embedder(hidden_states)

        # Scatter hidden_states along seq dim for sequence parallelism.
        # Done here (after x_embedder, before transformer_blocks) so that
        # CacheDiT sees already-sharded hidden_states when saving
        # original_hidden_states at the start of CachedBlocks.forward.
        if get_sequence_parallel_world_size() > 1:
            hidden_states, _pad_size = sp_shard_with_padding(hidden_states, dim=1)
            if _pad_size > 0:
                ctx = get_forward_context()
                if ctx.sp_original_seq_len is None:
                    ctx.sp_padding_size = _pad_size
                    ctx.sp_original_seq_len = hidden_states.shape[1] * get_sequence_parallel_world_size() - _pad_size

        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        encoder_hidden_states_cond_emb = self.cond_type_embed(
            torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_cond_emb

        encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)

        encoder_hidden_states_2_cond_emb = self.cond_type_embed(
            torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states_2 = encoder_hidden_states_2 + encoder_hidden_states_2_cond_emb

        encoder_hidden_states_3 = self.image_embedder(image_embeds)
        if image_embeds_mask is not None:
            encoder_attention_mask_3 = image_embeds_mask
        else:
            # Fallback: detect T2V by checking if image_embeds are all zeros
            is_t2v = torch.all(image_embeds == 0)
            if is_t2v:
                encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
                encoder_attention_mask_3 = torch.zeros(
                    (batch_size, encoder_hidden_states_3.shape[1]),
                    dtype=encoder_attention_mask.dtype,
                    device=encoder_attention_mask.device,
                )
            else:
                encoder_attention_mask_3 = torch.ones(
                    (batch_size, encoder_hidden_states_3.shape[1]),
                    dtype=encoder_attention_mask.dtype,
                    device=encoder_attention_mask.device,
                )
        encoder_hidden_states_3_cond_emb = self.cond_type_embed(
            2 * torch.ones_like(encoder_hidden_states_3[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states_3 = encoder_hidden_states_3 + encoder_hidden_states_3_cond_emb

        # Token reordering: [valid_image, valid_byte5, valid_mllm, padding]
        encoder_attention_mask = encoder_attention_mask.bool()
        encoder_attention_mask_2 = encoder_attention_mask_2.bool()
        encoder_attention_mask_3 = encoder_attention_mask_3.bool()
        new_encoder_hidden_states = []
        new_encoder_attention_mask = []

        for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            encoder_hidden_states_3,
            encoder_attention_mask_3,
        ):
            new_encoder_hidden_states.append(
                torch.cat(
                    [
                        image[image_mask],
                        text_2[text_mask_2],
                        text[text_mask],
                        image[~image_mask],
                        torch.zeros_like(text_2[~text_mask_2]),
                        torch.zeros_like(text[~text_mask]),
                    ],
                    dim=0,
                )
            )
            new_encoder_attention_mask.append(
                torch.cat(
                    [
                        image_mask[image_mask],
                        text_mask_2[text_mask_2],
                        text_mask[text_mask],
                        image_mask[~image_mask],
                        text_mask_2[~text_mask_2],
                        text_mask[~text_mask],
                    ],
                    dim=0,
                )
            )

        encoder_hidden_states = torch.stack(new_encoder_hidden_states)
        encoder_attention_mask = torch.stack(new_encoder_attention_mask)

        # Create explicit attn_mask for image tokens when SP auto_pad is active.
        ctx = get_forward_context()
        hidden_states_mask = None
        if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
            padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
            hidden_states_mask = torch.ones(
                batch_size,
                padded_seq_len,
                dtype=torch.bool,
                device=hidden_states.device,
            )
            hidden_states_mask[:, ctx.sp_original_seq_len :] = False

            # if mask is all true, set it to None
            if hidden_states_mask.all():
                hidden_states_mask = None

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                encoder_attention_mask,
                image_rotary_emb,
                hidden_states_mask=hidden_states_mask,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self-attn
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            # cross-attn
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        # Load buffers for beta and eps (if any normalizations use them)
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                if lookup_name not in params_dict:
                    break
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle .to_out.0. -> .to_out. remapping
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params

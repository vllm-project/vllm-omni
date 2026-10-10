# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PAN2 video generation transformer, ported from diffusers ``PAN2Transformer3DModel``."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from cache_dit import ForwardPattern
from diffusers.models.attention import FeedForward as DiffusersFeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp_utils import is_transformer_block_module
from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput, SequenceParallelOutput
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard_with_padding
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
    _fused_cuda_supported,
    fused_qk_norm_rope,
    fused_qk_norm_rope_min_tokens,
)
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.flux.flux_transformer import FeedForward

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)

# Fuse the video Q/K RMSNorm + RoPE only when a call covers at least this many tokens; below it the host launch
# overhead dominates. Override with VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS (0 = always fuse).
_FUSED_MIN_TOKENS = 2048


class PAN2RotaryPosEmbed(nn.Module):
    """3-axis ``(t, h, w)`` rotary embedding over the video tokens, returned as half-dim cos/sin."""

    def __init__(
        self, patch_size: int, patch_size_t: int, rope_dim: list[int], theta: float = 256.0, temporal_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta
        self.temporal_scale = temporal_scale

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, num_frames, height, width = hidden_states.shape
        rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]

        axes_grids = [torch.arange(0, size, device=hidden_states.device, dtype=torch.float32) for size in rope_sizes]
        grid = torch.stack(torch.meshgrid(*axes_grids, indexing="ij"), dim=0)
        # Only the frame positions are rescaled, to the frame rate the rotary embedding was trained at.
        grid[0] = grid[0] * self.temporal_scale

        freqs = []
        for i in range(3):
            # use_real=False returns the complex half-dim frequencies expected by RotaryEmbedding.
            freqs_cis = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=False)
            freqs.append((freqs_cis.real, freqs_cis.imag))

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1).float()
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1).float()
        return freqs_cos, freqs_sin


class PAN2RefinerAttention(nn.Module):
    """Self-attention of the text token refiner. The text is replicated across sequence-parallel ranks."""

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(dim, heads * dim_head, bias=False)
        self.norm_q = DiffusersRMSNorm(dim_head, eps=eps)
        self.norm_k = DiffusersRMSNorm(dim_head, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, dim, bias=False), nn.Identity()])
        self.attn = Attention(
            num_heads=heads,
            head_size=dim_head,
            softmax_scale=1.0 / (dim_head**0.5),
            causal=False,
            num_kv_heads=heads,
            role="pan2.text_refiner",
            role_category="self",
            skip_sequence_parallel=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

        query = self.norm_q(query).to(value.dtype)
        key = self.norm_k(key).to(value.dtype)

        hidden_states = self.attn(query, key, value, None)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        return self.to_out[0](hidden_states)


class PAN2TokenRefinerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.norm1 = DiffusersRMSNorm(dim, eps=eps)
        self.attn = PAN2RefinerAttention(dim, num_attention_heads, attention_head_dim, eps)
        self.norm2 = DiffusersRMSNorm(dim, eps=eps)
        self.ff = DiffusersFeedForward(dim, inner_dim=int(dim * mlp_ratio), activation_fn="swiglu", bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class PAN2Attention(nn.Module):
    """Joint attention over ``[video; text]`` with per-stream, tensor-parallel projections.

    RoPE is applied to the video stream only, before it is joined with the text stream.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.head_dim = dim_head

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=dim_head,
            total_num_heads=heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv",
        )
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    heads * dim_head,
                    dim,
                    bias=True,
                    input_is_parallel=True,
                    return_bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                ),
                nn.Identity(),
            ]
        )

        self.add_kv_proj = QKVParallelLinear(
            hidden_size=dim,
            head_size=dim_head,
            total_num_heads=heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.add_kv_proj",
        )
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)
        self.to_add_out = RowParallelLinear(
            heads * dim_head,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_add_out",
        )

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=dim_head,
            softmax_scale=1.0 / (dim_head**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def _project(
        self, projection: QKVParallelLinear, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, _ = projection(hidden_states.contiguous())
        q_size = projection.num_heads * self.head_dim
        kv_size = projection.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
        return (
            query.unflatten(-1, (projection.num_heads, -1)),
            key.unflatten(-1, (projection.num_kv_heads, -1)),
            value.unflatten(-1, (projection.num_kv_heads, -1)),
        )

    def _video_qk_norm_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head Q/K RMSNorm + RoPE of the video stream, in one kernel when a packed RoPE table is present."""
        cos, sin, rope_table = image_rotary_emb
        head_dim = query.shape[-1]
        if rope_table is not None and _fused_cuda_supported(query, key, head_dim, head_dim, interleaved=True):
            batch_size, seq_len, num_heads, _ = query.shape
            num_kv_heads = key.shape[2]
            rope_table = rope_table.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * seq_len, -1)
            query, key = fused_qk_norm_rope(
                query.reshape(batch_size * seq_len, num_heads, head_dim),
                key.reshape(batch_size * seq_len, num_kv_heads, head_dim),
                self.norm_q.weight,
                self.norm_k.weight,
                rope_table,
                self.norm_q.variance_epsilon,
                interleaved=True,
            )
            return (
                query.view(batch_size, seq_len, num_heads, head_dim),
                key.view(batch_size, seq_len, num_kv_heads, head_dim),
            )

        query = self.norm_q(query).to(query.dtype)
        key = self.norm_k(key).to(key.dtype)
        query = self.rope(query, cos.to(query.dtype), sin.to(query.dtype))
        key = self.rope(key, cos.to(key.dtype), sin.to(key.dtype))
        return query, key

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        hidden_states_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, key, value = self._project(self.to_qkv, hidden_states)
        query, key = self._video_qk_norm_rope(query, key, image_rotary_emb)

        encoder_query, encoder_key, encoder_value = self._project(self.add_kv_proj, encoder_hidden_states)
        encoder_query = self.norm_added_q(encoder_query).to(encoder_value.dtype)
        encoder_key = self.norm_added_k(encoder_key).to(encoder_value.dtype)

        if get_forward_context().sp_active:
            # Under Ulysses SP the text tokens are replicated on every rank; they travel as joint tensors so they
            # are head-sliced rather than all-to-all'd with the sharded video tokens.
            attn_metadata = AttentionMetadata(
                joint_query=encoder_query,
                joint_key=encoder_key,
                joint_value=encoder_value,
                joint_strategy="rear",
            )
            if hidden_states_mask is not None:
                attn_metadata.attn_mask = hidden_states_mask
            hidden_states = self.attn(query, key, value, attn_metadata)
        else:
            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)
            hidden_states = self.attn(query, key, value, None)

        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            [hidden_states.shape[1] - encoder_hidden_states.shape[1], encoder_hidden_states.shape[1]], dim=1
        )
        return self.to_out[0](hidden_states), self.to_add_out(encoder_hidden_states)


class PAN2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        context_mlp_ratio: float,
        eps: float = 1e-6,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim, norm_type="layer_norm")
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.scale_shift_table_context = nn.Parameter(torch.zeros(6 * dim))

        self.attn = PAN2Attention(
            dim, num_attention_heads, attention_head_dim, eps, quant_config=quant_config, prefix=f"{prefix}.attn"
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = FeedForward(dim, inner_dim=int(dim * mlp_ratio), quant_config=quant_config, prefix=f"{prefix}.ff")
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = FeedForward(
            dim, inner_dim=int(dim * context_mlp_ratio), quant_config=quant_config, prefix=f"{prefix}.ff_context"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        hidden_states_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        (
            context_shift_msa,
            context_scale_msa,
            context_gate_msa,
            context_shift_mlp,
            context_scale_mlp,
            context_gate_mlp,
        ) = self.scale_shift_table_context.to(temb.dtype).chunk(6, dim=-1)
        norm_encoder_hidden_states = (
            self.norm1_context(encoder_hidden_states) * (1 + context_scale_msa) + context_shift_msa
        )

        attn_output, context_attn_output = self.attn(
            norm_hidden_states, norm_encoder_hidden_states, image_rotary_emb, hidden_states_mask
        )
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * context_gate_msa

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = (
            self.norm2_context(encoder_hidden_states) * (1 + context_scale_mlp) + context_shift_mlp
        )

        hidden_states = hidden_states + self.ff(norm_hidden_states) * gate_mlp.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + self.ff_context(norm_encoder_hidden_states) * context_gate_mlp
        return hidden_states, encoder_hidden_states


class PAN2Transformer3DModel(nn.Module):
    """PAN2 video generation transformer with tensor-parallel joint-attention blocks.

    The video input stacks the noisy latents, the conditioning latents and a one-channel conditioning mask along
    channels, so the same model serves text-to-video and image-to-video.
    """

    # The pipeline switches has_separate_cfg off when CFG parallelism runs one CFG branch per rank.
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={"transformer_blocks": ForwardPattern.Pattern_0},
        has_separate_cfg=True,
    )
    _repeated_blocks = ["PAN2TransformerBlock"]
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
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 97,
        out_channels: int = 48,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 44,
        num_refiner_layers: int = 4,
        mlp_ratio: float = 8.0,
        context_mlp_ratio: float = 4.0,
        refiner_mlp_ratio: float = 4.0,
        text_embed_dim: int = 4096,
        rope_axes_dim: tuple[int, int, int] = (16, 56, 56),
        rope_theta: float = 256.0,
        rope_temporal_scale: float = 16 / 24,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.parallel_config = od_config.parallel_config
        self.patch_size = tuple(patch_size)
        inner_dim = num_attention_heads * attention_head_dim

        self.x_embedder = nn.Conv3d(in_channels, inner_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.rope = PAN2RotaryPosEmbed(
            self.patch_size[1], self.patch_size[0], list(rope_axes_dim), rope_theta, rope_temporal_scale
        )

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)

        self.context_embedder = nn.Sequential(
            DiffusersRMSNorm(text_embed_dim, eps=1e-5), nn.Linear(text_embed_dim, inner_dim)
        )
        self.context_refiner = nn.ModuleList(
            [
                PAN2TokenRefinerBlock(inner_dim, num_attention_heads, attention_head_dim, refiner_mlp_ratio, eps=1e-5)
                for _ in range(num_refiner_layers)
            ]
        )
        self.context_type_embedding = nn.Parameter(torch.zeros(inner_dim))

        self.transformer_blocks = nn.ModuleList(
            [
                PAN2TransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio,
                    context_mlp_ratio,
                    quant_config=quant_config,
                    prefix=f"transformer_blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(
            inner_dim, self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * out_channels
        )

    def _sp_padding_mask(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        """Mask for the video padding sequence parallelism adds, when ``mask_sp_padding`` asks for it."""
        ctx = get_forward_context()
        if ctx.sp_original_seq_len is None or ctx.sp_padding_size == 0:
            return None
        if not self.parallel_config.mask_sp_padding:
            logger.warning_once(
                "SP auto-padding added %d video token(s) that are not masked from attention "
                "(mask_sp_padding=False); set parallel_config.mask_sp_padding=True for strict masking.",
                ctx.sp_padding_size,
            )
            return None
        hidden_states_mask = torch.ones(
            batch_size, ctx.sp_original_seq_len + ctx.sp_padding_size, dtype=torch.bool, device=device
        )
        hidden_states_mask[:, ctx.sp_original_seq_len :] = False
        return hidden_states_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        image_rotary_emb = self.rope(hidden_states)

        hidden_states = self.x_embedder(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

        hidden_states_mask = None
        if get_sequence_parallel_world_size() > 1:
            hidden_states, pad_size = sp_shard_with_padding(hidden_states, dim=1)
            if pad_size > 0:
                ctx = get_forward_context()
                if ctx.sp_original_seq_len is None:
                    ctx.sp_padding_size = pad_size
                    ctx.sp_original_seq_len = hidden_states.shape[1] * get_sequence_parallel_world_size() - pad_size
                hidden_states_mask = self._sp_padding_mask(batch_size, hidden_states.device)

        # The fused video Q/K norm + RoPE reads cos | sin packed into one table; build it once for all blocks.
        cos, sin = image_rotary_emb
        rope_table = None
        if batch_size * cos.shape[0] >= fused_qk_norm_rope_min_tokens(_FUSED_MIN_TOKENS):
            rope_table = torch.cat((cos, sin), dim=-1)
        image_rotary_emb = (cos, sin, rope_table)

        temb = self.time_embedder(self.time_proj(timestep).to(hidden_states.dtype))

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        for block in self.context_refiner:
            encoder_hidden_states = block(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + self.context_type_embedding

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, image_rotary_emb, hidden_states_mask
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                fused_name = name.replace(weight_name, param_name)
                # The token refiner keeps separate, non-parallel q/k/v projections.
                if fused_name not in params_dict:
                    continue
                param = params_dict[fused_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(fused_name)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params

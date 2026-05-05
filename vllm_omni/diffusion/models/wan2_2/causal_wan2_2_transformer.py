# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Causal WanVideo Transformer with block-wise causal attention and KV cache.
# Based on CausVid (arXiv:2412.07772) and adapted from sglang's implementation.

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.models.wan2_2.kv_cache import (
    CrossAttentionKVCache,
    SelfAttentionKVCache,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    AdaLayerNorm,
    DistributedRMSNorm,
    WanCrossAttention,
    WanFeedForward,
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    apply_rotary_emb_wan,
)

logger = init_logger(__name__)

# Compile flex_attention with max-autotune for Wan 1.3B compatibility
# See https://github.com/pytorch/pytorch/issues/133254
_compiled_flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


class CausalWanSelfAttention(nn.Module):
    """
    Block-wise causal self-attention with TP support.

    Training: uses flex_attention with BlockMask for block-wise causal masking.
    Inference: uses sliding-window KV cache with sink tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # Fused QKV projection with TP
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            bias=True,
        )
        self.local_num_heads = self.to_qkv.num_heads
        self.local_num_kv_heads = self.to_qkv.num_kv_heads
        self.tp_inner_dim = self.local_num_heads * head_dim

        # QK normalization (TP-aware)
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps)
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps)

        # Output projection with TP
        self.to_out = RowParallelLinear(
            num_heads * head_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
        )

        # Local attention for inference mode (no SP)
        self.attn = Attention(
            num_heads=self.local_num_heads,
            head_size=head_dim,
            num_kv_heads=self.local_num_kv_heads,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask | None = None,
        kv_cache: dict | SelfAttentionKVCache | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        if cache_start is None:
            cache_start = current_start

        # Fused QKV projection
        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.local_num_heads * self.head_dim
        kv_size = self.local_num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Reshape: [B, S, local_heads * head_dim] -> [B, S, local_heads, head_dim]
        query = query.unflatten(2, (self.local_num_heads, self.head_dim))
        key = key.unflatten(2, (self.local_num_kv_heads, self.head_dim))
        value = value.unflatten(2, (self.local_num_kv_heads, self.head_dim))

        # Apply rotary embeddings
        freqs_cos, freqs_sin = rotary_emb
        query = apply_rotary_emb_wan(query, freqs_cos, freqs_sin)
        key = apply_rotary_emb_wan(key, freqs_cos, freqs_sin)

        if isinstance(kv_cache, SelfAttentionKVCache):
            out = self._forward_sa_kv_cache(query, key, value, kv_cache, current_start)
        elif kv_cache is not None:
            out = self._forward_kv_cache(query, key, value, kv_cache, current_start, cache_start)
        elif block_mask is not None:
            out = self._forward_flex_attention(query, key, value, block_mask)
        else:
            out = self.attn(query, key, value)

        # Output projection
        out = out.flatten(2, 3).type_as(query)
        out = self.to_out(out)
        return out

    def _forward_sa_kv_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: SelfAttentionKVCache,
        current_start: int,
    ) -> torch.Tensor:
        """Inference path using SelfAttentionKVCache with sliding window eviction."""
        kv_cache.append(key, value, current_start)
        active_k, active_v = kv_cache.get_active_kv(self.max_attention_size)
        return self.attn(query, active_k, active_v)

    def _forward_flex_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: BlockMask | None,
    ) -> torch.Tensor:
        """Training path: flex_attention with block-wise causal mask."""
        seq_len = query.shape[1]
        padded_length = math.ceil(seq_len / 128) * 128 - seq_len

        if padded_length > 0:
            pad_shape_q = [query.shape[0], padded_length, query.shape[2], query.shape[3]]
            pad_shape_kv = [key.shape[0], padded_length, key.shape[2], key.shape[3]]
            query = torch.cat([query, query.new_zeros(pad_shape_q)], dim=1)
            key = torch.cat([key, key.new_zeros(pad_shape_kv)], dim=1)
            value = torch.cat([value, value.new_zeros(pad_shape_kv)], dim=1)

        # flex_attention expects [B, H, S, D]
        out = _compiled_flex_attention(
            query=query.transpose(1, 2),
            key=key.transpose(1, 2),
            value=value.transpose(1, 2),
            block_mask=block_mask,
        ).transpose(1, 2)

        if padded_length > 0:
            out = out[:, :seq_len]

        return out

    def _forward_kv_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: dict,
        current_start: int,
        cache_start: int,
    ) -> torch.Tensor:
        """Inference path: sliding-window KV cache with sink tokens."""
        frame_seqlen = query.shape[1]
        current_end = current_start + query.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = query.shape[1]

        if (
            self.local_attn_size != -1
            and (current_end > kv_cache["global_end_index"].item())
            and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)
        ):
            # Evict old tokens, preserving sink tokens
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            src_start = sink_tokens + num_evicted_tokens
            src_end = src_start + num_rolled_tokens
            dst_start = sink_tokens
            dst_end = dst_start + num_rolled_tokens
            kv_cache["k"][:, dst_start:dst_end] = kv_cache["k"][:, src_start:src_end].clone()
            kv_cache["v"][:, dst_start:dst_end] = kv_cache["v"][:, src_start:src_end].clone()

            local_end_index = (
                kv_cache["local_end_index"].item()
                + current_end
                - kv_cache["global_end_index"].item()
                - num_evicted_tokens
            )
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = key
            kv_cache["v"][:, local_start_index:local_end_index] = value
        else:
            # Direct assignment without eviction
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"] = kv_cache["k"].detach()
            kv_cache["v"] = kv_cache["v"].detach()
            kv_cache["k"][:, local_start_index:local_end_index] = key
            kv_cache["v"][:, local_start_index:local_end_index] = value

        # Attend to recent window only
        attn_start = max(0, local_end_index - self.max_attention_size)
        cached_k = kv_cache["k"][:, attn_start:local_end_index]
        cached_v = kv_cache["v"][:, attn_start:local_end_index]

        out = self.attn(query, cached_k, cached_v)

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        return out


class CausalWanTransformerBlock(nn.Module):
    """
    Transformer block with block-wise causal self-attention, cross-attention, and FFN.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        cross_attn_norm: bool = False,
    ):
        super().__init__()

        # 1. Self-attention with causal masking
        self.norm1 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn1 = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            eps=eps,
        )

        # 2. Cross-attention (reuse existing TP-enabled implementation)
        self.attn2 = WanCrossAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward (reuse existing TP-enabled implementation)
        self.ffn = WanFeedForward(dim=dim, inner_dim=ffn_dim, dim_out=dim)
        self.norm3 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)

        # Scale-shift table for modulation
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask | None = None,
        kv_cache: dict | SelfAttentionKVCache | None = None,
        crossattn_cache: CrossAttentionKVCache | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # TI2V mode: [batch, seq_len, 6, inner_dim]
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # Standard mode: [batch, 6, inner_dim]
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states, scale_msa, shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            rotary_emb,
            block_mask=block_mask,
            kv_cache=kv_cache,
            current_start=current_start,
            cache_start=cache_start,
        )
        hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention (with optional caching)
        norm_hidden_states = self.norm2(hidden_states).type_as(hidden_states)
        if crossattn_cache is not None and crossattn_cache.is_initialized:
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
        else:
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
            if crossattn_cache is not None:
                crossattn_cache.update(encoder_hidden_states, encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states, c_scale_msa, c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)

        return hidden_states


class CausalWanTransformer3DModel(nn.Module):
    """
    Causal WanVideo Transformer with block-wise causal attention and KV cache.
    Supports TP via vLLM parallel layers.

    Compared to WanTransformer3DModel:
    - Uses block-wise causal attention instead of bidirectional
    - Supports KV cache for frame-by-frame autoregressive inference
    - Supports sliding-window attention with sink tokens
    - Does NOT support Sequence Parallelism (TP only)
    """

    _repeated_blocks = ["CausalWanTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["blocks"]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return "blocks" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
        # Causal-specific parameters
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        head_dim = attention_head_dim

        self.config = type(
            "Config",
            (),
            {
                "patch_size": patch_size,
                "num_attention_heads": num_attention_heads,
                "attention_head_dim": attention_head_dim,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "text_dim": text_dim,
                "freq_dim": freq_dim,
                "ffn_dim": ffn_dim,
                "num_layers": num_layers,
                "cross_attn_norm": cross_attn_norm,
                "eps": eps,
                "image_dim": image_dim,
                "added_kv_proj_dim": added_kv_proj_dim,
                "rope_max_seq_len": rope_max_seq_len,
                "pos_embed_seq_len": pos_embed_seq_len,
                "local_attn_size": local_attn_size,
                "sink_size": sink_size,
                "num_frame_per_block": num_frame_per_block,
            },
        )()

        self.inner_dim = inner_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.local_attn_size = local_attn_size
        self.num_frame_per_block = num_frame_per_block
        assert num_frame_per_block <= 3

        # 1. Patch & position embedding
        from vllm.model_executor.layers.conv import Conv3dLayer

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = Conv3dLayer(
            in_channels=in_channels,
            out_channels=inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                CausalWanTransformerBlock(
                    dim=inner_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_attention_heads,
                    head_dim=head_dim,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    eps=eps,
                    added_kv_proj_dim=added_kv_proj_dim,
                    cross_attn_norm=cross_attn_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = AdaLayerNorm(inner_dim, elementwise_affine=False, eps=eps)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # Causal state
        self.block_mask: BlockMask | None = None
        self.gradient_checkpointing = False

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
    ) -> BlockMask:
        """
        Build a block-wise causal attention mask for flex_attention.

        Divides the token sequence into blocks of `num_frame_per_block` frames.
        Each block can attend to itself and all prior blocks.
        """
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )
        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (
                    q_idx == kv_idx
                )

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(
                "Cached block-wise causal mask: block_size=%d frames, shape=%s",
                num_frame_per_block,
                block_mask,
            )

        return block_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: list[dict] | list[SelfAttentionKVCache] | None = None,
        crossattn_cache: list[CrossAttentionKVCache] | None = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
    ) -> torch.Tensor | Transformer2DModelOutput:
        if kv_cache is not None:
            return self._forward_inference(
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                return_dict,
                attention_kwargs,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                start_frame=start_frame,
            )
        else:
            return self._forward_train(
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                return_dict,
                attention_kwargs,
                start_frame=start_frame,
            )

    def _forward_common_prepare(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None,
        start_frame: int = 0,
    ) -> tuple:
        """Shared preparation logic for both train and inference paths."""
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # RoPE with start_frame offset for incremental inference
        rotary_emb = self._compute_rotary_emb(
            hidden_states, post_patch_num_frames, post_patch_height, post_patch_width, start_frame
        )

        # Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Timestep handling
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            timestep_seq_len=ts_seq_len,
        )

        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        return (
            hidden_states,
            encoder_hidden_states,
            temb,
            timestep_proj,
            rotary_emb,
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
        )

    def _compute_rotary_emb(
        self,
        hidden_states: torch.Tensor,
        ppf: int,
        pph: int,
        ppw: int,
        start_frame: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE with optional frame offset for incremental inference."""
        if start_frame == 0:
            return self.rope(hidden_states)

        head_dim = self.attention_head_dim
        split_sizes = [
            head_dim - 2 * (head_dim // 3),
            head_dim // 3,
            head_dim // 3,
        ]
        freqs_cos = self.rope.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.rope.freqs_sin.split(split_sizes, dim=1)

        end_frame = start_frame + ppf
        freqs_cos_f = freqs_cos[0][start_frame:end_frame].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][start_frame:end_frame].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        cos_emb = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        sin_emb = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return cos_emb.to(hidden_states.device), sin_emb.to(hidden_states.device)

    def _forward_train(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        start_frame: int = 0,
    ) -> torch.Tensor | Transformer2DModelOutput:
        (
            hidden_states,
            encoder_hidden_states,
            temb,
            timestep_proj,
            rotary_emb,
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
        ) = self._forward_common_prepare(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            start_frame,
        )
        p_t, p_h, p_w = self.config.patch_size

        # Build block mask (lazily cached)
        if self.block_mask is None:
            num_frames = hidden_states.shape[1] // (post_patch_height * post_patch_width)
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device=hidden_states.device,
                num_frames=num_frames,
                frame_seqlen=post_patch_height * post_patch_width,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
            )

        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                block_mask=self.block_mask,
            )

        # Output
        return self._forward_output(
            hidden_states,
            temb,
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            return_dict,
        )

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: list[dict] | list[SelfAttentionKVCache] | None = None,
        crossattn_cache: list[CrossAttentionKVCache] | None = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
    ) -> torch.Tensor | Transformer2DModelOutput:
        (
            hidden_states,
            encoder_hidden_states,
            temb,
            timestep_proj,
            rotary_emb,
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
        ) = self._forward_common_prepare(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            start_frame,
        )
        p_t, p_h, p_w = self.config.patch_size

        # Transformer blocks with KV cache
        for block_index, block in enumerate(self.blocks):
            block_kv_cache = kv_cache[block_index] if kv_cache is not None else None
            block_crossattn_cache = crossattn_cache[block_index] if crossattn_cache is not None else None
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                block_mask=self.block_mask,
                kv_cache=block_kv_cache,
                crossattn_cache=block_crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
            )

        # Output
        return self._forward_output(
            hidden_states,
            temb,
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            return_dict,
        )

    def _forward_output(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        batch_size: int,
        post_patch_num_frames: int,
        post_patch_height: int,
        post_patch_width: int,
        return_dict: bool,
    ) -> torch.Tensor | Transformer2DModelOutput:
        p_t, p_h, p_w = self.config.patch_size

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(hidden_states.device) + temb.unsqueeze(2)).chunk(
                2, dim=2
            )
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(hidden_states.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states, scale, shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights, handling Q/K/V fusion for self-attention (same as WanTransformer3DModel).
        """
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        stacked_params_mapping = [
            (".attn1.to_qkv", ".attn1.to_q", "q"),
            (".attn1.to_qkv", ".attn1.to_k", "k"),
            (".attn1.to_qkv", ".attn1.to_v", "v"),
        ]
        self.stacked_params_mapping = stacked_params_mapping

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name

            # Handle QKV fusion
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ".ffn.net.0." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.net.0.", ".ffn.net_0.")
                elif ".ffn.net.2." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.net.2.", ".ffn.net_2.")

                if ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")

                if lookup_name.endswith(".modulation"):
                    modulation_alias = lookup_name[: -len(".modulation")] + ".scale_shift_table"
                    if modulation_alias in params_dict:
                        lookup_name = modulation_alias

                if lookup_name not in params_dict:
                    logger.warning("Skipping weight %s -> %s", original_name, lookup_name)
                    continue

                param = params_dict[lookup_name]

                # Shard RMSNorm weights for TP
                if tp_size > 1 and any(
                    norm_name in lookup_name
                    for norm_name in [
                        ".attn1.norm_q.",
                        ".attn1.norm_k.",
                        ".attn2.norm_q.",
                        ".attn2.norm_k.",
                        ".attn2.norm_added_k.",
                    ]
                ):
                    shard_size = loaded_weight.shape[0] // tp_size
                    loaded_weight = loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params

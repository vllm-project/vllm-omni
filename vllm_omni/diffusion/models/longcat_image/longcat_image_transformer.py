# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, apply_rotary_emb, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, bias: bool = True):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.w_in = ReplicatedLinear(dim, inner_dim, bias=bias, return_bias=False)
        self.act = get_act_fn("gelu_pytorch_tanh")
        self.w_out = ReplicatedLinear(inner_dim, dim_out, bias=bias, return_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.w_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.w_out(hidden_states)
        return hidden_states


class LongCatImageAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # Fused QKV projection using vLLM's optimized layer
        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            disable_tp=True,
            bias=bias,
        )

        if not self.pre_only:
            self.to_out = torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)

        if self.added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=self.added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                disable_tp=True,
                bias=added_proj_bias,
            )

            self.to_add_out = ReplicatedLinear(self.inner_dim, query_dim, bias=out_bias)

        self.attn = Attention(
            num_heads=heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def _sp_attention_with_rope(
        self,
        img_query: torch.Tensor,
        img_key: torch.Tensor,
        img_value: torch.Tensor,
        text_query: torch.Tensor,
        text_key: torch.Tensor,
        text_value: torch.Tensor,
        text_seq_len: int,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        """
        Apply RoPE separately to text and image Q/K, then run SP attention with joint tensors.

        This is the common SP attention pattern used by both dual-stream (added_kv_proj_dim)
        and single-stream (no added_kv_proj_dim) blocks.

        Args:
            img_query/key/value: Image Q/K/V tensors (chunked in SP mode)
            text_query/key/value: Text Q/K/V tensors (full, not chunked)
            text_seq_len: Length of text sequence for splitting RoPE
            image_rotary_emb: (freqs_cos, freqs_sin) containing [txt_pos, img_pos]

        Returns:
            Attention output with shape (B, txt_len + img_len/SP, H, D)
        """
        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            txt_rotary_emb = (freqs_cos[:text_seq_len], freqs_sin[:text_seq_len])
            img_rotary_emb_split = (freqs_cos[text_seq_len:], freqs_sin[text_seq_len:])
            # Apply RoPE to image Q/K
            img_query = apply_rotary_emb(img_query, img_rotary_emb_split, sequence_dim=1)
            img_key = apply_rotary_emb(img_key, img_rotary_emb_split, sequence_dim=1)
            # Apply RoPE to text Q/K
            text_query = apply_rotary_emb(text_query, txt_rotary_emb, sequence_dim=1)
            text_key = apply_rotary_emb(text_key, txt_rotary_emb, sequence_dim=1)

        return self.attn(
            img_query,
            img_key,
            img_value,
            AttentionMetadata(
                joint_query=text_query,
                joint_key=text_key,
                joint_value=text_value,
                joint_strategy="front",
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with SP-aware joint attention.

        Input shapes (in SP mode):
            - hidden_states: (B, img_seq_len // SP, D) - image hidden states (chunked)
            - encoder_hidden_states: (B, txt_seq_len, D) - text hidden states (full)

        SP Mode (sequence_parallel_size > 1):
            - Image Q/K/V: processed with AllToAll or Ring communication
            - Text Q/K/V: passed as joint tensors, broadcasted to all ranks
            - Output: attention over (text + image) with proper SP handling

        Non-SP Mode (sequence_parallel_size = 1):
            - Standard concatenation of text + image Q/K/V
            - Regular attention over the full sequence
        """
        qkv, _ = self.to_qkv(hidden_states)

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
            encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)

            encoder_query = encoder_query.unflatten(-1, (self.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.heads, -1))

            # Apply RMSNorm to text Q/K
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            # Check if SP is enabled from forward context (set by LongCatImageTransformer2DModel)
            forward_ctx = get_forward_context()
            sp_size = forward_ctx.sequence_parallel_size
            use_sp_joint_attention = sp_size > 1 and not forward_ctx.split_text_embed_in_sp

            if use_sp_joint_attention:
                # SP Mode: Use common helper for RoPE + joint attention
                hidden_states = self._sp_attention_with_rope(
                    img_query=query,
                    img_key=key,
                    img_value=value,
                    text_query=encoder_query,
                    text_key=encoder_key,
                    text_value=encoder_value,
                    text_seq_len=encoder_query.shape[1],
                    image_rotary_emb=image_rotary_emb,
                )
            else:
                # Non-SP Mode: Concat first, then apply RoPE to full sequence
                joint_query = torch.cat([encoder_query, query], dim=1)
                joint_key = torch.cat([encoder_key, key], dim=1)
                joint_value = torch.cat([encoder_value, value], dim=1)

                if image_rotary_emb is not None:
                    # Apply RoPE to full (text + image) sequence
                    joint_query = apply_rotary_emb(joint_query, image_rotary_emb, sequence_dim=1)
                    joint_key = apply_rotary_emb(joint_key, image_rotary_emb, sequence_dim=1)

                hidden_states = self.attn(
                    joint_query,
                    joint_key,
                    joint_value,
                )
        else:
            # No added_kv_proj_dim: single stream attention (e.g., from SingleTransformerBlock)
            # hidden_states is the combined (text + image) sequence
            # In SP mode, image part is chunked: (B, txt_len + img_len/SP, D)

            # Check if SP is enabled and we have text_seq_len info
            forward_ctx = get_forward_context()
            sp_size = forward_ctx.sequence_parallel_size
            text_seq_len = kwargs.get("text_seq_len", None)
            use_sp_single_stream = sp_size > 1 and not forward_ctx.split_text_embed_in_sp and text_seq_len is not None

            if use_sp_single_stream:
                # SP Mode for single-stream block:
                # Split QKV into text and image parts, then use common helper
                hidden_states = self._sp_attention_with_rope(
                    img_query=query[:, text_seq_len:],
                    img_key=key[:, text_seq_len:],
                    img_value=value[:, text_seq_len:],
                    text_query=query[:, :text_seq_len],
                    text_key=key[:, :text_seq_len],
                    text_value=value[:, :text_seq_len],
                    text_seq_len=text_seq_len,
                    image_rotary_emb=image_rotary_emb,
                )
            else:
                # Non-SP Mode: standard path
                if image_rotary_emb is not None:
                    query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                    key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

                hidden_states = self.attn(
                    query,
                    key,
                    value,
                )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split output back into text and image portions
            # In SP mode: seq_len = txt_seq_len + img_seq_len // SP
            # In non-SP mode: seq_len = txt_seq_len + img_seq_len
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = self.to_out(hidden_states)
            encoder_hidden_states, _ = self.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class LongCatImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = LongCatImageAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class LongCatImagePosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class LongCatImageTimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        return timesteps_emb


class LongCatImageSingleTransformerBlock(nn.Module):
    """
    Single-stream Transformer block for LongCat with SP (Sequence Parallelism) support.

    SP handling is delegated to LongCatImageAttention via the text_seq_len parameter.
    This keeps the block logic clean and centralizes SP logic in the attention layer.
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        # SP handling is delegated to LongCatImageAttention via text_seq_len kwarg
        self.attn = LongCatImageAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for SingleTransformerBlock with SP support.

        SP handling is delegated to LongCatImageAttention.forward via text_seq_len kwarg.
        This keeps the block logic clean and centralizes SP logic in the attention layer.
        """
        text_seq_len = encoder_hidden_states.shape[1]

        # Concatenate text and image
        # In SP mode: image is chunked (B, img_len/SP, D), text is full (B, txt_len, D)
        combined = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = combined
        norm_hidden_states, gate = self.norm(combined, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        # Delegate SP handling to LongCatImageAttention by passing text_seq_len
        # LongCatImageAttention will detect SP mode and handle text/image splitting internally
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            text_seq_len=text_seq_len,  # Pass text_seq_len for SP mode handling
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


class LongCatImageTransformer2DModel(nn.Module):
    """
    The Transformer model introduced in Flux.

    Supports Sequence Parallelism (Ulysses and Ring) when configured via OmniDiffusionConfig.
    """

    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 3584,
        axes_dims_rope: list[int] = [16, 56, 56],
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pooled_projection_dim = pooled_projection_dim

        # Store parallel config for SP support
        self.parallel_config = od_config.parallel_config

        self.pos_embed = LongCatImagePosEmbed(theta=10000, axes_dim=axes_dims_rope)

        self.time_embed = LongCatImageTimestepEmbeddings(embedding_dim=self.inner_dim)

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                LongCatImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                LongCatImageSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.use_checkpoint = [True] * num_layers
        self.use_single_checkpoint = [True] * num_single_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
    ) -> torch.FloatTensor | Transformer2DModelOutput:
        # Before: hidden_states shape = (B, img_seq_len, in_channels)
        # After:  hidden_states shape = (B, img_seq_len // SP, in_channels)
        sp_size = self.parallel_config.sequence_parallel_size
        # Store SP size in forward context for sub-modules to access
        get_forward_context().sequence_parallel_size = sp_size
        if sp_size > 1:
            sp_world_size = get_sequence_parallel_world_size()
            sp_rank = get_sequence_parallel_rank()
            original_shape = hidden_states.shape
            hidden_states = torch.chunk(hidden_states, sp_world_size, dim=1)[sp_rank]
            # LongCat uses dual-stream (text + image) with joint attention
            # Text embeddings should be replicated across SP ranks for correctness
            get_forward_context().split_text_embed_in_sp = False
            # Debug log (only first forward)
            if not hasattr(self, "_sp_forward_logged"):
                self._sp_forward_logged = True
                logger.info(
                    f"[LongCat Transformer] SP enabled: sp_size={sp_size}, world_size={sp_world_size}, "
                    f"rank={sp_rank}, original_shape={original_shape}, chunked_shape={hidden_states.shape}"
                )
        else:
            if not hasattr(self, "_sp_forward_logged"):
                self._sp_forward_logged = True
                logger.info(f"[LongCat Transformer] SP disabled: sp_size={sp_size}")

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000

        temb = self.time_embed(timestep, hidden_states.dtype)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=0)

        if current_omni_platform.is_npu():
            freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else:
            image_rotary_emb = self.pos_embed(ids)

        # SP: Chunk RoPE embeddings along sequence dimension
        if self.parallel_config.sequence_parallel_size > 1:
            sp_world_size = get_sequence_parallel_world_size()
            sp_rank = get_sequence_parallel_rank()
            freqs_cos, freqs_sin = image_rotary_emb
            txt_len = txt_ids.shape[0]

            # Split RoPE into text and image portions
            # txt_freqs: (txt_seq_len, head_dim) - keep full for all ranks
            # img_freqs: (img_seq_len, head_dim) -> (img_seq_len // SP, head_dim)
            txt_freqs_cos = freqs_cos[:txt_len]
            txt_freqs_sin = freqs_sin[:txt_len]
            img_freqs_cos = freqs_cos[txt_len:]
            img_freqs_sin = freqs_sin[txt_len:]

            # Chunk image RoPE for each SP rank
            # img_freqs_cos: (img_seq_len // SP, head_dim)
            # img_freqs_sin: (img_seq_len // SP, head_dim)
            img_freqs_cos = torch.chunk(img_freqs_cos, sp_world_size, dim=0)[sp_rank]
            img_freqs_sin = torch.chunk(img_freqs_sin, sp_world_size, dim=0)[sp_rank]

            # Optionally chunk text RoPE if split_text_embed_in_sp is True
            if get_forward_context().split_text_embed_in_sp:
                txt_freqs_cos = torch.chunk(txt_freqs_cos, sp_world_size, dim=0)[sp_rank]
                txt_freqs_sin = torch.chunk(txt_freqs_sin, sp_world_size, dim=0)[sp_rank]

            # Reconstruct image_rotary_emb with chunked values
            # Final shape: (txt_seq_len + img_seq_len // SP, head_dim)
            image_rotary_emb = (
                torch.cat([txt_freqs_cos, img_freqs_cos], dim=0),
                torch.cat([txt_freqs_sin, img_freqs_sin], dim=0),
            )

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing and self.use_checkpoint[index_block]:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing and self.use_single_checkpoint[index_block]:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        # SP: All-gather output to reconstruct full sequence
        if self.parallel_config.sequence_parallel_size > 1:
            output = get_sp_group().all_gather(output, dim=1)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self attn
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            # cross attn
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if ".to_out.0" in name:
                name = name.replace(".to_out.0", ".to_out")
            # Handle FeedForward parameter mapping
            if ".ff.net." in name:
                # Map .ff.net.0.proj -> .ff.w_in
                if ".net.0.proj" in name:
                    name = name.replace(".net.0.proj", ".w_in")
                # Map .ff.net.2 -> .ff.w_out
                elif ".net.2" in name:
                    name = name.replace(".net.2", ".w_out")
            # Handle FeedForward context parameters
            if ".ff_context.net." in name:
                # Map .ff_context.net.0.proj -> .ff_context.w_in
                if ".net.0.proj" in name:
                    name = name.replace(".net.0.proj", ".w_in")
                # Map .ff_context.net.2 -> .ff_context.w_out
                elif ".net.2" in name:
                    name = name.replace(".net.2", ".w_out")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

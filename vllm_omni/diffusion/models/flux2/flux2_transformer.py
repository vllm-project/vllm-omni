# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, apply_rotary_emb, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention

logger = init_logger(__name__)


class Flux2SwiGLU(nn.Module):
    """
    Flux 2 uses a SwiGLU-style activation in the transformer feedforward sub-blocks.
    """

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        x = self.gate_fn(x1) * x2
        return x


class Flux2FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: float = 3.0,
        inner_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # Flux2SwiGLU will reduce the dimension by half
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


class Flux2Attention(nn.Module):
    """
    Dual-stream attention for Flux 2.0 transformer blocks.
    Uses QKVParallelLinear from vLLM for efficiency.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.added_kv_proj_dim = added_kv_proj_dim

        # Image QKV projection (use vLLM fused layer)
        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=dim_head,
            total_num_heads=heads,
            disable_tp=True,
            bias=bias,
        )

        # QK normalization (Flux innovation)
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # Output projection
        self.to_out = nn.ModuleList([])
        self.to_out.append(ReplicatedLinear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
            # Text QKV projection
            self.add_kv_proj = QKVParallelLinear(
                hidden_size=added_kv_proj_dim,
                head_size=dim_head,
                total_num_heads=heads,
                disable_tp=True,
                bias=added_proj_bias,
            )
            self.to_add_out = ReplicatedLinear(self.inner_dim, query_dim, bias=out_bias)

        # Unified attention layer
        self.attn = Attention(
            num_heads=heads,
            head_size=dim_head,
            softmax_scale=1.0 / (dim_head**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden_states.shape[0]
        hidden_states.shape[1]
        txt_seq_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0

        # Image QKV
        img_qkv, _ = self.to_qkv(hidden_states)
        img_q, img_k, img_v = img_qkv.chunk(3, dim=-1)

        # Reshape to [B, seq_len, heads, head_dim]
        img_q = img_q.unflatten(-1, (self.heads, self.head_dim))
        img_k = img_k.unflatten(-1, (self.heads, self.head_dim))
        img_v = img_v.unflatten(-1, (self.heads, self.head_dim))

        # QK normalization
        img_q = self.norm_q(img_q)
        img_k = self.norm_k(img_k)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            # Text QKV
            txt_qkv, _ = self.add_kv_proj(encoder_hidden_states)
            txt_q, txt_k, txt_v = txt_qkv.chunk(3, dim=-1)

            # Reshape to [B, seq_len, heads, head_dim]
            txt_q = txt_q.unflatten(-1, (self.heads, self.head_dim))
            txt_k = txt_k.unflatten(-1, (self.heads, self.head_dim))
            txt_v = txt_v.unflatten(-1, (self.heads, self.head_dim))

            txt_q = self.norm_added_q(txt_q)
            txt_k = self.norm_added_k(txt_k)

            # Apply RoPE
            if image_rotary_emb is not None:
                img_q = apply_rotary_emb(img_q, image_rotary_emb, sequence_dim=1)
                img_k = apply_rotary_emb(img_k, image_rotary_emb, sequence_dim=1)
                txt_q = apply_rotary_emb(txt_q, image_rotary_emb, sequence_dim=1)
                txt_k = apply_rotary_emb(txt_k, image_rotary_emb, sequence_dim=1)

            # Joint attention: concatenate [text, image]
            joint_q = torch.cat([txt_q, img_q], dim=1)
            joint_k = torch.cat([txt_k, img_k], dim=1)
            joint_v = torch.cat([txt_v, img_v], dim=1)

            # Prepare attention metadata
            attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None

            # Attention - keep tensors in [B, seq_len, heads, head_dim] format
            joint_attn = self.attn(
                joint_q,
                joint_k,
                joint_v,
                attn_metadata=attn_metadata,
            )

            # Flatten head dimensions: [B, txt_len+img_len, heads, head_dim] -> [B, txt_len+img_len, inner_dim]
            joint_attn = joint_attn.flatten(-2)

            # Split back to text and image
            txt_attn = joint_attn[:, :txt_seq_len, :]
            img_attn = joint_attn[:, txt_seq_len:, :]

            # Output projections
            img_out, _ = self.to_out[0](img_attn)
            img_out = self.to_out[1](img_out)
            txt_out, _ = self.to_add_out(txt_attn)

            return img_out, txt_out
        else:
            # Self-attention only
            if image_rotary_emb is not None:
                img_q = apply_rotary_emb(img_q, image_rotary_emb, sequence_dim=1)
                img_k = apply_rotary_emb(img_k, image_rotary_emb, sequence_dim=1)

            # Prepare attention metadata
            attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None

            # Attention - keep tensors in [B, seq_len, heads, head_dim] format
            img_attn = self.attn(
                img_q,
                img_k,
                img_v,
                attn_metadata=attn_metadata,
            )

            # Flatten head dimensions
            img_attn = img_attn.flatten(-2)
            img_out, _ = self.to_out[0](img_attn)
            img_out = self.to_out[1](img_out)

            return img_out


class Flux2ParallelSelfAttention(nn.Module):
    """
    Flux 2 parallel self-attention for single-stream transformer blocks.
    Fuses QKV projections with MLP input projections.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        mlp_ratio: float = 4.0,
        mlp_mult_factor: int = 2,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(query_dim * self.mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        # Fused QKV projections + MLP input projection
        self.to_qkv_mlp_proj = nn.Linear(
            self.query_dim, self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor, bias=bias
        )
        self.mlp_act_fn = Flux2SwiGLU()

        # QK Norm
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # Unified attention layer
        self.attn = Attention(
            num_heads=heads,
            head_size=dim_head,
            softmax_scale=1.0 / (dim_head**0.5),
            causal=False,
        )

        # Fused attention output projection + MLP output projection
        self.to_out = nn.Linear(self.inner_dim + self.mlp_hidden_dim, self.out_dim, bias=out_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Parallel in (QKV + MLP in) projection
        hidden_states_proj = self.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states_proj, [3 * self.inner_dim, self.mlp_hidden_dim * self.mlp_mult_factor], dim=-1
        )

        # Handle the attention logic
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Prepare attention metadata
        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None

        # Attention - keep tensors in [B, seq_len, heads, head_dim] format
        hidden_states_attn = self.attn(
            query,
            key,
            value,
            attn_metadata=attn_metadata,
        )
        # Flatten head dimensions
        hidden_states_attn = hidden_states_attn.flatten(-2)
        hidden_states_attn = hidden_states_attn.to(query.dtype)

        # Handle the feedforward (FF) logic
        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)

        # Concatenate and parallel output projection
        hidden_states = torch.cat([hidden_states_attn, mlp_hidden_states], dim=-1)
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        # Parallel transformer block: attention and MLP in parallel
        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb_mod_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        split_hidden_states: bool = False,
        text_seq_len: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # If encoder_hidden_states is None, hidden_states is assumed to have encoder_hidden_states already concatenated
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = temb_mod_params

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            return encoder_hidden_states, hidden_states
        else:
            return hidden_states


class Flux2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Flux2Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_params_img: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        temb_mod_params_txt: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}

        # Modulation parameters shape: [1, 1, self.dim] or [B, 1, self.dim]
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        # Img stream
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        # Conditioning txt stream
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        # Attention on concatenated img + txt stream
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        attn_output, context_attn_output = attention_outputs

        # Process attention outputs for the image stream (`hidden_states`).
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        # Process attention outputs for the text stream (`encoder_hidden_states`).
        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class Flux2PosEmbed(nn.Module):
    """
    4D Rotary Position Embeddings for Flux 2.0.
    Supports (T, H, W, L) dimensions.
    """

    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 4D RoPE frequencies from position IDs.

        Args:
            ids: Position IDs tensor [S, 4] where S is sequence length and 4 is (T, H, W, L)

        Returns:
            (freqs_cos, freqs_sin): Frequency tensors [S, total_dim]
        """
        # Expected ids shape: [S, len(self.axes_dim)]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        # Loop over len(self.axes_dim) rather than ids.shape[-1]
        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
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


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(self, in_channels: int = 256, embedding_dim: int = 6144, bias: bool = False):
        super().__init__()

        self.time_proj = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias
        )

        self.guidance_embedder = TimestepEmbedding(
            in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias
        )

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))  # (N, D)

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(guidance.dtype))  # (N, D)

        time_guidance_emb = timesteps_emb + guidance_emb

        return time_guidance_emb


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = mod_param_sets

        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        mod = self.act_fn(temb)
        mod = self.linear(mod)

        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)
        # Return tuple of 3-tuples of modulation params shift/scale/gate
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets))


class Flux2Transformer2DModel(nn.Module):
    """
    The Transformer model introduced in Flux 2.

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `8`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `48`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `48`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `15360`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        timestep_guidance_channels (`int`, defaults to `256`):
            Number of channels for timestep/guidance embeddings.
        mlp_ratio (`float`, defaults to `3.0`):
            MLP ratio for feed-forward networks.
        axes_dims_rope (`Tuple[int]`, defaults to `(32, 32, 32, 32)`):
            The dimensions to use for the rotary positional embeddings (T, H, W, L).
        rope_theta (`int`, defaults to `2000`):
            Theta parameter for RoPE.
        eps (`float`, defaults to `1e-6`):
            Epsilon for layer normalization.
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        od_config: "OmniDiffusionConfig | None" = None,
    ):
        super().__init__()
        # Load from config if available (for CI testing with smaller models)
        if od_config is not None and od_config.tf_model_config.params:
            model_config = od_config.tf_model_config
            num_layers = model_config.get("num_layers", num_layers)
            num_single_layers = model_config.get("num_single_layers", num_single_layers)
            attention_head_dim = model_config.get("attention_head_dim", attention_head_dim)
            num_attention_heads = model_config.get("num_attention_heads", num_attention_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # 1. Sinusoidal positional embedding for RoPE on image and text tokens
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))

        # 2. Combined timestep + guidance embedding
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels, embedding_dim=self.inner_dim, bias=False
        )

        # 3. Modulation (double stream and single stream blocks share modulation parameters, resp.)
        # Two sets of shift/scale/gate modulation parameters for the double stream attn and FF sub-blocks
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        # Only one set of modulation parameters as the attn and FF sub-blocks are run in parallel for single stream
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        # 4. Input projections
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        # 5. Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`Flux2Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            img_ids (`torch.Tensor` of shape `(batch_size, image_sequence_length, 4)`):
                4D position IDs for image tokens (T, H, W, L).
            txt_ids (`torch.Tensor` of shape `(batch_size, text_sequence_length, 4)`):
                4D position IDs for text tokens (T, H, W, L).
            guidance (`torch.Tensor`):
                Guidance scale tensor.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the attention layers.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        joint_attention_kwargs = joint_attention_kwargs or {}

        num_txt_tokens = encoder_hidden_states.shape[1]

        # 1. Calculate timestep embedding and modulation parameters
        timestep = timestep.to(hidden_states.dtype) * 1000
        guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]

        # 2. Input projection for image (hidden_states) and conditioning text (encoder_hidden_states)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 3. Calculate RoPE embeddings from image and text tokens
        # NOTE: the below logic means that we can't support batched inference with images of different resolutions or
        # text prompts of different lengths. Is this a use case we want to support?
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # 4. Double Stream Transformer Blocks
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        # Concatenate text and image streams for single-block inference
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 5. Single Stream Transformer Blocks
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        # Remove text tokens from concatenated stream
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # 6. Output layers
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """
        Load weights from state dict.
        """
        stacked_params_mapping = [
            (
                "to_qkv",
                [
                    {"aliases": ["q_proj", "to_q", "q", "query"], "order": 0},
                    {"aliases": ["k_proj", "to_k", "k", "key"], "order": 1},
                    {"aliases": ["v_proj", "to_v", "v", "value"], "order": 2},
                ],
            ),
            (
                "add_kv_proj",
                [
                    {"aliases": ["add_q_proj", "add_q"], "order": 0},
                    {"aliases": ["add_k_proj", "add_k"], "order": 1},
                    {"aliases": ["add_v_proj", "add_v"], "order": 2},
                ],
            ),
        ]

        # Collect incoming weights into a dict for flexible mapping and grouping
        weights_map: dict[str, torch.Tensor] = {name: tensor for name, tensor in weights}

        # Helper that attempts to gather shards per base key using aliases and merges them
        for target, shards in stacked_params_mapping:
            base_candidates: dict[str, dict[str, dict[int, tuple[str, torch.Tensor]]]] = {}
            for key in list(weights_map.keys()):
                for shard in shards:
                    for alias in shard["aliases"]:
                        weight_suffix = f".{alias}.weight"
                        bias_suffix = f".{alias}.bias"
                        if key.endswith(weight_suffix):
                            base = key[: -len(weight_suffix)]
                            base_entry = base_candidates.setdefault(base, {"weight": {}, "bias": {}})
                            base_entry["weight"][shard["order"]] = (
                                key,
                                weights_map[key],
                            )
                            break
                        if key.endswith(bias_suffix):
                            base = key[: -len(bias_suffix)]
                            base_entry = base_candidates.setdefault(base, {"weight": {}, "bias": {}})
                            base_entry["bias"][shard["order"]] = (key, weights_map[key])
                            break

            for base, entry in base_candidates.items():
                if len(entry["weight"]) == len(shards):
                    ordered = [entry["weight"][order] for order in sorted(entry["weight"])]
                    combined = torch.cat([tensor for _, tensor in ordered], dim=0)
                    for key, _ in ordered:
                        weights_map.pop(key, None)
                    weights_map[base + f".{target}.weight"] = combined
                if len(entry["bias"]) == len(shards):
                    ordered_bias = [entry["bias"][order] for order in sorted(entry["bias"])]
                    bcombined = torch.cat([tensor for _, tensor in ordered_bias], dim=0)
                    for key, _ in ordered_bias:
                        weights_map.pop(key, None)
                    weights_map[base + f".{target}.bias"] = bcombined

        # Now perform the parameter copy using the possibly-updated weights_map
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights_map.items():
            # Handle potential name mismatches for FeedForward layers
            if name.endswith(".proj.weight") and name not in params_dict:
                mapped_name = name.replace(".proj.weight", ".weight")
            elif name.endswith(".proj.bias") and name not in params_dict:
                mapped_name = name.replace(".proj.bias", ".bias")
            else:
                mapped_name = name

            if mapped_name in params_dict:
                param = params_dict[mapped_name]
                with torch.no_grad():
                    param.copy_(loaded_weight)
            else:
                logger.warning(f"Weight {name} (mapped to {mapped_name}) not found in model params.")


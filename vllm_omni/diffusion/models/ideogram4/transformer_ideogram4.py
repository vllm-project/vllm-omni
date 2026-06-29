# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from diffusers.models.transformers.transformer_ideogram4

import math
from collections.abc import Iterable
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

# Per-token role indicators used to label entries of the packed text+image sequence.
SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3

# Image grid coordinates start at this offset so they never collide with text token indices.
IMAGE_POSITION_OFFSET = 65536


def _join_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Ideogram4MRoPE(nn.Module):
    """Multi-axis (t, h, w) interleaved rotary position embedding."""

    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        base: int,
        mrope_section: tuple[int, ...],
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, L, 3) of int (axes are t, h, w).
        if position_ids.ndim != 3 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must have shape (B, L, 3), got {tuple(position_ids.shape)}.")
        batch_size, seq_len, _ = position_ids.shape

        # Ideogram4's image position ids start at IMAGE_POSITION_OFFSET (65536).
        # Disable autocast to avoid bfloat16 collapse at large position values.
        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(3, batch_size, -1, 1)
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = inv_freq @ pos.unsqueeze(2)
        freqs = freqs.transpose(2, 3)  # (3, B, L, inv_freq_size)

        # Interleaved mrope: pull H freqs into idx 1 mod 3, W freqs into idx 2 mod 3.
        freqs_t = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3, device=freqs_t.device)
            freqs_t[..., idx] = freqs[axis][..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos().float(), emb.sin().float()


class Ideogram4MLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        x = F.silu(x1) * x3
        x = self.w2(x)
        return x


class Ideogram4Attention(nn.Module):
    """Self-attention with merged QKV projection, q/k RMSNorm, and MRoPE.

    Note: This uses merged QKV projection (matching the checkpoint format)
    instead of separate to_q/to_k/to_v projections.
    """

    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Merged QKV projection (matches checkpoint: attention.qkv.weight)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        # Q/K normalization (per-head RMSNorm)
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)

        # Output projection (matches checkpoint: attention.o.weight)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.num_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # Merged QKV projection
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: (B, S, num_heads, head_dim)

        # Q/K normalization
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Apply MRoPE
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.unsqueeze(2)  # (B, L, 1, head_dim)
            sin = sin.unsqueeze(2)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)

        # Q/K/V shape: (B, S, num_heads, head_dim) - keep this format for SDPA

        # Attention
        attn_metadata = None
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            # Keep mask as [B, 1, S, S] for SDPA to broadcast
            attn_metadata = AttentionMetadata(attn_mask=attention_mask)

        hidden_states = self.attn(q, k, v, attn_metadata)
        hidden_states = hidden_states.reshape(B, S, self.hidden_size)

        # Output projection
        hidden_states = self.o(hidden_states.contiguous())
        return hidden_states


class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int, norm_eps: float, adaln_dim: int
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)

        self.attention_norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(hidden_size, eps=norm_eps)

        self.adaln_modulation = nn.Linear(adaln_dim, 4 * hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        adaln_input: torch.Tensor,
    ) -> torch.Tensor:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(hidden_states) * scale_msa,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + gate_msa * self.attention_norm2(attn_out)
        hidden_states = hidden_states + gate_mlp * self.ffn_norm2(
            self.feed_forward(self.ffn_norm1(hidden_states) * scale_mlp)
        )
        return hidden_states


def _sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4) -> torch.Tensor:
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    """Sinusoidal scalar embedding followed by a small MLP."""

    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        if self.range_max <= self.range_min:
            raise ValueError("input_range[1] must be greater than input_range[0]")
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(in_dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adaln_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adaln_dim, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(conditioning))
        output = self.linear(self.norm_final(hidden_states) * scale)
        return output


class Ideogram4Transformer2DModel(nn.Module):
    """The flow-matching transformer backbone used by the Ideogram 4 pipeline."""

    _repeated_blocks = ["Ideogram4TransformerBlock"]
    _skip_layerwise_casting_patterns = ["t_embedding", "adaln_proj", "embed_image_indicator"]

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return "layers" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        in_channels: int = 128,
        num_layers: int = 34,
        attention_head_dim: int = 256,
        num_attention_heads: int = 18,
        intermediate_size: int = 12288,
        adaln_dim: int = 512,
        llm_features_dim: int = 53248,
        rope_theta: int = 5_000_000,
        mrope_section: tuple[int, int, int] = (24, 20, 20),
        norm_eps: float = 1e-5,
        od_config: OmniDiffusionConfig | None = None,
    ) -> None:
        super().__init__()

        hidden_size = attention_head_dim * num_attention_heads
        head_dim = attention_head_dim

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.gradient_checkpointing = False

        self.config = SimpleNamespace(
            in_channels=in_channels,
            out_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            adaln_dim=adaln_dim,
            llm_features_dim=llm_features_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            norm_eps=norm_eps,
        )

        if od_config is not None:
            self.parallel_config = od_config.parallel_config
        else:
            self.parallel_config = DiffusionParallelConfig()

        self.input_proj = nn.Linear(in_channels, hidden_size, bias=True)
        self.llm_cond_norm = DiffusersRMSNorm(llm_features_dim, eps=1e-6, elementwise_affine=True)
        self.llm_cond_proj = nn.Linear(llm_features_dim, hidden_size, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(hidden_size, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(hidden_size, adaln_dim, bias=True)

        self.embed_image_indicator = nn.Embedding(2, hidden_size)

        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=rope_theta,
            mrope_section=mrope_section,
        )

        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_attention_heads,
                    norm_eps=norm_eps,
                    adaln_dim=adaln_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.final_layer = Ideogram4FinalLayer(hidden_size=hidden_size, out_channels=in_channels, adaln_dim=adaln_dim)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
        attention_kwargs: dict | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        batch_size, seq_len, in_channels = hidden_states.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected last dim {self.in_channels}, got {in_channels}.")

        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(hidden_states.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(hidden_states.dtype).unsqueeze(-1)

        encoder_hidden_states = encoder_hidden_states * llm_token_mask
        hidden_states = hidden_states * output_image_mask
        hidden_states = self.input_proj(hidden_states) * output_image_mask

        t_cond = self.t_embedding(timestep)
        if timestep.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        encoder_hidden_states = self.llm_cond_norm(encoder_hidden_states)
        encoder_hidden_states = self.llm_cond_proj(encoder_hidden_states) * llm_token_mask

        hidden_states = hidden_states + encoder_hidden_states

        image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
        hidden_states = hidden_states + image_indicator_embedding

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)
        image_rotary_emb = (cos, sin)

        # Block-diagonal mask from segment ids: tokens only attend within their segment.
        attention_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)

        for block in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, attention_mask, image_rotary_emb, adaln_input
                )
            else:
                hidden_states = block(hidden_states, attention_mask, image_rotary_emb, adaln_input)

        output = self.final_layer(hidden_states, conditioning=adaln_input)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Convert to list to allow multiple passes (FP8 detection + actual loading)
        weights_list = list(weights)

        # Check if checkpoint uses Ideogram's weight-only FP8 format
        from vllm_omni.diffusion.models.ideogram4.ideogram_fp8 import (
            is_ideogram_fp8_state_dict,
            swap_linears_to_fp8,
        )

        # Build state dict for FP8 detection
        state_dict = {name: tensor for name, tensor in weights_list}

        if is_ideogram_fp8_state_dict(state_dict):
            # Swap nn.Linear to Ideogram4Fp8Linear before loading
            # This must happen before we try to load weight_scale buffers
            swap_linears_to_fp8(self, state_dict, compute_dtype=self.dtype)

        # Now load the weights
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        loaded_params: set[str] = set()
        for original_name, loaded_weight in weights_list:
            # AutoWeightsLoader passes names with prefix like "transformer.layers.0..."
            # We need to strip the prefix to match params_dict
            name = original_name
            if name.startswith("transformer."):
                name = name[len("transformer.") :]
            elif name.startswith("unconditional_transformer."):
                name = name[len("unconditional_transformer.") :]

            # Check if it's a parameter or buffer
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(original_name)
            elif name in buffers_dict:
                # For weight_scale and other buffers
                buffer = buffers_dict[name]
                buffer.copy_(loaded_weight)
                loaded_params.add(original_name)

        return loaded_params

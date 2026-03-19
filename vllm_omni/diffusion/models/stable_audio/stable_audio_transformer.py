# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Stable Audio DiT Model for vLLM-Omni.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


def apply_rotary_emb_stable_audio(
    hidden_states: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors for Stable Audio.

    Args:
        hidden_states: Input tensor of shape [B, S, H, D] where D is head_dim
        freqs_cis: Tuple of (cos, sin) frequency tensors of shape [S, rotary_dim]
                   where rotary_dim = head_dim // 2

    Returns:
        Tensor with rotary embeddings applied to first rotary_dim dimensions only.
        The remaining dimensions are left unchanged (pass-through).
    """
    cos, sin = freqs_cis  # [S, rotary_dim]
    rotary_dim = cos.shape[-1]

    # Rotate only the first rotary_dim entries; leave the rest unchanged
    x_rot = hidden_states[..., :rotary_dim]
    x_pass = hidden_states[..., rotary_dim:]

    cos = cos[None, :, None, :]  # [1, S, 1, rotary_dim]
    sin = sin[None, :, None, :]  # [1, S, 1, rotary_dim]

    # [B, S, H, rotary_dim] -> [B, S, H, 2, rotary_dim//2] -> two halves
    x_real, x_imag = x_rot.reshape(*x_rot.shape[:-1], 2, rotary_dim // 2).unbind(-2)
    x_rotated = torch.cat([-x_imag, x_real], dim=-1)

    x_rot = (x_rot.float() * cos + x_rotated.float() * sin).to(hidden_states.dtype)
    return torch.cat([x_rot, x_pass], dim=-1)


class StableAudioGaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.

    Matches diffusers StableAudioGaussianFourierProjection with:
    - flip_sin_to_cos=True (output is [cos, sin] not [sin, cos])
    - log=False (no log transformation of input)
    """

    def __init__(self, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch] or [batch, 1]
        # Output: [batch, embedding_size * 2]
        x_proj = 2 * math.pi * x[:, None] @ self.weight[None, :]
        # flip_sin_to_cos=True means cos comes first
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class StableAudioSelfAttention(nn.Module):
    """
    Optimized self-attention for Stable Audio using vLLM layers.

    Self-attention uses full attention (all heads for Q, K, V).
    GQA is only used for cross-attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_key_value_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        # All projections use inner_dim for output
        self.to_q = ReplicatedLinear(dim, self.inner_dim, bias=False)
        self.to_k = ReplicatedLinear(dim, self.inner_dim, bias=False)
        self.to_v = ReplicatedLinear(dim, self.inner_dim, bias=False)

        # Output projection
        self.to_out = nn.ModuleList(
            [
                ReplicatedLinear(self.inner_dim, dim, bias=False),
                nn.Dropout(dropout),
            ]
        )

        # Full attention (no GQA for self-attention)
        self.attn = Attention(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            softmax_scale=1.0 / (attention_head_dim**0.5),
            causal=False,
            num_kv_heads=num_attention_heads,  # Same as query heads
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Projections - all output inner_dim
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)

        # Reshape for multi-head attention (all use full heads)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply rotary embeddings
        if rotary_emb is not None:
            query = apply_rotary_emb_stable_audio(query, rotary_emb)
            key = apply_rotary_emb_stable_audio(key, rotary_emb)

        # Compute attention
        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.view(batch_size, seq_len, self.inner_dim)

        # Output projection
        hidden_states, _ = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class StableAudioCrossAttention(nn.Module):
    """
    Optimized cross-attention for Stable Audio using vLLM layers.

    For cross-attention:
    - Q projection: outputs inner_dim (full heads)
    - K/V projections: outputs kv_dim (reduced heads for GQA)

    GQA is handled by manually expanding K/V heads to match Q heads
    since the SDPA backend doesn't handle this automatically.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_key_value_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.kv_dim = num_key_value_attention_heads * attention_head_dim

        # Number of times to repeat KV heads
        self.num_kv_groups = num_attention_heads // num_key_value_attention_heads

        # Q outputs inner_dim, K/V output kv_dim (GQA)
        self.to_q = ReplicatedLinear(dim, self.inner_dim, bias=False)
        self.to_k = ReplicatedLinear(cross_attention_dim, self.kv_dim, bias=False)
        self.to_v = ReplicatedLinear(cross_attention_dim, self.kv_dim, bias=False)

        # Output projection
        self.to_out = nn.ModuleList(
            [
                ReplicatedLinear(self.inner_dim, dim, bias=False),
                nn.Dropout(dropout),
            ]
        )

        # Use full heads for attention (KV will be expanded)
        self.attn = Attention(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            softmax_scale=1.0 / (attention_head_dim**0.5),
            causal=False,
            num_kv_heads=num_attention_heads,  # After expansion
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        encoder_seq_len = encoder_hidden_states.shape[1]

        # Projections
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(encoder_hidden_states)
        value, _ = self.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, encoder_seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, encoder_seq_len, self.num_kv_heads, self.head_dim)

        # Expand K/V heads to match Q heads for GQA
        # [B, S, kv_heads, D] -> [B, S, kv_heads, 1, D] -> [B, S, kv_heads, groups, D] -> [B, S, num_heads, D]
        key = key.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
        key = key.reshape(batch_size, encoder_seq_len, self.num_heads, self.head_dim)
        value = value.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
        value = value.reshape(batch_size, encoder_seq_len, self.num_heads, self.head_dim)

        # Compute attention
        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.view(batch_size, seq_len, self.inner_dim)

        # Output projection
        hidden_states, _ = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class SwiGLU(nn.Module):
    """SwiGLU activation - matches diffusers structure."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class StableAudioFeedForward(nn.Module):
    """
    Feed-forward network with SwiGLU activation for Stable Audio.
    Matches diffusers FeedForward structure with activation_fn="swiglu".
    """

    def __init__(self, dim: int, inner_dim: int, bias: bool = True):
        super().__init__()
        # Structure matches diffusers FeedForward:
        # net.0 = SwiGLU (proj.weight, proj.bias)
        # net.1 = Dropout
        # net.2 = Linear (weight, bias)
        self.net = nn.Sequential(
            SwiGLU(dim, inner_dim, bias=bias),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim, bias=bias),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class StableAudioDiTBlock(nn.Module):
    """
    Stable Audio DiT block with self-attention, cross-attention, and FFN.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_key_value_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        ff_mult: int = 4,
    ):
        super().__init__()

        # Self-attention with layer norm
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn1 = StableAudioSelfAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            num_key_value_attention_heads=num_key_value_attention_heads,
            attention_head_dim=attention_head_dim,
        )

        # Cross-attention with layer norm
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn2 = StableAudioCrossAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            num_key_value_attention_heads=num_key_value_attention_heads,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
        )

        # Feed-forward with SwiGLU activation
        # inner_dim = dim * ff_mult (e.g., 1536 * 4 = 6144)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=True)
        self.ff = StableAudioFeedForward(dim, inner_dim=dim * ff_mult)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with skip connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn1(hidden_states, rotary_emb=rotary_embedding, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # Cross-attention with skip connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = residual + hidden_states

        # Feed-forward with skip connection
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class StableAudioDiTModel(nn.Module):
    """
    Optimized Stable Audio DiT model using vLLM layers.

    This is an optimized version of the diffusers StableAudioDiTModel that uses
    vLLM's efficient linear layers and attention implementations.

    Architecture:
    - Input: [B, in_channels, L] (e.g., [B, 64, L])
    - preprocess_conv: residual conv layer (keeps 64 channels)
    - proj_in: projects 64 -> 1536 (inner_dim)
    - Global+time embeddings prepended to sequence
    - Transformer blocks work on 1536-dim
    - proj_out: projects 1536 -> 64 (out_channels)
    - postprocess_conv: residual conv layer (keeps 64 channels)
    - Output: [B, out_channels, L]
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        sample_size: int = 1024,
        in_channels: int = 64,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        num_key_value_attention_heads: int = 12,
        out_channels: int = 64,
        cross_attention_dim: int = 768,
        time_proj_dim: int = 256,
        global_states_input_dim: int = 1536,
        cross_attention_input_dim: int = 768,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads

        # inner_dim is the transformer hidden dimension
        self.inner_dim = num_attention_heads * attention_head_dim

        # Store config for compatibility
        self.config = type(
            "Config",
            (),
            {
                "sample_size": sample_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "num_layers": num_layers,
                "attention_head_dim": attention_head_dim,
                "num_attention_heads": num_attention_heads,
                "num_key_value_attention_heads": num_key_value_attention_heads,
                "cross_attention_dim": cross_attention_dim,
                "time_proj_dim": time_proj_dim,
                "global_states_input_dim": global_states_input_dim,
                "cross_attention_input_dim": cross_attention_input_dim,
            },
        )()

        # Time projection (Gaussian Fourier features)
        # time_proj_dim is the OUTPUT dimension (after sin/cos concatenation)
        # So embedding_size = time_proj_dim // 2
        self.time_proj = StableAudioGaussianFourierProjection(embedding_size=time_proj_dim // 2)

        # Timestep projection: time_proj_dim -> inner_dim
        self.timestep_proj = nn.Sequential(
            nn.Linear(time_proj_dim, self.inner_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim, bias=True),
        )

        # Global states projection (for audio duration conditioning)
        # Output is inner_dim, added to time embedding
        self.global_proj = nn.Sequential(
            nn.Linear(global_states_input_dim, self.inner_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim, bias=False),
        )

        # Cross-attention input projection
        # Always use Sequential(Linear, SiLU, Linear) to match diffusers structure
        self.cross_attention_proj = nn.Sequential(
            nn.Linear(cross_attention_input_dim, cross_attention_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cross_attention_dim, cross_attention_dim, bias=False),
        )

        # Pre-processing conv (residual connection)
        self.preprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)

        # Input projection: in_channels -> inner_dim (64 -> 1536)
        self.proj_in = nn.Linear(in_channels, self.inner_dim, bias=False)

        # Transformer blocks - work on inner_dim (1536)
        self.transformer_blocks = nn.ModuleList(
            [
                StableAudioDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_key_value_attention_heads=num_key_value_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection: inner_dim -> out_channels (1536 -> 64)
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)

        # Post-processing conv (residual connection)
        self.postprocess_conv = nn.Conv1d(out_channels, out_channels, 1, bias=False)

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model parameters."""
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor | None = None,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_dict: bool = True,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        Forward pass of the Stable Audio DiT model.

        Args:
            hidden_states: Input latent tensor [B, C, L] (C=in_channels=64)
            timestep: Timestep tensor [B] or [1]
            encoder_hidden_states: Text/condition embeddings [B, S, D]
            global_hidden_states: Global conditioning (duration) [B, 1, D]
            rotary_embedding: Precomputed rotary embeddings (cos, sin)
            return_dict: Whether to return a dataclass or tuple
            attention_mask: Attention mask for self-attention
            encoder_attention_mask: Attention mask for cross-attention

        Returns:
            Denoised latent tensor
        """
        # Project cross-attention inputs
        cross_attention_hidden_states = self.cross_attention_proj(encoder_hidden_states)

        # Global embedding projection [B, 1, D] -> [B, 1, inner_dim]
        global_hidden_states = self.global_proj(global_hidden_states)

        # Time embedding: timestep -> time_proj -> timestep_proj
        time_hidden_states = self.timestep_proj(self.time_proj(timestep.to(self.dtype)))

        # Combine global and time embeddings [B, 1, inner_dim]
        global_hidden_states = global_hidden_states + time_hidden_states.unsqueeze(1)

        # Pre-process with residual: [B, C, L]
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states

        # Transpose: [B, C, L] -> [B, L, C]
        hidden_states = hidden_states.transpose(1, 2)

        # Project to inner_dim: [B, L, C] -> [B, L, inner_dim]
        hidden_states = self.proj_in(hidden_states)

        # Prepend global states to hidden states: [B, 1+L, inner_dim]
        hidden_states = torch.cat([global_hidden_states, hidden_states], dim=1)

        # Update attention mask if provided
        if attention_mask is not None:
            prepend_mask = torch.ones(
                (hidden_states.shape[0], 1),
                device=hidden_states.device,
                dtype=torch.bool,
            )
            attention_mask = torch.cat([prepend_mask, attention_mask], dim=-1)

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                cross_attention_hidden_states,
                rotary_embedding=rotary_embedding,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )

        # Project back to out_channels: [B, 1+L, inner_dim] -> [B, 1+L, out_channels]
        hidden_states = self.proj_out(hidden_states)

        # Transpose and remove prepended global token: [B, L, C] -> [B, C, L]
        hidden_states = hidden_states.transpose(1, 2)[:, :, 1:]

        # Post-process with residual: [B, C, L]
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        if return_dict:
            return Transformer2DModelOutput(sample=hidden_states)
        return (hidden_states,)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights from a pretrained model.

        Maps diffusers weight names to our module structure.

        Returns:
            Set of parameter names that were successfully loaded.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Weight name mapping from diffusers to our implementation
        name_mapping = {
            # Timestep projection - diffusers uses index-based naming
            "timestep_proj.linear_1.weight": "timestep_proj.0.weight",
            "timestep_proj.linear_1.bias": "timestep_proj.0.bias",
            "timestep_proj.linear_2.weight": "timestep_proj.2.weight",
            "timestep_proj.linear_2.bias": "timestep_proj.2.bias",
            # Global projection - diffusers uses index-based naming
            "global_proj.linear_1.weight": "global_proj.0.weight",
            "global_proj.linear_2.weight": "global_proj.2.weight",
        }

        for name, loaded_weight in weights:
            # Apply name mapping if needed
            mapped_name = name_mapping.get(name, name)

            if mapped_name in params_dict:
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped_name)
            else:
                logger.debug(f"Skipping weight {name} - not found in model")

        return loaded_params

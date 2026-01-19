# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GLM-TTS DiT (Flow Matching) Model for vLLM-Omni.

This module implements the flow matching model that converts speech tokens
to mel-spectrograms, based on the GLM-TTS architecture.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


def apply_rotary_emb_glm_tts(
    hidden_states: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors for GLM-TTS.

    Args:
        hidden_states: Input tensor of shape [B, S, H, D] where D is head_dim
        freqs_cis: Tuple of (cos, sin) frequency tensors of shape [S, rotary_dim]

    Returns:
        Tensor with rotary embeddings applied.
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


class GLMTTSGaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for timestep conditioning.

    Matches diffusers pattern with flip_sin_to_cos=True.
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


class GLMTTSSelfAttention(nn.Module):
    """
    Optimized self-attention for GLM-TTS using vLLM layers.

    Uses full attention (all heads for Q, K, V).
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
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
            query = apply_rotary_emb_glm_tts(query, rotary_emb)
            key = apply_rotary_emb_glm_tts(key, rotary_emb)

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


class GLMTTSFeedForward(nn.Module):
    """
    Feed-forward network with SwiGLU activation for GLM-TTS.
    Matches diffusers FeedForward structure with activation_fn="swiglu".
    """

    def __init__(self, dim: int, inner_dim: int, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            SwiGLU(dim, inner_dim, bias=bias),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim, bias=bias),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class GLMTTSAdaLayerNorm(nn.Module):
    """Adaptive LayerNorm with timestep conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply adaptive layer normalization.

        Returns:
            Tuple of (normed_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        emb = self.linear(self.silu(timestep_emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class GLMTTSAdaLayerNormFinal(nn.Module):
    """Final adaptive LayerNorm (no MLP modulation needed)."""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.linear(self.silu(timestep_emb))
        scale, shift = emb.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return hidden_states


class GLMTTSDiTBlock(nn.Module):
    """
    GLM-TTS DiT block with self-attention and FFN.
    Uses adaptive layer norm for timestep conditioning.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ff_mult: int = 4,
    ):
        super().__init__()

        # Self-attention with adaptive layer norm
        self.attn_norm = GLMTTSAdaLayerNorm(dim)
        self.attn = GLMTTSSelfAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )

        # Feed-forward with layer norm
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = GLMTTSFeedForward(dim, inner_dim=dim * ff_mult)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with adaptive norm
        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(hidden_states, timestep_emb)
        attn_out = self.attn(norm_hidden, rotary_emb=rotary_embedding, attention_mask=attention_mask)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_out

        # Feed-forward with adaptive norm
        norm_hidden = self.ff_norm(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ff_out = self.ff(norm_hidden)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_out

        return hidden_states


class GLMTTSDiTModel(nn.Module):
    """
    Optimized GLM-TTS DiT model using vLLM layers.

    This model implements flow matching for speech token to mel-spectrogram generation.

    Architecture:
    - Input: Speech tokens + noisy mel + speaker embedding
    - Preprocess: Project and combine inputs
    - Transformer blocks with adaptive layer norm
    - Output: Predicted velocity for flow matching

    Args:
        od_config: OmniDiffusion configuration object
        sample_size: Maximum mel spectrogram length
        mel_dim: Mel spectrogram dimension (typically 80)
        num_layers: Number of transformer blocks
        attention_head_dim: Dimension per attention head
        num_attention_heads: Number of attention heads
        speech_token_vocab_size: Vocabulary size for speech tokens
        speech_token_dim: Embedding dimension for speech tokens
        speaker_embed_dim: Dimension of speaker embeddings
        time_proj_dim: Time projection dimension
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        sample_size: int = 2048,
        mel_dim: int = 80,
        num_layers: int = 22,
        attention_head_dim: int = 64,
        num_attention_heads: int = 16,
        speech_token_vocab_size: int = 100000,
        speech_token_dim: int = 512,
        speaker_embed_dim: int = 192,
        time_proj_dim: int = 256,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.mel_dim = mel_dim
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads

        # inner_dim is the transformer hidden dimension
        self.inner_dim = num_attention_heads * attention_head_dim

        # Store config for compatibility (like Stable Audio)
        self.config = type(
            "Config",
            (),
            {
                "sample_size": sample_size,
                "mel_dim": mel_dim,
                "num_layers": num_layers,
                "attention_head_dim": attention_head_dim,
                "num_attention_heads": num_attention_heads,
                "speech_token_vocab_size": speech_token_vocab_size,
                "speech_token_dim": speech_token_dim,
                "speaker_embed_dim": speaker_embed_dim,
                "time_proj_dim": time_proj_dim,
            },
        )()

        # Time projection (Gaussian Fourier features)
        self.time_proj = GLMTTSGaussianFourierProjection(embedding_size=time_proj_dim // 2)

        # Timestep projection: time_proj_dim -> inner_dim
        self.timestep_proj = nn.Sequential(
            nn.Linear(time_proj_dim, self.inner_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim, bias=True),
        )

        # Speech token embedding
        self.speech_token_embed = nn.Embedding(speech_token_vocab_size + 1, speech_token_dim)

        # Speaker embedding projection
        self.speaker_proj = nn.Sequential(
            nn.Linear(speaker_embed_dim, self.inner_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim, bias=False),
        )

        # Input projection: mel + speech_token_dim + inner_dim (speaker) -> inner_dim
        self.proj_in = nn.Linear(mel_dim + speech_token_dim + self.inner_dim, self.inner_dim, bias=False)

        # Rotary embedding (computed on the fly based on sequence length)
        self.rotary_embed_dim = attention_head_dim // 2

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                GLMTTSDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Final norm and output projection
        self.norm_out = GLMTTSAdaLayerNormFinal(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, mel_dim, bias=False)

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model parameters."""
        return next(self.parameters()).dtype

    def _get_rotary_embedding(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings for the given sequence length."""
        dim = self.rotary_embed_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        # Double the dimension by concatenating
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        speech_tokens: torch.Tensor,
        speaker_embedding: torch.Tensor,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_dict: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        Forward pass of the GLM-TTS DiT model.

        Args:
            hidden_states: Noisy mel-spectrogram [B, T, mel_dim]
            timestep: Timestep tensor [B] or [1]
            speech_tokens: Speech token indices [B, T_tokens]
            speaker_embedding: Speaker embedding [B, speaker_dim]
            rotary_embedding: Precomputed rotary embeddings (cos, sin)
            return_dict: Whether to return a dataclass or tuple
            attention_mask: Optional attention mask

        Returns:
            Predicted velocity for flow matching [B, T, mel_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Time embedding
        time_hidden_states = self.timestep_proj(self.time_proj(timestep.to(self.dtype)))

        # Speech token embedding - interpolate to match mel length
        token_emb = self.speech_token_embed(speech_tokens)  # [B, T_tokens, D]
        if token_emb.shape[1] != seq_len:
            token_emb = token_emb.transpose(1, 2)  # [B, D, T_tokens]
            token_emb = F.interpolate(token_emb, size=seq_len, mode="linear", align_corners=False)
            token_emb = token_emb.transpose(1, 2)  # [B, T, D]

        # Speaker embedding projection and expand to sequence
        spk_emb = self.speaker_proj(speaker_embedding)  # [B, inner_dim]
        spk_emb = spk_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, inner_dim]

        # Concatenate inputs and project
        combined = torch.cat([hidden_states, token_emb, spk_emb], dim=-1)
        hidden_states = self.proj_in(combined)

        # Compute rotary embeddings if not provided
        if rotary_embedding is None:
            rotary_embedding = self._get_rotary_embedding(seq_len, device, hidden_states.dtype)

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                time_hidden_states,
                rotary_embedding=rotary_embedding,
                attention_mask=attention_mask,
            )

        # Final norm and output projection
        hidden_states = self.norm_out(hidden_states, time_hidden_states)
        output = self.proj_out(hidden_states)

        if return_dict:
            return Transformer2DModelOutput(sample=output)
        return (output,)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights from a pretrained model.

        Maps diffusers/HF weight names to our module structure.

        Returns:
            Set of parameter names that were successfully loaded.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Weight name mapping from HF/diffusers to our implementation
        name_mapping = {
            # Timestep projection
            "timestep_proj.linear_1.weight": "timestep_proj.0.weight",
            "timestep_proj.linear_1.bias": "timestep_proj.0.bias",
            "timestep_proj.linear_2.weight": "timestep_proj.2.weight",
            "timestep_proj.linear_2.bias": "timestep_proj.2.bias",
            # Speaker projection
            "speaker_proj.linear_1.weight": "speaker_proj.0.weight",
            "speaker_proj.linear_2.weight": "speaker_proj.2.weight",
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

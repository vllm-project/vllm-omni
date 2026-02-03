# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_glm_image import GlmImageCombinedTimestepSizeEmbeddings
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.base import CachedTransformer
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

logger = init_logger(__name__)


class GlmImageImageProjector(nn.Module):
    """Projects latent image patches to transformer hidden dimension."""

    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = hidden_states.shape
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size

        # Reshape: [B, C, H, W] -> [B, H', W', C*p*p] -> [B, H'*W', C*p*p]
        hidden_states = hidden_states.reshape(
            batch_size, channel, post_patch_height, self.patch_size, post_patch_width, self.patch_size
        )
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class GlmImageRotaryPosEmbed(nn.Module):
    """Rotary positional embedding for 2D image patches."""

    def __init__(self, dim: int, patch_size: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size

        dim_h, dim_w = self.dim // 2, self.dim // 2
        h_inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, dim_h, 2, dtype=torch.float32)[: (dim_h // 2)].float() / dim_h)
        )
        w_inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, dim_w, 2, dtype=torch.float32)[: (dim_w // 2)].float() / dim_w)
        )
        h_seq = torch.arange(height, device=hidden_states.device)
        w_seq = torch.arange(width, device=hidden_states.device)
        h_inv_freq = h_inv_freq.to(hidden_states.device)
        w_inv_freq = w_inv_freq.to(hidden_states.device)

        freqs_h = torch.outer(h_seq, h_inv_freq)
        freqs_w = torch.outer(w_seq, w_inv_freq)

        # Create position matrices: [height, 1, dim//4] and [1, width, dim//4]
        freqs_h = freqs_h.unsqueeze(1).expand(height, width, -1)
        freqs_w = freqs_w.unsqueeze(0).expand(height, width, -1)

        # Concatenate: [height, width, dim//2] -> [height, width, dim]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        freqs = freqs.reshape(height * width, -1)
        return (freqs.cos(), freqs.sin())


class GlmImageAdaLayerNormZero(nn.Module):
    """Adaptive LayerNorm with zero initialization for both image and text streams."""

    def __init__(self, embedding_dim: int, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.linear = nn.Linear(embedding_dim, 12 * dim, bias=True)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        dtype = hidden_states.dtype
        norm_hidden_states = self.norm(hidden_states).to(dtype=dtype)
        norm_encoder_hidden_states = self.norm_context(encoder_hidden_states).to(dtype=dtype)

        emb = self.linear(temb)
        (
            shift_msa,
            c_shift_msa,
            scale_msa,
            c_scale_msa,
            gate_msa,
            c_gate_msa,
            shift_mlp,
            c_shift_mlp,
            scale_mlp,
            c_scale_mlp,
            gate_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)

        hidden_states = norm_hidden_states * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_msa.unsqueeze(1)) + c_shift_msa.unsqueeze(1)

        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        )


class GlmImageAdaLayerNormContinuous(nn.Module):
    """Final AdaLN for output projection (no activation before Linear)."""

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # NO SiLU here
        emb = self.linear(conditioning_embedding.to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class KVCacheMode(Enum):
    """Mode for KV cache operations.

    - WRITE: Store the K/V tensors from condition images
    - READ: Concatenate cached K/V with current K/V
    - SKIP: Do not use cache (pass-through)
    """

    WRITE = "write"
    READ = "read"
    SKIP = "skip"


class GlmImageLayerKVCache:
    """KV cache for a single attention layer.

    Stores key and value tensors for image editing. The cache accumulates
    KV pairs during write mode and provides them during read mode.

    Shape convention (vllm-omni):
        key/value: [batch_size, seq_length, num_heads, head_dim]
    """

    def __init__(self):
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def store(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store or accumulate KV tensors.

        If cache is empty, stores the tensors directly.
        If cache is not empty, concatenates new tensors along seq_length dim.

        Args:
            key: Key tensor of shape [B, S, H, D]
            value: Value tensor of shape [B, S, H, D]
        """
        if self.k_cache is None:
            self.k_cache = key
            self.v_cache = value
        else:
            # Concatenate along sequence dimension (dim=1 for [B, S, H, D])
            self.k_cache = torch.cat([self.k_cache, key], dim=1)
            self.v_cache = torch.cat([self.v_cache, value], dim=1)

    def get(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get cached KV tensors.

        Returns:
            Tuple of (k_cache, v_cache), both may be None if cache is empty.
        """
        return self.k_cache, self.v_cache

    def clear(self) -> None:
        """Clear the cache."""
        self.k_cache = None
        self.v_cache = None

    @property
    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return self.k_cache is None

    def __repr__(self) -> str:
        if self.is_empty:
            return "GlmImageLayerKVCache(empty)"
        return f"GlmImageLayerKVCache(k_shape={self.k_cache.shape}, v_shape={self.v_cache.shape})"


class GlmImageKVCache:
    """Container for all layers' KV caches.

    Manages KV cache for all transformer layers in GLM-Image model.
    Provides a unified interface for setting mode and clearing cache.

    Args:
        num_layers: Number of transformer layers in the model.

    Example:
        kv_cache = GlmImageKVCache(num_layers=28)
        kv_cache.set_mode(KVCacheMode.WRITE)
        # ... process condition image ...
        kv_cache.set_mode(KVCacheMode.READ)
        # ... process target image ...
        kv_cache.clear()
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.caches = [GlmImageLayerKVCache() for _ in range(num_layers)]
        self._mode: KVCacheMode | None = None

    def __getitem__(self, layer_idx: int) -> GlmImageLayerKVCache:
        """Get cache for a specific layer.

        Args:
            layer_idx: Index of the layer (0-indexed).

        Returns:
            GlmImageLayerKVCache for the specified layer.

        Raises:
            IndexError: If layer_idx is out of range.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")
        return self.caches[layer_idx]

    def __len__(self) -> int:
        """Return number of layers."""
        return self.num_layers

    @property
    def mode(self) -> KVCacheMode | None:
        """Get current cache mode."""
        return self._mode

    def set_mode(self, mode: KVCacheMode | str | None) -> None:
        """Set cache mode for all layers.

        Args:
            mode: Cache mode (WRITE, READ, SKIP) or string ("write", "read", "skip").
                  Use None to disable cache operations.

        Raises:
            ValueError: If mode is an invalid string.
        """
        if mode is None:
            self._mode = None
        elif isinstance(mode, str):
            try:
                self._mode = KVCacheMode(mode.lower())
            except ValueError:
                raise ValueError(f"Invalid mode: '{mode}', must be one of 'write', 'read', 'skip'")
        else:
            self._mode = mode

    def clear(self) -> None:
        """Clear cache for all layers and reset mode."""
        for cache in self.caches:
            cache.clear()
        self._mode = None

    @property
    def is_empty(self) -> bool:
        """Check if all layer caches are empty."""
        return all(cache.is_empty for cache in self.caches)

    def __repr__(self) -> str:
        mode_str = self._mode.value if self._mode else "None"
        return f"GlmImageKVCache(num_layers={self.num_layers}, mode={mode_str}, is_empty={self.is_empty})"


class GlmImageAttention(nn.Module):
    """
    Joint attention for GLM-Image model using vllm-omni's optimized attention.

    This combines text and image streams for joint attention computation.
    Supports KV caching for image editing workflows via external cache.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        out_bias: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        # QKV projection (fused for efficiency)
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            disable_tp=True,
            bias=True,
        )

        # QK normalization (LayerNorm, not RMSNorm for GLM-Image)
        self.norm_q = nn.LayerNorm(head_dim, elementwise_affine=False, eps=eps)
        self.norm_k = nn.LayerNorm(head_dim, elementwise_affine=False, eps=eps)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim, bias=out_bias),
            nn.Dropout(0.0),
        )

        # RoPE and attention
        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: GlmImageLayerKVCache | None = None,
        kv_cache_mode: KVCacheMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for joint attention.

        Args:
            hidden_states: Image hidden states [B, img_seq_len, D]
            encoder_hidden_states: Text hidden states [B, text_seq_len, D]
            image_rotary_emb: Tuple of (cos, sin) for RoPE
            attention_mask: Optional attention mask for text tokens
            kv_cache: Optional layer KV cache for image editing
            kv_cache_mode: Cache mode (WRITE, READ, SKIP)

        Returns:
            Tuple of (image_hidden_states, text_hidden_states)
        """
        dtype = encoder_hidden_states.dtype
        batch_size, text_seq_length, _ = encoder_hidden_states.shape

        # Concatenate text and image: [text, image]
        hidden_states_combined = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # QKV projection
        qkv, _ = self.to_qkv(hidden_states_combined)
        query, key, value = qkv.chunk(3, dim=-1)

        # Reshape: [B, S, H*D] -> [B, S, H, D]
        query = query.unflatten(-1, (self.num_heads, -1))
        key = key.unflatten(-1, (self.num_heads, -1))
        value = value.unflatten(-1, (self.num_heads, -1))

        # QK normalization
        query = self.norm_q(query).to(dtype=dtype)
        key = self.norm_k(key).to(dtype=dtype)

        # Apply RoPE only to image tokens (not text tokens)
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            # Only apply RoPE to image part (after text_seq_length)
            query_img = query[:, text_seq_length:, :, :]
            key_img = key[:, text_seq_length:, :, :]
            from diffusers.models.embeddings import apply_rotary_emb

            query_img = apply_rotary_emb(query_img, image_rotary_emb, sequence_dim=1, use_real_unbind_dim=-2)
            # key_img = self.rope(key_img, cos, sin)
            key_img = apply_rotary_emb(key_img, image_rotary_emb, sequence_dim=1, use_real_unbind_dim=-2)
            query = torch.cat([query[:, :text_seq_length, :, :], query_img], dim=1)
            key = torch.cat([key[:, :text_seq_length, :, :], key_img], dim=1)

        # Handle KV cache for image editing
        if kv_cache is not None and kv_cache_mode is not None:
            if kv_cache_mode == KVCacheMode.WRITE:
                kv_cache.store(key, value)
            elif kv_cache_mode == KVCacheMode.READ:
                k_cached, v_cached = kv_cache.get()
                if k_cached is not None:
                    key = torch.cat([k_cached, key], dim=1)
                    value = torch.cat([v_cached, value], dim=1)
            # KVCacheMode.SKIP: do nothing

        # Attention computation
        hidden_states_out = self.attn(query, key, value)
        hidden_states_out = hidden_states_out.flatten(2, 3)
        hidden_states_out = hidden_states_out.to(dtype)

        # Output projection
        hidden_states_out = self.to_out(hidden_states_out)

        # Split back to text and image
        encoder_hidden_states_out = hidden_states_out[:, :text_seq_length, :]
        hidden_states_out = hidden_states_out[:, text_seq_length:, :]

        return hidden_states_out, encoder_hidden_states_out


class GlmImageTransformerBlock(nn.Module):
    """Single transformer block for GLM-Image."""

    def __init__(
        self,
        dim: int = 2560,
        num_attention_heads: int = 64,
        attention_head_dim: int = 40,
        time_embed_dim: int = 512,
    ) -> None:
        super().__init__()

        # 1. Attention with AdaLN
        self.norm1 = GlmImageAdaLayerNormZero(time_embed_dim, dim)
        self.attn1 = GlmImageAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
        )

        # 2. Feedforward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: GlmImageLayerKVCache | None = None,
        kv_cache_mode: KVCacheMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for transformer block.

        Args:
            hidden_states: Image hidden states
            encoder_hidden_states: Text hidden states
            temb: Timestep embedding
            image_rotary_emb: RoPE embeddings
            attention_mask: Text attention mask
            attention_kwargs: Additional attention arguments
            kv_cache: Layer-specific KV cache for image editing
            kv_cache_mode: Cache mode (WRITE, READ, SKIP)

        Returns:
            Tuple of (image_hidden_states, text_hidden_states)
        """
        # 1. Timestep conditioning via AdaLN
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1(hidden_states, encoder_hidden_states, temb)

        # 2. Attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + attn_encoder_hidden_states * c_gate_msa.unsqueeze(1)

        # 3. Feedforward
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (
            1 + c_scale_mlp.unsqueeze(1)
        ) + c_shift_mlp.unsqueeze(1)

        ff_output = self.ff(norm_hidden_states)
        ff_output_context = self.ff(norm_encoder_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + ff_output_context * c_gate_mlp.unsqueeze(1)

        return hidden_states, encoder_hidden_states


class GlmImageTransformer2DModel(CachedTransformer):
    """
    GLM-Image Transformer model for 2D image generation.

    This is the vllm-omni optimized version of the GLM-Image DiT model.

    Args:
        od_config: OmniDiffusionConfig containing model configuration.
            Transformer hyper-parameters (e.g. patch size / channels / heads) are read from
            `od_config.tf_model_config`.
    """

    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
    ):
        super().__init__()

        patch_size = od_config.tf_model_config.patch_size
        in_channels = od_config.tf_model_config.in_channels
        out_channels = od_config.tf_model_config.out_channels
        num_attention_heads = od_config.tf_model_config.num_attention_heads
        attention_head_dim = od_config.tf_model_config.attention_head_dim
        time_embed_dim = od_config.tf_model_config.time_embed_dim
        condition_dim = od_config.tf_model_config.condition_dim
        prior_vq_quantizer_codebook_size = od_config.tf_model_config.prior_vq_quantizer_codebook_size
        text_embed_dim = od_config.tf_model_config.text_embed_dim

        # Get num_layers from config if available
        model_config = od_config.tf_model_config
        if model_config is not None and hasattr(model_config, "num_layers"):
            num_layers = model_config.num_layers

        self.od_config = od_config
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # GlmImage uses 2 additional SDXL-like conditions - target_size, crop_coords
        pooled_projection_dim = 2 * 2 * condition_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 1. RoPE
        self.rope = GlmImageRotaryPosEmbed(attention_head_dim, patch_size, theta=10000.0)

        # 2. Patch & Text-timestep embedding
        self.image_projector = GlmImageImageProjector(in_channels, inner_dim, patch_size)
        self.glyph_projector = FeedForward(text_embed_dim, inner_dim, inner_dim=inner_dim, activation_fn="gelu")
        self.prior_token_embedding = nn.Embedding(prior_vq_quantizer_codebook_size, inner_dim)
        self.prior_projector = FeedForward(inner_dim, inner_dim, inner_dim=inner_dim, activation_fn="linear-silu")

        self.time_condition_embed = GlmImageCombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=pooled_projection_dim,
            timesteps_dim=time_embed_dim,
        )

        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                GlmImageTransformerBlock(inner_dim, num_attention_heads, attention_head_dim, time_embed_dim)
                for _ in range(num_layers)
            ]
        )

        # 4. Output projection
        self.norm_out = GlmImageAdaLayerNormContinuous(inner_dim, time_embed_dim, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        prior_token_id: torch.Tensor,
        prior_token_drop: torch.Tensor,
        timestep: torch.LongTensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: GlmImageKVCache | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        Forward pass of the GLM-Image Transformer.

        Args:
            hidden_states: Input latent tensor of shape [B, C, H, W].
            encoder_hidden_states: Text embeddings of shape [B, S, D].
            prior_token_id: Prior VQ token IDs.
            prior_token_drop: Mask for dropping prior tokens (CFG).
            timestep: Diffusion timestep.
            target_size: Target image size for conditioning.
            crop_coords: Crop coordinates for conditioning.
            attention_kwargs: Additional attention arguments.
            return_dict: Whether to return a dataclass.
            attention_mask: Optional attention mask for text tokens.
            image_rotary_emb: Pre-computed rotary embeddings.
            kv_cache: Optional KV cache for image editing. When provided,
                      the cache's mode determines behavior:
                      - WRITE: Store KV from condition images
                      - READ: Use cached KV during generation
                      - SKIP: No caching (same as None)

        Returns:
            Output tensor or Transformer2DModelOutput.
        """
        batch_size, num_channels, height, width = hidden_states.shape

        # Get KV cache mode
        kv_cache_mode = kv_cache.mode if kv_cache is not None else None

        # 1. RoPE
        if image_rotary_emb is None:
            image_rotary_emb = self.rope(hidden_states)
            # Move to correct device
            image_rotary_emb = (
                image_rotary_emb[0].to(hidden_states.device),
                image_rotary_emb[1].to(hidden_states.device),
            )

        # 2. Patch & Timestep embeddings
        p = self.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states = self.image_projector(hidden_states)
        encoder_hidden_states = self.glyph_projector(encoder_hidden_states)

        # Prior embedding with dropout
        prior_embedding = self.prior_token_embedding(prior_token_id)
        prior_embedding[prior_token_drop] *= 0.0
        prior_hidden_states = self.prior_projector(prior_embedding)
        hidden_states = hidden_states + prior_hidden_states

        # Timestep conditioning
        temb = self.time_condition_embed(timestep, target_size, crop_coords, hidden_states.dtype)

        # 3. Transformer blocks
        for layer_idx, block in enumerate(self.transformer_blocks):
            # Get layer-specific KV cache if available
            layer_kv_cache = kv_cache[layer_idx] if kv_cache is not None else None

            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                attention_mask,
                attention_kwargs,
                kv_cache=layer_kv_cache,
                kv_cache_mode=kv_cache_mode,
            )

        # 4. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify: [B, H'*W', C*p*p] -> [B, C, H, W]
        hidden_states = hidden_states.reshape(batch_size, post_patch_height, post_patch_width, -1, p, p)
        output = hidden_states.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights from pretrained checkpoint.

        This method handles the mapping from diffusers weight names to vllm-omni weight names,
        especially for fused QKV projections.
        """
        stacked_params_mapping = [
            # Fused QKV projection: to_q, to_k, to_v -> to_qkv
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
        ]

        params_dict = dict(self.named_parameters())

        # Also include buffers (for any beta/eps parameters)
        for name, buffer in self.named_buffers():
            params_dict[name] = buffer

        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Handle fused QKV projections
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # Map diffusers name to vllm-omni name
                name = name.replace(weight_name, param_name)

                if name not in params_dict:
                    logger.warning(f"Skipping weight {name} - not found in model")
                    break

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Standard weight loading (not fused)
                if name not in params_dict:
                    logger.warning(f"Skipping weight {name} - not found in model")
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params

    def create_kv_cache(self) -> GlmImageKVCache:
        """
        Create a KV cache for image editing.

        Returns a new GlmImageKVCache instance sized for this model's
        number of transformer layers. Use this for image editing workflows.

        Example:
            kv_cache = transformer.create_kv_cache()
            kv_cache.set_mode("write")
            transformer(condition_image, kv_cache=kv_cache)
            kv_cache.set_mode("read")
            for t in timesteps:
                transformer(noisy_target, kv_cache=kv_cache)
            kv_cache.clear()

        Returns:
            GlmImageKVCache instance with correct number of layers.
        """
        return GlmImageKVCache(num_layers=len(self.transformer_blocks))

    @property
    def num_layers(self) -> int:
        """Return number of transformer layers."""
        return len(self.transformer_blocks)

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype of model parameters."""
        return next(self.parameters()).dtype

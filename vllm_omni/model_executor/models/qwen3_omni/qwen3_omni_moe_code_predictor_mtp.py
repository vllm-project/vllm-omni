"""Qwen3-Omni Code Predictor with MTP (Multi-Token Prediction) support.

This module implements the code predictor component for Qwen3-Omni talker models.

The code predictor generates residual RVQ (Residual Vector Quantization) codes
autoregressively, predicting layers 1 to N based on layer-0 codes from the talker.
"""

from collections import namedtuple
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Cache, PretrainedConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeRotaryEmbedding
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = init_logger(__name__)


# ============================================================================
# Rotary Embeddings and Helper Functions
# ============================================================================


class Qwen3OmniCodePredictorRotaryEmbedding(nn.Module):
    """Rotary positional embeddings for the code predictor."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position = position_ids[:, None, :].float()
        freqs = (inv_freq @ position).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for grouped query attention."""
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


# ============================================================================
# Code Predictor Attention Layer
# ============================================================================


class Qwen3OmniCodePredictorAttention(nn.Module):
    """Multi-head self-attention for code predictor with vLLM optimization."""

    def __init__(self, config, layer_idx: int, vllm_config: VllmConfig = None):
        super().__init__()

        self.num_heads = config.code_predictor_config.num_attention_heads
        self.num_key_value_heads = config.code_predictor_config.num_key_value_heads
        self.head_dim = getattr(
            config.code_predictor_config,
            "head_dim",
            config.code_predictor_config.hidden_size // config.code_predictor_config.num_attention_heads,
        )
        self.hidden_size = config.code_predictor_config.hidden_size

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Query/Key normalization
        self.q_norm = RMSNorm(self.head_dim, eps=config.code_predictor_config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.code_predictor_config.rms_norm_eps)
        self.is_causal = True
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q_heads = q.transpose(1, 2).contiguous()
        k_heads = k.transpose(1, 2).contiguous()
        q_heads, k_heads = apply_rotary_pos_emb(q_heads, k_heads, cos, sin)
        v_heads = v.transpose(1, 2).contiguous()

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k_heads, v_heads = past_key_values.update(k_heads, v_heads, self.layer_idx, cache_kwargs)

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        # Try attention backends in order of preference, with runtime error handling
        # This handles cases where the backend is registered but not actually available
        attention_backends = ["flash_attention_2", "xformers", "eager", "sdpa"]
        attn_output = None
        last_error = None

        for backend_name in attention_backends:
            if backend_name not in ALL_ATTENTION_FUNCTIONS:
                continue

            try:
                attention_interface = ALL_ATTENTION_FUNCTIONS[backend_name]
                attn_output, _ = attention_interface(
                    self,
                    q_heads,
                    k_heads,
                    v_heads,
                    None,
                    dropout=0.0 if not self.training else getattr(self, "attention_dropout", 0.0),
                    scaling=self.head_dim**-0.5,
                    sliding_window=None,
                    use_cache=use_cache,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    output_attentions=False,
                )
                # Success - log fallback if not using flash_attention_2
                if backend_name != "flash_attention_2":
                    logger.warning_once(
                        f"Using {backend_name} attention backend (flash_attention_2 not available or failed)"
                    )
                break
            except (ValueError, ImportError, RuntimeError, AttributeError) as e:
                # Store error and try next backend
                last_error = e
                continue

        if attn_output is None:
            raise RuntimeError(
                f"All attention backends failed. Last error: {last_error}. "
                "Please install flash-attn, or ensure PyTorch's scaled_dot_product_attention is available."
            )
        attn_output = attn_output.reshape(*(hidden_states.shape[:-1]), -1).contiguous()

        attn_output = self.o_proj(attn_output)
        return attn_output


# ============================================================================
# Code Predictor MLP Layer
# ============================================================================


class Qwen3OmniCodePredictorMLP(nn.Module):
    """Feed-forward network for code predictor."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.code_predictor_config.hidden_size, config.code_predictor_config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.code_predictor_config.hidden_size, config.code_predictor_config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.code_predictor_config.intermediate_size, config.code_predictor_config.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


# ============================================================================
# MTP Layer (Multi-Token Prediction Layer)
# ============================================================================


class Qwen3OmniCodePredictorMTPLayer(nn.Module):
    """MTP layer for speculative decoding - predicts next residual code layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Qwen3OmniCodePredictorDecoderLayer
        self.self_attn = Qwen3OmniCodePredictorAttention(
            config,
            layer_idx,
            vllm_config=type(
                "VllmConfig",
                (),
                {"cache_config": cache_config, "quant_config": quant_config, "model_config": model_config},
            )(),
        )
        self.mlp = Qwen3OmniCodePredictorMLP(config)
        self.input_layernorm = RMSNorm(
            config.code_predictor_config.hidden_size, eps=config.code_predictor_config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.code_predictor_config.hidden_size, eps=config.code_predictor_config.rms_norm_eps
        )

    def mtp_block(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, causal_mask, cos, sin, past_key_values, cache_position, use_cache, position_ids
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None, "inputs_embeds required for MTP"

        # Mask position 0 (not needed for MTP)
        inputs_embeds[positions == 0] = 0

        hidden_states = torch.cat([inputs_embeds, previous_hidden_states], dim=-1)

        # Get position info for RoPE
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        # Get RoPE embeddings
        head_dim = self.self_attn.head_dim
        rotary_emb = Qwen3OmniCodePredictorRotaryEmbedding(
            head_dim, max_position_embeddings=self.config.code_predictor_config.max_position_embeddings
        )
        rotary_emb = Qwen3OmniMoeRotaryEmbedding(self.config)
        cos, sin = rotary_emb(hidden_states, position_ids)

        # Create causal mask
        causal_mask = (
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        causal_mask = causal_mask.masked_fill(causal_mask, float("-inf"))

        # Forward through MTP block
        hidden_states = self.mtp_block(hidden_states, causal_mask, cos, sin)

        return hidden_states


class Qwen3OmniCodePredictorBaseModel(nn.Module):
    """
    Base model for code predictor - matches HF Qwen3OmniMoeTalkerCodePredictorModel structure.

    This is a simple transformer that processes inputs_embeds and outputs hidden states.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.code_predictor_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups

        # Codec embeddings (for layers 1-num_code_groups-1)
        self.codec_embedding = nn.ModuleList(
            [
                VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                )
                for _ in range(config.num_code_groups - 1)
            ]
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3OmniCodePredictorMTPLayer(
                    vllm_config.model_config.hf_config,
                    f"{prefix}.layers.{idx}",
                    model_config=vllm_config.model_config,
                    layer_idx=idx,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                )
                for idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE
        self.rotary_emb = Qwen3OmniMoeRotaryEmbedding(config=config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Forward pass matching HF structure.

        Args:
            inputs_embeds: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor
            position_ids: Optional position IDs tensor
            past_key_values: Optional cached key-value pairs
            use_cache: Whether to use cache
            cache_position: Optional cache position tensor
            **kwargs: Additional keyword arguments

        Returns:
            Named tuple with .last_hidden_state and .past_key_values attributes
        """
        batch_size, seq_len, _ = inputs_embeds.shape

        # Create positions tensor if not provided
        # positions must be [num_tokens] or [batch_size, seq_len]
        if position_ids is None:
            if cache_position is not None:
                position_ids = cache_position  # [num_tokens]
            else:
                position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)  # [1, seq_len]
        else:
            position_ids = position_ids.flatten()  # Ensure [num_tokens]

        # Extract cos/sin from rotary_emb cache
        # The cos_sin_cache is [max_pos, rotary_dim * 2]
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)

        # Create causal mask
        device = inputs_embeds.device
        causal_mask = (
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        causal_mask = causal_mask.masked_fill(causal_mask, float("-inf"))

        # Forward through decoder layers
        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer.mtp_block(
                hidden_states, causal_mask, cos, sin, past_key_values, cache_position, use_cache, position_ids
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Return in HF-compatible format
        Output = namedtuple("Output", ["last_hidden_state", "past_key_values"])
        return Output(last_hidden_state=hidden_states, past_key_values=None)  # [batch, num_code_groups-1, hidden_size]

    def get_input_embeddings(self):
        """Return codec embeddings for HF compatibility."""
        return self.codec_embedding


class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    """
    Code predictor wrapper matching HF structure.

    Structure:
    - self.model: Qwen3OmniCodePredictorBaseModel (transformer)
    - self.lm_head: ModuleList of output heads
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        talker_code_predictor_config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.config = talker_code_predictor_config
        self.vocab_size = self.config.code_predictor_config.vocab_size
        self.num_code_groups = self.config.code_predictor_config.num_code_groups

        # Base transformer model (matches HF structure)
        self.model = Qwen3OmniCodePredictorBaseModel(vllm_config=vllm_config, prefix=prefix)

        # Output heads for each residual layer (1-num_layers-1)
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(
                    self.config.code_predictor_config.hidden_size,
                    self.config.code_predictor_config.vocab_size,
                    bias=False,
                )
                for _ in range(self.num_code_groups - 1)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # [batch, seq_len, hidden_size]
        layer_idx: int,  # Which layer to predict (0-num_layers-2 for layers 1-num_layers-1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for code predictor.

        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            layer_idx: Which residual layer to predict (0-num_layers-2 for layers 1-num_layers-1)

        Returns:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            hidden_states: Output hidden states [batch_size, seq_len, hidden_size]
        """
        # Pass through base model
        hidden_states = self.model(inputs_embeds)

        # Get logits from corresponding head
        logits = self.lm_head[layer_idx](hidden_states)

        return logits, hidden_states

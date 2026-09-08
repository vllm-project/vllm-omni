# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the PersonaPlex Helium temporal transformer."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

__all__ = ["HeliumConfig"]


class HeliumConfig(PretrainedConfig):
    """Minimal HF config for the Moshi temporal LM backbone.

    The defaults are measured from the PersonaPlex checkpoint rather than inferred
    from Moshi's higher-level kwargs.
    """

    model_type = "helium"
    keys_to_ignore_at_inference = ("past_key_values",)

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        head_dim: int = 128,
        num_key_value_heads: int = 32,
        intermediate_size: int = 11264,
        max_position_embeddings: int = 3000,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-8,
        vocab_size: int = 32000,
        sliding_window: int = 3000,
        hidden_act: str = "silu",
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size
        self.sliding_window = sliding_window
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_cache = use_cache

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HuggingFace config for m-a-p/YuE2-3B.

The checkpoint's own ``config.json`` already carries every field the AR
backbone needs (it is Qwen3-1.7B-shaped with q_norm/k_norm and an extended
184,704-entry vocabulary), plus the acoustic-side fields (``latent_dim``,
``max_latent_frames``, ``timestep_shift``). Registering this class lets vLLM
build a ``ModelConfig`` without ``trust_remote_code``: the package's own
``Yue2ForCausalLM`` replaces the repository's auto_map modeling code, and the
deploy YAML pins ``hf_overrides.architectures`` to it.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class Yue2Config(PretrainedConfig):
    """Config for the YuE2-3B AR–NAR Mixture-of-Transformers checkpoint."""

    model_type = "yue2"

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 6144,
        vocab_size: int = 184704,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 24576,
        tie_word_embeddings: bool = False,
        latent_dim: int = 64,
        max_latent_frames: int = 24576,
        timestep_shift: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.latent_dim = latent_dim
        self.max_latent_frames = max_latent_frames
        self.timestep_shift = timestep_shift
        # The checkpoint's config.json (like upstream's qwen_config derivative,
        # fast.py) omits the standard Qwen3 fields the AR backbone expects;
        # synthesize the same defaults upstream does so vLLM's Qwen3Model and
        # MLP construct without AttributeError.
        self.hidden_act = str(kwargs.pop("hidden_act", "silu"))
        self.attention_bias = bool(kwargs.pop("attention_bias", False))
        self.attention_dropout = float(kwargs.pop("attention_dropout", 0.0))
        self.use_sliding_window = bool(kwargs.pop("use_sliding_window", False))
        self.sliding_window = kwargs.pop("sliding_window", None)
        self.max_window_layers = int(kwargs.pop("max_window_layers", 0))


AutoConfig.register("yue2", Yue2Config)

__all__ = ["Yue2Config"]

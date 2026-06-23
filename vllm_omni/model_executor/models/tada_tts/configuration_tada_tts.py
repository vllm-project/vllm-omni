"""TadaTTSConfig: vLLM-Omni config for HumeAI/tada-1b and tada-3b-ml."""

from __future__ import annotations

import math

from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class TadaTTSConfig(PretrainedConfig):
    """Exposes the fields vLLM-Omni needs from the TADA checkpoint. Defaults match tada-1b."""

    model_type = "tada_tts"

    def __init__(
        self,
        # LLM backbone dims (from Llama-3.2-1B / Llama-3.2-3B)
        hidden_size: int = 2048,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 8192,
        vocab_size: int = 128256,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        rope_scaling: dict | None = None,
        # Acoustic / diffusion dims
        acoustic_dim: int = 512,
        num_time_classes: int = 256,
        shift_acoustic: int = 5,
        acoustic_from_nth_hidden_state: int = -1,
        # Acoustic feature (de)normalisation (diffusion runs in normalised space)
        acoustic_mean: float = 0.0,
        acoustic_std: float = 1.5,
        # Diffusion head
        head_layers: int = 6,
        head_ffn_ratio: float = 4.0,
        diffusion_head_type: str = "vibevoice",
        bottleneck_dim: int | None = None,
        # Tokenizer
        tokenizer_name: str = "meta-llama/Llama-3.2-1B",
        # Output sample rate (set by the codec: 50 Hz frames × 480 upsample)
        output_sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.acoustic_dim = acoustic_dim
        self.num_time_classes = num_time_classes
        self.shift_acoustic = shift_acoustic
        self.acoustic_from_nth_hidden_state = acoustic_from_nth_hidden_state
        self.acoustic_mean = acoustic_mean
        self.acoustic_std = acoustic_std
        self.head_layers = head_layers
        self.head_ffn_ratio = head_ffn_ratio
        self.diffusion_head_type = diffusion_head_type
        self.bottleneck_dim = bottleneck_dim
        self.tokenizer_name = tokenizer_name
        self.output_sample_rate = output_sample_rate

    @property
    def num_time_bits(self) -> int:
        return math.ceil(math.log2(self.num_time_classes))

    @property
    def time_dim(self) -> int:
        return 2 * self.num_time_bits

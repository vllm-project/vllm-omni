"""Configuration classes for Fish Speech S2 Pro (fish_qwen3_omni).

The HuggingFace config uses field names from the original Fish Speech codebase
(dim, n_head, n_layer, etc.).  This module re-exports them under the standard
Transformers/Qwen3 attribute names so that vLLM's ``Qwen3Model`` can consume
the Slow AR config transparently.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class FishSpeechSlowARConfig(PretrainedConfig):
    """Slow AR (text_model) config -- Qwen3-based transformer.

    Maps Fish Speech field names to Qwen3-compatible attribute names so
    ``vllm.model_executor.models.qwen3.Qwen3Model`` works out of the box.
    """

    model_type = "fish_qwen3"

    def __init__(
        self,
        vocab_size: int = 155776,
        dim: int = 2560,
        n_head: int = 32,
        n_local_heads: int = 8,
        head_dim: int = 128,
        n_layer: int = 36,
        intermediate_size: int = 9728,
        attention_qk_norm: bool = True,
        rope_base: float = 1_000_000.0,
        max_seq_len: int = 32768,
        tie_word_embeddings: bool = True,
        codebook_size: int = 4096,
        num_codebooks: int = 10,
        semantic_begin_id: int = 0,
        semantic_end_id: int = 0,
        scale_codebook_embeddings: bool = False,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ):
        # Fish Speech field names → standard Transformers/Qwen3 names.
        self.hidden_size = dim
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_local_heads
        self.head_dim = head_dim
        self.num_hidden_layers = n_layer
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_seq_len
        self.rope_theta = rope_base
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = "silu"
        self.attention_bias = False
        self.attention_qk_norm = attention_qk_norm

        # Fish Speech codec / codebook fields.
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.semantic_begin_id = semantic_begin_id
        self.semantic_end_id = semantic_end_id
        self.scale_codebook_embeddings = scale_codebook_embeddings

        super().__init__(
            vocab_size=vocab_size,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class FishSpeechFastARConfig(PretrainedConfig):
    """Fast AR (audio_decoder) config -- 4-layer residual codebook predictor."""

    model_type = "fish_qwen3_audio_decoder"

    def __init__(
        self,
        vocab_size: int = 4096,
        num_codebooks: int = 10,
        dim: int = 2560,
        n_head: int = 32,
        n_local_heads: int = 8,
        head_dim: int = 128,
        n_layer: int = 4,
        intermediate_size: int = 9728,
        max_seq_len: int = 11,
        text_dim: int = 2560,
        audio_hidden_dim: int = 5120,
        attention_qk_norm: bool = False,
        rope_base: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ):
        self.hidden_size = dim
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_local_heads
        self.head_dim = head_dim
        self.num_hidden_layers = n_layer
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_seq_len
        self.rope_theta = rope_base
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = "silu"
        self.attention_bias = False
        self.attention_qk_norm = attention_qk_norm
        self.num_codebooks = num_codebooks
        self.text_dim = text_dim
        self.audio_hidden_dim = audio_hidden_dim

        super().__init__(vocab_size=vocab_size, **kwargs)


class FishSpeechConfig(PretrainedConfig):
    """Top-level config for Fish Speech S2 Pro (fish_qwen3_omni).

    Wraps ``text_config`` (Slow AR) and ``audio_decoder_config`` (Fast AR).
    """

    model_type = "fish_qwen3_omni"
    sub_configs = {
        "text_config": FishSpeechSlowARConfig,
        "audio_decoder_config": FishSpeechFastARConfig,
    }

    def __init__(
        self,
        text_config: dict | FishSpeechSlowARConfig | None = None,
        audio_decoder_config: dict | FishSpeechFastARConfig | None = None,
        semantic_start_token_id: int = 151678,
        semantic_end_token_id: int = 155773,
        audio_pad_token_id: int = 151677,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = FishSpeechSlowARConfig(**text_config)
        if isinstance(audio_decoder_config, dict):
            audio_decoder_config = FishSpeechFastARConfig(**audio_decoder_config)

        self.text_config = text_config or FishSpeechSlowARConfig()
        self.audio_decoder_config = audio_decoder_config or FishSpeechFastARConfig()

        self.semantic_start_token_id = semantic_start_token_id
        self.semantic_end_token_id = semantic_end_token_id
        self.audio_pad_token_id = audio_pad_token_id

        # Propagate semantic IDs into text_config for convenience.
        self.text_config.semantic_begin_id = semantic_start_token_id
        self.text_config.semantic_end_id = semantic_end_token_id

        super().__init__(**kwargs)

    def get_text_config(self, **kwargs) -> FishSpeechSlowARConfig:
        return self.text_config

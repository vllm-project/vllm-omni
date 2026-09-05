# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Configuration classes for Breeze-TTS-2."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3 import Qwen3Config


class BreezeTTS2Config(PretrainedConfig):
    """Configuration for ``BreezeForConditionalGeneration``."""

    model_type = "breeze"

    def __init__(
        self,
        backbone_config: dict | Qwen3Config | None = None,
        text_encoder_config: dict | BreezeTTS2TextEncoderConfig | None = None,
        depth_decoder_config: dict | BreezeTTS2DepthDecoderConfig | None = None,
        codec_config: dict | None = None,
        architectures: list[str] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        audio_embed_size: int = 2048,
        audio_eos_token_id: int = 262145,
        audio_num_codebooks: int = 16,
        audio_token_id: int = 262144,
        audio_vocab_size: int = 2051,
        backbone_flavor: str = "llama-1B",
        backbone_model_name_or_path: str | None = None,
        backbone_model_type: str = "qwen3",
        bos_token_id: int = 2,
        codebook_eos_token_id: int = 0,
        codebook_pad_token_id: int = 2050,
        decoder_flavor: str = "llama-100M",
        depth_header_loss_weight: float = 1.0,
        dtype: str = "bfloat16",
        eos_token_id: int = 1,
        head_dim: int = 128,
        hidden_act: str = "silu",
        hidden_size: int = 2048,
        initializer_range: float = 0.02,
        intermediate_size: int = 6144,
        max_position_embeddings: int = 2048,
        mlp_bias: bool = False,
        num_attention_heads: int = 16,
        num_codebooks: int = 16,
        num_hidden_layers: int = 28,
        num_key_value_heads: int = 8,
        pad_token_id: int = 0,
        rms_norm_eps: float = 1e-5,
        rope_scaling: dict | None = None,
        rope_theta: float = 500_000.0,
        text_encoder_bucket_max_length_ratio: float = 4.0,
        text_encoder_lora_config: dict | None = None,
        text_encoder_proj_type: str = "linear",
        text_encoder_special_tokens_config: dict | None = None,
        text_vocab_size: int = 262158,
        tie_codebooks_embeddings: bool = True,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        vocab_size: int = 2051,
        sampling_rate: int = 24000,
        codec_model_name_or_path: str = "kyutai/mimi",
        **kwargs: Any,
    ) -> None:
        serialized_rope_parameters = kwargs.pop("rope_parameters", None)
        if rope_scaling is None and serialized_rope_parameters is not None:
            rope_scaling = serialized_rope_parameters
        # Breeze stores the Qwen3 backbone in backbone_config. This follows the
        # MOSS-TTS pattern and deliberately does not declare sub_configs.
        if backbone_config is None:
            backbone_config = {}
        if isinstance(backbone_config, dict):
            backbone_config = dict(backbone_config)
            backbone_config.pop("model_type", None)
            self.backbone_config = Qwen3Config(**backbone_config)
        else:
            self.backbone_config = backbone_config

        if text_encoder_config is None:
            text_encoder_config = {}
        if isinstance(text_encoder_config, dict):
            text_encoder_config = dict(text_encoder_config)
            text_encoder_config.pop("model_type", None)
            self.text_encoder_config = BreezeTTS2TextEncoderConfig(**text_encoder_config)
        else:
            self.text_encoder_config = text_encoder_config

        if depth_decoder_config is None:
            depth_decoder_config = {}
        if isinstance(depth_decoder_config, dict):
            depth_decoder_config = dict(depth_decoder_config)
            depth_decoder_config.pop("model_type", None)
            self.depth_decoder_config = BreezeTTS2DepthDecoderConfig(**depth_decoder_config)
        else:
            self.depth_decoder_config = depth_decoder_config

        # Nested configs must exist before the parent constructor: Transformers
        # 5.x may call get_text_config() during config validation.
        super().__init__(
            architectures=architectures,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.audio_embed_size = audio_embed_size
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_num_codebooks = audio_num_codebooks
        self.audio_token_id = audio_token_id
        self.audio_vocab_size = audio_vocab_size
        # Keep Mimi's nested config as a plain checkpoint dictionary. Stage 1
        # owns codec construction; stage 0 only needs codebook_size here.
        self.codec_config = codec_config
        self.backbone_flavor = backbone_flavor
        self.backbone_model_name_or_path = backbone_model_name_or_path
        self.backbone_model_type = backbone_model_type
        self.codebook_eos_token_id = codebook_eos_token_id
        self.codebook_pad_token_id = codebook_pad_token_id
        self.decoder_flavor = decoder_flavor
        self.depth_header_loss_weight = depth_header_loss_weight
        self.dtype = dtype
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.mlp_bias = mlp_bias
        self.num_attention_heads = num_attention_heads
        self.num_codebooks = num_codebooks
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        if rope_scaling is not None:
            # Preserve the upstream legacy field while also satisfying the
            # normalized ``rope_parameters`` validation added in Transformers 5.
            rope_scaling = dict(rope_scaling)
            rope_scaling.setdefault("rope_type", rope_scaling.get("type", "default"))
            rope_scaling.setdefault("rope_theta", rope_theta)
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.text_encoder_bucket_max_length_ratio = text_encoder_bucket_max_length_ratio
        self.text_encoder_lora_config = text_encoder_lora_config or {}
        self.text_encoder_proj_type = text_encoder_proj_type
        self.text_encoder_special_tokens_config = text_encoder_special_tokens_config or {}
        self.text_vocab_size = text_vocab_size
        self.tie_codebooks_embeddings = tie_codebooks_embeddings
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.sampling_rate = sampling_rate
        self.codec_model_name_or_path = codec_model_name_or_path

        if not getattr(self, "architectures", None):
            self.architectures = ["BreezeForConditionalGeneration"]

    def get_text_config(self, **_: Any) -> Qwen3Config:
        return self.backbone_config


class BreezeTTS2TextEncoderConfig(PretrainedConfig):
    """Configuration for Breeze's T5Gemma2 text encoder."""

    model_type = "t5gemma2_text"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 6912,
        num_hidden_layers: int = 26,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 1,
        head_dim: int = 256,
        max_position_embeddings: int = 32768,
        hidden_activation: str = "gelu_pytorch_tanh",
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10_000.0,
        sliding_window: int = 512,
        query_pre_attn_scalar: int = 256,
        eoi_token_index: int = 256000,
        vocab_size: int = 262158,
        layer_types: list[str] | None = None,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        dropout_rate: float = 0.0,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        use_bidirectional_attention: bool = False,
        use_cache: bool = True,
        requires_grad: bool = False,
        preferred_attn_implementation: str = "flash_attention_2",
        dtype: str = "bfloat16",
        is_encoder_decoder: bool = False,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("model_type", None)

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_activation = hidden_activation
        self.hidden_act = hidden_activation
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.eoi_token_index = eoi_token_index
        self.vocab_size = vocab_size
        if layer_types is None:
            # T5Gemma2 uses sliding attention except every sixth layer.
            layer_types = [
                "full_attention" if (index + 1) % 6 == 0 else "sliding_attention" for index in range(num_hidden_layers)
            ]
        self.layer_types = list(layer_types)
        if rope_parameters is None:
            rope_parameters = {
                "full_attention": {
                    "factor": 8.0,
                    "rope_theta": 1_000_000.0,
                    "rope_type": "linear",
                },
                "sliding_attention": {
                    "rope_theta": 10_000.0,
                    "rope_type": "default",
                },
            }
        self.rope_parameters = dict(rope_parameters)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.use_bidirectional_attention = use_bidirectional_attention
        self.use_cache = use_cache
        self.requires_grad = requires_grad
        self.preferred_attn_implementation = preferred_attn_implementation
        self.dtype = dtype


class BreezeTTS2DepthDecoderConfig(PretrainedConfig):
    """Configuration for Breeze's per-frame depth decoder."""

    model_type = "breeze_depth_decoder_model"

    def __init__(
        self,
        hidden_size: int = 1024,
        backbone_hidden_size: int = 2048,
        audio_embed_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        max_position_embeddings: int = 33,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500_000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        num_codebooks: int = 16,
        vocab_size: int = 2051,
        codebook_loss_weights: list[int] | None = None,
        dtype: str = "bfloat16",
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("model_type", None)
        # Transformers 5 serializes the normalized field as
        # ``rope_parameters`` while the upstream Breeze checkpoint calls it
        # ``rope_scaling``.  Normalize before ``PretrainedConfig`` validates
        # so ``to_dict() -> BreezeTTS2DepthDecoderConfig(**dict)`` round-trips.
        serialized_rope_parameters = kwargs.pop("rope_parameters", None)
        if rope_scaling is None:
            rope_scaling = {
                "factor": 32.0,
                "high_freq_factor": 0.0078125,
                "low_freq_factor": 0.001953125,
                "original_max_position_embeddings": 16,
                "rope_type": "llama3",
            }
        if codebook_loss_weights is None:
            codebook_loss_weights = [3]
        if num_codebooks <= 0 or vocab_size <= 0:
            raise ValueError("num_codebooks and vocab_size must be positive")
        rope_parameters = dict(serialized_rope_parameters or rope_scaling)
        rope_parameters.setdefault("rope_type", rope_parameters.get("type", "default"))
        rope_parameters.setdefault("rope_theta", rope_theta)

        super().__init__(
            vocab_size=vocab_size,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        # The vLLM depth decoder is a plain ``nn.Module``, so it does not get
        # Transformers' PreTrainedModel default attention initialization.
        self._attn_implementation = getattr(self, "_attn_implementation", None) or "eager"
        self.hidden_size = hidden_size
        self.backbone_hidden_size = backbone_hidden_size
        self.audio_embed_size = audio_embed_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_parameters = dict(getattr(self, "rope_parameters", None) or rope_parameters)
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        # Transformers 4 exposes this as a plain attribute; Transformers 5
        # implements ``rope_scaling`` as a property backed by rope_parameters.
        if not hasattr(type(self), "rope_scaling"):
            self.rope_scaling = self.rope_parameters
        self.rope_type = rope_scaling.get("rope_type", "default")
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.codebook_loss_weights = list(codebook_loss_weights)
        self.dtype = dtype


AutoConfig.register("breeze", BreezeTTS2Config, exist_ok=True)


__all__ = [
    "BreezeTTS2Config",
    "BreezeTTS2DepthDecoderConfig",
    "BreezeTTS2TextEncoderConfig",
]

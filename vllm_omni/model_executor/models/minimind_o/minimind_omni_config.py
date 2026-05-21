from __future__ import annotations

import math
from typing import Any

from transformers import AutoConfig, PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    _text_keys = (
        "hidden_size",
        "num_hidden_layers",
        "use_moe",
        "dropout",
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "flash_attn",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_act",
        "intermediate_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
        "tie_word_embeddings",
        "inference_rope_scaling",
        "rope_scaling",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "norm_topk_prob",
        "router_aux_loss_coef",
    )

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        use_moe: bool = False,
        dropout: float = 0.0,
        vocab_size: int = 6400,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        flash_attn: bool = True,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        intermediate_size: int | None = None,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        tie_word_embeddings: bool = True,
        inference_rope_scaling: bool = False,
        rope_scaling: dict[str, Any] | None = None,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int | None = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.flash_attn = flash_attn
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size or math.ceil(hidden_size * math.pi / 64) * 64
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = (
            rope_scaling
            if rope_scaling is not None
            else {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

        # MoE specific configs (ignored if use_moe is False).
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size or self.intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

    def get_text_config(self, **kwargs: Any) -> "MiniMindConfig":
        return self


class MiniMindOmniVisionConfig(PretrainedConfig):
    model_type = "minimind-o-vision"

    def __init__(
        self,
        hidden_size: int = 768,
        image_ids: list[int] | None = None,
        image_special_token: str = "<|image_pad|>",
        image_hidden_size: int = 768,
        image_token_len: int = 64,
        vision_model_path: str = "./model/siglip2-base-p32-256-ve",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_ids = image_ids or [12]
        self.image_special_token = image_special_token
        self.image_hidden_size = image_hidden_size
        self.image_token_len = image_token_len
        self.vision_model_path = vision_model_path


class MiniMindOmniAudioConfig(PretrainedConfig):
    model_type = "minimind-o-audio"

    def __init__(
        self,
        hidden_size: int = 768,
        audio_ids: list[int] | None = None,
        audio_special_token: str = "<|audio_pad|>",
        audio_hidden_size: int = 512,
        audio_encoder_path: str = "./model/SenseVoiceSmall",
        audio_vocab_size: int = 2112,
        audio_pad_token: int = 2049,
        audio_stop_token: int = 2050,
        audio_spk_token: int = 2051,
        audio_sample_rate: int = 16000,
        audio_target_channels: int = 1,
        max_audio_tokens: int = 3000,
        spk_emb_size: int = 192,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.audio_ids = audio_ids or [16]
        self.audio_special_token = audio_special_token
        self.audio_hidden_size = audio_hidden_size
        self.audio_encoder_path = audio_encoder_path
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token = audio_pad_token
        self.audio_stop_token = audio_stop_token
        self.audio_spk_token = audio_spk_token
        self.audio_sample_rate = audio_sample_rate
        self.audio_target_channels = audio_target_channels
        self.max_audio_tokens = max_audio_tokens
        self.spk_emb_size = spk_emb_size


class OmniConfig(MiniMindConfig):
    model_type = "minimind-o"
    sub_configs = {
        "text_config": MiniMindConfig,
        "vision_config": MiniMindOmniVisionConfig,
        "audio_config": MiniMindOmniAudioConfig,
    }

    def __init__(
        self,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        vision_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_config: dict[str, Any] | PretrainedConfig | None = None,
        num_talker_hidden_layers: int = 4,
        talker_hidden_size: int = 768,
        audio_ids: list[int] | None = None,
        audio_special_token: str = "<|audio_pad|>",
        audio_hidden_size: int = 512,
        audio_encoder_path: str = "./model/SenseVoiceSmall",
        audio_vocab_size: int = 2112,
        audio_pad_token: int = 2049,
        audio_stop_token: int = 2050,
        audio_spk_token: int = 2051,
        audio_sample_rate: int = 16000,
        audio_target_channels: int = 1,
        max_audio_tokens: int = 3000,
        spk_emb_size: int = 192,
        think_end_ids: list[int] | None = None,
        image_ids: list[int] | None = None,
        image_special_token: str = "<|image_pad|>",
        image_hidden_size: int = 768,
        image_token_len: int = 64,
        vision_model_path: str = "./model/siglip2-base-p32-256-ve",
        bridge_layer: int | None = None,
        **kwargs: Any,
    ) -> None:
        text_defaults = {
            key: kwargs[key]
            for key in MiniMindConfig._text_keys
            if key in kwargs
        }
        if isinstance(text_config, dict):
            text_config = MiniMindConfig(**{**text_defaults, **text_config})
        elif text_config is None:
            text_config = MiniMindConfig(**text_defaults)

        text_cfg = text_config
        hidden_size = int(getattr(text_cfg, "hidden_size", kwargs.get("hidden_size", 768)))

        vision_defaults = {
            "hidden_size": hidden_size,
            "image_ids": image_ids,
            "image_special_token": image_special_token,
            "image_hidden_size": image_hidden_size,
            "image_token_len": image_token_len,
            "vision_model_path": vision_model_path,
        }
        audio_defaults = {
            "hidden_size": hidden_size,
            "audio_ids": audio_ids,
            "audio_special_token": audio_special_token,
            "audio_hidden_size": audio_hidden_size,
            "audio_encoder_path": audio_encoder_path,
            "audio_vocab_size": audio_vocab_size,
            "audio_pad_token": audio_pad_token,
            "audio_stop_token": audio_stop_token,
            "audio_spk_token": audio_spk_token,
            "audio_sample_rate": audio_sample_rate,
            "audio_target_channels": audio_target_channels,
            "max_audio_tokens": max_audio_tokens,
            "spk_emb_size": spk_emb_size,
        }
        if isinstance(vision_config, dict):
            vision_config = MiniMindOmniVisionConfig(**{**vision_defaults, **vision_config})
        elif vision_config is None:
            vision_config = MiniMindOmniVisionConfig(**vision_defaults)

        if isinstance(audio_config, dict):
            audio_config = MiniMindOmniAudioConfig(**{**audio_defaults, **audio_config})
        elif audio_config is None:
            audio_config = MiniMindOmniAudioConfig(**audio_defaults)

        vision_cfg = vision_config
        audio_cfg = audio_config

        # PretrainedConfig validation may call get_text_config during super().__init__.
        object.__setattr__(self, "text_config", text_cfg)
        object.__setattr__(self, "vision_config", vision_cfg)
        object.__setattr__(self, "audio_config", audio_cfg)

        text_init_kwargs = {
            key: getattr(text_cfg, key)
            for key in MiniMindConfig._text_keys
            if hasattr(text_cfg, key)
        }
        super().__init__(**{**kwargs, **text_init_kwargs})

        self.text_config = text_cfg
        self.vision_config = vision_cfg
        self.audio_config = audio_cfg

        self.num_talker_hidden_layers = num_talker_hidden_layers
        self.talker_hidden_size = talker_hidden_size
        self.think_end_ids = think_end_ids or [26, 234, 234]  # </think>\n\n

        self.audio_ids = self.audio_config.audio_ids
        self.audio_special_token = self.audio_config.audio_special_token
        self.audio_hidden_size = self.audio_config.audio_hidden_size
        self.audio_encoder_path = self.audio_config.audio_encoder_path
        self.audio_vocab_size = self.audio_config.audio_vocab_size
        self.audio_pad_token = self.audio_config.audio_pad_token
        self.audio_stop_token = self.audio_config.audio_stop_token
        self.audio_spk_token = self.audio_config.audio_spk_token
        self.audio_sample_rate = self.audio_config.audio_sample_rate
        self.audio_target_channels = self.audio_config.audio_target_channels
        self.max_audio_tokens = self.audio_config.max_audio_tokens
        self.spk_emb_size = self.audio_config.spk_emb_size

        self.image_ids = self.vision_config.image_ids
        self.image_special_token = self.vision_config.image_special_token
        self.image_hidden_size = self.vision_config.image_hidden_size
        self.image_token_len = self.vision_config.image_token_len
        self.vision_model_path = self.vision_config.vision_model_path
        self.bridge_layer = (
            bridge_layer
            if bridge_layer is not None
            else int(getattr(self.text_config, "num_hidden_layers", self.num_hidden_layers)) // 2 - 1
        )

    def get_text_config(self, **kwargs: Any) -> PretrainedConfig:
        return self.text_config


MiniMindOmniConfig = OmniConfig

try:
    AutoConfig.register(MiniMindConfig.model_type, MiniMindConfig)
    AutoConfig.register(OmniConfig.model_type, OmniConfig)
except ValueError:
    pass


__all__ = [
    "MiniMindConfig",
    "MiniMindOmniAudioConfig",
    "MiniMindOmniConfig",
    "MiniMindOmniVisionConfig",
    "OmniConfig",
]
